#!/usr/bin/env python
"""Generate revision-only SFT data from existing QA/rubric training data.

Input records can be old chat-format SFT rows or flat JSONL rows. The script
extracts:
- Question
- Candidate Answer
- Evaluation Dimension
- Complete 0-5 Score Criteria

It ignores any existing assistant label, including old score/reason/modified
answer, then calls local OpenAI-compatible vLLM endpoints to generate only:
- revision_suggestions
- modified_answer

Output rows are chat-format SFT messages whose assistant content is strict JSON:
{"revision_suggestions": "...", "modified_answer": "..."}
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from pipeline_common import (
    COLON_CLASS,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    extract_json_object,
    iter_jsonl,
    models_endpoint_ready,
    normalize_text,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
)
from regenerate_sft_score_reason_revision_local import load_samples

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_DIR = Path("datasets/train/revision_only_regen")

SYSTEM_PROMPT = (
    "You are an expert answer revision assistant. Given a question, a candidate answer, "
    "one evaluation dimension, and complete 0-5 scoring criteria for that dimension, "
    "produce actionable revision suggestions and a revised answer that would better "
    "satisfy the score-5 criterion. Use only the provided information. Do not add "
    "unsupported facts. Return strict JSON only with exactly these keys: "
    "revision_suggestions, modified_answer."
)

USER_TASK_INSTRUCTION = (
    "### Task Description:\n"
    "You are given a question, a candidate answer, one evaluation dimension, "
    "and the complete 0-5 scoring criteria for that dimension.\n"
    "\n"
    "Your task is to revise the candidate answer using only the provided question, "
    "candidate answer, evaluation dimension, and scoring criteria.\n"
    "\n"
    "Follow these steps:\n"
    "1. Identify concrete changes needed for the candidate answer to best satisfy "
    "the score-5 criterion.\n"
    "2. Write actionable revision_suggestions. The suggestions should be executable "
    "edit instructions, not generic advice. Explaining how the candidate answer should "
    "be revised to best satisfy the score-5 criterion.\n"
    "3. Rewrite the candidate answer into modified_answer. The modified_answer should "
    "address the concrete issues, stay aligned with the original question, optimize "
    "for the same evaluation dimension, and avoid unsupported facts.\n"
    "4. Return strict JSON only with exactly this schema:\n"
    "{\"revision_suggestions\": \"...\", \"modified_answer\": \"...\"}"
)

USER_TEMPLATE = (
    "{instruction}\n\n"
    "Question:\n{question}\n\n"
    "Candidate Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Score Criteria (0-5):\n{criteria_text}"
)

JSON_CODE_BLOCK_RE = re.compile(r"(?is)^\s*```(?:json)?\s*(.*?)\s*```\s*$")
REVISION_RE = re.compile(
    rf"(?is)(?:revision suggestions|revision_suggestions|edit intent)\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?:(?:\n\s*)?(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(
    rf"(?is)(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}\s*(.*)$"
)
JSON_FIELD_RE_TEMPLATE = r'(?is)"{field}"\s*:\s*"'


def strip_code_fence(text: str) -> str:
    match = JSON_CODE_BLOCK_RE.match(str(text or "").strip())
    if match:
        return match.group(1).strip()
    return str(text or "").strip()


def extract_loose_json_string_field(text: str, field: str) -> str:
    """Extract a JSON-like quoted string while preserving LaTeX backslashes.

    Some local models emit JSON-shaped text with LaTeX such as \psi or \quad
    inside strings. Those are invalid JSON escapes, so json.loads cannot parse
    them. This scanner only handles the two string fields we need.
    """
    match = re.search(JSON_FIELD_RE_TEMPLATE.format(field=re.escape(field)), text)
    if not match:
        return ""

    chars: list[str] = []
    index = match.end()
    while index < len(text):
        char = text[index]
        if char == '"':
            tail = text[index + 1 :]
            if re.match(r"\s*(?:,|\})", tail):
                break
            chars.append(char)
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            nxt = text[index + 1]
            if nxt == "n":
                chars.append("\n")
            elif nxt == "r":
                chars.append("\n")
            elif nxt == '"':
                chars.append('"')
            elif nxt == "\\":
                chars.append("\\")
            else:
                chars.append("\\" + nxt)
            index += 2
            continue
        chars.append(char)
        index += 1
    return normalize_text("".join(chars))


def build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        instruction=USER_TASK_INSTRUCTION,
        question=sample["question"],
        answer=sample["answer"],
        dimension_name=sample["dimension_name"],
        criteria_text=sample["criteria_text"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_revision_output(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip().replace("\r\n", "\n")
    cleaned_text = strip_code_fence(raw_text)
    parsed = extract_json_object(cleaned_text)
    strict_json_ok = parsed is not None

    revision_suggestions = ""
    modified_answer = ""

    if isinstance(parsed, dict):
        revision_suggestions = normalize_text(
            parsed.get("revision_suggestions")
            or parsed.get("edit_intent")
            or parsed.get("Revision Suggestions")
            or parsed.get("Edit Intent")
        )
        modified_answer = normalize_text(
            parsed.get("modified_answer")
            or parsed.get("revised_answer")
            or parsed.get("better_answer")
            or parsed.get("Modified Answer")
            or parsed.get("Revised Answer")
            or parsed.get("Better Answer")
        )
    else:
        revision_suggestions = extract_loose_json_string_field(cleaned_text, "revision_suggestions")
        modified_answer = extract_loose_json_string_field(cleaned_text, "modified_answer")
        revision_match = REVISION_RE.search(cleaned_text)
        modified_match = MODIFIED_RE.search(cleaned_text)
        if not revision_suggestions and revision_match:
            revision_suggestions = normalize_text(revision_match.group(1))
        if not modified_answer and modified_match:
            modified_answer = normalize_text(modified_match.group(1))

    ok = bool(revision_suggestions) and bool(modified_answer)
    return {
        "revision_suggestions": revision_suggestions,
        "modified_answer": modified_answer,
        "raw_output": raw_text,
        "ok": ok,
        "parse_error": None if ok else "Missing revision_suggestions or modified_answer.",
        "strict_json_ok": strict_json_ok and ok,
    }


def assistant_json(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "revision_suggestions": row["revision_suggestions"],
            "modified_answer": row["modified_answer"],
        },
        ensure_ascii=False,
    )


def build_sft_row(sample: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_messages(sample)[1]["content"]},
            {"role": "assistant", "content": assistant_json(pred)},
        ]
    }


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for row in iter_jsonl(path):
        if row.get("ok") is not True:
            reparsed = parse_revision_output(row.get("raw_output", ""))
            if reparsed.get("ok") is True:
                row.update(reparsed)
                row["ok"] = True
            else:
                continue
        if row.get("ok") is not True:
            continue
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            completed[sample_id] = row
    return completed


def filter_ready_base_urls(
    base_urls: list[str],
    *,
    timeout_seconds: int,
    workers: int,
) -> list[str]:
    ready_urls: list[str] = []
    lock = threading.Lock()

    def _probe(base_url: str) -> None:
        if models_endpoint_ready(base_url, timeout_seconds):
            with lock:
                ready_urls.append(base_url)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_probe, base_url) for base_url in base_urls]
        for future in as_completed(futures):
            future.result()

    ready_set = set(ready_urls)
    return [base_url for base_url in base_urls if base_url in ready_set]


def resolve_runtime(args: argparse.Namespace) -> tuple[list[str], str]:
    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if args.skip_health_check:
        pass
    elif args.wait_all_ports:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    else:
        requested = len(base_urls)
        base_urls = filter_ready_base_urls(
            base_urls,
            timeout_seconds=args.health_check_timeout,
            workers=args.probe_workers,
        )
        print(f"[ports] ready={len(base_urls)} / requested={requested}")
        if not base_urls:
            raise RuntimeError("No ready vLLM endpoints found.")
    model = args.model.strip() or resolve_model("", base_urls, args.health_check_timeout)
    return base_urls, model


def run_one(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "source_index": sample["source_index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "dimension_name": sample["dimension_name"],
        "criteria_text": sample["criteria_text"],
        "revision_suggestions": "",
        "modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "model": model,
        "latency_sec": None,
        "strict_json_ok": False,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.api_key,
            model=model,
            messages=build_messages(sample),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = parse_revision_output(raw_text)
        row.update(
            {
                "revision_suggestions": parsed["revision_suggestions"],
                "modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "ok": parsed["ok"],
                "parse_error": parsed["parse_error"],
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"],
            }
        )
    except Exception as exc:
        row.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


async def post_chat_completion_async(
    session: "aiohttp.ClientSession",
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with session.post(url, json=payload, headers=headers) as response:
        response.raise_for_status()
        response_payload = await response.json(content_type=None)

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned from {url}")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


async def call_chat_with_retries_async(
    session: "aiohttp.ClientSession",
    *,
    base_urls: list[str],
    preferred_endpoint_index: int,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_sleep: float,
) -> tuple[str, str]:
    last_error: Exception | None = None
    endpoint_count = max(1, len(base_urls))
    for attempt in range(retries + 1):
        base_url = base_urls[(preferred_endpoint_index + attempt) % endpoint_count]
        try:
            text = await post_chat_completion_async(
                session,
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return text, base_url
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(retry_sleep)
    raise RuntimeError(str(last_error))


async def run_one_async(
    session: "aiohttp.ClientSession",
    sample: dict[str, Any],
    task_index: int,
    preferred_endpoint_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "source_index": sample["source_index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "dimension_name": sample["dimension_name"],
        "criteria_text": sample["criteria_text"],
        "revision_suggestions": "",
        "modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "model": model,
        "latency_sec": None,
        "strict_json_ok": False,
    }
    try:
        raw_text, endpoint = await call_chat_with_retries_async(
            session,
            base_urls=base_urls,
            preferred_endpoint_index=preferred_endpoint_index,
            api_key=args.api_key,
            model=model,
            messages=build_messages(sample),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = parse_revision_output(raw_text)
        row.update(
            {
                "revision_suggestions": parsed["revision_suggestions"],
                "modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "ok": parsed["ok"],
                "parse_error": parsed["parse_error"],
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"],
            }
        )
    except Exception as exc:
        row.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


async def cache_writer(
    cache_path: Path,
    queue: "asyncio.Queue[dict[str, Any] | None]",
    completed: dict[str, dict[str, Any]],
    progress: Any,
    flush_every: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pending_flushes = 0
    with cache_path.open("a", encoding="utf-8") as f:
        while True:
            row = await queue.get()
            try:
                if row is None:
                    break
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                pending_flushes += 1
                if pending_flushes >= flush_every:
                    f.flush()
                    pending_flushes = 0
                if row.get("ok") is True:
                    completed[row["sample_id"]] = row
                if progress is not None:
                    progress.update(1)
            finally:
                queue.task_done()
        if pending_flushes:
            f.flush()


async def generate_split_async(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> None:
    if aiohttp is None:
        raise RuntimeError("aiohttp is required for async inference. Install aiohttp or pass --sync-http.")

    samples = load_samples(input_path, split, skip_missing=args.skip_missing, limit=args.limit)
    safe_unlink(cache_path, args.overwrite)
    safe_unlink(output_path, args.overwrite)
    completed = load_completed(cache_path)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[run] split={split} total={len(samples)} completed={len(completed)} pending={len(pending)}")
    print(f"[run] split={split} cache={cache_path}")
    print(f"[run] split={split} output={output_path}")
    print(f"[run] split={split} async_workers={args.workers} endpoints={len(base_urls)}")

    task_queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue(maxsize=args.queue_size)
    writer_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=max(args.queue_size, args.workers * 2))
    progress = tqdm(total=len(pending), desc=f"{split}_revision", ncols=100) if tqdm is not None else None

    async def producer() -> None:
        for index, sample in enumerate(pending):
            await task_queue.put((index, sample))
        for _ in range(args.workers):
            await task_queue.put(None)

    async def worker(worker_index: int) -> None:
        endpoint_index = worker_index % len(base_urls)
        while True:
            item = await task_queue.get()
            try:
                if item is None:
                    break
                task_index, sample = item
                row = await run_one_async(
                    session,
                    sample,
                    task_index,
                    endpoint_index,
                    args,
                    base_urls,
                    model,
                )
                await writer_queue.put(row)
            finally:
                task_queue.task_done()

    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=min(30, args.request_timeout))
    connector = aiohttp.TCPConnector(
        limit=max(args.workers + len(base_urls), 1),
        limit_per_host=max(1, args.workers_per_endpoint),
        ttl_dns_cache=300,
    )
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
            writer_task = asyncio.create_task(
                cache_writer(cache_path, writer_queue, completed, progress, max(1, args.flush_every))
            )
            producer_task = asyncio.create_task(producer())
            worker_tasks = [asyncio.create_task(worker(index)) for index in range(args.workers)]
            await producer_task
            await asyncio.gather(*worker_tasks)
            await writer_queue.put(None)
            await writer_task
    finally:
        if progress is not None:
            progress.close()

    write_sft_rows_streaming(split, output_path, samples, completed)


def write_sft_rows_streaming(
    split: str,
    output_path: Path,
    samples: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sft_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            pred = completed.get(sample["sample_id"])
            if pred is not None and pred.get("ok") is True:
                f.write(json.dumps(build_sft_row(sample, pred), ensure_ascii=False) + "\n")
                sft_count += 1
    print(f"[done] split={split} sft_rows={sft_count} failed_or_pending={len(samples) - sft_count}")


def generate_split(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> None:
    samples = load_samples(input_path, split, skip_missing=args.skip_missing, limit=args.limit)
    safe_unlink(cache_path, args.overwrite)
    safe_unlink(output_path, args.overwrite)
    completed = load_completed(cache_path)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[run] split={split} total={len(samples)} completed={len(completed)} pending={len(pending)}")
    print(f"[run] split={split} cache={cache_path}")
    print(f"[run] split={split} output={output_path}")

    progress = tqdm(total=len(pending), desc=f"{split}_revision", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            next_index = 0
            futures = {}
            while next_index < len(pending) and len(futures) < args.queue_size:
                sample = pending[next_index]
                futures[executor.submit(run_one, sample, next_index, args, base_urls, model)] = sample
                next_index += 1
            while futures:
                for future in as_completed(futures):
                    futures.pop(future)
                    row = future.result()
                    if row.get("ok") is True:
                        completed[row["sample_id"]] = row
                    append_jsonl(cache_path, row)
                    if progress is not None:
                        progress.update(1)
                    while next_index < len(pending) and len(futures) < args.queue_size:
                        sample = pending[next_index]
                        futures[executor.submit(run_one, sample, next_index, args, base_urls, model)] = sample
                        next_index += 1
                    break
    finally:
        if progress is not None:
            progress.close()

    write_sft_rows_streaming(split, output_path, samples, completed)


def run(args: argparse.Namespace) -> None:
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.queue_size <= 0:
        raise ValueError("--queue-size must be positive.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_urls, model = resolve_runtime(args)
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")

    if args.train_input:
        if args.sync_http:
            generate_split(
                split="train",
                input_path=Path(args.train_input),
                output_path=output_dir / args.train_output_name,
                cache_path=output_dir / args.train_cache_name,
                args=args,
                base_urls=base_urls,
                model=model,
            )
        else:
            asyncio.run(
                generate_split_async(
                    split="train",
                    input_path=Path(args.train_input),
                    output_path=output_dir / args.train_output_name,
                    cache_path=output_dir / args.train_cache_name,
                    args=args,
                    base_urls=base_urls,
                    model=model,
                )
            )
    if args.test_input:
        if args.sync_http:
            generate_split(
                split="test",
                input_path=Path(args.test_input),
                output_path=output_dir / args.test_output_name,
                cache_path=output_dir / args.test_cache_name,
                args=args,
                base_urls=base_urls,
                model=model,
            )
        else:
            asyncio.run(
                generate_split_async(
                    split="test",
                    input_path=Path(args.test_input),
                    output_path=output_dir / args.test_output_name,
                    cache_path=output_dir / args.test_cache_name,
                    args=args,
                    base_urls=base_urls,
                    model=model,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate revision_suggestions + modified_answer SFT data from existing QA/rubric rows."
    )
    parser.add_argument("--train-input", type=Path, default=Path("datasets/train/final_train_split.jsonl"))
    parser.add_argument("--test-input", type=Path, default=Path("datasets/train/final_test_split.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-output-name", default="revision_only_sft_train.jsonl")
    parser.add_argument("--test-output-name", default="revision_only_sft_test.jsonl")
    parser.add_argument("--train-cache-name", default="revision_only_train_cache.jsonl")
    parser.add_argument("--test-cache-name", default="revision_only_test_cache.jsonl")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows from each split.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8001-8007"], help="Ports like: 8001-8007 or 8001 8002.")
    parser.add_argument("--model", type=str, default="Qwen3.6-27B")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=224, help="Total concurrent requests across all endpoints.")
    parser.add_argument(
        "--workers-per-endpoint",
        type=int,
        default=32,
        help="Per-host HTTP connection cap for async mode. Keep this close to workers / endpoint_count.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2048,
        help="Maximum queued/submitted unfinished samples. Prevents creating one Future per input row.",
    )
    parser.add_argument("--flush-every", type=int, default=100, help="Flush cache file after this many rows.")
    parser.add_argument("--sync-http", action="store_true", help="Use the old synchronous urllib request path.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--wait-all-ports", action="store_true")
    parser.add_argument("--probe-workers", type=int, default=64)
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
