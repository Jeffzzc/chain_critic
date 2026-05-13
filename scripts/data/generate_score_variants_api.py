#!/usr/bin/env python
"""Generate 0-4 score variants from full-score seeds via multi-endpoint API."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT = Path("datasets/final/full_score_dimension_seeds_cleaned.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/generated_score_variants")
DEFAULT_TARGET_SCORES = (0, 1, 2, 3, 4)
# DEFAULT_TARGET_SCORES = (0, 1)
DEFAULT_PORTS = tuple(range(8000, 8004))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_MODEL = "Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_DISPATCH_POLL_INTERVAL = 0.05

# SYSTEM_PROMPT = (
#     "You are a data generation assistant for evaluation training. "
#     "You will receive a question, an original high-quality answer, one evaluation "
#     "dimension, the full-score criteria for that dimension, and a target score from 0 to 4. "
#     "Generate a new answer that would realistically receive exactly that target score on "
#     "the given dimension. The weakness must be reflected in the generated answer itself, "
#     "not merely justified later in the reason. Do not keep the answer fully correct while "
#     "only using the reason to argue for a low score. Then produce a revised answer that "
#     "satisfies the full-score criteria for that dimension. Return strict JSON only."
# )

SYSTEM_PROMPT = (
    "You are a data generation assistant for evaluation training. "
    "You will receive a question, an original high-quality answer, one evaluation "
    "dimension, the full-score criteria for that dimension, and a target score from 0 to 4. "
    "Your task is to generate a new answer that would realistically receive exactly that "
    "target score on the specified dimension.\n\n"

    "Critical requirements:\n"
    "1. The score difference must be caused by the content of the generated answer itself, "
    "not merely explained afterward in the reason.\n"
    "2. Do not keep the generated answer fully correct while only using the reason to justify "
    "a low score.\n"
    "3. The generated answer must remain relevant to the question and plausible as a real user response.\n"
    "4. The generated answer should differ from the original answer in a meaningful way.\n"
    "5. The weakness should primarily affect the specified dimension, while avoiding unnecessary damage "
    "to unrelated dimensions unless the target dimension naturally requires it.\n"
    "6. If the target score is low (especially 0 or 1), the generated answer should contain clear, "
    "observable deficiencies in the answer itself. Depending on the dimension, these may include: "
    "calculation mistakes, logical errors, missing justification, contradiction, unsupported claims, "
    "incorrect factual statements, weak reasoning steps, or incomplete argument structure.\n"
    "7. If the target score is medium (2 or 3), the generated answer should be partially correct but "
    "still contain noticeable flaws appropriate to that score level.\n"
    "8. If the target score is high (4), the generated answer should satisfy the full-score criteria well.\n"
    "9. The reason must explicitly point to concrete problems or strengths in the generated answer, "
    "rather than giving vague generic comments.\n"
    "10. Generate revision_suggestions with executable edits that would fix the generated answer.\n"
    "11. Also generate a modified answer that genuinely fixes the problems and satisfies the "
    "full-score criteria.\n"
    "12. The modified answer must not be a superficial rephrasing; it should correct the actual defects.\n"
    "13. Return strict JSON only."
)



USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Original Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Full-Score Criteria:\n{full_score_criteria}\n\n"
    "Target Score:\n{target_score}\n\n"
    "Requirements:\n"
    "1. Generate a plausible answer that would score exactly the target score on the given dimension.\n"
    "2. The returned score field must exactly equal the target score.\n"
    "3. Keep the answer relevant to the question.\n"
    "4. Make the score difference come primarily from the specified dimension.\n"
    "5. Also generate a modified answer that is revised to satisfy the given full-score criteria.\n"
    "6. Generate revision_suggestions as concrete edit instructions for moving the generated answer to score 5.\n"
    "7. Do not mention the rubric or target score inside the answers.\n"
    "8. Return JSON only with this schema:\n"
    "{{\"Score\": {target_score},\"Reason\": \"...\", \"revision_suggestions\": \"...\", \"generated_answer\": \"...\", \"modified_answer\": \"...\"}}"
)


def _extract_rows(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "samples", "records", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return None


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _extract_rows(payload)
    if rows is None:
        raise ValueError(f"Unsupported JSON structure: {path}")
    return [row for row in rows if isinstance(row, dict)]


def build_sample_id(record: dict[str, Any]) -> str:
    existing = record.get("sample_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    parts = [
        str(record.get("question", "")).strip(),
        str(record.get("answer", "")).strip(),
        str(record.get("dimension_name", "")).strip(),
        str(record.get("full_score_criteria", "")).strip(),
    ]
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest


def load_completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                completed.add(sample_id)
    return completed


def build_messages(record: dict[str, Any], target_score: int) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        question=str(record.get("question", "")).strip(),
        answer=str(record.get("answer", "")).strip(),
        dimension_name=str(record.get("dimension_name", "")).strip(),
        full_score_criteria=str(record.get("full_score_criteria", "")).strip(),
        target_score=target_score,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def get_payload_value(payload: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def parse_ports(values: list[str] | None) -> list[int]:
    if not values:
        return list(DEFAULT_PORTS)

    ports: list[int] = []
    for value in values:
        if "-" in value:
            start_text, end_text = value.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            ports.extend(range(start, end + step, step))
        else:
            ports.append(int(value))
    return ports


def build_base_urls(base_url_template: str, ports: list[int]) -> list[str]:
    return [base_url_template.format(port=port) for port in ports]


def _models_endpoint_ready(models_url: str, timeout_seconds: int) -> bool:
    try:
        with urllib_request.urlopen(models_url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (urllib_error.URLError, TimeoutError, ValueError):
        return False


async def _models_endpoint_ready_async(
    session: "aiohttp.ClientSession",
    models_url: str,
) -> bool:
    try:
        async with session.get(models_url) as response:
            return response.status == 200
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        return False


async def wait_for_servers(base_urls: list[str], *, timeout_seconds: int) -> None:
    if aiohttp is not None:
        connector = aiohttp.TCPConnector(limit=max(16, len(base_urls) * 2))
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for base_url in base_urls:
                models_url = base_url.rstrip("/") + "/models"
                while True:
                    ready = await _models_endpoint_ready_async(session, models_url)
                    if ready:
                        break
                    print(f"Waiting for {models_url} ...", end="\r")
                    await asyncio.sleep(2)
    else:
        for base_url in base_urls:
            models_url = base_url.rstrip("/") + "/models"
            while True:
                ready = await asyncio.to_thread(
                    _models_endpoint_ready,
                    models_url,
                    timeout_seconds,
                )
                if ready:
                    break
                print(f"Waiting for {models_url} ...", end="\r")
                await asyncio.sleep(2)
    print("All endpoints are ready.                    ")


def _post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in response from {url}")
    message = choices[0].get("message") or {}
    return message.get("content") or ""


async def _post_chat_completion_async(
    *,
    session: "aiohttp.ClientSession",
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(
        url,
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    ) as response:
        response.raise_for_status()
        response_payload = await response.json(content_type=None)
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in response from {url}")
    message = choices[0].get("message") or {}
    return message.get("content") or ""


async def call_model(
    base_url: str,
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    retries: int,
    timeout_seconds: int,
    http_session: "aiohttp.ClientSession | None" = None,
    request_executor: ThreadPoolExecutor | None = None,
) -> tuple[dict[str, Any] | None, str]:
    last_text = ""
    for attempt in range(retries):
        try:
            if http_session is not None:
                last_text = await _post_chat_completion_async(
                    session=http_session,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                loop = asyncio.get_running_loop()
                last_text = await loop.run_in_executor(
                    request_executor,
                    partial(
                        _post_chat_completion,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout_seconds=timeout_seconds,
                    ),
                )
            parsed = extract_json_object(last_text)
            if parsed is not None:
                return parsed, last_text
        except Exception as exc:
            last_text = f"[error] {exc}"

        if attempt < retries - 1:
            await asyncio.sleep(1)
    return None, last_text


async def multi_writer(
    queue: asyncio.Queue[tuple[Path, dict[str, Any]] | None],
) -> None:
    handles: dict[Path, Any] = {}
    try:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            output_path, row = item
            handle = handles.get(output_path)
            if handle is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                handle = output_path.open("a", encoding="utf-8")
                handles[output_path] = handle

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            queue.task_done()
    finally:
        for handle in handles.values():
            handle.close()


def build_generation_tasks(
    *,
    records: list[dict[str, Any]],
    targets: list[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    prepared_records = [
        {
            "record": record,
            "sample_id": build_sample_id(record),
        }
        for record in records
    ]

    for target_score in targets:
        output_path = output_dir / f"score_{target_score}.jsonl"
        completed_ids = load_completed_ids(output_path)
        remaining = 0
        for prepared in prepared_records:
            if prepared["sample_id"] in completed_ids:
                continue
            remaining += 1
            tasks.append(
                {
                    "record": prepared["record"],
                    "sample_id": prepared["sample_id"],
                    "target_score": target_score,
                    "output_path": output_path,
                }
            )
        print(
            f"[Target {target_score}] output={output_path} total={len(records)} "
            f"completed={len(completed_ids)} remaining={remaining}"
        )
    return tasks


async def dispatch_tasks_round_robin(
    tasks: list[dict[str, Any]],
    endpoint_queues: list[asyncio.Queue[dict[str, Any] | None]],
    *,
    poll_interval: float,
) -> None:
    task_index = 0
    endpoint_count = len(endpoint_queues)
    next_endpoint_index = 0

    while task_index < len(tasks):
        assigned = False
        for offset in range(endpoint_count):
            endpoint_index = (next_endpoint_index + offset) % endpoint_count
            endpoint_queue = endpoint_queues[endpoint_index]
            if endpoint_queue.full():
                continue
            await endpoint_queue.put(tasks[task_index])
            task_index += 1
            next_endpoint_index = (endpoint_index + 1) % endpoint_count
            assigned = True
            break
        if not assigned:
            await asyncio.sleep(poll_interval)

    for endpoint_queue in endpoint_queues:
        await endpoint_queue.put(None)


async def worker(
    *,
    base_url: str,
    api_key: str,
    model: str,
    input_queue: asyncio.Queue[dict[str, Any] | None],
    writer_queue: asyncio.Queue[tuple[Path, dict[str, Any]] | None],
    max_tokens: int,
    temperature: float,
    retries: int,
    request_timeout: int,
    progress_bar: Any,
    http_session: "aiohttp.ClientSession | None",
    request_executor: ThreadPoolExecutor | None,
) -> None:
    while True:
        try:
            task = await input_queue.get()
        except asyncio.CancelledError:
            break
        if task is None:
            input_queue.task_done()
            break

        record = task["record"]
        sample_id = task["sample_id"]
        target_score = task["target_score"]
        output_path = task["output_path"]
        messages = build_messages(record, target_score)
        parsed, raw_text = await call_model(
            base_url,
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            timeout_seconds=request_timeout,
            http_session=http_session,
            request_executor=request_executor,
        )
        if parsed is not None:
            parsed_score = get_payload_value(parsed, "Score", "score", default=target_score)
            output_row = {
                "sample_id": sample_id,
                "question": record.get("question"),
                "answer": get_payload_value(parsed, "generated_answer", "answer"),
                "dimension_name": record.get("dimension_name"),
                "full_score_criteria": record.get("full_score_criteria"),
                "Score": parsed_score,
                "Reason": get_payload_value(parsed, "Reason", "reason"),
                "revision_suggestions": get_payload_value(
                    parsed,
                    "revision_suggestions",
                    "edit_intent",
                    "Revision Suggestions",
                    "Edit Intent",
                ),
                "edit_intent": get_payload_value(
                    parsed,
                    "revision_suggestions",
                    "edit_intent",
                    "Revision Suggestions",
                    "Edit Intent",
                ),
                "modified_answer": get_payload_value(parsed, "modified_answer"),
                # "raw_response": raw_text,
            }
            await writer_queue.put((output_path, output_row))

        input_queue.task_done()
        if progress_bar is not None:
            progress_bar.update(1)


async def run_generation(
    *,
    base_urls: list[str],
    api_key: str,
    model: str,
    tasks: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    retries: int,
    concurrency_per_endpoint: int,
    request_timeout: int,
    dispatch_poll_interval: float,
) -> None:
    if not tasks:
        print("No pending tasks. Nothing to generate.")
        return

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=len(tasks),
            desc="Generating",
            unit="task",
            dynamic_ncols=True,
        )
    else:
        print("tqdm is not installed; progress bar disabled.")

    queue_capacity = max(1, concurrency_per_endpoint)
    endpoint_queues = [
        asyncio.Queue(maxsize=queue_capacity)
        for _ in base_urls
    ]
    writer_queue: asyncio.Queue[tuple[Path, dict[str, Any]] | None] = asyncio.Queue()
    writer_task = asyncio.create_task(multi_writer(writer_queue))
    dispatcher_task = asyncio.create_task(
        dispatch_tasks_round_robin(
            tasks,
            endpoint_queues,
            poll_interval=dispatch_poll_interval,
        )
    )
    workers = []
    worker_count = max(1, concurrency_per_endpoint)
    total_worker_count = max(1, len(base_urls) * worker_count)

    async def launch_workers(
        *,
        http_session: "aiohttp.ClientSession | None",
        request_executor: ThreadPoolExecutor | None,
    ) -> None:
        for base_url, input_queue in zip(base_urls, endpoint_queues):
            for _ in range(worker_count):
                workers.append(
                    asyncio.create_task(
                        worker(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            input_queue=input_queue,
                            writer_queue=writer_queue,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            retries=retries,
                            request_timeout=request_timeout,
                            progress_bar=progress_bar,
                            http_session=http_session,
                            request_executor=request_executor,
                        )
                    )
                )

        await dispatcher_task
        await asyncio.gather(*workers)

    try:
        if aiohttp is not None:
            connector = aiohttp.TCPConnector(limit=max(64, total_worker_count * 2))
            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
                await launch_workers(http_session=http_session, request_executor=None)
        else:
            with ThreadPoolExecutor(max_workers=max(32, total_worker_count)) as request_executor:
                await launch_workers(http_session=None, request_executor=request_executor)
    finally:
        await writer_queue.put(None)
        await writer_queue.join()
        await writer_task
        if progress_bar is not None:
            progress_bar.close()


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    ports = parse_ports(args.ports)
    base_urls = build_base_urls(args.base_url_template, ports)
    if not args.skip_health_check:
        await wait_for_servers(base_urls, timeout_seconds=args.health_check_timeout)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    tasks = build_generation_tasks(
        records=records,
        targets=args.targets,
        output_dir=output_dir,
    )
    await run_generation(
        base_urls=base_urls,
        api_key=api_key,
        model=args.model,
        tasks=tasks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        concurrency_per_endpoint=args.concurrency_per_endpoint,
        request_timeout=args.request_timeout,
        dispatch_poll_interval=args.dispatch_poll_interval,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 0/1/2/3/4-score answer variants from full-score seeds via "
            "multiple OpenAI-compatible endpoints."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Seed json/jsonl path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store score_0.jsonl ... score_4.jsonl",
    )
    parser.add_argument(
        "--base-url-template",
        default=DEFAULT_BASE_URL_TEMPLATE,
        help="Base URL template, e.g. http://localhost:{port}/v1",
    )
    parser.add_argument(
        "--ports",
        nargs="*",
        default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"],
        help="Port list or ranges, e.g. --ports 8000-8007 or --ports 8000 8001 8002",
    )
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by the API.")
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_SCORES),
        help="Target scores to generate, e.g. --targets 0 1 2 3 4",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per sample.")
    parser.add_argument(
        "--concurrency-per-endpoint",
        type=int,
        default=96,
        help="Worker count per endpoint. Use 1 to keep one active generator per GPU.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N seeds.")
    parser.add_argument(
        "--dispatch-poll-interval",
        type=float,
        default=DEFAULT_DISPATCH_POLL_INTERVAL,
        help="Polling interval in seconds when all endpoint queues are temporarily full.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip polling /models before generation.",
    )
    parser.add_argument(
        "--health-check-timeout",
        type=int,
        default=10,
        help="HTTP timeout in seconds for endpoint health checks.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for each generation request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
