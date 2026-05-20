#!/usr/bin/env python3
"""Build score=5 SFT data with plain-text assistant output.

Outputs messages in the same format as datasets/train/final_test_split.jsonl:
- system: evaluator prompt
- user: task description + Question/Answer/Evaluation_dimension/Criteria
- assistant: plain text with 4 lines (Score/Reason/Revision Suggestions/Modified Answer)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = ROOT / "scripts" / "evaluation_pipeline"
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    iter_jsonl,
    models_endpoint_ready,
    normalize_text,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT = Path("datasets/0-5/score_criteria_0_5.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/train/final_score_reason_plaintext_score5")

SYSTEM_PROMPT = (
    "You are an AI evaluator-and-rewriter.\n"
    "Evaluate the given answer strictly using ONLY the provided evaluation dimension "
    "and the complete 0-5 scoring criteria.\n"
    "Then revise the answer to better satisfy ONLY that dimension.\n"
    "Do not add unsupported facts. If an assumption is necessary, state it minimally "
    "and explicitly.\n"
    "Output plain text in exactly 4 lines, with exactly these prefixes and no numbering:\n"
    "Score: <an integer number from 0 to 5>\n"
    "Reason: <one-line concise explanation strictly based on the given dimension criteria>\n"
    "Revision Suggestions: <one-line actionable edit instructions>\n"
    "Modified Answer: <one-line revised answer optimized only for the given dimension criteria>\n"
    "Do not include any extra text, JSON, markdown, bullets, or line breaks inside any field."
)

USER_TASK_INSTRUCTION = (
    "###Task Description:\n"
    "You are given a question, a response to evaluate, and one evaluation dimension with its complete 0-5 scoring criteria.\n"
    "1. Write a score that reflects how well the response satisfies the given criteria.\n"
    "2. Write feedback that assesses the quality of the response strictly based on the given dimension criteria.\n"
    "3. Write actionable revision suggestions that directly address the issues in the reason.\n"
    "4. Then rewrite the answer so it better satisfies the criteria, without adding unsupported facts.\n"
    "5. The output format must be exactly:\n"
    "Score: <score>\n"
    "Reason: <feedback>\n"
    "Revision Suggestions: <edit instructions>\n"
    "Modified Answer: <rewritten answer>\n"
    "6. Do not generate any other opening, closing, JSON, or explanations."
)

USER_TEMPLATE = (
    "{instruction}\n\n"
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluation_dimension:\n{dimension_name}\n\n"
    "Criteria (0-5):\n{criteria_text}"
)

SCORE_LINE_RE = re.compile(r"(?im)^\s*Score\s*:\s*([0-5])\s*$")
REASON_LINE_RE = re.compile(r"(?im)^\s*Reason\s*:\s*(.*?)\s*$")
REVISION_LINE_RE = re.compile(r"(?im)^\s*Revision Suggestions\s*:\s*(.*?)\s*$")
MODIFIED_LINE_RE = re.compile(r"(?im)^\s*Modified Answer\s*:\s*(.*?)\s*$")


def sample_records(path: Path, count: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if count <= 0:
        return []
    if not shuffle:
        rows: List[Dict[str, Any]] = []
        for row in iter_jsonl(path):
            rows.append(row)
            if len(rows) >= count:
                break
        return rows
    reservoir: List[Dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path)):
        if len(reservoir) < count:
            reservoir.append(row)
        else:
            j = rng.randint(0, index)
            if j < count:
                reservoir[j] = row
    rng.shuffle(reservoir)
    return reservoir


def resolve_question(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("question"))


def resolve_dimension(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("dimension_name") or record.get("evaluation_dimension"))


def resolve_reference_answer(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("answer") or record.get("reference_answer") or record.get("model_answer"))


def parse_score_criteria_map(record: Dict[str, Any]) -> Dict[str, str]:
    if isinstance(record.get("score_criteria"), dict):
        return {str(k): str(v) for k, v in record["score_criteria"].items()}
    raw = record.get("0-5_Criteria") or record.get("score_criteria")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
    criteria: Dict[str, str] = {}
    for i in range(6):
        key = f"criteria_{i}"
        if key in record and record[key] is not None:
            criteria[str(i)] = str(record[key])
    return criteria


def format_criteria_text(record: Dict[str, Any]) -> str:
    score_map = parse_score_criteria_map(record)
    if not score_map:
        return ""
    lines: List[str] = []
    for i in range(6):
        key = str(i)
        if key in score_map and str(score_map[key]).strip():
            lines.append(f"{i}: {score_map[key]}")
    return "\n".join(lines)


def build_user_content(question: str, answer: str, dimension_name: str, criteria_text: str) -> str:
    return USER_TEMPLATE.format(
        instruction=USER_TASK_INSTRUCTION,
        question=question,
        answer=answer,
        dimension_name=dimension_name,
        criteria_text=criteria_text,
    )


def build_messages(question: str, answer: str, dimension_name: str, criteria_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(question, answer, dimension_name, criteria_text)},
    ]


def parse_plaintext_output(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip().replace("\r\n", "\n")
    score_match = SCORE_LINE_RE.search(raw)
    reason_match = REASON_LINE_RE.search(raw)
    revision_match = REVISION_LINE_RE.search(raw)
    modified_match = MODIFIED_LINE_RE.search(raw)
    score = int(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else ""
    revision_suggestions = revision_match.group(1).strip() if revision_match else ""
    modified_answer = modified_match.group(1).strip() if modified_match else ""
    return {
        "score": score,
        "reason": reason,
        "revision_suggestions": revision_suggestions,
        "modified_answer": modified_answer,
        "raw_output": raw,
        "parse_error": None if score is not None else "Failed to parse score line.",
    }


def build_assistant_content(parsed: Dict[str, Any]) -> str:
    return (
        f"Score: {parsed['score']}\n"
        f"Reason: {parsed['reason']}\n"
        f"Revision Suggestions: {parsed['revision_suggestions']}\n"
        f"Modified Answer: {parsed['modified_answer']}"
    )


def compute_sample_id(question: str, answer: str, dimension_name: str, criteria_text: str) -> str:
    payload = {
        "question": question,
        "answer": answer,
        "dimension_name": dimension_name,
        "criteria_text": criteria_text,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def filter_ready_base_urls(
    base_urls: List[str],
    *,
    timeout_seconds: int,
    workers: int,
) -> List[str]:
    ready_urls: List[str] = []
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


def resolve_runtime(args: argparse.Namespace) -> Tuple[List[str], str]:
    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if args.skip_health_check:
        pass
    elif args.wait_all_ports:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    else:
        before = len(base_urls)
        base_urls = filter_ready_base_urls(
            base_urls,
            timeout_seconds=args.health_check_timeout,
            workers=args.probe_workers,
        )
        print(f"[ports] ready={len(base_urls)} / requested={before}")
        if not base_urls:
            raise RuntimeError(
                "No ready vLLM endpoints found. Check --ports/--base-url-template, "
                "or pass --wait-all-ports if servers are still starting."
            )
    model = args.model.strip() or resolve_model("", base_urls, args.health_check_timeout)
    return base_urls, model


def load_completed_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    for row in iter_jsonl(path):
        if row.get("ok") is not True:
            continue
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            completed.add(sample_id)
    return completed


def run_one(
    *,
    record: Dict[str, Any],
    split: str,
    index: int,
    args: argparse.Namespace,
    base_urls: List[str],
    model: str,
) -> Dict[str, Any]:
    started = time.time()
    question = resolve_question(record)
    dimension_name = resolve_dimension(record)
    answer = resolve_reference_answer(record)
    criteria_text = format_criteria_text(record)

    result: Dict[str, Any] = {
        "sample_id": "",
        "split": split,
        "question": question,
        "dimension_name": dimension_name,
        "criteria_text": criteria_text,
        "candidate_answer": answer,
        "score": None,
        "reason": "",
        "revision_suggestions": "",
        "modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "model": model,
        "latency_sec": None,
        "sft_row": None,
    }

    if not (question and dimension_name and answer and criteria_text):
        result["parse_error"] = "Missing question/dimension/criteria/reference answer."
        return result

    sample_id = compute_sample_id(question, answer, dimension_name, criteria_text)
    result["sample_id"] = sample_id

    messages = build_messages(question, answer, dimension_name, criteria_text)

    for attempt in range(max(1, args.max_attempts)):
        try:
            raw_text, endpoint = call_chat_with_retries(
                base_urls=base_urls,
                task_index=index,
                api_key=args.api_key,
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_seconds=args.request_timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            parsed = parse_plaintext_output(raw_text)
            ok = (
                parsed["score"] is not None
                and 0 <= int(parsed["score"]) <= 5
                and bool(parsed["reason"])
                and bool(parsed["revision_suggestions"])
                and bool(parsed["modified_answer"])
            )
            if ok and args.enforce_score5 and int(parsed["score"]) != 5:
                ok = False
                parsed["parse_error"] = f"Score mismatch: got {parsed['score']} vs target 5"

            result.update(
                {
                    "score": parsed["score"],
                    "reason": parsed["reason"],
                    "revision_suggestions": parsed["revision_suggestions"],
                    "modified_answer": parsed["modified_answer"],
                    "raw_output": parsed["raw_output"],
                    "ok": ok,
                    "parse_error": parsed.get("parse_error") if not ok else None,
                    "endpoint": endpoint,
                    "latency_sec": round(time.time() - started, 4),
                }
            )
            if ok:
                assistant_content = build_assistant_content(parsed)
                result["sft_row"] = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_content(question, answer, dimension_name, criteria_text)},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            return result
        except Exception as exc:
            result.update({
                "request_error": str(exc),
                "latency_sec": round(time.time() - started, 4),
            })
            time.sleep(args.retry_sleep)

    return result


def generate_split(
    *,
    split: str,
    records: List[Dict[str, Any]],
    output_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    base_urls: List[str],
    model: str,
) -> None:
    safe_unlink(cache_path, args.overwrite)
    safe_unlink(output_path, args.overwrite)

    completed = load_completed_ids(cache_path)
    pending: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in enumerate(records):
        question = resolve_question(row)
        dimension_name = resolve_dimension(row)
        answer = resolve_reference_answer(row)
        criteria_text = format_criteria_text(row)
        if not (question and dimension_name and answer and criteria_text):
            continue
        sample_id = compute_sample_id(question, answer, dimension_name, criteria_text)
        row["_sample_id"] = sample_id
        if sample_id in completed:
            continue
        pending.append((idx, row))

    print(f"[run] split={split} output={output_path}")
    print(f"[run] split={split} cache={cache_path}")
    print(f"[run] split={split} total={len(records)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc=f"{split}_score5_plain", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    run_one,
                    record=row,
                    split=split,
                    index=idx,
                    args=args,
                    base_urls=base_urls,
                    model=model,
                ): row
                for idx, row in pending
            }
            for future in as_completed(futures):
                row = future.result()
                sft_row = row.pop("sft_row", None)
                with lock:
                    append_jsonl(cache_path, row)
                    if row.get("ok") is True and sft_row is not None:
                        append_jsonl(output_path, sft_row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    print(f"[done] split={split} sft_rows_written={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build score=5 SFT data with plain-text assistant output."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-output-name", default="final_score_reason_plaintext_train.jsonl")
    parser.add_argument("--test-output-name", default="final_score_reason_plaintext_test.jsonl")
    parser.add_argument("--train-cache-name", default="final_score_reason_plaintext_train_cache.jsonl")
    parser.add_argument("--test-cache-name", default="final_score_reason_plaintext_test_cache.jsonl")

    parser.add_argument("--train-count", type=int, default=85000)
    parser.add_argument("--test-count", type=int, default=17000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle", action="store_true", default=True)

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8001-8007"], help="Ports like: 8000-8003 or 8000 8001.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=784)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enforce-score5", action="store_true", default=True)
    parser.add_argument("--allow-non-5", action="store_false", dest="enforce_score5")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--wait-all-ports", action="store_true")
    parser.add_argument("--probe-workers", type=int, default=64)
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_count = int(args.train_count) + int(args.test_count)
    base_records = sample_records(args.input, total_count, args.seed, args.shuffle)
    if len(base_records) < total_count:
        raise ValueError(
            f"Not enough base records. Need {total_count}, got {len(base_records)} from {args.input}."
        )

    train_records = base_records[: int(args.train_count)]
    test_records = base_records[int(args.train_count) : int(args.train_count) + int(args.test_count)]

    base_urls, model = resolve_runtime(args)
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")

    if train_records:
        generate_split(
            split="train",
            records=train_records,
            output_path=output_dir / args.train_output_name,
            cache_path=output_dir / args.train_cache_name,
            args=args,
            base_urls=base_urls,
            model=model,
        )

    if test_records:
        generate_split(
            split="test",
            records=test_records,
            output_path=output_dir / args.test_output_name,
            cache_path=output_dir / args.test_cache_name,
            args=args,
            base_urls=base_urls,
            model=model,
        )


if __name__ == "__main__":
    main()
