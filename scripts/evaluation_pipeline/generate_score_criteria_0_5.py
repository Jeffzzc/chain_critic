#!/usr/bin/env python
"""Generate complete 0-5 score criteria for each QA evaluation dimension."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
from typing import Any

from pipeline_common import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    extract_json_object,
    first_non_empty,
    has_complete_score_criteria,
    load_completed_ids,
    load_jsonl,
    normalize_score_criteria,
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


SYSTEM_PROMPT = (
    "You are a strict rubric expansion assistant.\n"
    "Given a QA pair, one evaluation dimension, and its full-score criterion, "
    "generate complete score criteria for integer scores 0, 1, 2, 3, 4, and 5. "
    "Return strict JSON only."
)


def normalize_criteria_payload(payload: Any, full_score_criteria: str) -> dict[str, str]:
    if isinstance(payload, dict):
        criteria = normalize_score_criteria(payload, full_score_criteria)
        if has_complete_score_criteria(criteria):
            return criteria

        nested = payload.get("score_criteria") or payload.get("criteria")
        if isinstance(nested, dict):
            criteria = normalize_score_criteria({"score_criteria": nested}, full_score_criteria)
            if has_complete_score_criteria(criteria):
                return criteria

    return {str(score): "" for score in range(6)}


def dimension_items_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = row.get("evaluation_dimensions")
    if isinstance(dimensions, list):
        items: list[dict[str, Any]] = []
        for index, dim in enumerate(dimensions):
            if not isinstance(dim, dict):
                continue
            name = first_non_empty(dim.get("dimension_name"), dim.get("name"), dim.get("dimension"))
            full_score_criteria = first_non_empty(
                dim.get("full_score_criteria"),
                dim.get("criteria_5"),
                dim.get("score_criteria", {}).get("5") if isinstance(dim.get("score_criteria"), dict) else "",
            )
            if not name or not full_score_criteria:
                continue
            sample_id = f"{row.get('sample_id', row.get('index', 'sample'))}:dimension:{index + 1}"
            items.append(
                {
                    "sample_id": sample_id,
                    "parent_sample_id": row.get("sample_id"),
                    "index": row.get("index"),
                    "dimension_index": index,
                    "unique_id": row.get("unique_id"),
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                    "reference_answer": row.get("reference_answer"),
                    "reference_solution": row.get("reference_solution"),
                    "question": normalize_text(row.get("question")),
                    "answer": normalize_text(row.get("answer")),
                    "dimension_name": name,
                    "category": first_non_empty(dim.get("category"), "general"),
                    "full_score_criteria": full_score_criteria,
                    "source_dimension": dim,
                }
            )
        return items

    name = first_non_empty(row.get("dimension_name"), row.get("evaluation_dimension"), row.get("dimension"))
    full_score_criteria = first_non_empty(row.get("full_score_criteria"), row.get("criteria_5"))
    if not name or not full_score_criteria:
        return []

    return [
        {
            "sample_id": normalize_text(row.get("sample_id")) or f"{row.get('index', 0)}:dimension:1",
            "parent_sample_id": row.get("parent_sample_id"),
            "index": row.get("index"),
            "dimension_index": row.get("dimension_index", 0),
            "unique_id": row.get("unique_id"),
            "subject": row.get("subject"),
            "level": row.get("level"),
            "reference_answer": row.get("reference_answer"),
            "reference_solution": row.get("reference_solution"),
            "question": normalize_text(row.get("question")),
            "answer": normalize_text(row.get("answer")),
            "dimension_name": name,
            "category": first_non_empty(row.get("category"), "general"),
            "full_score_criteria": full_score_criteria,
            "source_dimension": row,
        }
    ]


def build_tasks(rows: list[dict[str, Any]], skip_missing: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        items = dimension_items_from_row(row)
        if not items and not skip_missing:
            raise ValueError(f"Record {row_index} has no usable dimension/full_score_criteria.")
        for item in items:
            missing = [
                name
                for name in ("question", "answer", "dimension_name", "full_score_criteria")
                if not normalize_text(item.get(name))
            ]
            if missing:
                if skip_missing:
                    continue
                raise ValueError(f"Record {row_index} dimension missing fields: {missing}")
            tasks.append(item)
    return tasks


def build_user_prompt(task: dict[str, Any]) -> str:
    return (
        "Question:\n"
        f"{task['question']}\n\n"
        "Candidate Answer:\n"
        f"{task['answer']}\n\n"
        "Evaluation Dimension:\n"
        f"{task['dimension_name']}\n\n"
        "Full-score criterion for score 5:\n"
        f"{task['full_score_criteria']}\n\n"
        "Generate criteria for scores 0 through 5.\n\n"
        "Requirements:\n"
        "- Criteria must be monotonic from 0 as near-complete failure to 5 as full satisfaction.\n"
        "- Criteria must be concrete and grounded in this QA pair and dimension.\n"
        "- Score 5 must preserve the meaning of the full-score criterion above.\n"
        "- Adjacent scores must be distinguishable.\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "score_criteria": {"0": "...", "1": "...", "2": "...", "3": "...", "4": "...", "5": "..."},\n'
        '  "criteria_0": "...", "criteria_1": "...", "criteria_2": "...",\n'
        '  "criteria_3": "...", "criteria_4": "...", "criteria_5": "..."\n'
        "}"
    )


def generate_one(
    task: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": task["sample_id"],
        "parent_sample_id": task.get("parent_sample_id"),
        "index": task.get("index"),
        "dimension_index": task.get("dimension_index"),
        "unique_id": task.get("unique_id"),
        "subject": task.get("subject"),
        "level": task.get("level"),
        "reference_answer": task.get("reference_answer"),
        "reference_solution": task.get("reference_solution"),
        "question": task["question"],
        "answer": task["answer"],
        "dimension_name": task["dimension_name"],
        "evaluation_dimension": task["dimension_name"],
        "category": task.get("category"),
        "full_score_criteria": task["full_score_criteria"],
        "score_criteria": {str(score): "" for score in range(6)},
        "ok": False,
        "error": "",
        "raw_output": "",
        "endpoint": "",
        "model": model,
        "latency_sec": None,
    }
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
        ]
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.api_key,
            model=model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = extract_json_object(raw_text)
        score_criteria = normalize_criteria_payload(parsed, task["full_score_criteria"])
        ok = has_complete_score_criteria(score_criteria)
        row.update(
            {
                "score_criteria": score_criteria,
                "ok": ok,
                "error": "" if ok else "No complete 0-5 score_criteria parsed.",
                "raw_output": raw_text,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
            }
        )
        for score in range(6):
            row[f"criteria_{score}"] = score_criteria[str(score)]
    except Exception as exc:
        row.update({"ok": False, "error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    tasks = build_tasks(rows, args.skip_missing)

    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if not args.skip_health_check:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    model = resolve_model(args.model, base_urls, args.health_check_timeout)

    safe_unlink(output_path, args.overwrite)
    completed = load_completed_ids(output_path, require_ok=True)
    pending = [task for task in tasks if task["sample_id"] not in completed]

    print(f"[input] {input_path}")
    print(f"[output] {output_path}")
    print(f"[model] {model}")
    print(f"[dimensions] total={len(tasks)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="criteria", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(generate_one, task, index, args, base_urls, model): task
                for index, task in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                with lock:
                    append_jsonl(output_path, row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complete 0-5 criteria for QA evaluation dimensions.")
    parser.add_argument("--input", type=Path, default="datasets/MATH500/evaluation_dim_clean.jsonl", help="Dimension JSONL from stage 1 or flat dimension JSONL.")
    parser.add_argument("--output", type=Path, default="datasets/MATH500/0-5_rubric.jsonl", help="Output flat 0-5 rubric JSONL.")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=8001-8004, help="Ports like: 8000 8001 or 8000-8003.")
    parser.add_argument("--model", type=str, default="", help="Model id. If empty, fetch from /models.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
