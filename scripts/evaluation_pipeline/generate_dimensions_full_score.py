#!/usr/bin/env python
"""Generate evaluation dimensions and full-score criteria for generic QA JSONL."""

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
    extract_question_answer,
    first_non_empty,
    load_completed_ids,
    load_jsonl,
    normalize_text,
    parse_ports,
    resolve_model,
    safe_unlink,
    stable_sample_id,
    wait_for_servers,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SYSTEM_PROMPT = (
    "You are a rigorous rubric-generator assistant for QA answer evaluation.\n"
    "Given one question and one candidate answer, generate evaluation dimensions "
    "and full-score criteria. Each dimension must describe one observable quality "
    "needed to judge or improve the candidate answer. Return strict JSON only."
)


def normalize_dimensions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        dimensions = payload.get("evaluation_dimensions") or payload.get("dimensions") or payload.get("rubrics")
    elif isinstance(payload, list):
        dimensions = payload
    else:
        dimensions = None

    if not isinstance(dimensions, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in dimensions:
        if not isinstance(item, dict):
            continue
        name = first_non_empty(item.get("dimension_name"), item.get("name"), item.get("dimension"))
        full_score_criteria = first_non_empty(
            item.get("full_score_criteria"),
            item.get("criteria_5"),
            item.get("score_5"),
            item.get("criteria"),
            item.get("standard"),
            item.get("rubric"),
        )
        category = first_non_empty(item.get("category"), "general")
        if not name or not full_score_criteria:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "dimension_name": name,
                "category": category,
                "full_score_criteria": full_score_criteria,
            }
        )
    return normalized


def build_user_prompt(sample: dict[str, Any], num_dimensions: str) -> str:
    return (
        "Question:\n"
        f"{sample['question']}\n\n"
        "Candidate Answer:\n"
        f"{sample['answer']}\n\n"
        f"Generate {num_dimensions} evaluation dimensions for this QA pair.\n\n"
        "Requirements:\n"
        "- Each dimension must be specific to this question and candidate answer.\n"
        "- Include correctness, reasoning, instruction-following, format, completeness, or safety only when relevant.\n"
        "- full_score_criteria must describe what a score-5 answer must satisfy for that dimension.\n"
        "- Do not generate duplicate or purely stylistic dimensions unless style is required by the question.\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "evaluation_dimensions": [\n'
        '    {"dimension_name": "...", "category": "...", "full_score_criteria": "..."}\n'
        "  ]\n"
        "}"
    )


def prepare_samples(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question, answer = extract_question_answer(
            record,
            question_field=args.question_field,
            answer_field=args.answer_field,
        )
        if not question or not answer:
            if args.skip_missing:
                continue
            raise ValueError(f"Record {index} missing question/answer. Use --question-field/--answer-field if needed.")
        sample_id = stable_sample_id(record, index, question, answer, args.id_field)
        samples.append(
            {
                "sample_id": sample_id,
                "index": index,
                "question": question,
                "answer": answer,
                "unique_id": record.get("unique_id"),
                "subject": record.get("subject"),
                "level": record.get("level"),
                "reference_answer": record.get("answer"),
                "reference_solution": record.get("solution"),
                "source_record": record,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt({"question": question, "answer": answer}, args.num_dimensions)},
                ],
            }
        )
    return samples


def generate_one(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "index": sample["index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "unique_id": sample.get("unique_id"),
        "subject": sample.get("subject"),
        "level": sample.get("level"),
        "reference_answer": sample.get("reference_answer"),
        "reference_solution": sample.get("reference_solution"),
        "evaluation_dimensions": [],
        "ok": False,
        "error": "",
        "raw_output": "",
        "endpoint": "",
        "model": model,
        "latency_sec": None,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.api_key,
            model=model,
            messages=sample["messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = extract_json_object(raw_text)
        dimensions = normalize_dimensions(parsed)
        row.update(
            {
                "evaluation_dimensions": dimensions,
                "ok": bool(dimensions),
                "error": "" if dimensions else "No valid evaluation_dimensions parsed.",
                "raw_output": raw_text,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
            }
        )
    except Exception as exc:
        row.update({"ok": False, "error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    samples = prepare_samples(records, args)

    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if not args.skip_health_check:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    model = resolve_model(args.model, base_urls, args.health_check_timeout)

    safe_unlink(output_path, args.overwrite)
    completed = load_completed_ids(output_path, require_ok=True)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[input] {input_path}")
    print(f"[output] {output_path}")
    print(f"[model] {model}")
    print(f"[samples] total={len(samples)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="dimensions", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(generate_one, sample, index, args, base_urls, model): sample
                for index, sample in enumerate(pending)
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
    parser = argparse.ArgumentParser(description="Generate QA evaluation dimensions and full-score criteria.")
    parser.add_argument("--input", type=Path, default="datasets/MATH500/test_swapped.jsonl", help="Input JSONL with QA records.")
    parser.add_argument("--output", type=Path, default="datasets/MATH500/evaluation_dim.jsonl", help="Output JSONL for dimension rows.")
    parser.add_argument("--question-field", type=str, default="", help="Optional explicit question field name.")
    parser.add_argument("--answer-field", type=str, default="", help="Optional explicit answer field name.")
    parser.add_argument("--id-field", type=str, default="", help="Optional explicit sample id field name.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip records without extractable question/answer.")
    parser.add_argument("--num-dimensions", type=str, default="4", help="Requested number of dimensions.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=None, help="Ports like: 8000 8001 or 8000-8003.")
    parser.add_argument("--model", type=str, default="", help="Model id. If empty, fetch from /models.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
