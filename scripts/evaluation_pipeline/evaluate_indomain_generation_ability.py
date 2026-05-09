#!/usr/bin/env python
"""LLM-judge in-domain generation ability with WIN/TIE/LOSE comparisons."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional

from pipeline_common import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    build_base_urls,
    call_chat_with_retries,
    fetch_model_id,
    load_jsonl,
    normalize_text,
    parse_ports,
    wait_for_servers,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


RESULT_PATTERN = re.compile(r"(?im)^\s*result\s*[:\uFF1A]\s*(win|tie|lose)\s*$")
REASON_PATTERN = re.compile(r"(?is)reason\s*[:\uFF1A]\s*(.*)$")

SYSTEM_PROMPT = """You are a strict in-domain QA generation judge.

Compare Candidate A against Candidate B for the same question. Judge from Candidate A's perspective.

Use the question, reference final answer when available, evaluation dimension, and 0-5 criteria as the judging standard. Candidate B may be a gold/reference solution or a baseline model answer.

Rules:
- Choose WIN if Candidate A is clearly better than Candidate B for solving the question and satisfying the dimension.
- Choose TIE if both are similarly correct and similarly useful under the dimension.
- Choose LOSE if Candidate A is clearly worse, less correct, less complete, or less aligned with the dimension.
- Do not reward verbosity by itself.
- Penalize unsupported facts, math errors, contradictions, and answers that do not address the question.
- Output exactly two lines:
Result: WIN|TIE|LOSE
Reason: <one concise sentence grounded in the question and criteria>
"""


def row_key(row: dict[str, Any]) -> str:
    sample_id = normalize_text(row.get("sample_id"))
    if sample_id:
        return f"sample_id::{sample_id}"
    unique_id = normalize_text(row.get("unique_id"))
    dimension = normalize_text(row.get("dimension_name") or row.get("evaluation_dimension"))
    if unique_id and dimension:
        return f"unique_id_dimension::{unique_id}::{dimension}"
    index = row.get("index")
    if isinstance(index, int):
        return f"index::{index}"
    raise ValueError(f"Row is missing a usable key: {row}")


def load_rows_by_key(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = row_key(row)
        if key in rows:
            raise ValueError(f"Duplicate key {key!r} found in {path}")
        rows[key] = row
    return rows


def first_field(row: dict[str, Any], fields: list[str]) -> tuple[str, str]:
    for field in fields:
        value = normalize_text(row.get(field))
        if value:
            return value, field
    return "", ""


def format_criteria(row: dict[str, Any]) -> str:
    score_criteria = row.get("score_criteria")
    if isinstance(score_criteria, dict):
        pieces = []
        for score in range(6):
            value = normalize_text(score_criteria.get(str(score)) or score_criteria.get(score))
            if value:
                pieces.append(f"Score {score}: {value}")
        if pieces:
            return "\n".join(pieces)
    return normalize_text(row.get("criteria_text") or row.get("criteria"))


def parse_result(text: str) -> tuple[Optional[str], str, Optional[str]]:
    raw_text = text.strip().replace("\r\n", "\n")
    result_match = RESULT_PATTERN.search(raw_text)
    reason_match = REASON_PATTERN.search(raw_text)
    result = result_match.group(1).lower() if result_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    parse_error = None if result in {"win", "tie", "lose"} else "Failed to parse Result as WIN/TIE/LOSE."
    return result, reason, parse_error


def build_user_prompt(sample: dict[str, Any]) -> str:
    return (
        "Question:\n"
        f"{sample['question']}\n\n"
        "Reference final answer:\n"
        f"{sample['reference_answer'] or '(not provided)'}\n\n"
        "Evaluation Dimension:\n"
        f"{sample['dimension_name'] or '(general answer quality)'}\n\n"
        "Score Criteria (0-5):\n"
        f"{sample['criteria'] or '(no explicit criteria provided)'}\n\n"
        "Original candidate answer before revision:\n"
        f"{sample['original_answer'] or '(not provided)'}\n\n"
        "Candidate A:\n"
        f"{sample['candidate_a']}\n\n"
        "Candidate B:\n"
        f"{sample['candidate_b']}\n\n"
        "Decide whether Candidate A wins, ties, or loses against Candidate B."
    )


def build_sample(
    row: dict[str, Any],
    *,
    target_fields: list[str],
    reference_fields: list[str],
    baseline_row: Optional[dict[str, Any]],
    baseline_fields: list[str],
) -> Optional[dict[str, Any]]:
    candidate_a, candidate_a_field = first_field(row, target_fields)
    if not candidate_a:
        return None

    if baseline_row is not None:
        candidate_b, candidate_b_field = first_field(baseline_row, baseline_fields)
        reference_source = "baseline"
    else:
        candidate_b, candidate_b_field = first_field(row, reference_fields)
        reference_source = "reference"
    if not candidate_b:
        return None

    sample = {
        "comparison_key": row_key(row),
        "sample_id": row.get("sample_id"),
        "index": row.get("index"),
        "unique_id": row.get("unique_id"),
        "parent_sample_id": row.get("parent_sample_id"),
        "subject": row.get("subject"),
        "level": row.get("level"),
        "question": normalize_text(row.get("question")),
        "original_answer": normalize_text(row.get("answer") or row.get("rubric_answer")),
        "reference_answer": normalize_text(row.get("reference_answer")),
        "dimension_name": normalize_text(row.get("dimension_name") or row.get("evaluation_dimension")),
        "criteria": format_criteria(row),
        "candidate_a": candidate_a,
        "candidate_a_field": candidate_a_field,
        "candidate_b": candidate_b,
        "candidate_b_field": candidate_b_field,
        "candidate_b_source": reference_source,
        "messages": [],
    }
    sample["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(sample)},
    ]
    return sample


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = normalize_text(row.get("comparison_key"))
        if key:
            existing[key] = row
    return existing


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def judge_one(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
    input_path: Path,
) -> dict[str, Any]:
    started = time.time()
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
        result, reason, parse_error = parse_result(raw_text)
        request_error = None
    except Exception as exc:
        raw_text = ""
        endpoint = ""
        result = None
        reason = ""
        parse_error = None
        request_error = str(exc)

    return {
        "comparison_key": sample["comparison_key"],
        "sample_id": sample.get("sample_id"),
        "index": sample.get("index"),
        "unique_id": sample.get("unique_id"),
        "parent_sample_id": sample.get("parent_sample_id"),
        "subject": sample.get("subject"),
        "level": sample.get("level"),
        "input_file": str(input_path),
        "question": sample.get("question"),
        "dimension_name": sample.get("dimension_name"),
        "criteria": sample.get("criteria"),
        "original_answer": sample.get("original_answer"),
        "reference_answer": sample.get("reference_answer"),
        "candidate_a": sample.get("candidate_a"),
        "candidate_a_field": sample.get("candidate_a_field"),
        "candidate_b": sample.get("candidate_b"),
        "candidate_b_field": sample.get("candidate_b_field"),
        "candidate_b_source": sample.get("candidate_b_source"),
        "judge_model": model,
        "judge_endpoint": endpoint,
        "result": result,
        "reason": reason,
        "raw_output": raw_text,
        "parse_error": parse_error,
        "request_error": request_error,
        "latency_sec": round(time.time() - started, 4),
    }


def collect_samples(
    input_path: Path,
    args: argparse.Namespace,
    baseline_rows: Optional[dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = load_jsonl(input_path)
    if args.sample_size is not None:
        records = records[: max(0, args.sample_size)]

    samples: list[dict[str, Any]] = []
    stats = {
        "rows": len(records),
        "missing_candidate_a": 0,
        "missing_candidate_b": 0,
        "missing_baseline_row": 0,
    }
    for row in records:
        baseline_row = None
        if baseline_rows is not None:
            baseline_row = baseline_rows.get(row_key(row))
            if baseline_row is None:
                stats["missing_baseline_row"] += 1
                continue
        sample = build_sample(
            row,
            target_fields=args.target_fields,
            reference_fields=args.reference_fields,
            baseline_row=baseline_row,
            baseline_fields=args.baseline_fields,
        )
        if sample is None:
            candidate_a, _ = first_field(row, args.target_fields)
            if not candidate_a:
                stats["missing_candidate_a"] += 1
            else:
                stats["missing_candidate_b"] += 1
            continue
        samples.append(sample)
    return samples, stats


def summarize(input_path: Path, output_path: Path, rows: list[dict[str, Any]], stats: dict[str, int]) -> dict[str, Any]:
    wins = sum(1 for row in rows if row.get("result") == "win")
    ties = sum(1 for row in rows if row.get("result") == "tie")
    losses = sum(1 for row in rows if row.get("result") == "lose")
    judged = wins + ties + losses
    return {
        "name": input_path.stem,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "judged_rows": judged,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": round(wins / judged, 6) if judged else None,
        "tie_rate": round(ties / judged, 6) if judged else None,
        "loss_rate": round(losses / judged, 6) if judged else None,
        "win_rate_excluding_ties": round(wins / (wins + losses), 6) if wins + losses else None,
        "net_win_rate": round((wins - losses) / judged, 6) if judged else None,
        "request_failures": sum(1 for row in rows if row.get("request_error")),
        "parse_failures": sum(1 for row in rows if row.get("parse_error")),
        **stats,
    }


def process_file(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
    baseline_rows: Optional[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    samples, stats = collect_samples(input_path, args, baseline_rows)
    existing = load_existing(output_path)
    pending = [sample for sample in samples if sample["comparison_key"] not in existing]
    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc=input_path.stem, ncols=100, leave=False) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(judge_one, sample, index, args, base_urls, model, input_path): sample
                for index, sample in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                existing[row["comparison_key"]] = row
                append_jsonl(output_path, row, lock)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    ordered_rows = [existing[sample["comparison_key"]] for sample in samples if sample["comparison_key"] in existing]
    return summarize(input_path, output_path, ordered_rows, stats)


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary_json = output_dir / "indomain_generation_summary.json"
    summary_csv = output_dir / "indomain_generation_summary.csv"
    summary_json.write_text(json.dumps({"summaries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "name",
        "judged_rows",
        "wins",
        "ties",
        "losses",
        "win_rate",
        "tie_rate",
        "loss_rate",
        "win_rate_excluding_ties",
        "net_win_rate",
        "request_failures",
        "parse_failures",
        "rows",
        "missing_candidate_a",
        "missing_candidate_b",
        "missing_baseline_row",
        "input_file",
        "output_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: -(item["win_rate"] or -1)):
            writer.writerow({field: row.get(field) for field in fieldnames})
    return summary_json, summary_csv


def input_files(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [path.resolve() for path in args.inputs]
    if args.input:
        return [args.input.resolve()]
    return sorted(path for path in args.input_dir.resolve().glob("*.jsonl") if path.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate in-domain generated-answer ability with LLM WIN/TIE/LOSE judge.")
    parser.add_argument("--input", type=Path, default=None, help="Single score_reason_rewrite JSONL file.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/MATH500-Bench"), help="Directory of JSONL files.")
    parser.add_argument("--inputs", nargs="*", type=Path, default=[], help="Explicit input JSONL files.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline prediction JSONL. If omitted, Candidate B is the reference solution/answer.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <input_parent>/indomain_generation_judge.")
    parser.add_argument("--target-fields", nargs="+", default=["predicted_modified_answer", "modified_answer"], help="Candidate A fields.")
    parser.add_argument("--reference-fields", nargs="+", default=["reference_solution", "reference_answer", "gold_solution", "gold_answer"], help="Candidate B reference fields when --baseline is omitted.")
    parser.add_argument("--baseline-fields", nargs="+", default=["predicted_modified_answer", "modified_answer", "answer"], help="Candidate B fields when --baseline is provided.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=None, help="Judge ports like: 8000 8001 or 8000-8003.")
    parser.add_argument("--model", type=str, default="", help="Judge model id. If empty, fetch from /models.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = input_files(args)
    if not files:
        raise FileNotFoundError("No input JSONL files found.")

    output_dir = args.output_dir or (files[0].parent / "indomain_generation_judge")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if not args.skip_health_check:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    model = args.model.strip() or fetch_model_id(base_urls[0], args.health_check_timeout)
    if not model:
        raise ValueError("No judge model provided and /models auto-fetch failed.")

    baseline_rows = load_rows_by_key(args.baseline.resolve()) if args.baseline else None

    summaries: list[dict[str, Any]] = []
    iterator = tqdm(files, desc="indomain files") if tqdm is not None else files
    for input_path in iterator:
        output_path = output_dir / f"{input_path.stem}.jsonl"
        summaries.append(process_file(input_path, output_path, args, base_urls, model, baseline_rows))

    summary_json, summary_csv = write_summary(output_dir, summaries)
    print(f"Summary saved to: {summary_json}")
    print(f"CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
