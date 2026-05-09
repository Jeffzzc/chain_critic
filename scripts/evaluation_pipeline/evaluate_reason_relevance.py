#!/usr/bin/env python
"""Evaluate reason-criteria relevance and compare it to a cortex-5 baseline.

This script merges the functionality of scripts/embedding:
1. Read prediction JSONL files, usually from datasets/MATH500/final.
2. For each row, extract predicted_reason and the 0-5 criterion matching
   predicted_score.
3. Embed both texts through OpenAI-compatible /embeddings endpoints and compute
   cosine similarity.
4. Use the cortex-5 similarity file as baseline and compute Pearson/Spearman
   correlations for each target model on common valid samples.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
from typing import Any, Optional

from pipeline_common import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    build_base_urls,
    cosine_similarity,
    embed_texts_with_retries,
    fetch_model_id,
    load_jsonl,
    normalize_score_criteria,
    normalize_text,
    parse_ports,
    wait_for_servers,
    write_jsonl,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT_DIR = Path("datasets/MATH500/final")
DEFAULT_OUTPUT_DIR = Path("datasets/MATH500/reason_relevance")
DEFAULT_BASELINE_HINT = "cortex-5"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

CRITERION_PATTERN = re.compile(
    r"(?ms)^\s*(?:Score\s*)?(?P<score>[0-5])\s*[:\uFF1A]\s*(?P<body>.*?)(?=^\s*(?:Score\s*)?[0-5]\s*[:\uFF1A]|\Z)"
)


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def resolve_api_key(raw_api_key: str, api_key_env: str) -> str:
    explicit = raw_api_key.strip()
    if explicit:
        return explicit
    from_env = os.getenv(api_key_env, "").strip()
    return from_env or DEFAULT_API_KEY


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = average(x)
    mean_y = average(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    sum_x2 = sum(value * value for value in dx)
    sum_y2 = sum(value * value for value in dy)
    if sum_x2 == 0 or sum_y2 == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sum_x2 * sum_y2)


def rankdata(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson_correlation(rankdata(x), rankdata(y))


def as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def score_key(score: Any) -> Optional[str]:
    if isinstance(score, bool) or score is None:
        return None
    if isinstance(score, (int, float)) and math.isfinite(float(score)):
        rounded = round(float(score))
        if abs(float(score) - rounded) < 1e-6 and 0 <= rounded <= 5:
            return str(int(rounded))
        return None
    text = str(score).strip()
    if re.fullmatch(r"[0-5]", text):
        return text
    match = re.search(r"\b([0-5])\b", text)
    return match.group(1) if match else None


def row_key(row: dict[str, Any]) -> tuple[str, Any]:
    sample_id = normalize_text(row.get("sample_id"))
    if sample_id:
        return ("sample_id", sample_id)
    unique_id = normalize_text(row.get("unique_id"))
    dimension = normalize_text(row.get("dimension_name") or row.get("evaluation_dimension"))
    if unique_id and dimension:
        return ("unique_id_dimension", f"{unique_id}::{dimension}")
    parent_sample_id = normalize_text(row.get("parent_sample_id"))
    if parent_sample_id and dimension:
        return ("parent_dimension", f"{parent_sample_id}::{dimension}")
    index = row.get("index")
    if isinstance(index, int):
        return ("index", index)
    raise ValueError(f"Row is missing a usable key: {row}")


def sort_key(key: tuple[str, Any]) -> tuple[int, Any]:
    order = {
        "sample_id": 0,
        "unique_id_dimension": 1,
        "parent_dimension": 2,
        "index": 3,
    }
    return (order.get(key[0], 99), key[1])


def row_quality_score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    similarity_ok = int(as_finite_float(row.get("similarity")) is not None)
    predicted_score_ok = int(score_key(row.get("predicted_score")) is not None)
    reason_ok = int(bool(normalize_text(row.get("predicted_reason"))))
    criterion_ok = int(bool(normalize_text(row.get("matched_criterion"))))
    clean_ok = int(not row.get("criteria_parse_error") and not row.get("embedding_error"))
    return (similarity_ok, predicted_score_ok, reason_ok, criterion_ok, clean_ok)

def load_rows_by_key(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    rows: dict[tuple[str, Any], dict[str, Any]] = {}
    duplicate_count = 0

    for row in load_jsonl(path):
        key = row_key(row)
        if key in rows:
            duplicate_count += 1
            prev = rows[key]
            if row_quality_score(row) > row_quality_score(prev):
                rows[key] = row
            continue
        rows[key] = row

    if duplicate_count:
        print(f"[dedup] {path}: resolved {duplicate_count} duplicate rows by keeping higher-quality entries")

    return rows


def extract_criterion(row: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    target_score = score_key(row.get("predicted_score"))
    if target_score is None:
        return None, "missing or invalid predicted_score"

    score_criteria = row.get("score_criteria")
    if isinstance(score_criteria, dict):
        criterion = normalize_text(score_criteria.get(target_score) or score_criteria.get(int(target_score)))
        if criterion:
            return f"{target_score}: {criterion}", None

    normalized = normalize_score_criteria(row, normalize_text(row.get("full_score_criteria")))
    criterion = normalize_text(normalized.get(target_score))
    if criterion:
        return f"{target_score}: {criterion}", None

    criteria_text = normalize_text(row.get("criteria_text") or row.get("criteria"))
    if not criteria_text:
        return None, "missing criteria"

    for match in CRITERION_PATTERN.finditer(criteria_text):
        if match.group("score") == target_score:
            body = normalize_text(match.group("body"))
            if body:
                return f"{target_score}: {body}", None
            return None, f"empty criterion for score {target_score}"

    return None, f"criterion for score {target_score} not found"


def build_similarity_row(record: dict[str, Any]) -> dict[str, Any]:
    reason = normalize_text(record.get("predicted_reason") or record.get("reason"))
    matched_criterion, parse_error = extract_criterion(record)
    return {
        "sample_id": record.get("sample_id"),
        "index": record.get("index"),
        "unique_id": record.get("unique_id"),
        "parent_sample_id": record.get("parent_sample_id"),
        "subject": record.get("subject"),
        "level": record.get("level"),
        "question": record.get("question"),
        "dimension_name": record.get("dimension_name") or record.get("evaluation_dimension"),
        "predicted_score": record.get("predicted_score"),
        "predicted_reason": reason or None,
        "matched_criterion": matched_criterion,
        "similarity": None,
        "criteria_parse_error": parse_error,
        "embedding_error": None,
    }


def endpoint_for_batch(base_urls: list[str], batch_index: int) -> str:
    return base_urls[batch_index % len(base_urls)]


def fill_similarity_scores(
    rows: list[dict[str, Any]],
    *,
    executor: ThreadPoolExecutor,
    base_urls: list[str],
    model: str,
    api_key: str,
    batch_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    include_embeddings: bool,
) -> None:
    pending: list[tuple[int, str, str]] = []
    for row_index, row in enumerate(rows):
        reason = row.get("predicted_reason")
        criterion = row.get("matched_criterion")
        if not reason:
            row["embedding_error"] = "missing predicted_reason"
            continue
        if not criterion:
            continue
        pending.append((row_index, "reason", str(reason)))
        pending.append((row_index, "criterion", str(criterion)))

    future_to_items = {}
    for batch_index, start in enumerate(range(0, len(pending), batch_size)):
        batch_items = pending[start : start + batch_size]
        future = executor.submit(
            embed_texts_with_retries,
            base_url=endpoint_for_batch(base_urls, batch_index),
            api_key=api_key,
            model=model,
            texts=[item[2] for item in batch_items],
            timeout_seconds=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        future_to_items[future] = batch_items

    embeddings_by_row: dict[int, dict[str, list[float]]] = {}
    for future in as_completed(future_to_items):
        batch_items = future_to_items[future]
        try:
            embeddings = future.result()
        except Exception as exc:
            for row_index, _, _ in batch_items:
                rows[row_index]["embedding_error"] = str(exc)
            continue
        for (row_index, name, _), embedding in zip(batch_items, embeddings):
            embeddings_by_row.setdefault(row_index, {})[name] = embedding

    for row_index, embeddings in embeddings_by_row.items():
        reason_embedding = embeddings.get("reason")
        criterion_embedding = embeddings.get("criterion")
        if reason_embedding is None or criterion_embedding is None:
            rows[row_index]["embedding_error"] = rows[row_index]["embedding_error"] or "missing embedding result"
            continue
        rows[row_index]["similarity"] = cosine_similarity(reason_embedding, criterion_embedding)
        if include_embeddings:
            rows[row_index]["predicted_reason_embedding"] = reason_embedding
            rows[row_index]["matched_criterion_embedding"] = criterion_embedding


def similarity_output_path(output_dir: Path, input_path: Path) -> Path:
    return output_dir / "similarities" / f"{input_path.stem}.jsonl"


def summarize_similarity_rows(input_path: Path, output_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["similarity"])
        for row in rows
        if isinstance(row.get("similarity"), (int, float)) and math.isfinite(float(row["similarity"]))
    ]
    return {
        "name": input_path.stem,
        "input_file": str(input_path),
        "similarity_file": str(output_path),
        "rows": len(rows),
        "valid_count": len(values),
        "missing_reason": sum(1 for row in rows if not row.get("predicted_reason")),
        "criteria_parse_errors": sum(1 for row in rows if row.get("criteria_parse_error")),
        "embedding_errors": sum(1 for row in rows if row.get("embedding_error")),
        "mean_similarity": format_float(average(values)) if values else None,
        "median_similarity": format_float(statistics.median(values)) if values else None,
        "min_similarity": format_float(min(values)) if values else None,
        "max_similarity": format_float(max(values)) if values else None,
    }


def process_prediction_file(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    records = load_jsonl(input_path)
    if args.sample_size is not None:
        records = records[: max(0, args.sample_size)]

    rows = [build_similarity_row(record) for record in records]
    fill_similarity_scores(
        rows,
        executor=executor,
        base_urls=base_urls,
        model=model,
        api_key=args.api_key,
        batch_size=args.batch_size,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        include_embeddings=args.include_embeddings,
    )
    write_jsonl(output_path, rows)
    return summarize_similarity_rows(input_path, output_path, rows)


def discover_prediction_files(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return [path.resolve() for path in args.inputs]
    if args.input:
        return [args.input.resolve()]
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())


def resolve_baseline_prediction_path(files: list[Path], baseline: str) -> Path:
    if baseline:
        path = Path(baseline)
        if path.is_file():
            return path.resolve()

        normalized = baseline.lower()
        candidates = [
            file
            for file in files
            if file.name.lower() == normalized
            or file.stem.lower() == normalized
            or normalized in file.name.lower()
            or normalized in file.stem.lower()
        ]
        if len(candidates) == 1:
            return candidates[0].resolve()
        if len(candidates) > 1:
            names = ", ".join(path.name for path in candidates)
            raise ValueError(f"Baseline hint {baseline!r} matched multiple files: {names}")
        raise FileNotFoundError(f"Baseline prediction file not found or not matched: {baseline}")

    candidates = [file for file in files if DEFAULT_BASELINE_HINT in file.name.lower()]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise ValueError("Unable to auto-detect cortex baseline. Pass --baseline with a path, stem, or filename hint.")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Multiple cortex-like baseline files found: {names}. Pass --baseline explicitly.")


def compute_correlation_metrics(baseline_values: list[float], target_values: list[float]) -> dict[str, Any]:
    differences = [target - baseline for baseline, target in zip(baseline_values, target_values)]
    abs_differences = [abs(value) for value in differences]
    sq_differences = [value * value for value in differences]
    return {
        "count": len(baseline_values),
        "baseline_mean": format_float(average(baseline_values)) if baseline_values else None,
        "target_mean": format_float(average(target_values)) if target_values else None,
        "mean_difference": format_float(average(differences)) if differences else None,
        "mae": format_float(average(abs_differences)) if abs_differences else None,
        "rmse": format_float(math.sqrt(average(sq_differences))) if sq_differences else None,
        "pearson": format_float(pearson_correlation(baseline_values, target_values)),
        "spearman": format_float(spearman_correlation(baseline_values, target_values)),
    }


def compare_similarity_file(
    baseline_path: Path,
    target_path: Path,
    sample_size: Optional[int],
) -> dict[str, Any]:
    baseline_rows = load_rows_by_key(baseline_path)
    target_rows = load_rows_by_key(target_path)
    common_keys = sorted(set(baseline_rows) & set(target_rows), key=sort_key)
    if sample_size is not None:
        common_keys = common_keys[: max(0, sample_size)]
    if not common_keys:
        raise ValueError(f"No overlapping rows between {baseline_path} and {target_path}")

    valid_pairs: list[tuple[float, float]] = []
    baseline_missing_similarity = 0
    target_missing_similarity = 0
    baseline_embedding_errors = 0
    target_embedding_errors = 0
    baseline_criteria_parse_errors = 0
    target_criteria_parse_errors = 0

    for key in common_keys:
        baseline_row = baseline_rows[key]
        target_row = target_rows[key]
        baseline_similarity = as_finite_float(baseline_row.get("similarity"))
        target_similarity = as_finite_float(target_row.get("similarity"))

        if baseline_similarity is None:
            baseline_missing_similarity += 1
        if target_similarity is None:
            target_missing_similarity += 1
        if baseline_row.get("embedding_error"):
            baseline_embedding_errors += 1
        if target_row.get("embedding_error"):
            target_embedding_errors += 1
        if baseline_row.get("criteria_parse_error"):
            baseline_criteria_parse_errors += 1
        if target_row.get("criteria_parse_error"):
            target_criteria_parse_errors += 1
        if baseline_similarity is not None and target_similarity is not None:
            valid_pairs.append((baseline_similarity, target_similarity))

    baseline_values = [pair[0] for pair in valid_pairs]
    target_values = [pair[1] for pair in valid_pairs]

    return {
        "baseline": {
            "name": baseline_path.stem,
            "similarity_file": str(baseline_path),
        },
        "target": {
            "name": target_path.stem,
            "similarity_file": str(target_path),
        },
        "common_samples": len(common_keys),
        "baseline_only_samples": len(baseline_rows) - len(common_keys),
        "target_only_samples": len(target_rows) - len(common_keys),
        "baseline_missing_similarity": baseline_missing_similarity,
        "target_missing_similarity": target_missing_similarity,
        "baseline_embedding_errors": baseline_embedding_errors,
        "target_embedding_errors": target_embedding_errors,
        "baseline_criteria_parse_errors": baseline_criteria_parse_errors,
        "target_criteria_parse_errors": target_criteria_parse_errors,
        "metrics": compute_correlation_metrics(baseline_values, target_values),
    }


def write_generation_summary(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summaries_dir / "reason_similarity_generation_summary.json"
    summary_csv = summaries_dir / "reason_similarity_generation_summary.csv"
    summary_json.write_text(json.dumps({"summaries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "name",
        "valid_count",
        "mean_similarity",
        "median_similarity",
        "min_similarity",
        "max_similarity",
        "rows",
        "missing_reason",
        "criteria_parse_errors",
        "embedding_errors",
        "input_file",
        "similarity_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: -(item["mean_similarity"] or -1)):
            writer.writerow({field: row.get(field) for field in fieldnames})
    return summary_json, summary_csv


def write_correlation_summary(
    output_dir: Path,
    baseline_similarity_path: Path,
    target_summaries: list[dict[str, Any]],
) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summaries_dir / "reason_relevance_correlation_summary.json"
    summary_csv = summaries_dir / "reason_relevance_correlation_summary.csv"

    ranking = sorted(
        target_summaries,
        key=lambda item: (
            item["metrics"]["spearman"] is None,
            -(item["metrics"]["spearman"] if item["metrics"]["spearman"] is not None else float("-inf")),
            -(item["metrics"]["pearson"] if item["metrics"]["pearson"] is not None else float("-inf")),
        ),
    )
    payload = {
        "baseline_similarity_file": str(baseline_similarity_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_summaries": target_summaries,
        "ranking_by_spearman": [
            {
                "target_name": item["target"]["name"],
                "valid_count": item["metrics"]["count"],
                "pearson": item["metrics"]["pearson"],
                "spearman": item["metrics"]["spearman"],
                "baseline_mean": item["metrics"]["baseline_mean"],
                "target_mean": item["metrics"]["target_mean"],
                "mean_difference": item["metrics"]["mean_difference"],
                "similarity_file": item["target"]["similarity_file"],
            }
            for item in ranking
        ],
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "target_name",
        "valid_count",
        "pearson",
        "spearman",
        "baseline_mean",
        "target_mean",
        "mean_difference",
        "mae",
        "rmse",
        "common_samples",
        "baseline_only_samples",
        "target_only_samples",
        "baseline_missing_similarity",
        "target_missing_similarity",
        "baseline_embedding_errors",
        "target_embedding_errors",
        "baseline_criteria_parse_errors",
        "target_criteria_parse_errors",
        "similarity_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranking:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "target_name": item["target"]["name"],
                    "valid_count": metrics["count"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "baseline_mean": metrics["baseline_mean"],
                    "target_mean": metrics["target_mean"],
                    "mean_difference": metrics["mean_difference"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "common_samples": item["common_samples"],
                    "baseline_only_samples": item["baseline_only_samples"],
                    "target_only_samples": item["target_only_samples"],
                    "baseline_missing_similarity": item["baseline_missing_similarity"],
                    "target_missing_similarity": item["target_missing_similarity"],
                    "baseline_embedding_errors": item["baseline_embedding_errors"],
                    "target_embedding_errors": item["target_embedding_errors"],
                    "baseline_criteria_parse_errors": item["baseline_criteria_parse_errors"],
                    "target_criteria_parse_errors": item["target_criteria_parse_errors"],
                    "similarity_file": item["target"]["similarity_file"],
                }
            )
    return summary_json, summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate predicted_reason/score-criterion similarities for prediction files, "
            "then compare every model's similarity sequence against a cortex-5 baseline "
            "with Pearson and Spearman correlations."
        )
    )
    parser.add_argument("--input", type=Path, default=None, help="Single raw prediction JSONL file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing raw prediction JSONL files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument("--inputs", nargs="*", type=Path, default=[], help="Explicit raw prediction JSONL files.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="",
        help=(
            "Cortex-5 baseline raw prediction file path, stem, filename, or substring. "
            "If omitted, auto-detects a file containing 'cortex'."
        ),
    )
    parser.add_argument(
        "--baseline-similarity",
        type=Path,
        default=None,
        help="Optional precomputed cortex-5 similarity JSONL. If omitted, the baseline raw file is embedded too.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for similarities and summaries. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row limit per raw file.")
    parser.add_argument("--include-baseline", action="store_true", help="Include baseline in correlation comparison.")

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8001-8004"], help="Embedding ports like: 8000 8001 or 8000-8003.")
    parser.add_argument("--model", type=str, default="", help="Embedding model id. If empty, fetch from /models.")
    parser.add_argument("--api-key", type=str, default="", help="Embedding API key. Defaults to --api-key-env, then EMPTY.")
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers-per-endpoint", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.api_key = resolve_api_key(args.api_key, args.api_key_env)
    prediction_files = discover_prediction_files(args)
    if not prediction_files:
        raise FileNotFoundError("No raw prediction JSONL files found.")

    output_dir = args.output_dir.resolve()
    similarities_dir = output_dir / "similarities"
    similarities_dir.mkdir(parents=True, exist_ok=True)

    baseline_prediction_path = resolve_baseline_prediction_path(prediction_files, args.baseline)
    prediction_files = sorted(set(path.resolve() for path in prediction_files) | {baseline_prediction_path.resolve()})

    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if not args.skip_health_check:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    model = args.model.strip() or fetch_model_id(base_urls[0], args.health_check_timeout)
    if not model:
        raise ValueError("No embedding model provided and /models auto-fetch failed.")

    print(f"[input_files] {len(prediction_files)}")
    print(f"[baseline_prediction] {baseline_prediction_path}")
    print(f"[output_dir] {output_dir}")
    print(f"[embedding_model] {model}")
    print(f"[embedding_base_urls] {base_urls}")

    generation_summaries: list[dict[str, Any]] = []
    max_workers = max(1, len(base_urls) * args.workers_per_endpoint)
    iterator = tqdm(prediction_files, desc="similarity files") if tqdm is not None else prediction_files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for input_path in iterator:
            output_path = similarity_output_path(output_dir, input_path)
            if output_path.exists() and not args.overwrite:
                generation_summaries.append(summarize_similarity_rows(input_path, output_path, load_jsonl(output_path)))
                continue
            generation_summaries.append(process_prediction_file(input_path, output_path, args, executor, base_urls, model))

    gen_json, gen_csv = write_generation_summary(output_dir, generation_summaries)

    baseline_similarity_path = (
        args.baseline_similarity.resolve()
        if args.baseline_similarity is not None
        else similarity_output_path(output_dir, baseline_prediction_path).resolve()
    )
    if not baseline_similarity_path.is_file():
        raise FileNotFoundError(f"Baseline similarity file not found: {baseline_similarity_path}")

    target_similarity_paths = [similarity_output_path(output_dir, path).resolve() for path in prediction_files]
    if not args.include_baseline:
        target_similarity_paths = [
            path for path in target_similarity_paths if path.resolve() != baseline_similarity_path.resolve()
        ]

    correlation_summaries = [
        compare_similarity_file(
            baseline_path=baseline_similarity_path,
            target_path=target_path,
            sample_size=args.sample_size,
        )
        for target_path in target_similarity_paths
    ]
    corr_json, corr_csv = write_correlation_summary(output_dir, baseline_similarity_path, correlation_summaries)

    print(f"Similarity generation summary saved to: {gen_json}")
    print(f"Similarity generation CSV saved to: {gen_csv}")
    print(f"Correlation summary saved to: {corr_json}")
    print(f"Correlation CSV saved to: {corr_csv}")


if __name__ == "__main__":
    main()
