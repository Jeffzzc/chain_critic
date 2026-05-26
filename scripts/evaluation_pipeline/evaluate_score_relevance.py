#!/usr/bin/env python
"""Evaluate predicted_score correlation against a baseline or ground-truth scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

from pipeline_common import load_jsonl, normalize_text


DEFAULT_INPUT_DIR = Path("datasets/MATH500/final")
DEFAULT_OUTPUT_DIR = Path("datasets/MATH500/score_relevance")
DEFAULT_BASELINE_HINT = "cortex-5"


def coalesce_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = normalize_text(row.get(key))
        if text:
            return text
    return ""


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


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

    numerator = sum(a * b for a, b in zip(dx, dy))
    return numerator / math.sqrt(sum_x2 * sum_y2)


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


def kendall_tau_b(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None

    unique_x = sorted(set(x))
    unique_y = sorted(set(y))
    x_index = {value: idx for idx, value in enumerate(unique_x)}
    y_index = {value: idx for idx, value in enumerate(unique_y)}

    counts = [[0 for _ in unique_y] for _ in unique_x]
    for x_value, y_value in zip(x, y):
        counts[x_index[x_value]][y_index[y_value]] += 1

    row_sums = [sum(row) for row in counts]
    col_sums = [sum(counts[i][j] for i in range(len(unique_x))) for j in range(len(unique_y))]
    tied_both = sum(cell * (cell - 1) // 2 for row in counts for cell in row)
    ties_x = sum(total * (total - 1) // 2 for total in row_sums) - tied_both
    ties_y = sum(total * (total - 1) // 2 for total in col_sums) - tied_both

    concordant = 0
    discordant = 0
    for i, row in enumerate(counts):
        for j, cell in enumerate(row):
            if cell == 0:
                continue
            lower_right = sum(
                counts[p][q]
                for p in range(i + 1, len(unique_x))
                for q in range(j + 1, len(unique_y))
            )
            lower_left = sum(
                counts[p][q]
                for p in range(i + 1, len(unique_x))
                for q in range(j)
            )
            concordant += cell * lower_right
            discordant += cell * lower_left

    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def row_key(row: dict[str, Any]) -> tuple[str, Any]:
    question = coalesce_text(row, "question", "orig_instruction", "instruction")
    answer = coalesce_text(row, "answer", "orig_response", "response")
    dimension = coalesce_text(row, "dimension_name", "evaluation_dimension", "orig_criteria", "criteria")
    if question and answer and dimension:
        raw = json.dumps(
            {
                "question": question,
                "answer": answer,
                "dimension": dimension,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return ("content", hashlib.sha1(raw.encode("utf-8")).hexdigest())

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
        "content": 0,
        "sample_id": 1,
        "unique_id_dimension": 2,
        "parent_dimension": 3,
        "index": 4,
    }
    return (order.get(key[0], 99), key[1])


def score_from_row(row: dict[str, Any]) -> Optional[float]:
    for key in ("predicted_score", "score", "reference_score", "orig_score"):
        score = as_finite_float(row.get(key))
        if score is not None:
            return score
    return None


def row_quality_score(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    predicted_score_ok = int(score_from_row(row) is not None)
    parse_ok = int(not row.get("parse_error"))
    request_ok = int(not row.get("request_error"))
    question_ok = int(bool(coalesce_text(row, "question", "orig_instruction", "instruction")))
    dimension_ok = int(bool(coalesce_text(row, "dimension_name", "evaluation_dimension", "orig_criteria", "criteria")))
    return (predicted_score_ok, parse_ok, request_ok, question_ok, dimension_ok)


def load_rows_by_key(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    rows: dict[tuple[str, Any], dict[str, Any]] = {}
    duplicate_count = 0

    for row in load_jsonl(path):
        key = row_key(row)
        if key in rows:
            duplicate_count += 1
            previous = rows[key]
            if row_quality_score(row) > row_quality_score(previous):
                rows[key] = row
            continue
        rows[key] = row

    if duplicate_count:
        print(f"[dedup] {path}: resolved {duplicate_count} duplicate rows by keeping higher-quality entries")

    return rows


def validate_aligned_rows(
    baseline_row: dict[str, Any],
    target_row: dict[str, Any],
    target_path: Path,
) -> None:
    baseline_question = coalesce_text(baseline_row, "question", "orig_instruction")
    target_question = coalesce_text(target_row, "question", "orig_instruction")
    if baseline_question and target_question and baseline_question != target_question:
        raise ValueError(
            f"Mismatched question for key={row_key(baseline_row)!r} in {target_path}"
        )

    baseline_dimension = coalesce_text(baseline_row, "dimension_name", "evaluation_dimension", "orig_criteria")
    target_dimension = coalesce_text(target_row, "dimension_name", "evaluation_dimension", "orig_criteria")
    if baseline_dimension and target_dimension and baseline_dimension != target_dimension:
        raise ValueError(
            f"Mismatched evaluation dimension for key={row_key(baseline_row)!r} in {target_path}"
        )


def compute_metrics(baseline_scores: list[float], target_scores: list[float]) -> dict[str, Any]:
    if len(baseline_scores) != len(target_scores):
        raise ValueError("Baseline and target score lengths do not match.")

    abs_errors = [abs(a - b) for a, b in zip(baseline_scores, target_scores)]
    sq_errors = [(a - b) ** 2 for a, b in zip(baseline_scores, target_scores)]
    exact_matches = sum(1 for a, b in zip(baseline_scores, target_scores) if a == b)

    return {
        "count": len(baseline_scores),
        "baseline_mean": format_float(average(baseline_scores)) if baseline_scores else None,
        "target_mean": format_float(average(target_scores)) if target_scores else None,
        "mean_difference": format_float(average(target_scores) - average(baseline_scores))
        if baseline_scores
        else None,
        "mae": format_float(average(abs_errors)) if abs_errors else None,
        "rmse": format_float(math.sqrt(average(sq_errors))) if sq_errors else None,
        "score_match_rate": format_float(exact_matches / len(baseline_scores)) if baseline_scores else None,
        "pearson": format_float(pearson_correlation(baseline_scores, target_scores)),
        "spearman": format_float(spearman_correlation(baseline_scores, target_scores)),
        "kendall_tau": format_float(kendall_tau_b(baseline_scores, target_scores)),
    }


def compare_prediction_file(
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
    baseline_missing_scores = 0
    target_missing_scores = 0
    baseline_request_failures = 0
    target_request_failures = 0
    baseline_parse_failures = 0
    target_parse_failures = 0

    for key in common_keys:
        baseline_row = baseline_rows[key]
        target_row = target_rows[key]
        validate_aligned_rows(baseline_row, target_row, target_path)

        baseline_score = score_from_row(baseline_row)
        target_score = score_from_row(target_row)

        if baseline_score is None:
            baseline_missing_scores += 1
        if target_score is None:
            target_missing_scores += 1
        if baseline_row.get("request_error"):
            baseline_request_failures += 1
        if target_row.get("request_error"):
            target_request_failures += 1
        if baseline_row.get("parse_error"):
            baseline_parse_failures += 1
        if target_row.get("parse_error"):
            target_parse_failures += 1

        if baseline_score is not None and target_score is not None:
            valid_pairs.append((baseline_score, target_score))

    baseline_scores = [pair[0] for pair in valid_pairs]
    target_scores = [pair[1] for pair in valid_pairs]
    first_target_row = target_rows[common_keys[0]]

    return {
        "baseline": {
            "name": baseline_path.stem,
            "prediction_file": str(baseline_path),
        },
        "target": {
            "name": target_path.stem,
            "model": normalize_text(
                first_target_row.get("model")
                or first_target_row.get("judge_model")
                or first_target_row.get("judge_name")
            )
            or target_path.stem,
            "prediction_file": str(target_path),
        },
        "common_samples": len(common_keys),
        "baseline_only_samples": len(baseline_rows) - len(common_keys),
        "target_only_samples": len(target_rows) - len(common_keys),
        "baseline_missing_scores": baseline_missing_scores,
        "target_missing_scores": target_missing_scores,
        "baseline_request_failures": baseline_request_failures,
        "target_request_failures": target_request_failures,
        "baseline_parse_failures": baseline_parse_failures,
        "target_parse_failures": target_parse_failures,
        "metrics": compute_metrics(baseline_scores, target_scores),
    }


def resolve_baseline_path(input_dir: Path, baseline: str) -> Path:
    direct = Path(baseline)
    if direct.is_file():
        return direct.resolve()

    joined = (input_dir / baseline)
    if joined.is_file():
        return joined.resolve()

    files = sorted(input_dir.glob("*.jsonl"))
    exact_matches = [path for path in files if path.stem == baseline or path.name == baseline]
    if len(exact_matches) == 1:
        return exact_matches[0].resolve()
    if len(exact_matches) > 1:
        raise ValueError(f"Multiple exact baseline matches found for {baseline!r}: {exact_matches}")

    fuzzy_matches = [path for path in files if baseline in path.stem or baseline in path.name]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0].resolve()
    if len(fuzzy_matches) > 1:
        names = ", ".join(path.name for path in fuzzy_matches)
        raise ValueError(f"Multiple baseline matches found for {baseline!r}: {names}")

    raise FileNotFoundError(f"Could not resolve baseline {baseline!r} under {input_dir}")


def discover_input_paths(
    *,
    input_dir: Path,
    explicit_inputs: Optional[list[Path]],
    baseline_path: Path,
    include_baseline: bool,
) -> list[Path]:
    if explicit_inputs:
        paths = [path.resolve() for path in explicit_inputs]
    else:
        paths = sorted(path.resolve() for path in input_dir.glob("*.jsonl"))

    filtered: list[Path] = []
    for path in paths:
        if not include_baseline and path == baseline_path:
            continue
        filtered.append(path)
    return filtered


def write_correlation_summary(
    output_dir: Path,
    baseline_path: Path,
    target_summaries: list[dict[str, Any]],
) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    summary_json = summaries_dir / "score_relevance_correlation_summary.json"
    summary_csv = summaries_dir / "score_relevance_correlation_summary.csv"

    ranking = sorted(
        target_summaries,
        key=lambda item: (
            item["metrics"]["spearman"] is None,
            -(item["metrics"]["spearman"] if item["metrics"]["spearman"] is not None else float("-inf")),
            -(item["metrics"]["pearson"] if item["metrics"]["pearson"] is not None else float("-inf")),
        ),
    )
    payload = {
        "baseline_or_ground_truth_file": str(baseline_path),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_summaries": target_summaries,
        "ranking_by_spearman": [
            {
                "target_name": item["target"]["name"],
                "target_model": item["target"]["model"],
                "valid_count": item["metrics"]["count"],
                "pearson": item["metrics"]["pearson"],
                "spearman": item["metrics"]["spearman"],
                "kendall_tau": item["metrics"]["kendall_tau"],
                "mae": item["metrics"]["mae"],
                "rmse": item["metrics"]["rmse"],
                "score_match_rate": item["metrics"]["score_match_rate"],
                "prediction_file": item["target"]["prediction_file"],
            }
            for item in ranking
        ],
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "target_name",
        "target_model",
        "valid_count",
        "pearson",
        "spearman",
        "kendall_tau",
        "baseline_mean",
        "target_mean",
        "mean_difference",
        "mae",
        "rmse",
        "score_match_rate",
        "common_samples",
        "baseline_only_samples",
        "target_only_samples",
        "baseline_missing_scores",
        "target_missing_scores",
        "baseline_request_failures",
        "target_request_failures",
        "baseline_parse_failures",
        "target_parse_failures",
        "prediction_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in ranking:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "target_name": item["target"]["name"],
                    "target_model": item["target"]["model"],
                    "valid_count": metrics["count"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "kendall_tau": metrics["kendall_tau"],
                    "baseline_mean": metrics["baseline_mean"],
                    "target_mean": metrics["target_mean"],
                    "mean_difference": metrics["mean_difference"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "score_match_rate": metrics["score_match_rate"],
                    "common_samples": item["common_samples"],
                    "baseline_only_samples": item["baseline_only_samples"],
                    "target_only_samples": item["target_only_samples"],
                    "baseline_missing_scores": item["baseline_missing_scores"],
                    "target_missing_scores": item["target_missing_scores"],
                    "baseline_request_failures": item["baseline_request_failures"],
                    "target_request_failures": item["target_request_failures"],
                    "baseline_parse_failures": item["baseline_parse_failures"],
                    "target_parse_failures": item["target_parse_failures"],
                    "prediction_file": item["target"]["prediction_file"],
                }
            )

    return summary_json, summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predicted_score JSONL files against a baseline prediction file or a ground-truth "
            "JSONL with score/orig_score, then report Pearson/Spearman/Kendall correlations."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing model prediction JSONL files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit JSONL files to compare. Defaults to all JSONL files in --input-dir.",
    )
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE_HINT,
        help=(
            "Baseline/ground-truth file path or fuzzy name/stem. Rows may contain predicted_score, "
            f"score, reference_score, or orig_score. Default resolves to the cortex-5 file under {DEFAULT_INPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Alias for --baseline when comparing predictions against a GT JSONL such as Feedback-Bench train.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for correlation summaries. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of sorted overlapping samples to include.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include the baseline file itself in the comparison set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    baseline_path = args.ground_truth.resolve() if args.ground_truth is not None else resolve_baseline_path(input_dir, args.baseline)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline/ground-truth file not found: {baseline_path}")
    input_paths = discover_input_paths(
        input_dir=input_dir,
        explicit_inputs=args.inputs,
        baseline_path=baseline_path,
        include_baseline=args.include_baseline,
    )
    if not input_paths:
        raise ValueError("No input prediction files found to compare.")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_summaries = [
        compare_prediction_file(
            baseline_path=baseline_path,
            target_path=path,
            sample_size=args.sample_size,
        )
        for path in input_paths
    ]
    summary_json, summary_csv = write_correlation_summary(output_dir, baseline_path, target_summaries)

    print(json.dumps(
        [
            {
                "target_name": item["target"]["name"],
                "target_model": item["target"]["model"],
                "pearson": item["metrics"]["pearson"],
                "spearman": item["metrics"]["spearman"],
                "kendall_tau": item["metrics"]["kendall_tau"],
                "mae": item["metrics"]["mae"],
                "score_match_rate": item["metrics"]["score_match_rate"],
                "valid_count": item["metrics"]["count"],
            }
            for item in sorted(
                target_summaries,
                key=lambda item: (
                    item["metrics"]["spearman"] is None,
                    -(item["metrics"]["spearman"] if item["metrics"]["spearman"] is not None else float("-inf")),
                    -(item["metrics"]["pearson"] if item["metrics"]["pearson"] is not None else float("-inf")),
                ),
            )
        ],
        ensure_ascii=False,
        indent=2,
    ))
    print(f"Summary saved to: {summary_json}")
    print(f"CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
