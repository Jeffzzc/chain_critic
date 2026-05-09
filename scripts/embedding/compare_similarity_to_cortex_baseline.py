#!/usr/bin/env python
"""Compare similarity JSONL files against the cortex-5 similarity baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_INPUT_DIR = Path("evaluation/baseline_new/predicted_reason_criteria_similarity")
DEFAULT_BASELINE_PATH = DEFAULT_INPUT_DIR / "openai_cortex-5_judge_outputs_with_reference.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "cortex_similarity_correlation"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            rows.append(payload)
    return rows


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute average of an empty list.")
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


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip())
        except ValueError:
            return None
    if not math.isfinite(number):
        return None
    return number


def normalize_index(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def row_key(row: dict[str, Any]) -> tuple[str, Any]:
    index = normalize_index(row.get("index"))
    if index is not None:
        return ("index", index)

    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return ("sample_id", sample_id.strip())

    raise ValueError(f"Row is missing both a usable 'index' and 'sample_id': {row}")


def load_rows_by_key(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    keyed_rows: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = row_key(row)
        if key in keyed_rows:
            raise ValueError(f"Duplicate key {key!r} found in {path}")
        keyed_rows[key] = row
    return keyed_rows


def sort_key(key: tuple[str, Any]) -> tuple[int, Any]:
    if key[0] == "index":
        return (0, key[1])
    return (1, str(key[1]))


def select_common_keys(
    all_common_keys: list[tuple[str, Any]],
    sample_size: Optional[int],
    sample_mode: str,
    seed: int,
) -> list[tuple[str, Any]]:
    if sample_size is None or sample_size >= len(all_common_keys):
        return list(all_common_keys)

    if sample_size < 0:
        raise ValueError("--sample-size must be >= 0")

    if sample_mode == "head":
        return all_common_keys[:sample_size]

    if sample_mode == "random":
        rng = random.Random(seed)
        sampled = rng.sample(all_common_keys, sample_size)
        return sorted(sampled, key=sort_key)

    raise ValueError(f"Unsupported sample mode: {sample_mode}")


def compute_metrics(baseline_values: list[float], target_values: list[float]) -> dict[str, Any]:
    if len(baseline_values) != len(target_values):
        raise ValueError("Baseline and target similarity lengths do not match.")

    if not baseline_values:
        return {
            "count": 0,
            "baseline_mean": None,
            "target_mean": None,
            "mean_difference": None,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
        }

    differences = [target - baseline for baseline, target in zip(baseline_values, target_values)]
    abs_differences = [abs(value) for value in differences]
    sq_differences = [value * value for value in differences]

    return {
        "count": len(baseline_values),
        "baseline_mean": format_float(average(baseline_values)),
        "target_mean": format_float(average(target_values)),
        "mean_difference": format_float(average(differences)),
        "mae": format_float(average(abs_differences)),
        "rmse": format_float(math.sqrt(average(sq_differences))),
        "pearson": format_float(pearson_correlation(baseline_values, target_values)),
        "spearman": format_float(spearman_correlation(baseline_values, target_values)),
    }


def compare_similarity_file(
    baseline_path: Path,
    target_path: Path,
    sample_size: Optional[int],
    sample_mode: str,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = load_rows_by_key(baseline_path)
    target_rows = load_rows_by_key(target_path)

    all_common_keys = sorted(set(baseline_rows) & set(target_rows), key=sort_key)
    if not all_common_keys:
        raise ValueError(f"No overlapping rows between {baseline_path} and {target_path}")

    evaluated_keys = select_common_keys(
        all_common_keys=all_common_keys,
        sample_size=sample_size,
        sample_mode=sample_mode,
        seed=seed,
    )

    valid_pairs: list[tuple[float, float]] = []
    baseline_missing_similarity = 0
    target_missing_similarity = 0
    baseline_embedding_errors = 0
    target_embedding_errors = 0
    baseline_criteria_parse_errors = 0
    target_criteria_parse_errors = 0

    for key in evaluated_keys:
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
        "common_samples": len(all_common_keys),
        "evaluated_samples": len(evaluated_keys),
        "baseline_only_samples": len(baseline_rows) - len(all_common_keys),
        "target_only_samples": len(target_rows) - len(all_common_keys),
        "baseline_missing_similarity": baseline_missing_similarity,
        "target_missing_similarity": target_missing_similarity,
        "baseline_embedding_errors": baseline_embedding_errors,
        "target_embedding_errors": target_embedding_errors,
        "baseline_criteria_parse_errors": baseline_criteria_parse_errors,
        "target_criteria_parse_errors": target_criteria_parse_errors,
        "metrics": compute_metrics(baseline_values, target_values),
    }


def discover_input_paths(input_dir: Path, baseline_path: Path, include_baseline: bool) -> list[Path]:
    paths = sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())
    if include_baseline:
        return paths
    baseline_resolved = baseline_path.resolve()
    return [path for path in paths if path.resolve() != baseline_resolved]


def metric_sort_key(item: dict[str, Any]) -> tuple[bool, float, float, str]:
    spearman = item["metrics"]["spearman"]
    pearson = item["metrics"]["pearson"]
    return (
        spearman is None,
        -(spearman if spearman is not None else float("-inf")),
        -(pearson if pearson is not None else float("-inf")),
        item["target"]["name"],
    )


def build_experiment_summary(
    baseline_path: Path,
    input_paths: list[Path],
    sample_size: Optional[int],
    sample_mode: str,
    seed: int,
    target_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    ranking = sorted(target_summaries, key=metric_sort_key)
    return {
        "baseline_path": str(baseline_path),
        "similarity_files": [str(path) for path in input_paths],
        "sample_size": sample_size,
        "sample_mode": sample_mode,
        "seed": seed,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_summaries": target_summaries,
        "ranking_by_spearman": [
            {
                "target_name": item["target"]["name"],
                "valid_count": item["metrics"]["count"],
                "common_samples": item["common_samples"],
                "evaluated_samples": item["evaluated_samples"],
                "pearson": item["metrics"]["pearson"],
                "spearman": item["metrics"]["spearman"],
                "baseline_mean": item["metrics"]["baseline_mean"],
                "target_mean": item["metrics"]["target_mean"],
                "mean_difference": item["metrics"]["mean_difference"],
                "mae": item["metrics"]["mae"],
                "rmse": item["metrics"]["rmse"],
                "similarity_file": item["target"]["similarity_file"],
            }
            for item in ranking
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare each predicted_reason/criteria similarity file against the cortex-5 "
            "similarity baseline and report Pearson/Spearman correlations."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing similarity JSONL files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help=f"Baseline similarity JSONL file. Default: {DEFAULT_BASELINE_PATH}",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional explicit similarity JSONL files to compare. "
            "Defaults to all JSONL files in --input-dir except baseline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary.json and summary.csv. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of overlapping samples to include.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("head", "random"),
        default="head",
        help="How to choose samples when --sample-size is set. Default: head",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-mode=random. Default: 42",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include the baseline file itself in comparisons when --inputs is not provided.",
    )
    return parser.parse_args()


def write_csv(path: Path, target_summaries: list[dict[str, Any]]) -> None:
    sorted_items = sorted(target_summaries, key=metric_sort_key)
    fieldnames = [
        "target_name",
        "valid_count",
        "common_samples",
        "evaluated_samples",
        "pearson",
        "spearman",
        "baseline_mean",
        "target_mean",
        "mean_difference",
        "mae",
        "rmse",
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted_items:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "target_name": item["target"]["name"],
                    "valid_count": metrics["count"],
                    "common_samples": item["common_samples"],
                    "evaluated_samples": item["evaluated_samples"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "baseline_mean": metrics["baseline_mean"],
                    "target_mean": metrics["target_mean"],
                    "mean_difference": metrics["mean_difference"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
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


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    baseline_path = args.baseline.resolve()
    output_dir = args.output_dir.resolve()

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline similarity file not found: {baseline_path}")

    if args.inputs is None:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        input_paths = discover_input_paths(input_dir, baseline_path, args.include_baseline)
    else:
        input_paths = [path.resolve() for path in args.inputs]

    if not input_paths:
        raise ValueError("No input similarity files to compare.")

    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input similarity file not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    target_summaries = [
        compare_similarity_file(
            baseline_path=baseline_path,
            target_path=path,
            sample_size=args.sample_size,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )
        for path in input_paths
    ]

    experiment_summary = build_experiment_summary(
        baseline_path=baseline_path,
        input_paths=input_paths,
        sample_size=args.sample_size,
        sample_mode=args.sample_mode,
        seed=args.seed,
        target_summaries=target_summaries,
    )

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    summary_path.write_text(
        json.dumps(experiment_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(csv_path, target_summaries)

    print(json.dumps(experiment_summary["ranking_by_spearman"], ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()