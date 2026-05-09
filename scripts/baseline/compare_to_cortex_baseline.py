#!/usr/bin/env python
"""Compare judge prediction files against a cortex-5 baseline file."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional


DEFAULT_BASELINE_PATH = Path(
    # "evaluation/baseline_new/predictions/openai_cortex-5_judge_outputs_with_reference.jsonl"
    "evaluation/baseline_new/mismatch/openai_cortex-5_judge_outputs_with_reference_backfilled_from_adjudicated_selected.jsonl"
)
DEFAULT_INPUT_PATHS = [
    Path("evaluation/baseline_new/predictions/Qwen2.5-7B-Instruct.jsonl"),
    Path("evaluation/baseline_new/predictions/Qwen2.5-14B-Instruct.jsonl"),
    Path("evaluation/baseline_new/predictions/Qwen2.5-32B-Instruct.jsonl"),
    Path("evaluation/baseline_new/predictions/Qwen3.5-9B.jsonl"),
    Path("evaluation/baseline_new/predictions/Mistral-7B-Instruct-v0.3.jsonl"),
    Path("evaluation/baseline_new/predictions/Ministral-3-14B-Instruct-2512.jsonl"),
    Path("evaluation/baseline_new/predictions/chaincritic-v8-6000.jsonl"),
    Path("evaluation/baseline_new/predictions/chaincritic-v10-20260407-105024.jsonl"),
    Path("evaluation/baseline_new/predictions/chaincritic-v11-70669.jsonl"),
    Path("evaluation/baseline_new/predictions/chaincritic-v11-62000.jsonl"),
    Path("evaluation/baseline_new/predictions/chaincritic-grpo-v11-11800.jsonl"),
    Path("evaluation/baseline_new/predictions/chaincritic-grpo-v11-1000.jsonl")

]
# DEFAULT_OUTPUT_DIR = Path("evaluation/baseline_new/predictions/summaries/cortex_baseline")
DEFAULT_OUTPUT_DIR = Path("evaluation/baseline_new/predictions/summaries/GT_baseline")


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


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def compute_metrics(baseline_scores: list[float], model_scores: list[float]) -> dict[str, Any]:
    if len(baseline_scores) != len(model_scores):
        raise ValueError("Baseline and model score lengths do not match.")

    abs_errors = [abs(a - b) for a, b in zip(baseline_scores, model_scores)]
    sq_errors = [(a - b) ** 2 for a, b in zip(baseline_scores, model_scores)]
    exact_matches = sum(1 for a, b in zip(baseline_scores, model_scores) if a == b)

    return {
        "count": len(baseline_scores),
        "baseline_mean": format_float(average(baseline_scores)) if baseline_scores else None,
        "model_mean": format_float(average(model_scores)) if model_scores else None,
        "mae": format_float(average(abs_errors)) if abs_errors else None,
        "rmse": format_float(math.sqrt(average(sq_errors))) if sq_errors else None,
        "score_match_rate": format_float(exact_matches / len(baseline_scores)) if baseline_scores else None,
        "pearson": format_float(pearson_correlation(baseline_scores, model_scores)),
        "spearman": format_float(spearman_correlation(baseline_scores, model_scores)),
        "kendall_tau": format_float(kendall_tau_b(baseline_scores, model_scores)),
    }


def row_key(row: dict[str, Any]) -> tuple[str, Any]:
    index = row.get("index")
    if isinstance(index, int):
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


def validate_aligned_rows(
    baseline_row: dict[str, Any],
    model_row: dict[str, Any],
    target_path: Path,
) -> None:
    baseline_question = str(baseline_row.get("question", "")).strip()
    model_question = str(model_row.get("question", "")).strip()
    if baseline_question and model_question and baseline_question != model_question:
        raise ValueError(
            f"Mismatched question for index={baseline_row.get('index')} in {target_path}"
        )

    baseline_dimension = str(baseline_row.get("evaluation_dimension", "")).strip()
    model_dimension = str(model_row.get("evaluation_dimension", "")).strip()
    if baseline_dimension and model_dimension and baseline_dimension != model_dimension:
        raise ValueError(
            f"Mismatched evaluation_dimension for index={baseline_row.get('index')} in {target_path}"
        )


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
    for key in common_keys:
        baseline_row = baseline_rows[key]
        model_row = target_rows[key]
        validate_aligned_rows(baseline_row, model_row, target_path)

        baseline_score = baseline_row.get("predicted_score")
        model_score = model_row.get("predicted_score")
        if isinstance(baseline_score, (int, float)) and isinstance(model_score, (int, float)):
            valid_pairs.append((float(baseline_score), float(model_score)))

    baseline_scores = [pair[0] for pair in valid_pairs]
    model_scores = [pair[1] for pair in valid_pairs]

    ordered_target_rows = [target_rows[key] for key in common_keys]
    request_failures = sum(1 for row in ordered_target_rows if row.get("request_error"))
    parse_failures = sum(1 for row in ordered_target_rows if row.get("parse_error"))
    baseline_request_failures = sum(1 for key in common_keys if baseline_rows[key].get("request_error"))
    baseline_parse_failures = sum(1 for key in common_keys if baseline_rows[key].get("parse_error"))

    first_target_row = ordered_target_rows[0]
    return {
        "baseline": {
            "name": str(baseline_path.stem),
            "prediction_file": str(baseline_path),
        },
        "judge": {
            "name": str(first_target_row.get("judge_name", "")).strip() or target_path.stem,
            "model": str(first_target_row.get("judge_model", "")).strip(),
            "base_urls": sorted(
                {
                    str(row.get("judge_base_url", "")).strip()
                    for row in ordered_target_rows
                    if str(row.get("judge_base_url", "")).strip()
                }
            ),
        },
        "prediction_file": str(target_path),
        "common_samples": len(common_keys),
        "baseline_only_samples": len(baseline_rows) - len(common_keys),
        "target_only_samples": len(target_rows) - len(common_keys),
        "baseline_request_failures": baseline_request_failures,
        "baseline_parse_failures": baseline_parse_failures,
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "metrics": compute_metrics(baseline_scores, model_scores),
    }


def build_experiment_summary(
    baseline_path: Path,
    input_paths: list[Path],
    sample_size: Optional[int],
    judge_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    def metric_sort_key(item: dict[str, Any]) -> tuple[bool, float, float]:
        spearman = item["metrics"]["spearman"]
        pearson = item["metrics"]["pearson"]
        return (
            spearman is None,
            -(spearman if spearman is not None else float("-inf")),
            -(pearson if pearson is not None else float("-inf")),
        )

    ranking = sorted(judge_summaries, key=metric_sort_key)
    return {
        "baseline_path": str(baseline_path),
        "prediction_files": [str(path) for path in input_paths],
        "sample_size": sample_size,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_summaries": judge_summaries,
        "ranking_by_spearman": [
            {
                "judge_name": item["judge"]["name"],
                "judge_model": item["judge"]["model"],
                "spearman": item["metrics"]["spearman"],
                "pearson": item["metrics"]["pearson"],
                "kendall_tau": item["metrics"]["kendall_tau"],
                "mae": item["metrics"]["mae"],
                "rmse": item["metrics"]["rmse"],
                "score_match_rate": item["metrics"]["score_match_rate"],
                "valid_count": item["metrics"]["count"],
            }
            for item in ranking
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare judge-output JSONL files against a cortex-5 baseline file."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help=f"Baseline JSONL file. Its 'predicted_score' is treated as ground truth. Default: {DEFAULT_BASELINE_PATH}",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=DEFAULT_INPUT_PATHS,
        help="One or more JSONL prediction files to compare against the baseline.",
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
        help="Optional number of sorted overlapping samples to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline.resolve()
    input_paths = [path.resolve() for path in args.inputs]

    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    judge_summaries = [
        compare_prediction_file(
            baseline_path=baseline_path,
            target_path=path,
            sample_size=args.sample_size,
        )
        for path in input_paths
    ]
    experiment_summary = build_experiment_summary(
        baseline_path=baseline_path,
        input_paths=input_paths,
        sample_size=args.sample_size,
        judge_summaries=judge_summaries,
    )

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    summary_path.write_text(json.dumps(experiment_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = [
        "judge_name,judge_model,valid_count,pearson,spearman,kendall_tau,mae,rmse,score_match_rate,common_samples,baseline_only_samples,target_only_samples,request_failures,parse_failures,prediction_file"
    ]
    for item in judge_summaries:
        metrics = item["metrics"]
        csv_lines.append(
            ",".join(
                [
                    str(item["judge"]["name"]),
                    str(item["judge"]["model"]),
                    str(metrics["count"]),
                    str(metrics["pearson"]),
                    str(metrics["spearman"]),
                    str(metrics["kendall_tau"]),
                    str(metrics["mae"]),
                    str(metrics["rmse"]),
                    str(metrics["score_match_rate"]),
                    str(item["common_samples"]),
                    str(item["baseline_only_samples"]),
                    str(item["target_only_samples"]),
                    str(item["request_failures"]),
                    str(item["parse_failures"]),
                    str(item["prediction_file"]),
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(json.dumps(experiment_summary["ranking_by_spearman"], ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
