#!/usr/bin/env python
"""Evaluate score alignment against FLASK gpt4/human labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def parse_score(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-8:
        return None
    if 0 <= rounded <= 5:
        return rounded
    return None


def parse_score_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            score = int(float(item))
        except (TypeError, ValueError):
            continue
        if 1 <= score <= 5:
            parsed.append(score)
    return parsed


def mean_or_none(values: list[int]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def round_half_up(value: Optional[float]) -> Optional[int]:
    if value is None or not math.isfinite(value):
        return None
    return int(math.floor(value + 0.5))


def majority_or_none(values: list[int]) -> Optional[int]:
    if not values:
        return None
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    top_count = max(counts.values())
    top_values = sorted(score for score, count in counts.items() if count == top_count)
    if len(top_values) != 1:
        return None
    return top_values[0]


def average(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
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
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][1] == pairs[start][1]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for index in range(start, end):
            ranks[pairs[index][0]] = avg_rank
        start = end
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson_correlation(rankdata(x), rankdata(y))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_no}: {path}")
            rows.append(payload)
    return rows


def row_key(row: dict[str, Any]) -> str:
    sample_id = normalize_text(row.get("sample_id"))
    if sample_id:
        return sample_id
    idx = normalize_text(row.get("idx"))
    source = normalize_text(row.get("response_source"))
    dimension = normalize_text(row.get("dimension_name") or row.get("evaluation_dimension"))
    if idx and source and dimension:
        return f"flask:{idx}:{source}:{dimension}"
    raise ValueError(f"Row missing usable key: {row}")


def compute_metrics(predicted: list[int], target_mean: list[float], rounded_target: list[int]) -> dict[str, Any]:
    if not predicted:
        return {
            "count": 0,
            "predicted_mean": None,
            "target_mean": None,
            "mae_to_target_mean": None,
            "rmse_to_target_mean": None,
            "pearson": None,
            "spearman": None,
            "rounded_exact_match_rate": None,
            "within_1_rate": None,
        }

    predicted_float = [float(value) for value in predicted]
    abs_errors = [abs(float(p) - t) for p, t in zip(predicted, target_mean)]
    sq_errors = [(float(p) - t) ** 2 for p, t in zip(predicted, target_mean)]
    rounded_matches = [int(p == r) for p, r in zip(predicted, rounded_target)]
    within_1 = [int(abs(float(p) - t) <= 1.0) for p, t in zip(predicted, target_mean)]

    return {
        "count": len(predicted),
        "predicted_mean": format_float(average(predicted_float)),
        "target_mean": format_float(average(target_mean)),
        "mae_to_target_mean": format_float(average(abs_errors)),
        "rmse_to_target_mean": format_float(math.sqrt(average(sq_errors) or 0.0)),
        "pearson": format_float(pearson_correlation(predicted_float, target_mean)),
        "spearman": format_float(spearman_correlation(predicted_float, target_mean)),
        "rounded_exact_match_rate": format_float((sum(rounded_matches) / len(rounded_matches)) if rounded_matches else None),
        "within_1_rate": format_float((sum(within_1) / len(within_1)) if within_1 else None),
    }


def evaluate_target(
    *,
    name: str,
    prediction_rows: dict[str, dict[str, Any]],
    reference_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    joined_keys = sorted(set(prediction_rows) & set(reference_rows))
    missing_reference = len(prediction_rows) - len(joined_keys)

    total_rows = len(joined_keys)
    missing_predicted_score = 0
    missing_target_label = 0
    majority_available = 0
    majority_match = 0

    predicted: list[int] = []
    target_mean: list[float] = []
    rounded_target: list[int] = []

    per_dimension: dict[str, dict[str, Any]] = {}
    for key in joined_keys:
        pred_row = prediction_rows[key]
        ref_row = reference_rows[key]
        pred_score = parse_score(pred_row.get("predicted_score"))
        if pred_score is None:
            missing_predicted_score += 1
            continue

        labels = parse_score_list(ref_row.get(f"{name}_score"))
        target_mean_value = mean_or_none(labels)
        rounded_target_value = round_half_up(target_mean_value)
        if target_mean_value is None or rounded_target_value is None:
            missing_target_label += 1
            continue

        predicted.append(pred_score)
        target_mean.append(target_mean_value)
        rounded_target.append(rounded_target_value)

        majority = majority_or_none(labels)
        if majority is not None:
            majority_available += 1
            if pred_score == majority:
                majority_match += 1

        dimension = normalize_text(
            pred_row.get("dimension_name")
            or pred_row.get("evaluation_dimension")
            or ref_row.get("dimension_name")
            or ref_row.get("evaluation_dimension")
        ) or "unknown"
        bucket = per_dimension.setdefault(
            dimension,
            {"predicted": [], "target_mean": [], "rounded_target": [], "majority_total": 0, "majority_match": 0},
        )
        bucket["predicted"].append(pred_score)
        bucket["target_mean"].append(target_mean_value)
        bucket["rounded_target"].append(rounded_target_value)
        if majority is not None:
            bucket["majority_total"] += 1
            if pred_score == majority:
                bucket["majority_match"] += 1

    summary = {
        "target": name,
        "rows_with_reference": total_rows,
        "missing_reference_rows": missing_reference,
        "missing_predicted_score": missing_predicted_score,
        "missing_target_label": missing_target_label,
        "majority_available": majority_available,
        "majority_match_rate": format_float((majority_match / majority_available) if majority_available else None),
        "metrics": compute_metrics(predicted, target_mean, rounded_target),
    }

    per_dimension_rows: list[dict[str, Any]] = []
    for dimension, bucket in sorted(per_dimension.items()):
        metrics = compute_metrics(bucket["predicted"], bucket["target_mean"], bucket["rounded_target"])
        per_dimension_rows.append(
            {
                "target": name,
                "dimension": dimension,
                "count": metrics["count"],
                "mae_to_target_mean": metrics["mae_to_target_mean"],
                "rmse_to_target_mean": metrics["rmse_to_target_mean"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
                "rounded_exact_match_rate": metrics["rounded_exact_match_rate"],
                "within_1_rate": metrics["within_1_rate"],
                "majority_match_rate": format_float(
                    (bucket["majority_match"] / bucket["majority_total"]) if bucket["majority_total"] else None
                ),
            }
        )

    return summary, per_dimension_rows


def run(args: argparse.Namespace) -> None:
    predictions_path = Path(args.predictions)
    reference_path = Path(args.reference)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")

    prediction_rows_raw = load_jsonl(predictions_path)
    reference_rows_raw = load_jsonl(reference_path)
    if args.limit is not None:
        prediction_rows_raw = prediction_rows_raw[: max(0, args.limit)]

    prediction_rows = {row_key(row): row for row in prediction_rows_raw}
    reference_rows = {row_key(row): row for row in reference_rows_raw}

    targets = ["gpt4", "human"] if args.target == "both" else [args.target]
    summaries: list[dict[str, Any]] = []
    per_dimension_rows: list[dict[str, Any]] = []
    for target in targets:
        summary, per_dim = evaluate_target(
            name=target,
            prediction_rows=prediction_rows,
            reference_rows=reference_rows,
        )
        summaries.append(summary)
        per_dimension_rows.extend(per_dim)

    output_dir = Path(args.output_dir) if args.output_dir else predictions_path.parent / "flask_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "alignment_summary.json"
    summary_csv = output_dir / "alignment_summary.csv"
    per_dimension_csv = output_dir / "alignment_by_dimension.csv"

    payload = {
        "predictions_file": str(predictions_path),
        "reference_file": str(reference_path),
        "targets": targets,
        "summaries": summaries,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "rows_with_reference",
                "missing_reference_rows",
                "missing_predicted_score",
                "missing_target_label",
                "count",
                "predicted_mean",
                "target_mean",
                "mae_to_target_mean",
                "rmse_to_target_mean",
                "pearson",
                "spearman",
                "rounded_exact_match_rate",
                "within_1_rate",
                "majority_available",
                "majority_match_rate",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            metrics = summary["metrics"]
            writer.writerow(
                {
                    "target": summary["target"],
                    "rows_with_reference": summary["rows_with_reference"],
                    "missing_reference_rows": summary["missing_reference_rows"],
                    "missing_predicted_score": summary["missing_predicted_score"],
                    "missing_target_label": summary["missing_target_label"],
                    "count": metrics["count"],
                    "predicted_mean": metrics["predicted_mean"],
                    "target_mean": metrics["target_mean"],
                    "mae_to_target_mean": metrics["mae_to_target_mean"],
                    "rmse_to_target_mean": metrics["rmse_to_target_mean"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "rounded_exact_match_rate": metrics["rounded_exact_match_rate"],
                    "within_1_rate": metrics["within_1_rate"],
                    "majority_available": summary["majority_available"],
                    "majority_match_rate": summary["majority_match_rate"],
                }
            )

    with per_dimension_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "dimension",
                "count",
                "mae_to_target_mean",
                "rmse_to_target_mean",
                "pearson",
                "spearman",
                "rounded_exact_match_rate",
                "within_1_rate",
                "majority_match_rate",
            ],
        )
        writer.writeheader()
        for row in per_dimension_rows:
            writer.writerow(row)

    print(f"[predictions] {predictions_path}")
    print(f"[reference] {reference_path}")
    print(f"[targets] {targets}")
    print(f"[summary_json] {summary_json}")
    print(f"[summary_csv] {summary_csv}")
    print(f"[per_dimension_csv] {per_dimension_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate score alignment to FLASK gpt4/human labels.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Stage-3 prediction JSONL from score_reason_rewrite_local/openai_api.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("datasets/FLASK/flask_eval_rubric_1_5.jsonl"),
        help="Converted FLASK rubric JSONL generated by prepare_flask_eval_for_pipeline.py.",
    )
    parser.add_argument(
        "--target",
        choices=["gpt4", "human", "both"],
        default="both",
        help="Which FLASK label set to align with.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
