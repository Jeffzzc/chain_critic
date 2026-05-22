#!/usr/bin/env python
"""Compare final_test_model_outputs scores against GT from the tagged test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import re
from typing import Any, Optional


DEFAULT_GT_PATH = Path(
    "datasets/train/final_score_reason_plaintext_sft_final_tagged_system_user_balanced_v2/"
    "final_score_reason_plaintext_test.jsonl"
)
DEFAULT_PREDICTIONS_DIR = Path("evaluation/final_test_model_outputs")
DEFAULT_OUTPUT_DIR = Path("evaluation/final_test_model_outputs/summaries/gt_baseline")

SCORE_TAG_RE = re.compile(r"(?is)<\s*s\s*>\s*(.*?)\s*</\s*s\s*>")
SCORE_LINE_RE = re.compile(r"(?im)^\s*score\s*[:\uFF1A]\s*([0-5])\s*$")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def parse_int_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    if isinstance(value, float) and value.is_integer() and 0 <= value <= 5:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-5]", value.strip()):
        return int(value.strip())
    return None


def parse_score_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and 0 <= float(value) <= 5:
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[0-5](?:\.\d+)?", text):
            return float(text)
    return None


def parse_tagged_score(text: str) -> int | None:
    raw = str(text or "").strip().replace("\r\n", "\n")
    match = SCORE_TAG_RE.search(raw)
    if match:
        return parse_int_score(normalize_text(match.group(1)))

    score_line = SCORE_LINE_RE.search(raw)
    if score_line:
        return parse_int_score(score_line.group(1))

    return None


def stable_sample_id(record: dict[str, Any], index: int) -> str:
    explicit = normalize_text(record.get("sample_id") or record.get("id") or record.get("unique_id"))
    if explicit:
        return explicit
    raw = json.dumps(record.get("messages", record), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{index}||{raw}".encode("utf-8")).hexdigest()
    return f"row:{index}:{digest}"


def extract_reference_output(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    assistant_contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        if role == "assistant":
            assistant_contents.append(str(message.get("content") or ""))
    return assistant_contents[-1] if assistant_contents else ""


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


def compute_metrics(gt_scores: list[float], model_scores: list[float]) -> dict[str, Any]:
    if len(gt_scores) != len(model_scores):
        raise ValueError("GT and model score lengths do not match.")

    abs_errors = [abs(a - b) for a, b in zip(gt_scores, model_scores)]
    sq_errors = [(a - b) ** 2 for a, b in zip(gt_scores, model_scores)]
    exact_matches = sum(1 for a, b in zip(gt_scores, model_scores) if a == b)

    return {
        "count": len(gt_scores),
        "gt_mean": format_float(average(gt_scores)) if gt_scores else None,
        "model_mean": format_float(average(model_scores)) if model_scores else None,
        "mae": format_float(average(abs_errors)) if abs_errors else None,
        "rmse": format_float(math.sqrt(average(sq_errors))) if sq_errors else None,
        "score_match_rate": format_float(exact_matches / len(gt_scores)) if gt_scores else None,
        "pearson": format_float(pearson_correlation(gt_scores, model_scores)),
        "spearman": format_float(spearman_correlation(gt_scores, model_scores)),
        "kendall_tau": format_float(kendall_tau_b(gt_scores, model_scores)),
    }


def load_gt_scores(gt_path: Path) -> tuple[dict[str, float], dict[int, float], int]:
    gt_by_sample_id: dict[str, float] = {}
    gt_by_index: dict[int, float] = {}
    parse_failures = 0

    rows = read_jsonl(gt_path)
    for index, record in enumerate(rows):
        sample_id = stable_sample_id(record, index)
        reference_output = extract_reference_output(record)
        score = parse_tagged_score(reference_output)
        if score is None:
            parse_failures += 1
            continue
        gt_by_sample_id[sample_id] = float(score)
        gt_by_index[index] = float(score)
    return gt_by_sample_id, gt_by_index, parse_failures


def resolve_gt_score(
    row: dict[str, Any],
    gt_by_sample_id: dict[str, float],
    gt_by_index: dict[int, float],
) -> Optional[float]:
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        score = gt_by_sample_id.get(sample_id.strip())
        if score is not None:
            return score

    index = row.get("index")
    if isinstance(index, int):
        return gt_by_index.get(index)

    return None


def compare_prediction_file(
    gt_by_sample_id: dict[str, float],
    gt_by_index: dict[int, float],
    gt_parse_failures: int,
    target_path: Path,
    sample_size: Optional[int],
) -> dict[str, Any]:
    rows = read_jsonl(target_path)

    valid_pairs: list[tuple[float, float]] = []
    missing_gt = 0
    invalid_predicted = 0

    if sample_size is not None:
        rows = rows[: max(0, sample_size)]

    request_failures = sum(1 for row in rows if row.get("request_error"))
    parse_failures = sum(1 for row in rows if row.get("parse_error"))

    for row in rows:
        gt_score = resolve_gt_score(row, gt_by_sample_id, gt_by_index)
        if gt_score is None:
            missing_gt += 1
            continue
        predicted_score = parse_score_value(row.get("predicted_score"))
        if predicted_score is None:
            invalid_predicted += 1
            continue
        valid_pairs.append((gt_score, predicted_score))

    gt_scores = [pair[0] for pair in valid_pairs]
    model_scores = [pair[1] for pair in valid_pairs]

    first_row = rows[0] if rows else {}
    return {
        "gt": {
            "path": str(DEFAULT_GT_PATH),
        },
        "prediction_file": str(target_path),
        "run_name": str(first_row.get("run_name", "")).strip() or target_path.stem,
        "model": str(first_row.get("model", "")).strip(),
        "endpoint": str(first_row.get("endpoint", "")).strip(),
        "total_rows": len(rows),
        "gt_parse_failures": gt_parse_failures,
        "missing_gt": missing_gt,
        "invalid_predicted_score": invalid_predicted,
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "metrics": compute_metrics(gt_scores, model_scores),
    }


def build_experiment_summary(
    gt_path: Path,
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
        "gt_path": str(gt_path),
        "prediction_files": [str(path) for path in input_paths],
        "sample_size": sample_size,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summaries": judge_summaries,
        "ranking_by_spearman": [
            {
                "run_name": item["run_name"],
                "model": item["model"],
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
        description="Compare final_test_model_outputs scores against GT from the tagged test set."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_PATH,
        help=f"GT JSONL file. Default: {DEFAULT_GT_PATH}",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory with prediction JSONL files. Default: {DEFAULT_PREDICTIONS_DIR}",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="Glob pattern for prediction files inside predictions-dir.",
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
        help="Optional number of samples from each prediction file to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = args.gt.resolve()
    predictions_dir = args.predictions_dir.resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions dir not found: {predictions_dir}")

    input_paths = sorted(predictions_dir.glob(args.pattern))
    if not input_paths:
        raise FileNotFoundError(f"No prediction files matched {predictions_dir / args.pattern}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_by_sample_id, gt_by_index, gt_parse_failures = load_gt_scores(gt_path)

    judge_summaries = [
        compare_prediction_file(
            gt_by_sample_id=gt_by_sample_id,
            gt_by_index=gt_by_index,
            gt_parse_failures=gt_parse_failures,
            target_path=path,
            sample_size=args.sample_size,
        )
        for path in input_paths
    ]

    experiment_summary = build_experiment_summary(
        gt_path=gt_path,
        input_paths=input_paths,
        sample_size=args.sample_size,
        judge_summaries=judge_summaries,
    )

    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    summary_path.write_text(json.dumps(experiment_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_lines = [
        "run_name,model,valid_count,pearson,spearman,kendall_tau,mae,rmse,score_match_rate,total_rows,missing_gt,invalid_predicted_score,request_failures,parse_failures,prediction_file"
    ]
    for item in judge_summaries:
        metrics = item["metrics"]
        csv_lines.append(
            ",".join(
                [
                    str(item["run_name"]),
                    str(item["model"]),
                    str(metrics["count"]),
                    str(metrics["pearson"]),
                    str(metrics["spearman"]),
                    str(metrics["kendall_tau"]),
                    str(metrics["mae"]),
                    str(metrics["rmse"]),
                    str(metrics["score_match_rate"]),
                    str(item["total_rows"]),
                    str(item["missing_gt"]),
                    str(item["invalid_predicted_score"]),
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
