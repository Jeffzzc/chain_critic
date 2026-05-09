#!/usr/bin/env python
"""Convert datasets/FLASK/flask_eval.json into stage-3-ready rubric JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Optional


SECTION_RE = re.compile(
    r"###The instruction to evaluate:\s*(.*?)\s*"
    r"###Response to evaluate:\s*(.*?)\s*"
    r"###Reference Answer \(Score 5\):\s*(.*?)\s*"
    r"###Score Rubrics:\s*(.*)\s*"
    r"###Feedback:\s*$",
    re.S,
)
SCORE_RE = re.compile(r"Score\s*([1-5])\s*:\s*(.*?)(?=\n\s*Score\s*[1-5]\s*:|\Z)", re.S)
BRACKET_DESC_RE = re.compile(r"\[(.*?)\]", re.S)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def parse_score_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            score = int(float(item))
        except (TypeError, ValueError):
            continue
        if 1 <= score <= 5:
            result.append(score)
    return result


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
    counts = Counter(values)
    top_count = max(counts.values())
    top_values = sorted(score for score, count in counts.items() if count == top_count)
    if len(top_values) != 1:
        return None
    return top_values[0]


def slugify(text: str) -> str:
    lowered = normalize_text(text).lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "dimension"


def parse_instruction_blocks(instruction: str) -> dict[str, Any]:
    match = SECTION_RE.search(str(instruction or ""))
    if not match:
        raise ValueError("Failed to parse FLASK instruction blocks.")

    question = normalize_text(match.group(1))
    candidate_answer = normalize_text(match.group(2))
    reference_answer = normalize_text(match.group(3))
    rubric_block = match.group(4).strip()

    score_matches = SCORE_RE.findall(rubric_block)
    score_criteria: dict[str, str] = {str(score): "" for score in range(1, 6)}
    for score, text in score_matches:
        score_criteria[str(score)] = normalize_text(text)

    missing = [score for score in ("1", "2", "3", "4", "5") if not score_criteria[score]]
    if missing:
        raise ValueError(f"Rubric missing score criteria: {missing}")

    desc_match = BRACKET_DESC_RE.search(rubric_block)
    criteria_description = normalize_text(desc_match.group(1)) if desc_match else ""

    return {
        "question": question,
        "answer": candidate_answer,
        "reference_answer": reference_answer,
        "score_criteria_1_to_5": score_criteria,
        "criteria_description": criteria_description,
    }


def build_row(
    source_row: dict[str, Any],
    *,
    zero_criteria: str,
) -> dict[str, Any]:
    parsed = parse_instruction_blocks(source_row.get("instruction"))
    dimension_name = normalize_text(source_row.get("criteria"))
    if not dimension_name:
        raise ValueError("Missing criteria field in source row.")

    idx = normalize_text(source_row.get("idx"))
    response_source = normalize_text(source_row.get("response_source"))
    uid = f"flask:{idx}:{response_source}".strip(":")
    sample_id = f"{uid}:{slugify(dimension_name)}"

    score_criteria = {
        # "0": normalize_text(zero_criteria),
        "1": parsed["score_criteria_1_to_5"]["1"],
        "2": parsed["score_criteria_1_to_5"]["2"],
        "3": parsed["score_criteria_1_to_5"]["3"],
        "4": parsed["score_criteria_1_to_5"]["4"],
        "5": parsed["score_criteria_1_to_5"]["5"],
    }

    human_scores = parse_score_list(source_row.get("human_score"))
    gpt4_scores = parse_score_list(source_row.get("gpt4_score"))
    human_mean = mean_or_none(human_scores)
    gpt4_mean = mean_or_none(gpt4_scores)

    return {
        "sample_id": sample_id,
        "parent_sample_id": uid,
        "unique_id": uid,
        "idx": idx,
        "response_source": response_source,
        "question": parsed["question"],
        "answer": parsed["answer"],
        "reference_answer": parsed["reference_answer"],
        "reference_solution": parsed["reference_answer"],
        "dimension_name": dimension_name,
        "evaluation_dimension": dimension_name,
        "criteria_description": parsed["criteria_description"],
        "full_score_criteria": score_criteria["5"],
        "score_criteria": score_criteria,
        # "criteria_0": score_criteria["0"],
        "criteria_1": score_criteria["1"],
        "criteria_2": score_criteria["2"],
        "criteria_3": score_criteria["3"],
        "criteria_4": score_criteria["4"],
        "criteria_5": score_criteria["5"],
        "human_score": human_scores,
        "human_score_mean": human_mean,
        "human_score_rounded_mean": round_half_up(human_mean),
        "human_score_majority": majority_or_none(human_scores),
        "gpt4_score": gpt4_scores,
        "gpt4_score_mean": gpt4_mean,
        "gpt4_score_rounded_mean": round_half_up(gpt4_mean),
        "gpt4_score_majority": majority_or_none(gpt4_scores),
    }


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    rows = raw[: max(0, args.limit)] if args.limit is not None else raw
    converted: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index}: not a JSON object")
            continue
        try:
            converted.append(build_row(row, zero_criteria=args.zero_criteria))
        except Exception as exc:  # pragma: no cover
            errors.append(f"row {index}: {exc}")
            if not args.skip_parse_errors:
                break

    if errors and not args.skip_parse_errors:
        error_preview = "\n".join(errors[:5])
        raise ValueError(f"Conversion failed with {len(errors)} error(s):\n{error_preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[input] {input_path}")
    print(f"[output] {output_path}")
    print(f"[rows] source={len(rows)} converted={len(converted)} errors={len(errors)}")
    if errors:
        print("[errors] first 5:")
        for message in errors[:5]:
            print(f"  - {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FLASK flask_eval.json to a flat stage-3 rubric JSONL for score_reason_rewrite scripts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/FLASK/flask_eval.json"),
        help="Input FLASK JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/FLASK/flask_eval_rubric_1_5.jsonl"),
        help="Output JSONL compatible with score_reason_rewrite_* scripts.",
    )
    parser.add_argument(
        "--zero-criteria",
        type=str,
        default="The response completely fails this dimension or is irrelevant.",
        help="Default score-0 criterion to prepend, because FLASK provides 1-5 rubrics.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument(
        "--skip-parse-errors",
        action="store_true",
        help="Skip malformed rows instead of failing fast.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
