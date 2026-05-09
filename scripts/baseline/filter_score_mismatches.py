#!/usr/bin/env python
"""Filter judge-output rows where predicted_score differs from reference_score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path(
    "evaluation/baseline_new/predictions/openai_cortex-5_judge_outputs_with_reference.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter JSONL rows whose predicted_score is different from reference_score "
            "and save them into a new JSONL file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL file. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL file path. Defaults to "
            "<input_dir>/<input_stem>_score_mismatches.jsonl"
        ),
    )
    parser.add_argument(
        "--judge-model-contains",
        type=str,
        default=None,
        help=(
            "Optional case-insensitive substring filter for judge_model. "
            "Example: gpt"
        ),
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return input_path.with_name(f"{input_path.stem}_score_mismatches.jsonl")


def row_matches_judge_filter(
    row: dict[str, Any],
    judge_model_contains: str | None,
) -> bool:
    if not judge_model_contains:
        return True
    judge_model = str(row.get("judge_model", "")).strip().lower()
    return judge_model_contains.lower() in judge_model


def extract_numeric_score(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def filter_score_mismatches(
    input_path: Path,
    output_path: Path,
    judge_model_contains: str | None,
) -> dict[str, int]:
    total_rows = 0
    filtered_rows = 0
    valid_score_rows = 0
    mismatch_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} in {input_path} is not a JSON object.")

            total_rows += 1

            if not row_matches_judge_filter(payload, judge_model_contains):
                continue

            filtered_rows += 1
            predicted_score = extract_numeric_score(payload, "predicted_score")
            reference_score = extract_numeric_score(payload, "reference_score")
            if predicted_score is None or reference_score is None:
                continue

            valid_score_rows += 1
            if predicted_score == reference_score:
                continue

            mismatch_rows += 1
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return {
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "valid_score_rows": valid_score_rows,
        "mismatch_rows": mismatch_rows,
    }


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.input, args.output)
    stats = filter_score_mismatches(
        input_path=args.input,
        output_path=output_path,
        judge_model_contains=args.judge_model_contains,
    )

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    if args.judge_model_contains:
        print(f"Judge model filter: contains '{args.judge_model_contains}'")
    print(f"Total rows read: {stats['total_rows']}")
    print(f"Rows after judge_model filter: {stats['filtered_rows']}")
    print(f"Rows with valid numeric scores: {stats['valid_score_rows']}")
    print(f"Rows written (score mismatches): {stats['mismatch_rows']}")


if __name__ == "__main__":
    main()
