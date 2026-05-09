#!/usr/bin/env python
"""Filter correct items from score files by answers extracted from train.jsonl.

Workflow:
1. Load {question -> gold_answer} from train.jsonl (`answer` field).
2. For each JSON file in score-dir, extract predicted answer from each item's `answer`.
3. Keep items whose extracted predicted answer equals the gold answer of same question.
4. Write one filtered file per score file + one summary file (optional).
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


# e.g. "#### 72"
HASH_ANSWER_RE = re.compile(r"####\s*([+-]?\d[\d,]*(?:\.\d+)?)")
# e.g. "\boxed{589}"
BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
# generic numeric token fallback
NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")

# score files to process in a score directory
SCORE_FILE_RE = re.compile(r"_score\.json$")


def normalize_number(text: str | None) -> str | None:
    """Normalize numeric string to canonical form, e.g. '010.0' -> '10'."""
    if text is None:
        return None

    value = text.strip().strip("$").replace(",", "").rstrip(".")
    if not value:
        return None
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None

    try:
        dec = Decimal(value)
    except InvalidOperation:
        return None

    norm = format(dec.normalize(), "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".")
    if norm == "-0":
        norm = "0"
    return norm


def extract_numeric_from_text(text: str) -> str | None:
    text = text.strip()
    direct = normalize_number(text)
    if direct is not None:
        return direct
    nums = NUMBER_RE.findall(text)
    if not nums:
        return None
    return normalize_number(nums[-1])


def extract_answer(answer_text: str) -> str | None:
    """Extract final numeric answer from chain-of-thought answer text."""
    if not answer_text:
        return None

    # 1) GSM8K style final marker
    hash_matches = HASH_ANSWER_RE.findall(answer_text)
    if hash_matches:
        candidate = extract_numeric_from_text(hash_matches[-1])
        if candidate is not None:
            return candidate

    # 2) boxed answer
    boxed_matches = BOXED_ANSWER_RE.findall(answer_text)
    if boxed_matches:
        candidate = extract_numeric_from_text(boxed_matches[-1])
        if candidate is not None:
            return candidate

    # 3) whole answer already plain numeric
    direct = normalize_number(answer_text)
    if direct is not None:
        return direct

    # 4) fallback: use the last numeric token in tail segment
    tail = answer_text[-1500:]
    nums = NUMBER_RE.findall(tail)
    if nums:
        return normalize_number(nums[-1])

    return None


def load_train_answer_map(train_jsonl: Path) -> dict[str, str]:
    """Load question -> normalized gold answer."""
    answer_map: dict[str, str] = {}
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            question = row.get("question")
            answer_text = str(row.get("answer", ""))
            answer = extract_answer(answer_text)
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid question at line {line_no} in {train_jsonl}")
            if answer is None:
                raise ValueError(
                    f"Cannot extract numeric answer at line {line_no} in {train_jsonl}"
                )
            answer_map[question] = answer
    return answer_map


def iter_score_files(score_dir: Path) -> list[Path]:
    return sorted(
        p for p in score_dir.glob("*.json") if p.is_file() and SCORE_FILE_RE.search(p.name)
    )


def filter_one_score_file(
    score_file: Path, answer_map: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Returns:
      - correct_items: list of ORIGINAL rows (no extra fields)
      - stats: file-level stats only
    """
    raw = score_file.read_text(encoding="utf-8").strip()
    if not raw:
        stats = {
            "source_file": score_file.name,
            "status": "skipped_empty",
            "total_items": 0,
            "matched_questions": 0,
            "extract_failures": 0,
            "correct_count": 0,
        }
        return [], stats

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        stats = {
            "source_file": score_file.name,
            "status": f"skipped_invalid_json: {exc}",
            "total_items": 0,
            "matched_questions": 0,
            "extract_failures": 0,
            "correct_count": 0,
        }
        return [], stats

    if not isinstance(payload, list):
        stats = {
            "source_file": score_file.name,
            "status": "skipped_non_array_json",
            "total_items": 0,
            "matched_questions": 0,
            "extract_failures": 0,
            "correct_count": 0,
        }
        return [], stats

    total = 0
    matched = 0
    extract_fail = 0
    correct_items: list[dict[str, Any]] = []

    for row in payload:
        if not isinstance(row, dict):
            continue
        total += 1

        question = row.get("question")
        if question not in answer_map:
            continue
        matched += 1

        predicted = extract_answer(str(row.get("answer", "")))
        if predicted is None:
            extract_fail += 1
            continue

        if predicted == answer_map[question]:
            # Keep ONLY original row
            correct_items.append(row)

    stats = {
        "source_file": score_file.name,
        "status": "ok",
        "total_items": total,
        "matched_questions": matched,
        "extract_failures": extract_fail,
        "correct_count": len(correct_items),
    }
    return correct_items, stats


def build_output_name(score_file: Path) -> str:
    return f"{score_file.stem}_correct_items.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter correct score items by comparing extracted answers with train.jsonl answers."
    )
    parser.add_argument(
        "--train",
        default="datasets/GSM8K/train.jsonl",
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--score-dir",
        default="datasets/GSM8K/score",
        help="Directory containing score JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/GSM8K/filter",
        help="Directory to save filtered JSON files",
    )
    parser.add_argument(
        "--summary-file",
        default="filter_summary.json",
        help="Summary file name under output-dir",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the summary file",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_map = load_train_answer_map(train_path)
    score_files = iter_score_files(score_dir)

    file_summaries: list[dict[str, Any]] = []
    total_correct = 0

    for score_file in score_files:
        correct_items, stats = filter_one_score_file(score_file, answer_map)

        out_file = output_dir / build_output_name(score_file)
        # IMPORTANT: output only filtered original rows (a JSON array)
        out_file.write_text(
            json.dumps(correct_items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        file_summaries.append(
            {
                **stats,
                "output_file": str(out_file),
            }
        )
        total_correct += int(stats["correct_count"])

    if not args.no_summary:
        summary = {
            "train_path": str(train_path),
            "score_dir": str(score_dir),
            "output_dir": str(output_dir),
            "score_file_count": len(score_files),
            "total_correct_items": total_correct,
            "file_summaries": file_summaries,
        }
        summary_path = output_dir / args.summary_file
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Processed {len(score_files)} score files.")
        print(f"Total correct items: {total_correct}")
        print(f"Summary: {summary_path}")
    else:
        print(f"Processed {len(score_files)} score files.")
        print(f"Total correct items: {total_correct}")
        print("Summary: (disabled)")


if __name__ == "__main__":
    main()
