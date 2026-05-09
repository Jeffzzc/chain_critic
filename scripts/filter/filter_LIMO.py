#!/usr/bin/env python
"""Filter correct items from LIMO score files by matching train.jsonl answers.

Outputs one filtered file per source score file under datasets/LIMO/filter by default.

Per-file outputs:
- ONLY the original rows that are correct (a JSON array), with formatting inferred from source file.

Summary:
- STILL keeps statistics data (counts, statuses, etc.).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


# 1) Prefer boxed answers: \boxed{25}
BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)

# 2) Common natural-language endings near the final answer
FINAL_ANSWER_PATTERNS = [
    re.compile(
        r"(?is)(?:final\s+answer|answer)\s*(?:is|=|:)?\s*(?:\\boxed\s*\{)?\s*([+-]?\d+(?:\.\d+)?)"
    ),
    re.compile(
        r"(?is)(?:thus|therefore|hence)\s*(?:,?\s*the)?\s*answer(?:\s*is)?\s*(?:is|=|:)?\s*(?:\\boxed\s*\{)?\s*([+-]?\d+(?:\.\d+)?)"
    ),
]

# 3) Final fallback: pick the last integer in the tail text
INT_PATTERN = re.compile(r"[+-]?\d+")

# Process files that look like score outputs only
SCORE_FILE_PATTERN = re.compile(r"(?:_result_score|_score)\.json$")


def normalize_int_string(value: str | None) -> str | None:
    """Normalize numeric string to integer string, e.g. '025' -> '25', '25.0' -> '25'."""
    if value is None:
        return None

    text = value.strip().replace(",", "")
    text = text.strip("$")
    text = text.rstrip(".")

    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))

    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return str(int(float(text)))

    return None


def extract_numeric_from_segment(segment: str) -> str | None:
    """Extract a likely integer answer from a short segment."""
    segment = segment.strip()
    normalized = normalize_int_string(segment)
    if normalized is not None:
        return normalized

    nums = INT_PATTERN.findall(segment)
    if not nums:
        return None
    return normalize_int_string(nums[-1])


def extract_predicted_answer(answer_text: str) -> tuple[str | None, str]:
    """Extract predicted integer answer and extraction method."""
    if not answer_text:
        return None, "empty_answer"

    boxed_matches = BOXED_PATTERN.findall(answer_text)
    if boxed_matches:
        candidate = extract_numeric_from_segment(boxed_matches[-1])
        if candidate is not None:
            return candidate, "boxed"

    tail_text = answer_text[-1200:]
    for pattern in FINAL_ANSWER_PATTERNS:
        matches = pattern.findall(tail_text)
        if matches:
            candidate = extract_numeric_from_segment(matches[-1])
            if candidate is not None:
                return candidate, "final_phrase"

    nums = INT_PATTERN.findall(tail_text)
    if nums:
        candidate = normalize_int_string(nums[-1])
        if candidate is not None:
            return candidate, "last_integer_fallback"

    return None, "not_found"


def load_train_answers(train_path: Path) -> dict[str, str]:
    """Load {question -> normalized answer} from train.jsonl."""
    answer_map: dict[str, str] = {}
    with train_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = row.get("question")
            answer = normalize_int_string(str(row.get("answer", "")).strip())
            if not question or answer is None:
                raise ValueError(
                    f"Invalid train row at line {line_no}: missing question or non-integer answer."
                )
            answer_map[question] = answer
    return answer_map


def iter_score_files(score_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in score_dir.glob("*.json")
        if p.is_file() and SCORE_FILE_PATTERN.search(p.name)
    )


def _infer_dump_style(source_text: str) -> tuple[dict[str, Any], bool]:
    """
    Infer json.dumps kwargs to mimic source formatting:
    - pretty vs compact
    - indent width guess
    - ensure_ascii guess (if source uses \\uXXXX heavily and has no real non-ascii chars)
    Returns: (dump_kwargs, keep_trailing_newline)
    """
    keep_trailing_newline = source_text.endswith("\n")

    has_non_ascii = any(ord(c) > 127 for c in source_text)
    has_unicode_esc = bool(re.search(r"\\u[0-9a-fA-F]{4}", source_text))
    ensure_ascii = bool(has_unicode_esc and not has_non_ascii)

    # pretty printed?
    if "\n" in source_text:
        # guess indent by finding leading spaces before a quote after newline
        # e.g. "\n  \"key\"" => indent=2
        m = re.search(r"\n([ \t]+)\"", source_text)
        indent = 2
        if m:
            ws = m.group(1)
            if "\t" in ws:
                indent = 2  # json.dumps uses spaces for indent anyway
            else:
                indent = len(ws)
                if indent <= 0:
                    indent = 2
                # clamp to sane range
                indent = max(2, min(indent, 8))

        dump_kwargs: dict[str, Any] = {
            "ensure_ascii": ensure_ascii,
            "indent": indent,
        }
        return dump_kwargs, keep_trailing_newline

    # compact
    dump_kwargs = {
        "ensure_ascii": ensure_ascii,
        "separators": (",", ":"),
    }
    return dump_kwargs, keep_trailing_newline


def process_score_file(score_path: Path, answer_map: dict[str, str]) -> dict[str, Any]:
    """
    Returns a dict with:
      - file/status/total/matched_questions/correct/unmatched_questions/extraction_failures
      - correct_rows: ONLY the original rows that match (no extra wrapping fields)
      - _dump_kwargs/_keep_trailing_newline: internal style info for per-file output
    """
    raw_text = score_path.read_text(encoding="utf-8")
    dump_kwargs, keep_trailing_newline = _infer_dump_style(raw_text)
    raw_stripped = raw_text.strip()

    if not raw_stripped:
        return {
            "file": score_path.name,
            "status": "skipped_empty",
            "total": 0,
            "matched_questions": 0,
            "correct": 0,
            "unmatched_questions": 0,
            "extraction_failures": 0,
            "correct_rows": [],
            "_dump_kwargs": dump_kwargs,
            "_keep_trailing_newline": keep_trailing_newline,
        }

    try:
        rows = json.loads(raw_stripped)
    except json.JSONDecodeError as exc:
        return {
            "file": score_path.name,
            "status": f"skipped_invalid_json: {exc}",
            "total": 0,
            "matched_questions": 0,
            "correct": 0,
            "unmatched_questions": 0,
            "extraction_failures": 0,
            "correct_rows": [],
            "_dump_kwargs": dump_kwargs,
            "_keep_trailing_newline": keep_trailing_newline,
        }

    if not isinstance(rows, list):
        return {
            "file": score_path.name,
            "status": "skipped_non_array_json",
            "total": 0,
            "matched_questions": 0,
            "correct": 0,
            "unmatched_questions": 0,
            "extraction_failures": 0,
            "correct_rows": [],
            "_dump_kwargs": dump_kwargs,
            "_keep_trailing_newline": keep_trailing_newline,
        }

    correct_rows: list[dict[str, Any]] = []
    matched_questions = 0
    unmatched_questions = 0
    extraction_failures = 0

    for row in rows:
        if not isinstance(row, dict):
            continue

        question = row.get("question")
        if question not in answer_map:
            unmatched_questions += 1
            continue
        matched_questions += 1

        predicted, _method = extract_predicted_answer(str(row.get("answer", "")))
        if predicted is None:
            extraction_failures += 1
            continue

        gold = answer_map[question]
        if predicted == gold:
            # Keep ONLY the original row (no metadata)
            correct_rows.append(row)

    return {
        "file": score_path.name,
        "status": "ok",
        "total": len(rows),
        "matched_questions": matched_questions,
        "correct": len(correct_rows),
        "unmatched_questions": unmatched_questions,
        "extraction_failures": extraction_failures,
        "correct_rows": correct_rows,
        "_dump_kwargs": dump_kwargs,
        "_keep_trailing_newline": keep_trailing_newline,
    }


def per_file_output_name(score_file: Path) -> str:
    return f"{score_file.stem}_correct_items.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter items with correct answers from score JSON files by comparing "
            "extracted answers against train.jsonl, and write one output file per score file."
        )
    )
    parser.add_argument(
        "--train",
        default="datasets/LIMO/train.jsonl",
        help="Path to train.jsonl containing question and answer fields.",
    )
    parser.add_argument(
        "--score-dir",
        default="datasets/LIMO/score",
        help="Directory containing score JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/LIMO/filter",
        help="Output directory for per-score filtered files.",
    )
    parser.add_argument(
        "--summary-file",
        default="filter_summary.json",
        help="Summary filename under output-dir.",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)

    answer_map = load_train_answers(train_path)
    score_files = iter_score_files(score_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_summaries: list[dict[str, Any]] = []
    total_correct_items = 0

    for score_file in score_files:
        result = process_score_file(score_file, answer_map)

        # Per-file output: ONLY correct original rows, with formatting inferred from source
        per_file_path = output_dir / per_file_output_name(score_file)
        out_text = json.dumps(result["correct_rows"], **result["_dump_kwargs"])
        if result["_keep_trailing_newline"]:
            out_text += "\n"
        per_file_path.write_text(out_text, encoding="utf-8")

        summary = {
            "file": result["file"],
            "status": result["status"],
            "total": result["total"],
            "matched_questions": result["matched_questions"],
            "correct": result["correct"],
            "unmatched_questions": result["unmatched_questions"],
            "extraction_failures": result["extraction_failures"],
            "output_file": str(per_file_path),
        }
        file_summaries.append(summary)
        total_correct_items += int(result["correct"])

    summary_payload = {
        "train_path": str(train_path),
        "score_dir": str(score_dir),
        "output_dir": str(output_dir),
        "total_score_files": len(score_files),
        "total_correct_items": total_correct_items,
        "file_summaries": file_summaries,
    }
    summary_path = output_dir / args.summary_file
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved per-file filtered outputs to: {output_dir}")
    print(f"Saved summary to: {summary_path}")
    print(f"Total correct items across files: {total_correct_items}")
    for s in file_summaries:
        print(
            f"- {s['file']}: status={s['status']}, "
            f"correct={s['correct']}/{s['matched_questions']} "
            f"(extraction_failures={s['extraction_failures']})"
        )


if __name__ == "__main__":
    main()
