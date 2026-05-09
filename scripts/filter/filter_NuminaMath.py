#!/usr/bin/env python
"""Filter correct items from score files using answers extracted from a reference JSONL.

Default use case:
- Reference: datasets/NuminaMath-CoT/NuminaMath-CoT_random_sample.jsonl
- Score dir:  datasets/NuminaMath-CoT/score

For each score JSON file, this script:
1. Matches rows by question text.
2. Extracts final answer from each row's `answer` field via regex heuristics.
3. Compares extracted prediction with extracted reference answer.
4. Writes one output file containing only correct rows.

CHANGE vs original:
- Per-score output file now preserves the **same top-level format as input**:
  it writes a JSON array (list of original rows), NOT a dict with stats.
- Each output row is kept **unchanged** (no injected debug keys like _gold/_pred).
- Stats are kept in the summary JSON only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s*answer|the\s*answer\s*is|answer)\s*(?::|\uFF1A)\s*(.+)",
    re.IGNORECASE,
)
FINAL_ANSWER_CN_RE = re.compile(
    r"(?:\u6700\u7EC8\u7B54\u6848|\u7B54\u6848\u662F)\s*(?::|\uFF1A)?\s*(.+)"
)
INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)|\$(.*?)\$", re.DOTALL)
NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class Extraction:
    canonical: str | None
    raw: str | None
    method: str


def normalize_number(text: str) -> str | None:
    value = text.strip().replace(",", "")
    value = value.strip("$")
    value = value.rstrip(".")
    if not value:
        return None
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None
    try:
        dec = Decimal(value)
    except InvalidOperation:
        return None
    normalized = format(dec.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if normalized == "-0":
        normalized = "0"
    return normalized


def normalize_choice(text: str) -> str | None:
    cleaned = text.strip()
    cleaned = cleaned.strip("()[]{}")
    cleaned = cleaned.strip().upper()
    if re.fullmatch(r"[A-E]", cleaned):
        return cleaned
    return None


def strip_wrappers(text: str) -> str:
    cleaned = str(text).strip()
    cleaned = cleaned.replace("\u200b", "")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove common trailing proof symbols.
    cleaned = re.sub(r"(?:\\blacksquare|\\square)\s*$", "", cleaned).strip()

    # Remove math wrappers.
    if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
        cleaned = cleaned[2:-2].strip()
    if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    # Remove markdown emphasis markers on boundaries.
    cleaned = cleaned.strip("*_`")
    return cleaned.strip()


def canonicalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = strip_wrappers(text)
    if not cleaned:
        return None

    # Normalize currency value like "$50" -> "50".
    money = re.fullmatch(r"\$+\s*([+-]?\d[\d,]*(?:\.\d+)?)", cleaned)
    if money:
        number = normalize_number(money.group(1))
        if number is not None:
            return number

    number = normalize_number(cleaned)
    if number is not None:
        return number

    choice = normalize_choice(cleaned)
    if choice is not None:
        return choice

    cleaned = cleaned.strip(".,;: ")
    if not cleaned:
        return None
    return cleaned


def candidate_from_tail_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    # Prefer last non-empty line (already ensured)
    tail = lines[-1]
    tail = re.sub(
        r"^(?:final\s*answer|answer|\u6700\u7EC8\u7B54\u6848|\u7B54\u6848\u662F)\s*(?::|\uFF1A)?\s*",
        "",
        tail,
        flags=re.IGNORECASE,
    )
    tail = strip_wrappers(tail)
    if not tail:
        return None
    return tail


def extract_answer(answer_text: str) -> Extraction:
    if not answer_text:
        return Extraction(canonical=None, raw=None, method="empty")

    text = THINK_RE.sub("", str(answer_text))

    boxed_matches = BOXED_ANSWER_RE.findall(text)
    if boxed_matches:
        raw = strip_wrappers(boxed_matches[-1])
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="boxed")

    hash_matches = HASH_ANSWER_RE.findall(text)
    if hash_matches:
        raw = strip_wrappers(hash_matches[-1])
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="hash")

    final_matches = FINAL_ANSWER_RE.findall(text)
    if final_matches:
        raw = strip_wrappers(final_matches[-1].splitlines()[0])
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="final_answer_en")

    final_cn_matches = FINAL_ANSWER_CN_RE.findall(text)
    if final_cn_matches:
        raw = strip_wrappers(final_cn_matches[-1].splitlines()[0])
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="final_answer_cn")

    math_matches = INLINE_MATH_RE.findall(text[-2000:])
    if math_matches:
        # take last non-empty inline math token
        for pair in reversed(math_matches):
            raw_candidate = pair[0] or pair[1]
            raw_candidate = strip_wrappers(raw_candidate)
            if raw_candidate:
                return Extraction(
                    canonical=canonicalize_answer(raw_candidate),
                    raw=raw_candidate,
                    method="inline_math_tail",
                )

    tail = candidate_from_tail_line(text)
    if tail is not None:
        return Extraction(canonical=canonicalize_answer(tail), raw=tail, method="tail_line")

    return Extraction(canonical=None, raw=None, method="not_found")


def load_reference_answers(reference_jsonl: Path) -> tuple[dict[str, str], dict[str, Any]]:
    answer_map: dict[str, str] = {}
    stats = {
        "total_lines": 0,
        "extract_failed": 0,
        "duplicate_questions": 0,
    }
    with reference_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["total_lines"] += 1
            row = json.loads(line)
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid question at line {line_no}: {reference_jsonl}")

            extracted = extract_answer(str(row.get("answer", "")))
            if extracted.canonical is None:
                stats["extract_failed"] += 1
                continue

            if question in answer_map:
                stats["duplicate_questions"] += 1
                # keep first seen for stability
                continue

            answer_map[question] = extracted.canonical

    return answer_map, stats


def iter_score_files(score_dir: Path, pattern: str) -> list[Path]:
    return sorted([p for p in score_dir.glob(pattern) if p.is_file()])


def build_output_name(score_file: Path) -> str:
    return f"{score_file.stem}_correct_items.json"


def filter_one_file(score_file: Path, reference_answers: dict[str, str]) -> dict[str, Any]:
    """
    Returns dict(stats + correct_items).
    NOTE: The caller will serialize ONLY correct_items to preserve original per-file format.
    """
    raw = score_file.read_text(encoding="utf-8").strip()
    if not raw:
        return {
            "source_file": score_file.name,
            "status": "skipped_empty",
            "total_items": 0,
            "matched_questions": 0,
            "extract_failures": 0,
            "correct_count": 0,
            "correct_items": [],
        }

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "source_file": score_file.name,
            "status": f"skipped_invalid_json: {exc}",
            "total_items": 0,
            "matched_questions": 0,
            "extract_failures": 0,
            "correct_count": 0,
            "correct_items": [],
        }

    if not isinstance(payload, list):
        return {
            "source_file": score_file.name,
            "status": "skipped_non_array_json",
            "total_items": 0,
            "matched_questions": 0,
            "extract_failures": 0,
            "correct_count": 0,
            "correct_items": [],
        }

    total_items = 0
    matched_questions = 0
    extract_failures = 0
    correct_items: list[dict[str, Any]] = []

    for row in payload:
        if not isinstance(row, dict):
            continue
        total_items += 1

        question = row.get("question")
        if question not in reference_answers:
            continue
        matched_questions += 1

        pred = extract_answer(str(row.get("answer", "")))
        if pred.canonical is None:
            extract_failures += 1
            continue

        gold = reference_answers[question]
        if pred.canonical == gold:
            # 关键：保留原始 row，不注入任何调试字段
            correct_items.append(row)

    return {
        "source_file": score_file.name,
        "status": "ok",
        "total_items": total_items,
        "matched_questions": matched_questions,
        "extract_failures": extract_failures,
        "correct_count": len(correct_items),
        "correct_items": correct_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter score rows that have correct answers according to a reference JSONL."
    )
    parser.add_argument(
        "--reference-jsonl",
        default="datasets/NuminaMath-CoT/NuminaMath-CoT_random_sample.jsonl",
        help="Path to reference JSONL containing question and answer fields.",
    )
    parser.add_argument(
        "--score-dir",
        default="datasets/NuminaMath-CoT/score",
        help="Directory containing score JSON files to filter.",
    )
    parser.add_argument(
        "--score-pattern",
        default="*.json",
        help="Glob pattern for score files inside score-dir. Default: *.json",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/NuminaMath-CoT/filter",
        help="Directory for filtered outputs.",
    )
    parser.add_argument(
        "--summary-file",
        default="filter_summary.json",
        help="Summary file name written under output-dir.",
    )
    args = parser.parse_args()

    reference_jsonl = Path(args.reference_jsonl)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_answers, reference_stats = load_reference_answers(reference_jsonl)
    score_files = iter_score_files(score_dir, args.score_pattern)

    per_file_summary: list[dict[str, Any]] = []
    total_correct = 0

    for score_file in score_files:
        result = filter_one_file(score_file, reference_answers)
        out_path = output_dir / build_output_name(score_file)

        # 关键：逐文件输出保持与输入一致：顶层是 JSON 数组；元素是原始 row dict
        per_file_payload = result["correct_items"] if result.get("status") == "ok" else []
        out_path.write_text(
            json.dumps(per_file_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_correct += int(result.get("correct_count", 0))
        per_file_summary.append(
            {
                "source_file": result.get("source_file", score_file.name),
                "status": result.get("status", "unknown"),
                "total_items": result.get("total_items", 0),
                "matched_questions": result.get("matched_questions", 0),
                "extract_failures": result.get("extract_failures", 0),
                "correct_count": result.get("correct_count", 0),
                "output_file": str(out_path),
            }
        )

    summary = {
        "reference_jsonl": str(reference_jsonl),
        "score_dir": str(score_dir),
        "score_pattern": args.score_pattern,
        "output_dir": str(output_dir),
        "reference_stats": reference_stats,
        "reference_usable_questions": len(reference_answers),
        "processed_files": len(score_files),
        "total_correct_items": total_correct,
        "files": per_file_summary,
    }
    summary_path = output_dir / args.summary_file
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed files: {len(score_files)}")
    print(f"Reference usable questions: {len(reference_answers)}")
    print(f"Total correct items: {total_correct}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
