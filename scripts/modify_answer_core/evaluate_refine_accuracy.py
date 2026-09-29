#!/usr/bin/env python3
"""Evaluate answer accuracy before and after refinement.

The script matches prediction rows to GSM8K/NuminaMath examples by question
text, extracts a single final numeric answer from the original and refined
answers, then compares both against the dataset gold answer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_FILES = [
    REPO_ROOT / "datasets/GSM8K/train.jsonl",
    REPO_ROOT / "datasets/GSM8K/test.jsonl",
    REPO_ROOT / "datasets/NuminaMath-CoT/NuminaMath-CoT_test.jsonl",
]
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "evaluation/baseline_new/predictions"
DEFAULT_BASELINE_FILE = DEFAULT_PREDICTIONS_DIR / "Qwen2.5-7B-Instruct.jsonl"

QUESTION_FIELDS = (
    "question",
    "problem",
    "prompt",
    "input",
    "query",
    "instruction",
)
GOLD_FIELDS = (
    "answer",
    "final_answer",
    "target",
    "output",
    "solution",
    "reference_answer",
    "reference",
)
REFINED_ANSWER_FIELDS = (
    "predicted_modified_answer",
    "modified_answer",
    "final_answer",
    "refined_answer",
    "revised_answer",
    "corrected_answer",
)
BASELINE_ANSWER_FIELDS = (
    "answer",
    "candidate_answer",
    "original_answer",
    "model_answer",
    "response",
)


@dataclass(frozen=True)
class NumericAnswer:
    value: Fraction
    source: str


@dataclass(frozen=True)
class DatasetExample:
    question: str
    gold_text: str
    gold_answer: NumericAnswer | None
    source_path: Path
    line_number: int


@dataclass
class FileStats:
    path: Path
    total_rows: int = 0
    matched_rows: int = 0
    before_correct: int = 0
    after_correct: int = 0
    fixed: int = 0
    broken: int = 0
    unextractable: int = 0
    unextractable_gold: int = 0
    unextractable_before: int = 0
    unextractable_after: int = 0
    missing_baseline: int = 0
    unmatched_rows: int = 0
    invalid_json_rows: int = 0


class JsonlReadError(RuntimeError):
    """Raised when a required JSONL file cannot be read."""


def normalize_question(text: Any) -> str:
    """Normalize question text for exact matching across files."""
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def first_text_field(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text
    return ""


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JsonlReadError(
                        f"Invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(data, dict):
                    raise JsonlReadError(
                        f"Expected JSON object in {path} at line {line_number}"
                    )
                yield line_number, data
    except OSError as exc:
        raise JsonlReadError(f"Could not read {path}: {exc}") from exc


def strip_latex_wrappers(text: str) -> str:
    text = text.strip()
    wrappers = (
        (r"\left", ""),
        (r"\right", ""),
        ("$", ""),
        ("\\,", ""),
        ("\\!", ""),
    )
    for old, new in wrappers:
        text = text.replace(old, new)
    return text.strip()


def find_balanced_brace_content(text: str, command: str) -> list[str]:
    """Return contents of LaTeX commands such as \\boxed{...}."""
    results: list[str] = []
    start = 0
    token = command + "{"
    while True:
        index = text.find(token, start)
        if index < 0:
            return results
        cursor = index + len(token)
        depth = 1
        chars: list[str] = []
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth:
                    chars.append(char)
            else:
                chars.append(char)
            cursor += 1
        if depth == 0:
            results.append("".join(chars))
            start = cursor
        else:
            start = index + len(command)


def replace_latex_fracs(text: str) -> str:
    pattern = re.compile(
        r"\\(?:dfrac|tfrac|frac)\s*\{\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*\}"
        r"\s*\{\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*\}"
    )
    return pattern.sub(r"\1/\2", text)


def normalize_numeric_text(text: str) -> str:
    """Normalize common numeric spelling variants before regex extraction."""
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    number = r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    text = re.sub(
        number + r"\s*(?:percent|per\s+cent|pct)\b",
        r"\1%",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"百分之\s*" + number, r"\1%", text)
    return text


def answer_focus_segments(text: str) -> list[str]:
    """Prefer explicit final-answer spans before falling back to full text."""
    text = normalize_numeric_text(strip_latex_wrappers(replace_latex_fracs(text)))
    boxed = find_balanced_brace_content(text, r"\boxed")
    fboxed = find_balanced_brace_content(text, r"\fbox")
    segments: list[str] = [strip_latex_wrappers(x) for x in boxed + fboxed if x.strip()]

    if "####" in text:
        segments.append(text.rsplit("####", 1)[-1])

    marker_patterns = [
        r"(?:final\s+answer|answer|result|therefore|thus|so)\s*(?:is|=|:)?",
        r"(?:the\s+answer\s+is)\s*",
        r"(?:答案|最终答案)\s*(?:是|为|:|：)?",
    ]
    lowered = text.lower()
    for pattern in marker_patterns:
        matches = list(re.finditer(pattern, lowered, flags=re.IGNORECASE))
        for match in matches[-3:]:
            segments.append(text[match.end() :])

    segments.append(text)
    return segments


def decimal_to_fraction(token: str) -> Fraction | None:
    try:
        return Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def parse_numeric_token(raw_token: str) -> NumericAnswer | None:
    token = strip_latex_wrappers(raw_token)
    token = token.strip().rstrip(".,;:)]}")
    token = token.lstrip("([{")
    token = token.replace(",", "")
    token = token.replace("−", "-")
    token = token.replace("–", "-")
    token = token.replace("—", "-")
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1].strip()

    if not token:
        return None

    mixed_match = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", token)
    if mixed_match:
        whole = int(mixed_match.group(1))
        numerator = int(mixed_match.group(2))
        denominator = int(mixed_match.group(3))
        if denominator == 0:
            return None
        sign = -1 if whole < 0 else 1
        value = Fraction(whole) + sign * Fraction(numerator, denominator)
    elif re.fullmatch(r"[+-]?\d+(?:\.\d+)?\s*/\s*[+-]?\d+(?:\.\d+)?", token):
        numerator_text, denominator_text = re.split(r"\s*/\s*", token, maxsplit=1)
        numerator = decimal_to_fraction(numerator_text)
        denominator = decimal_to_fraction(denominator_text)
        if numerator is None or denominator in (None, Fraction(0)):
            return None
        value = numerator / denominator
    else:
        value = decimal_to_fraction(token)
        if value is None:
            return None

    if is_percent:
        value /= 100
    return NumericAnswer(value=value, source=raw_token)


NUMBER_RE = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    [+-]?
    (?:
        \d{1,3}(?:,\d{3})+|\d+
    )
    (?:
        \s+\d+\s*/\s*\d+ |
        (?:\.\d+)?\s*/\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)? |
        \.\d+ |
    )
    \s*%?
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)


def extract_numeric_answer(text: Any) -> NumericAnswer | None:
    """Extract the final single numeric answer from free-form answer text."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    for segment in answer_focus_segments(text):
        candidates: list[NumericAnswer] = []
        normalized_segment = normalize_numeric_text(replace_latex_fracs(segment))
        for match in NUMBER_RE.finditer(normalized_segment):
            parsed = parse_numeric_token(match.group(0))
            if parsed is not None:
                candidates.append(parsed)
        if candidates:
            return candidates[-1]
    return None


def answers_equal(
    predicted: NumericAnswer | None,
    gold: NumericAnswer | None,
    tolerance: Fraction,
) -> bool:
    if predicted is None or gold is None:
        return False
    return abs(predicted.value - gold.value) <= tolerance


def load_dataset_examples(dataset_files: list[Path]) -> tuple[dict[str, DatasetExample], int]:
    examples: dict[str, DatasetExample] = {}
    duplicate_count = 0
    for path in dataset_files:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        for line_number, row in iter_jsonl(path):
            question = first_text_field(row, QUESTION_FIELDS)
            gold_text = first_text_field(row, GOLD_FIELDS)
            question_key = normalize_question(question)
            if not question_key:
                continue
            example = DatasetExample(
                question=question_key,
                gold_text=gold_text,
                gold_answer=extract_numeric_answer(gold_text),
                source_path=path,
                line_number=line_number,
            )
            if question_key in examples:
                duplicate_count += 1
                continue
            examples[question_key] = example
    return examples, duplicate_count


def load_baseline_answers(path: Path) -> dict[str, NumericAnswer | None]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")
    answers: dict[str, NumericAnswer | None] = {}
    for _, row in iter_jsonl(path):
        question_key = normalize_question(first_text_field(row, QUESTION_FIELDS))
        if not question_key or question_key in answers:
            continue
        answer_text = first_text_field(row, BASELINE_ANSWER_FIELDS)
        answers[question_key] = extract_numeric_answer(answer_text)
    return answers


def prediction_files(predictions_dir: Path, recursive: bool) -> list[Path]:
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(path for path in predictions_dir.glob(pattern) if path.is_file())


def evaluate_prediction_file(
    path: Path,
    dataset_examples: dict[str, DatasetExample],
    baseline_answers: dict[str, NumericAnswer | None],
    tolerance: Fraction,
    strict_json: bool,
) -> FileStats:
    stats = FileStats(path=path)
    try:
        rows = iter_jsonl(path)
        for _, row in rows:
            stats.total_rows += 1
            question_key = normalize_question(first_text_field(row, QUESTION_FIELDS))
            example = dataset_examples.get(question_key)
            if example is None:
                stats.unmatched_rows += 1
                continue

            stats.matched_rows += 1
            refined_text = first_text_field(row, REFINED_ANSWER_FIELDS)
            refined_answer = extract_numeric_answer(refined_text)
            baseline_answer = baseline_answers.get(question_key)
            if question_key not in baseline_answers:
                stats.missing_baseline += 1

            if example.gold_answer is None:
                stats.unextractable_gold += 1
            if baseline_answer is None:
                stats.unextractable_before += 1
            if refined_answer is None:
                stats.unextractable_after += 1
            if (
                example.gold_answer is None
                or baseline_answer is None
                or refined_answer is None
            ):
                stats.unextractable += 1

            before_ok = answers_equal(baseline_answer, example.gold_answer, tolerance)
            after_ok = answers_equal(refined_answer, example.gold_answer, tolerance)
            if before_ok:
                stats.before_correct += 1
            if after_ok:
                stats.after_correct += 1
            if not before_ok and after_ok:
                stats.fixed += 1
            if before_ok and not after_ok:
                stats.broken += 1
    except JsonlReadError:
        if strict_json:
            raise
        stats.invalid_json_rows += 1
    return stats


def format_accuracy(correct: int, total: int) -> str:
    if total <= 0:
        return "0.0000"
    return f"{correct / total:.4f}"


def stats_to_row(stats: FileStats) -> dict[str, Any]:
    matched = stats.matched_rows
    before_acc = stats.before_correct / matched if matched else 0.0
    after_acc = stats.after_correct / matched if matched else 0.0
    return {
        "file": str(stats.path),
        "total_rows": stats.total_rows,
        "matched_samples": stats.matched_rows,
        "before_correct": stats.before_correct,
        "before_accuracy": before_acc,
        "after_correct": stats.after_correct,
        "after_accuracy": after_acc,
        "accuracy_delta": after_acc - before_acc,
        "fixed_wrong_to_right": stats.fixed,
        "broken_right_to_wrong": stats.broken,
        "unextractable": stats.unextractable,
        "unextractable_gold": stats.unextractable_gold,
        "unextractable_before": stats.unextractable_before,
        "unextractable_after": stats.unextractable_after,
        "missing_baseline": stats.missing_baseline,
        "unmatched_rows": stats.unmatched_rows,
        "invalid_json_rows": stats.invalid_json_rows,
    }


def print_text_report(stats_list: list[FileStats]) -> None:
    for stats in stats_list:
        matched = stats.matched_rows
        before_acc = format_accuracy(stats.before_correct, matched)
        after_acc = format_accuracy(stats.after_correct, matched)
        delta = (stats.after_correct / matched - stats.before_correct / matched) if matched else 0.0
        print(f"\nFile: {stats.path.name}")
        print(f"  matched samples: {matched}")
        print(f"  before correct / accuracy: {stats.before_correct} / {before_acc}")
        print(f"  after correct / accuracy: {stats.after_correct} / {after_acc}")
        print(f"  accuracy delta: {delta:+.4f}")
        print(f"  fixed wrong->right: {stats.fixed}")
        print(f"  broken right->wrong: {stats.broken}")
        print(f"  unextractable: {stats.unextractable}")
        if stats.unmatched_rows or stats.missing_baseline or stats.invalid_json_rows:
            print(
                "  diagnostics: "
                f"unmatched={stats.unmatched_rows}, "
                f"missing_baseline={stats.missing_baseline}, "
                f"invalid_json_rows={stats.invalid_json_rows}"
            )


def write_json_report(path: Path, stats_list: list[FileStats]) -> None:
    rows = [stats_to_row(stats) for stats in stats_list]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_tolerance(value: str) -> Fraction:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid tolerance: {value}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise argparse.ArgumentTypeError("Tolerance must be a non-negative number")
    return Fraction(Decimal(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final-answer accuracy before and after answer refinement "
            "for GSM8K/NuminaMath prediction JSONL files."
        )
    )
    parser.add_argument(
        "--dataset-files",
        nargs="+",
        type=Path,
        default=DEFAULT_DATASET_FILES,
        help="Dataset JSONL files containing questions and gold answers.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help="Directory containing refinement prediction JSONL files.",
    )
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=DEFAULT_BASELINE_FILE,
        help="Prediction file used to extract pre-refinement original answers.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively evaluate JSONL files below --predictions-dir.",
    )
    parser.add_argument(
        "--tolerance",
        type=parse_tolerance,
        default=Fraction(1, 1_000_000_000),
        help="Absolute numeric tolerance for answer comparison.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON summary.",
    )
    parser.add_argument(
        "--strict-json",
        action="store_true",
        help="Fail immediately on invalid JSON instead of reporting diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_examples, duplicate_count = load_dataset_examples(args.dataset_files)
        if not dataset_examples:
            raise RuntimeError("No dataset examples were loaded.")

        baseline_answers = load_baseline_answers(args.baseline_file)
        files = prediction_files(args.predictions_dir, args.recursive)
        if not files:
            raise RuntimeError(f"No JSONL prediction files found in {args.predictions_dir}")

        stats_list = [
            evaluate_prediction_file(
                path=path,
                dataset_examples=dataset_examples,
                baseline_answers=baseline_answers,
                tolerance=args.tolerance,
                strict_json=args.strict_json,
            )
            for path in files
        ]

        print(
            f"Loaded {len(dataset_examples)} dataset examples "
            f"({duplicate_count} duplicate questions skipped)."
        )
        print(f"Loaded {len(baseline_answers)} baseline answers from {args.baseline_file}.")
        print_text_report(stats_list)

        if args.json_output is not None:
            write_json_report(args.json_output, stats_list)
            print(f"\nWrote JSON report to {args.json_output}")
        return 0
    except (FileNotFoundError, JsonlReadError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
