#!/usr/bin/env python
"""
Filter correct score items for natural_reasoning (high-precision short-answer match).

Core rules:
1) Reference answer extraction:
   - Prefer `reference_answer` in reference JSONL.
   - If empty/unparseable, fallback to last item in `responses` (responses[-1]).

2) For each score JSON file (array of rows):
   - Match rows by `question`.
   - Extract predicted final answer from row["answer"] via regex heuristics.
   - Keep only rows whose extracted predicted answer equals extracted reference answer
     (high-precision canonical match; optional numeric tolerance).

Outputs:
- Per-score filtered output JSON: **keeps the same top-level format as input** (a JSON array).
  Each element is the original row dict (no extra debug fields injected).
- One summary JSON under output-dir (contains stats per file + overall stats).

Notes:
- Better normalization for fractions (\\frac{a}{b}, a/b) and percents (e.g., 12.5%).
- When a candidate is long, refine to the most answer-like atom (choice/percent/fraction/last number).
- Inline math extraction prefers math *after* the last "answer/final/最终答案" hint near the tail.
- Duplicate question conflicts in reference JSONL are tracked (canonical mismatch).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional


# ---------------------------
# Regex
# ---------------------------
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
HASH_RE = re.compile(r"####\s*([^\n\r]+)")

FINAL_EN_RE = re.compile(
    r"(?:^|\n)\s*(?:final\s*answer|the\s*final\s*answer|the\s*answer|answer)\s*(?:is|:|=)?\s*(.+)",
    re.IGNORECASE,
)
FINAL_CN_RE = re.compile(
    r"(?:^|\n)\s*(?:\u6700\u7EC8\u7B54\u6848|\u7B54\u6848)\s*(?:\u662F|\u4E3A|:|\uFF1A|=)?\s*(.+)"
)

INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)|\$(.*?)\$", re.DOTALL)

NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")

# fractions, percents, answer hints
FRAC_LATEX_RE = re.compile(r"\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")
FRAC_SLASH_RE = re.compile(r"(?<!\d)([+-]?\d+)\s*/\s*([+-]?\d+)(?!\d)")
PERCENT_RE = re.compile(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*%")
ANSWER_HINT_RE = re.compile(r"(final\s*answer|the\s*final\s*answer|the\s*answer|answer|最终答案|答案)", re.IGNORECASE)


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Extraction:
    canonical: str | None
    raw: str | None
    method: str


@dataclass
class ReferenceAnswer:
    canonical: str
    raw: str
    source: str
    method: str


# ---------------------------
# Normalization helpers
# ---------------------------
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


def is_math_like(text: str) -> bool:
    if re.search(r"\\[a-zA-Z]+", text):
        return True
    if re.search(r"[=^_{}]", text):
        return True
    if re.search(r"\d+\s*[/+\-*]\s*\d+", text):
        return True
    return False


def strip_wrappers(text: str) -> str:
    cleaned = THINK_RE.sub("", str(text))
    cleaned = cleaned.replace("\u200b", "")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = cleaned.strip()
    cleaned = cleaned.strip("`")
    cleaned = cleaned.strip("\"'")
    cleaned = cleaned.strip("\u201c\u201d\u2018\u2019")

    cleaned = re.sub(r"(?:\\blacksquare|\\square)\s*$", "", cleaned).strip()

    if cleaned.startswith("\\(") and cleaned.endswith("\\)"):
        cleaned = cleaned[2:-2].strip()
    if cleaned.startswith("$") and cleaned.endswith("$") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()

    cleaned = cleaned.strip("*_")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" \t\r\n")
    cleaned = cleaned.strip("。．.!?;；")
    return cleaned.strip()


def normalize_fraction_exact(text: str) -> str | None:
    """If the entire cleaned text is exactly a fraction, return reduced 'a/b'."""
    t = strip_wrappers(text)

    m = FRAC_LATEX_RE.fullmatch(t)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{frac.numerator}/{frac.denominator}"
        except Exception:
            return None

    m = FRAC_SLASH_RE.fullmatch(t)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{frac.numerator}/{frac.denominator}"
        except Exception:
            return None

    return None


def pick_atom_from_long_text(text: str) -> str | None:
    """
    When candidate text is long, try to extract the most answer-like atom:
    choice -> percent -> latex fraction -> slash fraction -> last number.
    """
    t = strip_wrappers(text)
    if not t:
        return None

    # 1) Choice A-E (as a standalone token)
    m = re.search(r"\b([A-E])\b", t.upper())
    if m:
        return m.group(1)

    # 2) Percent: keep '%'
    pm = list(PERCENT_RE.finditer(t))
    if pm:
        num = normalize_number(pm[-1].group(1))
        if num is not None:
            return f"{num}%"

    # 3) LaTeX fraction anywhere
    fm = list(FRAC_LATEX_RE.finditer(t))
    if fm:
        a, b = fm[-1].group(1), fm[-1].group(2)
        frac = normalize_fraction_exact(f"\\frac{{{a}}}{{{b}}}")
        if frac:
            return frac

    # 4) Slash fraction anywhere
    sm = list(FRAC_SLASH_RE.finditer(t))
    if sm:
        a, b = sm[-1].group(1), sm[-1].group(2)
        frac = normalize_fraction_exact(f"{a}/{b}")
        if frac:
            return frac

    # 5) Last plain number
    nums = list(NUMBER_RE.finditer(t))
    if nums:
        return nums[-1].group(0)

    return None


def canonicalize_answer(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = strip_wrappers(text)
    if not cleaned:
        return None

    # 0) exact fraction first
    frac = normalize_fraction_exact(cleaned)
    if frac is not None:
        return frac

    # 1) exact percent
    pm = PERCENT_RE.fullmatch(cleaned)
    if pm:
        num = normalize_number(pm.group(1))
        if num is not None:
            return f"{num}%"

    # 2) money
    money = re.fullmatch(r"\$+\s*([+-]?\d[\d,]*(?:\.\d+)?)", cleaned)
    if money:
        number = normalize_number(money.group(1))
        if number is not None:
            return number

    # 3) number
    number = normalize_number(cleaned)
    if number is not None:
        return number

    # 4) choice
    choice = normalize_choice(cleaned)
    if choice is not None:
        return choice

    # 5) general normalize
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = cleaned.replace("\\left", "")
    cleaned = cleaned.replace("\\right", "")

    if is_math_like(cleaned):
        cleaned = re.sub(r"\s+", "", cleaned)
    else:
        cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

    cleaned = cleaned.strip(".,;: ")
    return cleaned or None


def candidate_from_tail_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    for raw_tail in reversed(lines[-8:]):
        tail = re.sub(
            r"^(?:final\s*answer|the\s*answer|answer|thus|therefore|hence)\s*(?:is|:|=)?\s*",
            "",
            raw_tail,
            flags=re.IGNORECASE,
        )
        tail = re.sub(
            r"^(?:\u6700\u7EC8\u7B54\u6848|\u7B54\u6848)\s*(?:\u662F|\u4E3A|:|\uFF1A|=)?\s*",
            "",
            tail,
        )
        tail = strip_wrappers(tail)
        if tail:
            return tail
    return None


# ---------------------------
# Extraction
# ---------------------------
def extract_answer(answer_text: str, long_refine_min_len: int = 80) -> Extraction:
    """
    Extract an answer candidate (raw + canonical) from a model output string.
    High-precision heuristics:
      boxed > hash > explicit final lines > inline math (after last answer hint) > tail line.
    """
    if not answer_text:
        return Extraction(canonical=None, raw=None, method="empty")

    text = THINK_RE.sub("", str(answer_text))

    # 1) boxed
    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        raw0 = strip_wrappers(boxed_matches[-1])
        refined = pick_atom_from_long_text(raw0) if len(raw0) > long_refine_min_len else None
        raw = refined or raw0
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="boxed")

    # 2) ####
    hash_matches = HASH_RE.findall(text)
    if hash_matches:
        raw0 = strip_wrappers(hash_matches[-1])
        refined = pick_atom_from_long_text(raw0) if len(raw0) > long_refine_min_len else None
        raw = refined or raw0
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="hash")

    # 3) explicit final lines (EN)
    final_en_matches = FINAL_EN_RE.findall(text)
    if final_en_matches:
        raw0 = strip_wrappers(final_en_matches[-1].splitlines()[0])
        refined = pick_atom_from_long_text(raw0) if len(raw0) > long_refine_min_len else None
        raw = refined or raw0
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="final_answer_en")

    # 4) explicit final lines (CN)
    final_cn_matches = FINAL_CN_RE.findall(text)
    if final_cn_matches:
        raw0 = strip_wrappers(final_cn_matches[-1].splitlines()[0])
        refined = pick_atom_from_long_text(raw0) if len(raw0) > long_refine_min_len else None
        raw = refined or raw0
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="final_answer_cn")

    # 5) inline math near tail, prefer after last answer hint
    tail_text = text[-2500:]
    hints = list(ANSWER_HINT_RE.finditer(tail_text))
    if hints:
        tail_text = tail_text[hints[-1].start() :]

    math_matches = INLINE_MATH_RE.findall(tail_text)
    if math_matches:
        for pair in reversed(math_matches):
            raw_candidate = strip_wrappers(pair[0] or pair[1])
            if not raw_candidate:
                continue
            refined = pick_atom_from_long_text(raw_candidate) if len(raw_candidate) > long_refine_min_len else None
            raw = refined or raw_candidate
            return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="inline_math_tail")

    # 6) tail line
    tail = candidate_from_tail_line(text)
    if tail is not None:
        refined = pick_atom_from_long_text(tail) if len(tail) > long_refine_min_len else None
        raw = refined or tail
        return Extraction(canonical=canonicalize_answer(raw), raw=raw, method="tail_line")

    return Extraction(canonical=None, raw=None, method="not_found")


# ---------------------------
# Reference loading
# ---------------------------
def get_last_response_text(row: dict[str, Any]) -> str | None:
    responses = row.get("responses")
    if not isinstance(responses, list) or not responses:
        return None

    last = responses[-1]
    if isinstance(last, str):
        return last
    if isinstance(last, dict):
        for key in ("response", "answer", "content", "text"):
            value = last.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def load_reference_answers(
    reference_jsonl: Path, long_refine_min_len: int = 80
) -> tuple[dict[str, ReferenceAnswer], dict[str, Any]]:
    answer_map: dict[str, ReferenceAnswer] = {}
    stats: dict[str, Any] = {
        "total_lines": 0,
        "used_reference_answer": 0,
        "used_responses_fallback": 0,
        "extract_failed": 0,
        "duplicate_questions": 0,
        "conflict_questions": 0,  # duplicate question but different canonical
        "conflict_examples": [],  # capped examples
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

            chosen: Optional[Extraction] = None
            source = "none"

            reference_answer = row.get("reference_answer")
            if isinstance(reference_answer, str) and reference_answer.strip():
                extracted = extract_answer(reference_answer, long_refine_min_len=long_refine_min_len)
                if extracted.canonical is not None:
                    chosen = extracted
                    source = "reference_answer"
                    stats["used_reference_answer"] += 1

            if chosen is None:
                fallback_text = get_last_response_text(row)
                if fallback_text:
                    extracted = extract_answer(fallback_text, long_refine_min_len=long_refine_min_len)
                    if extracted.canonical is not None:
                        chosen = extracted
                        source = "responses_last"
                        stats["used_responses_fallback"] += 1

            if chosen is None or chosen.canonical is None or chosen.raw is None:
                stats["extract_failed"] += 1
                continue

            if question in answer_map:
                stats["duplicate_questions"] += 1
                prev = answer_map[question]
                if prev.canonical != chosen.canonical:
                    stats["conflict_questions"] += 1
                    if len(stats["conflict_examples"]) < 20:
                        stats["conflict_examples"].append(
                            {
                                "question": question,
                                "prev_canonical": prev.canonical,
                                "new_canonical": chosen.canonical,
                                "prev_raw": prev.raw,
                                "new_raw": chosen.raw,
                            }
                        )
                    # 保持“先到先得”，不覆盖
                    continue

            answer_map[question] = ReferenceAnswer(
                canonical=chosen.canonical,
                raw=chosen.raw,
                source=source,
                method=chosen.method,
            )

    return answer_map, stats


# ---------------------------
# File iteration / output
# ---------------------------
def iter_score_files(score_dir: Path, pattern: str) -> list[Path]:
    return sorted(p for p in score_dir.glob(pattern) if p.is_file())


def build_output_name(score_file: Path) -> str:
    return f"{score_file.stem}_correct_items.json"


# ---------------------------
# Matching helpers
# ---------------------------
def try_parse_decimal_canonical(canon: str) -> Decimal | None:
    """
    Parse canonical string to Decimal if it is a plain number.
    Excludes fractions like 'a/b' and percents like 'x%'.
    """
    if canon is None:
        return None
    if "/" in canon or canon.endswith("%"):
        return None
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", canon):
        return None
    try:
        return Decimal(canon)
    except InvalidOperation:
        return None


def canon_equal(pred: str, gold: str, numeric_tol: float = 0.0) -> tuple[bool, str]:
    """
    Return (equal, reason).
    reason: "exact" | "numeric_tol" | "mismatch"
    """
    if pred == gold:
        return True, "exact"
    if numeric_tol and numeric_tol > 0:
        dp = try_parse_decimal_canonical(pred)
        dg = try_parse_decimal_canonical(gold)
        if dp is not None and dg is not None:
            diff = abs(dp - dg)
            try:
                tol = Decimal(str(numeric_tol))
            except InvalidOperation:
                tol = Decimal("0")
            if diff <= tol:
                return True, "numeric_tol"
    return False, "mismatch"


# ---------------------------
# Filtering per score file
# ---------------------------
def filter_one_file(
    score_file: Path,
    reference_answers: dict[str, ReferenceAnswer],
    *,
    long_refine_min_len: int = 80,
    numeric_tol: float = 0.0,
) -> dict[str, Any]:
    """
    Returns a dict with stats + correct_items (original rows).
    NOTE: The caller decides how to serialize output; to keep per-file output format
    identical to input, write only correct_items (a list of rows).
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
            "match_breakdown": {"exact": 0, "numeric_tol": 0},
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
            "match_breakdown": {"exact": 0, "numeric_tol": 0},
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
            "match_breakdown": {"exact": 0, "numeric_tol": 0},
        }

    total_items = 0
    matched_questions = 0
    extract_failures = 0
    correct_items: list[dict[str, Any]] = []
    match_breakdown = {"exact": 0, "numeric_tol": 0}

    for row in payload:
        if not isinstance(row, dict):
            continue
        total_items += 1

        question = row.get("question")
        if question not in reference_answers:
            continue
        matched_questions += 1

        pred = extract_answer(str(row.get("answer", "")), long_refine_min_len=long_refine_min_len)
        if pred.canonical is None or pred.raw is None:
            extract_failures += 1
            continue

        gold = reference_answers[question]
        ok, reason = canon_equal(pred.canonical, gold.canonical, numeric_tol=numeric_tol)
        if ok:
            match_breakdown[reason] = match_breakdown.get(reason, 0) + 1
            # 关键：保留输入 row 的原始结构（不注入任何额外字段）
            correct_items.append(row)

    return {
        "source_file": score_file.name,
        "status": "ok",
        "total_items": total_items,
        "matched_questions": matched_questions,
        "extract_failures": extract_failures,
        "correct_count": len(correct_items),
        "correct_items": correct_items,
        "match_breakdown": match_breakdown,
        "config": {
            "long_refine_min_len": long_refine_min_len,
            "numeric_tol": numeric_tol,
        },
    }


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "High-precision filter: keep score rows whose extracted final answer equals "
            "the extracted reference answer (short-answer focused)."
        )
    )
    parser.add_argument(
        "--reference-jsonl",
        default="datasets/natural_reasoning/natural_reasoning_random_sample.jsonl",
        help="Reference JSONL path. Expected fields: question, reference_answer, responses.",
    )
    parser.add_argument(
        "--score-dir",
        default="datasets/natural_reasoning/score",
        help="Directory containing score JSON files.",
    )
    parser.add_argument(
        "--score-pattern",
        default="*.json",
        help="Glob pattern for score files inside score-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/natural_reasoning/filter",
        help="Directory for per-score filtered output files.",
    )
    parser.add_argument(
        "--summary-file",
        default="filter_summary.json",
        help="Summary file name written under output-dir.",
    )
    parser.add_argument(
        "--long-refine-min-len",
        type=int,
        default=80,
        help="If extracted raw candidate is longer than this, try to refine to a short atom.",
    )
    parser.add_argument(
        "--numeric-tol",
        type=float,
        default=0.0,
        help="Optional numeric tolerance for plain-number canonical answers (0 disables).",
    )

    args = parser.parse_args()

    reference_jsonl = Path(args.reference_jsonl)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_answers, reference_stats = load_reference_answers(
        reference_jsonl, long_refine_min_len=args.long_refine_min_len
    )
    score_files = iter_score_files(score_dir, args.score_pattern)

    per_file_summary: list[dict[str, Any]] = []
    total_correct = 0
    total_match_breakdown = {"exact": 0, "numeric_tol": 0}

    for score_file in score_files:
        result = filter_one_file(
            score_file,
            reference_answers,
            long_refine_min_len=args.long_refine_min_len,
            numeric_tol=args.numeric_tol,
        )

        out_path = output_dir / build_output_name(score_file)

        # 关键：逐文件输出保持与输入一致（顶层是 JSON 数组；元素是原始 row dict）
        per_file_payload = result["correct_items"] if result.get("status") == "ok" else []
        out_path.write_text(
            json.dumps(per_file_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_correct += int(result.get("correct_count", 0))
        mb = result.get("match_breakdown", {})
        total_match_breakdown["exact"] += int(mb.get("exact", 0))
        total_match_breakdown["numeric_tol"] += int(mb.get("numeric_tol", 0))

        per_file_summary.append(
            {
                "source_file": result.get("source_file", score_file.name),
                "status": result.get("status", "unknown"),
                "total_items": result.get("total_items", 0),
                "matched_questions": result.get("matched_questions", 0),
                "extract_failures": result.get("extract_failures", 0),
                "correct_count": result.get("correct_count", 0),
                "match_breakdown": mb,
                "output_file": str(out_path),
            }
        )

    summary = {
        "reference_jsonl": str(reference_jsonl),
        "score_dir": str(score_dir),
        "score_pattern": args.score_pattern,
        "output_dir": str(output_dir),
        "config": {
            "long_refine_min_len": args.long_refine_min_len,
            "numeric_tol": args.numeric_tol,
        },
        "reference_stats": reference_stats,
        "reference_usable_questions": len(reference_answers),
        "processed_files": len(score_files),
        "total_correct_items": total_correct,
        "total_match_breakdown": total_match_breakdown,
        "files": per_file_summary,
    }
    summary_path = output_dir / args.summary_file
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed files: {len(score_files)}")
    print(f"Reference usable questions: {len(reference_answers)}")
    print(f"Total correct items: {total_correct}")
    print(f"Total match breakdown: {total_match_breakdown}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
