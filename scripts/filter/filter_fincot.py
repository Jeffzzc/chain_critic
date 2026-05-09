#!/usr/bin/env python
"""
Filter correct FinCoT score items by matching extracted final answers (high-precision, short-answer focused).

Workflow:
1) Load gold answers from datasets/FinCoT/basic/sft.jsonl.
2) Extract a comparable final answer "atom" (boxed/hash/final line/inline-math/tail-line).
   If candidate is long, refine to an answer-like atom (bool/percent/fraction/last number).
3) Match predicted vs gold:
   - boolean exact
   - numeric/fraction/percent canonical (with abs/rel tolerance and %<->decimal conversion)
   - multi-number list full coverage match (gold nums must all be matched by distinct pred nums)
   - strict text match (exact or containment with min length)
4) Write one *_correct_items.json per score file plus one summary file.
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
    r"(?:^|\n)\s*(?:最终答案|答案)\s*(?:是|为|:|：|=)?\s*(.+)",
    re.IGNORECASE,
)

INLINE_MATH_RE = re.compile(r"\\\((.*?)\\\)|\$(.*?)\$", re.DOTALL)
ANSWER_HINT_RE = re.compile(r"(final\s*answer|the\s*final\s*answer|the\s*answer|answer|最终答案|答案)", re.IGNORECASE)

# bool tokens (extendable)
BOOL_RE = re.compile(r"\b(?:yes|no|true|false)\b", re.IGNORECASE)
CN_BOOL_RE = re.compile(r"(?:\b)?(?:是|否|对|错|正确|错误)(?:\b)?")

# numbers / percents / fractions
NUM_TOKEN_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?\s*%?")
NUMBER_RE = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
PERCENT_RE = re.compile(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*%")
FRAC_LATEX_RE = re.compile(r"\\(?:d?frac)\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")
FRAC_SLASH_RE = re.compile(r"(?<!\d)([+-]?\d+)\s*/\s*([+-]?\d+)(?!\d)")


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class ExtractedAnswer:
    raw: str | None
    canonical: str | None          # single canonical atom (bool/num/fraction/percent/text)
    kind: str | None               # "bool" | "num" | "percent" | "fraction" | "text"
    numbers: list[tuple[Decimal, bool]]  # list of (value, is_percent) parsed from raw
    bool_val: str | None           # "true"/"false" if detected
    method: str                    # extraction method tag


# ---------------------------
# Text cleanup / normalization
# ---------------------------
def strip_wrappers(text: str) -> str:
    cleaned = THINK_RE.sub("", str(text))
    cleaned = cleaned.replace("\u00a0", " ").replace("\u200b", "")
    cleaned = cleaned.strip()
    cleaned = cleaned.strip("`\"'")
    cleaned = cleaned.strip("*_")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = re.sub(r"^\s*(?:therefore|thus|hence)\s*,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" \t\r\n.,;:!?()[]{}")
    return cleaned


def normalize_text(text: str) -> str:
    cleaned = strip_wrappers(text)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------
# Canonical atom parsing
# ---------------------------
def parse_bool_atom(text: str) -> str | None:
    t = normalize_text(text)
    m = BOOL_RE.search(t)
    if m:
        token = m.group(0).lower()
        if token in ("yes", "true"):
            return "true"
        if token in ("no", "false"):
            return "false"
    # Chinese (optional, conservative)
    # Only accept single-character/word forms when text is short-ish
    t2 = strip_wrappers(text)
    if len(t2) <= 8:
        if t2 in ("是", "对", "正确"):
            return "true"
        if t2 in ("否", "错", "错误"):
            return "false"
    return None


def parse_decimal_number(token: str) -> Decimal | None:
    t = token.strip().replace(",", "")
    t = t.strip("$")
    if not t:
        return None
    # reject if not a pure number
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t):
        return None
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def parse_fraction_exact(text: str) -> str | None:
    """If the whole text is exactly a fraction, return reduced 'a/b'."""
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


def canonicalize_atom(text: str | None) -> tuple[str | None, str | None]:
    """
    Return (canonical, kind) where kind in:
      bool / percent / fraction / num / text
    """
    if text is None:
        return None, None

    raw = strip_wrappers(text)
    if not raw:
        return None, None

    b = parse_bool_atom(raw)
    if b is not None:
        return b, "bool"

    frac = parse_fraction_exact(raw)
    if frac is not None:
        return frac, "fraction"

    pm = PERCENT_RE.fullmatch(raw)
    if pm:
        dec = parse_decimal_number(pm.group(1))
        if dec is not None:
            # normalize percent number (strip trailing zeros)
            canon_num = dec.normalize()
            # Decimal.normalize() may use exponent; convert to plain string
            canon_str = format(canon_num, "f").rstrip("0").rstrip(".") if "." in format(canon_num, "f") else format(canon_num, "f")
            if canon_str == "-0":
                canon_str = "0"
            return f"{canon_str}%", "percent"

    num = parse_decimal_number(raw)
    if num is not None:
        canon_num = num.normalize()
        canon_str = format(canon_num, "f")
        if "." in canon_str:
            canon_str = canon_str.rstrip("0").rstrip(".")
        if canon_str == "-0":
            canon_str = "0"
        return canon_str, "num"

    # text
    return normalize_text(raw), "text"


def extract_numbers(raw: str) -> list[tuple[Decimal, bool]]:
    out: list[tuple[Decimal, bool]] = []
    for m in NUM_TOKEN_RE.finditer(raw):
        tok = m.group(0).strip()
        is_percent = tok.endswith("%")
        if is_percent:
            tok = tok[:-1].strip()
        dec = parse_decimal_number(tok)
        if dec is not None:
            out.append((dec, is_percent))
    return out


def pick_atom_from_long_text(text: str) -> str | None:
    """
    When candidate is long, refine to an answer-like atom:
    bool -> percent -> fraction (latex) -> fraction (a/b) -> last number -> None
    """
    t = strip_wrappers(text)
    if not t:
        return None

    # bool anywhere
    b = parse_bool_atom(t)
    if b is not None:
        return "true" if b == "true" else "false"

    # last percent
    pm = list(PERCENT_RE.finditer(t))
    if pm:
        num_str = pm[-1].group(1)
        dec = parse_decimal_number(num_str)
        if dec is not None:
            canon = format(dec.normalize(), "f")
            if "." in canon:
                canon = canon.rstrip("0").rstrip(".")
            if canon == "-0":
                canon = "0"
            return f"{canon}%"

    # last latex fraction
    fm = list(FRAC_LATEX_RE.finditer(t))
    if fm:
        a, b2 = fm[-1].group(1), fm[-1].group(2)
        frac = parse_fraction_exact(f"\\frac{{{a}}}{{{b2}}}")
        if frac:
            return frac

    # last slash fraction
    sm = list(FRAC_SLASH_RE.finditer(t))
    if sm:
        a, b2 = sm[-1].group(1), sm[-1].group(2)
        frac = parse_fraction_exact(f"{a}/{b2}")
        if frac:
            return frac

    # last number
    nums = list(NUMBER_RE.finditer(t))
    if nums:
        return nums[-1].group(0)

    return None


# ---------------------------
# Answer span extraction
# ---------------------------
def tail_line_candidate(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    for line in reversed(lines[-10:]):
        line = re.sub(
            r"^(?:final\s*answer|the\s*final\s*answer|the\s*answer|answer)\s*(?:is|:|=)?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"^(?:最终答案|答案)\s*(?:是|为|:|：|=)?\s*", "", line, flags=re.IGNORECASE)
        line = strip_wrappers(line)
        if line:
            return line
    return None


def extract_answer_span(answer_text: str) -> tuple[str | None, str]:
    if not answer_text:
        return None, "empty"

    text = THINK_RE.sub("", str(answer_text))

    boxed = BOXED_RE.findall(text)
    if boxed:
        return strip_wrappers(boxed[-1]), "boxed"

    hashm = HASH_RE.findall(text)
    if hashm:
        return strip_wrappers(hashm[-1]), "hash"

    final_en = FINAL_EN_RE.findall(text)
    if final_en:
        return strip_wrappers(final_en[-1].splitlines()[0]), "final_answer_en"

    final_cn = FINAL_CN_RE.findall(text)
    if final_cn:
        return strip_wrappers(final_cn[-1].splitlines()[0]), "final_answer_cn"

    # inline math near tail, prefer after last answer hint
    tail = text[-2500:]
    hints = list(ANSWER_HINT_RE.finditer(tail))
    if hints:
        tail = tail[hints[-1].start():]
    mathm = INLINE_MATH_RE.findall(tail)
    if mathm:
        for pair in reversed(mathm):
            cand = strip_wrappers(pair[0] or pair[1])
            if cand:
                return cand, "inline_math_tail"

    tail_line = tail_line_candidate(text)
    if tail_line:
        return tail_line, "tail_line"

    compact = " ".join(text.split())
    if compact:
        return strip_wrappers(compact[-240:]), "tail_text"
    return None, "not_found"


def extract_answer(answer_text: str, long_refine_min_len: int = 80) -> ExtractedAnswer:
    raw, method = extract_answer_span(answer_text)
    if raw is None or not raw.strip():
        return ExtractedAnswer(raw=None, canonical=None, kind=None, numbers=[], bool_val=None, method=method)

    # refine if too long
    raw0 = raw
    if len(raw0) > long_refine_min_len:
        refined = pick_atom_from_long_text(raw0)
        if refined:
            raw0 = refined

    canonical, kind = canonicalize_atom(raw0)
    bool_val = canonical if kind == "bool" else parse_bool_atom(raw0)
    nums = extract_numbers(raw)

    return ExtractedAnswer(
        raw=raw0,
        canonical=canonical,
        kind=kind,
        numbers=nums,
        bool_val=bool_val,
        method=method,
    )


# ---------------------------
# Matching
# ---------------------------
def is_close_decimal(a: Decimal, b: Decimal, abs_tol: Decimal, rel_tol: Decimal) -> bool:
    diff = abs(a - b)
    if diff <= abs_tol:
        return True
    scale = max(abs(a), abs(b), Decimal("1"))
    return (diff / scale) <= rel_tol


def numeric_token_match(
    gold: tuple[Decimal, bool],
    pred: tuple[Decimal, bool],
    abs_tol: Decimal,
    rel_tol: Decimal,
) -> bool:
    g_val, g_pct = gold
    p_val, p_pct = pred

    # same percent-flag: compare directly
    if g_pct == p_pct and is_close_decimal(g_val, p_val, abs_tol, rel_tol):
        return True

    # percent <-> decimal conversion
    if g_pct != p_pct:
        # gold is decimal, pred is percent
        if (not g_pct) and p_pct:
            if is_close_decimal(g_val * Decimal("100"), p_val, abs_tol, rel_tol):
                return True
        # gold is percent, pred is decimal
        if g_pct and (not p_pct):
            if is_close_decimal(g_val, p_val * Decimal("100"), abs_tol, rel_tol):
                return True
    return False


def numeric_list_match(
    gold_numbers: list[tuple[Decimal, bool]],
    pred_numbers: list[tuple[Decimal, bool]],
    abs_tol: Decimal,
    rel_tol: Decimal,
) -> bool:
    """
    High-precision: require every gold number to be matched by a distinct pred number.
    """
    if not gold_numbers or not pred_numbers:
        return False

    used: set[int] = set()
    for g in gold_numbers:
        hit = None
        for i, p in enumerate(pred_numbers):
            if i in used:
                continue
            if numeric_token_match(g, p, abs_tol, rel_tol):
                hit = i
                break
        if hit is None:
            return False
        used.add(hit)
    return True


def fraction_to_decimal(frac: str) -> Decimal | None:
    if not frac or "/" not in frac:
        return None
    try:
        a, b = frac.split("/", 1)
        da = Decimal(a.strip())
        db = Decimal(b.strip())
        if db == 0:
            return None
        return da / db
    except Exception:
        return None


def percent_str_to_decimal(percent_canon: str) -> Decimal | None:
    if not percent_canon or not percent_canon.endswith("%"):
        return None
    num = percent_canon[:-1].strip()
    d = parse_decimal_number(num)
    if d is None:
        return None
    return d / Decimal("100")


def canonical_numeric_equal(
    gold: ExtractedAnswer,
    pred: ExtractedAnswer,
    abs_tol: Decimal,
    rel_tol: Decimal,
) -> bool:
    """
    Compare when both have single canonical atoms in numeric-ish kinds.
    Supports num/percent/fraction mixed with conversion.
    """
    if not gold.canonical or not pred.canonical:
        return False
    if gold.kind not in ("num", "percent", "fraction") or pred.kind not in ("num", "percent", "fraction"):
        return False

    # convert both to decimal if possible
    def to_dec(e: ExtractedAnswer) -> Decimal | None:
        if e.kind == "num":
            return parse_decimal_number(e.canonical or "")
        if e.kind == "percent":
            return percent_str_to_decimal(e.canonical or "")
        if e.kind == "fraction":
            return fraction_to_decimal(e.canonical or "")
        return None

    dg = to_dec(gold)
    dp = to_dec(pred)
    if dg is None or dp is None:
        return False
    return is_close_decimal(dg, dp, abs_tol, rel_tol)


def text_match(gold_text: str | None, pred_text: str | None, min_contain_len: int = 8) -> bool:
    if not gold_text or not pred_text:
        return False
    if gold_text == pred_text:
        return True
    # containment only when reasonably long to reduce false positives
    if len(gold_text) >= min_contain_len and gold_text in pred_text:
        return True
    if len(pred_text) >= min_contain_len and pred_text in gold_text:
        return True
    return False


def answers_match(
    gold: ExtractedAnswer,
    pred: ExtractedAnswer,
    abs_tol: Decimal,
    rel_tol: Decimal,
    min_contain_len: int,
) -> bool:
    # 1) boolean exact
    if gold.bool_val and pred.bool_val and gold.bool_val == pred.bool_val:
        return True

    # 2) canonical numeric-ish (single atom) comparison
    if canonical_numeric_equal(gold, pred, abs_tol, rel_tol):
        return True

    # 3) multi-number list match (use original raw-number lists)
    if numeric_list_match(gold.numbers, pred.numbers, abs_tol, rel_tol):
        return True

    # 4) strict-ish text match
    if gold.kind == "text" and pred.kind == "text":
        return text_match(gold.canonical, pred.canonical, min_contain_len=min_contain_len)

    return False


# ---------------------------
# Gold loading
# ---------------------------
def load_gold_answers(
    sft_jsonl: Path,
    long_refine_min_len: int,
) -> tuple[dict[str, list[ExtractedAnswer]], dict[str, Any]]:
    gold_by_question: dict[str, list[ExtractedAnswer]] = {}
    stats: dict[str, Any] = {
        "total_rows": 0,
        "usable_rows": 0,
        "empty_extractions": 0,
        "unique_questions": 0,
    }

    with sft_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["total_rows"] += 1
            row = json.loads(line)
            question = row.get("question")
            answer = row.get("answer", "")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid question at line {line_no}: {sft_jsonl}")

            extracted = extract_answer(str(answer), long_refine_min_len=long_refine_min_len)
            if extracted.raw is None or extracted.canonical is None:
                stats["empty_extractions"] += 1
                continue

            gold_by_question.setdefault(question, []).append(extracted)
            stats["usable_rows"] += 1

    stats["unique_questions"] = len(gold_by_question)
    return gold_by_question, stats


# ---------------------------
# File iteration / output
# ---------------------------
def iter_score_files(score_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in score_dir.glob(pattern) if path.is_file())


def output_name(score_file: Path) -> str:
    return f"{score_file.stem}_correct_items.json"


def filter_score_file(
    score_file: Path,
    gold_by_question: dict[str, list[ExtractedAnswer]],
    *,
    long_refine_min_len: int,
    abs_tol: Decimal,
    rel_tol: Decimal,
    min_contain_len: int,
    with_debug: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(score_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Score file is not a JSON array: {score_file}")

    total = 0
    matched_question = 0
    pred_empty = 0
    correct_items: list[dict[str, Any]] = []

    for row in data:
        if not isinstance(row, dict):
            continue
        total += 1

        question = row.get("question")
        if question not in gold_by_question:
            continue
        matched_question += 1

        pred = extract_answer(str(row.get("answer", "")), long_refine_min_len=long_refine_min_len)
        if pred.raw is None or pred.canonical is None:
            pred_empty += 1
            continue

        gold_candidates = gold_by_question[question]
        hit_gold: Optional[ExtractedAnswer] = None
        for g in gold_candidates:
            if answers_match(g, pred, abs_tol=abs_tol, rel_tol=rel_tol, min_contain_len=min_contain_len):
                hit_gold = g
                break

        if hit_gold is not None:
            if with_debug:
                out_row = dict(row)
                out_row["_gold_raw"] = hit_gold.raw
                out_row["_gold_canonical"] = hit_gold.canonical
                out_row["_gold_kind"] = hit_gold.kind
                out_row["_gold_method"] = hit_gold.method
                out_row["_pred_raw"] = pred.raw
                out_row["_pred_canonical"] = pred.canonical
                out_row["_pred_kind"] = pred.kind
                out_row["_pred_method"] = pred.method
                correct_items.append(out_row)
            else:
                correct_items.append(row)

    stats = {
        "source_file": score_file.name,
        "total_items": total,
        "matched_questions": matched_question,
        "prediction_extract_failures": pred_empty,
        "correct_items": len(correct_items),
        "config": {
            "long_refine_min_len": long_refine_min_len,
            "abs_tol": str(abs_tol),
            "rel_tol": str(rel_tol),
            "min_contain_len": min_contain_len,
            "with_debug": with_debug,
        },
    }
    return correct_items, stats


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Filter correct FinCoT score items using extracted final answers.")
    parser.add_argument(
        "--sft-jsonl",
        default="datasets/FinCoT/basic/sft.jsonl",
        help="Reference SFT JSONL path containing fields: question, answer.",
    )
    parser.add_argument(
        "--score-dir",
        default="datasets/FinCoT/score",
        help="Directory containing score JSON files.",
    )
    parser.add_argument(
        "--score-pattern",
        default="*.json",
        help="Glob pattern for score files under score-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/FinCoT/filter",
        help="Directory to write per-file *_correct_items.json outputs.",
    )
    parser.add_argument(
        "--summary-file",
        default="filter_summary.json",
        help="Summary file name under output-dir.",
    )
    parser.add_argument(
        "--long-refine-min-len",
        type=int,
        default=80,
        help="If extracted candidate is longer than this, try to refine it into a short answer atom.",
    )
    parser.add_argument(
        "--abs-tol",
        type=str,
        default="1e-6",
        help="Absolute tolerance for numeric comparisons (Decimal string).",
    )
    parser.add_argument(
        "--rel-tol",
        type=str,
        default="1e-3",
        help="Relative tolerance for numeric comparisons (Decimal string).",
    )
    parser.add_argument(
        "--min-contain-len",
        type=int,
        default=10,
        help="Min length to allow containment-based text match (high precision => keep it >= 10).",
    )
    parser.add_argument(
        "--with-debug",
        action="store_true",
        help="If set, attach _gold/_pred extraction debug fields to each kept row.",
    )
    args = parser.parse_args()

    sft_jsonl = Path(args.sft_jsonl)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        abs_tol = Decimal(args.abs_tol)
        rel_tol = Decimal(args.rel_tol)
    except InvalidOperation:
        raise ValueError("Invalid --abs-tol or --rel-tol. Please provide valid Decimal strings, e.g. 1e-6, 0.001")

    gold_by_question, gold_stats = load_gold_answers(
        sft_jsonl,
        long_refine_min_len=args.long_refine_min_len,
    )
    score_files = iter_score_files(score_dir, args.score_pattern)

    all_stats: list[dict[str, Any]] = []
    total_correct = 0

    for score_file in score_files:
        correct_items, file_stats = filter_score_file(
            score_file,
            gold_by_question,
            long_refine_min_len=args.long_refine_min_len,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            min_contain_len=args.min_contain_len,
            with_debug=args.with_debug,
        )
        out_file = output_dir / output_name(score_file)
        out_file.write_text(json.dumps(correct_items, ensure_ascii=False, indent=2), encoding="utf-8")

        file_stats["output_file"] = str(out_file)
        all_stats.append(file_stats)
        total_correct += int(file_stats["correct_items"])

    summary = {
        "sft_jsonl": str(sft_jsonl),
        "score_dir": str(score_dir),
        "score_pattern": args.score_pattern,
        "output_dir": str(output_dir),
        "config": {
            "long_refine_min_len": args.long_refine_min_len,
            "abs_tol": str(abs_tol),
            "rel_tol": str(rel_tol),
            "min_contain_len": args.min_contain_len,
            "with_debug": args.with_debug,
        },
        "gold_stats": gold_stats,
        "processed_files": len(score_files),
        "total_correct_items": total_correct,
        "files": all_stats,
    }
    summary_path = output_dir / args.summary_file
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed files: {len(score_files)}")
    print(f"Gold questions: {gold_stats['unique_questions']}")
    print(f"Total correct items: {total_correct}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
