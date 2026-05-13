#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate 0-4 score answers from 0-5 score criteria via multi-endpoint API.

Main features:
1. Two-stage generation:
   - Stage A: generate low-score answer WITHOUT reference answer.
   - Stage B: rewrite the low-score answer into score-5 answer WITH reference answer.
2. Enforces Score == target_score.
3. Uses only the provided 0-5 rubric to control score boundaries.
4. Adds strict prompt constraints + programmatic post-generation screening.
5. Supports keep / review / drop decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Attempt to import local module for API calls
try:
    import generate_score_variants_api as api_base
except ImportError:
    print("Error: generate_score_variants_api module not found.")
    raise

# ============================================================================
# Constants and Defaults
# ============================================================================

DEFAULT_INPUT = Path("datasets/0-5/score_criteria_0_5.jsonl")
DEFAULT_OUTPUT = Path("datasets/0-4/generated_answers_0_4.jsonl")
DEFAULT_PORTS = tuple(range(8007, 8008))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_MODEL = "Qwen3.5-27B"
DEFAULT_DISPATCH_POLL_INTERVAL = 0.05
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_RETRIES = 3
TARGET_SCORES = tuple(range(5))  # 0, 1, 2, 3, 4

# Repair attempts after validation failure
DEFAULT_REPAIR_ATTEMPTS = 1

# Quality screening thresholds
MIN_REASON_WORD_COUNT = 12
HIGH_SIMILARITY_THRESHOLD = 0.92
VERY_HIGH_SIMILARITY_THRESHOLD = 0.95
EXTREME_SIMILARITY_THRESHOLD = 0.97
NEAR_IDENTICAL_THRESHOLD = 0.985
GIBBERISH_COMMON_WORD_RATIO = 0.08
GIBBERISH_VOWEL_ALPHA_RATIO = 0.18
GIBBERISH_WEIRD_TOKEN_COUNT = 3

# ============================================================================
# Prompt Templates
# ============================================================================

LOW_SCORE_SYSTEM_PROMPT = (
    "You are a data generation assistant.\n\n"
    "You will receive:\n"
    "- a question,\n"
    "- one evaluation dimension,\n"
    "- the complete 0-5 score criteria for that dimension,\n"
    "- and a target score from 0 to 4.\n\n"
    "Your task:\n"
    "1. Generate one plausible student answer that should receive exactly the target score on the specified dimension.\n"
    "2. Generate one strict, rubric-grounded Reason explaining why the generated answer matches the target score.\n"
    "3. Generate executable revision_suggestions for improving the generated answer to score 5.\n\n"
    "Hard constraints:\n"
    "- Return strict JSON only.\n"
    "- The returned Score must be exactly equal to the Target Score.\n"
    "- Use only the provided 0-5 score criteria to control the score level.\n"
    "- Make the generated_answer fit the target score more closely than the adjacent score levels.\n"
    "- Degrade primarily the specified dimension and keep other dimensions as intact as reasonably possible.\n"
    "- Even low-score answers must look like realistic student responses; do not output gibberish or random text.\n"
    "- Do not invent unrelated story facts unless that naturally follows from the target rubric level.\n"
    "- Do not assume or copy any unseen perfect answer.\n"
    "- The Reason must be specific and strict: explicitly identify the flaw, missing element, inconsistency, or weakness "
    "that places the generated_answer at the target score, and briefly indicate why it does not fit the nearest adjacent score(s).\n"
    "- The revision_suggestions must be concrete edit instructions that would fix the identified flaw and move the answer to score 5.\n"
    "- Do not use vague comments such as 'could be improved', 'somewhat unclear', or 'a little weak'.\n"
    "- Do not include dirty formatting inside any field: no markdown headings, no bullet lists, no numbered lists, "
    "no code fences, no bold markers, no '####', and no field labels such as 'Score:' or 'Reason:' inside text fields.\n"
    "- Keep the answer realistic and consistent with the question format.\n\n"
    "Self-check before you answer:\n"
    "1. Score is exactly the Target Score.\n"
    "2. generated_answer matches the target criterion better than the adjacent criteria.\n"
    "3. Reason explicitly justifies the target score using the rubric language and the actual flaw in the generated_answer.\n"
    "4. revision_suggestions states executable edits, not generic encouragement.\n"
    "5. All text fields are plain text with no dirty formatting.\n"
)

REWRITE_SYSTEM_PROMPT = (
    "You are a rewriting assistant.\n\n"
    "You will receive:\n"
    "- a question,\n"
    "- a low-score generated answer,\n"
    "- a high-quality reference answer,\n"
    "- one evaluation dimension,\n"
    "- and the complete 0-5 score criteria for that dimension.\n\n"
    "Your task:\n"
    "Rewrite the generated answer into a clean, complete, high-quality modified_answer that fully satisfies score 5 "
    "on the specified dimension.\n\n"
    "Hard constraints:\n"
    "- Return strict JSON only.\n"
    "- Use the reference answer only at this stage, as a correctness and quality anchor for the rewrite.\n"
    "- The modified_answer must satisfy the score-5 criterion on the specified dimension.\n"
    "- Improve only as needed to reach score 5 on that dimension; keep the answer realistic and aligned with the question.\n"
    "- When possible, write a clean independent answer instead of copying the reference answer verbatim.\n"
    "- Do not copy dirty formatting from the generated_answer.\n"
    "- Do not include dirty formatting inside modified_answer: no markdown headings, no bullet lists, no numbered lists, "
    "no code fences, no bold markers, no '####', and no field labels.\n"
    "- Write modified_answer as plain text only.\n\n"
    "Self-check before you answer:\n"
    "1. modified_answer fully satisfies the score-5 criterion for the specified dimension.\n"
    "2. modified_answer is clean, realistic, and consistent with the question.\n"
    "3. modified_answer contains no dirty formatting.\n"
)

GEN_REPAIR_SYSTEM_PROMPT = (
    "You are a JSON repair assistant.\n\n"
    "You will receive an invalid or incomplete JSON response for a low-score answer generation task.\n"
    "Rewrite it into valid strict JSON only.\n\n"
    "Preserve the original meaning, score level, and failure pattern as much as possible.\n"
    "Do NOT improve answer quality.\n"
    "Do NOT change the intended target score.\n"
    "Only repair formatting or fill clearly missing fields conservatively.\n\n"
    "Required keys:\n"
    "- Score\n"
    "- Reason\n"
    "- revision_suggestions\n"
    "- generated_answer\n"
)

REWRITE_REPAIR_SYSTEM_PROMPT = (
    "You are a JSON repair assistant.\n\n"
    "You will receive an invalid or incomplete JSON response for a rewrite task.\n"
    "Rewrite it into valid strict JSON only.\n\n"
    "Preserve the original meaning as much as possible.\n"
    "Do NOT change the substantive content unless required to restore missing structure.\n"
    "Only repair formatting or fill clearly missing fields conservatively.\n\n"
    "Required keys:\n"
    "- modified_answer\n"
)

LOW_SCORE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Dimension:\n{dimension_name}\n\n"
    "0-5 Score Criteria:\n{score_criteria}\n\n"
    "Target Score:\n{target_score}\n\n"
    "Rubric focus for exact score control:\n{score_band_guidance}\n\n"
    "Dimension-specific guidance:\n{dimension_guidance}\n\n"
    "Return JSON only with keys:\n"
    "{{\"Score\": <int>, \"Reason\": \"...\", \"revision_suggestions\": \"...\", \"generated_answer\": \"...\"}}"
)

REWRITE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Generated Answer:\n{generated_answer}\n\n"
    "Original Reason:\n{reason}\n\n"
    "Reference Answer:\n{reference_answer}\n\n"
    "Dimension:\n{dimension_name}\n\n"
    "0-5 Score Criteria:\n{score_criteria}\n\n"
    "Score-5 Criterion:\n{score_5_guidance}\n\n"
    "Dimension-specific guidance:\n{dimension_guidance}\n\n"
    "Return JSON only with keys:\n"
    "{{\"modified_answer\": \"...\"}}"
)

# Dimension-specific guidance for controlling degradation
DIMENSION_GUIDANCE: Dict[str, str] = {
    "data accuracy": (
        "- Focus on arithmetic correctness only.\n"
        "- Prefer numerical or calculation-level degradation rather than unrelated story changes.\n"
        "- Do not mainly degrade by removing explanation unless the rubric itself requires missing calculations.\n"
    ),
    "factual correctness": (
        "- Focus on whether stated facts, values, or quantities match the question.\n"
        "- Prefer incorrect facts or corrupted values.\n"
        "- Avoid degrading mainly through writing style.\n"
    ),
    "logical consistency": (
        "- Focus on whether reasoning steps logically follow from one another.\n"
        "- Prefer invalid operations, broken inference links, or unit/step mismatches.\n"
        "- Do not rely only on a trivial arithmetic typo unless it creates a genuine logic break under the rubric.\n"
    ),
    "internal coherence": (
        "- Focus on consistency between different parts of the answer.\n"
        "- Prefer contradictions between steps, unresolved self-corrections, or mismatched intermediate values.\n"
    ),
    "argument rigor": (
        "- Focus on justification quality.\n"
        "- Prefer omitted justification, vague unsupported leaps, or compressed reasoning.\n"
        "- Do not mainly degrade by changing unrelated facts unless needed by the rubric.\n"
    ),
    "answer completeness": (
        "- Focus on whether all requested parts are present.\n"
        "- Prefer omitting one required component instead of making the whole answer random or nonsensical.\n"
    ),
    "expression naturalness": (
        "- Focus on human-like flow and phrasing.\n"
        "- Low scores should still look like realistic weak student writing, not random gibberish.\n"
    ),
    "readability": (
        "- Focus on spacing, structure, and ease of reading.\n"
        "- Prefer awkward but readable presentation over factual corruption.\n"
    ),
    "comprehensibility": (
        "- Focus on whether a basic reader can follow how the answer was obtained.\n"
        "- Prefer missing or unclear explanatory links rather than changing all facts.\n"
    ),
}

# Patterns to detect dirty formatting
DIRTY_TEXT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"```"), "contains code fence"),
    (re.compile(r"(?m)^\s*#{1,6}\s*"), "contains markdown heading"),
    (re.compile(r"(?m)^\s*[-*]\s+"), "contains bullet list"),
    (re.compile(r"(?m)^\s*\d+[.)]\s+"), "contains numbered list"),
    (re.compile(r"\*\*|__"), "contains bold marker"),
    (re.compile(r"####"), "contains #### marker"),
    (re.compile(r"(?im)(^|\n)\s*(score|reason|revision suggestions|edit intent|modified answer|generated_answer|modified_answer)\s*:"), "contains field label"),
]

# Common English words for gibberish detection
COMMON_ENGLISH_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
    "for", "from", "has", "have", "he", "her", "his", "how", "if", "in",
    "is", "it", "may", "more", "of", "on", "or", "she", "so", "than",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "total", "was", "were", "what", "which", "with", "worked", "sold",
    "clips", "pages", "pounds", "pieces", "pizza", "wallet", "need",
    "needs", "earned", "money", "today", "yesterday", "altogether",
    "april", "may", "month", "months", "answer", "final", "remaining",
    "read", "book", "hour", "hours", "minutes", "minute", "per", "each",
    "equals", "now", "still",
}

# Phrases that indicate a too-generic reason
GENERIC_REASON_PHRASES = {
    "could be improved",
    "somewhat unclear",
    "a little weak",
    "needs improvement",
    "slightly weak",
    "could be clearer",
}

# Regex for word and number extraction
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_dimension_name(value: Any) -> str:
    """Convert dimension name to a canonical lowercase string."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_hash(text: str) -> str:
    """Return SHA1 hash of the input text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_sample_id(record: Dict[str, Any]) -> str:
    """Generate a unique ID for a sample based on question, answer, dimension, and criteria."""
    if "sample_id" in record and record["sample_id"]:
        return str(record["sample_id"])
    payload = {
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "dimension_name": record.get("dimension_name", ""),
        "score_criteria": record.get("score_criteria", record.get("0-5_Criteria", "")),
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_task_id(sample_id: str, target_score: int) -> str:
    """Create a unique task identifier from sample_id and target score."""
    return f"{sample_id}::score={target_score}"


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records from the given path."""
    return api_base.load_records(path)


def derive_review_output_path(output_path: Path) -> Path:
    """Generate review output path by inserting '.review' before the extension."""
    return output_path.with_name(f"{output_path.stem}.review{output_path.suffix}")


# ============================================================================
# Completed Tasks Loading
# ============================================================================

def _load_completed_task_ids_from_path(path: Path) -> set[str]:
    """Load completed task IDs from a single JSONL output file."""
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            task_id = row.get("task_id")
            if task_id:
                completed.add(str(task_id))
                continue
            sample_id = row.get("sample_id")
            score = row.get("Score")
            if sample_id is not None and score is not None:
                try:
                    completed.add(build_task_id(str(sample_id), int(score)))
                except Exception:
                    pass
    return completed


def load_completed_task_ids(output_paths: List[Path]) -> set[str]:
    """Load all completed task IDs from multiple output files."""
    completed: set[str] = set()
    for path in output_paths:
        completed |= _load_completed_task_ids_from_path(path)
    return completed


# ============================================================================
# Rubric Helpers
# ============================================================================

def parse_score_criteria_map(record: Dict[str, Any]) -> Dict[str, str]:
    """Extract a dictionary mapping score (0-5) to criterion text."""
    if isinstance(record.get("score_criteria"), dict):
        return {str(k): str(v) for k, v in record["score_criteria"].items()}
    raw = record.get("0-5_Criteria")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
    # Fallback: try individual criteria_0..5 fields
    maybe_map: Dict[str, str] = {}
    for i in range(6):
        key = f"criteria_{i}"
        if key in record:
            maybe_map[str(i)] = str(record[key])
    return maybe_map


def normalize_score_criteria(record: Dict[str, Any]) -> str:
    """Return a JSON string of the score criteria map."""
    return json.dumps(parse_score_criteria_map(record), ensure_ascii=False)


def format_score_criteria_for_prompt(record: Dict[str, Any]) -> str:
    """Format the rubric as human-readable lines."""
    score_map = parse_score_criteria_map(record)
    lines: List[str] = []
    for i in range(6):
        key = str(i)
        if key in score_map:
            lines.append(f"{i}: {score_map[key]}")
    return "\n".join(lines) if lines else "{}"


def build_score_band_guidance(record: Dict[str, Any], target_score: int) -> str:
    """Create guidance that includes target, lower, and upper adjacent criteria."""
    score_map = parse_score_criteria_map(record)
    parts: List[str] = []
    parts.append(f"Target criterion ({target_score}): {score_map.get(str(target_score), '(missing)')}")
    if target_score - 1 >= 0:
        parts.append(f"Lower adjacent criterion ({target_score - 1}): {score_map.get(str(target_score - 1), '(missing)')}")
    if target_score + 1 <= 5:
        parts.append(f"Upper adjacent criterion ({target_score + 1}): {score_map.get(str(target_score + 1), '(missing)')}")
    parts.append("The generated_answer must fit the target criterion more closely than either adjacent criterion.")
    return "\n".join(parts)


def build_score_5_guidance(record: Dict[str, Any]) -> str:
    """Return the criterion for score 5."""
    score_map = parse_score_criteria_map(record)
    return score_map.get("5", "(missing)")


def get_dimension_guidance(dimension_name: str) -> str:
    """Return dimension-specific degradation guidance."""
    dim = normalize_dimension_name(dimension_name)
    for key, guidance in DIMENSION_GUIDANCE.items():
        if key == dim or key in dim or dim in key:
            return guidance
    return (
        "- Stay tightly focused on the named dimension.\n"
        "- Use the rubric itself to separate the target score from adjacent scores.\n"
        "- Keep other dimensions as intact as reasonably possible.\n"
    )


# ============================================================================
# Prompt Builders
# ============================================================================

def build_low_score_messages(record: Dict[str, Any], target_score: int) -> List[Dict[str, str]]:
    """Construct messages for generating a low-score answer."""
    user_content = LOW_SCORE_USER_TEMPLATE.format(
        question=str(record.get("question", "")).strip(),
        dimension_name=str(record.get("dimension_name", "")).strip(),
        score_criteria=format_score_criteria_for_prompt(record),
        target_score=target_score,
        score_band_guidance=build_score_band_guidance(record, target_score),
        dimension_guidance=get_dimension_guidance(str(record.get("dimension_name", ""))),
    )
    return [
        {"role": "system", "content": LOW_SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_rewrite_messages(
    record: Dict[str, Any],
    generated_answer: str,
    reason: str,
) -> List[Dict[str, str]]:
    """Construct messages for rewriting a low-score answer to score 5."""
    user_content = REWRITE_USER_TEMPLATE.format(
        question=str(record.get("question", "")).strip(),
        generated_answer=generated_answer.strip(),
        reason=reason.strip(),
        reference_answer=str(record.get("answer", "")).strip(),
        dimension_name=str(record.get("dimension_name", "")).strip(),
        score_criteria=format_score_criteria_for_prompt(record),
        score_5_guidance=build_score_5_guidance(record),
        dimension_guidance=get_dimension_guidance(str(record.get("dimension_name", ""))),
    )
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_generation_repair_messages(
    record: Dict[str, Any],
    target_score: int,
    bad_payload: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Construct messages to repair an invalid generation response."""
    repair_user = (
        f"Question:\n{str(record.get('question', '')).strip()}\n\n"
        f"Dimension:\n{str(record.get('dimension_name', '')).strip()}\n\n"
        f"0-5 Score Criteria:\n{format_score_criteria_for_prompt(record)}\n\n"
        f"Target Score:\n{target_score}\n\n"
        f"Rubric focus for exact score control:\n{build_score_band_guidance(record, target_score)}\n\n"
        "Previous invalid/incomplete JSON:\n"
        f"{json.dumps(bad_payload or {}, ensure_ascii=False, indent=2)}\n\n"
        "Rewrite into strict JSON only with keys:\n"
        '{"Score": <int>, "Reason": "...", "revision_suggestions": "...", "generated_answer": "..."}'
    )
    return [
        {"role": "system", "content": GEN_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": repair_user},
    ]


def build_rewrite_repair_messages(
    record: Dict[str, Any],
    generated_answer: str,
    reason: str,
    bad_payload: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Construct messages to repair an invalid rewrite response."""
    repair_user = (
        f"Question:\n{str(record.get('question', '')).strip()}\n\n"
        f"Generated Answer:\n{generated_answer.strip()}\n\n"
        f"Original Reason:\n{reason.strip()}\n\n"
        f"Reference Answer:\n{str(record.get('answer', '')).strip()}\n\n"
        f"Dimension:\n{str(record.get('dimension_name', '')).strip()}\n\n"
        f"0-5 Score Criteria:\n{format_score_criteria_for_prompt(record)}\n\n"
        f"Score-5 Criterion:\n{build_score_5_guidance(record)}\n\n"
        "Previous invalid/incomplete JSON:\n"
        f"{json.dumps(bad_payload or {}, ensure_ascii=False, indent=2)}\n\n"
        "Rewrite into strict JSON only with keys:\n"
        '{"modified_answer": "..."}'
    )
    return [
        {"role": "system", "content": REWRITE_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": repair_user},
    ]


# ============================================================================
# Text Analysis Helpers
# ============================================================================

def normalize_text_for_compare(text: str) -> str:
    """Normalize text for similarity comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text: str) -> set[str]:
    """Return a set of lowercase word tokens."""
    return {w.lower() for w in WORD_RE.findall(text) if w.strip()}


def token_list(text: str) -> List[str]:
    """Return a list of lowercase word tokens."""
    return [w.lower() for w in WORD_RE.findall(text) if w.strip()]


def word_count(text: str) -> int:
    """Count the number of word tokens."""
    return len(WORD_RE.findall(text))


def number_count(text: str) -> int:
    """Count the number of numeric tokens."""
    return len(NUMBER_RE.findall(text))


def common_word_ratio(text: str) -> float:
    """Proportion of tokens that are common English words."""
    tokens = token_list(text)
    if not tokens:
        return 0.0
    common = sum(1 for t in tokens if t in COMMON_ENGLISH_WORDS)
    return common / len(tokens)


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity of token sets."""
    sa = token_set(a)
    sb = token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def looks_like_gibberish(text: str) -> bool:
    """Heuristic to detect gibberish or nonsensical text."""
    stripped = text.strip()
    if not stripped:
        return True
    words = WORD_RE.findall(stripped)
    if len(words) <= 2 and len(stripped) <= 10:
        return False

    # Check vowel density
    vowel_count = sum(ch.lower() in "aeiou" for ch in stripped if ch.isalpha())
    alpha_count = sum(ch.isalpha() for ch in stripped)
    if alpha_count >= 20 and vowel_count / max(alpha_count, 1) < GIBBERISH_VOWEL_ALPHA_RATIO:
        return True

    # Check for weird tokens (long words without vowels)
    weird_token_count = 0
    for w in words:
        if len(w) >= 5 and not re.search(r"[aeiouAEIOU]", w):
            weird_token_count += 1
    if weird_token_count >= GIBBERISH_WEIRD_TOKEN_COUNT:
        return True

    # Check for very low common word ratio
    if (len(words) >= 4 and
            not any(ch.isdigit() for ch in stripped) and
            all(w.isalpha() for w in words) and
            common_word_ratio(stripped) < GIBBERISH_COMMON_WORD_RATIO):
        return True

    return False


def count_calculation_patterns(text: str) -> int:
    """Count explicit arithmetic or numeric operation patterns."""
    patterns = [
        r"\d+\s*[\+\-\*/×÷]\s*\d+",
        r"half of",
        r"twice",
        r"triple",
        r"double",
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, text, flags=re.I))
    return count


def find_dirty_format_reasons(text: str) -> List[str]:
    """Return a list of dirty formatting issues found in the text."""
    reasons: List[str] = []
    for pattern, label in DIRTY_TEXT_PATTERNS:
        if pattern.search(text):
            reasons.append(label)
    return reasons


# ============================================================================
# Quality Checks
# ============================================================================

def quality_check_generated(
    normalized_generation: Dict[str, Any],
    record: Dict[str, Any],
    target_score: int,
) -> Tuple[bool, List[str]]:
    """
    Check quality of the generated low-score answer.
    Returns (is_ok, list_of_issues).
    """
    issues: List[str] = []
    reference_answer = str(record.get("answer", "")).strip()
    generated_answer = str(normalized_generation.get("generated_answer", "")).strip()
    reason = str(normalized_generation.get("Reason", "")).strip()
    dimension = normalize_dimension_name(record.get("dimension_name", ""))

    if not generated_answer:
        issues.append("generated_answer is empty")
    if not reason:
        issues.append("Reason is empty")

    issues.extend([f"generated_answer {x}" for x in find_dirty_format_reasons(generated_answer)])
    issues.extend([f"Reason {x}" for x in find_dirty_format_reasons(reason)])

    if looks_like_gibberish(generated_answer):
        issues.append("generated_answer appears gibberish-like")

    sim_ref_gen = jaccard_similarity(reference_answer, generated_answer)
    if target_score <= 2 and sim_ref_gen >= HIGH_SIMILARITY_THRESHOLD:
        issues.append("low-score generated_answer too similar to reference_answer")

    if "expression naturalness" in dimension and looks_like_gibberish(generated_answer):
        issues.append("unnatural gibberish for expression naturalness")

    if target_score == 0 and dimension in {"data accuracy", "argument rigor", "comprehensibility"}:
        if generated_answer and sim_ref_gen >= VERY_HIGH_SIMILARITY_THRESHOLD:
            issues.append("score-0 generated_answer too similar to reference_answer")

    if target_score == 0 and count_calculation_patterns(generated_answer) >= 4:
        issues.append("score-0 answer contains too many explicit calculations")

    return len(issues) == 0, issues


def quality_check_final(
    normalized_generation: Dict[str, Any],
    normalized_rewrite: Dict[str, Any],
    record: Dict[str, Any],
    target_score: int,
) -> Tuple[bool, List[str]]:
    """
    Check quality of the final pair (low-score + rewritten score-5).
    Returns (is_ok, list_of_issues).
    """
    issues: List[str] = []
    generated_answer = str(normalized_generation.get("generated_answer", "")).strip()
    modified_answer = str(normalized_rewrite.get("modified_answer", "")).strip()

    if not modified_answer:
        issues.append("modified_answer is empty")
    issues.extend([f"modified_answer {x}" for x in find_dirty_format_reasons(modified_answer)])

    if normalize_text_for_compare(generated_answer) == normalize_text_for_compare(modified_answer):
        issues.append("generated_answer identical to modified_answer")

    sim_gen_mod = jaccard_similarity(generated_answer, modified_answer)
    if target_score == 4 and sim_gen_mod >= EXTREME_SIMILARITY_THRESHOLD:
        issues.append("score-4 generated_answer too similar to modified_answer")

    if looks_like_gibberish(modified_answer):
        issues.append("modified_answer appears gibberish-like")

    return len(issues) == 0, issues


def collect_review_flags(
    normalized_generation: Dict[str, Any],
    normalized_rewrite: Dict[str, Any],
    record: Dict[str, Any],
    target_score: int,
) -> List[str]:
    """Collect non-fatal flags that suggest the sample should be reviewed."""
    flags: List[str] = []
    dimension = normalize_dimension_name(record.get("dimension_name", ""))
    generated_answer = str(normalized_generation.get("generated_answer", "")).strip()
    reason = str(normalized_generation.get("Reason", "")).strip()
    modified_answer = str(normalized_rewrite.get("modified_answer", "")).strip()
    reference_answer = str(record.get("answer", "")).strip()

    reason_lower = reason.lower()
    if word_count(reason) < MIN_REASON_WORD_COUNT:
        flags.append("reason_too_short")

    if not any(marker in reason_lower for marker in [
        "because", "fails", "missing", "incorrect", "omits",
        "does not", "contradict", "invalid"
    ]):
        flags.append("reason_may_be_too_generic")

    if any(phrase in reason_lower for phrase in GENERIC_REASON_PHRASES):
        flags.append("reason_contains_generic_phrase")

    if jaccard_similarity(reference_answer, modified_answer) >= NEAR_IDENTICAL_THRESHOLD:
        flags.append("modified_answer_nearly_identical_to_reference")

    if dimension in {"argument rigor", "logical consistency", "comprehensibility"} and target_score == 0:
        if word_count(generated_answer) <= 3:
            flags.append("zero_short_answer")

    if dimension == "expression naturalness":
        if target_score == 0 and common_word_ratio(generated_answer) < 0.12:
            flags.append("naturalness_score0_may_be_too_nonsensical")

    if dimension == "answer completeness":
        n_nums = number_count(generated_answer)
        if target_score == 0 and n_nums >= 2:
            flags.append("answer_completeness_score0_possible_rubric_issue")
        if target_score in {1, 2} and n_nums >= 3:
            flags.append("answer_completeness_mid_score_has_many_numbers")

    if dimension == "readability":
        formatting_signal = 0
        if " " in generated_answer:
            formatting_signal += 1
        if "\n\n" in generated_answer:
            formatting_signal += 1
        if re.search(r"\s+[.,;:!?]", generated_answer):
            formatting_signal += 1
        if target_score == 3 and formatting_signal == 0:
            flags.append("readability_score3_weak_format_signal")
        if target_score == 4 and formatting_signal >= 3:
            flags.append("readability_score4_too_noisy")

    return sorted(set(flags))


def assess_sample_quality(
    normalized_generation: Dict[str, Any],
    normalized_rewrite: Dict[str, Any],
    record: Dict[str, Any],
    target_score: int,
) -> Tuple[str, List[str]]:
    """
    Determine final decision: 'keep', 'review', or 'drop'.
    Returns (decision, list_of_flags).
    """
    hard_flags: List[str] = []

    gen_ok, gen_flags = quality_check_generated(
        normalized_generation=normalized_generation,
        record=record,
        target_score=target_score,
    )
    if not gen_ok:
        hard_flags.extend(gen_flags)

    final_ok, final_flags = quality_check_final(
        normalized_generation=normalized_generation,
        normalized_rewrite=normalized_rewrite,
        record=record,
        target_score=target_score,
    )
    if not final_ok:
        hard_flags.extend(final_flags)

    review_flags = collect_review_flags(
        normalized_generation=normalized_generation,
        normalized_rewrite=normalized_rewrite,
        record=record,
        target_score=target_score,
    )

    if hard_flags:
        return "drop", sorted(set(hard_flags + review_flags))
    if review_flags:
        return "review", review_flags
    return "keep", []


# ============================================================================
# Payload Normalization / Validation
# ============================================================================

def normalize_generation_payload(
    parsed: Optional[Dict[str, Any]],
    target_score: int,
) -> Optional[Dict[str, Any]]:
    """Normalize a generation API response into a standard dict."""
    if not isinstance(parsed, dict):
        return None
    if "generated_answer" not in parsed:
        return None
    score = parsed.get("Score", target_score)
    try:
        score = int(score)
    except Exception:
        return None
    return {
        "Score": score,
        "Reason": str(parsed.get("Reason", "")).strip(),
        "revision_suggestions": str(
            parsed.get("revision_suggestions")
            or parsed.get("edit_intent")
            or parsed.get("Revision Suggestions")
            or parsed.get("Edit Intent")
            or ""
        ).strip(),
        "generated_answer": str(parsed.get("generated_answer", "")).strip(),
    }


def validate_generation_payload(
    normalized: Optional[Dict[str, Any]],
    target_score: int,
) -> Tuple[bool, List[str]]:
    """Validate a normalized generation payload."""
    issues: List[str] = []
    if normalized is None:
        return False, ["normalized generation payload is None"]
    for key in ["Score", "Reason", "revision_suggestions", "generated_answer"]:
        if key not in normalized:
            issues.append(f"missing key: {key}")
    score = normalized.get("Score")
    if not isinstance(score, int):
        issues.append("Score is not int")
    elif score != target_score:
        issues.append(f"Score {score} does not match target_score {target_score}")
    if not str(normalized.get("Reason", "")).strip():
        issues.append("Reason is empty")
    if not str(normalized.get("revision_suggestions", "")).strip():
        issues.append("revision_suggestions is empty")
    if not str(normalized.get("generated_answer", "")).strip():
        issues.append("generated_answer is empty")
    return len(issues) == 0, issues


def normalize_rewrite_payload(parsed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize a rewrite API response into a standard dict."""
    if not isinstance(parsed, dict):
        return None
    if "modified_answer" not in parsed:
        return None
    return {"modified_answer": str(parsed.get("modified_answer") or "").strip()}


def validate_rewrite_payload(normalized: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Validate a normalized rewrite payload."""
    issues: List[str] = []
    if normalized is None:
        return False, ["normalized rewrite payload is None"]
    if "modified_answer" not in normalized:
        issues.append("missing key: modified_answer")
    if not str(normalized.get("modified_answer", "")).strip():
        issues.append("modified_answer is empty")
    return len(issues) == 0, issues


# ============================================================================
# Repair Helpers
# ============================================================================

async def maybe_repair_generation_payload(
    base_url: str,
    api_key: str,
    model: str,
    parsed: Optional[Dict[str, Any]],
    record: Dict[str, Any],
    target_score: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    request_timeout: int,
    http_session: Optional["api_base.aiohttp.ClientSession"],
    request_executor: Optional[ThreadPoolExecutor],
    repair_attempts: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Attempt to repair an invalid generation payload."""
    normalized = normalize_generation_payload(parsed, target_score=target_score)
    is_valid, _ = validate_generation_payload(normalized, target_score=target_score)
    if is_valid:
        return normalized, "ok"

    last_bad = parsed
    for attempt in range(repair_attempts):
        repair_messages = build_generation_repair_messages(
            record=record,
            target_score=target_score,
            bad_payload=last_bad,
        )
        repaired_parsed, _ = await api_base.call_model(
            base_url, model=model, api_key=api_key,
            messages=repair_messages, max_tokens=max_tokens, temperature=temperature,
            retries=retries, timeout_seconds=request_timeout,
            http_session=http_session, request_executor=request_executor,
        )
        repaired_normalized = normalize_generation_payload(repaired_parsed, target_score=target_score)
        is_valid, _ = validate_generation_payload(repaired_normalized, target_score=target_score)
        if is_valid:
            return repaired_normalized, f"generation_repaired_{attempt + 1}"
        last_bad = repaired_parsed
    return None, "generation_failed_validation"


async def maybe_repair_rewrite_payload(
    base_url: str,
    api_key: str,
    model: str,
    parsed: Optional[Dict[str, Any]],
    record: Dict[str, Any],
    generated_answer: str,
    reason: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    request_timeout: int,
    http_session: Optional["api_base.aiohttp.ClientSession"],
    request_executor: Optional[ThreadPoolExecutor],
    repair_attempts: int,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Attempt to repair an invalid rewrite payload."""
    normalized = normalize_rewrite_payload(parsed)
    is_valid, _ = validate_rewrite_payload(normalized)
    if is_valid:
        return normalized, "ok"

    last_bad = parsed
    for attempt in range(repair_attempts):
        repair_messages = build_rewrite_repair_messages(
            record=record,
            generated_answer=generated_answer,
            reason=reason,
            bad_payload=last_bad,
        )
        repaired_parsed, _ = await api_base.call_model(
            base_url, model=model, api_key=api_key,
            messages=repair_messages, max_tokens=max_tokens, temperature=temperature,
            retries=retries, timeout_seconds=request_timeout,
            http_session=http_session, request_executor=request_executor,
        )
        repaired_normalized = normalize_rewrite_payload(repaired_parsed)
        is_valid, _ = validate_rewrite_payload(repaired_normalized)
        if is_valid:
            return repaired_normalized, f"rewrite_repaired_{attempt + 1}"
        last_bad = repaired_parsed
    return None, "rewrite_failed_validation"


# ============================================================================
# Task Building
# ============================================================================

def build_generation_tasks(
    records: List[Dict[str, Any]],
    output_path: Path,
    review_output_path: Optional[Path],
) -> List[Dict[str, Any]]:
    """Create a list of pending generation tasks based on already completed ones."""
    completed_paths = [output_path]
    if review_output_path is not None:
        completed_paths.append(review_output_path)
    completed_task_ids = load_completed_task_ids(completed_paths)

    tasks: List[Dict[str, Any]] = []
    for record in records:
        sample_id = build_sample_id(record)
        for target_score in TARGET_SCORES:
            task_id = build_task_id(sample_id, target_score)
            if task_id in completed_task_ids:
                continue
            tasks.append({
                "record": record,
                "sample_id": sample_id,
                "target_score": target_score,
                "task_id": task_id,
                "output_path": output_path,
                "review_output_path": review_output_path,
            })
    print(
        f"[Answer Generation] output={output_path} "
        f"review_output={review_output_path} "
        f"records={len(records)} "
        f"completed_tasks={len(completed_task_ids)} "
        f"remaining_tasks={len(tasks)}"
    )
    return tasks


# ============================================================================
# Statistics Collector
# ============================================================================

class StatsCollector:
    """Collect and aggregate generation statistics per endpoint."""

    def __init__(self) -> None:
        self.global_stats: defaultdict[str, int] = defaultdict(int)
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "keep_written": 0,
                "review_written": 0,
                "quality_dropped": 0,
                "parse_failed": 0,
                "validation_failed": 0,
                "exceptions": 0,
                "latency_ms_sum": 0.0,
            }
        )

    def record_keep(self, endpoint: str, latency_ms: float) -> None:
        self.global_stats["written"] += 1
        self.global_stats["keep_written"] += 1
        self.endpoint_stats[endpoint]["keep_written"] += 1
        self.endpoint_stats[endpoint]["latency_ms_sum"] += latency_ms

    def record_review(self, endpoint: str, latency_ms: float) -> None:
        self.global_stats["review_written"] += 1
        self.endpoint_stats[endpoint]["review_written"] += 1
        self.endpoint_stats[endpoint]["latency_ms_sum"] += latency_ms

    def record_quality_drop(self, endpoint: str, latency_ms: float) -> None:
        self.global_stats["quality_dropped"] += 1
        self.endpoint_stats[endpoint]["quality_dropped"] += 1
        self.endpoint_stats[endpoint]["latency_ms_sum"] += latency_ms

    def record_parse_failed(self, endpoint: str, latency_ms: float) -> None:
        self.global_stats["parse_failed"] += 1
        self.endpoint_stats[endpoint]["parse_failed"] += 1
        self.endpoint_stats[endpoint]["latency_ms_sum"] += latency_ms

    def record_validation_failed(self, endpoint: str, latency_ms: float) -> None:
        self.global_stats["validation_failed"] += 1
        self.endpoint_stats[endpoint]["validation_failed"] += 1
        self.endpoint_stats[endpoint]["latency_ms_sum"] += latency_ms

    def record_exception(self, endpoint: str) -> None:
        self.global_stats["exceptions"] += 1
        self.endpoint_stats[endpoint]["exceptions"] += 1

    def as_dict(self) -> Dict[str, Any]:
        endpoint_payload = {}
        for endpoint, stat in self.endpoint_stats.items():
            total_seen = (
                stat["keep_written"] + stat["review_written"] +
                stat["quality_dropped"] + stat["parse_failed"] +
                stat["validation_failed"]
            )
            endpoint_payload[endpoint] = {
                **stat,
                "avg_latency_ms": round(stat["latency_ms_sum"] / max(1, total_seen), 2),
            }
        return {
            "global": dict(self.global_stats),
            "endpoint_stats": endpoint_payload,
        }


# ============================================================================
# Worker
# ============================================================================

async def worker(
    base_url: str,
    api_key: str,
    model: str,
    input_queue: asyncio.Queue[Optional[Dict[str, Any]]],
    writer_queue: asyncio.Queue[Optional[Tuple[Path, Dict[str, Any]]]],
    max_tokens: int,
    temperature: float,
    retries: int,
    request_timeout: int,
    progress_bar: Any,
    http_session: Optional["api_base.aiohttp.ClientSession"],
    request_executor: Optional[ThreadPoolExecutor],
    repair_attempts: int,
    verbose_failures: bool,
    stats: StatsCollector,
) -> None:
    """Single worker that processes tasks from an endpoint-specific queue."""
    while True:
        task = None
        start_time = 0.0
        try:
            task = await input_queue.get()
        except asyncio.CancelledError:
            break
        try:
            if task is None:
                break

            start_time = time.perf_counter()
            record = task["record"]
            sample_id = task["sample_id"]
            target_score = int(task["target_score"])
            task_id = task["task_id"]
            output_path = task["output_path"]
            review_output_path: Optional[Path] = task["review_output_path"]

            # ---- Stage A: generate low-score answer ----
            generation_messages = build_low_score_messages(record, target_score)
            generation_parsed, _ = await api_base.call_model(
                base_url, model=model, api_key=api_key,
                messages=generation_messages, max_tokens=max_tokens, temperature=temperature,
                retries=retries, timeout_seconds=request_timeout,
                http_session=http_session, request_executor=request_executor,
            )
            if generation_parsed is None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                stats.record_parse_failed(base_url, latency_ms)
                if verbose_failures:
                    print(f"[WARN] generation parse failed task_id={task_id}")
                continue

            normalized_generation, generation_status = await maybe_repair_generation_payload(
                base_url=base_url, api_key=api_key, model=model,
                parsed=generation_parsed, record=record, target_score=target_score,
                max_tokens=max_tokens, temperature=temperature, retries=retries,
                request_timeout=request_timeout, http_session=http_session,
                request_executor=request_executor, repair_attempts=repair_attempts,
            )
            if normalized_generation is None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                stats.record_validation_failed(base_url, latency_ms)
                if verbose_failures:
                    print(f"[WARN] generation validation failed task_id={task_id} "
                          f"dimension={record.get('dimension_name')}")
                continue

            # ---- Stage B: rewrite to score-5 answer ----
            rewrite_messages = build_rewrite_messages(
                record=record,
                generated_answer=normalized_generation["generated_answer"],
                reason=normalized_generation["Reason"],
            )
            rewrite_parsed, _ = await api_base.call_model(
                base_url, model=model, api_key=api_key,
                messages=rewrite_messages, max_tokens=max_tokens, temperature=temperature,
                retries=retries, timeout_seconds=request_timeout,
                http_session=http_session, request_executor=request_executor,
            )
            if rewrite_parsed is None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                stats.record_parse_failed(base_url, latency_ms)
                if verbose_failures:
                    print(f"[WARN] rewrite parse failed task_id={task_id}")
                continue

            normalized_rewrite, rewrite_status = await maybe_repair_rewrite_payload(
                base_url=base_url, api_key=api_key, model=model,
                parsed=rewrite_parsed, record=record,
                generated_answer=normalized_generation["generated_answer"],
                reason=normalized_generation["Reason"],
                max_tokens=max_tokens, temperature=temperature, retries=retries,
                request_timeout=request_timeout, http_session=http_session,
                request_executor=request_executor, repair_attempts=repair_attempts,
            )
            if normalized_rewrite is None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                stats.record_validation_failed(base_url, latency_ms)
                if verbose_failures:
                    print(f"[WARN] rewrite validation failed task_id={task_id} "
                          f"dimension={record.get('dimension_name')}")
                continue

            # ---- Quality assessment ----
            quality_decision, quality_flags = assess_sample_quality(
                normalized_generation=normalized_generation,
                normalized_rewrite=normalized_rewrite,
                record=record,
                target_score=target_score,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            output_row = {
                "task_id": task_id,
                "sample_id": sample_id,
                "question": record.get("question"),
                "reference_answer": record.get("answer"),
                "dimension_name": record.get("dimension_name"),
                "0-5_Criteria": normalize_score_criteria(record),
                "Score": normalized_generation["Score"],
                "Reason": normalized_generation["Reason"],
                "revision_suggestions": normalized_generation["revision_suggestions"],
                "edit_intent": normalized_generation["revision_suggestions"],
                "answer": normalized_generation["generated_answer"],
                "generated_answer": normalized_generation["generated_answer"],
                "modified_answer": normalized_rewrite["modified_answer"],
                "generation_status": f"{generation_status};{rewrite_status}",
                "quality_decision": quality_decision,
                "quality_flags": quality_flags,
            }

            if quality_decision == "keep":
                await writer_queue.put((output_path, output_row))
                stats.record_keep(base_url, latency_ms)
            elif quality_decision == "review":
                if review_output_path is not None:
                    await writer_queue.put((review_output_path, output_row))
                    stats.record_review(base_url, latency_ms)
                else:
                    stats.record_quality_drop(base_url, latency_ms)
                    if verbose_failures:
                        print(f"[WARN] review sample dropped because review_output is disabled "
                              f"task_id={task_id} flags={quality_flags}")
            else:  # drop
                stats.record_quality_drop(base_url, latency_ms)
                if verbose_failures:
                    print(f"[WARN] quality drop task_id={task_id} flags={quality_flags}")

        except Exception as exc:
            stats.record_exception(base_url)
            if verbose_failures:
                print(f"[ERROR] worker failed on {base_url}: {exc}")
        finally:
            input_queue.task_done()
            if task is not None and progress_bar is not None:
                progress_bar.update(1)


# ============================================================================
# Main Generation Orchestration
# ============================================================================

async def run_generation(
    base_urls: List[str],
    api_key: str,
    model: str,
    tasks: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    retries: int,
    concurrency_per_endpoint: int,
    request_timeout: int,
    dispatch_poll_interval: float,
    repair_attempts: int,
    verbose_failures: bool,
) -> Dict[str, Any]:
    """Orchestrate the entire generation process across endpoints."""
    if not tasks:
        print("No pending tasks. Nothing to generate.")
        return {"global": {"written": 0}, "endpoint_stats": {}}

    stats = StatsCollector()

    # Progress bar (if tqdm is available)
    progress_bar = None
    if api_base.tqdm is not None:
        progress_bar = api_base.tqdm(
            total=len(tasks), desc="Generating 0-4 Answers",
            unit="task", dynamic_ncols=True,
        )
    else:
        print("tqdm is not installed; progress bar disabled.")

    queue_capacity = max(1, concurrency_per_endpoint * 2)
    endpoint_queues = [asyncio.Queue(maxsize=queue_capacity) for _ in base_urls]
    writer_queue: asyncio.Queue[Optional[Tuple[Path, Dict[str, Any]]]] = asyncio.Queue()
    writer_task = asyncio.create_task(api_base.multi_writer(writer_queue))
    dispatcher_task = asyncio.create_task(
        api_base.dispatch_tasks_round_robin(
            tasks, endpoint_queues, poll_interval=dispatch_poll_interval,
        )
    )

    workers: List[asyncio.Task] = []
    worker_count = max(1, concurrency_per_endpoint)
    total_worker_count = max(1, len(base_urls) * worker_count)

    async def launch_workers(
        http_session: Optional["api_base.aiohttp.ClientSession"],
        request_executor: Optional[ThreadPoolExecutor],
    ) -> None:
        for base_url, input_queue in zip(base_urls, endpoint_queues):
            for _ in range(worker_count):
                workers.append(asyncio.create_task(
                    worker(
                        base_url=base_url, api_key=api_key, model=model,
                        input_queue=input_queue, writer_queue=writer_queue,
                        max_tokens=max_tokens, temperature=temperature,
                        retries=retries, request_timeout=request_timeout,
                        progress_bar=progress_bar, http_session=http_session,
                        request_executor=request_executor, repair_attempts=repair_attempts,
                        verbose_failures=verbose_failures, stats=stats,
                    )
                ))
        await dispatcher_task
        await asyncio.gather(*workers)

    try:
        if api_base.aiohttp is not None:
            connector = api_base.aiohttp.TCPConnector(limit=max(64, total_worker_count * 2))
            timeout = api_base.aiohttp.ClientTimeout(total=request_timeout)
            async with api_base.aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
                await launch_workers(http_session=http_session, request_executor=None)
        else:
            with ThreadPoolExecutor(max_workers=max(32, total_worker_count)) as request_executor:
                await launch_workers(http_session=None, request_executor=request_executor)
    finally:
        await writer_queue.put(None)
        await writer_queue.join()
        await writer_task
        if progress_bar is not None:
            progress_bar.close()

    return stats.as_dict()


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 0-4 score answers from 0-5 score criteria via multiple OpenAI-compatible endpoints."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input json/jsonl path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Main keep-output jsonl path.")
    parser.add_argument(
        "--review-output", default=None,
        help="Optional review-output jsonl path. Defaults to <output>.review.jsonl unless disabled.",
    )
    parser.add_argument(
        "--disable-review-output", action="store_true",
        help="Disable writing review-flagged samples.",
    )
    parser.add_argument(
        "--base-url-template", default=DEFAULT_BASE_URL_TEMPLATE,
        help="Base URL template, e.g. http://localhost:{port}/v1",
    )
    parser.add_argument(
        "--ports", nargs="*", default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"],
        help="Port list or ranges, e.g. --ports 8000-8003",
    )
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by the API.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retry count per request.")
    parser.add_argument(
        "--concurrency-per-endpoint", type=int, default=32,
        help="Worker count per endpoint for local GPU-backed endpoints.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument(
        "--dispatch-poll-interval", type=float, default=DEFAULT_DISPATCH_POLL_INTERVAL,
        help="Polling interval in seconds when all endpoint queues are temporarily full.",
    )
    parser.add_argument(
        "--skip-health-check", action="store_true",
        help="Skip polling /models before generation.",
    )
    parser.add_argument(
        "--health-check-timeout", type=int, default=10,
        help="HTTP timeout in seconds for endpoint health checks.",
    )
    parser.add_argument(
        "--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout in seconds for each generation request.",
    )
    parser.add_argument(
        "--repair-attempts", type=int, default=DEFAULT_REPAIR_ATTEMPTS,
        help="Automatic repair attempts after JSON validation fails.",
    )
    parser.add_argument(
        "--verbose-failures", action="store_true",
        help="Print parse / validation / quality failures for debugging.",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.disable_review_output:
        review_output_path = None
    else:
        review_output_path = (
            Path(args.review_output)
            if args.review_output is not None
            else derive_review_output_path(output_path)
        )
    if review_output_path is not None and review_output_path.resolve() == output_path.resolve():
        raise ValueError("review_output_path must be different from output_path")

    records = load_records(input_path)
    if args.limit is not None:
        records = records[:max(0, args.limit)]

    ports = api_base.parse_ports(args.ports)
    base_urls = api_base.build_base_urls(args.base_url_template, ports)
    if not args.skip_health_check:
        await api_base.wait_for_servers(base_urls, timeout_seconds=args.health_check_timeout)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    tasks = build_generation_tasks(
        records=records,
        output_path=output_path,
        review_output_path=review_output_path,
    )

    summary = await run_generation(
        base_urls=base_urls,
        api_key=api_key,
        model=args.model,
        tasks=tasks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        concurrency_per_endpoint=args.concurrency_per_endpoint,
        request_timeout=args.request_timeout,
        dispatch_poll_interval=args.dispatch_poll_interval,
        repair_attempts=args.repair_attempts,
        verbose_failures=args.verbose_failures,
    )

    print(f"Done! Main output written to: {output_path}")
    if review_output_path is not None:
        print(f"Review output written to: {review_output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
