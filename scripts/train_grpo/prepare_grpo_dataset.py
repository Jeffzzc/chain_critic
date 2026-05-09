#!/usr/bin/env python
"""Prepare ChainCritic GRPO data for ms-swift.

GRPO should train from prompts only. This script converts the repository's flat
score/reason/rewrite JSONL rows into ms-swift chat-format rows:

  {
    "messages": [{"role": "system", ...}, {"role": "user", ...}],
    "question": "...",
    "answer": "...",
    "dimension_name": "...",
    "criteria_text": "Score 0: ...\n...",
    "target_score": 3,
    "target_reason": "...",
    "target_modified_answer": "...",
    "gt_answer": "..."
  }

The extra columns are consumed by scripts/train/grpo_reward_plugin.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = REPO_ROOT / "scripts" / "evaluation_pipeline"
if str(EVAL_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import normalize_score_criteria, normalize_text  # noqa: E402


SYSTEM_PROMPT = """You are an AI evaluator-and-rewriter.
Evaluate the given answer strictly using ONLY the provided evaluation dimension and its complete score criteria.
Then revise the answer to better satisfy ONLY that dimension.
Do not add unsupported facts. If an assumption is necessary, state it minimally and explicitly.
Output plain text in exactly 3 lines, with exactly these prefixes and no numbering:
Score: <an integer from 0 to 5>
Reason: <one-line concise explanation strictly based on the given dimension criteria>
Modified Answer: <one-line revised answer optimized only for the given dimension criteria>
Do not include any extra text, JSON, markdown, bullets, or extra line breaks inside any field."""

USER_TEMPLATE = """###Task Description:
You are given a question, a response to evaluate, and one evaluation dimension with its complete 0-5 score criteria.
1. Write a score that reflects how well the response satisfies the criteria.
2. Write feedback that assesses the quality of the response strictly based on the dimension criteria.
3. Then rewrite the answer so it better satisfies the criteria, without adding unsupported facts.
4. The output format must be exactly:
Score: <score>
Reason: <feedback>
Modified Answer: <rewritten answer>
5. Do not generate any other opening, closing, JSON, or explanations.

Question:
{question}

Answer:
{answer}

Evaluation Dimension:
{dimension_name}

Score Criteria (0-5):
{criteria_text}"""

CRITERION_RE = re.compile(
    r"(?ms)(?:^|\n|\s)(?:Score\s*)?([0-5])\s*[:\uFF1A]\s*(.*?)(?=(?:^|\n|\s)(?:Score\s*)?[0-5]\s*[:\uFF1A]|\Z)"
)


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def parse_int_score(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            rounded = round(number)
            if abs(number - rounded) < 1e-6 and 0 <= rounded <= 5:
                return int(rounded)
        return None
    match = re.search(r"\b([0-5])\b", str(value))
    return int(match.group(1)) if match else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object: {path}")
            yield payload


def format_score_criteria_text(row: dict[str, Any]) -> str:
    fallback_full = first_non_empty(row.get("full_score_criteria"), row.get("criteria_5"))
    criteria = normalize_score_criteria(row, fallback_full)
    raw_criteria = row.get("criteria")
    if isinstance(raw_criteria, str) and raw_criteria.strip():
        parsed = {
            score: text
            for score, text in (
                (match.group(1), normalize_text(match.group(2)))
                for match in CRITERION_RE.finditer(raw_criteria)
            )
            if text
        }
        if len(parsed) >= 5:
            criteria.update(parsed)

    pieces = []
    for score in range(6):
        text = normalize_text(criteria.get(str(score)))
        if text:
            pieces.append(f"Score {score}: {text}")
    if pieces:
        return "\n".join(pieces)
    return normalize_text(row.get("criteria") or row.get("criteria_text"))


def extract_question(row: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("question"),
        row.get("prompt"),
        row.get("instruction"),
        row.get("query"),
        row.get("problem"),
    )


def extract_answer(row: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("answer"),
        row.get("response"),
        row.get("output"),
        row.get("candidate_answer"),
        row.get("model_answer"),
    )


def extract_dimension(row: dict[str, Any]) -> str:
    return first_non_empty(row.get("dimension_name"), row.get("evaluation_dimension"), row.get("name"))


def extract_target_score(row: dict[str, Any]) -> Optional[int]:
    for field in ("reference_score", "Score", "score", "predicted_score", "target_score"):
        score = parse_int_score(row.get(field))
        if score is not None:
            return score
    return None


def extract_target_reason(row: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("reference_reason"),
        row.get("Reason"),
        row.get("reason"),
        row.get("predicted_reason"),
        row.get("target_reason"),
    )


def extract_target_modified_answer(row: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("reference_modified_answer"),
        row.get("modified_answer"),
        row.get("Modified Answer"),
        row.get("predicted_modified_answer"),
        row.get("target_modified_answer"),
    )


def extract_gt_answer(row: dict[str, Any]) -> str:
    """Return a standard answer for rewrite similarity.

    Priority:
    1. Original dataset gold/reference answer fields.
    2. GPT/reference rewritten answer fields generated by the evaluation pipeline.
    """
    return first_non_empty(
        row.get("gt_answer"),
        row.get("ground_truth"),
        row.get("ground_truth_answer"),
        row.get("gold_answer"),
        row.get("standard_answer"),
        row.get("reference_answer"),
        row.get("final_answer"),
        row.get("solution"),
        row.get("target_modified_answer"),
        row.get("reference_modified_answer"),
        row.get("modified_answer"),
        row.get("predicted_modified_answer"),
    )


def convert_row(row: dict[str, Any], index: int) -> Optional[dict[str, Any]]:
    question = extract_question(row)
    answer = extract_answer(row)
    dimension = extract_dimension(row)
    criteria_text = format_score_criteria_text(row)
    target_score = extract_target_score(row)

    if not question or not answer or not dimension or not criteria_text:
        return None
    if target_score is None:
        return None

    user_content = USER_TEMPLATE.format(
        question=question,
        answer=answer,
        dimension_name=dimension,
        criteria_text=criteria_text,
    )
    return {
        "sample_id": first_non_empty(row.get("sample_id"), row.get("id"), row.get("unique_id"), f"row:{index}"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "question": question,
        "answer": answer,
        "dimension_name": dimension,
        "criteria_text": criteria_text,
        "target_score": target_score,
        "target_reason": extract_target_reason(row),
        "target_modified_answer": extract_target_modified_answer(row),
        "gt_answer": extract_gt_answer(row),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ms-swift GRPO JSONL data.")
    parser.add_argument("--input", type=Path, required=True, help="Input flat JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output ms-swift GRPO JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(iter_jsonl(args.input), start=1):
            if args.limit is not None and written >= args.limit:
                break
            converted = convert_row(row, index)
            if converted is None:
                skipped += 1
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {args.output}; skipped {skipped} rows.")


if __name__ == "__main__":
    main()
