#!/usr/bin/env python3
"""Prepare ChainCritic GRPO data from message-format JSONL.

Input row example:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...Question: ... Answer: ... Evaluation_dimension: ... Criteria (0-5): ..."},
    {"role": "assistant", "content": "Score: 3\\nReason: ...\\nModified Answer: ..."}
  ]
}

Output row example:
{
  "sample_id": "...",
  "messages": [{"role": "system", ...}, {"role": "user", ...}],
  "question": "...",
  "answer": "...",
  "dimension_name": "...",
  "criteria_text": "Score 0: ...\\n...\\nScore 5: ...",
  "target_score": 3,
  "target_reason": "...",
  "target_modified_answer": "...",
  "gt_answer": "..."
}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_SYSTEM_PROMPT = """You are an AI evaluator-and-rewriter.
Evaluate the given answer strictly using ONLY the provided evaluation dimension and the complete 0-5 scoring criteria.
Then revise the answer to better satisfy ONLY that dimension.
Do not add unsupported facts. If an assumption is necessary, state it minimally and explicitly.
Output plain text in exactly 3 lines, with exactly these prefixes and no numbering:
Score: <an integer number from 0 to 5>
Reason: <one-line concise explanation strictly based on the given dimension criteria>
Modified Answer: <one-line revised answer optimized only for the given dimension criteria>
Do not include any extra text, JSON, markdown, bullets, or line breaks inside any field."""

CRITERION_RE = re.compile(
    r"(?ms)(?:^|\n|\s)(?:Score\s*)?([0-5])\s*[:\uFF1A]\s*(.*?)(?=(?:^|\n|\s)(?:Score\s*)?[0-5]\s*[:\uFF1A]|\Z)"
)
SCORE_LINE_RE = re.compile(r"(?im)^\s*score\s*[:\uFF1A]\s*([0-5])\s*$")
REASON_RE = re.compile(
    r"(?is)(?:^|\n)\s*reason\s*[:\uFF1A]\s*(.*?)\s*(?:(?:\n\s*)?(?:modified answer|revised answer)\s*[:\uFF1A]|$)"
)
MODIFIED_RE = re.compile(r"(?is)(?:^|\n)\s*(?:modified answer|revised answer)\s*[:\uFF1A]\s*(.*)$")

SECTION_PATTERNS = {
    "question": re.compile(r"(?im)^\s*Question\s*:\s*"),
    "answer": re.compile(r"(?im)^\s*Answer\s*:\s*"),
    "dimension_name": re.compile(r"(?im)^\s*Evaluation(?:[_ ]+Dimension)?\s*:\s*"),
    "criteria_text": re.compile(r"(?im)^\s*(?:Criteria|Score Criteria)\s*(?:\(\s*0\s*-\s*5\s*\))?\s*:\s*"),
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


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


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


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


def extract_messages(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", "", ""

    system_content = ""
    user_content = ""
    assistant_content = ""

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
        content = str(content or "")

        if role == "system" and not system_content:
            system_content = content
        elif role == "user" and not user_content:
            user_content = content
        elif role == "assistant" and not assistant_content:
            assistant_content = content

    return system_content, user_content, assistant_content


def parse_labeled_sections(text: str) -> dict[str, str]:
    matches = []
    for key, pattern in SECTION_PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            matches.append((match.start(), match.end(), key))
    matches.sort()

    sections: dict[str, str] = {}
    for idx, (_, start_content, key) in enumerate(matches):
        end_content = matches[idx + 1][0] if idx + 1 < len(matches) else len(text)
        sections[key] = normalize_text(text[start_content:end_content])
    return sections


def normalize_criteria_text(criteria_text: str) -> str:
    raw = str(criteria_text or "").strip()
    if not raw:
        return ""

    parsed = {}
    for match in CRITERION_RE.finditer(raw):
        score = match.group(1)
        text = normalize_text(match.group(2))
        if text:
            parsed[score] = text

    if parsed:
        pieces = []
        for score in range(6):
            text = parsed.get(str(score))
            if text:
                pieces.append(f"Score {score}: {text}")
        return "\n".join(pieces)

    return normalize_text(raw)


def parse_assistant_output(text: str) -> dict[str, Any]:
    raw = str(text or "").strip().replace("\r\n", "\n")
    score_match = SCORE_LINE_RE.search(raw)
    reason_match = REASON_RE.search(raw)
    modified_match = MODIFIED_RE.search(raw)

    score = int(score_match.group(1)) if score_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    modified_answer = normalize_text(modified_match.group(1)) if modified_match else ""

    return {
        "score": score,
        "reason": reason,
        "modified_answer": modified_answer,
    }


def extract_target_score(row: dict[str, Any], parsed_assistant: dict[str, Any]) -> Optional[int]:
    for field in ("reference_score", "Score", "score", "predicted_score", "target_score"):
        value = parse_int_score(row.get(field))
        if value is not None:
            return value
    return parsed_assistant.get("score")


def extract_target_reason(row: dict[str, Any], parsed_assistant: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("reference_reason"),
        row.get("Reason"),
        row.get("reason"),
        row.get("predicted_reason"),
        row.get("target_reason"),
        parsed_assistant.get("reason"),
    )


def extract_target_modified_answer(row: dict[str, Any], parsed_assistant: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("reference_modified_answer"),
        row.get("modified_answer"),
        row.get("Modified Answer"),
        row.get("predicted_modified_answer"),
        row.get("target_modified_answer"),
        parsed_assistant.get("modified_answer"),
    )


def extract_gt_answer(row: dict[str, Any], target_modified_answer: str) -> str:
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
        target_modified_answer,
    )


def convert_row(row: dict[str, Any], index: int) -> tuple[Optional[dict[str, Any]], str]:
    system_content, user_content, assistant_content = extract_messages(row)
    if not user_content:
        return None, "missing_user_message"

    sections = parse_labeled_sections(user_content)

    question = first_non_empty(
        row.get("question"),
        row.get("prompt"),
        row.get("instruction"),
        row.get("query"),
        row.get("problem"),
        sections.get("question"),
    )
    answer = first_non_empty(
        row.get("answer"),
        row.get("response"),
        row.get("output"),
        row.get("candidate_answer"),
        row.get("model_answer"),
        sections.get("answer"),
    )
    dimension_name = first_non_empty(
        row.get("dimension_name"),
        row.get("evaluation_dimension"),
        row.get("name"),
        sections.get("dimension_name"),
    )
    criteria_text = normalize_criteria_text(
        first_non_empty(
            row.get("criteria_text"),
            row.get("criteria"),
            row.get("full_score_criteria"),
            sections.get("criteria_text"),
        )
    )

    parsed_assistant = parse_assistant_output(assistant_content)
    target_score = extract_target_score(row, parsed_assistant)
    target_reason = extract_target_reason(row, parsed_assistant)
    target_modified_answer = extract_target_modified_answer(row, parsed_assistant)
    gt_answer = extract_gt_answer(row, target_modified_answer)

    if not question:
        return None, "missing_question"
    if not answer:
        return None, "missing_answer"
    if not dimension_name:
        return None, "missing_dimension"
    if not criteria_text:
        return None, "missing_criteria"
    if target_score is None:
        return None, "missing_target_score"

    output = {
        "sample_id": first_non_empty(row.get("sample_id"), row.get("id"), row.get("unique_id"), f"row:{index}"),
        "messages": [
            {"role": "system", "content": system_content or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "question": question,
        "answer": answer,
        "dimension_name": dimension_name,
        "criteria_text": criteria_text,
        "target_score": target_score,
        "target_reason": target_reason,
        "target_modified_answer": target_modified_answer,
        "gt_answer": gt_answer,
    }
    return output, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ms-swift GRPO JSONL data from message-format rows.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with messages.")
    parser.add_argument("--output", type=Path, required=True, help="Output GRPO JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    stats: dict[str, int] = {}

    with args.output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(iter_jsonl(args.input), start=1):
            if args.limit is not None and written >= args.limit:
                break

            converted, reason = convert_row(row, index)
            stats[reason] = stats.get(reason, 0) + 1

            if converted is None:
                continue

            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    summary = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
    print(f"Wrote {written} rows to {args.output}")
    print(f"Stats: {summary}")


if __name__ == "__main__":
    main()