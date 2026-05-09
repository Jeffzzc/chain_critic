#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert existing judge-style training data into a new SFT dataset for the task:
Input: Question + Answer
Output: Evaluation_dimension + full 0-5 scoring criteria

Supported source format (one JSON object per line):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...Question: ... Answer: ... Evaluation_dimension: ... Criteria (0-5): ..."},
    {"role": "assistant", "content": "..."}
  ]
}

The script extracts the fields from the user message and writes either:
1) messages format (recommended)
2) query/response format

Examples:
python convert_qa_to_rubric_sft.py --input data.jsonl --output train_rubric_messages.jsonl --output-format messages
python convert_qa_to_rubric_sft.py --input data.jsonl --output train_rubric_qr.jsonl --output-format query_response
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert evaluation rubric writer. "
    "Given a question and an answer, generate exactly one evaluation dimension and its complete 0-5 scoring criteria. "
    "The rubric must be specific to the answer quality being judged, internally consistent, and written in plain text. "
    "Output exactly in this format:\n"
    "Evaluation_dimension: <dimension name>\n"
    "Criteria (0-5):\n"
    "0: <criterion>\n"
    "1: <criterion>\n"
    "2: <criterion>\n"
    "3: <criterion>\n"
    "4: <criterion>\n"
    "5: <criterion>\n"
    "Do not output anything else."
)

DEFAULT_USER_TEMPLATE = (
    "Task: Write one evaluation dimension and its complete 0-5 scoring criteria for assessing the given answer.\n\n"
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n"
)


SECTION_PATTERNS = {
    "question": [r"Question\s*:\s*", r"QUESTION\s*:\s*"],
    "answer": [r"Answer\s*:\s*", r"ANSWER\s*:\s*"],
    "evaluation_dimension": [
        r"Evaluation_dimension\s*:\s*",
        r"Evaluation Dimension\s*:\s*",
        r"EVALUATION_DIMENSION\s*:\s*",
    ],
    "criteria": [
        r"Criteria\s*\(\s*0\s*-\s*5\s*\)\s*:\s*",
        r"Criteria\s*:\s*",
        r"CRITERIA\s*\(\s*0\s*-\s*5\s*\)\s*:\s*",
    ],
}


def build_header_regex(patterns: List[str]) -> str:
    return "(?:" + "|".join(patterns) + ")"


QUESTION_RE = build_header_regex(SECTION_PATTERNS["question"])
ANSWER_RE = build_header_regex(SECTION_PATTERNS["answer"])
DIM_RE = build_header_regex(SECTION_PATTERNS["evaluation_dimension"])
CRITERIA_RE = build_header_regex(SECTION_PATTERNS["criteria"])

FULL_PARSE_RE = re.compile(
    rf"{QUESTION_RE}(?P<question>.*?)"
    rf"{ANSWER_RE}(?P<answer>.*?)"
    rf"{DIM_RE}(?P<dimension>.*?)"
    rf"{CRITERIA_RE}(?P<criteria>.*)$",
    flags=re.DOTALL,
)

FALLBACK_QA_RE = re.compile(
    rf"{QUESTION_RE}(?P<question>.*?)"
    rf"{ANSWER_RE}(?P<answer>.*)$",
    flags=re.DOTALL,
)

CRITERION_LINE_RE = re.compile(r"^\s*([0-5])\s*:\s*(.*?)\s*$")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return text



def extract_user_content(obj: Dict[str, Any]) -> str:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Missing or invalid 'messages' field")

    user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
    if not user_contents:
        raise ValueError("No user message found")

    # Most samples contain a single task-style user prompt. If there are multiple,
    # prefer the last one because it typically contains the actual sample content.
    return normalize_text(str(user_contents[-1]))



def parse_task_text(text: str) -> Tuple[str, str, str, str]:
    text = normalize_text(text)
    match = FULL_PARSE_RE.search(text)
    if not match:
        raise ValueError("Failed to parse Question/Answer/Evaluation_dimension/Criteria from user content")

    question = normalize_text(match.group("question"))
    answer = normalize_text(match.group("answer"))
    dimension = normalize_text(match.group("dimension"))
    criteria_block = normalize_criteria_block(match.group("criteria"))

    if not question or not answer or not dimension or not criteria_block:
        raise ValueError("Parsed empty required field")

    return question, answer, dimension, criteria_block



def normalize_criteria_block(criteria: str) -> str:
    criteria = normalize_text(criteria)
    lines = [line.strip() for line in criteria.split("\n") if line.strip()]

    collected: Dict[str, str] = {}
    current_key: Optional[str] = None

    for raw_line in lines:
        m = CRITERION_LINE_RE.match(raw_line)
        if m:
            current_key = m.group(1)
            collected[current_key] = m.group(2).strip()
        elif current_key is not None:
            collected[current_key] = (collected[current_key] + " " + raw_line.strip()).strip()

    if all(str(i) in collected for i in range(6)):
        return "\n".join(f"{i}: {collected[str(i)]}" for i in range(6))

    # If structured 0-5 lines were not fully recognized, keep the raw block.
    return criteria



def build_input_text(question: str, answer: str, user_template: str) -> str:
    return user_template.format(question=question, answer=answer).strip()



def build_output_text(dimension: str, criteria_block: str) -> str:
    return f"Evaluation_dimension: {dimension}\nCriteria (0-5):\n{criteria_block}".strip()



def convert_record(
    obj: Dict[str, Any],
    output_format: str,
    system_prompt: str,
    user_template: str,
    keep_extra_fields: bool,
) -> Dict[str, Any]:
    user_content = extract_user_content(obj)
    question, answer, dimension, criteria_block = parse_task_text(user_content)

    input_text = build_input_text(question, answer, user_template)
    output_text = build_output_text(dimension, criteria_block)

    if output_format == "messages":
        new_obj: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": output_text},
            ]
        }
    elif output_format == "query_response":
        new_obj = {
            "system": system_prompt,
            "query": input_text,
            "response": output_text,
        }
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    if keep_extra_fields:
        for key, value in obj.items():
            if key not in new_obj:
                new_obj[f"source_{key}"] = value

    return new_obj



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_idx}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_idx} is not a JSON object")
            rows.append(obj)
    return rows



def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="Convert judge-style JSONL into rubric-generation SFT data")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--output-format",
        choices=["messages", "query_response"],
        default="messages",
        help="Target dataset format for ms-swift",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used in output dataset",
    )
    parser.add_argument(
        "--user-template",
        default=DEFAULT_USER_TEMPLATE,
        help="User template. Must contain {question} and {answer}",
    )
    parser.add_argument(
        "--keep-extra-fields",
        action="store_true",
        help="Preserve original fields with 'source_' prefix for debugging",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)
    converted_rows: List[Dict[str, Any]] = []
    failed = 0

    for idx, obj in enumerate(rows, start=1):
        try:
            converted = convert_record(
                obj=obj,
                output_format=args.output_format,
                system_prompt=args.system_prompt,
                user_template=args.user_template,
                keep_extra_fields=args.keep_extra_fields,
            )
            converted_rows.append(converted)
        except Exception as e:
            failed += 1
            print(f"[WARN] Skip line {idx}: {e}")

    save_jsonl(output_path, converted_rows)

    print(f"[Done] Input rows: {len(rows)}")
    print(f"[Done] Output rows: {len(converted_rows)}")
    print(f"[Done] Failed rows: {failed}")
    print(f"[Done] Saved to: {output_path}")


if __name__ == "__main__":
    main()
