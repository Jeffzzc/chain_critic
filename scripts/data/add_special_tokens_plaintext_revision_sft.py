#!/usr/bin/env python3
"""Add special tokens to assistant content in plaintext revision SFT data.

Reads plaintext revision SFT JSONL files and rewrites assistant content to:
<s>score</s><r>reason</r><rs>revision suggetion</rs><ra>refined answer</ra>
with values inside each tag, for example:
<s>1</s><r>...</r><rs>...</rs><ra>...</ra>
Original data is left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = ROOT / "scripts" / "evaluation_pipeline"
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import iter_jsonl  # noqa: E402


DEFAULT_INPUT_DIR = Path("datasets/train/final_score_reason_plaintext_revision_sft")
DEFAULT_OUTPUT_DIR = Path("datasets/train/final_score_reason_plaintext_revision_sft_tagged")
DEFAULT_TRAIN_NAME = "final_score_reason_plaintext_train.jsonl"
DEFAULT_TEST_NAME = "final_score_reason_plaintext_test.jsonl"
DEFAULT_OUTPUT_TRAIN_NAME = "final_score_reason_plaintext_train.jsonl"
DEFAULT_OUTPUT_TEST_NAME = "final_score_reason_plaintext_test.jsonl"

TAG_SCORE_OPEN = "<s>"
TAG_SCORE_CLOSE = "</s>"
TAG_REASON_OPEN = "<r>"
TAG_REASON_CLOSE = "</r>"
TAG_REVISION_OPEN = "<rs>"
TAG_REVISION_CLOSE = "</rs>"
TAG_REFINED_OPEN = "<ra>"
TAG_REFINED_CLOSE = "</ra>"

PREFIX_SCORE = "score:"
PREFIX_REASON = "reason:"
PREFIX_REVISION = "revision suggestions:"
PREFIX_MODIFIED = "modified answer:"


def parse_plaintext_fields(content: str) -> Optional[Dict[str, str]]:
    if not content:
        return None
    score = ""
    reason = ""
    revision = ""
    modified = ""
    for raw_line in str(content).replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith(PREFIX_SCORE):
            score = line[len(PREFIX_SCORE) :].strip()
            continue
        if lower.startswith(PREFIX_REASON):
            reason = line[len(PREFIX_REASON) :].strip()
            continue
        if lower.startswith(PREFIX_REVISION):
            revision = line[len(PREFIX_REVISION) :].strip()
            continue
        if lower.startswith(PREFIX_MODIFIED):
            modified = line[len(PREFIX_MODIFIED) :].strip()
            continue
    if not score and not reason and not revision and not modified:
        return None
    return {
        "score": score,
        "reason": reason,
        "revision": revision,
        "modified": modified,
    }


def build_tagged_content(fields: Dict[str, str]) -> str:
    return (
        f"{TAG_SCORE_OPEN}{fields.get('score', '')}{TAG_SCORE_CLOSE}"
        f"{TAG_REASON_OPEN}{fields.get('reason', '')}{TAG_REASON_CLOSE}"
        f"{TAG_REVISION_OPEN}{fields.get('revision', '')}{TAG_REVISION_CLOSE}"
        f"{TAG_REFINED_OPEN}{fields.get('modified', '')}{TAG_REFINED_CLOSE}"
    )


def convert_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    updated = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            updated.append(message)
            continue
        if str(message.get("role") or "").lower() != "assistant":
            updated.append(message)
            continue
        content = str(message.get("content") or "")
        fields = parse_plaintext_fields(content)
        if fields is None:
            updated.append(message)
            continue
        new_message = dict(message)
        new_message["content"] = build_tagged_content(fields)
        updated.append(new_message)
        changed = True
    if not changed:
        return None
    new_record = dict(record)
    new_record["messages"] = updated
    return new_record


def convert_file(input_path: Path, output_path: Path) -> Dict[str, int]:
    stats = {"rows": 0, "written": 0, "skipped": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            stats["rows"] += 1
            converted = convert_record(record)
            if converted is None:
                stats["skipped"] += 1
                continue
            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["written"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add special tokens to assistant content for plaintext revision SFT JSONL."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-name", default=DEFAULT_TRAIN_NAME)
    parser.add_argument("--test-name", default=DEFAULT_TEST_NAME)
    parser.add_argument("--output-train-name", default=DEFAULT_OUTPUT_TRAIN_NAME)
    parser.add_argument("--output-test-name", default=DEFAULT_OUTPUT_TEST_NAME)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    train_input = input_dir / args.train_name
    test_input = input_dir / args.test_name

    train_output = output_dir / args.output_train_name
    test_output = output_dir / args.output_test_name

    if not train_input.is_file():
        print(f"Missing train input: {train_input}")
        return 1
    if not test_input.is_file():
        print(f"Missing test input: {test_input}")
        return 1

    train_stats = convert_file(train_input, train_output)
    test_stats = convert_file(test_input, test_output)

    print("Conversion complete.")
    print(f"Train: {train_stats}")
    print(f"Test: {test_stats}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
