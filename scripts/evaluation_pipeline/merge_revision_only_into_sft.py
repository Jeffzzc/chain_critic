#!/usr/bin/env python
"""Merge revision-only SFT labels into score/reason SFT data.

This script builds final SFT JSONL rows whose assistant label is strict JSON:

{"score": <int>, "reason": "...", "revision_suggestions": "...", "modified_answer": "..."}

It first combines revision-only train/test JSONL files into one revision pool,
then uses that pool to fill revision_suggestions and modified_answer for
final_train_split.jsonl and final_test_split.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = ROOT / "scripts" / "evaluation_pipeline"

sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import COLON_CLASS, iter_jsonl, normalize_text, write_jsonl
from regenerate_sft_revision_only_local import parse_revision_output
from regenerate_sft_score_reason_revision_local import sample_from_record
from score_reason_rewrite_local import parse_model_output


DEFAULT_OUTPUT_DIR = Path("datasets/train/final_score_reason_revision_sft")
DEFAULT_MERGED_REVISION_NAME = "merged_revision_only_sft.jsonl"

SYSTEM_PROMPT = (
    "You are an expert evaluator and answer rewriter. Given a question, a "
    "candidate answer, one evaluation dimension, and complete 0-5 scoring "
    "criteria for that dimension, evaluate the candidate answer and revise it "
    "to better satisfy the score-5 criterion. Use only the provided question, "
    "candidate answer, evaluation dimension, and scoring criteria. Do not add "
    "unsupported facts. Return strict JSON only with exactly these keys: "
    "score, reason, revision_suggestions, modified_answer."
)

USER_TASK_INSTRUCTION = (
    "### Task Description:\n"
    "You are given a question, a candidate answer, one evaluation dimension, "
    "and the complete 0-5 scoring criteria for that dimension.\n"
    "\n"
    "Your task is to evaluate and improve the candidate answer using only the "
    "provided question, candidate answer, evaluation dimension, and scoring "
    "criteria.\n"
    "\n"
    "Follow these steps:\n"
    "1. Assign one integer score from 0 to 5 according to the provided scoring criteria.\n"
    "2. Write a concise, concrete reason for the score. The reason must be grounded "
    "in the evaluation dimension and the 0-5 criteria, and it should identify the "
    "main strengths or errors of the candidate answer.\n"
    "3. Write actionable revision_suggestions. The suggestions should be executable "
    "edit instructions that directly address the errors in the reason and explain "
    "how to move the answer toward the score-5 criterion.\n"
    "4. Rewrite the candidate answer into modified_answer. The modified_answer should "
    "follow the revision_suggestions, answer the original question, optimize only for "
    "the same evaluation dimension, and avoid unsupported facts.\n"
    "5. Return strict JSON only with exactly this schema:\n"
    "{\"score\": <int>, \"reason\": \"...\", \"revision_suggestions\": \"...\", "
    "\"modified_answer\": \"...\"}"
)

USER_TEMPLATE = (
    "{instruction}\n\n"
    "Question:\n{question}\n\n"
    "Candidate Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Score Criteria (0-5):\n{criteria_text}"
)

SCORE_RE = re.compile(rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$")
REASON_RE = re.compile(
    rf"(?is)(?:^|\n)\s*reason\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:revision suggestions|revision_suggestions|modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}\s*(.*)$"
)


def assistant_content(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if normalize_text(message.get("role")).lower() == "assistant":
            contents.append(str(message.get("content") or ""))
    return contents[-1] if contents else ""


def parse_score_reason_label(text: str) -> dict[str, Any]:
    """Parse old score/reason/modified_answer labels.

    Some source rows are plain text with exactly:
    Score: ...
    Reason: ...
    Modified Answer: ...

    The generic JSON-oriented parser treats those as partial JSON-like output,
    so this parser handles the legacy plain-text shape first and then falls
    back to the broader parser.
    """
    raw_text = str(text or "").strip().replace("\r\n", "\n")
    score: int | None = None
    reason = ""
    modified_answer = ""

    score_match = SCORE_RE.search(raw_text)
    if score_match:
        score = int(score_match.group(1))

    reason_match = REASON_RE.search(raw_text)
    if reason_match:
        reason = normalize_text(reason_match.group(1))

    modified_match = MODIFIED_RE.search(raw_text)
    if modified_match:
        modified_answer = normalize_text(modified_match.group(1))

    if score is not None and reason:
        return {
            "score": score,
            "reason": reason,
            "modified_answer": modified_answer,
        }

    parsed = parse_model_output(raw_text)
    return {
        "score": parsed.get("score"),
        "reason": normalize_text(parsed.get("reason")),
        "modified_answer": normalize_text(parsed.get("modified_answer")),
    }


def sample_key(sample: dict[str, Any]) -> str:
    payload = {
        "question": normalize_text(sample.get("question")),
        "answer": normalize_text(sample.get("answer")),
        "dimension_name": normalize_text(sample.get("dimension_name")).lower(),
        "criteria_text": normalize_text(sample.get("criteria_text")),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def sample_loose_key(sample: dict[str, Any]) -> str:
    payload = {
        "question": normalize_text(sample.get("question")),
        "answer": normalize_text(sample.get("answer")),
        "dimension_name": normalize_text(sample.get("dimension_name")).lower(),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def build_messages(sample: dict[str, Any], label: dict[str, Any]) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        instruction=USER_TASK_INSTRUCTION,
        question=sample["question"],
        answer=sample["answer"],
        dimension_name=sample["dimension_name"],
        criteria_text=sample["criteria_text"],
    )
    assistant_payload = {
        "score": int(label["score"]),
        "reason": label["reason"],
        "revision_suggestions": label["revision_suggestions"],
        "modified_answer": label["modified_answer"],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)},
    ]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def write_merged_revision_jsonl(paths: list[Path], output_path: Path) -> dict[str, int]:
    stats = {"input_files": 0, "missing_files": 0, "rows": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for path in paths:
            if not path.is_file():
                stats["missing_files"] += 1
                print(f"[revision-pool] missing input: {path}")
                continue
            stats["input_files"] += 1
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line.rstrip("\n") + "\n")
                    stats["rows"] += 1
    return stats


def load_revision_pool(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
    labels_by_full_key: dict[str, dict[str, str]] = {}
    labels_by_loose_key: dict[str, dict[str, str]] = {}
    ambiguous_loose_keys: set[str] = set()
    stats = {
        "rows": 0,
        "parsed_samples": 0,
        "parsed_labels": 0,
        "duplicate_full_keys": 0,
        "duplicate_loose_keys": 0,
        "bad_samples": 0,
        "bad_labels": 0,
    }
    for index, record in enumerate(iter_jsonl(path)):
        stats["rows"] += 1
        sample = sample_from_record(record, index, "revision")
        if sample is None:
            stats["bad_samples"] += 1
            continue
        stats["parsed_samples"] += 1

        parsed = parse_revision_output(assistant_content(record))
        if not parsed.get("ok"):
            stats["bad_labels"] += 1
            continue
        stats["parsed_labels"] += 1

        label = {
            "revision_suggestions": parsed["revision_suggestions"],
            "modified_answer": parsed["modified_answer"],
        }

        full_key = sample_key(sample)
        if full_key in labels_by_full_key:
            stats["duplicate_full_keys"] += 1
        else:
            labels_by_full_key[full_key] = label

        loose_key = sample_loose_key(sample)
        if loose_key in ambiguous_loose_keys:
            continue
        if loose_key in labels_by_loose_key:
            stats["duplicate_loose_keys"] += 1
            ambiguous_loose_keys.add(loose_key)
            labels_by_loose_key.pop(loose_key, None)
        else:
            labels_by_loose_key[loose_key] = label
    return labels_by_full_key, labels_by_loose_key, stats


def count_key_matches(
    score_reason_input: Path,
    split: str,
    labels_by_full_key: dict[str, dict[str, str]],
    labels_by_loose_key: dict[str, dict[str, str]],
) -> dict[str, int]:
    stats = {"full_key_matches": 0, "loose_key_matches": 0}
    for index, record in enumerate(iter_jsonl(score_reason_input)):
        sample = sample_from_record(record, index, split)
        if sample is None:
            continue
        if sample_key(sample) in labels_by_full_key:
            stats["full_key_matches"] += 1
        elif sample_loose_key(sample) in labels_by_loose_key:
            stats["loose_key_matches"] += 1
    return stats


def merge_split(
    *,
    score_reason_input: Path,
    output_path: Path,
    split: str,
    missing_revision: str,
    labels_by_full_key: dict[str, dict[str, str]],
    labels_by_loose_key: dict[str, dict[str, str]],
    revision_stats: dict[str, int],
) -> dict[str, int]:
    key_match_stats = count_key_matches(score_reason_input, split, labels_by_full_key, labels_by_loose_key)
    rows: list[dict[str, Any]] = []
    stats = {
        "score_reason_rows": 0,
        "output_rows": 0,
        "full_key_matches": key_match_stats["full_key_matches"],
        "loose_key_matches": key_match_stats["loose_key_matches"],
        "bad_score_reason_samples": 0,
        "bad_score_reason_labels": 0,
        "missing_revision": 0,
        "revision_rows": revision_stats["rows"],
        "revision_parsed_labels": revision_stats["parsed_labels"],
        "revision_bad_samples": revision_stats["bad_samples"],
        "revision_bad_labels": revision_stats["bad_labels"],
        "revision_duplicate_full_keys": revision_stats["duplicate_full_keys"],
        "revision_duplicate_loose_keys": revision_stats["duplicate_loose_keys"],
    }

    for index, record in enumerate(iter_jsonl(score_reason_input)):
        stats["score_reason_rows"] += 1
        sample = sample_from_record(record, index, split)
        if sample is None:
            stats["bad_score_reason_samples"] += 1
            continue

        parsed = parse_score_reason_label(assistant_content(record))
        if parsed.get("score") is None or not parsed.get("reason"):
            stats["bad_score_reason_labels"] += 1
            continue

        revision = labels_by_full_key.get(sample_key(sample))
        if revision is None:
            revision = labels_by_loose_key.get(sample_loose_key(sample))

        if revision is None:
            stats["missing_revision"] += 1
            if missing_revision == "error":
                raise ValueError(
                    f"Missing revision-only label for {score_reason_input} line {index + 1}."
                )
            if missing_revision == "keep":
                revision = {
                    "revision_suggestions": parsed.get("revision_suggestions") or "",
                    "modified_answer": parsed.get("modified_answer") or "",
                }
                if not revision["revision_suggestions"] or not revision["modified_answer"]:
                    continue
            else:
                continue

        label = {
            "score": int(parsed["score"]),
            "reason": normalize_text(parsed["reason"]),
            "revision_suggestions": normalize_text(revision["revision_suggestions"]),
            "modified_answer": normalize_text(revision["modified_answer"]),
        }
        rows.append({"messages": build_messages(sample, label)})

    write_jsonl(output_path, rows)
    stats["output_rows"] = len(rows)
    return stats


def print_stats(split: str, output_path: Path, stats: dict[str, int]) -> None:
    print(f"[{split}] wrote {stats['output_rows']} rows -> {output_path}")
    print(
        f"[{split}] full_key_matches={stats['full_key_matches']} "
        f"loose_key_matches={stats['loose_key_matches']} "
        f"missing_revision={stats['missing_revision']}"
    )
    print(
        f"[{split}] score_reason_rows={stats['score_reason_rows']} "
        f"bad_score_reason_samples={stats['bad_score_reason_samples']} "
        f"bad_score_reason_labels={stats['bad_score_reason_labels']}"
    )
    print(
        f"[{split}] revision_rows={stats['revision_rows']} "
        f"revision_parsed_labels={stats['revision_parsed_labels']} "
        f"revision_bad_samples={stats['revision_bad_samples']} "
        f"revision_bad_labels={stats['revision_bad_labels']} "
        f"revision_duplicate_full_keys={stats['revision_duplicate_full_keys']} "
        f"revision_duplicate_loose_keys={stats['revision_duplicate_loose_keys']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build final score+reason+revision_suggestions+modified_answer SFT JSONL "
            "by merging revision-only labels into score/reason SFT rows."
        )
    )
    parser.add_argument(
        "--train-score-reason-input",
        type=Path,
        default=Path("datasets/train/final_train_split.jsonl"),
        help="Score/reason source for the final train split.",
    )
    parser.add_argument(
        "--train-revision-input",
        type=Path,
        default=Path("datasets/train/revision_only_regen/revision_only_sft_train.jsonl"),
        help="Revision-only train source used to build the merged revision pool.",
    )
    parser.add_argument(
        "--test-score-reason-input",
        type=Path,
        default=Path("datasets/train/final_test_split.jsonl"),
        help="Score/reason source for the final test split.",
    )
    parser.add_argument(
        "--test-revision-input",
        type=Path,
        default=Path("datasets/train/revision_only_regen/revision_only_sft_test.jsonl"),
        help="Revision-only test source used to build the merged revision pool.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--merged-revision-output-name",
        default=DEFAULT_MERGED_REVISION_NAME,
        help="File name for the concatenated revision-only train+test JSONL.",
    )
    parser.add_argument(
        "--train-output-name",
        default="final_score_reason_revision_sft_train.jsonl",
    )
    parser.add_argument(
        "--test-output-name",
        default="final_score_reason_revision_sft_test.jsonl",
    )
    parser.add_argument(
        "--missing-revision",
        choices=("skip", "error", "keep"),
        default="skip",
        help=(
            "How to handle score/reason rows that do not have a matching revision-only row. "
            "'skip' keeps only fully merged rows."
        ),
    )
    parser.add_argument(
        "--require-train",
        action="store_true",
        help="Fail if the train input pair is not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_revision_input = resolve_path(args.train_revision_input)
    test_revision_input = resolve_path(args.test_revision_input)
    merged_revision_output = args.output_dir / args.merged_revision_output_name
    merge_stats = write_merged_revision_jsonl(
        [train_revision_input, test_revision_input],
        merged_revision_output,
    )
    print(
        f"[revision-pool] wrote {merge_stats['rows']} rows from "
        f"{merge_stats['input_files']} files -> {merged_revision_output}"
    )
    if merge_stats["rows"] == 0:
        raise FileNotFoundError("No revision-only rows were loaded; check --train-revision-input and --test-revision-input.")

    labels_by_full_key, labels_by_loose_key, revision_stats = load_revision_pool(merged_revision_output)
    print(
        f"[revision-pool] parsed_labels={revision_stats['parsed_labels']} "
        f"full_key_index={len(labels_by_full_key)} loose_key_index={len(labels_by_loose_key)}"
    )

    jobs = [
        (
            "train",
            resolve_path(args.train_score_reason_input),
            args.output_dir / args.train_output_name,
        ),
        (
            "test",
            resolve_path(args.test_score_reason_input),
            args.output_dir / args.test_output_name,
        ),
    ]

    for split, score_reason_input, output_path in jobs:
        if not score_reason_input.is_file():
            message = (
                f"[{split}] skip: missing score/reason input "
                f"score_reason={score_reason_input}"
            )
            if split == "train" and args.require_train:
                raise FileNotFoundError(message)
            print(message)
            continue
        stats = merge_split(
            score_reason_input=score_reason_input,
            output_path=output_path,
            split=split,
            missing_revision=args.missing_revision,
            labels_by_full_key=labels_by_full_key,
            labels_by_loose_key=labels_by_loose_key,
            revision_stats=revision_stats,
        )
        print_stats(split, output_path, stats)


if __name__ == "__main__":
    main()
