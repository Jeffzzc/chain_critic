#!/usr/bin/env python
"""Filter revision_suggestions similarity JSONL files.

Actions:
1) For the chaincritic v17 file only, remove sentences containing
   "To achieve a score of 5," from predicted_revision_suggestions.
2) For all files, drop rows where reference_score == 5.

Outputs are written to a new directory; original files are not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Optional

from pipeline_common import parse_int_score

PHRASE = "To achieve a score of 5,"
SENTENCE_WITH_PHRASE = re.compile(r"[^.!?]*To achieve a score of 5,[^.!?]*[.!?]?", re.DOTALL)


def normalize_after_removal(text: str) -> str:
    cleaned = text.replace("\n", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def remove_sentence_with_phrase(text: str) -> str:
    if PHRASE not in text:
        return text
    cleaned = SENTENCE_WITH_PHRASE.sub("", text)
    return normalize_after_removal(cleaned)


def parse_score(value: Any) -> Optional[int]:
    return parse_int_score(value)


def should_drop_row(row: dict[str, Any]) -> bool:
    score = parse_score(row.get("reference_score"))
    return score == 5


def process_file(path: Path, output_path: Path, *, apply_phrase_removal: bool) -> dict[str, int]:
    total = 0
    kept = 0
    dropped = 0
    modified = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)

            if should_drop_row(row):
                dropped += 1
                continue

            if apply_phrase_removal:
                pred = row.get("predicted_revision_suggestions")
                if isinstance(pred, str) and PHRASE in pred:
                    new_pred = remove_sentence_with_phrase(pred)
                    if new_pred != pred:
                        row["predicted_revision_suggestions"] = new_pred
                        modified += 1

            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    return {"total": total, "kept": kept, "dropped": dropped, "modified": modified}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter revision_suggestions similarity JSONL files without modifying originals."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("evaluation/revision_suggestions_relevance/similarities"),
        help="Directory containing similarity JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/revision_suggestions_relevance/similarities_filtered"),
        help="Directory to write filtered JSONL files.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jsonl",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--chaincritic-file",
        type=str,
        default="chaincritic-7B-v17-17785_final_test_outputs.jsonl",
        help="Filename that receives the phrase-removal step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob} in {input_dir}")

    for path in files:
        output_path = output_dir / path.name
        stats = process_file(
            path,
            output_path,
            apply_phrase_removal=path.name == args.chaincritic_file,
        )
        print(
            f"{path.name}\t"
            f"total={stats['total']}\t"
            f"kept={stats['kept']}\t"
            f"dropped_score5={stats['dropped']}\t"
            f"modified_phrase={stats['modified']}"
        )


if __name__ == "__main__":
    main()
