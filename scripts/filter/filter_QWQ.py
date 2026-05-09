#!/usr/bin/env python
"""Filter correct score items for QwQ-LongCoT-130K.

This script matches score rows by `question`, extracts final answers from both
reference and prediction `answer` fields, and keeps only correct rows.

CHANGE vs original:
- Per-score output file now preserves the **same top-level format as input**:
  it writes a JSON array (list of original rows), NOT a dict with stats.
- Each output row is kept **unchanged** (no injected debug keys).
- Stats are kept in the summary JSON only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from filter_NuminaMath import (
    build_output_name,
    filter_one_file,
    iter_score_files,
    load_reference_answers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter correct rows in QwQ-LongCoT-130K score files."
    )
    parser.add_argument(
        "--reference-jsonl",
        default="datasets/QwQ-LongCoT-130K/train_random_sample.jsonl",
        help="Reference JSONL containing ground-truth answers.",
    )
    parser.add_argument(
        "--score-dir",
        default="datasets/QwQ-LongCoT-130K/score",
        help="Directory containing score JSON files.",
    )
    parser.add_argument(
        "--score-pattern",
        default="*.json",
        help="Glob pattern for score files.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/QwQ-LongCoT-130K/filter",
        help="Directory to write filtered files.",
    )
    parser.add_argument(
        "--summary-file",
        default="filter_summary.json",
        help="Summary filename under output-dir.",
    )
    args = parser.parse_args()

    reference_jsonl = Path(args.reference_jsonl)
    score_dir = Path(args.score_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_answers, reference_stats = load_reference_answers(reference_jsonl)
    score_files = iter_score_files(score_dir, args.score_pattern)

    file_summaries: list[dict[str, Any]] = []
    total_correct = 0

    for score_file in score_files:
        result = filter_one_file(score_file, reference_answers)
        out_file = output_dir / build_output_name(score_file)

        # 关键：逐文件输出保持与输入一致：顶层是 JSON 数组；元素为原始 row dict（不注入任何字段）
        per_file_payload = result["correct_items"] if result.get("status") == "ok" else []
        out_file.write_text(
            json.dumps(per_file_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        total_correct += int(result.get("correct_count", 0))
        file_summaries.append(
            {
                "source_file": result.get("source_file", score_file.name),
                "status": result.get("status", "unknown"),
                "total_items": result.get("total_items", 0),
                "matched_questions": result.get("matched_questions", 0),
                "extract_failures": result.get("extract_failures", 0),
                "correct_count": result.get("correct_count", 0),
                "output_file": str(out_file),
            }
        )

    summary = {
        "reference_jsonl": str(reference_jsonl),
        "score_dir": str(score_dir),
        "score_pattern": args.score_pattern,
        "output_dir": str(output_dir),
        "reference_stats": reference_stats,
        "reference_usable_questions": len(reference_answers),
        "processed_files": len(score_files),
        "total_correct_items": total_correct,
        "files": file_summaries,
    }
    summary_path = output_dir / args.summary_file
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Processed files: {len(score_files)}")
    print(f"Reference usable questions: {len(reference_answers)}")
    print(f"Total correct items: {total_correct}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
