#!/usr/bin/env python
"""Compute average character length for input text built as question+answer.

Default behavior scans:
  datasets/*/filter/*.json

The script prints per-file stats and optional JSON output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_GLOBS = ("datasets/*/filter/*.json",)
DEFAULT_SKIP_NAMES = {"filter_summary.json"}


def _records_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "samples", "records", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _iter_files(repo_root: Path, files: list[str], globs: list[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    for raw_path in files:
        file_path = Path(raw_path)
        if not file_path.is_absolute():
            file_path = repo_root / file_path
        resolved = str(file_path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(file_path)

    for pattern in globs:
        for file_path in sorted(repo_root.glob(pattern)):
            resolved = str(file_path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(file_path)
    return candidates


def _compute_one(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "file": str(path),
        "status": "ok",
        "total_items": 0,
        "used_items": 0,
        "skipped_items": 0,
        "avg_question_chars": 0.0,
        "avg_answer_chars": 0.0,
        "avg_input_chars": 0.0,
        "total_question_chars": 0,
        "total_answer_chars": 0,
        "total_input_chars": 0,
    }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        stats["status"] = f"error: {exc}"
        return stats

    records = _records_from_payload(payload)
    stats["total_items"] = len(records)

    for row in records:
        if not isinstance(row, dict):
            stats["skipped_items"] += 1
            continue

        if "question" not in row or "answer" not in row:
            stats["skipped_items"] += 1
            continue

        question = _to_text(row.get("question"))
        answer = _to_text(row.get("answer"))
        q_chars = len(question)
        a_chars = len(answer)

        stats["used_items"] += 1
        stats["total_question_chars"] += q_chars
        stats["total_answer_chars"] += a_chars
        stats["total_input_chars"] += q_chars + a_chars

    used = stats["used_items"]
    if used > 0:
        stats["avg_question_chars"] = stats["total_question_chars"] / used
        stats["avg_answer_chars"] = stats["total_answer_chars"] / used
        stats["avg_input_chars"] = stats["total_input_chars"] / used

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average character counts for question+answer inputs."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Specific JSON file paths to process.",
    )
    parser.add_argument(
        "--globs",
        nargs="*",
        default=[],
        help="Glob patterns (repo-root relative) used when --files is not enough.",
    )
    parser.add_argument(
        "--skip-names",
        nargs="*",
        default=sorted(DEFAULT_SKIP_NAMES),
        help="Basename list to skip.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    globs = args.globs if args.globs else ([] if args.files else list(DEFAULT_GLOBS))
    candidates = _iter_files(repo_root, args.files, globs)
    skip_names = set(args.skip_names)

    files = [p for p in candidates if p.is_file() and p.name not in skip_names]
    if not files:
        print("No files matched.")
        return

    results = [_compute_one(path) for path in files]

    overall_used = sum(item["used_items"] for item in results)
    overall_total_input = sum(item["total_input_chars"] for item in results)
    overall_avg = (overall_total_input / overall_used) if overall_used else 0.0

    print(
        "file\tused_items\ttotal_items\tavg_input_chars\tavg_question_chars\tavg_answer_chars\tstatus"
    )
    for item in results:
        print(
            f"{item['file']}\t{item['used_items']}\t{item['total_items']}\t"
            f"{item['avg_input_chars']:.2f}\t{item['avg_question_chars']:.2f}\t"
            f"{item['avg_answer_chars']:.2f}\t{item['status']}"
        )

    print(
        f"OVERALL\t{overall_used}\t-\t{overall_avg:.2f}\t-\t-\t"
        f"(weighted by used_items)"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = repo_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "overall_used_items": overall_used,
            "overall_avg_input_chars": overall_avg,
            "results": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON summary: {out_path}")


if __name__ == "__main__":
    main()
