#!/usr/bin/env python3
"""Split a pairwise DPO JSONL dataset into train/val splits.

The input is expected to contain rows like:
  {
    "prompt": "...",
    "chosen": "...",
    "rejected": "...",
    "meta": { ... }
  }

This script keeps the split deterministic and roughly stratified by score
distance when that information exists in `meta`.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any, DefaultDict, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_number} in {path} is not a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_key(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    for key in ("score_diff", "score_abs_diff", "chosen_score", "rejected_score"):
        value = meta.get(key)
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            return f"diff_{int(math.floor(number))}"
    sample_id = meta.get("sample_id")
    if sample_id is not None:
        return f"sid_{sample_id}"
    return "default"


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: DefaultDict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[split_key(row)].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for _, bucket in sorted(grouped.items(), key=lambda item: item[0]):
        rng.shuffle(bucket)
        if len(bucket) <= 1:
            train_rows.extend(bucket)
            continue
        val_size = max(1, int(round(len(bucket) * val_ratio)))
        val_size = min(val_size, len(bucket) - 1)
        val_rows.extend(bucket[:val_size])
        train_rows.extend(bucket[val_size:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: DefaultDict[str, int] = collections.defaultdict(int)
    for row in rows:
        counts[split_key(row)] += 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split DPO JSONL into train/val")
    parser.add_argument("--input", type=Path, required=True, help="Input DPO JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    summary_path = args.output_dir / "split_summary.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_buckets": summarize(train_rows),
        "val_buckets": summarize(val_rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()