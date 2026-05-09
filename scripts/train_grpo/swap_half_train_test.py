#!/usr/bin/env python3
"""Swap half of final_test_split with an equal amount from final_train_split.

Default behavior:
- Read from datasets/train/final_test_split.jsonl
- Read from datasets/train/final_train_split.jsonl
- Save swapped outputs to datasets/train_grpo/

Usage:
    python scripts/train/swap_half_train_test.py
    python scripts/train/swap_half_train_test.py --seed 123
    python scripts/train/swap_half_train_test.py \
        --test-input datasets/0-5/final_test_split.jsonl \
        --train-input datasets/0-5/final_train_split.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_jsonl_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line:
                lines.append(line)
    return lines


def write_jsonl_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def swap_samples(
    test_lines: list[str], train_lines: list[str], swap_count: int, seed: int
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)

    test_indices = rng.sample(range(len(test_lines)), swap_count)
    train_indices = rng.sample(range(len(train_lines)), swap_count)

    swapped_test = test_lines.copy()
    swapped_train = train_lines.copy()

    for test_idx, train_idx in zip(test_indices, train_indices):
        swapped_test[test_idx], swapped_train[train_idx] = (
            swapped_train[train_idx],
            swapped_test[test_idx],
        )

    return swapped_test, swapped_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Swap half of test set samples with an equal number of train set samples."
        )
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path("datasets/train/final_test_split.jsonl"),
        help="Path to final_test_split.jsonl",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("datasets/train/final_train_split.jsonl"),
        help="Path to final_train_split.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/train_grpo"),
        help="Directory to save swapped output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible swapping",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    test_lines = read_jsonl_lines(args.test_input)
    train_lines = read_jsonl_lines(args.train_input)

    if not test_lines:
        raise ValueError(f"Test file is empty: {args.test_input}")

    swap_count = len(test_lines) // 2
    if swap_count == 0:
        raise ValueError("Test set is too small to swap half of it.")

    if len(train_lines) < swap_count:
        raise ValueError(
            "Train set does not have enough samples for swap: "
            f"need {swap_count}, got {len(train_lines)}"
        )

    swapped_test, swapped_train = swap_samples(
        test_lines=test_lines,
        train_lines=train_lines,
        swap_count=swap_count,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_test = args.output_dir / "final_test_split.jsonl"
    out_train = args.output_dir / "final_train_split.jsonl"

    write_jsonl_lines(out_test, swapped_test)
    write_jsonl_lines(out_train, swapped_train)

    print("Swap finished.")
    print(f"Test input : {args.test_input}")
    print(f"Train input: {args.train_input}")
    print(f"Swap count : {swap_count}")
    print(f"Output test: {out_test}")
    print(f"Output train: {out_train}")


if __name__ == "__main__":
    main()
