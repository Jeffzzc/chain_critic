#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计每个 judgment 文件中的：
(win + tie) / (win + tie + lose)

默认输入目录：
evaluation/baseline_new/rewrite_win_rate/judgments/vllm_vllm_rewrite_judge_Qwen3.5-27B

默认输出文件：
evaluation/baseline_new/rewrite_win_rate/judgments/vllm_vllm_rewrite_judge_Qwen3.5-27B_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "datasets/Feedback-Bench/pairwise_judge"
)
DEFAULT_OUTPUT_CSV = Path(
    "datasets/Feedback-Bench/pairwise_judge/summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计每个 judgment 文件中的 win/tie/lose，并输出 CSV 汇总。"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"输入目录，默认: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"输出 CSV 路径，默认: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归遍历子目录；默认只处理当前目录下的文件。",
    )
    return parser.parse_args()


def iter_jsonl_files(input_dir: Path, recursive: bool):
    if recursive:
        yield from sorted(input_dir.rglob("*.jsonl"))
    else:
        yield from sorted(input_dir.glob("*.jsonl"))


def normalize_result(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"win", "tie", "lose"}:
        return text
    return None


def process_file(file_path: Path) -> dict:
    win = 0
    tie = 0
    lose = 0
    invalid_result_count = 0
    json_error_count = 0
    total_lines = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                json_error_count += 1
                continue

            result = normalize_result(record.get("outcome"))

            if result == "win":
                win += 1
            elif result == "tie":
                tie += 1
            elif result == "lose":
                lose += 1
            else:
                invalid_result_count += 1

    denominator = win + tie + lose
    win_tie_rate = (win + tie) / denominator if denominator > 0 else 0.0

    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "total_lines": total_lines,
        "win": win,
        "tie": tie,
        "lose": lose,
        "valid_result_count": denominator,
        "invalid_result_count": invalid_result_count,
        "json_error_count": json_error_count,
        "win_plus_tie": win + tie,
        "win_tie_rate": f"{win_tie_rate:.6f}",
    }


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_csv: Path = args.output_csv

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {input_dir}")

    files = list(iter_jsonl_files(input_dir, args.recursive))
    if not files:
        print(f"未找到任何 jsonl 文件: {input_dir}")
        return

    rows = []
    for file_path in files:
        row = process_file(file_path)
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file_name",
        "file_path",
        "total_lines",
        "win",
        "tie",
        "lose",
        "valid_result_count",
        "invalid_result_count",
        "json_error_count",
        "win_plus_tie",
        "win_tie_rate",
    ]

    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"处理完成，共统计 {len(rows)} 个文件。")
    print(f"结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()