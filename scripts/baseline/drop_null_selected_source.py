#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {path}, line={line_no}, error={e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def keep_row(row: dict[str, Any]) -> bool:
    # 只删除 selected_source 为 None/null 的行
    return row.get("selected_source") is not None


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_drop_null_selected_source{input_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="删除 JSONL 中 selected_source 为 null 的样本"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 JSONL 文件路径；不传则自动生成新文件名",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="直接覆盖原文件",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    rows = read_jsonl(args.input)
    kept_rows = [row for row in rows if keep_row(row)]
    removed = len(rows) - len(kept_rows)

    if args.inplace:
        output_path = args.input
    else:
        output_path = args.output or default_output_path(args.input)

    write_jsonl(output_path, kept_rows)

    print(f"[Input] {args.input}")
    print(f"[Output] {output_path}")
    print(f"[Total] {len(rows)}")
    print(f"[Removed selected_source=null] {removed}")
    print(f"[Kept] {len(kept_rows)}")


if __name__ == "__main__":
    main()