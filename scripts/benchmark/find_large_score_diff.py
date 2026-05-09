#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
找出 score_generate_chaincritic_v11-70669.jsonl 中 predicted_score
与 train.jsonl 中 orig_score 分差超过 2 分的样本，并做简单合并输出。

默认输入：
- datasets/Feedback-Bench/data/score_generate_chaincritic_v11-70669.jsonl
- datasets/Feedback-Bench/data/train.jsonl

默认输出：
- datasets/Feedback-Bench/data/score_diff_gt2_merged.jsonl
- datasets/Feedback-Bench/data/score_diff_gt2_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {path} 第 {line_no} 行: {e}") from e
    return data


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").strip()


def build_train_index(train_data: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    用 train.jsonl 中的三元组构建索引：
    (orig_instruction, orig_response, orig_criteria)
    """
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for row in train_data:
        key = (
            normalize_text(row.get("orig_instruction")),
            normalize_text(row.get("orig_response")),
            normalize_text(row.get("orig_criteria")),
        )

        if key in index:
            raise ValueError("train.jsonl 中发现重复匹配键，无法唯一对齐，请检查数据。")

        index[key] = row

    return index


def safe_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"字段 {field_name} 无法转换为 int，原值={value!r}") from e


def merge_and_filter(
    pred_data: Iterable[Dict[str, Any]],
    train_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    threshold: int,
) -> List[Dict[str, Any]]:
    """
    从预测结果里找出 |predicted_score - orig_score| > threshold 的数据，并合并信息。
    """
    results: List[Dict[str, Any]] = []
    unmatched_count = 0

    for row in pred_data:
        key = (
            normalize_text(row.get("question")),
            normalize_text(row.get("answer")),
            normalize_text(row.get("dimension_name")),
        )

        train_row = train_index.get(key)
        if train_row is None:
            unmatched_count += 1
            continue

        predicted_score = safe_int(row.get("predicted_score"), "predicted_score")
        orig_score = safe_int(train_row.get("orig_score"), "orig_score")
        score_diff = abs(predicted_score - orig_score)

        if score_diff > threshold:
            merged = {
                # 核心匹配信息
                "sample_id": row.get("sample_id"),
                "score_diff": score_diff,
                "predicted_score": predicted_score,
                "orig_score": orig_score,

                # 预测文件里的信息
                "question": row.get("question"),
                "answer": row.get("answer"),
                "dimension_name": row.get("dimension_name"),
                "score_criteria": row.get("score_criteria"),
                "predicted_reason": row.get("predicted_reason"),
                "predicted_modified_answer": row.get("predicted_modified_answer"),
                "ok": row.get("ok"),
                "parse_error": row.get("parse_error"),

                # train 文件里的信息
                "orig_instruction": train_row.get("orig_instruction"),
                "orig_response": train_row.get("orig_response"),
                "orig_reference_answer": train_row.get("orig_reference_answer"),
                "orig_criteria": train_row.get("orig_criteria"),
                "orig_feedback": train_row.get("orig_feedback"),
                "orig_score1_description": train_row.get("orig_score1_description"),
                "orig_score2_description": train_row.get("orig_score2_description"),
                "orig_score3_description": train_row.get("orig_score3_description"),
                "orig_score4_description": train_row.get("orig_score4_description"),
                "orig_score5_description": train_row.get("orig_score5_description"),
                "train_index": train_row.get("__index_level_0__"),
            }
            results.append(merged)

    print(f"未匹配上的预测样本数: {unmatched_count}")
    return results


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    写一份便于快速查看的 CSV。
    只保留比较核心、适合表格查看的字段。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "score_diff",
        "predicted_score",
        "orig_score",
        "dimension_name",
        "question",
        "answer",
        "predicted_reason",
        "orig_feedback",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选 predicted_score 与 orig_score 分差超过阈值的数据，并做简单合并。")
    parser.add_argument(
        "--pred-file",
        type=Path,
        default=Path("datasets/Feedback-Bench/data/score_generate_chaincritic_v11-70669.jsonl"),
        help="预测结果 JSONL 路径",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("datasets/Feedback-Bench/data/train.jsonl"),
        help="训练集 JSONL 路径",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="分差阈值，筛选 abs(predicted_score - orig_score) > threshold 的样本",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("datasets/Feedback-Bench/data/score_diff_gt2_merged.jsonl"),
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("datasets/Feedback-Bench/data/score_diff_gt2_merged.csv"),
        help="输出 CSV 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"读取预测文件: {args.pred_file}")
    pred_data = load_jsonl(args.pred_file)

    print(f"读取训练文件: {args.train_file}")
    train_data = load_jsonl(args.train_file)

    print("构建 train 索引...")
    train_index = build_train_index(train_data)

    print(f"开始筛选 |predicted_score - orig_score| > {args.threshold} 的样本...")
    merged_rows = merge_and_filter(pred_data, train_index, args.threshold)

    # 按分差从大到小排序，便于查看
    merged_rows.sort(key=lambda x: (-x["score_diff"], x["predicted_score"], x["orig_score"]))

    print(f"命中样本数: {len(merged_rows)}")

    print(f"写出 JSONL: {args.out_jsonl}")
    write_jsonl(args.out_jsonl, merged_rows)

    print(f"写出 CSV: {args.out_csv}")
    write_csv(args.out_csv, merged_rows)

    print("处理完成。")


if __name__ == "__main__":
    main()