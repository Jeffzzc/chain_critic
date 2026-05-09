#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make swift infer jsonl results more readable.

Input:
    /data/dhf/chain_critic/output/v1-20260307-231012/checkpoint-27165_test100_pred.jsonl

Output:
    A markdown report with extracted key fields for each sample.

Usage:
    python make_readable_report.py \
      --input /data/dhf/chain_critic/output/v1-20260307-231012/checkpoint-27165_test100_pred.jsonl \
      --output /data/dhf/chain_critic/output/v1-20260307-231012/checkpoint-27165_test100_pred_readable.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def safe_strip(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def extract_user_content(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return safe_strip(msg.get("content", ""))
    return ""


def extract_block(text: str, start_label: str, end_label: Optional[str] = None) -> str:
    text = safe_strip(text)
    if not text:
        return ""

    start_idx = text.find(start_label)
    if start_idx == -1:
        return ""

    start_idx += len(start_label)

    if end_label is None:
        return text[start_idx:].strip()

    end_idx = text.find(end_label, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()

    return text[start_idx:end_idx].strip()


def parse_prediction_text(text: str) -> Dict[str, str]:
    """
    Parse plain text in this format:
    Score: ...
    Reason: ...
    Modified Answer: ...
    """
    text = safe_strip(text)
    if not text:
        return {"score": "", "reason": "", "modified_answer": ""}

    score = extract_block(text, "Score:", "Reason:")
    reason = extract_block(text, "Reason:", "Modified Answer:")
    modified_answer = extract_block(text, "Modified Answer:", None)

    return {
        "score": score,
        "reason": reason,
        "modified_answer": modified_answer,
    }


def parse_prompt_fields(user_content: str) -> Dict[str, str]:
    """
    Extract:
    - Question
    - Answer
    - Evaluation_dimension
    - Criteria
    from the original user prompt.
    """
    question = extract_block(user_content, "Question:\n", "\n\nAnswer:")
    answer = extract_block(user_content, "Answer:\n", "\n\nEvaluation_dimension:")
    dimension = extract_block(user_content, "Evaluation_dimension:\n", "\n\nCriteria:")
    criteria = extract_block(user_content, "Criteria:\n", None)

    return {
        "question": question,
        "answer": answer,
        "dimension": dimension,
        "criteria": criteria,
    }


def try_parse_float(text: str) -> Optional[float]:
    text = safe_strip(text)
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None


def format_multiline_block(title: str, content: str) -> str:
    content = safe_strip(content)
    if not content:
        content = "[EMPTY]"
    return f"### {title}\n\n{content}\n"


def build_sample_markdown(idx: int, sample: Dict[str, Any]) -> str:
    response_text = safe_strip(sample.get("response", ""))
    labels_text = safe_strip(sample.get("labels", ""))
    messages = sample.get("messages", [])

    user_content = extract_user_content(messages)
    prompt_fields = parse_prompt_fields(user_content)

    pred = parse_prediction_text(response_text)
    gold = parse_prediction_text(labels_text)

    pred_score_f = try_parse_float(pred["score"])
    gold_score_f = try_parse_float(gold["score"])

    score_match = "N/A"
    score_diff = "N/A"
    if pred_score_f is not None and gold_score_f is not None:
        diff = abs(pred_score_f - gold_score_f)
        score_diff = f"{diff:.3f}"
        score_match = "YES" if diff == 0 else "NO"

    md = []
    md.append(f"# Sample {idx}")
    md.append("")
    md.append(f"- **Dimension**: {prompt_fields['dimension'] or '[EMPTY]'}")
    md.append(f"- **Pred Score**: {pred['score'] or '[EMPTY]'}")
    md.append(f"- **Gold Score**: {gold['score'] or '[EMPTY]'}")
    md.append(f"- **Score Exact Match**: {score_match}")
    md.append(f"- **Score Abs Diff**: {score_diff}")
    md.append("")

    md.append(format_multiline_block("Question", prompt_fields["question"]))
    md.append(format_multiline_block("Original Answer", prompt_fields["answer"]))
    md.append(format_multiline_block("Criteria", prompt_fields["criteria"]))

    md.append("## Prediction")
    md.append("")
    md.append(format_multiline_block("Predicted Reason", pred["reason"]))
    md.append(format_multiline_block("Predicted Modified Answer", pred["modified_answer"]))

    md.append("## Label")
    md.append("")
    md.append(format_multiline_block("Gold Reason", gold["reason"]))
    md.append(format_multiline_block("Gold Modified Answer", gold["modified_answer"]))

    md.append("## Raw Output")
    md.append("")
    md.append("```text")
    md.append(response_text or "[EMPTY]")
    md.append("```")
    md.append("")
    md.append("---")
    md.append("")

    return "\n".join(md)


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip bad json line {line_no}: {e}")
    return items


def build_summary(samples: list[Dict[str, Any]]) -> str:
    total = len(samples)
    parsed_pred_score = 0
    parsed_gold_score = 0
    exact_match = 0

    for sample in samples:
        pred = parse_prediction_text(safe_strip(sample.get("response", "")))
        gold = parse_prediction_text(safe_strip(sample.get("labels", "")))

        pred_score = try_parse_float(pred["score"])
        gold_score = try_parse_float(gold["score"])

        if pred_score is not None:
            parsed_pred_score += 1
        if gold_score is not None:
            parsed_gold_score += 1
        if pred_score is not None and gold_score is not None and pred_score == gold_score:
            exact_match += 1

    lines = []
    lines.append("# Readable Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total samples: **{total}**")
    lines.append(f"- Parsed predicted scores: **{parsed_pred_score}**")
    lines.append(f"- Parsed gold scores: **{parsed_gold_score}**")
    lines.append(f"- Score exact match count: **{exact_match}**")
    if total > 0:
        lines.append(f"- Score exact match rate: **{exact_match / total:.2%}**")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input jsonl path")
    parser.add_argument("--output", type=str, required=True, help="Output markdown path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    samples = read_jsonl(input_path)

    report_parts = [build_summary(samples)]
    for i, sample in enumerate(samples, start=1):
        report_parts.append(build_sample_markdown(i, sample))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report_parts), encoding="utf-8")

    print(f"[OK] Wrote readable report to: {output_path}")


if __name__ == "__main__":
    main()