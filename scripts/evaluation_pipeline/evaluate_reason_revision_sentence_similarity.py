#!/usr/bin/env python
"""Evaluate sentence-level similarity for reasons and revision suggestions.

For each prediction row, this script compares:
1. predicted/generated reason vs. ground-truth reason
2. predicted/generated revision_suggestions vs. ground-truth revision_suggestions

Each text is split into sentence-level units. For a field, the script embeds all
ground-truth and generated units, builds the pairwise cosine-similarity matrix,
and reports:

- Recall: average over ground-truth units of their best generated-unit match.
- Precision: average over generated units of their best ground-truth-unit match.
- F1: harmonic mean of precision and recall.

Summaries macro-average Precision, Recall, and F1 over samples, with no baseline
or GPT-5 comparison.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Optional

from evaluate_revision_suggestions_relevance import (
    DEFAULT_REFERENCE,
    parse_label,
    sample_maps,
)
from pipeline_common import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    build_base_urls,
    cosine_similarity,
    embed_texts_with_retries,
    iter_jsonl,
    load_jsonl,
    normalize_text,
    parse_ports,
    resolve_model,
    wait_for_servers,
    write_jsonl,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_PREDICTION_DIR = Path("evaluation/revision_suggestions_relevance")
DEFAULT_OUTPUT_DIR = Path("evaluation/revision_suggestions_relevance")
DEFAULT_PREDICTION_GLOB = "*.jsonl"
EXCLUDED_DISCOVERY_DIRS = {
    "sentence_unit_similarities",
    "similarities",
    "similarities_filtered",
    "summaries",
}

TAG_RE_TEMPLATE = r"(?is)<{tag}>\s*(.*?)\s*</{tag}>"
COLON_CLASS = r"[:\uFF1A]"
HEADER_REASON_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:reason|feedback|generated_reason|predicted_reason)\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:revision suggestions|revision_suggestions|edit intent|modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
HEADER_REVISION_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:revision suggestions|revision_suggestions|edit intent|generated_revision_suggestions|predicted_revision_suggestions)\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|[\r\n]+")
LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def harmonic_mean(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denominator


def split_sentence_units(text: Any) -> list[str]:
    raw = str(text or "").strip().replace("\r\n", "\n")
    if not raw:
        return []
    raw = LIST_MARKER_RE.sub("\n", raw)
    units = [normalize_text(part) for part in SENTENCE_SPLIT_RE.split(raw)]
    return [unit for unit in units if unit]


def decode_text_payload(value: Any) -> list[str]:
    """Extract plausible generated text strings from raw response payloads."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        candidates = [text]
        try:
            decoded = json.loads(text)
        except Exception:
            decoded = None
        if decoded is not None:
            candidates.extend(decode_text_payload(decoded))
        return candidates
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(decode_text_payload(item))
        return texts
    if isinstance(value, dict):
        texts: list[str] = []
        for key in ("raw_output", "output", "response", "completion", "content"):
            if key in value:
                texts.extend(decode_text_payload(value.get(key)))
        text_value = value.get("text")
        if isinstance(text_value, list):
            texts.extend(str(item) for item in text_value if item is not None)
        elif text_value is not None:
            texts.append(str(text_value))
        choices = value.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    message = choice.get("message")
                    if isinstance(message, dict) and message.get("content") is not None:
                        texts.append(str(message.get("content")))
                    elif choice.get("text") is not None:
                        texts.append(str(choice.get("text")))
        return texts
    return [str(value)]


def is_placeholder_tag_value(value: str, field: str) -> bool:
    normalized = normalize_text(value).lower()
    placeholders = {
        "reason": {"reason", "feedback", "one concise sentence"},
        "revision": {"revision suggestions", "edit instructions", "edit intent"},
    }
    return normalized in placeholders[field]


def extract_tag_value(text: str, tag: str, field: str) -> str:
    pattern = re.compile(TAG_RE_TEMPLATE.format(tag=re.escape(tag)))
    matches = [normalize_text(match.group(1)) for match in pattern.finditer(text or "")]
    for value in reversed(matches):
        if value and not is_placeholder_tag_value(value, field):
            return value
    return ""


def extract_header_value(text: str, field: str) -> str:
    pattern = HEADER_REASON_RE if field == "reason" else HEADER_REVISION_RE
    match = pattern.search(text or "")
    return normalize_text(match.group(1)) if match else ""


def extract_field_from_text(text: str, field: str) -> tuple[str, Optional[str]]:
    if not text:
        return "", None

    parsed = parse_label(text)
    parsed_value = normalize_text(
        parsed.get("reason") if field == "reason" else parsed.get("revision_suggestions")
    )
    if parsed_value and not is_placeholder_tag_value(parsed_value, field):
        return parsed_value, parsed.get("parse_error")

    tag = "r" if field == "reason" else "rs"
    tagged_value = extract_tag_value(text, tag, field)
    if tagged_value:
        return tagged_value, None

    header_value = extract_header_value(text, field)
    if header_value and not is_placeholder_tag_value(header_value, field):
        return header_value, None

    return "", None


def prediction_text_candidates(row: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in (
        "raw_output",
        "generate_response",
        "request_response",
        "response",
        "output",
        "completion",
        "text",
    ):
        candidates.extend(decode_text_payload(row.get(key)))

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = normalize_text(message.get("role")).lower()
            if role == "assistant":
                candidates.append(str(message.get("content") or ""))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def prediction_field_from_row(row: dict[str, Any], field: str) -> tuple[str, Optional[str]]:
    if field == "reason":
        direct_values = (
            row.get("generated_reason"),
            row.get("predicted_reason"),
            row.get("reason"),
            row.get("target_reason"),
            row.get("predicted_rationale"),
            row.get("rationale"),
        )
        missing = "missing predicted_reason"
    else:
        direct_values = (
            row.get("generated_revision_suggestions"),
            row.get("predicted_revision_suggestions"),
            row.get("revision_suggestions"),
            row.get("target_revision_suggestions"),
            row.get("edit_intent"),
            row.get("predicted_edit_intent"),
        )
        missing = "missing predicted_revision_suggestions"

    for value in direct_values:
        text = normalize_text(value)
        if text and not is_placeholder_tag_value(text, field):
            return text, None

    for candidate in prediction_text_candidates(row):
        text, parse_error = extract_field_from_text(candidate, field)
        if text:
            return text, parse_error

    return "", missing


def reference_field_from_sample(sample: dict[str, Any], prediction: dict[str, Any], field: str) -> str:
    if field == "reason":
        return normalize_text(sample.get("reference_reason") or prediction.get("reference_reason") or prediction.get("ground_truth_reason"))
    return normalize_text(
        sample.get("reference_revision_suggestions")
        or prediction.get("reference_revision_suggestions")
        or prediction.get("ground_truth_revision_suggestions")
        or prediction.get("ground_truth_suggestion")
    )


def load_reference_samples(path: Path, limit: Optional[int]) -> list[dict[str, Any]]:
    from evaluate_revision_suggestions_relevance import load_reference_samples as load_samples

    samples = load_samples(path, limit)
    records = load_jsonl(path)
    if limit is not None:
        records = records[: max(0, limit)]

    for sample, record in zip(samples, records):
        if not normalize_text(sample.get("reference_reason")):
            sample["reference_reason"] = normalize_text(
                record.get("reference_reason")
                or record.get("predicted_reason")
                or record.get("reason")
                or record.get("orig_feedback")
                or record.get("feedback")
            )
        if not normalize_text(sample.get("reference_revision_suggestions")):
            sample["reference_revision_suggestions"] = normalize_text(
                record.get("reference_revision_suggestions")
                or record.get("predicted_revision_suggestions")
                or record.get("revision_suggestions")
                or record.get("target_revision_suggestions")
                or record.get("edit_intent")
            )
        if sample.get("reference_score") is None:
            sample["reference_score"] = record.get("reference_score") or record.get("predicted_score") or record.get("score") or record.get("orig_score")
    return samples


def coalesce_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = normalize_text(row.get(key))
        if text:
            return text
    return ""


def content_key(row: dict[str, Any]) -> Optional[str]:
    question = coalesce_text(row, "question", "orig_instruction", "instruction")
    answer = coalesce_text(row, "answer", "orig_response", "response")
    dimension = coalesce_text(row, "dimension_name", "evaluation_dimension", "orig_criteria", "criteria")
    if not (question and answer and dimension):
        return None
    raw = json.dumps(
        {
            "question": question,
            "answer": answer,
            "dimension": dimension,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def sample_maps_by_key(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    samples_by_id, samples_by_index = sample_maps(samples)
    samples_by_content: dict[str, dict[str, Any]] = {}
    for sample in samples:
        key = content_key(sample)
        if key and key not in samples_by_content:
            samples_by_content[key] = sample
    return samples_by_id, samples_by_index, samples_by_content


def resolve_sample_for_prediction(
    row: dict[str, Any],
    fallback_index: int,
    samples_by_id: dict[str, dict[str, Any]],
    samples_by_index: dict[int, dict[str, Any]],
    samples_by_content: Optional[dict[str, dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    key = content_key(row)
    if key and samples_by_content is not None and key in samples_by_content:
        return samples_by_content[key]

    sample_id = normalize_text(row.get("sample_id"))
    if sample_id and sample_id in samples_by_id:
        return samples_by_id[sample_id]

    index = row.get("index")
    if isinstance(index, int) and index in samples_by_index:
        return samples_by_index[index]
    if isinstance(index, str) and index.strip().isdigit():
        numeric_index = int(index.strip())
        if numeric_index in samples_by_index:
            return samples_by_index[numeric_index]

    if sample_id.isdigit():
        numeric_id = int(sample_id)
        if numeric_id in samples_by_index:
            return samples_by_index[numeric_id]

    return samples_by_index.get(fallback_index)


def normalized_selector(text: str) -> str:
    return str(text or "").strip().strip("\"'").replace("\\", "/").lower()


def positive_limit(value: Optional[int]) -> Optional[int]:
    if value is None or value <= 0:
        return None
    return value


def build_metric_rows(
    prediction_path: Path,
    samples: list[dict[str, Any]],
    sample_size: Optional[int],
) -> list[dict[str, Any]]:
    samples_by_id, samples_by_index, samples_by_content = sample_maps_by_key(samples)
    rows: list[dict[str, Any]] = []
    for fallback_index, prediction in enumerate(iter_jsonl(prediction_path)):
        if sample_size is not None and fallback_index >= sample_size:
            break
        sample = resolve_sample_for_prediction(
            prediction,
            fallback_index,
            samples_by_id,
            samples_by_index,
            samples_by_content,
        )
        row: dict[str, Any] = {
            "sample_id": normalize_text(prediction.get("sample_id")) or None,
            "index": prediction.get("index", fallback_index),
            "prediction_file": str(prediction_path),
            "reference_score": prediction.get("reference_score"),
            "predicted_score": prediction.get("predicted_score") or prediction.get("score"),
            "parse_error": prediction.get("parse_error"),
            "request_error": prediction.get("request_error"),
            "embedding_error": None,
        }
        if sample is None:
            row.update(
                {
                    "reference_reason": None,
                    "predicted_reason": None,
                    "reference_revision_suggestions": None,
                    "predicted_revision_suggestions": None,
                    "alignment_error": "prediction row could not be aligned to reference sample",
                }
            )
            rows.append(row)
            continue

        predicted_reason, reason_parse_error = prediction_field_from_row(prediction, "reason")
        predicted_revision, revision_parse_error = prediction_field_from_row(prediction, "revision")
        row.update(
            {
                "sample_id": sample["sample_id"],
                "index": sample["index"],
                "reference_score": sample.get("reference_score") or prediction.get("reference_score"),
                "reference_reason": reference_field_from_sample(sample, prediction, "reason") or None,
                "predicted_reason": predicted_reason or None,
                "reference_revision_suggestions": reference_field_from_sample(sample, prediction, "revision") or None,
                "predicted_revision_suggestions": predicted_revision or None,
                "reason_parse_error": reason_parse_error,
                "revision_parse_error": revision_parse_error,
                "alignment_error": None,
            }
        )
        rows.append(row)
    return rows


def endpoint_for_batch(base_urls: list[str], batch_index: int) -> str:
    return base_urls[batch_index % len(base_urls)]


def embed_unique_texts(
    texts: list[str],
    *,
    executor: ThreadPoolExecutor,
    base_urls: list[str],
    model: str,
    api_key: str,
    batch_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> tuple[dict[str, list[float]], Optional[str]]:
    unique_texts = list(dict.fromkeys(texts))
    embeddings_by_text: dict[str, list[float]] = {}
    future_to_texts = {}
    for batch_index, start in enumerate(range(0, len(unique_texts), batch_size)):
        batch_texts = unique_texts[start : start + batch_size]
        future = executor.submit(
            embed_texts_with_retries,
            base_url=endpoint_for_batch(base_urls, batch_index),
            api_key=api_key,
            model=model,
            texts=batch_texts,
            timeout_seconds=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        future_to_texts[future] = batch_texts

    errors: list[str] = []
    for future in as_completed(future_to_texts):
        batch_texts = future_to_texts[future]
        try:
            embeddings = future.result()
        except Exception as exc:
            errors.append(str(exc))
            continue
        for text, embedding in zip(batch_texts, embeddings):
            embeddings_by_text[text] = embedding

    return embeddings_by_text, "; ".join(errors) if errors else None


def compute_unit_metrics(
    ground_truth_units: list[str],
    generated_units: list[str],
    embeddings_by_text: dict[str, list[float]],
) -> dict[str, Any]:
    if not ground_truth_units:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "error": "missing ground_truth units",
        }
    if not generated_units:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "error": "missing generated units",
        }

    matrix: list[list[Optional[float]]] = []
    for gt_unit in ground_truth_units:
        gt_embedding = embeddings_by_text.get(gt_unit)
        row: list[Optional[float]] = []
        for gen_unit in generated_units:
            gen_embedding = embeddings_by_text.get(gen_unit)
            similarity = (
                cosine_similarity(gt_embedding, gen_embedding)
                if gt_embedding is not None and gen_embedding is not None
                else None
            )
            row.append(similarity)
        matrix.append(row)

    if any(similarity is None for row in matrix for similarity in row):
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "error": "missing unit embedding or invalid cosine similarity",
        }

    numeric_matrix = [[float(value) for value in row] for row in matrix]
    recall_matches = [max(row) for row in numeric_matrix]
    precision_matches = [max(numeric_matrix[row_index][col_index] for row_index in range(len(numeric_matrix))) for col_index in range(len(generated_units))]
    recall = average(recall_matches)
    precision = average(precision_matches)
    f1 = harmonic_mean(precision, recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error": None,
        "matrix": numeric_matrix,
        "ground_truth_unit_max_similarities": recall_matches,
        "generated_unit_max_similarities": precision_matches,
    }


def fill_sentence_metrics(
    rows: list[dict[str, Any]],
    *,
    executor: ThreadPoolExecutor,
    base_urls: list[str],
    model: str,
    api_key: str,
    batch_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    include_matrices: bool,
    include_unit_scores: bool,
) -> None:
    all_units: list[str] = []
    for row in rows:
        row["reference_reason_units"] = split_sentence_units(row.get("reference_reason"))
        row["predicted_reason_units"] = split_sentence_units(row.get("predicted_reason"))
        row["reference_revision_suggestions_units"] = split_sentence_units(row.get("reference_revision_suggestions"))
        row["predicted_revision_suggestions_units"] = split_sentence_units(row.get("predicted_revision_suggestions"))
        all_units.extend(row["reference_reason_units"])
        all_units.extend(row["predicted_reason_units"])
        all_units.extend(row["reference_revision_suggestions_units"])
        all_units.extend(row["predicted_revision_suggestions_units"])

    embeddings_by_text, embedding_error = embed_unique_texts(
        all_units,
        executor=executor,
        base_urls=base_urls,
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )

    for row in rows:
        if embedding_error:
            row["embedding_error"] = embedding_error

        reason_metrics = compute_unit_metrics(
            row["reference_reason_units"],
            row["predicted_reason_units"],
            embeddings_by_text,
        )
        revision_metrics = compute_unit_metrics(
            row["reference_revision_suggestions_units"],
            row["predicted_revision_suggestions_units"],
            embeddings_by_text,
        )

        for prefix, metrics in (("reason", reason_metrics), ("revision_suggestions", revision_metrics)):
            row[f"{prefix}_precision"] = format_float(metrics["precision"])
            row[f"{prefix}_recall"] = format_float(metrics["recall"])
            row[f"{prefix}_f1"] = format_float(metrics["f1"])
            row[f"{prefix}_metric_error"] = metrics["error"]
            if include_matrices and metrics.get("matrix") is not None:
                row[f"{prefix}_similarity_matrix"] = [
                    [format_float(value) for value in matrix_row]
                    for matrix_row in metrics["matrix"]
                ]
            if include_unit_scores and metrics.get("ground_truth_unit_max_similarities") is not None:
                row[f"{prefix}_ground_truth_unit_max_similarities"] = [
                    format_float(value) for value in metrics["ground_truth_unit_max_similarities"]
                ]
                row[f"{prefix}_generated_unit_max_similarities"] = [
                    format_float(value) for value in metrics["generated_unit_max_similarities"]
                ]

        metric_values = [
            row.get("reason_precision"),
            row.get("revision_suggestions_precision"),
            row.get("reason_recall"),
            row.get("revision_suggestions_recall"),
            row.get("reason_f1"),
            row.get("revision_suggestions_f1"),
        ]
        if all(isinstance(value, (int, float)) for value in metric_values):
            row["combined_precision"] = format_float(average([float(row["reason_precision"]), float(row["revision_suggestions_precision"])]))
            row["combined_recall"] = format_float(average([float(row["reason_recall"]), float(row["revision_suggestions_recall"])]))
            row["combined_f1"] = format_float(average([float(row["reason_f1"]), float(row["revision_suggestions_f1"])]))
        else:
            row["combined_precision"] = None
            row["combined_recall"] = None
            row["combined_f1"] = None


def finite_metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def summarize_metric_group(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    precision_values = finite_metric_values(rows, f"{prefix}_precision")
    recall_values = finite_metric_values(rows, f"{prefix}_recall")
    f1_values = finite_metric_values(rows, f"{prefix}_f1")
    return {
        f"{prefix}_valid_count": len(f1_values),
        f"{prefix}_macro_precision": format_float(average(precision_values)) if precision_values else None,
        f"{prefix}_macro_recall": format_float(average(recall_values)) if recall_values else None,
        f"{prefix}_macro_f1": format_float(average(f1_values)) if f1_values else None,
        f"{prefix}_median_f1": format_float(statistics.median(f1_values)) if f1_values else None,
        f"{prefix}_missing_ground_truth": sum(1 for row in rows if not row.get(f"reference_{prefix}_units")),
        f"{prefix}_missing_generated": sum(1 for row in rows if not row.get(f"predicted_{prefix}_units")),
        f"{prefix}_metric_errors": sum(1 for row in rows if row.get(f"{prefix}_metric_error")),
    }


def summarize_rows(prediction_path: Path, output_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    combined_precision = finite_metric_values(rows, "combined_precision")
    combined_recall = finite_metric_values(rows, "combined_recall")
    combined_f1 = finite_metric_values(rows, "combined_f1")
    summary = {
        "name": prediction_path.stem,
        "prediction_file": str(prediction_path),
        "metrics_file": str(output_path),
        "rows": len(rows),
        "alignment_errors": sum(1 for row in rows if row.get("alignment_error")),
        "parse_errors": sum(1 for row in rows if row.get("parse_error") or row.get("reason_parse_error") or row.get("revision_parse_error")),
        "request_errors": sum(1 for row in rows if row.get("request_error")),
        "embedding_errors": sum(1 for row in rows if row.get("embedding_error")),
        "combined_valid_count": len(combined_f1),
        "combined_macro_precision": format_float(average(combined_precision)) if combined_precision else None,
        "combined_macro_recall": format_float(average(combined_recall)) if combined_recall else None,
        "combined_macro_f1": format_float(average(combined_f1)) if combined_f1 else None,
    }
    summary.update(summarize_metric_group(rows, "reason"))
    summary.update(summarize_metric_group(rows, "revision_suggestions"))
    return summary


def metrics_output_path(output_dir: Path, prediction_path: Path) -> Path:
    return output_dir / "sentence_unit_similarities" / f"{prediction_path.stem}.jsonl"


def process_prediction_file(
    *,
    prediction_path: Path,
    output_path: Path,
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    rows = build_metric_rows(prediction_path, samples, positive_limit(args.sample_size))
    fill_sentence_metrics(
        rows,
        executor=executor,
        base_urls=base_urls,
        model=model,
        api_key=args.embedding_api_key,
        batch_size=args.embedding_batch_size,
        timeout=args.embedding_timeout,
        retries=args.embedding_retries,
        retry_sleep=args.embedding_retry_sleep,
        include_matrices=args.include_matrices,
        include_unit_scores=args.include_unit_scores,
    )
    write_jsonl(output_path, rows)
    return summarize_rows(prediction_path, output_path, rows)


def is_discoverable_prediction_file(path: Path, root: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False
    try:
        relative_parts = set(path.relative_to(root).parts[:-1])
    except ValueError:
        relative_parts = set(path.parts[:-1])
    return not (relative_parts & EXCLUDED_DISCOVERY_DIRS)


def selector_matches_prediction(path: Path, selector: str, root: Optional[Path]) -> bool:
    target = normalized_selector(selector)
    if not target:
        return False

    candidates = {
        normalized_selector(path.name),
        normalized_selector(path.stem),
        normalized_selector(str(path)),
        normalized_selector(str(path.resolve())),
    }
    if root is not None:
        try:
            candidates.add(normalized_selector(str(path.relative_to(root))))
        except ValueError:
            pass

    return target in candidates or any(target in candidate for candidate in candidates)


def apply_prediction_file_selectors(
    paths: list[Path],
    selectors: list[str],
    root: Optional[Path],
) -> list[Path]:
    if not selectors:
        return paths

    selected: list[Path] = []
    unmatched: list[str] = []
    for selector in selectors:
        selector_path = Path(selector).expanduser()
        if selector_path.is_file():
            selected.append(selector_path.resolve())
            continue

        matches = [path for path in paths if selector_matches_prediction(path, selector, root)]
        if matches:
            selected.extend(matches)
        else:
            unmatched.append(selector)

    if unmatched:
        raise FileNotFoundError(
            "No prediction file matched --prediction-file selector(s): "
            + ", ".join(repr(item) for item in unmatched)
        )
    return selected


def discover_prediction_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.predictions:
        paths.extend(path.resolve() for path in args.predictions)
    root: Optional[Path] = None
    if args.prediction_dir is not None:
        root = args.prediction_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {root}")
        globber = root.rglob if args.recursive else root.glob
        paths.extend(
            path.resolve()
            for path in globber(args.prediction_glob)
            if is_discoverable_prediction_file(path, root)
        )
    paths = apply_prediction_file_selectors(paths, args.prediction_file, root)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths):
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def write_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summaries_dir / "reason_revision_sentence_similarity_summary.json"
    summary_csv = summaries_dir / "reason_revision_sentence_similarity_summary.csv"
    ranking = sorted(
        summaries,
        key=lambda item: (
            item["combined_macro_f1"] is None,
            -(item["combined_macro_f1"] if item["combined_macro_f1"] is not None else float("-inf")),
        ),
    )
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summaries": summaries,
        "ranking_by_combined_macro_f1": ranking,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "name",
        "combined_valid_count",
        "combined_macro_precision",
        "combined_macro_recall",
        "combined_macro_f1",
        "reason_valid_count",
        "reason_macro_precision",
        "reason_macro_recall",
        "reason_macro_f1",
        "reason_median_f1",
        "revision_suggestions_valid_count",
        "revision_suggestions_macro_precision",
        "revision_suggestions_macro_recall",
        "revision_suggestions_macro_f1",
        "revision_suggestions_median_f1",
        "rows",
        "alignment_errors",
        "parse_errors",
        "request_errors",
        "embedding_errors",
        "reason_missing_ground_truth",
        "reason_missing_generated",
        "revision_suggestions_missing_ground_truth",
        "revision_suggestions_missing_generated",
        "prediction_file",
        "metrics_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return summary_json, summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute sentence-level embedding Precision/Recall/F1 for predicted reason and "
            "revision_suggestions against ground-truth reference labels."
        )
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Tagged SFT test JSONL.")
    parser.add_argument("--predictions", nargs="*", type=Path, default=[], help="Explicit prediction JSONL files.")
    parser.add_argument(
        "--prediction-file",
        "--file",
        action="append",
        default=[],
        help=(
            "Run only matching prediction file(s). Accepts a full path, file name, stem, "
            "or substring under --prediction-dir. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help=f"Directory containing generated prediction JSONL files. Default: {DEFAULT_PREDICTION_DIR}",
    )
    parser.add_argument("--prediction-glob", default=DEFAULT_PREDICTION_GLOB)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Rows sampled from the start of each prediction file. Use <=0 to process all rows. Default: 10000.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional reference row limit. If omitted, defaults to --sample-size.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--embedding-base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument(
        "--embedding-ports",
        nargs="*",
        default=["8000-8003"],
        help="Embedding ports, e.g. 8000 or 8000-8003. Default: 8000-8003.",
    )
    parser.add_argument("--embedding-model", type=str, default="", help="Embedding model id. Empty means fetch from /models.")
    parser.add_argument("--embedding-api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-workers-per-endpoint", type=int, default=1)
    parser.add_argument("--embedding-timeout", type=float, default=120.0)
    parser.add_argument("--embedding-retries", type=int, default=3)
    parser.add_argument("--embedding-retry-sleep", type=float, default=1.0)
    parser.add_argument("--skip-embedding-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)

    parser.add_argument("--include-matrices", action="store_true", help="Write per-sample pairwise similarity matrices.")
    parser.add_argument("--include-unit-scores", action="store_true", help="Write per-unit best-match similarity lists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_size = positive_limit(args.sample_size)
    reference_limit = positive_limit(args.limit) if args.limit is not None else sample_size
    samples = load_reference_samples(args.reference.resolve(), reference_limit)
    if not samples:
        raise FileNotFoundError(f"No reference samples loaded from {args.reference}")
    print(f"[reference] {args.reference.resolve()} samples={len(samples)}")

    prediction_files = discover_prediction_files(args)
    if not prediction_files:
        raise ValueError("No prediction files found. Pass --predictions or --prediction-dir.")
    for path in prediction_files:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")

    embedding_base_urls = build_base_urls(args.embedding_base_url_template, parse_ports(args.embedding_ports))
    if not args.skip_embedding_health_check:
        wait_for_servers(embedding_base_urls, args.health_check_timeout, args.health_check_interval)
    embedding_model = args.embedding_model.strip() or resolve_model("", embedding_base_urls, args.health_check_timeout)

    print(f"[prediction_files] {len(prediction_files)}")
    print(f"[sample_size] {sample_size if sample_size is not None else 'all'}")
    print(f"[output_dir] {args.output_dir}")
    print(f"[embedding_model] {embedding_model}")
    print(f"[embedding_base_urls] {embedding_base_urls}")

    summaries: list[dict[str, Any]] = []
    max_workers = max(1, len(embedding_base_urls) * args.embedding_workers_per_endpoint)
    iterator = tqdm(prediction_files, desc="sentence similarity files") if tqdm is not None else prediction_files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for prediction_path in iterator:
            output_path = metrics_output_path(args.output_dir, prediction_path)
            if output_path.exists() and not args.overwrite:
                summaries.append(summarize_rows(prediction_path, output_path, load_jsonl(output_path)))
                continue
            summaries.append(
                process_prediction_file(
                    prediction_path=prediction_path,
                    output_path=output_path,
                    samples=samples,
                    args=args,
                    executor=executor,
                    base_urls=embedding_base_urls,
                    model=embedding_model,
                )
            )

    summary_json, summary_csv = write_summary(args.output_dir, summaries)
    print(f"Sentence similarity summary saved to: {summary_json}")
    print(f"Sentence similarity CSV saved to: {summary_csv}")


if __name__ == "__main__":
    main()
