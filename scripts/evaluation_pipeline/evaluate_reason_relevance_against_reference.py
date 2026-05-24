#!/usr/bin/env python
"""Evaluate reason relevance against SFT reference labels.

For each test row, compare the model's predicted reason with the reference
reason from the tagged SFT assistant label:

<s>score</s><r>reason</r><rs>revision suggestions</rs><ra>refined answer</ra>

The script can either:
1. generate predictions from a local OpenAI-compatible chat endpoint, or
2. read existing prediction JSONL files.

It embeds predicted/reference reasons through an OpenAI-compatible /embeddings
endpoint and writes cosine similarities plus summary files. Optionally, it
computes Pearson/Spearman correlations against a baseline model's similarity
file (e.g., cortex-5).
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

from pipeline_common import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    cosine_similarity,
    embed_texts_with_retries,
    extract_json_object,
    load_jsonl,
    models_endpoint_ready,
    normalize_text,
    parse_int_score,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
    write_jsonl,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_REFERENCE = Path(
    "datasets/train/final_score_reason_plaintext_sft_final_tagged_system_user_balanced_v2/"
    "final_score_reason_plaintext_test.jsonl"
)
DEFAULT_PREDICTION_DIR = Path("evaluation/final_test_model_outputs")
DEFAULT_OUTPUT_DIR = Path("evaluation/reason_relevance")
DEFAULT_PREDICTION_GLOB = "*.jsonl"
DEFAULT_BASELINE_PREDICTION = Path(
    "evaluation/final_test_model_outputs/cortex-5_final_test_outputs.jsonl"
)

COLON_CLASS = r"[:\uFF1A]"
TAGGED_LABEL_RE = re.compile(
    r"(?is)<s>\s*(?P<score>.*?)\s*</s>\s*"
    r"<r>\s*(?P<reason>.*?)\s*</r>\s*"
    r"<rs>\s*(?P<revision>.*?)\s*</rs>\s*"
    r"<ra>\s*(?P<modified>.*?)\s*</ra>"
)
SCORE_LINE_RE = re.compile(rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$")
REASON_RE = re.compile(
    rf"(?is)(?:^|\n)\s*reason\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:revision suggestions|edit intent|modified answer|revised answer)\s*{COLON_CLASS}|$)"
)
REVISION_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:revision suggestions|revision_suggestions|edit intent)\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}\s*(.*)$"
)


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = average(x)
    mean_y = average(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    sum_x2 = sum(value * value for value in dx)
    sum_y2 = sum(value * value for value in dy)
    if sum_x2 == 0 or sum_y2 == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(sum_x2 * sum_y2)


def rankdata(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[pairs[k][0]] = avg_rank
        i = j
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson_correlation(rankdata(x), rankdata(y))


def as_finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return cleaned.strip("._-") or "model"


def stable_sample_id(record: dict[str, Any], index: int) -> str:
    explicit = normalize_text(record.get("sample_id") or record.get("id") or record.get("unique_id"))
    if explicit:
        return explicit
    raw = json.dumps(record.get("messages", record), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{index}||{raw}".encode("utf-8")).hexdigest()
    return f"row:{index}:{digest}"


def split_messages(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")

    prompt_messages: list[dict[str, str]] = []
    assistant_contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = str(message.get("content") or "")
        if role == "assistant":
            assistant_contents.append(content)
        elif role in {"system", "user"}:
            prompt_messages.append({"role": role, "content": content})

    if not prompt_messages:
        raise ValueError("record has no prompt messages")
    if not assistant_contents:
        raise ValueError("record has no assistant reference label")
    return prompt_messages, assistant_contents[-1]


def parse_label(text: str) -> dict[str, Any]:
    raw = str(text or "").strip().replace("\r\n", "\n")
    tagged = TAGGED_LABEL_RE.search(raw)
    if tagged:
        score = parse_int_score(normalize_text(tagged.group("score")))
        return {
            "score": score,
            "reason": normalize_text(tagged.group("reason")),
            "revision_suggestions": normalize_text(tagged.group("revision")),
            "modified_answer": normalize_text(tagged.group("modified")),
            "raw_output": raw,
            "parse_error": None if score is not None else "Failed to parse tagged score.",
            "strict_json_ok": False,
            "tagged_ok": score is not None,
        }

    parsed = extract_json_object(raw)
    strict_json_ok = parsed is not None
    if parsed is None:
        try:
            loaded = json.loads(raw)
        except Exception:
            loaded = None
        parsed = loaded if isinstance(loaded, dict) else None

    if isinstance(parsed, dict):
        score = parse_int_score(parsed.get("score") or parsed.get("Score"))
        reason = normalize_text(parsed.get("reason") or parsed.get("Reason"))
        revision = normalize_text(
            parsed.get("revision_suggestions")
            or parsed.get("Revision Suggestions")
            or parsed.get("edit_intent")
            or parsed.get("Edit Intent")
        )
        modified = normalize_text(
            parsed.get("modified_answer")
            or parsed.get("Modified Answer")
            or parsed.get("revised_answer")
            or parsed.get("Revised Answer")
        )
        ok = score is not None and bool(reason) and bool(revision) and bool(modified)
        return {
            "score": score,
            "reason": reason,
            "revision_suggestions": revision,
            "modified_answer": modified,
            "raw_output": raw,
            "parse_error": None if ok else "Parsed JSON but required fields are missing.",
            "strict_json_ok": strict_json_ok and ok,
            "tagged_ok": False,
        }

    score_match = SCORE_LINE_RE.search(raw)
    reason_match = REASON_RE.search(raw)
    revision_match = REVISION_RE.search(raw)
    modified_match = MODIFIED_RE.search(raw)
    score = int(score_match.group(1)) if score_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    revision = normalize_text(revision_match.group(1)) if revision_match else ""
    modified = normalize_text(modified_match.group(1)) if modified_match else ""
    ok = score is not None and bool(reason) and bool(revision) and bool(modified)
    return {
        "score": score,
        "reason": reason,
        "revision_suggestions": revision,
        "modified_answer": modified,
        "raw_output": raw,
        "parse_error": None if ok else "Failed to parse required fields.",
        "strict_json_ok": False,
        "tagged_ok": False,
    }


def load_reference_samples(path: Path, limit: Optional[int]) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    if limit is not None:
        records = records[: max(0, limit)]

    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt_messages, reference_output = split_messages(record)
        reference = parse_label(reference_output)
        samples.append(
            {
                "sample_id": stable_sample_id(record, index),
                "index": index,
                "prompt_messages": prompt_messages,
                "reference_output": reference_output,
                "reference_score": reference["score"],
                "reference_reason": reference["reason"],
                "reference_revision_suggestions": reference["revision_suggestions"],
                "reference_modified_answer": reference["modified_answer"],
                "reference_parse_error": reference["parse_error"],
            }
        )
    return samples


def prediction_reason_from_row(row: dict[str, Any]) -> tuple[str, Optional[str]]:
    direct = normalize_text(
        row.get("predicted_reason")
        or row.get("reason")
        or row.get("target_reason")
        or row.get("predicted_rationale")
        or row.get("rationale")
    )
    if direct:
        return direct, None

    raw_output = normalize_text(row.get("raw_output"))
    if raw_output:
        parsed = parse_label(raw_output)
        reason = normalize_text(parsed.get("reason"))
        if reason:
            return reason, parsed.get("parse_error")

    messages = row.get("messages")
    if isinstance(messages, list):
        assistant_contents = [
            str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict) and normalize_text(message.get("role")).lower() == "assistant"
        ]
        if assistant_contents:
            parsed = parse_label(assistant_contents[-1])
            reason = normalize_text(parsed.get("reason"))
            if reason:
                return reason, parsed.get("parse_error")

    return "", "missing predicted_reason"


def row_key_from_prediction(row: dict[str, Any], fallback_index: int) -> tuple[str, Any]:
    sample_id = normalize_text(row.get("sample_id"))
    if sample_id:
        return ("sample_id", sample_id)
    index = row.get("index")
    if isinstance(index, int):
        return ("index", index)
    if isinstance(index, str) and index.strip().isdigit():
        return ("index", int(index.strip()))
    return ("index", fallback_index)


def sample_maps(samples: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    by_id = {sample["sample_id"]: sample for sample in samples}
    by_index = {int(sample["index"]): sample for sample in samples}
    return by_id, by_index


def resolve_sample_for_prediction(
    row: dict[str, Any],
    fallback_index: int,
    samples_by_id: dict[str, dict[str, Any]],
    samples_by_index: dict[int, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    key_type, key_value = row_key_from_prediction(row, fallback_index)
    if key_type == "sample_id":
        return samples_by_id.get(str(key_value))
    return samples_by_index.get(int(key_value))


def endpoint_for_batch(base_urls: list[str], batch_index: int) -> str:
    return base_urls[batch_index % len(base_urls)]


def fill_similarity_scores(
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
    include_embeddings: bool,
) -> None:
    pending: list[tuple[int, str, str]] = []
    for row_index, row in enumerate(rows):
        predicted = normalize_text(row.get("predicted_reason"))
        reference = normalize_text(row.get("reference_reason"))
        if not predicted:
            row["embedding_error"] = "missing predicted_reason"
            continue
        if not reference:
            row["embedding_error"] = "missing reference_reason"
            continue
        pending.append((row_index, "predicted", predicted))
        pending.append((row_index, "reference", reference))

    future_to_items = {}
    for batch_index, start in enumerate(range(0, len(pending), batch_size)):
        batch_items = pending[start : start + batch_size]
        future = executor.submit(
            embed_texts_with_retries,
            base_url=endpoint_for_batch(base_urls, batch_index),
            api_key=api_key,
            model=model,
            texts=[item[2] for item in batch_items],
            timeout_seconds=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
        )
        future_to_items[future] = batch_items

    embeddings_by_row: dict[int, dict[str, list[float]]] = {}
    for future in as_completed(future_to_items):
        batch_items = future_to_items[future]
        try:
            embeddings = future.result()
        except Exception as exc:
            for row_index, _, _ in batch_items:
                rows[row_index]["embedding_error"] = str(exc)
            continue
        for (row_index, name, _), embedding in zip(batch_items, embeddings):
            embeddings_by_row.setdefault(row_index, {})[name] = embedding

    for row_index, embeddings in embeddings_by_row.items():
        predicted_embedding = embeddings.get("predicted")
        reference_embedding = embeddings.get("reference")
        if predicted_embedding is None or reference_embedding is None:
            rows[row_index]["embedding_error"] = rows[row_index]["embedding_error"] or "missing embedding result"
            continue
        rows[row_index]["similarity"] = cosine_similarity(predicted_embedding, reference_embedding)
        if include_embeddings:
            rows[row_index]["predicted_reason_embedding"] = predicted_embedding
            rows[row_index]["reference_reason_embedding"] = reference_embedding


def build_similarity_rows(prediction_path: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples_by_id, samples_by_index = sample_maps(samples)
    rows: list[dict[str, Any]] = []
    for fallback_index, prediction in enumerate(load_jsonl(prediction_path)):
        sample = resolve_sample_for_prediction(prediction, fallback_index, samples_by_id, samples_by_index)
        if sample is None:
            rows.append(
                {
                    "sample_id": normalize_text(prediction.get("sample_id")) or None,
                    "index": prediction.get("index", fallback_index),
                    "prediction_file": str(prediction_path),
                    "reference_reason": None,
                    "predicted_reason": None,
                    "similarity": None,
                    "parse_error": "prediction row could not be aligned to reference sample",
                    "embedding_error": None,
                }
            )
            continue
        predicted_reason, parse_error = prediction_reason_from_row(prediction)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "index": sample["index"],
                "prediction_file": str(prediction_path),
                "reference_score": sample.get("reference_score"),
                "reference_reason": sample.get("reference_reason") or None,
                "predicted_score": prediction.get("predicted_score") or prediction.get("score"),
                "predicted_reason": predicted_reason or None,
                "similarity": None,
                "parse_error": parse_error or prediction.get("parse_error"),
                "request_error": prediction.get("request_error"),
                "embedding_error": None,
            }
        )
    return rows


def similarity_output_path(output_dir: Path, prediction_path: Path) -> Path:
    return output_dir / "similarities" / f"{prediction_path.stem}.jsonl"


def summarize_similarity_rows(prediction_path: Path, output_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["similarity"])
        for row in rows
        if isinstance(row.get("similarity"), (int, float)) and math.isfinite(float(row["similarity"]))
    ]
    return {
        "name": prediction_path.stem,
        "prediction_file": str(prediction_path),
        "similarity_file": str(output_path),
        "rows": len(rows),
        "valid_count": len(values),
        "missing_reference_reason": sum(1 for row in rows if not row.get("reference_reason")),
        "missing_predicted_reason": sum(1 for row in rows if not row.get("predicted_reason")),
        "parse_errors": sum(1 for row in rows if row.get("parse_error")),
        "request_errors": sum(1 for row in rows if row.get("request_error")),
        "embedding_errors": sum(1 for row in rows if row.get("embedding_error")),
        "mean_similarity": format_float(average(values)) if values else None,
        "median_similarity": format_float(statistics.median(values)) if values else None,
        "min_similarity": format_float(min(values)) if values else None,
        "max_similarity": format_float(max(values)) if values else None,
    }


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
    rows = build_similarity_rows(prediction_path, samples)
    fill_similarity_scores(
        rows,
        executor=executor,
        base_urls=base_urls,
        model=model,
        api_key=args.embedding_api_key,
        batch_size=args.embedding_batch_size,
        timeout=args.embedding_timeout,
        retries=args.embedding_retries,
        retry_sleep=args.embedding_retry_sleep,
        include_embeddings=args.include_embeddings,
    )
    write_jsonl(output_path, rows)
    return summarize_similarity_rows(prediction_path, output_path, rows)


def run_generation_one(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "index": sample["index"],
        "judge_model": model,
        "reference_score": sample["reference_score"],
        "reference_reason": sample["reference_reason"],
        "predicted_score": None,
        "predicted_reason": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "latency_sec": None,
        "strict_json_ok": False,
        "tagged_ok": False,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.generation_api_key,
            model=model,
            messages=sample["prompt_messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.generation_timeout,
            retries=args.generation_retries,
            retry_sleep=args.generation_retry_sleep,
        )
        parsed = parse_label(raw_text)
        ok = parsed["score"] is not None and bool(parsed["reason"])
        row.update(
            {
                "predicted_score": parsed["score"],
                "predicted_reason": parsed["reason"],
                "raw_output": parsed["raw_output"],
                "ok": ok,
                "parse_error": parsed["parse_error"] if not ok else None,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"],
                "tagged_ok": parsed["tagged_ok"],
            }
        )
    except Exception as exc:
        row.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


def load_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for row in load_jsonl(path):
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            completed[sample_id] = row
    return completed


def ready_base_urls(base_urls: list[str], timeout_seconds: int) -> list[str]:
    return [base_url for base_url in base_urls if models_endpoint_ready(base_url, timeout_seconds)]


def generate_predictions(args: argparse.Namespace, samples: list[dict[str, Any]]) -> Path:
    base_urls = build_base_urls(args.generation_base_url_template, parse_ports(args.generation_ports))
    if args.skip_generation_health_check:
        ready = base_urls
    elif args.wait_all_generation_ports:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
        ready = base_urls
    else:
        ready = ready_base_urls(base_urls, args.health_check_timeout)
        print(f"[generation_ports] ready={len(ready)} / requested={len(base_urls)}")
        if not ready:
            raise RuntimeError("No ready generation endpoints found.")

    model = args.generation_model.strip() or resolve_model("", ready, args.health_check_timeout)
    prediction_path = args.generation_output
    if prediction_path is None:
        prediction_path = args.output_dir / "predictions" / f"{safe_name(model)}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    safe_unlink(prediction_path, args.overwrite)

    completed = load_completed_predictions(prediction_path)
    pending = [
        sample
        for sample in samples
        if sample["sample_id"] not in completed or completed[sample["sample_id"]].get("ok") is not True
    ]

    print(f"[generate] model={model}")
    print(f"[generate] endpoints={ready}")
    print(f"[generate] output={prediction_path}")
    print(f"[generate] total={len(samples)} completed={len(completed)} pending={len(pending)}")

    progress = tqdm(total=len(pending), desc="generate", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.generation_workers)) as executor:
            futures = {
                executor.submit(run_generation_one, sample, task_index, args, ready, model): sample
                for task_index, sample in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                completed[row["sample_id"]] = row
                append_jsonl(prediction_path, row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return prediction_path


def write_generation_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summaries_dir / "reason_similarity_summary.json"
    summary_csv = summaries_dir / "reason_similarity_summary.csv"
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summaries": summaries,
        "ranking_by_mean_similarity": sorted(
            summaries,
            key=lambda item: (
                item["mean_similarity"] is None,
                -(item["mean_similarity"] if item["mean_similarity"] is not None else float("-inf")),
            ),
        ),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "name",
        "valid_count",
        "mean_similarity",
        "median_similarity",
        "min_similarity",
        "max_similarity",
        "rows",
        "missing_reference_reason",
        "missing_predicted_reason",
        "parse_errors",
        "request_errors",
        "embedding_errors",
        "prediction_file",
        "similarity_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload["ranking_by_mean_similarity"]:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return summary_json, summary_csv


def load_similarity_rows_by_key(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    rows: dict[tuple[str, Any], dict[str, Any]] = {}
    for fallback_index, row in enumerate(load_jsonl(path)):
        key = row_key_from_prediction(row, fallback_index)
        if key not in rows:
            rows[key] = row
    return rows


def compute_correlation_against_baseline(baseline_path: Path, target_path: Path) -> dict[str, Any]:
    baseline_rows = load_similarity_rows_by_key(baseline_path)
    target_rows = load_similarity_rows_by_key(target_path)
    common_keys = sorted(set(baseline_rows) & set(target_rows), key=lambda item: (item[0], item[1]))
    valid_pairs: list[tuple[float, float]] = []
    for key in common_keys:
        baseline_similarity = as_finite_float(baseline_rows[key].get("similarity"))
        target_similarity = as_finite_float(target_rows[key].get("similarity"))
        if baseline_similarity is not None and target_similarity is not None:
            valid_pairs.append((baseline_similarity, target_similarity))
    baseline_values = [pair[0] for pair in valid_pairs]
    target_values = [pair[1] for pair in valid_pairs]
    differences = [target - baseline for baseline, target in valid_pairs]
    return {
        "baseline_similarity_file": str(baseline_path),
        "target_similarity_file": str(target_path),
        "target_name": target_path.stem,
        "common_samples": len(common_keys),
        "valid_count": len(valid_pairs),
        "baseline_mean": format_float(average(baseline_values)) if baseline_values else None,
        "target_mean": format_float(average(target_values)) if target_values else None,
        "mean_difference": format_float(average(differences)) if differences else None,
        "pearson": format_float(pearson_correlation(baseline_values, target_values)),
        "spearman": format_float(spearman_correlation(baseline_values, target_values)),
    }


def write_correlation_summary(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_json = summaries_dir / "reason_baseline_correlation_summary.json"
    summary_csv = summaries_dir / "reason_baseline_correlation_summary.csv"
    ranking = sorted(
        rows,
        key=lambda item: (
            item["spearman"] is None,
            -(item["spearman"] if item["spearman"] is not None else float("-inf")),
            -(item["pearson"] if item["pearson"] is not None else float("-inf")),
        ),
    )
    summary_json.write_text(
        json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "ranking_by_spearman": ranking}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "target_name",
        "valid_count",
        "common_samples",
        "pearson",
        "spearman",
        "baseline_mean",
        "target_mean",
        "mean_difference",
        "baseline_similarity_file",
        "target_similarity_file",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranking:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return summary_json, summary_csv


def discover_prediction_files(args: argparse.Namespace, generated_path: Optional[Path]) -> list[Path]:
    paths: list[Path] = []
    if args.predictions:
        paths.extend(path.resolve() for path in args.predictions)
    if args.prediction_dir is not None:
        paths.extend(
            sorted(
                path.resolve()
                for path in args.prediction_dir.glob(args.prediction_glob)
                if path.is_file()
            )
        )
    if args.baseline_prediction is not None:
        paths.append(args.baseline_prediction.resolve())
    if generated_path is not None:
        paths.append(generated_path.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate/evaluate reason relevance by embedding predicted reasons and reference reasons, "
            "then computing cosine similarity."
        )
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Tagged SFT test JSONL.")
    parser.add_argument("--predictions", nargs="*", type=Path, default=[], help="Existing prediction JSONL files.")
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
        help=(
            "Directory of existing prediction JSONL files. "
            f"Default: {DEFAULT_PREDICTION_DIR}"
        ),
    )
    parser.add_argument(
        "--prediction-glob",
        default=DEFAULT_PREDICTION_GLOB,
        help=f"Glob used under --prediction-dir. Default: {DEFAULT_PREDICTION_GLOB}",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit from the reference test set.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--generate", action="store_true", help="Call a local chat endpoint to generate predictions first.")
    parser.add_argument("--generation-output", type=Path, default=None, help="Prediction JSONL path used with --generate.")
    parser.add_argument("--generation-base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--generation-ports", nargs="*", default=["8000"], help="Generation ports, e.g. 8000 or 8000-8003.")
    parser.add_argument("--generation-model", type=str, default="", help="Chat model id. Empty means fetch from /models.")
    parser.add_argument("--generation-api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--generation-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--generation-timeout", type=int, default=180)
    parser.add_argument("--generation-retries", type=int, default=2)
    parser.add_argument("--generation-retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-generation-health-check", action="store_true")
    parser.add_argument("--wait-all-generation-ports", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)

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
    parser.add_argument("--include-embeddings", action="store_true")

    parser.add_argument(
        "--baseline-similarity",
        type=Path,
        default=None,
        help="Optional similarity JSONL used as baseline for Pearson/Spearman comparison.",
    )
    parser.add_argument(
        "--baseline-prediction",
        type=Path,
        default=DEFAULT_BASELINE_PREDICTION,
        help=(
            "Optional prediction JSONL used as baseline for Pearson/Spearman comparison. "
            "If set, its similarity file is computed in this run and used as the baseline."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_similarity is not None and args.baseline_prediction is not None:
        raise ValueError("Use only one of --baseline-similarity or --baseline-prediction.")

    samples = load_reference_samples(args.reference.resolve(), args.limit)
    if not samples:
        raise FileNotFoundError(f"No reference samples loaded from {args.reference}")
    print(f"[reference] {args.reference.resolve()} samples={len(samples)}")

    generated_path = generate_predictions(args, samples) if args.generate else None
    prediction_files = discover_prediction_files(args, generated_path)
    if not prediction_files:
        raise ValueError("No prediction files provided. Pass --predictions/--prediction-dir or use --generate.")
    for path in prediction_files:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
    if args.baseline_prediction is not None and not args.baseline_prediction.is_file():
        raise FileNotFoundError(f"Baseline prediction file not found: {args.baseline_prediction}")

    embedding_base_urls = build_base_urls(args.embedding_base_url_template, parse_ports(args.embedding_ports))
    if not args.skip_embedding_health_check:
        wait_for_servers(embedding_base_urls, args.health_check_timeout, args.health_check_interval)
    embedding_model = args.embedding_model.strip() or resolve_model("", embedding_base_urls, args.health_check_timeout)

    print(f"[embedding_model] {embedding_model}")
    print(f"[embedding_base_urls] {embedding_base_urls}")

    summaries: list[dict[str, Any]] = []
    max_workers = max(1, len(embedding_base_urls) * args.embedding_workers_per_endpoint)
    iterator = tqdm(prediction_files, desc="reason similarity files") if tqdm is not None else prediction_files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for prediction_path in iterator:
            output_path = similarity_output_path(args.output_dir, prediction_path)
            if output_path.exists() and not args.overwrite:
                summaries.append(summarize_similarity_rows(prediction_path, output_path, load_jsonl(output_path)))
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

    summary_json, summary_csv = write_generation_summary(args.output_dir, summaries)
    print(f"Similarity summary saved to: {summary_json}")
    print(f"Similarity CSV saved to: {summary_csv}")

    baseline_path: Optional[Path] = None
    if args.baseline_similarity is not None:
        baseline_path = args.baseline_similarity.resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline similarity file not found: {baseline_path}")
    elif args.baseline_prediction is not None:
        baseline_path = similarity_output_path(args.output_dir, args.baseline_prediction.resolve()).resolve()

    if baseline_path is not None:
        if not baseline_path.is_file():
            raise FileNotFoundError(
                "Baseline similarity file not found. "
                "Ensure the baseline prediction was included in this run or provide --baseline-similarity."
            )
        correlation_rows = [
            compute_correlation_against_baseline(
                baseline_path,
                similarity_output_path(args.output_dir, path).resolve(),
            )
            for path in prediction_files
            if similarity_output_path(args.output_dir, path).resolve() != baseline_path
        ]
        corr_json, corr_csv = write_correlation_summary(args.output_dir, correlation_rows)
        print(f"Correlation summary saved to: {corr_json}")
        print(f"Correlation CSV saved to: {corr_csv}")


if __name__ == "__main__":
    main()
