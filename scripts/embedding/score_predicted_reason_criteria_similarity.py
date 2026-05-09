#!/usr/bin/env python
"""Score similarity between predicted reasons and their predicted-score criteria."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_PORTS = "8000,8001,8002,8003"
DEFAULT_BASE_URL_TEMPLATE = "http://127.0.0.1:{port}/v1"
CRITERION_PATTERN = re.compile(
    r"(?ms)^\s*(?P<score>-?\d+(?:\.\d+)?)\s*[:：]\s*(?P<body>.*?)(?=^\s*-?\d+(?:\.\d+)?\s*[:：]|\Z)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read prediction JSONL files, embed predicted_reason and the criterion matching "
            "predicted_score through OpenAI-compatible vLLM embedding endpoints, then write "
            "their cosine similarity."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("evaluation/baseline_new/predictions"),
        help="Directory containing source prediction JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/baseline_new/predicted_reason_criteria_similarity"),
        help="Directory where per-file similarity JSONL outputs will be written.",
    )
    parser.add_argument(
        "--base-urls",
        type=str,
        default="",
        help=(
            "Comma-separated embedding base URLs. If omitted, URLs are built from "
            "--base-url-template and --ports."
        ),
    )
    parser.add_argument(
        "--base-url-template",
        type=str,
        default=DEFAULT_BASE_URL_TEMPLATE,
        help="Template used with --ports, e.g. http://127.0.0.1:{port}/v1.",
    )
    parser.add_argument(
        "--ports",
        type=str,
        default=DEFAULT_PORTS,
        help="Comma-separated ports used with --base-url-template.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Embedding model name. If omitted, the first endpoint's /models result is used.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="Bearer token for OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of text inputs sent in one /embeddings request.",
    )
    parser.add_argument(
        "--row-buffer-size",
        type=int,
        default=256,
        help="Number of JSONL records buffered before writing results.",
    )
    parser.add_argument(
        "--workers-per-endpoint",
        type=int,
        default=1,
        help="Concurrent embedding requests per endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for failed embedding requests.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=1.0,
        help="Base sleep seconds between retries.",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Also write predicted_reason_embedding and matched_criterion_embedding.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSONL files.",
    )
    parser.add_argument(
        "--no-sample-progress",
        action="store_true",
        help="Disable per-file sample-level progress bars.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def append_jsonl_rows(handle: Any, rows: Iterable[Dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("Empty base URL.")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def parse_ports(raw_ports: str) -> List[int]:
    ports: List[int] = []
    for item in raw_ports.split(","):
        item = item.strip()
        if not item:
            continue
        ports.append(int(item))
    if not ports:
        raise ValueError("No ports provided.")
    return ports


def build_base_urls(raw_base_urls: str, template: str, raw_ports: str) -> List[str]:
    if raw_base_urls.strip():
        values = [normalize_base_url(item) for item in raw_base_urls.split(",") if item.strip()]
    else:
        values = [normalize_base_url(template.format(port=port)) for port in parse_ports(raw_ports)]

    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    if not deduped:
        raise ValueError("No embedding endpoints resolved.")
    return deduped


def fetch_model_id(base_url: str, timeout: float) -> Optional[str]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib_request.urlopen(url, timeout=min(timeout, 10.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if isinstance(first, dict):
        model_id = first.get("id")
        if model_id:
            return str(model_id)
    return None


def post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def embed_texts(
    base_url: str,
    model: str,
    api_key: str,
    texts: Sequence[str],
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> List[List[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = {
        "model": model,
        "input": list(texts),
    }
    last_error: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            response = post_json(url, payload, api_key=api_key, timeout=timeout)
            data = response.get("data")
            if not isinstance(data, list):
                raise ValueError(f"Embedding response has no data list from {base_url}")
            sorted_data = sorted(data, key=lambda item: int(item.get("index", 0)))
            embeddings = [item.get("embedding") for item in sorted_data]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch from {base_url}: expected {len(texts)}, got {len(embeddings)}"
                )
            if not all(isinstance(item, list) for item in embeddings):
                raise ValueError(f"Embedding response contains invalid embedding values from {base_url}")
            return embeddings  # type: ignore[return-value]
        except (urllib_error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"Embedding request failed after {retries + 1} attempts at {base_url}: {last_error}")


def score_key(score: Any) -> Optional[str]:
    if score is None:
        return None
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)):
        if not math.isfinite(float(score)):
            return None
        rounded = round(float(score))
        if abs(float(score) - rounded) < 1e-6:
            return str(int(rounded))
        return str(score)

    text = str(score).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group(0))
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return match.group(0)


def extract_criterion(criteria: Any, score: Any) -> Tuple[Optional[str], Optional[str]]:
    criteria_text = "" if criteria is None else str(criteria)
    target_score = score_key(score)
    if not criteria_text.strip():
        return None, "missing criteria"
    if target_score is None:
        return None, "missing or invalid predicted_score"

    matches = list(CRITERION_PATTERN.finditer(criteria_text))
    for match in matches:
        current_score = score_key(match.group("score"))
        if current_score == target_score:
            body = match.group("body").strip()
            if not body:
                return None, f"empty criterion for score {target_score}"
            return f"{target_score}: {body}", None
    return None, f"criterion for score {target_score} not found"


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        lv = float(left_value)
        rv = float(right_value)
        dot += lv * rv
        left_norm += lv * lv
        right_norm += rv * rv
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def list_prediction_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.rglob("*.jsonl") if path.is_file())


def maybe_tqdm(items: Sequence[Path], desc: str) -> Iterable[Path]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def build_output_row(record: Dict[str, Any]) -> Dict[str, Any]:
    matched_criterion, parse_error = extract_criterion(record.get("criteria"), record.get("predicted_score"))
    predicted_reason = record.get("predicted_reason")
    if isinstance(predicted_reason, str):
        predicted_reason = predicted_reason.strip()
    elif predicted_reason is not None:
        predicted_reason = str(predicted_reason)

    return {
        "sample_id": record.get("sample_id"),
        "index": record.get("index"),
        "predicted_score": record.get("predicted_score"),
        "predicted_score_key": score_key(record.get("predicted_score")),
        "predicted_reason": predicted_reason or None,
        "matched_criterion": matched_criterion,
        "similarity": None,
        "criteria_parse_error": parse_error,
        "embedding_error": None,
    }


def endpoint_for_batch(base_urls: Sequence[str], batch_index: int) -> str:
    return base_urls[batch_index % len(base_urls)]


def fill_similarity_scores(
    rows: List[Dict[str, Any]],
    executor: ThreadPoolExecutor,
    base_urls: Sequence[str],
    model: str,
    api_key: str,
    batch_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    include_embeddings: bool,
) -> None:
    pending_items: List[Tuple[int, str, str]] = []
    for row_index, row in enumerate(rows):
        reason = row.get("predicted_reason")
        criterion = row.get("matched_criterion")
        if not reason:
            row["embedding_error"] = "missing predicted_reason"
            continue
        if not criterion:
            continue
        pending_items.append((row_index, "predicted_reason", str(reason)))
        pending_items.append((row_index, "matched_criterion", str(criterion)))

    if not pending_items:
        return

    future_to_items: Dict[Future[List[List[float]]], List[Tuple[int, str, str]]] = {}
    for batch_index, start in enumerate(range(0, len(pending_items), batch_size)):
        batch_items = pending_items[start : start + batch_size]
        texts = [item[2] for item in batch_items]
        base_url = endpoint_for_batch(base_urls, batch_index)
        future = executor.submit(
            embed_texts,
            base_url,
            model,
            api_key,
            texts,
            timeout,
            retries,
            retry_sleep,
        )
        future_to_items[future] = batch_items

    embeddings_by_row: Dict[int, Dict[str, List[float]]] = {}
    for future in as_completed(future_to_items):
        batch_items = future_to_items[future]
        try:
            embeddings = future.result()
        except Exception as exc:
            message = str(exc)
            for row_index, _, _ in batch_items:
                rows[row_index]["embedding_error"] = message
            continue

        for (row_index, field_name, _), embedding in zip(batch_items, embeddings):
            embeddings_by_row.setdefault(row_index, {})[field_name] = embedding

    for row_index, embeddings in embeddings_by_row.items():
        reason_embedding = embeddings.get("predicted_reason")
        criterion_embedding = embeddings.get("matched_criterion")
        if reason_embedding is None or criterion_embedding is None:
            rows[row_index]["embedding_error"] = rows[row_index]["embedding_error"] or "missing embedding result"
            continue
        rows[row_index]["similarity"] = cosine_similarity(reason_embedding, criterion_embedding)
        if include_embeddings:
            rows[row_index]["predicted_reason_embedding"] = reason_embedding
            rows[row_index]["matched_criterion_embedding"] = criterion_embedding


def process_file(
    input_path: Path,
    output_path: Path,
    executor: ThreadPoolExecutor,
    base_urls: Sequence[str],
    model: str,
    api_key: str,
    batch_size: int,
    row_buffer_size: int,
    timeout: float,
    retries: int,
    retry_sleep: float,
    include_embeddings: bool,
    show_sample_progress: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    row_buffer: List[Dict[str, Any]] = []
    progress = None
    if show_sample_progress and tqdm is not None:
        progress = tqdm(
            total=count_jsonl_rows(input_path),
            desc=input_path.name,
            unit="sample",
            leave=False,
        )

    try:
        with output_path.open("w", encoding="utf-8") as f:
            for record in iter_jsonl(input_path):
                row_buffer.append(build_output_row(record))
                if progress is not None:
                    progress.update(1)
                if len(row_buffer) < row_buffer_size:
                    continue

                fill_similarity_scores(
                    row_buffer,
                    executor=executor,
                    base_urls=base_urls,
                    model=model,
                    api_key=api_key,
                    batch_size=batch_size,
                    timeout=timeout,
                    retries=retries,
                    retry_sleep=retry_sleep,
                    include_embeddings=include_embeddings,
                )
                append_jsonl_rows(f, row_buffer)
                row_count += len(row_buffer)
                row_buffer.clear()

            if row_buffer:
                fill_similarity_scores(
                    row_buffer,
                    executor=executor,
                    base_urls=base_urls,
                    model=model,
                    api_key=api_key,
                    batch_size=batch_size,
                    timeout=timeout,
                    retries=retries,
                    retry_sleep=retry_sleep,
                    include_embeddings=include_embeddings,
                )
                append_jsonl_rows(f, row_buffer)
                row_count += len(row_buffer)
    finally:
        if progress is not None:
            progress.close()

    return row_count


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.row_buffer_size <= 0:
        raise ValueError("--row-buffer-size must be positive.")
    if args.workers_per_endpoint <= 0:
        raise ValueError("--workers-per-endpoint must be positive.")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    prediction_files = list_prediction_files(input_dir)
    if not prediction_files:
        raise FileNotFoundError(f"No JSONL files found under {input_dir}")

    base_urls = build_base_urls(args.base_urls, args.base_url_template, args.ports)
    model = args.model.strip() or fetch_model_id(base_urls[0], args.timeout)
    if not model:
        raise ValueError("No model provided and auto-fetch from /models failed. Pass --model explicitly.")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Embedding endpoints: {base_urls}")
    print(f"Embedding model: {model}")
    print(f"Found JSONL files: {len(prediction_files)}")

    max_workers = max(1, len(base_urls) * args.workers_per_endpoint)
    total_rows = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for input_path in maybe_tqdm(prediction_files, desc="Scoring files"):
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing file: {output_path}")
                continue

            row_count = process_file(
                input_path=input_path,
                output_path=output_path,
                executor=executor,
                base_urls=base_urls,
                model=model,
                api_key=args.api_key,
                batch_size=args.batch_size,
                row_buffer_size=args.row_buffer_size,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
                include_embeddings=args.include_embeddings,
                show_sample_progress=not args.no_sample_progress,
            )
            total_rows += row_count
            print(f"Wrote {row_count} rows to {output_path}")

    print(f"Finished. Total rows written: {total_rows}")


if __name__ == "__main__":
    main()
