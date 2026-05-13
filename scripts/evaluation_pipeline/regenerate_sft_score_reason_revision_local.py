#!/usr/bin/env python
"""Regenerate SFT labels with score, reason, revision suggestions, and rewrite.

This script is for existing SFT-style JSONL files such as:

datasets/train/final_train_split.jsonl
datasets/train/final_test_split.jsonl

Each input record may be either:
1. Chat messages, where the user message contains Question / Answer /
   Evaluation_dimension / Criteria (0-5), or
2. A flat record with question, answer, dimension_name/evaluation_dimension,
   and score_criteria/criteria_0..criteria_5 fields.

The script calls local OpenAI-compatible vLLM endpoints and writes clean SFT
messages whose assistant content is strict JSON:
{"score": ..., "reason": "...", "revision_suggestions": "...", "modified_answer": "..."}
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

from pipeline_common import (
    COLON_CLASS,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    detect_complete_score_range,
    format_score_criteria,
    load_jsonl,
    models_endpoint_ready,
    normalize_score_criteria,
    normalize_text,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
    write_jsonl,
)
from score_reason_rewrite_local import parse_model_output

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_OUTPUT_DIR = Path("datasets/train/score_reason_revision_regen")

SYSTEM_PROMPT = (
    "You are an expert evaluator and answer rewriter. Evaluate a candidate answer "
    "to a given question under the specified evaluation dimension and 0-5 scoring "
    "criteria. Use only the provided question, candidate answer, evaluation dimension, and scoring "
    "criteria. Do not add unsupported facts. Return strict JSON only, with exactly these "
    "keys: score, reason, revision_suggestions, modified_answer."
    "The reason must be at most 2 short sentences. "
    "The revision_suggestions must be at most 2 short sentences. "
    "The modified_answer must be at most 4 short sentences. "
)

USER_TASK_INSTRUCTION = (
    "### Task Description:\n"
    "You are given a question, a candidate answer, one evaluation dimension, "
    "and the complete 0-5 scoring criteria for that dimension.\n"
    "\n"
    "Your task is to evaluate and improve the candidate answer using only the provided "
    "question, candidate answer, evaluation dimension, and scoring criteria.\n"
    "\n"
    "Follow these steps:\n"
    "1. Assign one integer score from 0 to 5 according to the provided scoring criteria.\n"
    "2. Provide a concise but concrete reason for the score. The reason should explain "
    "the candidate answer's strengths, weaknesses to satisfy the evaluation dimension. "
    "When identifying a specific problem, explicitly mark it with the prefix \"error:\".\n"
    "3. Based on the scoring reason, provide actionable revision suggestions explaining "
    "how the candidate answer should be revised to better satisfy the score-5 criterion.\n"
    "4. Rewrite the candidate answer into a stronger modified_answer for the same "
    "question and the same evaluation dimension. The modified_answer should address the "
    "identified errors, follow the revision suggestions, and avoid adding unsupported "
    "facts.\n"
    "5. Return strict JSON only with exactly this schema:\n"
    "{\"score\": <int>, \"reason\": \"...\", \"revision_suggestions\": \"...\", "
    "\"modified_answer\": \"...\"}"
)

USER_TEMPLATE = (
    "{instruction}\n\n"
    "Question:\n{question}\n\n"
    "Candidate Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Score Criteria (0-5):\n{criteria_text}"
)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("Question",),
    "answer": ("Answer", "Candidate Answer", "Response"),
    "dimension_name": ("Evaluation_dimension", "Evaluation Dimension", "Dimension"),
    "criteria": ("Criteria (0-5)", "Score Criteria (0-5)", "Criteria", "0-5 Score Criteria"),
}


def _is_empty(value: Any) -> bool:
    return normalize_text(value) == ""


def _canonical_header(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower())


def _alias_pattern(alias: str) -> str:
    tokens = [re.escape(token) for token in re.split(r"[\s_\-]+", alias.strip()) if token]
    return r"[\s_\-]*".join(tokens)


def extract_sections(text: str) -> dict[str, str]:
    header_to_key: dict[str, str] = {}
    patterns: list[str] = []
    for key, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            header_to_key[_canonical_header(alias)] = key
            patterns.append(_alias_pattern(alias))

    header_union = "|".join(sorted(patterns, key=len, reverse=True))
    pattern = re.compile(
        rf"(?im)^\s*(?P<header>{header_union})(?:\s*\([^)\n]*\))?\s*{COLON_CLASS}\s*"
    )
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = header_to_key.get(_canonical_header(match.group("header")))
        if not key:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[key] = text[start:end].strip()
    return sections


def last_message_content(record: dict[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if normalize_text(message.get("role")).lower() == role:
            contents.append(str(message.get("content") or ""))
    return contents[-1] if contents else ""


def normalize_criteria_text_from_record(record: dict[str, Any]) -> str:
    full_score_criteria = normalize_text(record.get("full_score_criteria"))
    score_criteria = normalize_score_criteria(record, full_score_criteria)
    score_range = detect_complete_score_range(score_criteria)
    if score_range:
        return format_score_criteria(score_criteria, record)

    raw = record.get("score_criteria") or record.get("criteria") or record.get("0-5_Criteria")
    if isinstance(raw, dict):
        return format_score_criteria(raw, record)
    return str(raw or "").strip()


def sample_from_record(record: dict[str, Any], index: int, split: str) -> dict[str, Any] | None:
    user_content = last_message_content(record, "user")
    sections = extract_sections(user_content) if user_content else {}

    question = normalize_text(record.get("question")) or normalize_text(sections.get("question"))
    answer = normalize_text(record.get("answer")) or normalize_text(sections.get("answer"))
    dimension_name = (
        normalize_text(record.get("dimension_name"))
        or normalize_text(record.get("evaluation_dimension"))
        or normalize_text(sections.get("dimension_name"))
    )
    criteria_text = normalize_criteria_text_from_record(record) or str(sections.get("criteria") or "").strip()

    if _is_empty(question) or _is_empty(answer) or _is_empty(dimension_name) or _is_empty(criteria_text):
        return None

    explicit_id = normalize_text(record.get("sample_id")) or normalize_text(record.get("id"))
    if explicit_id:
        sample_id = f"{split}:{explicit_id}"
    else:
        digest = hashlib.sha1(
            json.dumps(
                {
                    "split": split,
                    "index": index,
                    "question": question,
                    "answer": answer,
                    "dimension_name": dimension_name,
                    "criteria_text": criteria_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        sample_id = f"{split}:{digest}"

    user_prompt = USER_TEMPLATE.format(
        instruction=USER_TASK_INSTRUCTION,
        question=question,
        answer=answer,
        dimension_name=dimension_name,
        criteria_text=criteria_text,
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "source_index": index,
        "question": question,
        "answer": answer,
        "dimension_name": dimension_name,
        "criteria_text": criteria_text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def load_samples(
    path: Path,
    split: str,
    *,
    skip_missing: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if limit is not None:
        rows = rows[: max(0, limit)]
    samples: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(rows):
        sample = sample_from_record(record, index, split)
        if sample is None:
            skipped += 1
            if not skip_missing:
                raise ValueError(f"{path} line {index + 1} cannot be parsed into Q/A/D/Cs.")
            continue
        samples.append(sample)
    print(f"[load] split={split} input={path} rows={len(rows)} samples={len(samples)} skipped={skipped}")
    return samples


def assistant_json_from_prediction(row: dict[str, Any]) -> str:
    payload = {
        "score": row["score"],
        "reason": row["reason"],
        "revision_suggestions": row["revision_suggestions"],
        "modified_answer": row["modified_answer"],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_sft_row(sample: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            sample["messages"][0],
            sample["messages"][1],
            {"role": "assistant", "content": assistant_json_from_prediction(pred)},
        ]
    }


def load_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for row in load_jsonl(path):
        if row.get("ok") is not True:
            continue
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            completed[sample_id] = row
    return completed


def run_one(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "source_index": sample["source_index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "dimension_name": sample["dimension_name"],
        "criteria_text": sample["criteria_text"],
        "score": None,
        "reason": "",
        "revision_suggestions": "",
        "modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "model": model,
        "latency_sec": None,
        "strict_json_ok": False,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.api_key,
            model=model,
            messages=sample["messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = parse_model_output(raw_text)
        ok = (
            parsed["score"] is not None
            and 0 <= int(parsed["score"]) <= 5
            and bool(parsed["reason"])
            and bool(parsed["revision_suggestions"])
            and bool(parsed["modified_answer"])
        )
        row.update(
            {
                "score": parsed["score"],
                "reason": parsed["reason"],
                "revision_suggestions": parsed["revision_suggestions"],
                "modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "ok": ok,
                "parse_error": parsed["parse_error"] if not ok else None,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"],
            }
        )
    except Exception as exc:
        row.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


def filter_ready_base_urls(
    base_urls: list[str],
    *,
    timeout_seconds: int,
    workers: int,
) -> list[str]:
    ready_urls: list[str] = []
    lock = threading.Lock()

    def _probe(base_url: str) -> None:
        if models_endpoint_ready(base_url, timeout_seconds):
            with lock:
                ready_urls.append(base_url)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_probe, base_url) for base_url in base_urls]
        for future in as_completed(futures):
            future.result()

    ready_set = set(ready_urls)
    return [base_url for base_url in base_urls if base_url in ready_set]


def resolve_runtime(args: argparse.Namespace) -> tuple[list[str], str]:
    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if args.skip_health_check:
        pass
    elif args.wait_all_ports:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    else:
        before = len(base_urls)
        base_urls = filter_ready_base_urls(
            base_urls,
            timeout_seconds=args.health_check_timeout,
            workers=args.probe_workers,
        )
        print(f"[ports] ready={len(base_urls)} / requested={before}")
        if not base_urls:
            raise RuntimeError(
                "No ready vLLM endpoints found. Check --ports/--base-url-template, "
                "or pass --wait-all-ports if servers are still starting."
            )
    model = args.model.strip() or resolve_model("", base_urls, args.health_check_timeout)
    return base_urls, model


def generate_split(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> None:
    samples = load_samples(input_path, split, skip_missing=args.skip_missing, limit=args.limit)
    safe_unlink(cache_path, args.overwrite)
    safe_unlink(output_path, args.overwrite)

    completed = load_completed_predictions(cache_path)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[run] split={split} output={output_path}")
    print(f"[run] split={split} cache={cache_path}")
    print(f"[run] split={split} total={len(samples)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc=f"{split}_regen", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, sample, index, args, base_urls, model): sample
                for index, sample in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                if row.get("ok") is True:
                    completed[row["sample_id"]] = row
                with lock:
                    append_jsonl(cache_path, row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    sft_rows: list[dict[str, Any]] = []
    for sample in samples:
        pred = completed.get(sample["sample_id"])
        if pred is not None and pred.get("ok") is True:
            sft_rows.append(build_sft_row(sample, pred))
    write_jsonl(output_path, sft_rows)

    ok_count = len(sft_rows)
    print(f"[done] split={split} sft_rows={ok_count} failed_or_pending={len(samples) - ok_count}")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_urls, model = resolve_runtime(args)
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")

    if args.train_input:
        generate_split(
            split="train",
            input_path=Path(args.train_input),
            output_path=output_dir / args.train_output_name,
            cache_path=output_dir / args.train_cache_name,
            args=args,
            base_urls=base_urls,
            model=model,
        )

    if args.test_input:
        generate_split(
            split="test",
            input_path=Path(args.test_input),
            output_path=output_dir / args.test_output_name,
            cache_path=output_dir / args.test_cache_name,
            args=args,
            base_urls=base_urls,
            model=model,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate train/test SFT data with JSON assistant labels: "
            "score, reason, revision_suggestions, modified_answer."
        )
    )
    parser.add_argument("--train-input", type=Path, default=Path("datasets/train/final_train_split.jsonl"))
    parser.add_argument("--test-input", type=Path, default=Path("datasets/train/final_test_split.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-output-name", default="score_reason_revision_sft_train.jsonl")
    parser.add_argument("--test-output-name", default="score_reason_revision_sft_test.jsonl")
    parser.add_argument("--train-cache-name", default="score_reason_revision_train_cache.jsonl")
    parser.add_argument("--test-cache-name", default="score_reason_revision_test_cache.jsonl")
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows from each split.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8001-8007"], help="Ports like: 8001-8007 or 8001 8002.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument(
        "--wait-all-ports",
        action="store_true",
        help="Wait until every requested port is ready. By default, only ready ports are used.",
    )
    parser.add_argument(
        "--probe-workers",
        type=int,
        default=32,
        help="Concurrent workers used to probe requested vLLM endpoints.",
    )
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
