#!/usr/bin/env python
"""Generate model outputs on the final tagged SFT test set via external GPT API.

The input JSONL is expected to contain OpenAI-style messages:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<s>...</s><r>...</r><rs>...</rs><ra>...</ra>"}
  ]
}

For each row, this script sends only the system/user messages to an
OpenAI-compatible API endpoint, parses the model output, and writes
reference_* and predicted_* fields for downstream evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = ROOT / "scripts" / "evaluation_pipeline"
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import (  # noqa: E402
    append_jsonl,
    call_chat_with_retries,
    extract_json_object,
    extract_sections_by_headers,
    fetch_model_id,
    is_loopback_url,
    load_jsonl,
    normalize_text,
    parse_int_score,
    safe_unlink,
    write_jsonl,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

if load_dotenv is not None:
    load_dotenv()


DEFAULT_INPUT = Path(
    "datasets/train/final_score_reason_plaintext_sft_final_tagged_system_user_balanced_v2/"
    "final_score_reason_plaintext_test.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("evaluation/final_test_model_outputs")

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_BEARER_TOKEN_ENV = "OPENAI_BEARER_TOKEN"
DEFAULT_BASE_URL_ENV = "OPENAI_BASE_URL"

TAGGED_LABEL_RE = re.compile(
    r"(?is)<s>\s*(?P<score>.*?)\s*</s>\s*"
    r"<r>\s*(?P<reason>.*?)\s*</r>\s*"
    r"<rs>\s*(?P<revision>.*?)\s*</rs>\s*"
    r"<ra>\s*(?P<modified>.*?)\s*</ra>"
)
COLON_CLASS = r"[:\uFF1A]"
SCORE_LINE_RE = re.compile(rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$")
REASON_RE = re.compile(
    rf"(?is)(?:^|\n)\s*reason\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:revision suggestions|revision_suggestions|edit intent|modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
REVISION_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:revision suggestions|revision_suggestions|edit intent)\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}\s*(.*)$"
)

USER_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question",),
    "answer": ("answer", "candidate answer", "response"),
    "evaluation_dimension": ("evaluation_dimension", "evaluation dimension", "dimension"),
    "criteria": ("criteria", "score criteria", "rubric", "0-5 scoring criteria"),
}


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return cleaned.strip("._-") or "model"


def normalize_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Empty base URL.")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def normalize_api_key(value: str) -> str:
    token = str(value or "").strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def parse_label(text: str) -> dict[str, Any]:
    raw = str(text or "").strip().replace("\r\n", "\n")

    tagged = TAGGED_LABEL_RE.search(raw)
    if tagged:
        score = parse_int_score(normalize_text(tagged.group("score")))
        reason = normalize_text(tagged.group("reason"))
        revision = normalize_text(tagged.group("revision"))
        modified = normalize_text(tagged.group("modified"))
        ok = score is not None and bool(reason) and bool(revision) and bool(modified)
        return {
            "score": score,
            "reason": reason,
            "revision_suggestions": revision,
            "modified_answer": modified,
            "raw_output": raw,
            "parse_error": None if ok else "Tagged output is missing required fields.",
            "tagged_ok": ok,
            "strict_json_ok": False,
        }

    parsed = extract_json_object(raw)
    strict_json_ok = parsed is not None
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
            "tagged_ok": False,
            "strict_json_ok": strict_json_ok and ok,
        }

    score_match = SCORE_LINE_RE.search(raw)
    fallback_score = re.search(rf"(?i)\bscore\s*{COLON_CLASS}\s*([0-5])", raw)
    reason_match = REASON_RE.search(raw)
    revision_match = REVISION_RE.search(raw)
    modified_match = MODIFIED_RE.search(raw)

    score = None
    if score_match:
        score = int(score_match.group(1))
    elif fallback_score:
        score = int(fallback_score.group(1))

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
        "tagged_ok": False,
        "strict_json_ok": False,
    }


def stable_sample_id(record: dict[str, Any], index: int) -> str:
    explicit = normalize_text(record.get("sample_id") or record.get("id") or record.get("unique_id"))
    if explicit:
        return explicit
    raw = json.dumps(record.get("messages", record), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{index}||{raw}".encode("utf-8")).hexdigest()
    return f"row:{index}:{digest}"


def split_messages(record: dict[str, Any]) -> tuple[list[dict[str, str]], str, str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no messages list")

    prompt_messages: list[dict[str, str]] = []
    system_text = ""
    user_text = ""
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
            if role == "system" and not system_text:
                system_text = content
            if role == "user":
                user_text = content

    if not prompt_messages:
        raise ValueError("record has no prompt messages")
    if not assistant_contents:
        raise ValueError("record has no assistant reference label")

    return prompt_messages, assistant_contents[-1], system_text, user_text


def prepare_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt_messages, reference_output, system_text, user_text = split_messages(record)
        reference = parse_label(reference_output)
        sections = extract_sections_by_headers(user_text, USER_SECTION_ALIASES)
        sample_id = stable_sample_id(record, index)

        samples.append(
            {
                "sample_id": sample_id,
                "index": index,
                "prompt_messages": prompt_messages,
                "question": sections.get("question", "").strip(),
                "answer": sections.get("answer", "").strip(),
                "evaluation_dimension": sections.get("evaluation_dimension", "").strip(),
                "criteria": sections.get("criteria", "").strip(),
                "original_system_message": system_text,
                "original_user_message": user_text,
                "reference_output": reference_output,
                "reference_score": reference["score"],
                "reference_reason": reference["reason"],
                "reference_revision_suggestions": reference["revision_suggestions"],
                "reference_modified_answer": reference["modified_answer"],
                "reference_parse_error": reference["parse_error"],
                "reference_tagged_ok": reference["tagged_ok"],
            }
        )
    return samples


def load_completed_predictions(path: Path, *, resume_failed: bool) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for row in load_jsonl(path):
        sample_id = normalize_text(row.get("sample_id"))
        if not sample_id:
            continue
        if resume_failed or row.get("ok") is True:
            completed[sample_id] = row
    return completed


def parse_base_urls(raw: str) -> list[str]:
    cleaned = str(raw or "").strip()
    if not cleaned:
        return [normalize_base_url(DEFAULT_BASE_URL)]
    return [normalize_base_url(value) for value in cleaned.split(",") if value.strip()]


def resolve_api_key(args: argparse.Namespace) -> str:
    explicit = normalize_api_key(args.api_key)
    if explicit:
        return explicit
    env_token = normalize_api_key(os.getenv(args.api_key_env, ""))
    if env_token:
        return env_token
    fallback = normalize_api_key(os.getenv(DEFAULT_OPENAI_BEARER_TOKEN_ENV, ""))
    if fallback:
        return fallback
    raise ValueError(
        f"Missing API key. Pass --api-key or set env {args.api_key_env} (or {DEFAULT_OPENAI_BEARER_TOKEN_ENV})."
    )


def resolve_model(args: argparse.Namespace, base_urls: list[str]) -> str:
    model = first_non_empty(args.model, os.getenv(DEFAULT_MODEL_ENV, ""))
    if model:
        return model
    if base_urls and is_loopback_url(base_urls[0]):
        resolved = fetch_model_id(base_urls[0], args.health_check_timeout)
        if resolved:
            return resolved
    raise ValueError(f"Missing --model (or env {DEFAULT_MODEL_ENV}).")


def resolve_runtime(args: argparse.Namespace) -> tuple[list[str], str, str]:
    raw_base_url = first_non_empty(args.base_url, os.getenv(DEFAULT_BASE_URL_ENV, ""), DEFAULT_BASE_URL)
    base_urls = parse_base_urls(raw_base_url)
    api_key = resolve_api_key(args)

    if args.health_check and base_urls and is_loopback_url(base_urls[0]):
        from pipeline_common import wait_for_servers  # lazy import

        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)

    model = resolve_model(args, base_urls)
    return base_urls, api_key, model


def default_output_path(args: argparse.Namespace, model: str) -> Path:
    run_name = args.run_name.strip() or model
    return args.output_dir / f"{safe_name(run_name)}_final_test_outputs.jsonl"


def run_one(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    api_key: str,
    model: str,
    run_name: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "index": sample["index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "evaluation_dimension": sample["evaluation_dimension"],
        "criteria": sample["criteria"],
        "reference_score": sample["reference_score"],
        "reference_reason": sample["reference_reason"],
        "reference_revision_suggestions": sample["reference_revision_suggestions"],
        "reference_modified_answer": sample["reference_modified_answer"],
        "reference_output": sample["reference_output"],
        "reference_parse_error": sample["reference_parse_error"],
        "reference_tagged_ok": sample["reference_tagged_ok"],
        "original_system_message": sample["original_system_message"],
        "original_user_message": sample["original_user_message"],
        "run_name": run_name,
        "model": model,
        "predicted_score": None,
        "predicted_reason": "",
        "revision_suggestions": "",
        "predicted_revision_suggestions": "",
        "edit_intent": "",
        "predicted_modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "latency_sec": None,
        "tagged_ok": False,
        "strict_json_ok": False,
    }

    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=api_key,
            model=model,
            messages=sample["prompt_messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = parse_label(raw_text)
        ok = (
            parsed["score"] is not None
            and bool(parsed["reason"])
            and bool(parsed["revision_suggestions"])
            and bool(parsed["modified_answer"])
        )
        row.update(
            {
                "predicted_score": parsed["score"],
                "predicted_reason": parsed["reason"],
                "revision_suggestions": parsed["revision_suggestions"],
                "predicted_revision_suggestions": parsed["revision_suggestions"],
                "edit_intent": parsed["revision_suggestions"],
                "predicted_modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "ok": ok,
                "parse_error": parsed["parse_error"],
                "request_error": None,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "tagged_ok": parsed["tagged_ok"],
                "strict_json_ok": parsed["strict_json_ok"],
            }
        )
    except Exception as exc:
        row.update(
            {
                "request_error": str(exc),
                "latency_sec": round(time.time() - started, 4),
            }
        )
    return row


def reorder_output(output_path: Path, samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
    ordered = [completed[sample["sample_id"]] for sample in samples if sample["sample_id"] in completed]
    write_jsonl(output_path, ordered)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    samples = prepare_samples(records)

    base_urls, api_key, model = resolve_runtime(args)
    run_name = args.run_name.strip() or model
    output_path = Path(args.output) if args.output else default_output_path(args, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safe_unlink(output_path, args.overwrite)
    completed = load_completed_predictions(output_path, resume_failed=args.resume_failed)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[input] {input_path}")
    print(f"[output] {output_path}")
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")
    print(f"[run_name] {run_name}")
    print(f"[samples] total={len(samples)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="final_test_generate_api", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    run_one,
                    sample,
                    task_index,
                    args,
                    base_urls,
                    api_key,
                    model,
                    run_name,
                ): sample
                for task_index, sample in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                completed[row["sample_id"]] = row
                with lock:
                    append_jsonl(output_path, row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if args.reorder:
        reorder_output(output_path, samples, completed)

    rows = list(completed.values())
    ok_count = sum(1 for row in rows if row.get("ok") is True)
    tagged_count = sum(1 for row in rows if row.get("tagged_ok") is True)
    parse_fail_count = sum(1 for row in rows if row.get("parse_error"))
    request_fail_count = sum(1 for row in rows if row.get("request_error"))
    print(
        f"[summary] ok={ok_count} tagged_ok={tagged_count} "
        f"parse_fail={parse_fail_count} request_fail={request_fail_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPT outputs for the final tagged test set via an external API."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Input JSONL. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL. Defaults to output-dir/run-name file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="", help="Name used in output filename and metadata. Defaults to model.")
    parser.add_argument("--model", default="", help=f"API model id. Falls back to env {DEFAULT_MODEL_ENV}.")

    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL(s), comma-separated.")
    parser.add_argument("--api-key", default="", help="Explicit API key (bearer token).")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output before running.")
    parser.add_argument("--resume-failed", action="store_true", help="Also skip existing failed rows when resuming.")
    parser.add_argument("--reorder", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--health-check", action="store_true", help="Enable /models health check for loopback URLs.")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()