#!/usr/bin/env python
"""Score, explain, and rewrite QA answers through an external OpenAI-compatible API.

This is the API-only counterpart of score_reason_rewrite_local.py. It does not
support local vLLM port fan-out or health checks; every request goes to
--base-url/chat/completions with an external API key.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional
from urllib import request as urllib_request

from pipeline_common import (
    COLON_CLASS,
    append_jsonl,
    detect_complete_score_range,
    extract_json_object,
    extract_question_answer,
    first_non_empty,
    format_score_criteria,
    load_jsonl,
    normalize_score_criteria,
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


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "cortex-5")
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

SCORE_LINE_RE = re.compile(rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$")
REASON_RE = re.compile(
    rf"(?is)reason\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?(?:revision suggestions|edit intent|modified answer|revised answer)\s*{COLON_CLASS}|$)"
)
REVISION_RE = re.compile(
    rf"(?is)(?:revision suggestions|edit intent)\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?(?:modified answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(rf"(?is)(?:modified answer|revised answer)\s*{COLON_CLASS}\s*(.*)$")

SYSTEM_PROMPT = (
    "You are a strict answer evaluation and revision model.\n"
    "Evaluate the candidate answer using ONLY the provided question, evaluation dimension, "
    "and complete score criteria. Then rewrite the candidate answer into a better "
    "answer for the original question, optimized for the same evaluation dimension.\n"
    "Do not introduce unsupported facts. If the question is underspecified, make the "
    "minimum necessary assumption explicit.\n"
    "Return strict JSON only, with this exact schema:\n"
    '{"score": 1, "reason": "...", "revision_suggestions": "...", "modified_answer": "..."}\n'
    "The score must be an integer within the provided score range. The reason must be concise and based on the rubric. "
    "The revision_suggestions field must give executable edits that directly address the stated problems. "
    "The modified_answer must be a complete improved answer to the question."
)


def normalize_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Empty --base-url.")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def normalize_api_key(value: Any) -> str:
    token = str(value or "").strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def resolve_api_key(args: argparse.Namespace) -> str:
    api_key = normalize_api_key(args.api_key)
    if api_key:
        return api_key
    api_key = normalize_api_key(os.getenv(args.api_key_env, ""))
    if api_key:
        return api_key
    api_key = normalize_api_key(os.getenv("OPENAI_BEARER_TOKEN", ""))
    if api_key:
        return api_key
    raise ValueError(f"Missing API key. Pass --api-key or set env {args.api_key_env}.")


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned from {url}")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def call_chat_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    retries: int,
    retry_sleep: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(str(last_error))


def parse_model_output(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip().replace("\r\n", "\n")
    parsed = extract_json_object(raw_text)
    if parsed is not None:
        score = parse_int_score(parsed.get("score"))
        reason = normalize_text(parsed.get("reason"))
        revision_suggestions = normalize_text(
            parsed.get("revision_suggestions")
            or parsed.get("edit_intent")
            or parsed.get("Revision Suggestions")
            or parsed.get("Edit Intent")
        )
        modified_answer = normalize_text(
            parsed.get("modified_answer") or parsed.get("revised_answer") or parsed.get("better_answer")
        )
        return {
            "score": score,
            "reason": reason,
            "revision_suggestions": revision_suggestions,
            "modified_answer": modified_answer,
            "raw_output": raw_text,
            "parse_error": None
            if score is not None and reason and revision_suggestions and modified_answer
            else "Parsed JSON but some required fields are missing.",
            "strict_json_ok": score is not None and bool(reason) and bool(revision_suggestions) and bool(modified_answer),
        }

    score: int | None = None
    reason = ""
    revision_suggestions = ""
    modified_answer = ""
    parse_error: str | None = None

    score_match = SCORE_LINE_RE.search(raw_text)
    if score_match:
        score = int(score_match.group(1))
    else:
        fallback_score = re.search(rf"(?i)\bscore\s*{COLON_CLASS}\s*([0-5])", raw_text)
        if fallback_score:
            score = int(fallback_score.group(1))
        else:
            parse_error = "Failed to parse score."

    reason_match = REASON_RE.search(raw_text)
    revision_match = REVISION_RE.search(raw_text)
    modified_match = MODIFIED_RE.search(raw_text)
    if reason_match:
        reason = normalize_text(reason_match.group(1))
    if revision_match:
        revision_suggestions = normalize_text(revision_match.group(1))
    if modified_match:
        modified_answer = normalize_text(modified_match.group(1))

    return {
        "score": score,
        "reason": reason,
        "revision_suggestions": revision_suggestions,
        "modified_answer": modified_answer,
        "raw_output": raw_text,
        "parse_error": parse_error,
        "strict_json_ok": False,
    }


def answer_record_id(record: dict[str, Any], id_field: str) -> str:
    if id_field:
        return normalize_text(record.get(id_field))
    return first_non_empty(record.get("sample_id"), record.get("id"), record.get("unique_id"))


def load_answer_overrides(
    path: Optional[Path],
    *,
    id_field: str,
    answer_field: str,
    question_field: str,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Answer override file not found: {path}")

    overrides: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(load_jsonl(path)):
        key = answer_record_id(record, id_field)
        if not key:
            raise ValueError(f"Answer override record {index} has no usable id. Set --answer-id-field if needed.")
        question, answer = extract_question_answer(
            record,
            question_field=question_field,
            answer_field=answer_field,
        )
        if not answer:
            raise ValueError(f"Answer override record {index} has no usable answer. Set --answer-field if needed.")
        overrides[key] = {
            "question": question,
            "answer": answer,
            "source_record": record,
        }
    return overrides


def base_sample_keys(record: dict[str, Any]) -> list[str]:
    keys = [
        first_non_empty(record.get("parent_sample_id")),
        first_non_empty(record.get("unique_id")),
        first_non_empty(record.get("source_unique_id")),
    ]
    sample_id = first_non_empty(record.get("sample_id"), record.get("id"))
    if sample_id:
        keys.append(sample_id)
        if ":dimension:" in sample_id:
            keys.append(sample_id.split(":dimension:", 1)[0])
    return [key for index, key in enumerate(keys) if key and key not in keys[:index]]


def resolve_candidate_answer(
    record: dict[str, Any],
    answer_overrides: dict[str, dict[str, Any]],
) -> tuple[str, str, Optional[dict[str, Any]], str]:
    for key in base_sample_keys(record):
        override = answer_overrides.get(key)
        if override is None:
            continue
        question = normalize_text(override.get("question")) or normalize_text(record.get("question"))
        answer = normalize_text(override.get("answer"))
        return question, answer, override.get("source_record"), key
    return normalize_text(record.get("question")), normalize_text(record.get("answer")), None, ""


def build_sample_id(record: dict[str, Any], index: int) -> str:
    explicit = first_non_empty(record.get("sample_id"), record.get("id"), record.get("unique_id"))
    if explicit:
        return explicit
    return f"row:{index}"


def build_user_prompt(
    question: str,
    answer: str,
    dimension_name: str,
    criteria_text: str,
    score_range: list[str],
) -> str:
    score_range_label = f"{score_range[0]}-{score_range[-1]}"
    return (
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{answer}\n\n"
        "Evaluation Dimension:\n"
        f"{dimension_name}\n\n"
        f"Score Criteria ({score_range_label}):\n"
        f"{criteria_text}\n\n"
        "Tasks:\n"
        f"1. Assign one integer score from {score_range_label} to the candidate answer.\n"
        "2. Give a concise reason grounded in the score criteria.\n"
        "3. Give executable revision suggestions that directly state how to fix the answer.\n"
        "4. Rewrite a better answer to the original question that would satisfy the dimension better.\n\n"
        "Return strict JSON only."
    )


def prepare_samples(
    records: list[dict[str, Any]],
    skip_missing: bool,
    answer_overrides: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question, answer, answer_source_record, answer_source_id = resolve_candidate_answer(record, answer_overrides)
        dimension_name = first_non_empty(record.get("dimension_name"), record.get("evaluation_dimension"))
        full_score_criteria = normalize_text(record.get("full_score_criteria"))
        score_criteria = normalize_score_criteria(record, full_score_criteria)
        score_range = detect_complete_score_range(score_criteria)
        criteria_text = format_score_criteria(score_criteria, record)

        missing = []
        if not question:
            missing.append("question")
        if not answer:
            missing.append("answer")
        if not dimension_name:
            missing.append("dimension_name/evaluation_dimension")
        if not score_range:
            missing.append("score_criteria (need complete 0-5 or 1-5)")
        if missing:
            if skip_missing:
                continue
            raise ValueError(f"Record {index} missing required fields: {missing}")

        sample_id = build_sample_id(record, index)
        samples.append(
            {
                "sample_id": sample_id,
                "index": index,
                "unique_id": record.get("unique_id"),
                "parent_sample_id": record.get("parent_sample_id"),
                "subject": record.get("subject"),
                "level": record.get("level"),
                "question": question,
                "answer": answer,
                "rubric_answer": normalize_text(record.get("answer")),
                "reference_answer": record.get("reference_answer"),
                "reference_solution": record.get("reference_solution"),
                "answer_source_id": answer_source_id,
                "answer_source_record": answer_source_record,
                "dimension_name": dimension_name,
                "evaluation_dimension": dimension_name,
                "full_score_criteria": full_score_criteria,
                "score_criteria": score_criteria,
                "score_range": score_range,
                "criteria_text": criteria_text,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(question, answer, dimension_name, criteria_text, score_range),
                    },
                ],
            }
        )
    return samples


def run_one(
    sample: dict[str, Any],
    args: argparse.Namespace,
    base_url: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    started = time.time()
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "index": sample["index"],
        "unique_id": sample.get("unique_id"),
        "parent_sample_id": sample.get("parent_sample_id"),
        "subject": sample.get("subject"),
        "level": sample.get("level"),
        "question": sample["question"],
        "answer": sample["answer"],
        "rubric_answer": sample.get("rubric_answer"),
        "reference_answer": sample.get("reference_answer"),
        "reference_solution": sample.get("reference_solution"),
        "answer_source_id": sample.get("answer_source_id"),
        "dimension_name": sample["dimension_name"],
        "evaluation_dimension": sample["evaluation_dimension"],
        "full_score_criteria": sample["full_score_criteria"],
        "score_criteria": sample["score_criteria"],
        "score_range": sample.get("score_range"),
        "criteria_text": sample["criteria_text"],
        "predicted_score": None,
        "predicted_reason": "",
        "revision_suggestions": "",
        "edit_intent": "",
        "predicted_modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": base_url,
        "model": model,
        "latency_sec": None,
        "strict_json_ok": False,
    }
    try:
        raw_text = call_chat_with_retries(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=sample["messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = parse_model_output(raw_text)
        allowed_scores = {int(score) for score in (sample.get("score_range") or [])}
        score_in_range = parsed["score"] is not None and parsed["score"] in allowed_scores
        parse_error = parsed["parse_error"]
        if parsed["score"] is not None and not score_in_range:
            parse_error = f"Score {parsed['score']} is outside allowed range {sorted(allowed_scores)}."
        ok = (
            score_in_range
            and bool(parsed["reason"])
            and bool(parsed["revision_suggestions"])
            and bool(parsed["modified_answer"])
        )
        row.update(
            {
                "predicted_score": parsed["score"],
                "predicted_reason": parsed["reason"],
                "revision_suggestions": parsed["revision_suggestions"],
                "edit_intent": parsed["revision_suggestions"],
                "predicted_modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "ok": ok,
                "parse_error": parse_error,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"] and score_in_range,
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
        if row.get("ok") is not True:
            continue
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            completed[sample_id] = row
    return completed


def reorder_output(output_path: Path, samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
    ordered = [completed[sample["sample_id"]] for sample in samples if sample["sample_id"] in completed]
    write_jsonl(output_path, ordered)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    answer_overrides = load_answer_overrides(
        Path(args.answers) if args.answers else None,
        id_field=args.answer_id_field,
        answer_field=args.answer_field,
        question_field=args.answer_question_field,
    )
    samples = prepare_samples(records, args.skip_missing, answer_overrides)

    base_url = normalize_base_url(args.base_url or os.getenv("OPENAI_BASE_URL", "") or DEFAULT_BASE_URL)
    model = args.model or os.getenv("OPENAI_MODEL", "") or DEFAULT_MODEL
    api_key = resolve_api_key(args)

    safe_unlink(output_path, args.overwrite)
    completed = load_completed_predictions(output_path) if args.resume else {}
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[input] {input_path}")
    if args.answers:
        print(f"[answers] {Path(args.answers)} overrides={len(answer_overrides)}")
    print(f"[output] {output_path}")
    print(f"[base_url] {base_url}")
    print(f"[model] {model}")
    print(f"[samples] total={len(samples)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="openai_score_rewrite", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, sample, args, base_url, api_key, model): sample
                for sample in pending
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run API-only QA score+reason+rewrite generation through an external OpenAI-compatible API."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="datasets/MATH500/0-5_rubric.jsonl",
        help="Flat rubric JSONL (supports complete 0-5 or 1-5 criteria).",
    )
    parser.add_argument("--output", type=Path, default="datasets/MATH500/final/cortex-5.jsonl", help="Prediction output JSONL.")
    parser.add_argument("--answers", type=Path, default=None, help="Optional QA model answer JSONL.")
    parser.add_argument("--answer-id-field", type=str, default="", help="ID field in --answers.")
    parser.add_argument("--answer-field", type=str, default="", help="Answer field in --answers.")
    parser.add_argument("--answer-question-field", type=str, default="", help="Optional question field in --answers.")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--reorder", action="store_true", help="Rewrite output in input order after completion.")

    parser.add_argument("--base-url", type=str, default="", help=f"OpenAI-compatible base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--model", type=str, default="", help=f"Model name. Default: env OPENAI_MODEL or {DEFAULT_MODEL}.")
    parser.add_argument("--api-key", type=str, default="", help="Explicit API key.")
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV, help="API key environment variable.")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
