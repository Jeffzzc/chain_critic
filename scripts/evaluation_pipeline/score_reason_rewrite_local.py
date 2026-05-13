#!/usr/bin/env python
"""Score, explain, and rewrite QA answers with OpenAI-compatible endpoints."""

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

from pipeline_common import (
    COLON_CLASS,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    detect_complete_score_range,
    extract_question_answer,
    extract_json_object,
    first_non_empty,
    format_score_criteria,
    is_loopback_url,
    load_jsonl,
    normalize_score_criteria,
    normalize_text,
    parse_int_score,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
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


SCORE_LINE_RE = re.compile(rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$")
REASON_RE = re.compile(
    rf"(?is)reason\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?(?:revision suggestions|edit intent|modified answer|revised answer)\s*{COLON_CLASS}|$)"
)
REVISION_RE = re.compile(
    rf"(?is)(?:revision suggestions|edit intent)\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?(?:modified answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(rf"(?is)(?:modified answer|revised answer)\s*{COLON_CLASS}\s*(.*)$")

JSON_CODE_BLOCK_RE = re.compile(r"(?is)^\s*```(?:json)?\s*(.*?)\s*```\s*$")
JSON_SCORE_RE = re.compile(r'(?is)["\']?score["\']?\s*:\s*([0-5])')
JSON_REASON_RE = re.compile(r'(?is)["\']?reason["\']?\s*:\s*"((?:\\.|[^"\\])*)"')
JSON_REVISION_RES = [
    re.compile(r'(?is)["\']?revision_suggestions["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
    re.compile(r'(?is)["\']?edit_intent["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
    re.compile(r'(?is)["\']?revision suggestions["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
]
JSON_MODIFIED_RES = [
    re.compile(r'(?is)["\']?modified_answer["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
    re.compile(r'(?is)["\']?revised_answer["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
    re.compile(r'(?is)["\']?better_answer["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
    re.compile(r'(?is)["\']?modified answer["\']?\s*:\s*"((?:\\.|[^"\\])*)"'),
]


SYSTEM_PROMPT = (
    "You are an expert answer evaluator and rewriter.\n"
    "Your task is to evaluate a candidate answer to a given question under the provided "
    "evaluation dimension and complete 0-5 scoring criteria, then rewrite the candidate answer "
    "into a stronger answer optimized for the same evaluation dimension.\n\n"

    "Use ONLY the provided question, candidate answer, evaluation dimension, and scoring criteria. "
    "Do not introduce unsupported facts, external knowledge, or assumptions beyond the given context. "
    "If the question is underspecified, make only the minimum necessary assumption explicit in the rewritten answer.\n\n"

    "Evaluation requirements:\n"
    "1. Assign one integer score from 0 to 5 according to the provided scoring criteria.\n"
    "2. The reason must be concise, concrete, and based directly on the rubric. "
    "It should identify the candidate answer's strengths, weaknesses, missing elements, "
    "unsupported claims, or failures to satisfy the evaluation dimension. "
    "When pointing out a concrete problem, mark it with the prefix \"error:\".\n"
    "3. The revision_suggestions field must give actionable edits that directly address "
    "the problems identified in the reason and explain how to better satisfy the score-5 criterion.\n"
    "4. The modified_answer must be a complete improved answer to the original question, "
    "optimized for the same evaluation dimension, and must address the identified errors without adding unsupported facts.\n\n"

    "Return strict JSON only. Do not include markdown, explanations, comments, or extra text. "
    "Use exactly this schema:\n"
    "{\"score\": 1, \"reason\": \"...\", \"revision_suggestions\": \"...\", \"modified_answer\": \"...\"}"
)


def strip_code_fence(text: str) -> str:
    match = JSON_CODE_BLOCK_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def try_load_json_dict(text: str) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def try_repair_json_dict(text: str) -> Optional[dict[str, Any]]:
    stripped = text.strip()
    if not stripped or not stripped.startswith("{"):
        return None

    candidates: list[str] = []

    if not stripped.endswith("}"):
        candidates.append(stripped + "}")

    brace_gap = stripped.count("{") - stripped.count("}")
    if brace_gap > 0:
        candidates.append(stripped + ("}" * brace_gap))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        parsed = try_load_json_dict(candidate)
        if parsed is not None:
            return parsed
    return None


def decode_json_string_fragment(fragment: str) -> str:
    text = str(fragment or "").strip()
    if not text:
        return ""

    try:
        return normalize_text(json.loads(f'"{text}"'))
    except Exception:
        pass

    # 不再用 unicode_escape，避免把 LaTeX 里的反斜杠误当成非法转义
    text = (
        text.replace('\\"', '"')
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
    )
    return normalize_text(text)

def extract_json_like_fields(text: str) -> dict[str, Any]:
    score: Optional[int] = None
    reason = ""
    revision_suggestions = ""
    modified_answer = ""

    score_match = JSON_SCORE_RE.search(text)
    if score_match:
        score = parse_int_score(score_match.group(1))

    reason_match = JSON_REASON_RE.search(text)
    if reason_match:
        reason = decode_json_string_fragment(reason_match.group(1))

    for pattern in JSON_REVISION_RES:
        revision_match = pattern.search(text)
        if revision_match:
            revision_suggestions = decode_json_string_fragment(revision_match.group(1))
            break

    for pattern in JSON_MODIFIED_RES:
        modified_match = pattern.search(text)
        if modified_match:
            modified_answer = decode_json_string_fragment(modified_match.group(1))
            break

    return {
        "score": score,
        "reason": reason,
        "revision_suggestions": revision_suggestions,
        "modified_answer": modified_answer,
    }


def parse_model_output(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip().replace("\r\n", "\n")
    cleaned_text = strip_code_fence(raw_text)

    parsed = extract_json_object(cleaned_text)
    strict_json_ok = parsed is not None

    if parsed is None:
        parsed = try_load_json_dict(cleaned_text)

    if parsed is None:
        parsed = try_repair_json_dict(cleaned_text)

    if parsed is not None:
        score = parse_int_score(
            parsed.get("score")
            or parsed.get("Score")
        )
        reason = normalize_text(
            parsed.get("reason")
            or parsed.get("Reason")
        )
        revision_suggestions = normalize_text(
            parsed.get("revision_suggestions")
            or parsed.get("edit_intent")
            or parsed.get("Revision Suggestions")
            or parsed.get("Edit Intent")
        )
        modified_answer = normalize_text(
            parsed.get("modified_answer")
            or parsed.get("revised_answer")
            or parsed.get("better_answer")
            or parsed.get("Modified Answer")
            or parsed.get("Revised Answer")
            or parsed.get("Better Answer")
        )
        ok = score is not None and bool(reason) and bool(revision_suggestions) and bool(modified_answer)
        return {
            "score": score,
            "reason": reason,
            "revision_suggestions": revision_suggestions,
            "modified_answer": modified_answer,
            "raw_output": raw_text,
            "parse_error": None if ok else "Parsed JSON but some required fields are missing.",
            "strict_json_ok": strict_json_ok and ok,
        }

    json_like = extract_json_like_fields(cleaned_text)
    if json_like["score"] is not None or json_like["reason"] or json_like["modified_answer"]:
        ok = (
            json_like["score"] is not None
            and bool(json_like["reason"])
            and bool(json_like["revision_suggestions"])
            and bool(json_like["modified_answer"])
        )
        return {
            "score": json_like["score"],
            "reason": json_like["reason"],
            "revision_suggestions": json_like["revision_suggestions"],
            "modified_answer": json_like["modified_answer"],
            "raw_output": raw_text,
            "parse_error": None if ok else "Recovered partial fields from malformed JSON-like output.",
            "strict_json_ok": False,
        }

    score: int | None = None
    reason = ""
    revision_suggestions = ""
    modified_answer = ""
    parse_error: str | None = None

    score_match = SCORE_LINE_RE.search(cleaned_text)
    if score_match:
        score = int(score_match.group(1))
    else:
        fallback_score = re.search(rf"(?i)\bscore\s*{COLON_CLASS}\s*([0-5])", cleaned_text)
        if fallback_score:
            score = int(fallback_score.group(1))
        else:
            parse_error = "Failed to parse score."

    reason_match = REASON_RE.search(cleaned_text)
    revision_match = REVISION_RE.search(cleaned_text)
    modified_match = MODIFIED_RE.search(cleaned_text)
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


def build_sample_id(record: dict[str, Any], index: int) -> str:
    explicit = first_non_empty(record.get("sample_id"), record.get("id"), record.get("unique_id"))
    if explicit:
        return explicit
    return f"row:{index}"


def answer_record_id(record: dict[str, Any], id_field: str) -> str:
    if id_field:
        return normalize_text(record.get(id_field))
    return first_non_empty(record.get("sample_id"), record.get("id"), record.get("unique_id"))


def load_answer_overrides(
    path: Path | None,
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
                "source_record": record,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt_with_range(
                            question,
                            answer,
                            dimension_name,
                            criteria_text,
                            score_range,
                        ),
                    },
                ],
            }
        )
    return samples


def build_user_prompt_with_range(
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
        f"1. Assign one integer score from {score_range_label} to the candidate answer based on the given evaluation dimension and the provided rubric.\n"
        "2. Give a concise reason grounded in the score criteria.\n"
        "3. Give executable revision suggestions that directly state how to fix the answer.\n"
        "4. Rewrite a better answer to the original question that would satisfy the dimension better.\n\n"
        "Return strict JSON only."
    )


def build_user_prompt(question: str, answer: str, dimension_name: str, criteria_text: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{answer}\n\n"
        "Evaluation Dimension:\n"
        f"{dimension_name}\n\n"
        "Score Criteria (0-5):\n"
        f"{criteria_text}\n\n"
        "Tasks:\n"
        "1. Assign one integer score from 0 to 5 to the candidate answer based on the given evaluation dimension and the provided 0–5 scoring rubric.\n"
        "2. Give a concise reason grounded in the score criteria.\n"
        "3. Give executable revision suggestions that directly state how to fix the answer.\n"
        "4. Rewrite a better answer to the original question that would satisfy the dimension better.\n\n"
        "Return strict JSON only."
    )


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
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"] and score_in_range,
            }
        )
    except Exception as exc:
        row.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return row


def reorder_output(output_path: Path, samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
    ordered = [completed[sample["sample_id"]] for sample in samples if sample["sample_id"] in completed]
    from pipeline_common import write_jsonl

    write_jsonl(output_path, ordered)


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


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("Empty base URL.")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def split_base_url_values(values: list[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if text:
                resolved.append(normalize_base_url(text))
    return resolved


def dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def resolve_base_urls(args: argparse.Namespace) -> tuple[list[str], bool]:
    direct_values: list[str] = []
    if args.base_url:
        direct_values.append(args.base_url)
    if args.base_urls:
        direct_values.extend(args.base_urls)

    if direct_values:
        return dedupe_preserve_order(split_base_url_values(direct_values)), True

    return build_base_urls(args.base_url_template, parse_ports(args.ports)), False


def normalize_api_key(value: str) -> str:
    token = str(value or "").strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def resolve_api_key(args: argparse.Namespace, base_urls: list[str]) -> str:
    explicit = normalize_api_key(args.api_key)
    if explicit and explicit != DEFAULT_API_KEY:
        return explicit

    env_token = normalize_api_key(os.getenv(args.api_key_env, ""))
    if env_token:
        return env_token

    fallback_token = normalize_api_key(os.getenv("OPENAI_BEARER_TOKEN", ""))
    if fallback_token:
        return fallback_token

    if all(is_loopback_url(base_url) for base_url in base_urls):
        return explicit or DEFAULT_API_KEY

    raise ValueError(
        "Missing API key for non-local endpoint. Pass --api-key or set "
        f"env {args.api_key_env}."
    )


def should_health_check(args: argparse.Namespace, base_urls: list[str], direct_base_url: bool) -> bool:
    if args.skip_health_check:
        return False
    if args.health_check:
        return True
    if direct_base_url and not all(is_loopback_url(base_url) for base_url in base_urls):
        return False
    return True


def resolve_runtime(args: argparse.Namespace) -> tuple[list[str], str, str]:
    base_urls, direct_base_url = resolve_base_urls(args)
    api_key = resolve_api_key(args, base_urls)

    if should_health_check(args, base_urls, direct_base_url):
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)

    if args.model.strip():
        model = args.model.strip()
    elif direct_base_url and not all(is_loopback_url(base_url) for base_url in base_urls):
        raise ValueError("External API endpoint requires --model because unauthenticated /models fetch is not reliable.")
    else:
        model = resolve_model(args.model, base_urls, args.health_check_timeout)

    return base_urls, api_key, model


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

    base_urls, api_key, model = resolve_runtime(args)
    args.api_key = api_key

    safe_unlink(output_path, args.overwrite)
    completed = load_completed_predictions(output_path)
    pending = [sample for sample in samples if sample["sample_id"] not in completed]

    print(f"[input] {input_path}")
    if args.answers:
        print(f"[answers] {Path(args.answers)} overrides={len(answer_overrides)}")
    print(f"[output] {output_path}")
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")
    print(f"[samples] total={len(samples)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc="score_rewrite", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, sample, index, args, base_urls, model): sample
                for index, sample in enumerate(pending)
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
    
    ok_count = sum(1 for row in completed.values() if row.get("ok"))
    strict_json_count = sum(1 for row in completed.values() if row.get("strict_json_ok"))
    parse_fail_count = sum(1 for row in completed.values() if row.get("parse_error"))
    request_fail_count = sum(1 for row in completed.values() if row.get("request_error"))

    print(
        f"[summary] ok={ok_count} strict_json_ok={strict_json_count} "
        f"parse_fail={parse_fail_count} request_fail={request_fail_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local QA score+reason+rewrite generation from complete score rubrics.")
    parser.add_argument(
        "--input",
        type=Path,
        default="datasets/0-5/score_criteria_0_5.jsonl",
        help="Flat rubric JSONL (supports complete 0-5 or 1-5 criteria).",
    )
    parser.add_argument("--output", type=Path, default="datasets/0-5/score_reason_rewrite_original.jsonl", help="Prediction output JSONL.")
    parser.add_argument(
        "--answers",
        type=Path,
        default=None,
        help="Optional QA model answer JSONL. When set, candidate answers override the rubric file answer by id.",
    )
    parser.add_argument(
        "--answer-id-field",
        type=str,
        default="",
        help="ID field in --answers. Defaults to sample_id/id/unique_id auto-detection.",
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default="",
        help="Answer field in --answers. Defaults to common answer/response/output fields.",
    )
    parser.add_argument(
        "--answer-question-field",
        type=str,
        default="",
        help="Optional question field in --answers, used only when present.",
    )
    parser.add_argument("--skip-missing", default=True, action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reorder", action="store_true", help="Rewrite output in input order after completion.")

    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Single OpenAI-compatible base URL, e.g. https://api.openai.com/v1 or https://vendor.example.com/v1.",
    )
    parser.add_argument(
        "--base-urls",
        nargs="*",
        default=[],
        help="One or more OpenAI-compatible base URLs. Comma-separated values are also accepted.",
    )
    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8001-8007"], help="Ports like: 8000 8001 or 8000-8003.")
    parser.add_argument("--model", type=str, default="", help="Model id. If empty, fetch from /models.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable used for external API key when --api-key is not provided.",
    )
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Force /models health check. By default, external direct --base-url endpoints skip health check.",
    )
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
