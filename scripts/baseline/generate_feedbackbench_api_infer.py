#!/usr/bin/env python3
"""Run Feedback-Bench rubric JSONL through an OpenAI-compatible API.

The output schema matches generate_feedbackbench_vllm_infer.py:
predicted_score / predicted_reason / predicted_revision_suggestions /
predicted_modified_answer, plus run metadata. By default outputs are written as
score_generate_<run_name>.jsonl so downstream evaluation scripts can read them
alongside the existing Feedback-Bench score_generate files.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


DEFAULT_INPUT = Path("datasets/Feedback-Bench/data/score_rubric_0_5.jsonl")
DEFAULT_OUTDIR = Path("datasets/Feedback-Bench/data")
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_API_MODE = "auto"

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
    rf"(.*?)\s*(?=(?:\n\s*)?(?:revision suggestions|revision_suggestions|edit intent|modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
REVISION_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:revision suggestions|revision_suggestions|edit intent)\s*{COLON_CLASS}\s*"
    rf"(.*?)\s*(?=(?:\n\s*)?(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}|$)"
)
MODIFIED_RE = re.compile(
    rf"(?is)(?:^|\n)\s*(?:modified answer|modified_answer|revised answer)\s*{COLON_CLASS}\s*(.*)$"
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return cleaned.strip("._-") or "model"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            yield payload


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_base_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        raise ValueError("Empty base URL.")
    return text if text.endswith("/v1") else text + "/v1"


def parse_base_urls(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return [normalize_base_url(DEFAULT_BASE_URL)]
    return [normalize_base_url(item) for item in text.split(",") if item.strip()]


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


def resolve_api_key(args: argparse.Namespace) -> str:
    api_key = normalize_api_key(args.api_key)
    if api_key:
        return api_key
    api_key = normalize_api_key(os.getenv(args.api_key_env, ""))
    if api_key:
        return api_key
    raise ValueError(f"Missing API key. Pass --api-key or set env {args.api_key_env}.")


def resolve_model(args: argparse.Namespace) -> str:
    model = first_non_empty(args.model, os.getenv(args.model_env, ""))
    if not model:
        raise ValueError(f"Missing model. Pass --model or set env {args.model_env}.")
    return model


def build_criteria_text(sample: Dict[str, Any]) -> str:
    lines: List[str] = []
    score_criteria = sample.get("score_criteria")
    if isinstance(score_criteria, dict):
        for key in ["0", "1", "2", "3", "4", "5"]:
            value = score_criteria.get(key) or score_criteria.get(int(key))
            if value:
                lines.append(f"{key}: {value}")
        if lines:
            return "\n".join(lines)

    for key in range(6):
        value = sample.get(f"criteria_{key}")
        if value:
            lines.append(f"{key}: {value}")
    if lines:
        return "\n".join(lines)

    return str(sample.get("full_score_criteria", "") or "")


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are an AI evaluator-and-rewriter.\n"
        "Evaluate the given answer strictly using ONLY the provided evaluation dimension and the complete 0-5 scoring criteria.\n"
        "Then revise the answer to better satisfy ONLY that dimension.\n"
        "Do not add unsupported facts. If an assumption is necessary, state it minimally and explicitly.\n"
        "Output plain text in exactly 1 line using these tags and no numbering:\n"
        "<s>score</s><r>reason</r><rs>revision suggestions</rs><ra>refined answer</ra>\n"
        "Do not include any extra text, JSON, markdown, bullets, or line breaks inside any field."
        "Keep <r> under 80 words, <rs> under 50 words, and <ra> under 90 words."
    )
    user = "\n".join(
        [
            "###Task Description:",
            "You are given a question, a response to evaluate, and one evaluation dimension with its complete 0-5 scoring criteria.",
            "1. Write a score that reflects how well the response satisfies the given criteria.",
            "2. Write feedback that assesses the quality relative to the criteria.",
            "3. Provide concise revision suggestions to improve the response for that dimension.",
            "4. Provide a refined answer that better satisfies the dimension (only change what's needed).",
            "---",
            f"Question: {sample.get('question', '')}",
            f"Answer: {sample.get('answer', '')}",
            f"Evaluation dimension: {sample.get('dimension_name', '')}",
            "Scoring criteria:",
            build_criteria_text(sample),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    if "```" in raw:
        for block in raw.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            if block:
                candidates.append(block)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def parse_int_score(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    if isinstance(value, float) and value.is_integer() and 0 <= value <= 5:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-5]", value.strip()):
        return int(value.strip())
    return None


def parse_output(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip().replace("\r\n", "\n")
    out: Dict[str, Any] = {
        "predicted_score": None,
        "predicted_reason": "",
        "predicted_revision_suggestions": "",
        "revision_suggestions": "",
        "edit_intent": "",
        "predicted_modified_answer": "",
        "raw_output": raw,
        "ok": False,
        "parse_error": None,
        "tagged_ok": False,
        "strict_json_ok": False,
    }

    tagged = TAGGED_LABEL_RE.search(raw)
    if tagged:
        score = parse_int_score(normalize_text(tagged.group("score")))
        reason = normalize_text(tagged.group("reason"))
        revision = normalize_text(tagged.group("revision"))
        modified = normalize_text(tagged.group("modified"))
        ok = score is not None and bool(reason) and bool(revision) and bool(modified)
        out.update(
            {
                "predicted_score": score,
                "predicted_reason": reason,
                "predicted_revision_suggestions": revision,
                "revision_suggestions": revision,
                "edit_intent": revision,
                "predicted_modified_answer": modified,
                "ok": ok,
                "parse_error": None if ok else "Tagged output is missing required fields.",
                "tagged_ok": ok,
            }
        )
        return out

    parsed = extract_json_object(raw)
    if isinstance(parsed, dict):
        score_value = parsed["score"] if "score" in parsed else parsed.get("Score")
        score = parse_int_score(score_value)
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
        out.update(
            {
                "predicted_score": score,
                "predicted_reason": reason,
                "predicted_revision_suggestions": revision,
                "revision_suggestions": revision,
                "edit_intent": revision,
                "predicted_modified_answer": modified,
                "ok": ok,
                "parse_error": None if ok else "Parsed JSON but required fields are missing.",
                "strict_json_ok": ok,
            }
        )
        return out

    score_match = SCORE_LINE_RE.search(raw)
    reason_match = REASON_RE.search(raw)
    revision_match = REVISION_RE.search(raw)
    modified_match = MODIFIED_RE.search(raw)
    score = int(score_match.group(1)) if score_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    revision = normalize_text(revision_match.group(1)) if revision_match else ""
    modified = normalize_text(modified_match.group(1)) if modified_match else ""
    ok = score is not None and bool(reason) and bool(revision) and bool(modified)
    out.update(
        {
            "predicted_score": score,
            "predicted_reason": reason,
            "predicted_revision_suggestions": revision,
            "revision_suggestions": revision,
            "edit_intent": revision,
            "predicted_modified_answer": modified,
            "ok": ok,
            "parse_error": None if ok else "Failed to parse required fields.",
        }
    )
    return out


def extract_text_from_response(payload: Dict[str, Any]) -> str:
    """Extract visible assistant text from common OpenAI-compatible responses."""

    def content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = (
                        item.get("text")
                        or item.get("content")
                        or item.get("output_text")
                    )
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip()

    choices = payload.get("choices") or []
    if choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                # Prefer normal visible answer.
                for key in ["content", "output_text"]:
                    text = content_to_text(message.get(key))
                    if text:
                        return text

                # Some reasoning-model proxies put text here.
                # Only use this as fallback, because it may contain hidden reasoning
                # rather than the final answer.
                for key in ["reasoning_content", "reasoning", "thought"]:
                    text = content_to_text(message.get(key))
                    if text:
                        return text

            text = content_to_text(first.get("text"))
            if text:
                return text

            text = content_to_text(first.get("delta", {}).get("content") if isinstance(first.get("delta"), dict) else "")
            if text:
                return text

    for key in ["text", "output_text", "content"]:
        text = content_to_text(payload.get(key))
        if text:
            return text

    output_text = content_to_text(payload.get("output_text"))
    if output_text:
        return output_text

    output = payload.get("output")
    if isinstance(output, list):
        parts: List[str] = []
        for item in output:
            if isinstance(item, dict):
                text = content_to_text(item.get("content") or item.get("output_text") or item.get("text"))
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()

    return ""


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def post_responses(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    import requests

    payload = {
        "model": model,
        "input": messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "response_format": {"type": "text"},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(
        base_url.rstrip("/") + "/responses",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def post_completion(
    *,
    api_mode: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, Any]:
    if api_mode == "responses":
        return post_responses(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    if api_mode == "chat":
        return post_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    payload = post_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if extract_text_from_response(payload):
        return payload
    return post_responses(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def call_api_with_retries(
    *,
    base_urls: List[str],
    task_index: int,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    api_mode: str,
) -> Tuple[Dict[str, Any], str]:
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        endpoint = base_urls[(task_index + attempt) % len(base_urls)]
        try:
            return (
                post_completion(
                    api_mode=api_mode,
                    base_url=endpoint,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                ),
                endpoint,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(str(last_error))


def prepare_samples(input_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(input_path)):
        if limit is not None and index >= max(0, limit):
            break
        sample = dict(row)
        sample["index"] = index
        samples.append(sample)
    return samples


def load_completed(path: Path, *, resume_failed: bool) -> Dict[str, Dict[str, Any]]:
    completed: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for fallback_index, row in enumerate(read_jsonl(path)):
        sample_id = normalize_text(row.get("sample_id")) or f"index:{row.get('index', fallback_index)}"
        if resume_failed or row.get("ok") is True:
            completed[sample_id] = row
    return completed


def sample_key(sample: Dict[str, Any]) -> str:
    sample_id = normalize_text(sample.get("sample_id"))
    if sample_id:
        return sample_id
    return f"index:{sample.get('index')}"


def run_one(
    sample: Dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: List[str],
    api_key: str,
    model: str,
    run_name: str,
) -> Dict[str, Any]:
    started = time.time()
    row: Dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "index": sample.get("index"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "dimension_name": sample.get("dimension_name"),
        "full_score_criteria": sample.get("full_score_criteria"),
        "score_criteria": sample.get("score_criteria"),
        "run_name": run_name,
        "model": model,
        "endpoint": "",
        "latency": None,
        "request_error": None,
    }
    try:
        response_payload, endpoint = call_api_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=api_key,
            model=model,
            messages=build_messages(sample),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            api_mode=args.api_mode,
        )
        raw_text = extract_text_from_response(response_payload)
        parsed = parse_output(raw_text)
        row.update(parsed)
        row.update(
            {
                "endpoint": endpoint,
                "latency": time.time() - started,
                "raw_response": response_payload if args.include_raw_response else None,
            }
        )
    except Exception as exc:
        row.update(
            {
                "predicted_score": None,
                "predicted_reason": "",
                "predicted_revision_suggestions": "",
                "revision_suggestions": "",
                "edit_intent": "",
                "predicted_modified_answer": "",
                "raw_output": "",
                "ok": False,
                "parse_error": None,
                "request_error": str(exc),
                "latency": time.time() - started,
                "tagged_ok": False,
                "strict_json_ok": False,
            }
        )
    return row


def reorder_output(path: Path, samples: List[Dict[str, Any]], completed: Dict[str, Dict[str, Any]]) -> None:
    ordered = [completed[sample_key(sample)] for sample in samples if sample_key(sample) in completed]
    write_jsonl(path, ordered)


def default_output_path(args: argparse.Namespace, run_name: str) -> Path:
    output_name = args.output_name.strip() or f"score_generate_{safe_name(run_name)}.jsonl"
    return args.outdir / output_name


def run(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_url_raw = first_non_empty(args.base_url, os.getenv(args.base_url_env, ""), DEFAULT_BASE_URL)
    base_urls = parse_base_urls(base_url_raw)
    api_key = resolve_api_key(args)
    model = resolve_model(args)
    run_name = args.run_name.strip() or model
    output_path = args.output.resolve() if args.output is not None else default_output_path(args, run_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = prepare_samples(input_path, args.limit)
    if output_path.exists() and args.overwrite:
        output_path.unlink()
    elif output_path.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --overwrite or --resume.")

    completed = load_completed(output_path, resume_failed=args.resume_failed) if args.resume else {}
    pending = [sample for sample in samples if sample_key(sample) not in completed]

    print(f"[input] {input_path}")
    print(f"[output] {output_path}")
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")
    print(f"[run_name] {run_name}")
    print(f"[samples] total={len(samples)} completed={len(completed)} pending={len(pending)}")
    print(f"[workers] {args.workers}")

    ok_count = sum(1 for row in completed.values() if row.get("ok") is True)
    request_error_count = sum(1 for row in completed.values() if row.get("request_error"))
    parse_fail_count = sum(1 for row in completed.values() if row.get("parse_error"))
    lock = threading.Lock()
    batch: List[Dict[str, Any]] = []

    progress = tqdm(total=len(pending), desc="Feedback-Bench API", ncols=120) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, sample, task_index, args, base_urls, api_key, model, run_name): sample
                for task_index, sample in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                completed[sample_key(row)] = row
                batch.append(row)

                if row.get("request_error"):
                    request_error_count += 1
                elif row.get("parse_error"):
                    parse_fail_count += 1
                elif row.get("ok") is True:
                    ok_count += 1

                if len(batch) >= args.flush_every:
                    with lock:
                        append_jsonl(output_path, batch)
                    batch = []

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(ok=ok_count, request_error=request_error_count, parse_fail=parse_fail_count)
                else:
                    done = ok_count + request_error_count + parse_fail_count
                    if done % 100 == 0 or done == len(samples):
                        print(
                            f"[progress] {done}/{len(samples)} "
                            f"ok={ok_count} request_error={request_error_count} parse_fail={parse_fail_count}"
                        )
    finally:
        if progress is not None:
            progress.close()

    if batch:
        with lock:
            append_jsonl(output_path, batch)

    if args.reorder:
        reorder_output(output_path, samples, completed)

    rows = list(completed.values())
    ok_count = sum(1 for row in rows if row.get("ok") is True)
    request_error_count = sum(1 for row in rows if row.get("request_error"))
    parse_fail_count = sum(1 for row in rows if row.get("parse_error"))
    print(
        f"[summary] total={len(samples)} ok={ok_count} "
        f"request_error={request_error_count} parse_fail={parse_fail_count}"
    )
    print(f"Wrote outputs to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Feedback-Bench through an OpenAI-compatible API.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output", type=Path, default=None, help="Full output path. Overrides --outdir/--output-name.")
    parser.add_argument("--output-name", default="", help="Output filename. Defaults to score_generate_<run_name>.jsonl.")
    parser.add_argument("--run-name", default="", help="Name used in output filename and row metadata. Defaults to model.")

    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL(s), comma-separated.")
    parser.add_argument("--base-url-env", default=DEFAULT_BASE_URL_ENV)
    parser.add_argument("--model", default="", help=f"API model id. Falls back to env {DEFAULT_MODEL_ENV}.")
    parser.add_argument("--model-env", default=DEFAULT_MODEL_ENV)
    parser.add_argument("--api-key", default="", help="Explicit API key.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument(
        "--api-mode",
        choices=["auto", "chat", "responses"],
        default=DEFAULT_API_MODE,
        help="API mode to use. auto tries chat first, then responses if empty.",
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--include-raw-response", action="store_true")

    parser.add_argument("--overwrite", action="store_true", help="Delete existing output before running.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output file.")
    parser.add_argument("--resume-failed", action="store_true", help="When resuming, also skip existing failed rows.")
    parser.add_argument("--reorder", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
