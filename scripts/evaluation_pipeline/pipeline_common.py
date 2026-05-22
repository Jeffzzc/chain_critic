#!/usr/bin/env python
"""Shared helpers for the generic evaluation pipeline scripts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse


DEFAULT_PORTS = tuple(range(8000, 8004))
DEFAULT_BASE_URL_TEMPLATE = "http://127.0.0.1:{port}/v1"
DEFAULT_API_KEY = "EMPTY"

COLON_CLASS = r"[:\uFF1A]"

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question", "prompt", "instruction", "query", "problem"),
    "answer": ("answer", "response", "candidate answer", "model answer", "assistant answer"),
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_no}: {path}")
            rows.append(payload)
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_no}: {path}")
            yield payload


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ports(values: list[str] | None) -> list[int]:
    if not values:
        return list(DEFAULT_PORTS)

    ports: list[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            ports.extend(range(start, end + step, step))
        else:
            ports.append(int(text))
    return ports


def build_base_urls(template: str, ports: list[int]) -> list[str]:
    return [template.format(port=port).rstrip("/") for port in ports]


def is_loopback_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


@contextmanager
def suspended_proxy_env_if_loopback(url: str) -> Any:
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    if not is_loopback_url(url):
        yield
        return

    old_values = {key: os.environ.get(key) for key in proxy_keys}
    try:
        for key in proxy_keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def models_endpoint_ready(base_url: str, timeout_seconds: int) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with suspended_proxy_env_if_loopback(url):
            with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
                return response.status == 200
    except (OSError, urllib_error.URLError, TimeoutError, ValueError):
        return False


def wait_for_servers(base_urls: list[str], timeout_seconds: int, interval_seconds: float) -> None:
    for base_url in base_urls:
        while not models_endpoint_ready(base_url, timeout_seconds):
            print(f"Waiting for {base_url.rstrip('/')}/models ...", end="\r")
            time.sleep(interval_seconds)
    print("All endpoints are ready.                    ")


def fetch_model_id(base_url: str, timeout_seconds: int) -> str:
    url = base_url.rstrip("/") + "/models"
    with suspended_proxy_env_if_loopback(url):
        with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    return ""


def resolve_model(model: str, base_urls: list[str], timeout_seconds: int) -> str:
    if model.strip():
        return model.strip()
    for base_url in base_urls:
        try:
            resolved = fetch_model_id(base_url, timeout_seconds)
        except Exception:
            resolved = ""
        if resolved:
            return resolved
    raise ValueError("No --model was provided and no model id could be fetched from /models.")


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with suspended_proxy_env_if_loopback(url):
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned from {url}")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "").strip()


def post_embedding_request(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout_seconds: int | float,
) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = {
        "model": model,
        "input": texts,
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
    with suspended_proxy_env_if_loopback(url):
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

    data = response_payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Embedding response has no data list from {base_url}")
    sorted_data = sorted(data, key=lambda item: int(item.get("index", 0)))
    embeddings = [item.get("embedding") for item in sorted_data]
    if len(embeddings) != len(texts):
        raise ValueError(f"Embedding count mismatch from {base_url}: expected {len(texts)}, got {len(embeddings)}")
    if not all(isinstance(item, list) for item in embeddings):
        raise ValueError(f"Embedding response contains invalid embedding values from {base_url}")
    return embeddings  # type: ignore[return-value]


def embed_texts_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout_seconds: int | float,
    retries: int,
    retry_sleep: float,
) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return post_embedding_request(
                base_url=base_url,
                api_key=api_key,
                model=model,
                texts=texts,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
    raise RuntimeError(f"Embedding request failed after {retries + 1} attempts at {base_url}: {last_error}")


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
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
    return dot / ((left_norm ** 0.5) * (right_norm ** 0.5))


def call_chat_with_retries(
    *,
    base_urls: list[str],
    task_index: int,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    retries: int,
    retry_sleep: float,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[str, str]:
    last_error: Exception | None = None
    endpoint_count = max(1, len(base_urls))
    for attempt in range(retries + 1):
        base_url = base_urls[(task_index + attempt) % endpoint_count]
        try:
            text = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                chat_template_kwargs=chat_template_kwargs,
            )
            return text, base_url
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(str(last_error))


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    candidates = [raw]
    if "```" in raw:
        for block in raw.split("```"):
            block = block.strip()
            if not block:
                continue
            if block.lower().startswith("json"):
                block = block[4:].strip()
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
        if start < 0 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def canonicalize_header_name(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower())


def alias_to_regex(alias: str) -> str:
    tokens = [re.escape(token) for token in re.split(r"[\s_\-]+", alias.strip()) if token]
    return r"[\s_\-]*".join(tokens)


def extract_sections_by_headers(text: str, header_map: dict[str, tuple[str, ...]]) -> dict[str, str]:
    if not text:
        return {}

    header_to_key: dict[str, str] = {}
    pattern_parts: list[str] = []
    for key, aliases in header_map.items():
        for alias in aliases:
            alias_pattern = alias_to_regex(alias)
            if not alias_pattern:
                continue
            header_to_key[canonicalize_header_name(alias)] = key
            pattern_parts.append(alias_pattern)

    if not pattern_parts:
        return {}

    header_union = "|".join(sorted(pattern_parts, key=len, reverse=True))
    pattern = re.compile(
        rf"(?im)^\s*(?:[>#\-]+\s*)?(?P<header>{header_union})(?:\s*\([^)\n]*\))?\s*{COLON_CLASS}\s*"
    )
    matches = list(pattern.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        section_key = header_to_key.get(canonicalize_header_name(match.group("header")))
        if not section_key:
            continue
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[section_key] = text[value_start:value_end].strip()
    return sections


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def extract_from_messages(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return "", ""

    user_texts: list[str] = []
    assistant_texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = str(message.get("content") or "")
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)

    user_text = user_texts[-1] if user_texts else ""
    sections = extract_sections_by_headers(user_text, SECTION_ALIASES)
    question = normalize_text(sections.get("question")) or normalize_text(user_text)
    answer = normalize_text(sections.get("answer")) or (normalize_text(assistant_texts[-1]) if assistant_texts else "")
    return question, answer


def extract_question_answer(
    record: dict[str, Any],
    *,
    question_field: str = "",
    answer_field: str = "",
) -> tuple[str, str]:
    if question_field:
        question = normalize_text(record.get(question_field))
    else:
        question = first_non_empty(
            record.get("question"),
            record.get("prompt"),
            record.get("instruction"),
            record.get("query"),
            record.get("problem"),
            record.get("orig_instruction"),
        )

    if answer_field:
        answer = normalize_text(record.get(answer_field))
    else:
        answer = first_non_empty(
            record.get("answer"),
            record.get("response"),
            record.get("output"),
            record.get("candidate_answer"),
            record.get("model_answer"),
            record.get("orig_response"),
        )

    if question and answer:
        return question, answer

    message_question, message_answer = extract_from_messages(record)
    return question or message_question, answer or message_answer


def stable_sample_id(record: dict[str, Any], index: int, question: str, answer: str, id_field: str = "") -> str:
    explicit_id = (
        normalize_text(record.get(id_field))
        if id_field
        else first_non_empty(record.get("sample_id"), record.get("id"), record.get("unique_id"))
    )
    if explicit_id:
        return explicit_id
    payload = json.dumps(
        {
            "index": index,
            "question": question,
            "answer": answer,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def parse_int_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 5:
        return value
    if isinstance(value, float) and value.is_integer() and 0 <= value <= 5:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-5]", value.strip()):
        return int(value.strip())
    return None


def normalize_score_criteria(item: dict[str, Any], full_score_criteria: str = "") -> dict[str, str]:
    raw_score_criteria = item.get("score_criteria") or item.get("criteria")
    criteria: dict[str, str] = {}
    if isinstance(raw_score_criteria, dict):
        for score in range(6):
            value = normalize_text(raw_score_criteria.get(str(score)) or raw_score_criteria.get(score))
            if value:
                criteria[str(score)] = value

    for score in range(6):
        value = first_non_empty(
            item.get(f"criteria_{score}"),
            item.get(f"score_{score}"),
            item.get(f"score_{score}_criteria"),
        )
        if value:
            criteria[str(score)] = value

    if full_score_criteria and not criteria.get("5"):
        criteria["5"] = full_score_criteria

    return {str(score): criteria.get(str(score), "") for score in range(6)}


def has_complete_score_criteria(criteria: dict[str, str]) -> bool:
    return all(normalize_text(criteria.get(str(score))) for score in range(6))


def detect_complete_score_range(criteria: dict[str, str]) -> list[str]:
    if all(normalize_text(criteria.get(str(score))) for score in range(6)):
        return [str(score) for score in range(6)]
    if all(normalize_text(criteria.get(str(score))) for score in range(1, 6)):
        return [str(score) for score in range(1, 6)]
    return []


def format_score_criteria(criteria: Any, source: dict[str, Any] | None = None) -> str:
    source = source or {}
    if isinstance(criteria, dict):
        normalized = normalize_score_criteria({"score_criteria": criteria})
    else:
        normalized = normalize_score_criteria(source)
    score_range = detect_complete_score_range(normalized)
    if score_range:
        return "\n".join(f"Score {score}: {normalized[score]}" for score in score_range)
    return normalize_text(criteria)


def load_completed_ids(path: Path, *, id_field: str = "sample_id", require_ok: bool = False) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for row in iter_jsonl(path):
        if require_ok and row.get("ok") is not True:
            continue
        sample_id = normalize_text(row.get(id_field))
        if sample_id:
            completed.add(sample_id)
    return completed


def safe_unlink(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        path.unlink()
