#!/usr/bin/env python
"""Pairwise judge Feedback-Bench reference answers against generated rewrites.

The script compares each train.jsonl row's orig_reference_answer with each
candidate file row's predicted_modified_answer. A win means the candidate
predicted_modified_answer is judged better than orig_reference_answer.

It supports both local vLLM deployments and any OpenAI-compatible API.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = REPO_ROOT / "datasets/Feedback-Bench/data/train.jsonl"
DEFAULT_DATA_DIR = DEFAULT_TRAIN_FILE.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets/Feedback-Bench/pairwise_judge"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "feedback_bench_pairwise_summary.csv"

DEFAULT_PORTS = tuple(range(8000, 8004))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_VLLM_MODEL = ""
DEFAULT_API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_API_MODEL = os.environ.get("OPENAI_MODEL", "")

WINNER_VALUES = {"candidate", "predicted", "predicted_modified_answer", "modified", "b", "answer_b"}
REFERENCE_VALUES = {"reference", "orig_reference_answer", "original_reference", "ref", "a", "answer_a"}
TIE_VALUES = {"tie", "draw", "equal", "same", "equivalent"}
RESPONSE_HEADER_RE = re.compile(r"(?im)^\s*Response\s*[:\uFF1A]\s*")

SYSTEM_PROMPT = (
    "You are an impartial pairwise answer judge.\n"
    "Compare two answers to the same instruction using ONLY the provided evaluation "
    "dimension and score criteria.\n"
    "Your job is to determine which answer better satisfies the stated dimension for "
    "the original instruction. If both answers are similarly strong, similarly weak, "
    "or the difference is too small to justify a preference, choose tie.\n\n"

    "Important judging rules:\n"
    "1. Focus strictly on the stated evaluation dimension and score criteria.\n"
    "2. Do NOT reward extra length, verbosity, or additional detail unless it clearly "
    "improves performance on the stated dimension.\n"
    "3. Do NOT prefer an answer merely because it sounds more polished, more elaborate, "
    "or more concrete if those differences are not required by the dimension.\n"
    "4. Do NOT penalize an answer for being shorter if it already satisfies the dimension well.\n"
    "5. Ignore superficial style differences unless they directly affect the dimension.\n"
    "6. Do not infer missing requirements beyond the original instruction and rubric.\n"
    "7. If one answer is more explicit while the other is equally effective for the dimension, "
    "prefer tie rather than over-crediting explicitness.\n"
    "8. Use the score criteria as the main anchor: first estimate how well Answer A satisfies "
    "the dimension, then how well Answer B satisfies the dimension, then compare them.\n\n"

    "In the reason, explicitly state:\n"
    "1. which answer better satisfies the evaluation dimension, or why they are tied,\n"
    "2. one specific strength of the preferred answer (if any),\n"
    "3. one specific weakness, omission, or limitation of the other answer (if any),\n"
    "4. how these differences relate to the score criteria.\n"
    "Do not give vague reasons such as 'better overall' or 'more aligned' without explanation.\n"
    "Keep the reason concise but specific, ideally 2-4 sentences.\n\n"

    "Return strict JSON only with this exact schema:\n"
    "{\"winner\": \"A|B|tie\", \"reason\": \"...\"}"
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def progress_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "progress", True)) and tqdm is not None


def progress_write(message: str, args: argparse.Namespace) -> None:
    if progress_enabled(args):
        tqdm.write(message)
    else:
        print(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object on line {line_no}: {path}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_match_key(question: Any, answer: Any, dimension_name: Any) -> str:
    return stable_hash(
        {
            "question": normalize_text(question),
            "answer": normalize_text(answer),
            "dimension_name": normalize_text(dimension_name),
        }
    )


def get_train_question(row: dict[str, Any]) -> str:
    return normalize_text(row.get("orig_instruction") or row.get("question"))


def get_train_question_raw(row: dict[str, Any]) -> str:
    return str(row.get("orig_instruction") or row.get("question") or "").strip()


def get_train_original_answer(row: dict[str, Any]) -> str:
    return normalize_text(row.get("orig_response") or row.get("answer"))


def get_train_reference_answer(row: dict[str, Any]) -> str:
    return normalize_text(row.get("orig_reference_answer") or row.get("reference_answer"))


def split_embedded_response(question: Any) -> tuple[str, str]:
    text = str(question or "").strip()
    matches = list(RESPONSE_HEADER_RE.finditer(text))
    if not matches:
        return text, ""
    match = matches[-1]
    question_without_response = text[: match.start()].strip()
    embedded_response = text[match.end() :].strip()
    return question_without_response, embedded_response


def select_reference_and_question(
    *,
    train_row: dict[str, Any],
    candidate_row: dict[str, Any],
    reference_source: str,
) -> tuple[str, str, str]:
    raw_question = str(candidate_row.get("question") or "").strip() or get_train_question_raw(train_row)
    stripped_question, embedded_reference = split_embedded_response(raw_question)
    train_reference = get_train_reference_answer(train_row)

    if reference_source == "embedded-response":
        return normalize_text(embedded_reference), normalize_text(stripped_question or raw_question), "embedded_response"
    if reference_source == "train":
        return train_reference, normalize_text(raw_question), "train_reference"
    if embedded_reference:
        return normalize_text(embedded_reference), normalize_text(stripped_question or raw_question), "embedded_response"
    return train_reference, normalize_text(raw_question), "train_reference"


def get_train_dimension(row: dict[str, Any]) -> str:
    return normalize_text(row.get("orig_criteria") or row.get("dimension_name"))


def train_score_criteria(row: dict[str, Any]) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for score in range(6):
        value = normalize_text(
            row.get(f"orig_score{score}_description")
            or row.get(f"criteria_{score}")
            or row.get(f"score_{score}")
        )
        if value:
            criteria[str(score)] = value
    return criteria


def coerce_score_criteria(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {}

    if not isinstance(value, dict):
        return {}

    criteria: dict[str, str] = {}
    for key, item in value.items():
        key_text = normalize_text(key)
        item_text = normalize_text(item)
        if key_text and item_text:
            criteria[key_text] = item_text
    return criteria


def format_score_criteria(criteria: dict[str, str]) -> str:
    if not criteria:
        return ""

    def sort_key(key: str) -> tuple[int, str]:
        return (0, f"{int(key):02d}") if str(key).isdigit() else (1, str(key))

    return "\n".join(f"Score {key}: {normalize_text(criteria[key])}" for key in sorted(criteria, key=sort_key))


def build_train_lookup(train_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for index, row in enumerate(train_rows):
        key = make_match_key(get_train_question(row), get_train_original_answer(row), get_train_dimension(row))
        if key in lookup:
            duplicates += 1
            continue
        lookup[key] = {"row": row, "train_index": index}
    if duplicates:
        print(f"[extract] skipped duplicate train match keys: {duplicates}")
    return lookup


def candidate_match_key(row: dict[str, Any]) -> str:
    return make_match_key(row.get("question"), row.get("answer"), row.get("dimension_name"))


def candidate_files_from_args(args: argparse.Namespace) -> list[Path]:
    train_path = Path(args.train_file).resolve()
    if args.candidate_files:
        paths = [Path(value) for value in args.candidate_files]
    else:
        data_dir = Path(args.data_dir)
        paths = []
        for pattern in args.candidate_glob:
            paths.extend(data_dir.glob(pattern))

    unique: dict[Path, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved == train_path:
            continue
        if path.suffix.lower() != ".jsonl":
            continue
        unique[resolved] = path

    discovered: list[Path] = []
    for resolved in sorted(unique):
        path = unique[resolved]
        try:
            with path.open("r", encoding="utf-8") as f:
                first = next((line.strip() for line in f if line.strip()), "")
            if not first:
                continue
            first_row = json.loads(first)
        except (OSError, StopIteration, json.JSONDecodeError):
            continue
        if isinstance(first_row, dict) and "predicted_modified_answer" in first_row:
            discovered.append(path)

    if args.max_files is not None:
        discovered = discovered[: max(0, args.max_files)]
    return discovered


def make_comparison_id(candidate_file: Path, candidate_row: dict[str, Any], row_index: int) -> str:
    sample_id = normalize_text(candidate_row.get("sample_id"))
    if sample_id:
        return f"{candidate_file.name}:{sample_id}"
    return f"{candidate_file.name}:{row_index}:{candidate_match_key(candidate_row)}"


def choose_answer_order(comparison_id: str, order_mode: str) -> tuple[str, str]:
    if order_mode == "reference-first":
        return "reference", "candidate"
    if order_mode == "candidate-first":
        return "candidate", "reference"
    digest = hashlib.sha1(comparison_id.encode("utf-8")).hexdigest()
    return ("candidate", "reference") if int(digest[-1], 16) % 2 else ("reference", "candidate")


def build_user_prompt(task: dict[str, Any]) -> str:
    answer_a = task["reference_answer"] if task["answer_a_role"] == "reference" else task["candidate_answer"]
    answer_b = task["reference_answer"] if task["answer_b_role"] == "reference" else task["candidate_answer"]
    criteria_text = task["criteria_text"] or "(No explicit score criteria were provided.)"
    return (
        "Original Instruction:\n"
        f"{task['question']}\n\n"
        "Evaluation Dimension:\n"
        f"{task['dimension_name']}\n\n"
        "Score Criteria:\n"
        f"{criteria_text}\n\n"
        "Answer A:\n"
        f"{answer_a}\n\n"
        "Answer B:\n"
        f"{answer_b}\n\n"
        "Judge which answer is better under the evaluation dimension and criteria. "
        "In the reason, compare the two answers concretely and explain the key evidence "
        "for the decision based on the dimension and score criteria. "
        "Return JSON only."
    )


def prepare_tasks_for_file(
    *,
    candidate_file: Path,
    candidate_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    train_lookup: dict[str, dict[str, Any]],
    order_mode: str,
    fallback_line_index: bool,
    reference_source: str,
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tasks: list[dict[str, Any]] = []
    stats = {
        "total_rows": 0,
        "matched_rows": 0,
        "composite_key_matches": 0,
        "line_index_matches": 0,
        "embedded_response_references": 0,
        "train_reference_answers": 0,
        "skipped_unmatched": 0,
        "skipped_empty_reference": 0,
        "skipped_empty_candidate": 0,
    }
    rows = candidate_rows[: max(0, limit)] if limit is not None else candidate_rows

    for row_index, candidate_row in enumerate(rows):
        stats["total_rows"] += 1
        train_entry = train_lookup.get(candidate_match_key(candidate_row))
        match_method = "composite_key"
        if train_entry is None and fallback_line_index and row_index < len(train_rows):
            train_entry = {"row": train_rows[row_index], "train_index": row_index}
            match_method = "line_index"

        if train_entry is None:
            stats["skipped_unmatched"] += 1
            continue
        stats[f"{match_method}_matches"] += 1

        train_row = train_entry["row"]
        reference_answer, question, selected_reference_source = select_reference_and_question(
            train_row=train_row,
            candidate_row=candidate_row,
            reference_source=reference_source,
        )
        candidate_answer = normalize_text(candidate_row.get("predicted_modified_answer"))
        if not reference_answer:
            stats["skipped_empty_reference"] += 1
            continue
        if not candidate_answer:
            stats["skipped_empty_candidate"] += 1
            continue

        stats["matched_rows"] += 1
        if selected_reference_source == "embedded_response":
            stats["embedded_response_references"] += 1
        else:
            stats["train_reference_answers"] += 1
        comparison_id = make_comparison_id(candidate_file, candidate_row, row_index)
        answer_a_role, answer_b_role = choose_answer_order(comparison_id, order_mode)
        criteria = coerce_score_criteria(candidate_row.get("score_criteria")) or train_score_criteria(train_row)
        dimension_name = normalize_text(candidate_row.get("dimension_name")) or get_train_dimension(train_row)
        task = {
            "comparison_id": comparison_id,
            "row_index": row_index,
            "train_index": train_entry["train_index"],
            "match_method": match_method,
            "source_file": candidate_file.name,
            "sample_id": normalize_text(candidate_row.get("sample_id")),
            "reference_source": selected_reference_source,
            "question": question,
            "dimension_name": dimension_name,
            "criteria_text": format_score_criteria(criteria),
            "score_criteria": criteria,
            "reference_answer": reference_answer,
            "candidate_answer": candidate_answer,
            "answer_a_role": answer_a_role,
            "answer_b_role": answer_b_role,
        }
        task["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(task)},
        ]
        tasks.append(task)

    return tasks, stats


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


def build_vllm_base_urls(template: str, ports: list[int]) -> list[str]:
    return [template.format(port=port).rstrip("/") for port in ports]


def models_endpoint_ready(base_url: str, timeout_seconds: int) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
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
    with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
    return ""


def resolve_backend(args: argparse.Namespace) -> tuple[list[str], str, str]:
    if args.backend == "vllm":
        base_urls = build_vllm_base_urls(args.base_url_template, parse_ports(args.ports))
        if not args.skip_health_check:
            wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
        model = args.model.strip()
        if not model:
            for base_url in base_urls:
                try:
                    model = fetch_model_id(base_url, args.health_check_timeout)
                except Exception:
                    model = ""
                if model:
                    break
        if not model:
            raise ValueError("No vLLM --model was provided and no model id could be fetched from /models.")
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        return base_urls, model, api_key

    base_url = args.api_base_url.rstrip("/")
    model = args.api_model.strip()
    if not model:
        raise ValueError("API backend requires --api-model or OPENAI_MODEL.")
    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_BEARER_TOKEN")
        or os.environ.get("ZYUNCS_TOKEN")
        or ""
    )
    if not api_key:
        raise ValueError("API backend requires --api-key, OPENAI_API_KEY, OPENAI_BEARER_TOKEN, or ZYUNCS_TOKEN.")
    return [base_url], model, api_key


def auth_header_value(api_key: str, authorization_prefix: str) -> str:
    text = str(api_key or "").strip()
    if not text:
        return ""
    if re.match(r"(?i)^(bearer|basic)\s+", text):
        return text
    prefix = authorization_prefix.strip()
    return f"{prefix} {text}" if prefix else text


def content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                pieces.append(
                    content_to_text(
                        item.get("text")
                        or item.get("content")
                        or item.get("output_text")
                        or item.get("reasoning_content")
                    )
                )
        return "\n".join(piece for piece in pieces if piece).strip()
    if isinstance(value, dict):
        return content_to_text(
            value.get("text")
            or value.get("content")
            or value.get("output_text")
            or value.get("reasoning_content")
        )
    return str(value).strip()


def extract_chat_completion_text(response_payload: dict[str, Any], url: str) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned from {url}")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError(f"Unexpected choice format from {url}: {str(first_choice)[:500]}")

    message = first_choice.get("message") or {}
    candidates = []
    if isinstance(message, dict):
        candidates.extend(
            [
                message.get("content"),
                message.get("reasoning_content"),
                message.get("reasoning"),
                message.get("answer"),
                message.get("text"),
            ]
        )
    candidates.extend(
        [
            first_choice.get("text"),
            first_choice.get("content"),
            response_payload.get("output_text"),
        ]
    )

    for candidate in candidates:
        text = content_to_text(candidate)
        if text:
            return text

    payload_preview = json.dumps(response_payload, ensure_ascii=False)[:2000]
    raise RuntimeError(f"Empty chat completion content from {url}. Payload preview: {payload_preview}")


def post_chat_completion(
    *,
    base_url: str,
    api_key: str,
    authorization_prefix: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    authorization = auth_header_value(api_key, authorization_prefix)
    if authorization:
        headers["Authorization"] = authorization
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    return extract_chat_completion_text(response_payload, url)


def call_chat_with_retries(
    *,
    base_urls: list[str],
    task_index: int,
    api_key: str,
    authorization_prefix: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    retries: int,
    retry_sleep: float,
) -> tuple[str, str]:
    last_error: Exception | None = None
    endpoint_count = max(1, len(base_urls))
    for attempt in range(retries + 1):
        base_url = base_urls[(task_index + attempt) % endpoint_count]
        try:
            text = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                authorization_prefix=authorization_prefix,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
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


def normalize_winner_value(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text


def parse_model_output(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    parsed = extract_json_object(raw_text)
    if parsed is not None:
        winner = (
            parsed.get("winner")
            or parsed.get("decision")
            or parsed.get("result")
            or parsed.get("better_answer")
        )
        return {
            "winner": normalize_winner_value(winner),
            "reason": normalize_text(parsed.get("reason") or parsed.get("explanation")),
            "raw_output": raw_text,
            "parse_error": None if winner is not None else "Failed to parse JSON winner.",
            "strict_json_ok": winner is not None,
        }

    lowered = raw_text.lower()
    winner = ""
    if re.search(r"\b(tie|draw|equal|equivalent)\b", lowered):
        winner = "tie"
    elif re.search(r"\b(?:winner|decision|result|better answer)\s*[:\uFF1A]?\s*(?:answer\s*)?a\b", lowered):
        winner = "a"
    elif re.search(r"\banswer\s*a\b", lowered):
        winner = "a"
    elif re.search(r"\b(?:winner|decision|result|better answer)\s*[:\uFF1A]?\s*(?:answer\s*)?b\b", lowered):
        winner = "b"
    elif re.search(r"\banswer\s*b\b", lowered):
        winner = "b"
    elif re.search(r"\bcandidate\b|\bpredicted\b", lowered):
        winner = "candidate"
    elif re.search(r"\breference\b", lowered):
        winner = "reference"

    return {
        "winner": winner,
        "reason": normalize_text(raw_text),
        "raw_output": raw_text,
        "parse_error": None if winner else "Failed to parse winner.",
        "strict_json_ok": False,
    }


def winner_to_outcome(winner: str, answer_a_role: str, answer_b_role: str) -> str | None:
    value = normalize_winner_value(winner)
    if value in TIE_VALUES:
        return "tie"
    if value in {"win", "candidate_win", "candidate_wins"}:
        return "win"
    if value in {"lose", "loss", "candidate_lose", "candidate_loses"}:
        return "lose"
    if value in {"a", "answer_a"}:
        return "win" if answer_a_role == "candidate" else "lose"
    if value in {"b", "answer_b"}:
        return "win" if answer_b_role == "candidate" else "lose"
    if value in WINNER_VALUES:
        return "win"
    if value in REFERENCE_VALUES:
        return "lose"
    return None


def run_one_task(
    task: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
    api_key: str,
) -> dict[str, Any]:
    started = time.time()
    row = {
        "comparison_id": task["comparison_id"],
        "source_file": task["source_file"],
        "row_index": task["row_index"],
        "train_index": task["train_index"],
        "match_method": task["match_method"],
        "sample_id": task["sample_id"],
        "reference_source": task["reference_source"],
        "question": task["question"],
        "dimension_name": task["dimension_name"],
        "score_criteria": task["score_criteria"],
        "reference_answer": task["reference_answer"],
        "candidate_answer": task["candidate_answer"],
        "answer_a_role": task["answer_a_role"],
        "answer_b_role": task["answer_b_role"],
        "winner": "",
        "outcome": "",
        "judge_reason": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "strict_json_ok": False,
        "endpoint": "",
        "judge_model": model,
        "latency_sec": None,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=api_key,
            authorization_prefix=args.authorization_prefix,
            model=model,
            messages=task["messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = parse_model_output(raw_text)
        outcome = winner_to_outcome(parsed["winner"], task["answer_a_role"], task["answer_b_role"])
        row.update(
            {
                "winner": parsed["winner"],
                "outcome": outcome or "",
                "judge_reason": parsed["reason"],
                "raw_output": parsed["raw_output"],
                "ok": outcome in {"win", "lose", "tie"},
                "parse_error": parsed["parse_error"] if outcome else (parsed["parse_error"] or "Unknown winner."),
                "strict_json_ok": parsed["strict_json_ok"],
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
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


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            comparison_id = row.get("comparison_id")
            if isinstance(comparison_id, str) and comparison_id:
                completed[comparison_id] = row
    return completed


def reorder_output(path: Path, tasks: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
    ordered = [completed[task["comparison_id"]] for task in tasks if task["comparison_id"] in completed]
    write_jsonl(path, ordered)


def summarize_file(
    *,
    source_file: Path,
    output_path: Path,
    extract_stats: dict[str, int],
    rows: list[dict[str, Any]],
    backend: str,
    model: str,
) -> dict[str, Any]:
    wins = sum(1 for row in rows if row.get("outcome") == "win")
    losses = sum(1 for row in rows if row.get("outcome") == "lose")
    ties = sum(1 for row in rows if row.get("outcome") == "tie")
    judged = wins + losses + ties
    request_errors = sum(1 for row in rows if row.get("request_error"))
    parse_errors = sum(1 for row in rows if row.get("parse_error"))

    def rate(value: float) -> str:
        return f"{value:.6f}" if judged else ""

    return {
        "source_file": source_file.name,
        "backend": backend,
        "judge_model": model,
        "total_rows": extract_stats["total_rows"],
        "matched_rows": extract_stats["matched_rows"],
        "composite_key_matches": extract_stats["composite_key_matches"],
        "line_index_matches": extract_stats["line_index_matches"],
        "embedded_response_references": extract_stats["embedded_response_references"],
        "train_reference_answers": extract_stats["train_reference_answers"],
        "skipped_unmatched": extract_stats["skipped_unmatched"],
        "skipped_empty_reference": extract_stats["skipped_empty_reference"],
        "skipped_empty_candidate": extract_stats["skipped_empty_candidate"],
        "judged": judged,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "request_errors": request_errors,
        "parse_errors": parse_errors,
        "win_rate": rate(wins / judged) if judged else "",
        "loss_rate": rate(losses / judged) if judged else "",
        "tie_rate": rate(ties / judged) if judged else "",
        "tie_adjusted_win_rate": rate((wins + 0.5 * ties) / judged) if judged else "",
        "detail_output": str(output_path),
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_file",
        "backend",
        "judge_model",
        "total_rows",
        "matched_rows",
        "composite_key_matches",
        "line_index_matches",
        "embedded_response_references",
        "train_reference_answers",
        "skipped_unmatched",
        "skipped_empty_reference",
        "skipped_empty_candidate",
        "judged",
        "wins",
        "losses",
        "ties",
        "request_errors",
        "parse_errors",
        "win_rate",
        "loss_rate",
        "tie_rate",
        "tie_adjusted_win_rate",
        "detail_output",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_rows = read_jsonl(train_path)
    train_lookup = build_train_lookup(train_rows)
    candidate_files = candidate_files_from_args(args)
    if not candidate_files:
        raise FileNotFoundError(f"No candidate JSONL files with predicted_modified_answer found in {args.data_dir}")

    progress_write(f"[extract] train={train_path} rows={len(train_rows)} candidates={len(candidate_files)}", args)
    for path in candidate_files:
        progress_write(f"[extract] candidate={path}", args)

    if args.dry_run:
        dry_file_iter = candidate_files
        if progress_enabled(args):
            dry_file_iter = tqdm(
                candidate_files,
                total=len(candidate_files),
                desc="Files",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
        for candidate_file in dry_file_iter:
            candidate_rows = read_jsonl(candidate_file)
            _, stats = prepare_tasks_for_file(
                candidate_file=candidate_file,
                candidate_rows=candidate_rows,
                train_rows=train_rows,
                train_lookup=train_lookup,
                order_mode=args.order_mode,
                fallback_line_index=args.fallback_line_index,
                reference_source=args.reference_source,
                limit=args.limit,
            )
            progress_write(f"[dry-run] {candidate_file.name}: {stats}", args)
        return

    base_urls, model, api_key = resolve_backend(args)
    progress_write(f"[judge] backend={args.backend} endpoints={base_urls} model={model}", args)

    output_dir = Path(args.output_dir)
    summary_rows: list[dict[str, Any]] = []
    file_iter = candidate_files
    if progress_enabled(args):
        file_iter = tqdm(
            candidate_files,
            total=len(candidate_files),
            desc="Files",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
    for candidate_file in file_iter:
        candidate_rows = read_jsonl(candidate_file)
        tasks, extract_stats = prepare_tasks_for_file(
            candidate_file=candidate_file,
            candidate_rows=candidate_rows,
            train_rows=train_rows,
            train_lookup=train_lookup,
            order_mode=args.order_mode,
            fallback_line_index=args.fallback_line_index,
            reference_source=args.reference_source,
            limit=args.limit,
        )
        output_path = output_dir / f"{candidate_file.stem}.pairwise_judge.jsonl"
        if args.overwrite and output_path.exists():
            output_path.unlink()

        completed = load_completed(output_path) if args.resume else {}
        pending = [task for task in tasks if task["comparison_id"] not in completed]
        progress_write(
            f"[judge] file={candidate_file.name} total={extract_stats['total_rows']} "
            f"matched={extract_stats['matched_rows']} completed={len(completed)} "
            f"pending={len(pending)} output={output_path}",
            args,
        )

        write_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one_task, task, index, args, base_urls, model, api_key): task
                for index, task in enumerate(pending)
            }
            task_iter = as_completed(futures)
            if progress_enabled(args):
                task_iter = tqdm(
                    task_iter,
                    total=len(futures),
                    desc=candidate_file.name,
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                )
            for done, future in enumerate(task_iter, start=1):
                row = future.result()
                completed[row["comparison_id"]] = row
                append_jsonl(output_path, row, write_lock)
                if done % args.log_every == 0 or done == len(pending):
                    judged_so_far = sum(1 for item in completed.values() if item.get("ok"))
                    if progress_enabled(args) and hasattr(task_iter, "set_postfix"):
                        task_iter.set_postfix(judged=judged_so_far)
                    progress_write(
                        f"[judge] {candidate_file.name} done {done}/{len(pending)} "
                        f"pending, judged_total={judged_so_far}",
                        args,
                    )

        if args.reorder:
            reorder_output(output_path, tasks, completed)

        ordered_rows = [completed[task["comparison_id"]] for task in tasks if task["comparison_id"] in completed]
        summary_rows.append(
            summarize_file(
                source_file=candidate_file,
                output_path=output_path,
                extract_stats=extract_stats,
                rows=ordered_rows,
                backend=args.backend,
                model=model,
            )
        )
        write_summary_csv(Path(args.summary_csv), summary_rows)

    write_summary_csv(Path(args.summary_csv), summary_rows)
    progress_write(f"[summary] wrote {args.summary_csv}", args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Feedback-Bench orig_reference_answer against each "
            "predicted_modified_answer file with an LLM pairwise judge."
        )
    )
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--candidate-files", nargs="*", default=None)
    parser.add_argument("--candidate-glob", nargs="*", default=["*.jsonl"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--backend", choices=("vllm", "api"), default="vllm")

    parser.add_argument("--model", default=DEFAULT_VLLM_MODEL, help="vLLM model id. Empty means fetch from /models.")
    parser.add_argument("--base-url-template", default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"])
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--authorization-prefix",
        default="Bearer",
        help="Authorization prefix. Use an empty string if the API expects the raw token.",
    )

    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per candidate file.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    parser.add_argument(
        "--reference-source",
        choices=("auto", "train", "embedded-response"),
        default="auto",
        help=(
            "Reference answer source. auto uses a trailing 'Response:' block in the "
            "question when present, otherwise train orig_reference_answer."
        ),
    )
    parser.add_argument("--order-mode", choices=("alternate", "reference-first", "candidate-first"), default="alternate")
    parser.add_argument("--no-fallback-line-index", dest="fallback_line_index", action="store_false")
    parser.set_defaults(fallback_line_index=True)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--no-reorder", dest="reorder", action="store_false")
    parser.set_defaults(reorder=True)
    parser.add_argument("--dry-run", action="store_true", help="Only test extraction and matching; do not call LLM.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
