#!/usr/bin/env python
"""Two-stage ChainCritic benchmark pipeline for FastChat MT-Bench.

Stage 1 generates per-turn evaluation dimensions plus complete 0-5 score
criteria from MT-Bench QA pairs using a GPT/OpenAI-compatible API.

Stage 2 runs the score+generation model on each turn: it scores the candidate
answer against the generated dimensions and produces a revised answer. The
revised answers are also exported in FastChat model_answer JSONL format.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTION_FILE = REPO_ROOT / "FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
DEFAULT_REFERENCE_FILE = REPO_ROOT / "FastChat/fastchat/llm_judge/data/mt_bench/model_answer/gpt-4.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets/mt_bench_chaincritic"
DEFAULT_DIMENSIONS_FILE = DEFAULT_OUTPUT_DIR / "mt_bench_dimensions.jsonl"
DEFAULT_FLAT_RUBRIC_FILE = DEFAULT_OUTPUT_DIR / "mt_bench_score_criteria_0_5.jsonl"
DEFAULT_RESULTS_FILE = DEFAULT_OUTPUT_DIR / "mt_bench_score_generate.jsonl"
DEFAULT_FASTCHAT_ANSWER_FILE = DEFAULT_OUTPUT_DIR / "Qwen3.5-27B_answers.jsonl"

DEFAULT_PORTS = tuple(range(8001, 8005))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_DIM_API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_DIM_MODEL = os.environ.get("OPENAI_MODEL", "cortex-5")
DEFAULT_EVAL_MODEL = "Qwen3.5-27B"


DIMENSION_SYSTEM_PROMPT = (
    "You are a rigorous rubric generation assistant for MT-Bench answer evaluation.\n"
    "For the given QA pair, generate evaluation dimensions and complete 0-5 score "
    "criteria for each dimension.\n"
    "Each dimension must evaluate one observable aspect of the answer. The six "
    "score levels must be monotonic, concrete, and directly usable by a downstream "
    "scoring-and-revision model. Return strict JSON only."
)


EVAL_GENERATE_SYSTEM_PROMPT = (
    "You are a strict answer evaluation and revision model. "
    "Given a conversation turn, a candidate answer, and evaluation dimensions "
    "with 0-5 criteria, score the candidate answer and produce a revised "
    "answer that reaches full score under the same dimensions."
)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


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
    return [template.format(port=port) for port in ports]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                completed.add(sample_id)
    return completed


def load_fastchat_answers(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.is_file():
        return {}
    answers: dict[int, list[str]] = {}
    for row in load_jsonl(path):
        try:
            question_id = int(row.get("question_id"))
        except (TypeError, ValueError):
            continue
        turns: list[str] = []
        choices = row.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict) and isinstance(first_choice.get("turns"), list):
                turns = [_normalize_text(item) for item in first_choice["turns"]]
        answers[question_id] = turns
    return answers


def merge_answer_turns(
    target: dict[int, list[str]],
    source: dict[int, list[str]],
    *,
    overwrite: bool,
) -> None:
    for question_id, turns in source.items():
        resolved = target.setdefault(question_id, [])
        if len(resolved) < len(turns):
            resolved.extend([""] * (len(turns) - len(resolved)))
        for index, answer in enumerate(turns):
            normalized_answer = _normalize_text(answer)
            if normalized_answer and (overwrite or not resolved[index]):
                resolved[index] = normalized_answer


def load_dimension_row_answers(path: Path | None) -> dict[int, list[str]]:
    if path is None or not path.is_file():
        return {}
    answers: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                question_id = int(row["question_id"])
                turn_index = int(row["turn_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            answer = _normalize_text(row.get("answer"))
            if not answer:
                continue
            turns = answers.setdefault(question_id, [])
            if len(turns) <= turn_index:
                turns.extend([""] * (turn_index + 1 - len(turns)))
            if not turns[turn_index]:
                turns[turn_index] = answer
    return answers


def read_fastchat_model_id(path: Path) -> str:
    try:
        for row in load_jsonl(path):
            model_id = _normalize_text(row.get("model_id"))
            if model_id:
                return model_id
            break
    except Exception:
        pass
    return path.stem


def sanitize_filename(value: str) -> str:
    text = _normalize_text(value) or "model"
    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "model"


def with_model_suffix(path: Path, model_id: str) -> Path:
    return path.with_name(f"{path.stem}_{sanitize_filename(model_id)}{path.suffix}")


def inline_reference_answers(question: dict[str, Any]) -> list[str]:
    refs = question.get("reference")
    if not isinstance(refs, list):
        return []
    return [_normalize_text(item) for item in refs]


def answer_turns_for_question(
    question: dict[str, Any],
    external_answers: dict[int, list[str]],
    *,
    prefer_inline_reference: bool,
) -> list[str]:
    question_id = int(question["question_id"])
    turns = question.get("turns") or []
    inline_answers = inline_reference_answers(question) if prefer_inline_reference else []
    external_turns = external_answers.get(question_id, [])

    resolved: list[str] = []
    for turn_index in range(len(turns)):
        answer = ""
        if turn_index < len(inline_answers):
            answer = _normalize_text(inline_answers[turn_index])
        if not answer and turn_index < len(external_turns):
            answer = _normalize_text(external_turns[turn_index])
        resolved.append(answer)
    return resolved


def build_sample_id(question_id: int, turn_index: int) -> str:
    return f"mt_bench:{question_id}:turn:{turn_index + 1}"


def stable_answer_id(question_id: int, model_id: str, turns: list[str]) -> str:
    payload = json.dumps(
        {"question_id": question_id, "model_id": model_id, "turns": turns},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:22]


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = _normalize_text(text)
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


def normalize_score_criteria(item: dict[str, Any], full_score_criteria: str) -> dict[str, str]:
    raw_score_criteria = item.get("score_criteria") or item.get("criteria")
    criteria: dict[str, str] = {}
    if isinstance(raw_score_criteria, dict):
        for score in range(6):
            value = _normalize_text(raw_score_criteria.get(str(score)) or raw_score_criteria.get(score))
            if value:
                criteria[str(score)] = value

    for score in range(6):
        value = _normalize_text(
            item.get(f"criteria_{score}")
            or item.get(f"score_{score}")
            or item.get(f"score_{score}_criteria")
        )
        if value:
            criteria[str(score)] = value

    if full_score_criteria and not criteria.get("5"):
        criteria["5"] = full_score_criteria
    return {str(score): criteria.get(str(score), "") for score in range(6)}


def has_complete_score_criteria(criteria: dict[str, str]) -> bool:
    return all(_normalize_text(criteria.get(str(score))) for score in range(6))


def dimension_row_has_complete_rubrics(row: dict[str, Any]) -> bool:
    dimensions = row.get("evaluation_dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return False

    for dim in dimensions:
        if not isinstance(dim, dict):
            return False
        raw_score_criteria = dim.get("score_criteria")
        score_criteria = raw_score_criteria if isinstance(raw_score_criteria, dict) else {}
        full_score_criteria = _normalize_text(
            dim.get("full_score_criteria")
            or score_criteria.get("5")
            or dim.get("criteria_5")
            or dim.get("score_5")
        )
        normalized_score_criteria = normalize_score_criteria(dim, full_score_criteria)
        if not _normalize_text(dim.get("dimension_name")):
            return False
        if not full_score_criteria or not has_complete_score_criteria(normalized_score_criteria):
            return False
    return True


def load_completed_dimension_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or not dimension_row_has_complete_rubrics(row):
                continue
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                completed.add(sample_id)
    return completed


def criteria_text(score_criteria: dict[str, str]) -> str:
    return "\n".join(
        f"Score {score}: {_normalize_text(score_criteria.get(str(score)))}"
        for score in range(6)
    )


def normalize_dimensions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        dims = payload.get("evaluation_dimensions")
        if dims is None:
            dims = payload.get("dimensions")
        if dims is None:
            dims = payload.get("rubrics")
    elif isinstance(payload, list):
        dims = payload
    else:
        dims = None
    if not isinstance(dims, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in dims:
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("dimension_name") or item.get("name") or item.get("dimension"))
        full_score_criteria = _normalize_text(
            item.get("full_score_criteria")
            or item.get("criteria_5")
            or item.get("score_5")
            or item.get("criteria")
            or item.get("standard")
            or item.get("rubric")
        )
        score_criteria = normalize_score_criteria(item, full_score_criteria)
        if not full_score_criteria:
            full_score_criteria = _normalize_text(score_criteria.get("5"))
        category = _normalize_text(item.get("category") or "derived_constraint")
        if not name or not full_score_criteria or not has_complete_score_criteria(score_criteria):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized_item: dict[str, Any] = {
            "dimension_name": name,
            "full_score_criteria": full_score_criteria,
            "score_criteria": score_criteria,
            "category": category,
        }
        for score in range(6):
            normalized_item[f"criteria_{score}"] = score_criteria[str(score)]
        normalized.append(normalized_item)
    return normalized


def normalize_ratings(payload: Any, dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ratings = payload.get("ratings") if isinstance(payload, dict) else None
    if not isinstance(ratings, list):
        ratings = []
    dim_by_lower = {
        _normalize_text(dim.get("dimension_name")).lower(): dim
        for dim in dimensions
        if isinstance(dim, dict) and _normalize_text(dim.get("dimension_name"))
    }

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ratings:
        if not isinstance(item, dict):
            continue
        name = _normalize_text(item.get("dimension_name") or item.get("name"))
        if not name:
            continue
        score = item.get("score")
        if isinstance(score, str):
            try:
                score = float(score.strip())
            except ValueError:
                score = None
        elif isinstance(score, (int, float)):
            score = float(score)
        else:
            score = None
        ref = dim_by_lower.get(name.lower(), {})
        normalized.append(
            {
                "dimension_name": name,
                "score": score,
                "reason": _normalize_text(item.get("reason")),
                "category": _normalize_text(item.get("category") or ref.get("category")),
                "full_score_criteria": _normalize_text(ref.get("full_score_criteria")),
                "score_criteria": ref.get("score_criteria"),
            }
        )
        seen.add(name.lower())

    for dim in dimensions:
        name = _normalize_text(dim.get("dimension_name"))
        if name and name.lower() not in seen:
            normalized.append(
                {
                    "dimension_name": name,
                    "score": None,
                    "reason": "No rating returned by model.",
                    "category": _normalize_text(dim.get("category")),
                    "full_score_criteria": _normalize_text(dim.get("full_score_criteria")),
                    "score_criteria": dim.get("score_criteria"),
                }
            )
    return normalized


def compute_overall_score(ratings: list[dict[str, Any]]) -> float | None:
    scores = [float(item["score"]) for item in ratings if isinstance(item.get("score"), (int, float))]
    if not scores:
        return None
    return sum(scores) / len(scores)


def models_endpoint_ready(base_url: str, timeout_seconds: int) -> bool:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
            return response.status == 200
    except (OSError, urllib_error.URLError, TimeoutError, ValueError):
        return False


def wait_for_servers(base_urls: list[str], timeout_seconds: int) -> None:
    for base_url in base_urls:
        while not models_endpoint_ready(base_url, timeout_seconds):
            print(f"Waiting for {base_url.rstrip('/')}/models ...", end="\r")
            time.sleep(2)
    print("All endpoints are ready.                    ")


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
        data=json.dumps(payload).encode("utf-8"),
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
    return _normalize_text(message.get("content"))


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
            )
            return text, base_url
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(str(last_error))


def format_history(turns: list[str], answers: list[str], *, upto_turn_index: int) -> str:
    chunks: list[str] = []
    for i in range(upto_turn_index):
        chunks.append(f"User turn {i + 1}:\n{_normalize_text(turns[i])}")
        answer = _normalize_text(answers[i]) if i < len(answers) else ""
        chunks.append(f"Assistant turn {i + 1}:\n{answer or '(not provided)'}")
    return "\n\n".join(chunks) if chunks else "(none)"


def build_dimension_messages(
    question: dict[str, Any],
    turn_index: int,
    context_answers: list[str],
    current_answer: str,
) -> list[dict[str, str]]:
    turns = question.get("turns") or []
    category = _normalize_text(question.get("category"))
    current_turn = _normalize_text(turns[turn_index])
    history = format_history(turns, context_answers, upto_turn_index=turn_index)
    reference = ""
    refs = question.get("reference")
    if isinstance(refs, list) and turn_index < len(refs):
        reference = _normalize_text(refs[turn_index])

    user_content = (
        "Benchmark: MT-Bench\n"
        f"Question ID: {question.get('question_id')}\n"
        f"Category: {category}\n"
        f"Current turn: {turn_index + 1}\n\n"
        f"Previous conversation:\n{history}\n\n"
        f"Current user instruction:\n{current_turn}\n\n"
        f"Candidate answer for the current turn:\n{_normalize_text(current_answer) or '(not provided)'}\n\n"
    )
    if reference:
        user_content += f"Known reference hint for this turn:\n{reference}\n\n"
    user_content += (
        "Generate 3 to 6 evaluation dimensions for judging this QA pair. Include "
        "objective correctness when factual, mathematical, extraction, coding, or "
        "reasoning accuracy matters; include instruction-following and format "
        "constraints when the prompt specifies style, length, structure, language, "
        "or output format.\n\n"
        "For every dimension, generate a complete 0-5 scoring rubric. The score "
        "levels must be monotonic: score 0 is near-complete failure on that "
        "dimension, and score 5 is full satisfaction of the dimension. Make "
        "criteria_5 identical in meaning to full_score_criteria.\n\n"
        "Return strict JSON only with this schema:\n"
        '{"evaluation_dimensions": ['
        '{"dimension_name": "...", "category": "objective|subjective|derived_constraint|format_structure|instruction_following", '
        '"full_score_criteria": "...", '
        '"score_criteria": {"0": "...", "1": "...", "2": "...", "3": "...", "4": "...", "5": "..."}, '
        '"criteria_0": "...", "criteria_1": "...", "criteria_2": "...", '
        '"criteria_3": "...", "criteria_4": "...", "criteria_5": "..."}'
        "]}"
    )
    return [
        {"role": "system", "content": DIMENSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_eval_generate_messages(
    *,
    question: dict[str, Any],
    turn_index: int,
    history_answers: list[str],
    candidate_answer: str,
    dimensions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    turns = question.get("turns") or []
    history = format_history(turns, history_answers, upto_turn_index=turn_index)
    current_turn = _normalize_text(turns[turn_index])
    dimensions_json = json.dumps(dimensions, ensure_ascii=False, indent=2)
    answer = _normalize_text(candidate_answer)
    user_content = (
        "Benchmark: MT-Bench\n"
        f"Question ID: {question.get('question_id')}\n"
        f"Category: {_normalize_text(question.get('category'))}\n"
        f"Current turn: {turn_index + 1}\n\n"
        f"Previous conversation:\n{history}\n\n"
        f"Current user instruction:\n{current_turn}\n\n"
        f"Candidate answer:\n{answer or '(empty candidate answer; generate a full answer from scratch)'}\n\n"
        f"Evaluation dimensions:\n{dimensions_json}\n\n"
        "Tasks:\n"
        "1. Score the candidate answer on every dimension from 0 to 5. "
        "If the candidate answer is empty, use null for scores and explain that no candidate was provided.\n"
        "2. Provide a concise reason for each score.\n"
        "3. Produce modified_answer: a complete answer to the current user instruction "
        "that satisfies all dimensions at full-score level. For turn 2, it must be "
        "consistent with the previous conversation.\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "ratings": [{"dimension_name": "...", "score": 0, "reason": "..."}],\n'
        '  "overall_score": 0,\n'
        '  "modified_answer": "..."\n'
        "}"
    )
    return [
        {"role": "system", "content": EVAL_GENERATE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_dimension_tasks(
    questions: list[dict[str, Any]],
    dimension_answers: dict[int, list[str]],
    prefer_inline_reference: bool,
    require_answer: bool,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for question in questions:
        question_id = int(question["question_id"])
        turns = question.get("turns") or []
        resolved_answers = answer_turns_for_question(
            question,
            dimension_answers,
            prefer_inline_reference=prefer_inline_reference,
        )
        for turn_index in range(len(turns)):
            current_answer = resolved_answers[turn_index] if turn_index < len(resolved_answers) else ""
            if require_answer and not _normalize_text(current_answer):
                continue
            tasks.append(
                {
                    "sample_id": build_sample_id(question_id, turn_index),
                    "question": question,
                    "turn_index": turn_index,
                    "context_answers": resolved_answers,
                    "current_answer": current_answer,
                }
            )
    return tasks


def generate_one_dimension_row(
    task: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
) -> dict[str, Any]:
    question = task["question"]
    turn_index = int(task["turn_index"])
    current_answer = _normalize_text(task.get("current_answer"))
    messages = build_dimension_messages(question, turn_index, task["context_answers"], current_answer)
    started = time.time()
    row = {
        "sample_id": task["sample_id"],
        "question_id": question.get("question_id"),
        "category": question.get("category"),
        "turn_index": turn_index,
        "turn_number": turn_index + 1,
        "turns": question.get("turns", []),
        "current_prompt": (question.get("turns") or [""])[turn_index],
        "answer": current_answer,
        "evaluation_dimensions": [],
        "ok": False,
        "error": "",
        "raw_dimension_output": "",
        "endpoint": "",
        "latency_sec": None,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.dim_api_key,
            model=args.dim_model,
            messages=messages,
            temperature=args.dim_temperature,
            max_tokens=args.dim_max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed = extract_json_object(raw_text)
        dimensions = normalize_dimensions(parsed)
        row.update(
            {
                "evaluation_dimensions": dimensions,
                "ok": bool(dimensions),
                "error": "" if dimensions else "No valid dimensions with complete 0-5 criteria parsed.",
                "raw_dimension_output": raw_text,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
            }
        )
    except Exception as exc:
        row.update(
            {
                "ok": False,
                "error": str(exc),
                "latency_sec": round(time.time() - started, 4),
            }
        )
    return row


def flatten_dimension_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        question = _normalize_text(row.get("current_prompt"))
        answer = _normalize_text(row.get("answer"))
        dimensions = row.get("evaluation_dimensions")
        if not isinstance(dimensions, list):
            continue
        for index, dim in enumerate(dimensions):
            if not isinstance(dim, dict):
                continue
            score_criteria = dim.get("score_criteria") if isinstance(dim.get("score_criteria"), dict) else {}
            full_score_criteria = _normalize_text(dim.get("full_score_criteria") or score_criteria.get("5"))
            flat_row: dict[str, Any] = {
                "sample_id": f"{row.get('sample_id')}:dimension:{index + 1}",
                "question": question,
                "answer": answer,
                "dimension_name": _normalize_text(dim.get("dimension_name")),
                "full_score_criteria": full_score_criteria,
                "score_criteria": {str(score): _normalize_text(score_criteria.get(str(score))) for score in range(6)},
                "generation_status": "ok" if row.get("ok") else "failed",
                "question_id": row.get("question_id"),
                "category": row.get("category"),
                "turn_index": row.get("turn_index"),
                "turn_number": row.get("turn_number"),
            }
            for score in range(6):
                flat_row[f"criteria_{score}"] = flat_row["score_criteria"][str(score)]
            flat_rows.append(flat_row)
    return flat_rows


def run_dimension_stage(args: argparse.Namespace) -> None:
    questions = load_jsonl(Path(args.question_file))
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]

    output_path = Path(args.dimensions_file)
    flat_output_path = Path(args.flat_rubric_file) if args.flat_rubric_file else None

    dimension_answers: dict[int, list[str]] = {}
    context_answers = load_fastchat_answers(Path(args.context_answer_file)) if args.context_answer_file else {}
    merge_answer_turns(dimension_answers, context_answers, overwrite=False)
    if not args.overwrite:
        merge_answer_turns(dimension_answers, load_dimension_row_answers(output_path), overwrite=False)
    if args.dimension_answer_file:
        merge_answer_turns(
            dimension_answers,
            load_fastchat_answers(Path(args.dimension_answer_file)),
            overwrite=True,
        )
    tasks = build_dimension_tasks(
        questions,
        dimension_answers,
        args.prefer_inline_reference,
        args.require_dimension_answer,
    )
    if args.overwrite and output_path.exists():
        output_path.unlink()
    if args.overwrite and flat_output_path is not None and flat_output_path.exists():
        flat_output_path.unlink()

    completed = load_completed_dimension_ids(output_path)
    pending = [task for task in tasks if task["sample_id"] not in completed]
    print(
        f"[dimensions] total={len(tasks)} completed={len(completed)} pending={len(pending)} "
        f"output={output_path}"
    )
    if not pending:
        return

    if not args.dim_api_key:
        raise ValueError("Missing GPT API key. Set OPENAI_API_KEY or pass --dim-api-key.")
    base_urls = [args.dim_api_base_url.rstrip("/")]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(generate_one_dimension_row, task, idx, args, base_urls): task
            for idx, task in enumerate(pending)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            if done % args.flush_every == 0:
                append_jsonl(output_path, results)
                if flat_output_path is not None:
                    append_jsonl(flat_output_path, flatten_dimension_rows(results))
                results.clear()
            if done % 10 == 0 or done == len(pending):
                print(f"[dimensions] done {done}/{len(pending)}")
    append_jsonl(output_path, results)
    if flat_output_path is not None:
        append_jsonl(flat_output_path, flatten_dimension_rows(results))


def index_dimensions_by_question(path: Path) -> dict[int, dict[int, dict[str, Any]]]:
    rows = load_jsonl(path)
    indexed: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or not dimension_row_has_complete_rubrics(row):
            continue
        try:
            question_id = int(row["question_id"])
            turn_index = int(row["turn_index"])
        except (KeyError, TypeError, ValueError):
            continue
        indexed.setdefault(question_id, {})[turn_index] = row
    return indexed


def run_one_question_eval(
    question: dict[str, Any],
    question_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    dimensions_by_turn: dict[int, dict[str, Any]],
    candidate_answers: dict[int, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_id = int(question["question_id"])
    turns = question.get("turns") or []
    candidate_turns = answer_turns_for_question(
        question,
        candidate_answers,
        prefer_inline_reference=args.prefer_inline_reference,
    )
    history_answers: list[str] = []
    result_rows: list[dict[str, Any]] = []
    modified_turns: list[str] = []

    for turn_index in range(len(turns)):
        dim_row = dimensions_by_turn.get(turn_index, {})
        candidate_answer = candidate_turns[turn_index] if turn_index < len(candidate_turns) else ""
        sample_id = build_sample_id(question_id, turn_index)

        valid_dim_row = isinstance(dim_row, dict) and dimension_row_has_complete_rubrics(dim_row)
        dimensions = dim_row.get("evaluation_dimensions") if valid_dim_row else []
        if not isinstance(dimensions, list):
            dimensions = []

        row = {
            "sample_id": sample_id,
            "question_id": question_id,
            "candidate_model_id": getattr(args, "active_candidate_model_id", ""),
            "category": question.get("category"),
            "turn_index": turn_index,
            "turn_number": turn_index + 1,
            "current_prompt": turns[turn_index],
            "candidate_answer": candidate_answer,
            "evaluation_dimensions": dimensions,
            "ratings": [],
            "overall_score": None,
            "modified_answer": "",
            "ok": False,
            "skipped": False,
            "error": "",
            "raw_eval_generate_output": "",
            "endpoint": "",
            "latency_sec": None,
        }

        if not valid_dim_row:
            row.update(
                {
                    "skipped": True,
                    "modified_answer": candidate_answer,
                    "error": "Skipped because no valid evaluation dimensions were found.",
                    "latency_sec": 0.0,
                }
            )
            answer_for_history = candidate_answer
            history_answers.append(answer_for_history)
            modified_turns.append(row["modified_answer"])
            result_rows.append(row)
            continue

        messages = build_eval_generate_messages(
            question=question,
            turn_index=turn_index,
            history_answers=history_answers,
            candidate_answer=candidate_answer,
            dimensions=dimensions,
        )
        started = time.time()

        try:
            raw_text, endpoint = call_chat_with_retries(
                base_urls=base_urls,
                task_index=question_index + turn_index,
                api_key=args.api_key,
                model=args.eval_model,
                messages=messages,
                temperature=args.eval_temperature,
                max_tokens=args.eval_max_tokens,
                timeout_seconds=args.request_timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            parsed = extract_json_object(raw_text) or {}
            ratings = normalize_ratings(parsed, dimensions)
            modified_answer = _normalize_text(
                parsed.get("modified_answer")
                or parsed.get("revised_answer")
                or parsed.get("corrected_answer")
                or parsed.get("answer")
            )
            overall = parsed.get("overall_score")
            if isinstance(overall, str):
                try:
                    overall = float(overall.strip())
                except ValueError:
                    overall = None
            elif isinstance(overall, (int, float)):
                overall = float(overall)
            else:
                overall = compute_overall_score(ratings)

            row.update(
                {
                    "ratings": ratings,
                    "overall_score": overall,
                    "modified_answer": modified_answer,
                    "ok": bool(modified_answer),
                    "error": "" if modified_answer else "No modified_answer parsed.",
                    "raw_eval_generate_output": raw_text,
                    "endpoint": endpoint,
                    "latency_sec": round(time.time() - started, 4),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "ok": False,
                    "error": str(exc),
                    "latency_sec": round(time.time() - started, 4),
                }
            )

        answer_for_history = row["modified_answer"] if args.history_source == "modified" else candidate_answer
        if not answer_for_history:
            answer_for_history = row["modified_answer"] or candidate_answer
        history_answers.append(answer_for_history)
        modified_turns.append(row["modified_answer"] or candidate_answer)
        result_rows.append(row)

    answer_row = {
        "question_id": question_id,
        "answer_id": stable_answer_id(
            question_id,
            getattr(args, "active_fastchat_model_id", args.fastchat_model_id),
            modified_turns,
        ),
        "model_id": getattr(args, "active_fastchat_model_id", args.fastchat_model_id),
        "choices": [{"index": 0, "turns": modified_turns}],
        "tstamp": time.time(),
    }
    return result_rows, answer_row


def run_eval_generate_stage(args: argparse.Namespace) -> None:
    candidate_files = [Path(path) for path in args.candidate_answer_files]
    if not candidate_files and args.candidate_answer_file:
        candidate_files = [Path(args.candidate_answer_file)]
    if not candidate_files:
        raise ValueError("Provide --candidate-answer-file or --candidate-answer-files.")

    multi_model = len(candidate_files) > 1
    for candidate_file in candidate_files:
        run_eval_generate_for_candidate_file(args, candidate_file, multi_model=multi_model)


def run_eval_generate_for_candidate_file(
    args: argparse.Namespace,
    candidate_file: Path,
    *,
    multi_model: bool,
) -> None:
    questions = load_jsonl(Path(args.question_file))
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]

    dimensions_index = index_dimensions_by_question(Path(args.dimensions_file))
    if not candidate_file.is_file():
        raise FileNotFoundError(f"Candidate answer file not found: {candidate_file}")
    candidate_answers = load_fastchat_answers(candidate_file)
    candidate_model_id = args.candidate_model_id or read_fastchat_model_id(candidate_file)
    args.active_candidate_model_id = candidate_model_id
    output_path = with_model_suffix(Path(args.results_file), candidate_model_id) if multi_model else Path(args.results_file)
    fastchat_output_path = (
        with_model_suffix(Path(args.fastchat_answer_file), candidate_model_id)
        if multi_model
        else Path(args.fastchat_answer_file)
    )
    fastchat_model_id = args.fastchat_model_id or f"chaincritic-modified-{candidate_model_id}"
    args.active_fastchat_model_id = fastchat_model_id
    if args.overwrite:
        if output_path.exists():
            output_path.unlink()
        if fastchat_output_path.exists():
            fastchat_output_path.unlink()

    completed = load_completed_ids(output_path)
    pending_questions = [
        question
        for question in questions
        if any(
            build_sample_id(int(question["question_id"]), turn_index) not in completed
            for turn_index in range(len(question.get("turns") or []))
        )
    ]
    print(
        f"[eval-generate:{candidate_model_id}] questions={len(questions)} "
        f"pending_questions={len(pending_questions)} output={output_path}"
    )
    if not pending_questions:
        return

    base_urls = build_base_urls(args.eval_base_url_template, parse_ports(args.eval_ports))
    if not args.skip_health_check:
        wait_for_servers(base_urls, args.health_check_timeout)

    all_answer_rows: list[dict[str, Any]] = []
    buffered_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, question in enumerate(pending_questions):
            qid = int(question["question_id"])
            futures[
                executor.submit(
                    run_one_question_eval,
                    question,
                    idx,
                    args,
                    base_urls,
                    dimensions_index.get(qid, {}),
                    candidate_answers,
                )
            ] = qid

        for done, future in enumerate(as_completed(futures), start=1):
            result_rows, answer_row = future.result()
            new_rows = [row for row in result_rows if row["sample_id"] not in completed]
            buffered_rows.extend(new_rows)
            all_answer_rows.append(answer_row)
            if len(buffered_rows) >= args.flush_every:
                append_jsonl(output_path, buffered_rows)
                buffered_rows.clear()
            if done % 5 == 0 or done == len(pending_questions):
                print(f"[eval-generate:{candidate_model_id}] done {done}/{len(pending_questions)} questions")
    append_jsonl(output_path, buffered_rows)

    existing_answer_rows = []
    if fastchat_output_path.is_file() and not args.overwrite:
        existing_answer_rows = load_jsonl(fastchat_output_path)
    merged_answers = {int(row["question_id"]): row for row in existing_answer_rows if "question_id" in row}
    for row in all_answer_rows:
        row["model_id"] = fastchat_model_id
        merged_answers[int(row["question_id"])] = row
    write_jsonl(fastchat_output_path, [merged_answers[qid] for qid in sorted(merged_answers)])
    print(f"[eval-generate:{candidate_model_id}] FastChat modified answers: {fastchat_output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MT-Bench 0-5 rubrics via GPT API, then run ChainCritic score+generation via vLLM."
    )
    parser.add_argument(
        "--stage",
        choices=["dimensions", "eval-generate", "all"],
        default="eval-generate",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--question-file", default=str(DEFAULT_QUESTION_FILE))
    parser.add_argument(
        "--context-answer-file",
        default=str(DEFAULT_REFERENCE_FILE),
        help="Fallback FastChat answer JSONL for previous-turn context during rubric generation.",
    )
    parser.add_argument(
        "--dimension-answer-file",
        default=str(DEFAULT_REFERENCE_FILE),
        help="FastChat answer JSONL used as answers for GPT QA rubric generation. Defaults to mt_bench/gpt-4.jsonl.",
    )
    parser.add_argument(
        "--candidate-answer-file",
        default=str(DEFAULT_REFERENCE_FILE),
        help="Single FastChat answer JSONL providing candidate answers to score before revision.",
    )
    parser.add_argument(
        "--candidate-answer-files",
        nargs="*",
        default=[],
        help="Multiple FastChat answer JSONL files. Each model gets separate result outputs.",
    )
    parser.add_argument(
        "--candidate-model-id",
        default="",
        help="Override candidate model id for a single --candidate-answer-file run.",
    )
    parser.add_argument("--dimensions-file", default=str(DEFAULT_DIMENSIONS_FILE))
    parser.add_argument(
        "--flat-rubric-file",
        default=str(DEFAULT_FLAT_RUBRIC_FILE),
        help="Flat per-dimension output similar to datasets/0-5/score_criteria_0_5.jsonl.",
    )
    parser.add_argument("--results-file", default=str(DEFAULT_RESULTS_FILE))
    parser.add_argument("--fastchat-answer-file", default=str(DEFAULT_FASTCHAT_ANSWER_FILE))
    parser.add_argument(
        "--fastchat-model-id",
        default="",
        help="Model id for exported modified answers. Defaults to chaincritic-modified-{candidate_model_id}.",
    )

    parser.add_argument("--dim-model", default=DEFAULT_DIM_MODEL)
    parser.add_argument("--eval-model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument("--dim-api-base-url", default=DEFAULT_DIM_API_BASE_URL)
    parser.add_argument("--dim-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--eval-base-url-template", default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--eval-ports", nargs="*", default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--dim-temperature", type=float, default=0.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--dim-max-tokens", type=int, default=4096)
    parser.add_argument("--eval-max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--require-dimension-answer",
        action="store_true",
        help="Skip MT-Bench turns that do not have an answer in --dimension-answer-file.",
    )
    parser.add_argument(
        "--prefer-inline-reference",
        dest="prefer_inline_reference",
        action="store_true",
        help="Prefer question.jsonl inline reference answers over external answer files.",
    )
    parser.add_argument(
        "--no-prefer-inline-reference",
        dest="prefer_inline_reference",
        action="store_false",
        help="Use only external answer files for answers.",
    )
    parser.set_defaults(prefer_inline_reference=False)
    parser.add_argument(
        "--history-source",
        choices=["modified", "candidate"],
        default="modified",
        help="For turn 2 eval/generation, use turn-1 modified answer or original candidate as history.",
    )
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.stage in {"dimensions", "all"}:
        run_dimension_stage(args)
    if args.stage in {"eval-generate", "all"}:
        run_eval_generate_stage(args)


if __name__ == "__main__":
    main()
