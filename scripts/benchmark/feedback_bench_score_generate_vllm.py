#!/usr/bin/env python
"""Run Feedback-Bench score+rewrite inference through vLLM endpoints.

The input JSONL is expected to contain the fields generated in
datasets/Feedback-Bench/score_rubric_0_5.jsonl:
question, answer, dimension_name, and score_criteria.
"""

from __future__ import annotations

import argparse
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

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_FILE = REPO_ROOT / "datasets/Feedback-Bench/data/score_rubric_0_5.jsonl"
DEFAULT_OUTPUT_FILE = REPO_ROOT / "datasets/Feedback-Bench/data/score_generate_chaincritic-v11-70669.jsonl"
DEFAULT_PORTS = tuple(range(8001, 8005))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_MODEL = "chaincritic-v11-70669"

COLON_CLASS = r"[:=\-\uFF1A]"
SCORE_LINE_RE = re.compile(
    rf"(?im)^\s*(?:score|rating)\s*{COLON_CLASS}\s*([1-5](?:\.0+)?)\s*$"
)
SCORE_FALLBACK_RE = re.compile(
    rf"(?is)\b(?:score|rating)\b\s*{COLON_CLASS}\s*['\"]?([1-5](?:\.0+)?)['\"]?"
)

REASON_RE = re.compile(
    rf"(?is)(?:reason|feedback|rationale|justification)\s*{COLON_CLASS}\s*(.*?)\s*"
    rf"(?:(?:\n\s*)?(?:modified(?:\s+|_)?answer|revised(?:\s+|_)?answer|better(?:\s+|_)?answer|improved(?:\s+|_)?answer|rewrite)\s*{COLON_CLASS}|$)"
)

MODIFIED_RE = re.compile(
    rf"(?is)(?:modified(?:\s+|_)?answer|revised(?:\s+|_)?answer|better(?:\s+|_)?answer|improved(?:\s+|_)?answer|rewrite)\s*{COLON_CLASS}\s*(.*)$"
)

SYSTEM_PROMPT = (
    "You are a strict answer evaluation and revision model.\n"
    "Evaluate the candidate answer using ONLY the provided question, evaluation dimension, "
    "and complete 1-5 score criteria.\n"
    "Then rewrite the candidate answer into the single best final answer for the original question, optimized "
    "for the same evaluation dimension.\n"
    "Your rewrite must be the actual answer itself, not commentary about what a good answer would do, not a plan, "
    "and not a summary of someone else's performance.\n"
    "Prefer specific, situation-grounded wording over generic abstractions. Show rather than tell whenever possible: "
    "use concrete next steps, examples, contingencies, or short sample wording when they help the dimension.\n"
    "Do not introduce unsupported facts. If the question is underspecified, make the minimum necessary assumption explicit "
    "and still provide a useful answer.\n"
    "Do not use placeholders like [Event Name], [specific occasion], or similar unless they are already present in the user's question and unavoidable. "
    "If a detail is unknown, use a natural generic phrase such as 'the event' instead of a bracketed placeholder.\n"
    "Use one language consistently in modified_answer unless the user clearly asks for mixed-language output.\n"
    "Return strict JSON only, with this exact schema:\n"
    '{"score": 1, "reason": "...", "modified_answer": "..."}\n'
    "The score must be an integer from 1 to 5. "
    "The reason must be concise and based on the rubric. "
    "The modified_answer must be a single plain string, not a JSON object, not a list, and not markdown. "
    "The modified_answer must be concise, directly answer the question, and stay under 220 words. "
    "Do not repeat phrases or restate the same point multiple times."
)


def preserve_text(value: Any) -> str:
    """Keep original line structure as much as possible, only normalize line endings and trim edges."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def compact_text(value: Any) -> str:
    """Compact whitespace for validation / dedup / ID generation only."""
    return " ".join(preserve_text(value).split()).strip()


def has_content(value: Any) -> bool:
    return bool(preserve_text(value))


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


def _score_key_to_int(score_key: Any) -> int | None:
    text = preserve_text(score_key)
    if re.fullmatch(r"[1-5]", text):
        return int(text)
    return None


def _format_one_score_item(score_key: Any, value: Any) -> str:
    text = preserve_text(value)
    if not text:
        return ""
    if "\n" in text:
        return f"Score {score_key}:\n{text}"
    return f"Score {score_key}: {text}"


def _filter_score_criteria_text_to_1_5(value: Any) -> str:
    raw = preserve_text(value)
    if not raw:
        return ""

    heading_re = re.compile(rf"(?im)^\s*score\s*([0-9]+)\s*{COLON_CLASS}\s*")
    matches = list(heading_re.finditer(raw))

    if not matches:
        return raw

    kept_chunks: list[str] = []
    for i, match in enumerate(matches):
        try:
            score = int(match.group(1))
        except ValueError:
            continue

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end].strip()

        if 1 <= score <= 5 and chunk:
            kept_chunks.append(chunk)

    return "\n".join(kept_chunks).strip()


def format_score_criteria(record: dict[str, Any]) -> str:
    score_criteria = record.get("score_criteria")

    if isinstance(score_criteria, dict):
        filtered_items: list[tuple[int, Any]] = []
        for key, value in score_criteria.items():
            score = _score_key_to_int(key)
            if score is not None:
                filtered_items.append((score, value))

        filtered_items.sort(key=lambda item: item[0])
        parts = [_format_one_score_item(score, value) for score, value in filtered_items]
        return "\n".join(part for part in parts if part)

    pieces: list[str] = []
    for score in range(1, 6):
        value = record.get(f"criteria_{score}")
        part = _format_one_score_item(score, value)
        if part:
            pieces.append(part)
    if pieces:
        return "\n".join(pieces)

    return _filter_score_criteria_text_to_1_5(score_criteria)


def build_sample_id(record: dict[str, Any], index: int) -> str:
    existing = compact_text(record.get("sample_id"))
    if existing:
        return existing
    payload = json.dumps(
        {
            "index": index,
            "question": preserve_text(record.get("question")),
            "answer": preserve_text(record.get("answer")),
            "dimension_name": preserve_text(record.get("dimension_name")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_dimension_specific_guidance(question: str, dimension_name: str) -> str:
    text = f"{question}\n{dimension_name}".lower()
    rules: list[str] = [
        "Write the modified_answer as the final user-facing answer itself, not as a description of what the answer or speaker would do.",
        "Replace generic filler with concrete content that a judge can directly verify from the text.",
        "Keep the answer self-contained and natural; do not use bracketed placeholders.",
    ]

    if any(token in text for token in ["emotion", "emotional", "empat", "frustrat", "upset", "angry", "sad", "stress", "burnout"]):
        rules.extend(
            [
                "If emotion is relevant, briefly name or acknowledge the user's emotional state in a natural way.",
                "Include at least one concrete next step, remedy option, or contingency so the reassurance is actionable rather than generic.",
            ]
        )

    if any(token in text for token in ["cultural", "culture", "multinational", "international", "background", "stereotype", "divers", "japan", "united states", "united arab emirates", "uae"]):
        rules.extend(
            [
                "For cultural sensitivity, avoid fixed national stereotypes; describe tendencies as possibilities, not rules about all people.",
                "When relevant, include voluntary participation, comfort level, language accommodation, or room for people to share their own background.",
            ]
        )

    if any(token in text for token in ["humor", "humour", "light-hearted", "light hearted", "funny", "alleviating stress"]):
        rules.extend(
            [
                "If humor is required, include at least one brief, workplace-safe, non-sarcastic humorous line or analogy inside the answer itself.",
                "Do not merely say the answer is humorous; make the humor visible in the wording.",
            ]
        )

    if any(token in text for token in ["communication", "team", "manager", "project", "workshop", "meeting", "collaboration"]):
        rules.append(
            "Prefer concrete process steps, communication norms, or examples over abstract management slogans."
        )

    if any(token in text for token in ["context", "previous", "earlier", "follow-up", "follow up", "last discussion"]):
        rules.append(
            "If prior context is relevant, refer to it explicitly so the answer reads like a continuation rather than a standalone response."
        )

    if any(token in text for token in ["underspecified", "unclear", "missing information", "ambiguous"]):
        rules.append(
            "If key details are missing, state the smallest reasonable assumption explicitly and still provide a useful answer."
        )

    return "\n".join(f"- {rule}" for rule in rules)


def build_user_prompt(record: dict[str, Any]) -> str:
    question = preserve_text(record.get("question"))
    answer = preserve_text(record.get("answer"))
    dimension_name = preserve_text(record.get("dimension_name"))
    criteria_text = format_score_criteria(record)
    dimension_guidance = build_dimension_specific_guidance(question, dimension_name)

    return (
        "Question:\n"
        f"{question}\n\n"
        "Candidate Answer:\n"
        f"{answer}\n\n"
        "Evaluation Dimension:\n"
        f"{dimension_name}\n\n"
        "Score Criteria (1-5):\n"
        f"{criteria_text}\n\n"
        "Rewrite Requirements:\n"
        "- First identify the single biggest reason the candidate answer underperforms on this dimension, and fix that in the rewrite.\n"
        "- The modified_answer must directly answer the original question and should read like the best final answer, not analysis, commentary, or a summary of what a good answer would do.\n"
        "- Prefer concrete details, explicit wording, and visible evidence of the target skill over generic phrases.\n"
        f"{dimension_guidance}\n\n"
        "Tasks:\n"
        "1. Assign one integer score from 1 to 5 to the candidate answer.\n"
        "2. Give a concise reason grounded in the score criteria.\n"
        "3. Rewrite a better answer to the original question that would satisfy the dimension better.\n\n"
        "Return strict JSON only."
    )


def prepare_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    required_fields = ("question", "answer", "dimension_name")

    for index, record in enumerate(records):
        missing = [field for field in required_fields if not has_content(record.get(field))]
        criteria_text = format_score_criteria(record)
        if not criteria_text:
            missing.append("score_criteria")
        if missing:
            raise ValueError(f"Record {index} missing required fields: {missing}")

        sample_id = build_sample_id(record, index)
        user_prompt = build_user_prompt(record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        samples.append(
            {
                "sample_id": sample_id,
                "index": index,
                "question": preserve_text(record.get("question")),
                "answer": preserve_text(record.get("answer")),
                "dimension_name": preserve_text(record.get("dimension_name")),
                "full_score_criteria": record.get("full_score_criteria"),
                "score_criteria": record.get("score_criteria"),
                "criteria_text": criteria_text,
                "source_record": record,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "messages": messages,
            }
        )
    return samples


def extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    candidates: list[str] = [raw]

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        cleaned = candidate
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def parse_int_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if 1 <= value <= 5 else None

    if isinstance(value, float):
        if value.is_integer() and 1 <= int(value) <= 5:
            return int(value)
        return None

    if isinstance(value, str):
        text = value.strip()

        if re.fullmatch(r"[1-5]", text):
            return int(text)

        if re.fullmatch(r"[1-5]\.0+", text):
            return int(float(text))

        m = re.search(r"\b([1-5])(?:\.0+)?\b", text)
        if m:
            return int(float(m.group(1)))

    return None


def parse_model_output(text: str) -> dict[str, Any]:
    raw_text = preserve_text(text)

    def normalize_key(key: str) -> str:
        return re.sub(r"[^a-z0-9]", "", key.lower())

    def pick_field(data: dict[str, Any], aliases: list[str]) -> Any:
        if not isinstance(data, dict):
            return None

        normalized_map: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(k, str):
                normalized_map[normalize_key(k)] = v

        for alias in aliases:
            key = normalize_key(alias)
            if key in normalized_map:
                return normalized_map[key]
        return None

    parsed = extract_json_object(raw_text)

    if parsed is not None:
        score_value = pick_field(parsed, ["score", "rating", "grade"])
        reason_value = pick_field(parsed, ["reason", "feedback", "rationale", "justification"])
        modified_value = pick_field(
            parsed,
            [
                "modified_answer",
                "modified answer",
                "revised_answer",
                "revised answer",
                "better_answer",
                "better answer",
                "improved_answer",
                "improved answer",
                "rewrite",
                "rewritten_answer",
                "rewritten answer",
            ],
        )

        score = parse_int_score(score_value)
        reason = preserve_text(reason_value)
        modified_answer = preserve_text(modified_value)

        parse_error = None
        if score is None:
            parse_error = "Failed to parse JSON score."
        elif not reason:
            parse_error = "Missing JSON field: reason."
        elif not modified_answer:
            parse_error = "Missing JSON field: modified_answer."

        return {
            "score": score,
            "reason": reason,
            "modified_answer": modified_answer,
            "raw_output": raw_text,
            "parsed_json": parsed,
            "parse_mode": "json",
            "parse_error": parse_error,
            "strict_json_ok": score is not None and bool(reason) and bool(modified_answer),
        }

    score: int | None = None
    reason = ""
    modified_answer = ""
    parse_error: str | None = None

    score_match = SCORE_LINE_RE.search(raw_text)
    if score_match:
        score = parse_int_score(score_match.group(1))
    else:
        fallback_score = SCORE_FALLBACK_RE.search(raw_text)
        if fallback_score:
            score = parse_int_score(fallback_score.group(1))
        else:
            nearby = re.search(
                rf"(?is)\b(?:score|rating)\b.{{0,20}}?\b([1-5](?:\.0+)?)\b",
                raw_text,
            )
            if nearby:
                score = parse_int_score(nearby.group(1))
            else:
                parse_error = "Failed to parse score."

    reason_match = REASON_RE.search(raw_text)
    modified_match = MODIFIED_RE.search(raw_text)

    if reason_match:
        reason = preserve_text(reason_match.group(1))

    if modified_match:
        modified_answer = preserve_text(modified_match.group(1))

    if not reason:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if re.match(rf"(?is)^(?:reason|feedback|rationale|justification)\s*{COLON_CLASS}", line):
                reason = preserve_text(re.split(COLON_CLASS, line, maxsplit=1)[-1])
                break

    if not modified_answer:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if re.match(
                rf"(?is)^(?:modified(?:\s+|_)?answer|revised(?:\s+|_)?answer|better(?:\s+|_)?answer|improved(?:\s+|_)?answer|rewrite)\s*{COLON_CLASS}",
                line,
            ):
                modified_answer = preserve_text(re.split(COLON_CLASS, line, maxsplit=1)[-1])
                break

    if score is not None and not reason and parse_error is None:
        parse_error = "Failed to parse reason."
    if score is not None and not modified_answer and parse_error is None:
        parse_error = "Failed to parse modified_answer."

    return {
        "score": score,
        "reason": reason,
        "modified_answer": modified_answer,
        "raw_output": raw_text,
        "parsed_json": None,
        "parse_mode": "regex",
        "parse_error": parse_error,
        "strict_json_ok": False,
    }


def _send_chat_request(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
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
        return json.loads(response.read().decode("utf-8"))


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

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    try:
        response_payload = _send_chat_request(
            url=url,
            api_key=api_key,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
    except urllib_error.HTTPError as exc:
        if exc.code not in {400, 404, 422}:
            raise
        payload.pop("response_format", None)
        response_payload = _send_chat_request(
            url=url,
            api_key=api_key,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )

    choices = response_payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned from {url}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
        return "\n".join(texts).strip()

    return str(content or "").strip()


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


def run_one_sample(
    sample: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()

    row = {
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "dimension_name": sample["dimension_name"],
        "score_criteria": sample["score_criteria"],
        "predicted_score": None,
        "predicted_reason": "",
        "predicted_modified_answer": "",
        "raw_output": "",
        "parse_mode": "",
        "strict_json_ok": False,
        "endpoint": "",
        "elapsed_seconds": None,
        "ok": False,
        "parse_error": None,
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
        ok = parsed["score"] is not None and bool(parsed["reason"]) and bool(parsed["modified_answer"])

        row.update(
            {
                "sample_id": sample["sample_id"],
                "predicted_score": parsed["score"],
                "predicted_reason": parsed["reason"],
                "predicted_modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "parse_mode": parsed["parse_mode"],
                "strict_json_ok": parsed["strict_json_ok"],
                "endpoint": endpoint,
                "elapsed_seconds": round(time.time() - started, 4),
                "ok": ok,
                "parse_error": parsed["parse_error"],
            }
        )
    except Exception as exc:
        row.update(
            {
                "elapsed_seconds": round(time.time() - started, 4),
                "ok": False,
                "parse_error": str(exc),
            }
        )

    return row


def load_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
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

            sample_id = row.get("sample_id")
            if isinstance(sample_id, str) and sample_id:
                completed[sample_id] = row

    return completed


def reorder_output(path: Path, samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
    ordered = [completed[sample["sample_id"]] for sample in samples if sample["sample_id"] in completed]
    write_jsonl(path, ordered)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    samples = prepare_samples(records)

    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if not args.skip_health_check:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    model = resolve_model(args.model, base_urls, args.health_check_timeout)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed = load_completed_predictions(output_path) if args.resume else {}
    pending = [
        sample for sample in samples
        if not (
            sample["sample_id"] in completed
            and completed[sample["sample_id"]].get("ok") is True
        )
    ]

    print(
        f"[score-generate] total={len(samples)} completed={len(completed)} "
        f"pending={len(pending)} endpoints={base_urls} model={model} output={output_path}"
    )

    if not pending:
        reorder_output(output_path, samples, completed)
        return

    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_one_sample, sample, index, args, base_urls, model): sample
            for index, sample in enumerate(pending)
        }

        pbar = tqdm(total=len(pending), desc="score-generate", dynamic_ncols=True)

        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            completed[row["sample_id"]] = row
            append_jsonl(output_path, row, write_lock)

            ok_count = sum(1 for item in completed.values() if item.get("ok"))
            pbar.update(1)
            pbar.set_postfix(ok_total=ok_count)

            if done % args.log_every == 0 or done == len(pending):
                print(f"[score-generate] done {done}/{len(pending)} pending, ok_total={ok_count}")

        pbar.close()

    if args.reorder:
        reorder_output(output_path, samples, completed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score Feedback-Bench answers and generate improved answers through "
            "OpenAI-compatible vLLM endpoints."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name served by vLLM. Use an empty string to auto-fetch from /models.",
    )
    parser.add_argument("--base-url-template", default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--no-reorder",
        dest="reorder",
        action="store_false",
        help="Keep append completion order instead of rewriting output in input order at the end.",
    )
    parser.set_defaults(reorder=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()



# #!/usr/bin/env python
# """Run Feedback-Bench score+rewrite inference through vLLM endpoints.

# The input JSONL is expected to contain the fields generated in
# datasets/Feedback-Bench/score_rubric_0_5.jsonl:
# question, answer, dimension_name, and score_criteria.
# """

# from __future__ import annotations

# import argparse
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import hashlib
# import json
# import os
# from pathlib import Path
# import re
# import threading
# import time
# from typing import Any
# from urllib import error as urllib_error
# from urllib import request as urllib_request

# from tqdm import tqdm


# REPO_ROOT = Path(__file__).resolve().parents[2]
# DEFAULT_INPUT_FILE = REPO_ROOT / "datasets/Feedback-Bench/data/score_rubric_0_5.jsonl"
# DEFAULT_OUTPUT_FILE = REPO_ROOT / "datasets/Feedback-Bench/data/score_generate_chaincritic-v11-62000.jsonl"
# DEFAULT_PORTS = tuple(range(8000, 8004))
# DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
# DEFAULT_MODEL = "chaincritic-v11-62000"

# COLON_CLASS = r"[:=\-\uFF1A]"
# SCORE_LINE_RE = re.compile(
#     rf"(?im)^\s*(?:score|rating)\s*{COLON_CLASS}\s*([1-5](?:\.0+)?)\s*$"
# )
# SCORE_FALLBACK_RE = re.compile(
#     rf"(?is)\b(?:score|rating)\b\s*{COLON_CLASS}\s*['\"]?([1-5](?:\.0+)?)['\"]?"
# )

# REASON_RE = re.compile(
#     rf"(?is)(?:reason|feedback|rationale|justification)\s*{COLON_CLASS}\s*(.*?)\s*"
#     rf"(?:(?:\n\s*)?(?:modified(?:\s+|_)?answer|revised(?:\s+|_)?answer|better(?:\s+|_)?answer|improved(?:\s+|_)?answer|rewrite)\s*{COLON_CLASS}|$)"
# )

# MODIFIED_RE = re.compile(
#     rf"(?is)(?:modified(?:\s+|_)?answer|revised(?:\s+|_)?answer|better(?:\s+|_)?answer|improved(?:\s+|_)?answer|rewrite)\s*{COLON_CLASS}\s*(.*)$"
# )

# SYSTEM_PROMPT = (
#     "You are a strict answer evaluation and revision model.\n"
#     "Evaluate the candidate answer using ONLY the provided question, evaluation dimension, "
#     "and complete 1-5 score criteria.\n"
#     "Then rewrite the candidate answer into a better answer for the original question, optimized "
#     "for the same evaluation dimension.\n"
#     "Do not introduce unsupported facts. If the question is underspecified, make the minimum "
#     "necessary assumption explicit.\n"
#     "Return strict JSON only, with this exact schema:\n"
#     '{"score": 1, "reason": "...", "modified_answer": "..."}\n'
#     "The score must be an integer from 1 to 5. "
#     "The reason must be concise and based on the rubric. "
#     "The modified_answer must be a single plain string, not a JSON object, not a list, and not markdown."
#     "The modified_answer must be concise, directly answer the question, and stay under 220 words."
#     "Do not repeat phrases or restate the same point multiple times."
# )


# def preserve_text(value: Any) -> str:
#     """Keep original line structure as much as possible, only normalize line endings and trim edges."""
#     if value is None:
#         return ""
#     text = str(value)
#     text = text.replace("\r\n", "\n").replace("\r", "\n")
#     return text.strip()


# def compact_text(value: Any) -> str:
#     """Compact whitespace for validation / dedup / ID generation only."""
#     return " ".join(preserve_text(value).split()).strip()


# def has_content(value: Any) -> bool:
#     return bool(preserve_text(value))


# def load_jsonl(path: Path) -> list[dict[str, Any]]:
#     rows: list[dict[str, Any]] = []
#     with path.open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             text = line.strip()
#             if not text:
#                 continue
#             row = json.loads(text)
#             if not isinstance(row, dict):
#                 raise ValueError(f"Expected JSON object on line {line_no}: {path}")
#             rows.append(row)
#     return rows


# def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with lock:
#         with path.open("a", encoding="utf-8") as f:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as f:
#         for row in rows:
#             f.write(json.dumps(row, ensure_ascii=False) + "\n")


# def parse_ports(values: list[str] | None) -> list[int]:
#     if not values:
#         return list(DEFAULT_PORTS)

#     ports: list[int] = []
#     for value in values:
#         text = str(value).strip()
#         if not text:
#             continue
#         if "-" in text:
#             start_text, end_text = text.split("-", 1)
#             start = int(start_text)
#             end = int(end_text)
#             step = 1 if end >= start else -1
#             ports.extend(range(start, end + step, step))
#         else:
#             ports.append(int(text))
#     return ports


# def build_base_urls(template: str, ports: list[int]) -> list[str]:
#     return [template.format(port=port).rstrip("/") for port in ports]


# def models_endpoint_ready(base_url: str, timeout_seconds: int) -> bool:
#     url = base_url.rstrip("/") + "/models"
#     try:
#         with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
#             return response.status == 200
#     except (OSError, urllib_error.URLError, TimeoutError, ValueError):
#         return False


# def wait_for_servers(base_urls: list[str], timeout_seconds: int, interval_seconds: float) -> None:
#     for base_url in base_urls:
#         while not models_endpoint_ready(base_url, timeout_seconds):
#             print(f"Waiting for {base_url.rstrip('/')}/models ...", end="\r")
#             time.sleep(interval_seconds)
#     print("All endpoints are ready.                    ")


# def fetch_model_id(base_url: str, timeout_seconds: int) -> str:
#     url = base_url.rstrip("/") + "/models"
#     with urllib_request.urlopen(url, timeout=timeout_seconds) as response:
#         payload = json.loads(response.read().decode("utf-8"))
#     data = payload.get("data")
#     if isinstance(data, list) and data:
#         first = data[0]
#         if isinstance(first, dict) and first.get("id"):
#             return str(first["id"])
#     return ""


# def resolve_model(model: str, base_urls: list[str], timeout_seconds: int) -> str:
#     if model.strip():
#         return model.strip()
#     for base_url in base_urls:
#         try:
#             resolved = fetch_model_id(base_url, timeout_seconds)
#         except Exception:
#             resolved = ""
#         if resolved:
#             return resolved
#     raise ValueError("No --model was provided and no model id could be fetched from /models.")


# def _score_key_to_int(score_key: Any) -> int | None:
#     text = preserve_text(score_key)
#     if re.fullmatch(r"[1-5]", text):
#         return int(text)
#     return None


# def _format_one_score_item(score_key: Any, value: Any) -> str:
#     text = preserve_text(value)
#     if not text:
#         return ""
#     if "\n" in text:
#         return f"Score {score_key}:\n{text}"
#     return f"Score {score_key}: {text}"


# def _filter_score_criteria_text_to_1_5(value: Any) -> str:
#     raw = preserve_text(value)
#     if not raw:
#         return ""

#     heading_re = re.compile(rf"(?im)^\s*score\s*([0-9]+)\s*{COLON_CLASS}\s*")
#     matches = list(heading_re.finditer(raw))

#     if not matches:
#         return raw

#     kept_chunks: list[str] = []
#     for i, match in enumerate(matches):
#         try:
#             score = int(match.group(1))
#         except ValueError:
#             continue

#         start = match.start()
#         end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
#         chunk = raw[start:end].strip()

#         if 1 <= score <= 5 and chunk:
#             kept_chunks.append(chunk)

#     return "\n".join(kept_chunks).strip()


# def format_score_criteria(record: dict[str, Any]) -> str:
#     score_criteria = record.get("score_criteria")

#     if isinstance(score_criteria, dict):
#         filtered_items: list[tuple[int, Any]] = []
#         for key, value in score_criteria.items():
#             score = _score_key_to_int(key)
#             if score is not None:
#                 filtered_items.append((score, value))

#         filtered_items.sort(key=lambda item: item[0])
#         parts = [_format_one_score_item(score, value) for score, value in filtered_items]
#         return "\n".join(part for part in parts if part)

#     pieces: list[str] = []
#     for score in range(1, 6):
#         value = record.get(f"criteria_{score}")
#         part = _format_one_score_item(score, value)
#         if part:
#             pieces.append(part)
#     if pieces:
#         return "\n".join(pieces)

#     return _filter_score_criteria_text_to_1_5(score_criteria)


# def build_sample_id(record: dict[str, Any], index: int) -> str:
#     existing = compact_text(record.get("sample_id"))
#     if existing:
#         return existing
#     payload = json.dumps(
#         {
#             "index": index,
#             "question": preserve_text(record.get("question")),
#             "answer": preserve_text(record.get("answer")),
#             "dimension_name": preserve_text(record.get("dimension_name")),
#         },
#         ensure_ascii=False,
#         sort_keys=True,
#     )
#     return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# def build_user_prompt(record: dict[str, Any]) -> str:
#     question = preserve_text(record.get("question"))
#     answer = preserve_text(record.get("answer"))
#     dimension_name = preserve_text(record.get("dimension_name"))
#     criteria_text = format_score_criteria(record)

#     return (
#         "Question:\n"
#         f"{question}\n\n"
#         "Candidate Answer:\n"
#         f"{answer}\n\n"
#         "Evaluation Dimension:\n"
#         f"{dimension_name}\n\n"
#         "Score Criteria (1-5):\n"
#         f"{criteria_text}\n\n"
#         "Tasks:\n"
#         "1. Assign one integer score from 1 to 5 to the candidate answer.\n"
#         "2. Give a concise reason grounded in the score criteria.\n"
#         "3. Rewrite a better answer to the original question that would satisfy the dimension better.\n\n"
#         "Return strict JSON only."
#     )


# def prepare_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
#     samples: list[dict[str, Any]] = []
#     required_fields = ("question", "answer", "dimension_name")

#     for index, record in enumerate(records):
#         missing = [field for field in required_fields if not has_content(record.get(field))]
#         criteria_text = format_score_criteria(record)
#         if not criteria_text:
#             missing.append("score_criteria")
#         if missing:
#             raise ValueError(f"Record {index} missing required fields: {missing}")

#         sample_id = build_sample_id(record, index)
#         user_prompt = build_user_prompt(record)
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_prompt},
#         ]

#         samples.append(
#             {
#                 "sample_id": sample_id,
#                 "index": index,
#                 "question": preserve_text(record.get("question")),
#                 "answer": preserve_text(record.get("answer")),
#                 "dimension_name": preserve_text(record.get("dimension_name")),
#                 "full_score_criteria": record.get("full_score_criteria"),
#                 "score_criteria": record.get("score_criteria"),
#                 "criteria_text": criteria_text,
#                 "source_record": record,
#                 "system_prompt": SYSTEM_PROMPT,
#                 "user_prompt": user_prompt,
#                 "messages": messages,
#             }
#         )
#     return samples


# def extract_json_object(text: str) -> dict[str, Any] | None:
#     raw = str(text or "").strip()
#     if not raw:
#         return None

#     candidates: list[str] = [raw]

#     fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
#     candidates.extend(block.strip() for block in fenced_blocks if block.strip())

#     start = raw.find("{")
#     end = raw.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         candidates.append(raw[start:end + 1].strip())

#     for candidate in candidates:
#         if not candidate:
#             continue

#         try:
#             parsed = json.loads(candidate)
#             if isinstance(parsed, dict):
#                 return parsed
#         except json.JSONDecodeError:
#             pass

#         cleaned = candidate
#         start = cleaned.find("{")
#         end = cleaned.rfind("}")
#         if start != -1 and end != -1 and end > start:
#             cleaned = cleaned[start:end + 1]

#         cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

#         try:
#             parsed = json.loads(cleaned)
#             if isinstance(parsed, dict):
#                 return parsed
#         except json.JSONDecodeError:
#             pass

#     return None


# def parse_int_score(value: Any) -> int | None:
#     if isinstance(value, bool):
#         return None

#     if isinstance(value, int):
#         return value if 1 <= value <= 5 else None

#     if isinstance(value, float):
#         if value.is_integer() and 1 <= int(value) <= 5:
#             return int(value)
#         return None

#     if isinstance(value, str):
#         text = value.strip()

#         if re.fullmatch(r"[1-5]", text):
#             return int(text)

#         if re.fullmatch(r"[1-5]\.0+", text):
#             return int(float(text))

#         m = re.search(r"\b([1-5])(?:\.0+)?\b", text)
#         if m:
#             return int(float(m.group(1)))

#     return None


# def parse_model_output(text: str) -> dict[str, Any]:
#     raw_text = preserve_text(text)

#     def normalize_key(key: str) -> str:
#         return re.sub(r"[^a-z0-9]", "", key.lower())

#     def pick_field(data: dict[str, Any], aliases: list[str]) -> Any:
#         if not isinstance(data, dict):
#             return None

#         normalized_map: dict[str, Any] = {}
#         for k, v in data.items():
#             if isinstance(k, str):
#                 normalized_map[normalize_key(k)] = v

#         for alias in aliases:
#             key = normalize_key(alias)
#             if key in normalized_map:
#                 return normalized_map[key]
#         return None

#     parsed = extract_json_object(raw_text)

#     if parsed is not None:
#         score_value = pick_field(parsed, ["score", "rating", "grade"])
#         reason_value = pick_field(parsed, ["reason", "feedback", "rationale", "justification"])
#         modified_value = pick_field(
#             parsed,
#             [
#                 "modified_answer",
#                 "modified answer",
#                 "revised_answer",
#                 "revised answer",
#                 "better_answer",
#                 "better answer",
#                 "improved_answer",
#                 "improved answer",
#                 "rewrite",
#                 "rewritten_answer",
#                 "rewritten answer",
#             ],
#         )

#         score = parse_int_score(score_value)
#         reason = preserve_text(reason_value)
#         modified_answer = preserve_text(modified_value)

#         parse_error = None
#         if score is None:
#             parse_error = "Failed to parse JSON score."
#         elif not reason:
#             parse_error = "Missing JSON field: reason."
#         elif not modified_answer:
#             parse_error = "Missing JSON field: modified_answer."

#         return {
#             "score": score,
#             "reason": reason,
#             "modified_answer": modified_answer,
#             "raw_output": raw_text,
#             "parsed_json": parsed,
#             "parse_mode": "json",
#             "parse_error": parse_error,
#             "strict_json_ok": score is not None and bool(reason) and bool(modified_answer),
#         }

#     score: int | None = None
#     reason = ""
#     modified_answer = ""
#     parse_error: str | None = None

#     score_match = SCORE_LINE_RE.search(raw_text)
#     if score_match:
#         score = parse_int_score(score_match.group(1))
#     else:
#         fallback_score = SCORE_FALLBACK_RE.search(raw_text)
#         if fallback_score:
#             score = parse_int_score(fallback_score.group(1))
#         else:
#             nearby = re.search(
#                 rf"(?is)\b(?:score|rating)\b.{{0,20}}?\b([1-5](?:\.0+)?)\b",
#                 raw_text,
#             )
#             if nearby:
#                 score = parse_int_score(nearby.group(1))
#             else:
#                 parse_error = "Failed to parse score."

#     reason_match = REASON_RE.search(raw_text)
#     modified_match = MODIFIED_RE.search(raw_text)

#     if reason_match:
#         reason = preserve_text(reason_match.group(1))

#     if modified_match:
#         modified_answer = preserve_text(modified_match.group(1))

#     if not reason:
#         lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
#         for line in lines:
#             if re.match(rf"(?is)^(?:reason|feedback|rationale|justification)\s*{COLON_CLASS}", line):
#                 reason = preserve_text(re.split(COLON_CLASS, line, maxsplit=1)[-1])
#                 break

#     if not modified_answer:
#         lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
#         for line in lines:
#             if re.match(
#                 rf"(?is)^(?:modified(?:\s+|_)?answer|revised(?:\s+|_)?answer|better(?:\s+|_)?answer|improved(?:\s+|_)?answer|rewrite)\s*{COLON_CLASS}",
#                 line,
#             ):
#                 modified_answer = preserve_text(re.split(COLON_CLASS, line, maxsplit=1)[-1])
#                 break

#     if score is not None and not reason and parse_error is None:
#         parse_error = "Failed to parse reason."
#     if score is not None and not modified_answer and parse_error is None:
#         parse_error = "Failed to parse modified_answer."

#     return {
#         "score": score,
#         "reason": reason,
#         "modified_answer": modified_answer,
#         "raw_output": raw_text,
#         "parsed_json": None,
#         "parse_mode": "regex",
#         "parse_error": parse_error,
#         "strict_json_ok": False,
#     }


# def _send_chat_request(
#     *,
#     url: str,
#     api_key: str,
#     payload: dict[str, Any],
#     timeout_seconds: int,
# ) -> dict[str, Any]:
#     req = urllib_request.Request(
#         url,
#         data=json.dumps(payload).encode("utf-8"),
#         headers={
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {api_key}",
#         },
#         method="POST",
#     )
#     with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
#         return json.loads(response.read().decode("utf-8"))


# def post_chat_completion(
#     *,
#     base_url: str,
#     api_key: str,
#     model: str,
#     messages: list[dict[str, str]],
#     temperature: float,
#     max_tokens: int,
#     timeout_seconds: int,
# ) -> str:
#     url = base_url.rstrip("/") + "/chat/completions"

#     payload: dict[str, Any] = {
#         "model": model,
#         "messages": messages,
#         "temperature": temperature,
#         "max_tokens": max_tokens,
#         "response_format": {"type": "json_object"},
#     }

#     try:
#         response_payload = _send_chat_request(
#             url=url,
#             api_key=api_key,
#             payload=payload,
#             timeout_seconds=timeout_seconds,
#         )
#     except urllib_error.HTTPError as exc:
#         if exc.code not in {400, 404, 422}:
#             raise
#         payload.pop("response_format", None)
#         response_payload = _send_chat_request(
#             url=url,
#             api_key=api_key,
#             payload=payload,
#             timeout_seconds=timeout_seconds,
#         )

#     choices = response_payload.get("choices") or []
#     if not choices:
#         raise RuntimeError(f"No choices returned from {url}")

#     message = choices[0].get("message") or {}
#     content = message.get("content")

#     if isinstance(content, str):
#         return content.strip()

#     if isinstance(content, list):
#         texts: list[str] = []
#         for item in content:
#             if isinstance(item, dict) and item.get("type") == "text":
#                 texts.append(str(item.get("text", "")))
#         return "\n".join(texts).strip()

#     return str(content or "").strip()


# def call_chat_with_retries(
#     *,
#     base_urls: list[str],
#     task_index: int,
#     api_key: str,
#     model: str,
#     messages: list[dict[str, str]],
#     temperature: float,
#     max_tokens: int,
#     timeout_seconds: int,
#     retries: int,
#     retry_sleep: float,
# ) -> tuple[str, str]:
#     last_error: Exception | None = None
#     endpoint_count = max(1, len(base_urls))

#     for attempt in range(retries + 1):
#         base_url = base_urls[(task_index + attempt) % endpoint_count]
#         try:
#             text = post_chat_completion(
#                 base_url=base_url,
#                 api_key=api_key,
#                 model=model,
#                 messages=messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 timeout_seconds=timeout_seconds,
#             )
#             return text, base_url
#         except Exception as exc:
#             last_error = exc
#             if attempt < retries:
#                 time.sleep(retry_sleep)

#     raise RuntimeError(str(last_error))


# def run_one_sample(
#     sample: dict[str, Any],
#     task_index: int,
#     args: argparse.Namespace,
#     base_urls: list[str],
#     model: str,
# ) -> dict[str, Any]:
#     started = time.time()

#     row = {
#         "sample_id": sample["sample_id"],
#         "question": sample["question"],
#         "answer": sample["answer"],
#         "dimension_name": sample["dimension_name"],
#         "score_criteria": sample["score_criteria"],
#         "predicted_score": None,
#         "predicted_reason": "",
#         "predicted_modified_answer": "",
#         "raw_output": "",
#         "parse_mode": "",
#         "strict_json_ok": False,
#         "endpoint": "",
#         "elapsed_seconds": None,
#         "ok": False,
#         "parse_error": None,
#     }

#     try:
#         raw_text, endpoint = call_chat_with_retries(
#             base_urls=base_urls,
#             task_index=task_index,
#             api_key=args.api_key,
#             model=model,
#             messages=sample["messages"],
#             temperature=args.temperature,
#             max_tokens=args.max_tokens,
#             timeout_seconds=args.request_timeout,
#             retries=args.retries,
#             retry_sleep=args.retry_sleep,
#         )

#         parsed = parse_model_output(raw_text)
#         ok = parsed["score"] is not None and bool(parsed["reason"]) and bool(parsed["modified_answer"])

#         row.update(
#             {
#                 "sample_id": sample["sample_id"],
#                 "predicted_score": parsed["score"],
#                 "predicted_reason": parsed["reason"],
#                 "predicted_modified_answer": parsed["modified_answer"],
#                 "raw_output": parsed["raw_output"],
#                 "parse_mode": parsed["parse_mode"],
#                 "strict_json_ok": parsed["strict_json_ok"],
#                 "endpoint": endpoint,
#                 "elapsed_seconds": round(time.time() - started, 4),
#                 "ok": ok,
#                 "parse_error": parsed["parse_error"],
#             }
#         )
#     except Exception as exc:
#         row.update(
#             {
#                 "elapsed_seconds": round(time.time() - started, 4),
#                 "ok": False,
#                 "parse_error": str(exc),
#             }
#         )

#     return row


# def load_completed_predictions(path: Path) -> dict[str, dict[str, Any]]:
#     completed: dict[str, dict[str, Any]] = {}
#     if not path.is_file():
#         return completed

#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             text = line.strip()
#             if not text:
#                 continue
#             try:
#                 row = json.loads(text)
#             except json.JSONDecodeError:
#                 continue

#             sample_id = row.get("sample_id")
#             if isinstance(sample_id, str) and sample_id:
#                 completed[sample_id] = row

#     return completed


# def reorder_output(path: Path, samples: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> None:
#     ordered = [completed[sample["sample_id"]] for sample in samples if sample["sample_id"] in completed]
#     write_jsonl(path, ordered)


# def run(args: argparse.Namespace) -> None:
#     input_path = Path(args.input)
#     output_path = Path(args.output)

#     if not input_path.is_file():
#         raise FileNotFoundError(f"Input file not found: {input_path}")

#     records = load_jsonl(input_path)
#     if args.limit is not None:
#         records = records[: max(0, args.limit)]
#     samples = prepare_samples(records)

#     base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
#     if not args.skip_health_check:
#         wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
#     model = resolve_model(args.model, base_urls, args.health_check_timeout)

#     if args.overwrite and output_path.exists():
#         output_path.unlink()

#     completed = load_completed_predictions(output_path) if args.resume else {}
#     pending = [
#         sample for sample in samples
#         if not (
#             sample["sample_id"] in completed
#             and completed[sample["sample_id"]].get("ok") is True
#         )
#     ]

#     print(
#         f"[score-generate] total={len(samples)} completed={len(completed)} "
#         f"pending={len(pending)} endpoints={base_urls} model={model} output={output_path}"
#     )

#     if not pending:
#         reorder_output(output_path, samples, completed)
#         return

#     write_lock = threading.Lock()
#     with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
#         futures = {
#             executor.submit(run_one_sample, sample, index, args, base_urls, model): sample
#             for index, sample in enumerate(pending)
#         }

#         pbar = tqdm(total=len(pending), desc="score-generate", dynamic_ncols=True)

#         for done, future in enumerate(as_completed(futures), start=1):
#             row = future.result()
#             completed[row["sample_id"]] = row
#             append_jsonl(output_path, row, write_lock)

#             ok_count = sum(1 for item in completed.values() if item.get("ok"))
#             pbar.update(1)
#             pbar.set_postfix(ok_total=ok_count)

#             if done % args.log_every == 0 or done == len(pending):
#                 print(f"[score-generate] done {done}/{len(pending)} pending, ok_total={ok_count}")

#         pbar.close()

#     if args.reorder:
#         reorder_output(output_path, samples, completed)


# def build_arg_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "Score Feedback-Bench answers and generate improved answers through "
#             "OpenAI-compatible vLLM endpoints."
#         )
#     )
#     parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
#     parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
#     parser.add_argument(
#         "--model",
#         default=DEFAULT_MODEL,
#         help="Model name served by vLLM. Use an empty string to auto-fetch from /models.",
#     )
#     parser.add_argument("--base-url-template", default=DEFAULT_BASE_URL_TEMPLATE)
#     parser.add_argument("--ports", nargs="*", default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"])
#     parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
#     parser.add_argument("--workers", type=int, default=32)
#     parser.add_argument("--temperature", type=float, default=0.0)
#     parser.add_argument("--max-tokens", type=int, default=768)
#     parser.add_argument("--request-timeout", type=int, default=300)
#     parser.add_argument("--retries", type=int, default=2)
#     parser.add_argument("--retry-sleep", type=float, default=1.0)
#     parser.add_argument("--limit", type=int, default=None)
#     parser.add_argument("--log-every", type=int, default=20)
#     parser.add_argument("--skip-health-check", action="store_true")
#     parser.add_argument("--health-check-timeout", type=int, default=10)
#     parser.add_argument("--health-check-interval", type=float, default=2.0)
#     parser.add_argument("--overwrite", action="store_true")
#     parser.add_argument("--no-resume", dest="resume", action="store_false")
#     parser.set_defaults(resume=True)
#     parser.add_argument(
#         "--no-reorder",
#         dest="reorder",
#         action="store_false",
#         help="Keep append completion order instead of rewriting output in input order at the end.",
#     )
#     parser.set_defaults(reorder=True)
#     return parser


# def main() -> None:
#     args = build_arg_parser().parse_args()
#     run(args)


# if __name__ == "__main__":
#     main()