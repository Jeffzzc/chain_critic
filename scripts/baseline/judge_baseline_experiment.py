#!/usr/bin/env python
"""
Run score-centric judge-model agreement experiments on 0-5 criteria datasets,
while also collecting rewritten answers from judges.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


load_dotenv()


# ============================================================
# Defaults
# ============================================================

DEFAULT_INPUT_PATH = Path("/data/dhf/chain_critic/datasets/train/final_test_split.jsonl")
DEFAULT_OUTPUT_DIR = Path("/data/dhf/chain_critic/evaluation/baseline_new")
DEFAULT_JUDGE_CONFIG_PATH = Path("scripts/baseline/judge_model_config.example.json")

COLON_CLASS = r"[:\uFF1A]"


# ============================================================
# Regex patterns
# Reference output format and judge output format are both 3 lines:
# Score:
# Reason:
# Modified Answer:
# ============================================================

REFERENCE_SCORE_PATTERN = re.compile(
    rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$"
)
REFERENCE_REASON_PATTERN = re.compile(
    rf"(?is)reason\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?modified answer\s*{COLON_CLASS}|$)"
)
REFERENCE_MODIFIED_ANSWER_PATTERN = re.compile(
    rf"(?is)modified answer\s*{COLON_CLASS}\s*(.*)$"
)

JUDGE_SCORE_PATTERN = re.compile(
    rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$"
)
JUDGE_REASON_PATTERN = re.compile(
    rf"(?is)reason\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?modified answer\s*{COLON_CLASS}|$)"
)
JUDGE_MODIFIED_ANSWER_PATTERN = re.compile(
    rf"(?is)modified answer\s*{COLON_CLASS}\s*(.*)$"
)


# ============================================================
# User message section aliases
# ============================================================

USER_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question",),
    "answer": ("answer",),
    "evaluation_dimension": ("evaluation_dimension", "evaluation dimension", "dimension"),
    "criteria": ("criteria",),
}


# ============================================================
# System prompt
# ============================================================

SCORE_WITH_REWRITE_SYSTEM_PROMPT = (
    "You are an AI evaluator-and-rewriter.\n"
    "Evaluate the given answer strictly using ONLY the provided evaluation dimension and the complete 0-5 scoring criteria.\n"
    "Then revise the answer to better satisfy ONLY that dimension.\n"
    "Do not add unsupported facts. If an assumption is necessary, state it minimally and explicitly.\n"
    "Output plain text in exactly 3 lines, with exactly these prefixes and no numbering:\n"
    "Score: <an integer number from 0 to 5>\n"
    "Reason: <one-line concise explanation strictly based on the given dimension criteria>\n"
    "Modified Answer: <one-line revised answer optimized only for the given dimension criteria>\n"
    "Do not include any extra text, JSON, markdown, bullets, or line breaks inside any field."
)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class JudgeConfig:
    name: str
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: float = 120.0
    max_retries: int = 5
    retry_delay: float = 3.0
    system_override: Optional[str] = None

    host: str = "127.0.0.1"
    port: Optional[int] = None
    base_urls: tuple[str, ...] = ()
    ports: tuple[int, ...] = ()

    auto_fetch_model: bool = True
    healthcheck: bool = True
    healthcheck_retries: int = 60
    healthcheck_interval: float = 2.0


@dataclass
class ParsedReferenceOutput:
    score: Optional[float]
    reason: str
    modified_answer: str
    raw_text: str
    parse_error: Optional[str] = None
    strict_format_ok: bool = False


@dataclass
class ParsedJudgeOutput:
    score: Optional[float]
    reason: str
    modified_answer: str
    raw_text: str
    parse_error: Optional[str] = None
    strict_format_ok: bool = False


# ============================================================
# IO helpers
# ============================================================

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            rows.append(payload)
    return rows


# ============================================================
# Text parsing helpers
# ============================================================

def canonicalize_header_name(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text).strip().lower())


def alias_to_regex(alias: str) -> str:
    tokens = [re.escape(token) for token in re.split(r"[\s_\-]+", alias.strip()) if token]
    return r"[\s_\-]*".join(tokens)


def extract_sections_by_headers(
    text: str,
    header_map: dict[str, tuple[str, ...]],
) -> dict[str, str]:
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
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        raw_header = canonicalize_header_name(match.group("header"))
        section_key = header_to_key.get(raw_header)
        if not section_key:
            continue

        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[section_key] = text[value_start:value_end].strip()

    return sections


def build_sample_id(messages: list[dict[str, str]], fallback_index: int) -> str:
    user_content = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "user"),
        "",
    )
    assistant_content = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "assistant"),
        "",
    )
    raw = f"{fallback_index}||{user_content}||{assistant_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalize_single_line(text: str) -> str:
    return " ".join(str(text).split()).strip()


# ============================================================
# Output parsers
# ============================================================

def parse_reference_output(text: str) -> ParsedReferenceOutput:
    raw_text = text.strip().replace("\r\n", "\n")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    if len(lines) == 3:
        if (
            lines[0].startswith("Score:")
            and lines[1].startswith("Reason:")
            and lines[2].startswith("Modified Answer:")
        ):
            score_text = lines[0][len("Score:"):].strip()
            if re.fullmatch(r"[0-5]", score_text):
                return ParsedReferenceOutput(
                    score=float(score_text),
                    reason=normalize_single_line(lines[1][len("Reason:"):].strip()),
                    modified_answer=normalize_single_line(lines[2][len("Modified Answer:"):].strip()),
                    raw_text=raw_text,
                    parse_error=None,
                    strict_format_ok=True,
                )

    score: Optional[float] = None
    parse_error: Optional[str] = None
    reason = ""
    modified_answer = ""

    score_match = REFERENCE_SCORE_PATTERN.search(raw_text)
    if score_match:
        score = float(score_match.group(1))
    else:
        fallback = re.search(rf"(?i)\bscore\s*{COLON_CLASS}\s*([0-5])", raw_text)
        if fallback:
            score = float(fallback.group(1))
        else:
            parse_error = "Failed to parse reference score."

    reason_match = REFERENCE_REASON_PATTERN.search(raw_text)
    modified_answer_match = REFERENCE_MODIFIED_ANSWER_PATTERN.search(raw_text)

    if reason_match:
        reason = normalize_single_line(reason_match.group(1))
    if modified_answer_match:
        modified_answer = normalize_single_line(modified_answer_match.group(1))

    return ParsedReferenceOutput(
        score=score,
        reason=reason,
        modified_answer=modified_answer,
        raw_text=raw_text,
        parse_error=parse_error,
        strict_format_ok=False,
    )


def parse_judge_output(text: str) -> ParsedJudgeOutput:
    raw_text = text.strip().replace("\r\n", "\n")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    if len(lines) == 3:
        if (
            lines[0].startswith("Score:")
            and lines[1].startswith("Reason:")
            and lines[2].startswith("Modified Answer:")
        ):
            score_text = lines[0][len("Score:"):].strip()
            if re.fullmatch(r"[0-5]", score_text):
                return ParsedJudgeOutput(
                    score=float(score_text),
                    reason=normalize_single_line(lines[1][len("Reason:"):].strip()),
                    modified_answer=normalize_single_line(lines[2][len("Modified Answer:"):].strip()),
                    raw_text=raw_text,
                    parse_error=None,
                    strict_format_ok=True,
                )

    score: Optional[float] = None
    parse_error: Optional[str] = None
    reason = ""
    modified_answer = ""

    score_match = JUDGE_SCORE_PATTERN.search(raw_text)
    if score_match:
        score = float(score_match.group(1))
    else:
        fallback = re.search(rf"(?i)\bscore\s*{COLON_CLASS}\s*([0-5])", raw_text)
        if fallback:
            score = float(fallback.group(1))
        else:
            parse_error = "Failed to parse score."

    reason_match = JUDGE_REASON_PATTERN.search(raw_text)
    modified_answer_match = JUDGE_MODIFIED_ANSWER_PATTERN.search(raw_text)

    if reason_match:
        reason = normalize_single_line(reason_match.group(1))
    if modified_answer_match:
        modified_answer = normalize_single_line(modified_answer_match.group(1))

    return ParsedJudgeOutput(
        score=score,
        reason=reason,
        modified_answer=modified_answer,
        raw_text=raw_text,
        parse_error=parse_error,
        strict_format_ok=False,
    )


# ============================================================
# Prompt building
# ============================================================

def build_user_message(
    *,
    question: str,
    answer: str,
    evaluation_dimension: str,
    criteria: str,
) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Evaluation_dimension:\n"
        f"{evaluation_dimension}\n\n"
        "Criteria (0-5):\n"
        f"{criteria}"
    )


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Record {index} is missing valid messages.")

        user_message = None
        assistant_message = None

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            if role == "user" and user_message is None:
                user_message = str(message.get("content", ""))
            elif role == "assistant" and assistant_message is None:
                assistant_message = str(message.get("content", ""))

        if not user_message or not assistant_message:
            raise ValueError(f"Record {index} is missing user/assistant content.")

        sections = extract_sections_by_headers(user_message, USER_SECTION_ALIASES)
        question = sections.get("question", "").strip()
        answer = sections.get("answer", "").strip()
        evaluation_dimension = sections.get("evaluation_dimension", "").strip()
        criteria = sections.get("criteria", "").strip()

        missing_fields = [
            name
            for name, value in (
                ("question", question),
                ("answer", answer),
                ("evaluation_dimension", evaluation_dimension),
                ("criteria", criteria),
            )
            if not value
        ]
        if missing_fields:
            raise ValueError(
                f"Record {index} missing required fields after parsing user content: {missing_fields}"
            )

        reference_parsed = parse_reference_output(assistant_message)
        if reference_parsed.score is None:
            raise ValueError(
                f"Record {index} reference assistant output has no parsable score. "
                f"parse_error={reference_parsed.parse_error}"
            )

        sample_id = build_sample_id(messages, index)

        prompt_messages = [
            {"role": "system", "content": SCORE_WITH_REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_message(
                    question=question,
                    answer=answer,
                    evaluation_dimension=evaluation_dimension,
                    criteria=criteria,
                ),
            },
        ]

        prepared.append(
            {
                "sample_id": sample_id,
                "index": index,
                "question": question,
                "answer": answer,
                "evaluation_dimension": evaluation_dimension,
                "criteria": criteria,
                "prompt_messages": prompt_messages,
                "reference_output": assistant_message,
                "reference_score": reference_parsed.score,
                "reference_reason": reference_parsed.reason,
                "reference_modified_answer": reference_parsed.modified_answer,
                "reference_parse_error": reference_parsed.parse_error,
                "reference_format_ok": reference_parsed.strict_format_ok,
                "original_user_message": user_message,
            }
        )

    return prepared


# ============================================================
# Judge config loading
# ============================================================

def load_judge_configs(path: Path) -> list[JudgeConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        judges_payload = payload.get("judges", [])
    elif isinstance(payload, list):
        judges_payload = payload
    else:
        raise ValueError("Judge config must be a list or an object with key 'judges'.")

    judges: list[JudgeConfig] = []

    for idx, item in enumerate(judges_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Judge entry {idx} is not a JSON object.")

        name = str(item.get("name", "")).strip()
        host = str(item.get("host", "127.0.0.1")).strip()
        port = item.get("port")
        ports = parse_ports(item.get("ports"))

        raw_base_url = str(item.get("base_url", "")).strip()
        raw_base_urls = item.get("base_urls")
        base_urls = normalize_base_urls(
            raw_base_url,
            raw_base_urls,
            host=host,
            port=port,
            ports=ports,
        )
        base_url = base_urls[0] if base_urls else ""

        model = str(item.get("model", "")).strip()

        if not name or not base_urls:
            raise ValueError(
                f"Judge entry {idx} must contain name and either "
                f"base_url/base_urls or host+port/ports."
            )

        api_key = item.get("api_key")
        api_key_env = item.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env), "")
        if not api_key:
            api_key = "EMPTY"

        judges.append(
            JudgeConfig(
                name=name,
                base_url=base_url,
                model=model,
                api_key=str(api_key),
                temperature=float(item.get("temperature", 0.0)),
                max_tokens=int(item.get("max_tokens", 512)),
                timeout=float(item.get("timeout", 120.0)),
                max_retries=int(item.get("max_retries", 5)),
                retry_delay=float(item.get("retry_delay", 3.0)),
                system_override=(
                    str(item.get("system_override")).strip()
                    if item.get("system_override") is not None
                    else None
                ),
                host=host,
                port=int(port) if port is not None else None,
                base_urls=tuple(base_urls),
                ports=ports,
                auto_fetch_model=bool(item.get("auto_fetch_model", True)),
                healthcheck=bool(item.get("healthcheck", True)),
                healthcheck_retries=int(item.get("healthcheck_retries", 60)),
                healthcheck_interval=float(item.get("healthcheck_interval", 2.0)),
            )
        )

    return judges


def parse_ports(raw_ports: Any) -> tuple[int, ...]:
    if raw_ports is None or raw_ports == "":
        return ()

    if not isinstance(raw_ports, list):
        raise ValueError("'ports' must be a JSON array of integers.")

    parsed_ports: list[int] = []
    for idx, raw_port in enumerate(raw_ports):
        try:
            parsed_ports.append(int(raw_port))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid port at ports[{idx}]: {raw_port}") from exc

    return tuple(parsed_ports)


def normalize_base_url(raw_base_url: str, *, host: str, port: Any) -> str:
    if raw_base_url:
        base_url = raw_base_url.rstrip("/")
    elif port is not None:
        client_host = normalize_client_host(host)
        base_url = f"http://{client_host}:{int(port)}/v1"
    else:
        return ""

    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    return base_url


def normalize_base_urls(
    raw_base_url: str,
    raw_base_urls: Any,
    *,
    host: str,
    port: Any,
    ports: tuple[int, ...],
) -> list[str]:
    base_urls: list[str] = []

    if raw_base_urls is not None:
        if not isinstance(raw_base_urls, list):
            raise ValueError("'base_urls' must be a JSON array of strings.")
        for idx, raw_url in enumerate(raw_base_urls):
            text = str(raw_url).strip()
            if not text:
                raise ValueError(f"Invalid base_urls[{idx}]: empty string")
            base_urls.append(normalize_base_url(text, host=host, port=None))
    elif ports:
        for item_port in ports:
            base_urls.append(normalize_base_url("", host=host, port=item_port))
    else:
        single_base_url = normalize_base_url(raw_base_url, host=host, port=port)
        if single_base_url:
            base_urls.append(single_base_url)

    return dedupe_preserve_order(base_urls)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)

    return deduped


def normalize_client_host(host: str) -> str:
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized or "127.0.0.1"


def is_loopback_base_url(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


# ============================================================
# Proxy / client helpers
# ============================================================

@contextmanager
def suspended_proxy_env() -> Any:
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
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


def build_openai_client(base_url: str, api_key: str, timeout: float) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required to run judge requests. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    if is_loopback_base_url(base_url):
        with suspended_proxy_env():
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def fetch_model_id(base_url: str, timeout: float) -> Optional[str]:
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib_request.urlopen(models_url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None

    if isinstance(payload, dict):
        models = payload.get("data", [])
        if models and isinstance(models[0], dict):
            model_id = models[0].get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()

    return None


def wait_for_vllm_ready(
    base_url: str,
    *,
    timeout: float,
    retries: int,
    interval: float,
) -> None:
    if retries <= 0:
        return

    base_url = base_url.rstrip("/")
    health_urls = [
        base_url[:-3] + "/health" if base_url.endswith("/v1") else base_url + "/health",
        base_url[:-3] + "/ping" if base_url.endswith("/v1") else base_url + "/ping",
        base_url + "/models",
    ]

    timeout = min(10.0, timeout)
    last_error: Optional[str] = None

    for _ in range(max(1, retries)):
        for url in health_urls:
            try:
                with urllib_request.urlopen(url, timeout=timeout) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = str(exc)

        time.sleep(max(0.1, interval))

    raise RuntimeError(
        "vLLM endpoint not ready. Tried "
        f"{health_urls}. Last error: {last_error}"
    )


# ============================================================
# Metrics
# ============================================================

def average(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None

    mean_x = average(x)
    mean_y = average(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]

    sum_x2 = sum(value * value for value in dx)
    sum_y2 = sum(value * value for value in dy)
    if sum_x2 == 0 or sum_y2 == 0:
        return None

    numerator = sum(a * b for a, b in zip(dx, dy))
    return numerator / math.sqrt(sum_x2 * sum_y2)


def rankdata(values: list[float]) -> list[float]:
    pairs = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1

        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            original_idx = pairs[k][0]
            ranks[original_idx] = avg_rank

        i = j

    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson_correlation(rankdata(x), rankdata(y))


def kendall_tau_b(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue

            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1

    denominator = math.sqrt(
        (concordant + discordant + ties_x) *
        (concordant + discordant + ties_y)
    )
    if denominator == 0:
        return None

    return (concordant - discordant) / denominator


def format_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value, 6)


def compute_metrics(reference_scores: list[float], predicted_scores: list[float]) -> dict[str, Any]:
    if len(reference_scores) != len(predicted_scores):
        raise ValueError("Reference and predicted score lengths do not match.")

    abs_errors = [abs(a - b) for a, b in zip(reference_scores, predicted_scores)]
    sq_errors = [(a - b) ** 2 for a, b in zip(reference_scores, predicted_scores)]

    exact_match = sum(1 for a, b in zip(reference_scores, predicted_scores) if a == b)
    off_by_1 = sum(1 for a, b in zip(reference_scores, predicted_scores) if abs(a - b) <= 1)

    return {
        "count": len(reference_scores),
        "reference_mean": format_float(average(reference_scores)) if reference_scores else None,
        "predicted_mean": format_float(average(predicted_scores)) if predicted_scores else None,
        "mae": format_float(average(abs_errors)) if abs_errors else None,
        "rmse": format_float(math.sqrt(average(sq_errors))) if sq_errors else None,
        "pearson": format_float(pearson_correlation(reference_scores, predicted_scores)),
        "spearman": format_float(spearman_correlation(reference_scores, predicted_scores)),
        "kendall_tau": format_float(kendall_tau_b(reference_scores, predicted_scores)),
        "exact_match_accuracy": format_float(exact_match / len(reference_scores)) if reference_scores else None,
        "off_by_1_accuracy": format_float(off_by_1 / len(reference_scores)) if reference_scores else None,
    }


# ============================================================
# Judge calling
# ============================================================

def call_judge(
    sample: dict[str, Any],
    judge: JudgeConfig,
    client: OpenAI,
) -> dict[str, Any]:
    messages = []
    for message in sample["prompt_messages"]:
        if message["role"] == "system" and judge.system_override:
            messages.append({"role": "system", "content": judge.system_override})
        else:
            messages.append(message)

    last_error: Optional[str] = None

    for attempt in range(judge.max_retries):
        try:
            completion = client.chat.completions.create(
                model=judge.model,
                messages=messages,
                temperature=judge.temperature,
                max_tokens=judge.max_tokens,
            )
            text = completion.choices[0].message.content or ""
            parsed = parse_judge_output(text)

            return {
                "sample_id": sample["sample_id"],
                "index": sample["index"],
                "question": sample["question"],
                "answer": sample["answer"],
                "evaluation_dimension": sample["evaluation_dimension"],
                "criteria": sample["criteria"],
                "reference_score": sample["reference_score"],
                "reference_reason": sample["reference_reason"],
                "reference_modified_answer": sample["reference_modified_answer"],
                "reference_output": sample["reference_output"],
                "reference_format_ok": sample["reference_format_ok"],
                "judge_name": judge.name,
                "judge_model": judge.model,
                "judge_base_url": judge.base_url,
                "predicted_score": parsed.score,
                "predicted_reason": parsed.reason,
                "predicted_modified_answer": parsed.modified_answer,
                "raw_output": parsed.raw_text,
                "strict_format_ok": parsed.strict_format_ok,
                "parse_error": parsed.parse_error,
                "request_error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < judge.max_retries - 1:
                delay = judge.retry_delay * (attempt + 1)
                time.sleep(delay)
            else:
                break

    return {
        "sample_id": sample["sample_id"],
        "index": sample["index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "evaluation_dimension": sample["evaluation_dimension"],
        "criteria": sample["criteria"],
        "reference_score": sample["reference_score"],
        "reference_reason": sample["reference_reason"],
        "reference_modified_answer": sample["reference_modified_answer"],
        "reference_output": sample["reference_output"],
        "reference_format_ok": sample["reference_format_ok"],
        "judge_name": judge.name,
        "judge_model": judge.model,
        "judge_base_url": judge.base_url,
        "predicted_score": None,
        "predicted_reason": "",
        "predicted_modified_answer": "",
        "raw_output": "",
        "strict_format_ok": False,
        "parse_error": None,
        "request_error": last_error,
    }


# ============================================================
# Prediction file helpers
# ============================================================

def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return existing

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            if isinstance(sample_id, str):
                existing[sample_id] = payload

    return existing


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ============================================================
# Main per-judge runner
# ============================================================

def run_single_judge(
    samples: list[dict[str, Any]],
    judge: JudgeConfig,
    output_dir: Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{judge.name}.jsonl"

    if not resume and prediction_path.exists():
        prediction_path.unlink()

    endpoint_base_urls = list(judge.base_urls) if judge.base_urls else [judge.base_url]
    resolved_endpoints: list[JudgeConfig] = []
    endpoint_ports = judge.ports if judge.ports else ((judge.port,) if judge.port is not None else ())

    for idx, endpoint_base_url in enumerate(endpoint_base_urls):
        if judge.healthcheck:
            wait_for_vllm_ready(
                endpoint_base_url,
                timeout=judge.timeout,
                retries=judge.healthcheck_retries,
                interval=judge.healthcheck_interval,
            )

        resolved_model = judge.model.strip()
        if not resolved_model and judge.auto_fetch_model:
            resolved_model = fetch_model_id(
                endpoint_base_url,
                timeout=min(10.0, judge.timeout),
            ) or ""

        if not resolved_model:
            raise ValueError(
                f"Judge '{judge.name}' has no model configured, and auto-fetch from "
                f"{endpoint_base_url}/models did not return a usable model id."
            )

        resolved_endpoints.append(
            JudgeConfig(
                name=judge.name,
                base_url=endpoint_base_url,
                model=resolved_model,
                api_key=judge.api_key,
                temperature=judge.temperature,
                max_tokens=judge.max_tokens,
                timeout=judge.timeout,
                max_retries=judge.max_retries,
                retry_delay=judge.retry_delay,
                system_override=judge.system_override,
                host=judge.host,
                port=endpoint_ports[idx] if idx < len(endpoint_ports) else None,
                base_urls=(endpoint_base_url,),
                ports=(endpoint_ports[idx],) if idx < len(endpoint_ports) else (),
                auto_fetch_model=judge.auto_fetch_model,
                healthcheck=judge.healthcheck,
                healthcheck_retries=judge.healthcheck_retries,
                healthcheck_interval=judge.healthcheck_interval,
            )
        )

    primary_endpoint = resolved_endpoints[0]
    resolved_judge = JudgeConfig(
        name=judge.name,
        base_url=primary_endpoint.base_url,
        model=primary_endpoint.model,
        api_key=judge.api_key,
        temperature=judge.temperature,
        max_tokens=judge.max_tokens,
        timeout=judge.timeout,
        max_retries=judge.max_retries,
        retry_delay=judge.retry_delay,
        system_override=judge.system_override,
        host=judge.host,
        port=primary_endpoint.port,
        base_urls=tuple(endpoint.base_url for endpoint in resolved_endpoints),
        ports=judge.ports,
        auto_fetch_model=judge.auto_fetch_model,
        healthcheck=judge.healthcheck,
        healthcheck_retries=judge.healthcheck_retries,
        healthcheck_interval=judge.healthcheck_interval,
    )

    print(
        f"[Judge] {resolved_judge.name} -> "
        f"endpoints={len(resolved_endpoints)}, "
        f"base_urls={list(resolved_judge.base_urls)}, "
        f"models={sorted({endpoint.model for endpoint in resolved_endpoints})}"
    )

    existing_predictions = load_existing_predictions(prediction_path) if resume else {}
    samples_to_run = [sample for sample in samples if sample["sample_id"] not in existing_predictions]

    clients = {
        endpoint.base_url: build_openai_client(
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            timeout=endpoint.timeout,
        )
        for endpoint in resolved_endpoints
    }

    write_lock = threading.Lock()
    progress = tqdm(total=len(samples), desc=judge.name, ncols=100) if tqdm is not None else None
    if progress is not None and existing_predictions:
        progress.update(len(existing_predictions))

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(
                    call_judge,
                    sample,
                    resolved_endpoints[index % len(resolved_endpoints)],
                    clients[resolved_endpoints[index % len(resolved_endpoints)].base_url],
                )
                for index, sample in enumerate(samples_to_run)
            ]

            for future in as_completed(futures):
                result = future.result()
                existing_predictions[result["sample_id"]] = result
                append_jsonl(prediction_path, result, write_lock)
                if progress is not None:
                    progress.update(1)

    finally:
        if progress is not None:
            progress.close()

        for client in clients.values():
            try:
                client.close()
            except Exception:
                pass

    ordered_predictions = [
        existing_predictions[sample["sample_id"]]
        for sample in sorted(samples, key=lambda item: item["index"])
        if sample["sample_id"] in existing_predictions
    ]

    valid_pairs = [
        (row["reference_score"], row["predicted_score"])
        for row in ordered_predictions
        if isinstance(row.get("reference_score"), (int, float))
        and isinstance(row.get("predicted_score"), (int, float))
    ]

    reference_scores = [pair[0] for pair in valid_pairs]
    predicted_scores = [pair[1] for pair in valid_pairs]

    request_failures = sum(1 for row in ordered_predictions if row.get("request_error"))
    parse_failures = sum(1 for row in ordered_predictions if row.get("parse_error"))
    strict_format_successes = sum(1 for row in ordered_predictions if row.get("strict_format_ok"))

    summary = {
        "judge": asdict(resolved_judge),
        "resolved_endpoints": [
            {
                "base_url": endpoint.base_url,
                "model": endpoint.model,
            }
            for endpoint in resolved_endpoints
        ],
        "prediction_file": str(prediction_path),
        "total_samples": len(samples),
        "completed_samples": len(ordered_predictions),
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "strict_format_successes": strict_format_successes,
        "strict_format_success_rate": (
            format_float(strict_format_successes / len(ordered_predictions))
            if ordered_predictions
            else None
        ),
        "metrics": compute_metrics(reference_scores, predicted_scores),
    }

    return summary


# ============================================================
# Filtering / experiment summary
# ============================================================

def maybe_limit_samples(samples: list[dict[str, Any]], sample_size: Optional[int]) -> list[dict[str, Any]]:
    if sample_size is None:
        return samples
    return samples[: max(0, sample_size)]


def filter_judges(judges: list[JudgeConfig], only_models: list[str]) -> list[JudgeConfig]:
    if not only_models:
        return judges
    wanted = {name.strip() for name in only_models if name.strip()}
    return [judge for judge in judges if judge.name in wanted]


def build_experiment_summary(
    input_path: Path,
    judge_config_path: Path,
    sample_count: int,
    judge_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    def metric_sort_key(item: dict[str, Any]) -> tuple[bool, float, float, float]:
        exact_match = item["metrics"]["exact_match_accuracy"]
        off_by_1 = item["metrics"]["off_by_1_accuracy"]
        spearman = item["metrics"]["spearman"]
        return (
            exact_match is None,
            -(exact_match if exact_match is not None else float("-inf")),
            -(off_by_1 if off_by_1 is not None else float("-inf")),
            -(spearman if spearman is not None else float("-inf")),
        )

    ranking = sorted(judge_summaries, key=metric_sort_key)

    return {
        "input_path": str(input_path),
        "judge_config_path": str(judge_config_path),
        "sample_count": sample_count,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_mode": "score_with_rewrite_but_score_metric_only",
        "score_range": "0-5",
        "judge_summaries": judge_summaries,
        "ranking": [
            {
                "judge_name": item["judge"]["name"],
                "exact_match_accuracy": item["metrics"]["exact_match_accuracy"],
                "off_by_1_accuracy": item["metrics"]["off_by_1_accuracy"],
                "spearman": item["metrics"]["spearman"],
                "pearson": item["metrics"]["pearson"],
                "kendall_tau": item["metrics"]["kendall_tau"],
                "valid_count": item["metrics"]["count"],
                "strict_format_success_rate": item["strict_format_success_rate"],
            }
            for item in ranking
        ],
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run score-centric judge-model baseline experiments on a 0-5 criteria "
            "JSONL test set, while collecting modified answers."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the JSONL test file. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=DEFAULT_JUDGE_CONFIG_PATH,
        help=f"Path to judge model config JSON file. Default: {DEFAULT_JUDGE_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for predictions and summary files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of samples to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=96,
        help="Worker threads per judge model.",
    )
    parser.add_argument(
        "--only-models",
        nargs="*",
        default=[],
        help="Optional judge names to run from the config file.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from existing prediction files.",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.judge_config.is_file():
        raise FileNotFoundError(f"Judge config file not found: {args.judge_config}")

    records = read_jsonl(args.input)
    samples = maybe_limit_samples(prepare_dataset(records), args.sample_size)
    judges = filter_judges(load_judge_configs(args.judge_config), args.only_models)

    if not judges:
        raise ValueError("No judge models selected to run.")

    predictions_dir = args.output_dir / "predictions"
    summaries_dir = args.output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    judge_summaries: list[dict[str, Any]] = []

    for judge in judges:
        summary = run_single_judge(
            samples=samples,
            judge=judge,
            output_dir=predictions_dir,
            workers=args.workers,
            resume=not args.no_resume,
        )
        judge_summaries.append(summary)

    experiment_summary = build_experiment_summary(
        input_path=args.input,
        judge_config_path=args.judge_config,
        sample_count=len(samples),
        judge_summaries=judge_summaries,
    )

    summary_path = summaries_dir / "summary.json"
    csv_path = summaries_dir / "summary.csv"

    summary_path.write_text(
        json.dumps(experiment_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_lines = [
        "judge_name,valid_count,exact_match_accuracy,off_by_1_accuracy,pearson,spearman,kendall_tau,mae,rmse,strict_format_success_rate,request_failures,parse_failures"
    ]
    for item in judge_summaries:
        metrics = item["metrics"]
        csv_lines.append(
            ",".join(
                [
                    item["judge"]["name"],
                    str(metrics["count"]),
                    str(metrics["exact_match_accuracy"]),
                    str(metrics["off_by_1_accuracy"]),
                    str(metrics["pearson"]),
                    str(metrics["spearman"]),
                    str(metrics["kendall_tau"]),
                    str(metrics["mae"]),
                    str(metrics["rmse"]),
                    str(item["strict_format_success_rate"]),
                    str(item["request_failures"]),
                    str(item["parse_failures"]),
                ]
            )
        )

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(json.dumps(experiment_summary["ranking"], ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
