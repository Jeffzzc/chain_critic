#!/usr/bin/env python
"""
Generate judge outputs from final_test_split.jsonl via OpenAI API or local vLLM.

This version is designed for datasets in the following format:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "...Question / Answer / Evaluation_dimension / Criteria..."},
    {"role": "assistant", "content": "Score: ...\nReason: ...\nModified Answer: ..."}
  ]
}

It will:
1. Send the original system+user messages to the judge model.
2. Parse the assistant message as reference output.
3. Save both reference_* and predicted_* fields for later comparison.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

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
DEFAULT_OUTPUT_DIR = Path("/data/dhf/chain_critic/evaluation/baseline_new/predictions")

DEFAULT_OPENAI_BASE_URL = "https://llm.api.zyuncs.com/v1"
DEFAULT_OPENAI_MODEL = "cortex-5"
DEFAULT_OPENAI_WORKERS = 5
DEFAULT_VLLM_WORKERS = 16

DEFAULT_OPENAI_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_OPENAI_BEARER_TOKEN_ENV = "OPENAI_BEARER_TOKEN"
DEFAULT_JUDGE_NAME_ENV = "JUDGE_NAME"

DEFAULT_VLLM_HOST_ENV = "VLLM_HOST"
DEFAULT_VLLM_PORT_ENV = "VLLM_PORT"
DEFAULT_VLLM_MODEL_ENV = "VLLM_MODEL"
DEFAULT_VLLM_BASE_URL_ENV = "VLLM_BASE_URL"

DEFAULT_STRICT_JUDGING_ENV = "STRICT_JUDGING"
DEFAULT_STRICT_PROMPT_ENV = "STRICT_JUDGING_PROMPT"


DEFAULT_STRICT_JUDGING_PROMPT = """Apply a strict rubric-based grading policy. Evaluate the ORIGINAL answer only.

Use ONLY the following inputs for scoring:
- the question (Q),
- the ORIGINAL answer (A),
- the target evaluation dimension,
- the complete 0-5 rubric for that dimension.

Scoring rules:
- Assign exactly one integer score from 0 to 5.
- Treat the rubric as the source of truth. Match the ORIGINAL answer to the single rubric level it best fits.
- Read all six score levels before deciding. Do not score by checking only the top level or by asking whether the answer deserves full marks.
- Judge in this order:
  1) correctness of the content and final conclusion relative to Q,
  2) logical soundness of the reasoning in the ORIGINAL answer,
  3) which 0-5 rubric level the ORIGINAL answer most closely matches for the target evaluation dimension.
- First check for hard errors before assigning any score. Hard errors include:
  - numerical mistakes,
  - semantic misunderstanding of the question,
  - contradictions,
  - invalid logical steps,
  - unsupported assumptions,
  - conclusions that do not follow from the reasoning.
- Any clear hard error MUST lower the score to a rubric level consistent with that error.
- Never give a 4 or 5 to an answer with a clear numerical, semantic, factual, or strict logical error.
- Do not let fluency, formatting, or partial alignment with one rubric phrase offset a clear hard error.
- Use the language of the provided rubric levels directly. If the answer clearly matches the description of 0, 1, 2, 3, or 4, assign that score even if some parts look superficially strong.
- For missing steps, missing units, incomplete conclusions, ambiguous phrasing, or partial satisfaction of the dimension, choose the rubric level that explicitly describes that failure mode.
- Treat 5 as exceptional, not normal. Give 5 only when the ORIGINAL answer fully matches the score-5 rubric description with no meaningful weakness relative to the dimension.
- If the answer has any noticeable weakness, omission, ambiguity, incomplete justification, unsupported assumption, or criterion-level gap relative to the rubric, do NOT give 5.
- When deciding between adjacent scores, prefer the lower score unless the higher score is clearly supported by the rubric text.
- Before assigning the final score, compare the answer against the neighboring rubric levels and choose the best fit, not the most generous fit.
- The Reason line should justify the chosen score by citing the relevant rubric-aligned behavior of the ORIGINAL answer, ideally making clear why it is this score instead of nearby scores.
"""


# ============================================================
# Regex
# ============================================================

COLON_CLASS = r"[:\uFF1A]"

SCORE_PATTERN = re.compile(rf"(?im)^\s*score\s*{COLON_CLASS}\s*([0-5])\s*$")
REASON_PATTERN = re.compile(
    rf"(?is)reason\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?modified answer\s*{COLON_CLASS}|$)"
)
MODIFIED_ANSWER_PATTERN = re.compile(
    rf"(?is)modified answer\s*{COLON_CLASS}\s*(.*)$"
)

USER_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "question": ("question",),
    "answer": ("answer",),
    "evaluation_dimension": ("evaluation_dimension", "evaluation dimension", "dimension"),
    "criteria": ("criteria", "rubric", "0-5 rubric", "full_score_criteria", "full score criteria"),
}


# ============================================================
# Data classes
# ============================================================

@dataclass
class ParsedOutput:
    score: Optional[float]
    reason: str
    modified_answer: str
    raw_output: str
    parse_error: Optional[str] = None
    strict_format_ok: bool = False


# ============================================================
# Basic helpers
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


def normalize_single_line(text: str) -> str:
    return " ".join(str(text).split()).strip()


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
    system_content = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "system"),
        "",
    )
    user_content = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "user"),
        "",
    )
    assistant_content = next(
        (str(message.get("content", "")) for message in messages if message.get("role") == "assistant"),
        "",
    )
    raw = f"{fallback_index}||{system_content}||{user_content}||{assistant_content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ============================================================
# Output parsing
# ============================================================

def parse_three_line_output(text: str) -> ParsedOutput:
    raw_text = text.strip().replace("\r\n", "\n")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

    # Strict case: exactly 3 lines with exact prefixes
    if len(lines) == 3:
        if (
            lines[0].startswith("Score:")
            and lines[1].startswith("Reason:")
            and lines[2].startswith("Modified Answer:")
        ):
            score_text = lines[0][len("Score:"):].strip()
            if re.fullmatch(r"[0-5]", score_text):
                return ParsedOutput(
                    score=float(score_text),
                    reason=normalize_single_line(lines[1][len("Reason:"):].strip()),
                    modified_answer=normalize_single_line(lines[2][len("Modified Answer:"):].strip()),
                    raw_output=raw_text,
                    parse_error=None,
                    strict_format_ok=True,
                )

    # Fallback case: regex parse
    score: Optional[float] = None
    parse_error: Optional[str] = None
    reason = ""
    modified_answer = ""

    score_match = SCORE_PATTERN.search(raw_text)
    if score_match:
        score = float(score_match.group(1))
    else:
        fallback = re.search(rf"(?i)\bscore\s*{COLON_CLASS}\s*([0-5](?:\.\d)?)", raw_text)
        if fallback:
            score = float(fallback.group(1))
        else:
            parse_error = "Failed to parse score."

    reason_match = REASON_PATTERN.search(raw_text)
    modified_answer_match = MODIFIED_ANSWER_PATTERN.search(raw_text)

    if reason_match:
        reason = normalize_single_line(reason_match.group(1))
    if modified_answer_match:
        modified_answer = normalize_single_line(modified_answer_match.group(1))

    return ParsedOutput(
        score=score,
        reason=reason,
        modified_answer=modified_answer,
        raw_output=raw_text,
        parse_error=parse_error,
        strict_format_ok=False,
    )


# ============================================================
# Dataset preparation
# ============================================================

def prepare_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Record {index} is missing valid messages.")

        system_message = None
        user_message = None
        assistant_message = None

        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            if role == "system" and system_message is None:
                system_message = str(message.get("content", ""))
            elif role == "user" and user_message is None:
                user_message = str(message.get("content", ""))
            elif role == "assistant" and assistant_message is None:
                assistant_message = str(message.get("content", ""))

        if not user_message:
            raise ValueError(f"Record {index} has no user content.")

        sections = extract_sections_by_headers(user_message, USER_SECTION_ALIASES)

        reference_parsed = parse_three_line_output(assistant_message or "")

        prompt_messages: list[dict[str, str]] = []
        if system_message:
            prompt_messages.append({"role": "system", "content": system_message})
        prompt_messages.append({"role": "user", "content": user_message})

        samples.append(
            {
                "sample_id": build_sample_id(messages, index),
                "index": index,
                "question": sections.get("question", "").strip(),
                "answer": sections.get("answer", "").strip(),
                "evaluation_dimension": sections.get("evaluation_dimension", "").strip(),
                "criteria": sections.get("criteria", "").strip(),
                "prompt_messages": prompt_messages,
                "reference_output": assistant_message or "",
                "reference_score": reference_parsed.score,
                "reference_reason": reference_parsed.reason,
                "reference_modified_answer": reference_parsed.modified_answer,
                "reference_parse_error": reference_parsed.parse_error,
                "reference_format_ok": reference_parsed.strict_format_ok,
                "original_system_message": system_message or "",
                "original_user_message": user_message,
            }
        )

    return samples


# ============================================================
# Prompt injection
# ============================================================

def build_effective_messages(
    prompt_messages: list[dict[str, str]],
    strict_judging: bool,
    strict_prompt: str,
) -> list[dict[str, str]]:
    if not strict_judging or not strict_prompt.strip():
        return [dict(message) for message in prompt_messages]

    effective_messages = [dict(message) for message in prompt_messages]

    if effective_messages and effective_messages[0].get("role") == "system":
        original_system = effective_messages[0].get("content", "").strip()
        effective_messages[0]["content"] = f"{original_system}\n\n{strict_prompt.strip()}".strip()
        return effective_messages

    return [{"role": "system", "content": strict_prompt.strip()}, *effective_messages]


# ============================================================
# Client / URL helpers
# ============================================================

def normalize_client_host(host: str) -> str:
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized or "127.0.0.1"


def normalize_base_url(raw_base_url: str, *, host: str, port: int) -> str:
    if raw_base_url:
        base_url = raw_base_url.rstrip("/")
    else:
        base_url = f"http://{normalize_client_host(host)}:{int(port)}/v1"

    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url


def normalize_openai_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return base_url


def is_loopback_base_url(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


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


def build_openai_client(base_url: Optional[str], api_key: str, timeout: float) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError(
            "The 'openai' package is required. Install dependencies with: pip install -r requirements.txt"
        )

    if base_url and is_loopback_base_url(base_url):
        with suspended_proxy_env():
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    return OpenAI(api_key=api_key, timeout=timeout)


def wait_for_vllm_ready(base_url: str, timeout: float, retries: int, interval: float) -> None:
    base_url = base_url.rstrip("/")
    health_urls = [
        base_url[:-3] + "/health" if base_url.endswith("/v1") else base_url + "/health",
        base_url[:-3] + "/ping" if base_url.endswith("/v1") else base_url + "/ping",
        base_url + "/models",
    ]

    last_error: Optional[str] = None
    for _ in range(max(1, retries)):
        for url in health_urls:
            try:
                with urllib_request.urlopen(url, timeout=min(10.0, timeout)) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(max(0.1, interval))

    raise RuntimeError(f"vLLM endpoint not ready. Tried {health_urls}. Last error: {last_error}")


def fetch_model_id(base_url: str, timeout: float) -> Optional[str]:
    models_url = base_url.rstrip("/") + "/models"
    try:
        with urllib_request.urlopen(models_url, timeout=min(10.0, timeout)) as response:
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


# ============================================================
# Output file helpers
# ============================================================

def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return results

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            if isinstance(sample_id, str):
                results[sample_id] = payload

    return results


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ============================================================
# Generation
# ============================================================

def generate_single(
    sample: dict[str, Any],
    client: "OpenAI",
    *,
    judge_name: str,
    resolved_model: str,
    resolved_base_url: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_delay: float,
    strict_judging: bool,
    strict_prompt: str,
) -> dict[str, Any]:
    last_error: Optional[str] = None

    effective_messages = build_effective_messages(
        sample["prompt_messages"],
        strict_judging=strict_judging,
        strict_prompt=strict_prompt,
    )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=resolved_model,
                messages=effective_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = completion.choices[0].message.content or ""
            parsed = parse_three_line_output(text)

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
                "reference_parse_error": sample["reference_parse_error"],
                "judge_name": judge_name,
                "judge_model": resolved_model,
                "judge_base_url": resolved_base_url,
                "predicted_score": parsed.score,
                "predicted_reason": parsed.reason,
                "predicted_modified_answer": parsed.modified_answer,
                "raw_output": parsed.raw_output,
                "strict_format_ok": parsed.strict_format_ok,
                "parse_error": parsed.parse_error,
                "request_error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
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
        "reference_parse_error": sample["reference_parse_error"],
        "judge_name": judge_name,
        "judge_model": resolved_model,
        "judge_base_url": resolved_base_url,
        "predicted_score": None,
        "predicted_reason": "",
        "predicted_modified_answer": "",
        "raw_output": "",
        "strict_format_ok": False,
        "parse_error": None,
        "request_error": last_error,
    }


# ============================================================
# Arg / env helpers
# ============================================================

def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def parse_env_bool(value: str) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def normalize_auth_token(value: Any) -> str:
    token = str(value or "").strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def resolve_openai_auth_token(explicit_token: str, env_name: str) -> str:
    return first_non_empty(
        normalize_auth_token(explicit_token),
        normalize_auth_token(os.getenv(env_name, "")),
        normalize_auth_token(os.getenv(DEFAULT_OPENAI_BEARER_TOKEN_ENV, "")),
    )


def resolve_workers(args: argparse.Namespace) -> int:
    if args.workers and args.workers > 0:
        return args.workers
    return DEFAULT_OPENAI_WORKERS if args.backend == "openai" else DEFAULT_VLLM_WORKERS


def resolve_strict_judging(args: argparse.Namespace) -> tuple[bool, str]:
    env_flag = parse_env_bool(os.getenv(DEFAULT_STRICT_JUDGING_ENV, ""))

    if args.no_strict_judging:
        strict_judging = False
    elif args.strict_judging:
        strict_judging = True
    elif env_flag is not None:
        strict_judging = env_flag
    else:
        strict_judging = True

    strict_prompt = first_non_empty(
        args.strict_prompt,
        os.getenv(DEFAULT_STRICT_PROMPT_ENV, ""),
        DEFAULT_STRICT_JUDGING_PROMPT,
    )
    return strict_judging, strict_prompt


def resolve_runtime(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.backend == "openai":
        api_key = resolve_openai_auth_token(args.api_key, args.api_key_env)
        if not api_key:
            raise ValueError(
                "Missing OpenAI-compatible bearer token. "
                f"Set --api-key, env {args.api_key_env}, or env {DEFAULT_OPENAI_BEARER_TOKEN_ENV}."
            )

        resolved_model = first_non_empty(
            args.model,
            os.getenv(DEFAULT_OPENAI_MODEL_ENV, ""),
            DEFAULT_OPENAI_MODEL,
        )
        if not resolved_model:
            raise ValueError(
                f"--model is required when backend=openai, or set env {DEFAULT_OPENAI_MODEL_ENV}."
            )

        base_url = normalize_openai_base_url(
            first_non_empty(
                args.base_url,
                os.getenv(DEFAULT_OPENAI_BASE_URL_ENV, ""),
                DEFAULT_OPENAI_BASE_URL,
            )
        )
        return api_key, base_url, resolved_model

    vllm_host = first_non_empty(args.host, os.getenv(DEFAULT_VLLM_HOST_ENV, ""), "127.0.0.1")
    vllm_port_text = first_non_empty(str(args.port), os.getenv(DEFAULT_VLLM_PORT_ENV, ""), "8000")

    try:
        vllm_port = int(vllm_port_text)
    except ValueError as exc:
        raise ValueError(f"Invalid vLLM port: {vllm_port_text}") from exc

    raw_base_url = first_non_empty(args.base_url, os.getenv(DEFAULT_VLLM_BASE_URL_ENV, ""))
    base_url = normalize_base_url(raw_base_url, host=vllm_host, port=vllm_port)

    wait_for_vllm_ready(base_url, args.timeout, args.healthcheck_retries, args.healthcheck_interval)

    resolved_model = first_non_empty(
        args.model,
        os.getenv(DEFAULT_VLLM_MODEL_ENV, ""),
        fetch_model_id(base_url, args.timeout) or "",
    )
    if not resolved_model:
        raise ValueError("Unable to resolve vLLM model id from /v1/models. Set --model explicitly.")

    return "EMPTY", base_url, resolved_model


def build_default_output_path(args: argparse.Namespace, resolved_model: str) -> Path:
    sanitized_model = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved_model) or "unknown_model"
    return DEFAULT_OUTPUT_DIR / f"{args.backend}_{sanitized_model}_judge_outputs_with_reference.jsonl"


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Score/Reason/Modified Answer outputs from final_test_split.jsonl."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL dataset path. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSONL file path. If omitted, it is generated under {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--backend",
        choices=("openai", "vllm"),
        required=True,
        help="Generation backend: official OpenAI API or local vLLM endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "Model name. For openai backend it falls back to env OPENAI_MODEL. "
            "For vLLM it falls back to env VLLM_MODEL or auto-fetch from /v1/models."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker threads. 0 means auto: openai=6, vllm=16.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample count limit.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per sample.")
    parser.add_argument("--retry-delay", type=float, default=4.0, help="Retry delay seconds.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing output.")

    parser.add_argument("--api-key", type=str, default="", help="Explicit API key for OpenAI backend.")
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help=(
            "Env var name for the OpenAI-compatible bearer token. "
            f"Falls back to {DEFAULT_OPENAI_BEARER_TOKEN_ENV}."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="",
        help=(
            "Optional custom base URL. Falls back to env OPENAI_BASE_URL for openai backend "
            "or env VLLM_BASE_URL for vllm backend."
        ),
    )
    parser.add_argument(
        "--judge-name",
        type=str,
        default="",
        help=f"Optional display name written to output. Falls back to env {DEFAULT_JUDGE_NAME_ENV}.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=f"vLLM host. Falls back to env {DEFAULT_VLLM_HOST_ENV}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help=f"vLLM port. Falls back to env {DEFAULT_VLLM_PORT_ENV}.",
    )
    parser.add_argument("--healthcheck-retries", type=int, default=60, help="vLLM health check retries.")
    parser.add_argument("--healthcheck-interval", type=float, default=2.0, help="vLLM health check interval seconds.")

    parser.add_argument(
        "--strict-judging",
        default=True,
        action="store_true",
        help="Enable stricter grading instructions. Defaults to env STRICT_JUDGING or on.",
    )
    parser.add_argument(
        "--no-strict-judging",
        action="store_true",
        help="Disable stricter grading instructions.",
    )
    parser.add_argument(
        "--strict-prompt",
        type=str,
        default="",
        help="Override the strict judging prompt text. Falls back to env STRICT_JUDGING_PROMPT.",
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    api_key, base_url, resolved_model = resolve_runtime(args)
    strict_judging, strict_prompt = resolve_strict_judging(args)

    output_path = args.output or build_default_output_path(args, resolved_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.input)
    samples = prepare_samples(records)

    if args.sample_size is not None:
        samples = samples[: max(0, args.sample_size)]

    if not args.no_resume and output_path.exists():
        existing = load_existing_results(output_path)
    else:
        existing = {}
        if args.no_resume and output_path.exists():
            output_path.unlink()

    samples_to_run = [sample for sample in samples if sample["sample_id"] not in existing]

    print(f"[Backend] {args.backend}")
    print(f"[Base URL] {base_url}")
    print(f"[Model] {resolved_model}")
    print(f"[Input] {args.input}")
    print(f"[Output] {output_path}")
    print(f"[Workers] {resolve_workers(args)}")
    print(f"[Strict Judging] {strict_judging}")
    print(f"[Total Samples] {len(samples)}")
    print(f"[Remaining Samples] {len(samples_to_run)}")

    client = build_openai_client(base_url=base_url or None, api_key=api_key, timeout=args.timeout)
    judge_name = first_non_empty(args.judge_name, os.getenv(DEFAULT_JUDGE_NAME_ENV, ""), resolved_model)

    write_lock = threading.Lock()
    progress = tqdm(total=len(samples), desc="Generating", ncols=100) if tqdm is not None else None
    if progress is not None and existing:
        progress.update(len(existing))

    try:
        # probe
        if samples_to_run:
            probe_sample = samples_to_run[0]
            print(f"[Probe] sending first request for sample index={probe_sample['index']}")

            probe_result = generate_single(
                probe_sample,
                client,
                judge_name=judge_name,
                resolved_model=resolved_model,
                resolved_base_url=base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=1,
                retry_delay=args.retry_delay,
                strict_judging=strict_judging,
                strict_prompt=strict_prompt,
            )

            if probe_result["request_error"]:
                raise RuntimeError(f"Probe request failed: {probe_result['request_error']}")

            existing[probe_result["sample_id"]] = probe_result
            append_jsonl(output_path, probe_result, write_lock)
            if progress is not None:
                progress.update(1)

            print(
                f"[Probe] success: predicted_score={probe_result['predicted_score']}, "
                f"parse_error={probe_result['parse_error']}, "
                f"strict_format_ok={probe_result['strict_format_ok']}"
            )

            samples_to_run = samples_to_run[1:]

        workers = resolve_workers(args)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(
                    generate_single,
                    sample,
                    client,
                    judge_name=judge_name,
                    resolved_model=resolved_model,
                    resolved_base_url=base_url,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                    strict_judging=strict_judging,
                    strict_prompt=strict_prompt,
                )
                for sample in samples_to_run
            ]

            for future in as_completed(futures):
                result = future.result()
                existing[result["sample_id"]] = result
                append_jsonl(output_path, result, write_lock)
                if progress is not None:
                    progress.update(1)

    finally:
        if progress is not None:
            progress.close()
        try:
            client.close()
        except Exception:
            pass

    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
