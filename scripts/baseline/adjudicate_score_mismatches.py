#!/usr/bin/env python
"""Adjudicate rows where predicted_score and reference_score disagree."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
import threading
import time
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

DEFAULT_INPUT_PATH = Path(
    "evaluation/baseline_new/mismatch/openai_cortex-5_judge_outputs_with_reference_score_mismatches.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("evaluation/baseline_new/mismatch")
DEFAULT_JUDGE_CONFIG_PATH = Path("scripts/baseline/judge_model_config.example.json")

DEFAULT_OPENAI_BASE_URL = "https://llm.api.zyuncs.com/v1"
DEFAULT_OPENAI_MODEL = "cortex-5"
DEFAULT_OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_OPENAI_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_OPENAI_BEARER_TOKEN_ENV = "OPENAI_BEARER_TOKEN"
DEFAULT_VLLM_MODEL_ENV = "VLLM_MODEL"

DEFAULT_OPENAI_WORKERS = 10
DEFAULT_LOCAL_WORKERS = 16

COLON_CLASS = r"[:\uFF1A]"
PREFERRED_PATTERN = re.compile(rf"(?im)^\s*preferred\s*{COLON_CLASS}\s*([AB])\s*$")
REASON_PATTERN = re.compile(
    rf"(?is)reason\s*{COLON_CLASS}\s*(.*?)\s*(?:(?:\n\s*)?aligned score\s*{COLON_CLASS}|$)"
)
ALIGNED_SCORE_PATTERN = re.compile(rf"(?im)^\s*aligned score\s*{COLON_CLASS}\s*([0-5])\s*$")

DEFAULT_SYSTEM_PROMPT = """You are a strict rubric-based adjudicator.
Judge which candidate score is more reasonable for the ORIGINAL answer.
Use the question, original answer, evaluation dimension, and full rubric as ground truth.
Candidate reasons are auxiliary only and may be wrong.
Do not choose based on which candidate reason is better written; judge only whether the score matches the original answer under the rubric.
Choose exactly one candidate label, A or B.
On the Aligned Score line, repeat the chosen candidate's score exactly.
Output exactly 3 lines:
Preferred: <A or B>
Reason: <one-line rubric-grounded justification>
Aligned Score: <0-5>
Do not output anything else."""


@dataclass
class JudgeConfig:
    name: str
    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 1024
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
class ParsedAdjudication:
    preferred_label: Optional[str]
    aligned_score: Optional[float]
    reason: str
    raw_output: str
    parse_error: Optional[str] = None
    strict_format_ok: bool = False


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


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def normalize_auth_token(value: Any) -> str:
    token = str(value or "").strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()) or "unknown"


def build_row_id(row: dict[str, Any]) -> str:
    explicit_row_id = str(row.get("row_id", "")).strip()
    if explicit_row_id:
        return explicit_row_id
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return f"index::{row.get('index')}"


def extract_numeric_score(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_candidate_mapping(row_id: str) -> dict[str, str]:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return {"A": "predicted", "B": "reference"} if int(digest[-1], 16) % 2 == 0 else {"A": "reference", "B": "predicted"}


def build_candidate_block(sample: dict[str, Any], label: str, source: str) -> str:
    return (
        f"Candidate {label}:\n"
        f"Score: {int(sample[f'{source}_score'])}\n"
        f"Reason: {normalize_single_line(sample.get(f'{source}_reason', ''))}\n"
    )


def build_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    mapping = sample["candidate_mapping"]
    user_content = (
        "Question:\n"
        f"{sample.get('question', '')}\n\n"
        "Original Answer:\n"
        f"{sample.get('answer', '')}\n\n"
        "Evaluation Dimension:\n"
        f"{sample.get('evaluation_dimension', '')}\n\n"
        "Criteria (0-5):\n"
        f"{sample.get('criteria', '')}\n\n"
        f"{build_candidate_block(sample, 'A', mapping['A'])}\n\n"
        f"{build_candidate_block(sample, 'B', mapping['B'])}\n\n"
        "Task:\n"
        "Choose the more reasonable candidate score for the ORIGINAL answer."
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prepare_samples(rows: list[dict[str, Any]], sample_size: Optional[int]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        predicted_score = extract_numeric_score(row, "predicted_score")
        reference_score = extract_numeric_score(row, "reference_score")
        if predicted_score is None or reference_score is None or predicted_score == reference_score:
            continue
        sample = dict(row)
        sample["row_id"] = build_row_id(row)
        sample["predicted_score"] = predicted_score
        sample["reference_score"] = reference_score
        sample["candidate_mapping"] = build_candidate_mapping(sample["row_id"])
        sample["prompt_messages"] = build_messages(sample)
        samples.append(sample)
    samples.sort(key=lambda item: (not isinstance(item.get("index"), int), item.get("index", 0)))
    return samples[: max(0, sample_size)] if sample_size is not None else samples


def parse_output(text: str) -> ParsedAdjudication:
    raw_text = text.strip().replace("\r\n", "\n")
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    if len(lines) == 3 and lines[0].startswith("Preferred:") and lines[1].startswith("Reason:") and lines[2].startswith("Aligned Score:"):
        preferred = lines[0][len("Preferred:"):].strip().upper()
        aligned = lines[2][len("Aligned Score:"):].strip()
        if preferred in {"A", "B"} and re.fullmatch(r"[0-5]", aligned):
            return ParsedAdjudication(
                preferred_label=preferred,
                aligned_score=float(aligned),
                reason=normalize_single_line(lines[1][len("Reason:"):].strip()),
                raw_output=raw_text,
                strict_format_ok=True,
            )

    preferred_match = PREFERRED_PATTERN.search(raw_text)
    aligned_match = ALIGNED_SCORE_PATTERN.search(raw_text)
    reason_match = REASON_PATTERN.search(raw_text)
    errors: list[str] = []
    preferred = None
    aligned_score = None
    if preferred_match:
        preferred = preferred_match.group(1).upper()
    else:
        errors.append("Failed to parse preferred label.")
    if aligned_match:
        aligned_score = float(aligned_match.group(1))
    else:
        errors.append("Failed to parse aligned score.")
    return ParsedAdjudication(
        preferred_label=preferred,
        aligned_score=aligned_score,
        reason=normalize_single_line(reason_match.group(1)) if reason_match else "",
        raw_output=raw_text,
        parse_error=" ".join(errors) if errors else None,
        strict_format_ok=False,
    )


def selected_source(sample: dict[str, Any], preferred_label: str | None) -> Optional[str]:
    if preferred_label is None:
        return None
    return sample["candidate_mapping"].get(preferred_label)


def source_output(sample: dict[str, Any], source: str | None) -> str:
    if source == "reference":
        return str(sample.get("reference_output", ""))
    if source == "predicted":
        return str(sample.get("raw_output", ""))
    return ""


def build_clean_base_result(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": sample.get("row_id"),
        "sample_id": sample.get("sample_id"),
        "index": sample.get("index"),
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "evaluation_dimension": sample.get("evaluation_dimension", ""),
        "criteria": sample.get("criteria", ""),
        "predicted_score": sample.get("predicted_score"),
        "reference_score": sample.get("reference_score"),
    }


def build_success_result(sample: dict[str, Any], judge: JudgeConfig, parsed: ParsedAdjudication) -> dict[str, Any]:
    preferred_source = selected_source(sample, parsed.preferred_label)
    selected_score = sample.get(f"{preferred_source}_score") if preferred_source else None
    selected_reason = sample.get(f"{preferred_source}_reason", "") if preferred_source else ""
    selected_modified_answer = sample.get(f"{preferred_source}_modified_answer", "") if preferred_source else ""
    selected_output = source_output(sample, preferred_source) if preferred_source else ""

    parse_error = parsed.parse_error
    if preferred_source and selected_score is not None and parsed.aligned_score is not None:
        if float(selected_score) != float(parsed.aligned_score):
            extra = (
                f"Aligned score {parsed.aligned_score} does not match "
                f"{preferred_source}_score {selected_score}."
            )
            parse_error = f"{parse_error} {extra}".strip() if parse_error else extra

    result = build_clean_base_result(sample)
    result.update(
        {
            "judge_name": judge.name,
            "judge_model": judge.model,
            "selected_source": preferred_source,
            "selected_score": selected_score,
            "selected_reason": selected_reason,
            "selected_modified_answer": selected_modified_answer,
            "selected_output": selected_output,
            "adjudicator_reason": parsed.reason,
            "adjudicator_raw_output": parsed.raw_output,
            "strict_format_ok": parsed.strict_format_ok,
            "parse_error": parse_error,
            "request_error": None,
        }
    )
    return result


def build_failure_result(sample: dict[str, Any], judge: JudgeConfig, request_error: str) -> dict[str, Any]:
    result = build_clean_base_result(sample)
    result.update(
        {
            "judge_name": judge.name,
            "judge_model": judge.model,
            "selected_source": None,
            "selected_score": None,
            "selected_reason": "",
            "selected_modified_answer": "",
            "selected_output": "",
            "adjudicator_reason": "",
            "adjudicator_raw_output": "",
            "strict_format_ok": False,
            "parse_error": None,
            "request_error": request_error,
        }
    )
    return result


def call_adjudicator(sample: dict[str, Any], judge: JudgeConfig, client: "OpenAI") -> dict[str, Any]:
    messages: list[dict[str, str]] = []
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
            return build_success_result(sample, judge, parse_output(text))
        except Exception as exc:
            last_error = str(exc)
            if attempt < judge.max_retries - 1:
                time.sleep(judge.retry_delay * (attempt + 1))
    return build_failure_result(sample, judge, last_error or "Unknown request error.")


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return existing
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            existing[build_row_id(payload)] = payload
    return existing


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_ports(raw_ports: Any) -> tuple[int, ...]:
    if raw_ports is None or raw_ports == "":
        return ()
    if not isinstance(raw_ports, list):
        raise ValueError("'ports' must be a JSON array of integers.")
    return tuple(int(item) for item in raw_ports)


def normalize_client_host(host: str) -> str:
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized or "127.0.0.1"


def normalize_base_url(raw_base_url: str, *, host: str, port: Any) -> str:
    if raw_base_url:
        base_url = raw_base_url.rstrip("/")
    elif port is not None:
        base_url = f"http://{normalize_client_host(host)}:{int(port)}/v1"
    else:
        return ""
    return base_url if base_url.endswith("/v1") else base_url + "/v1"


def normalize_openai_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)]
    return base_url


def normalize_base_urls(raw_base_url: str, raw_base_urls: Any, *, host: str, port: Any, ports: tuple[int, ...]) -> list[str]:
    values: list[str] = []
    if raw_base_urls is not None:
        if not isinstance(raw_base_urls, list):
            raise ValueError("'base_urls' must be a JSON array of strings.")
        values = [normalize_base_url(str(item).strip(), host=host, port=None) for item in raw_base_urls]
    elif ports:
        values = [normalize_base_url("", host=host, port=item) for item in ports]
    else:
        single = normalize_base_url(raw_base_url, host=host, port=port)
        values = [single] if single else []
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def is_loopback_base_url(base_url: str) -> bool:
    hostname = (urlparse(base_url).hostname or "").strip().lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


@contextmanager
def suspended_proxy_env() -> Any:
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    old_values = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
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
        raise RuntimeError("The 'openai' package is required. Install dependencies with: pip install -r requirements.txt")
    if is_loopback_base_url(base_url):
        with suspended_proxy_env():
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def fetch_model_id(base_url: str, timeout: float) -> Optional[str]:
    try:
        with urllib_request.urlopen(base_url.rstrip("/") + "/models", timeout=min(10.0, timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    if isinstance(payload, dict):
        data = payload.get("data", [])
        if data and isinstance(data[0], dict):
            model_id = data[0].get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    return None


def wait_for_vllm_ready(base_url: str, *, timeout: float, retries: int, interval: float) -> None:
    urls = [
        base_url[:-3] + "/health" if base_url.endswith("/v1") else base_url + "/health",
        base_url[:-3] + "/ping" if base_url.endswith("/v1") else base_url + "/ping",
        base_url.rstrip("/") + "/models",
    ]
    last_error: Optional[str] = None
    for _ in range(max(1, retries)):
        for url in urls:
            try:
                with urllib_request.urlopen(url, timeout=min(10.0, timeout)) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(max(0.1, interval))
    raise RuntimeError(f"vLLM endpoint not ready. Tried {urls}. Last error: {last_error}")


def load_judge_configs(path: Path) -> list[JudgeConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    judges_payload = payload.get("judges", []) if isinstance(payload, dict) else payload
    if not isinstance(judges_payload, list):
        raise ValueError("Judge config must be a list or an object with key 'judges'.")

    judges: list[JudgeConfig] = []
    for idx, item in enumerate(judges_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Judge entry {idx} is not a JSON object.")
        name = str(item.get("name", "")).strip()
        host = str(item.get("host", "127.0.0.1")).strip()
        port = item.get("port")
        ports = parse_ports(item.get("ports"))
        base_urls = normalize_base_urls(
            str(item.get("base_url", "")).strip(),
            item.get("base_urls"),
            host=host,
            port=port,
            ports=ports,
        )
        if not name or not base_urls:
            raise ValueError(f"Judge entry {idx} must contain name and base_url/base_urls or host+port/ports.")
        api_key = item.get("api_key") or os.getenv(str(item.get("api_key_env", "")), "") or "EMPTY"
        judges.append(
            JudgeConfig(
                name=name,
                base_url=base_urls[0],
                model=str(item.get("model", "")).strip(),
                api_key=str(api_key),
                temperature=float(item.get("temperature", 0.0)),
                max_tokens=int(item.get("max_tokens", 256)),
                timeout=float(item.get("timeout", 120.0)),
                max_retries=int(item.get("max_retries", 5)),
                retry_delay=float(item.get("retry_delay", 3.0)),
                system_override=str(item.get("system_override", "")).strip() or None,
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


def resolve_openai_judge(args: argparse.Namespace) -> JudgeConfig:
    api_key = first_non_empty(
        normalize_auth_token(args.api_key),
        normalize_auth_token(os.getenv(args.api_key_env, "")),
        normalize_auth_token(os.getenv(DEFAULT_OPENAI_BEARER_TOKEN_ENV, "")),
    )
    if not api_key:
        raise ValueError(
            "Missing OpenAI-compatible bearer token. "
            f"Set --api-key, env {args.api_key_env}, or env {DEFAULT_OPENAI_BEARER_TOKEN_ENV}."
        )
    base_url = normalize_openai_base_url(
        first_non_empty(args.base_url, os.getenv(DEFAULT_OPENAI_BASE_URL_ENV, ""), DEFAULT_OPENAI_BASE_URL)
    ).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    model = first_non_empty(args.model, os.getenv(DEFAULT_OPENAI_MODEL_ENV, ""), DEFAULT_OPENAI_MODEL)
    return JudgeConfig(
        name=first_non_empty(args.judge_name, model),
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        system_override=args.system_prompt or None,
        base_urls=(base_url,),
        healthcheck=False,
    )


def resolve_local_judge(args: argparse.Namespace) -> JudgeConfig:
    judges = load_judge_configs(args.judge_config)
    if args.local_judge_name:
        judges = [judge for judge in judges if judge.name == args.local_judge_name]
    if not judges:
        raise ValueError("No local judge config matched the requested judge name.")
    if len(judges) > 1:
        raise ValueError("Local backend requires exactly one judge config. Use --local-judge-name to select one.")
    judge = judges[0]
    if args.system_prompt:
        judge.system_override = args.system_prompt
    if args.model:
        judge.model = args.model
    judge.temperature = args.temperature
    judge.max_tokens = args.max_tokens
    judge.timeout = args.timeout
    judge.max_retries = args.max_retries
    judge.retry_delay = args.retry_delay
    return judge


def resolve_endpoints(judge: JudgeConfig) -> list[JudgeConfig]:
    base_urls = list(judge.base_urls) if judge.base_urls else [judge.base_url]
    ports = judge.ports if judge.ports else ((judge.port,) if judge.port is not None else ())
    endpoints: list[JudgeConfig] = []
    for idx, base_url in enumerate(base_urls):
        if judge.healthcheck:
            wait_for_vllm_ready(
                base_url,
                timeout=judge.timeout,
                retries=judge.healthcheck_retries,
                interval=judge.healthcheck_interval,
            )
        model = judge.model.strip() or os.getenv(DEFAULT_VLLM_MODEL_ENV, "").strip()
        if not model and judge.auto_fetch_model:
            model = fetch_model_id(base_url, judge.timeout) or ""
        if not model:
            raise ValueError(f"Judge '{judge.name}' has no model configured and auto-fetch failed for {base_url}.")
        endpoints.append(
            JudgeConfig(
                name=judge.name,
                base_url=base_url,
                model=model,
                api_key=judge.api_key,
                temperature=judge.temperature,
                max_tokens=judge.max_tokens,
                timeout=judge.timeout,
                max_retries=judge.max_retries,
                retry_delay=judge.retry_delay,
                system_override=judge.system_override,
                host=judge.host,
                port=ports[idx] if idx < len(ports) else None,
                base_urls=(base_url,),
                ports=(ports[idx],) if idx < len(ports) else (),
                auto_fetch_model=judge.auto_fetch_model,
                healthcheck=judge.healthcheck,
                healthcheck_retries=judge.healthcheck_retries,
                healthcheck_interval=judge.healthcheck_interval,
            )
        )
    return endpoints


def resolve_workers(args: argparse.Namespace) -> int:
    if args.workers and args.workers > 0:
        return args.workers
    return DEFAULT_OPENAI_WORKERS if args.backend == "openai" else DEFAULT_LOCAL_WORKERS


def default_output_path(input_path: Path, judge_name: str, judge_model: str) -> Path:
    target_dir = input_path.parent if input_path.parent != Path(".") else DEFAULT_OUTPUT_DIR
    return target_dir / f"{input_path.stem}_adjudicated_by_{sanitize_name(judge_name or judge_model)}.jsonl"


def build_summary(results: list[dict[str, Any]], judge: JudgeConfig, endpoints: list[JudgeConfig]) -> dict[str, Any]:
    preferred_counts = {"predicted": 0, "reference": 0, "unknown": 0}
    request_failures = 0
    parse_failures = 0
    strict_format_successes = 0

    for row in results:
        if row.get("request_error"):
            request_failures += 1
        if row.get("parse_error"):
            parse_failures += 1
        if row.get("strict_format_ok"):
            strict_format_successes += 1

        preferred = row.get("selected_source")
        if preferred in {"predicted", "reference"}:
            preferred_counts[str(preferred)] += 1
        else:
            preferred_counts["unknown"] += 1

    total = len(results)
    return {
        "judge": asdict(judge),
        "resolved_endpoints": [{"base_url": item.base_url, "model": item.model} for item in endpoints],
        "completed_samples": total,
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "strict_format_successes": strict_format_successes,
        "strict_format_success_rate": round(strict_format_successes / total, 6) if total else None,
        "preferred_counts": preferred_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adjudicate score-mismatch rows via API or multi-endpoint local inference."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help=f"Input JSONL. Default: {DEFAULT_INPUT_PATH}")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path.")
    parser.add_argument("--backend", choices=("openai", "local"), required=True, help="Adjudication backend.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample count limit.")
    parser.add_argument("--workers", type=int, default=0, help="Worker threads. 0 means auto.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing output.")
    parser.add_argument("--model", type=str, default="", help="Override model name.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per sample.")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Retry delay seconds.")
    parser.add_argument("--system-prompt", type=str, default="", help="Optional override system prompt.")
    parser.add_argument("--api-key", type=str, default="", help="Explicit API key for openai backend.")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help=f"API key env var. Falls back to {DEFAULT_OPENAI_BEARER_TOKEN_ENV}.")
    parser.add_argument("--base-url", type=str, default="", help="Custom API base URL for openai backend.")
    parser.add_argument("--judge-name", type=str, default="", help="Display name for openai backend.")
    parser.add_argument("--judge-config", type=Path, default=DEFAULT_JUDGE_CONFIG_PATH, help=f"Local judge config. Default: {DEFAULT_JUDGE_CONFIG_PATH}")
    parser.add_argument("--local-judge-name", type=str, default="", help="Select one judge from local config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    rows = read_jsonl(args.input)
    samples = prepare_samples(rows, args.sample_size)
    if not samples:
        raise ValueError("No mismatch rows with valid numeric scores found in the input file.")

    unresolved_judge = resolve_openai_judge(args) if args.backend == "openai" else resolve_local_judge(args)
    endpoints = resolve_endpoints(unresolved_judge)
    primary = endpoints[0]
    judge = JudgeConfig(
        name=unresolved_judge.name,
        base_url=primary.base_url,
        model=primary.model,
        api_key=unresolved_judge.api_key,
        temperature=unresolved_judge.temperature,
        max_tokens=unresolved_judge.max_tokens,
        timeout=unresolved_judge.timeout,
        max_retries=unresolved_judge.max_retries,
        retry_delay=unresolved_judge.retry_delay,
        system_override=unresolved_judge.system_override,
        host=unresolved_judge.host,
        port=primary.port,
        base_urls=tuple(item.base_url for item in endpoints),
        ports=unresolved_judge.ports,
        auto_fetch_model=unresolved_judge.auto_fetch_model,
        healthcheck=unresolved_judge.healthcheck,
        healthcheck_retries=unresolved_judge.healthcheck_retries,
        healthcheck_interval=unresolved_judge.healthcheck_interval,
    )

    output_path = args.output or default_output_path(args.input, judge.name, judge.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_results(output_path) if (output_path.exists() and not args.no_resume) else {}
    if args.no_resume and output_path.exists():
        output_path.unlink()
        existing = {}

    samples_to_run = [sample for sample in samples if sample["row_id"] not in existing]
    workers = resolve_workers(args)

    print(f"[Backend] {args.backend}")
    print(f"[Judge] {judge.name}")
    print(f"[Model] {judge.model}")
    print(f"[Input] {args.input}")
    print(f"[Output] {output_path}")
    print(f"[Endpoints] {[item.base_url for item in endpoints]}")
    print(f"[Workers] {workers}")
    print(f"[Total mismatch samples] {len(samples)}")
    print(f"[Remaining samples] {len(samples_to_run)}")

    clients = {
        endpoint.base_url: build_openai_client(endpoint.base_url, endpoint.api_key, endpoint.timeout)
        for endpoint in endpoints
    }
    write_lock = threading.Lock()
    progress = tqdm(total=len(samples), desc="Adjudicating", ncols=100) if tqdm is not None else None
    if progress is not None and existing:
        progress.update(len(existing))

    try:
        if samples_to_run:
            probe = samples_to_run[0]
            probe_result = call_adjudicator(probe, endpoints[0], clients[endpoints[0].base_url])
            if probe_result.get("request_error"):
                raise RuntimeError(f"Probe request failed: {probe_result['request_error']}")
            existing[probe_result["row_id"]] = probe_result
            append_jsonl(output_path, probe_result, write_lock)
            if progress is not None:
                progress.update(1)
            print(
                "[Probe] success: "
                f"selected_source={probe_result.get('selected_source')}, "
                f"selected_score={probe_result.get('selected_score')}, "
                f"parse_error={probe_result.get('parse_error')}"
            )
            samples_to_run = samples_to_run[1:]

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(
                    call_adjudicator,
                    sample,
                    endpoints[index % len(endpoints)],
                    clients[endpoints[index % len(endpoints)].base_url],
                )
                for index, sample in enumerate(samples_to_run)
            ]
            for future in as_completed(futures):
                result = future.result()
                existing[result["row_id"]] = result
                append_jsonl(output_path, result, write_lock)
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

    ordered_results = [existing[sample["row_id"]] for sample in samples if sample["row_id"] in existing]
    summary = build_summary(ordered_results, judge, endpoints)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Summary] {summary_path}")
    print(f"[Preferred predicted] {summary['preferred_counts']['predicted']}")
    print(f"[Preferred reference] {summary['preferred_counts']['reference']}")
    print(f"[Request failures] {summary['request_failures']}")
    print(f"[Parse failures] {summary['parse_failures']}")


if __name__ == "__main__":
    main()