#!/usr/bin/env python
"""Judge rewritten-answer quality against the cortex-5 rewrite baseline.

For each prediction JSONL under evaluation/baseline_new/predictions, this script
compares that file's predicted_modified_answer with the matching
predicted_modified_answer from openai_cortex-5_judge_outputs_with_reference.jsonl.
The comparison is judged by an OpenAI-compatible chat-completions endpoint and
reported as WIN/TIE/LOSE from the target prediction file's perspective.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable, Optional
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


DEFAULT_PREDICTIONS_DIR = Path("evaluation/baseline_new/predictions")
DEFAULT_BASELINE_NAME = "openai_cortex-5_judge_outputs_with_reference.jsonl"
DEFAULT_OUTPUT_DIR = Path("evaluation/baseline_new/rewrite_win_rate")

DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_VLLM_PORTS = (8001, 8002, 8003, 8004)
DEFAULT_VLLM_API_KEY = "EMPTY"

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "cortex-5"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

RESULT_PATTERN = re.compile(r"(?im)^\s*result\s*[:\uFF1A]\s*(win|tie|lose)\s*$")
REASON_PATTERN = re.compile(r"(?is)reason\s*[:\uFF1A]\s*(.*)$")


REWRITE_JUDGE_SYSTEM_PROMPT = """You are a strict judge of rewritten answers.

You will compare two candidate rewrites of the same original answer. Judge only how well each rewrite improves the original answer for the provided evaluation dimension and 0-5 criteria.

Rules:
- Use the evaluation dimension and criteria as the only quality standard.
- Prefer the rewrite that is more correct, internally consistent, and better aligned with the target evaluation dimension.
- Do not reward extra unsupported facts, verbosity, or style unless the criteria require them.
- If both rewrites satisfy the criteria to a similar degree, choose TIE.
- If one rewrite is clearly better, choose WIN or LOSE from Candidate A's perspective.
- Output exactly two lines:
Result: WIN|TIE|LOSE
Reason: <one concise sentence>
"""


@dataclass(frozen=True)
class JudgeEndpoint:
    name: str
    backend: str
    base_url: str
    model: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout: float
    max_retries: int
    retry_delay: float
    healthcheck: bool = False
    healthcheck_retries: int = 60
    healthcheck_interval: float = 2.0


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
            yield payload


def row_key(row: dict[str, Any]) -> tuple[str, Any]:
    index = row.get("index")
    if isinstance(index, int):
        return ("index", index)

    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return ("sample_id", sample_id.strip())

    raise ValueError(f"Row is missing both a usable 'index' and 'sample_id': {row}")


def sort_key(key: tuple[str, Any]) -> tuple[int, Any]:
    if key[0] == "index":
        return (0, key[1])
    return (1, str(key[1]))


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def load_rows_by_key(path: Path) -> dict[tuple[str, Any], dict[str, Any]]:
    keyed_rows: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = row_key(row)
        if key in keyed_rows:
            raise ValueError(f"Duplicate key {key!r} found in {path}")
        keyed_rows[key] = row
    return keyed_rows


def validate_aligned_rows(
    baseline_row: dict[str, Any],
    target_row: dict[str, Any],
    target_path: Path,
) -> None:
    baseline_question = normalize_text(baseline_row.get("question"))
    target_question = normalize_text(target_row.get("question"))
    if baseline_question and target_question and baseline_question != target_question:
        raise ValueError(f"Mismatched question for key={row_key(target_row)!r} in {target_path}")

    baseline_dimension = normalize_text(baseline_row.get("evaluation_dimension"))
    target_dimension = normalize_text(target_row.get("evaluation_dimension"))
    if baseline_dimension and target_dimension and baseline_dimension != target_dimension:
        raise ValueError(f"Mismatched evaluation_dimension for key={row_key(target_row)!r} in {target_path}")


def build_user_prompt(
    *,
    question: str,
    original_answer: str,
    evaluation_dimension: str,
    criteria: str,
    target_rewrite: str,
    baseline_rewrite: str,
) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Original Answer:\n"
        f"{original_answer}\n\n"
        "Evaluation Dimension:\n"
        f"{evaluation_dimension}\n\n"
        "Criteria:\n"
        f"{criteria}\n\n"
        "Candidate A rewrite:\n"
        f"{target_rewrite}\n\n"
        "Candidate B rewrite:\n"
        f"{baseline_rewrite}\n\n"
        "Decide whether Candidate A is better than, tied with, or worse than Candidate B."
    )


def build_comparison_sample(
    *,
    baseline_row: dict[str, Any],
    target_row: dict[str, Any],
    target_path: Path,
) -> Optional[dict[str, Any]]:
    validate_aligned_rows(baseline_row, target_row, target_path)

    target_rewrite = normalize_text(target_row.get("predicted_modified_answer"))
    baseline_rewrite = normalize_text(baseline_row.get("predicted_modified_answer"))
    if not target_rewrite or not baseline_rewrite:
        return None

    question = normalize_text(target_row.get("question")) or normalize_text(baseline_row.get("question"))
    original_answer = normalize_text(target_row.get("answer")) or normalize_text(baseline_row.get("answer"))
    evaluation_dimension = normalize_text(target_row.get("evaluation_dimension")) or normalize_text(
        baseline_row.get("evaluation_dimension")
    )
    criteria = str(target_row.get("criteria") or baseline_row.get("criteria") or "").strip()

    return {
        "key": list(row_key(target_row)),
        "sample_id": target_row.get("sample_id") or baseline_row.get("sample_id"),
        "index": target_row.get("index", baseline_row.get("index")),
        "question": question,
        "answer": original_answer,
        "evaluation_dimension": evaluation_dimension,
        "criteria": criteria,
        "target_modified_answer": target_rewrite,
        "baseline_modified_answer": baseline_rewrite,
        "target_judge_name": target_row.get("judge_name"),
        "target_judge_model": target_row.get("judge_model"),
        "baseline_judge_name": baseline_row.get("judge_name"),
        "baseline_judge_model": baseline_row.get("judge_model"),
        "messages": [
            {"role": "system", "content": REWRITE_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    question=question,
                    original_answer=original_answer,
                    evaluation_dimension=evaluation_dimension,
                    criteria=criteria,
                    target_rewrite=target_rewrite,
                    baseline_rewrite=baseline_rewrite,
                ),
            },
        ],
    }


def parse_result(text: str) -> tuple[Optional[str], str, Optional[str]]:
    raw_text = text.strip().replace("\r\n", "\n")
    result_match = RESULT_PATTERN.search(raw_text)
    reason_match = REASON_PATTERN.search(raw_text)

    result = result_match.group(1).lower() if result_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    parse_error = None if result in {"win", "tie", "lose"} else "Failed to parse Result as WIN/TIE/LOSE."
    return result, reason, parse_error


def normalize_client_host(host: str) -> str:
    normalized = host.strip()
    if normalized in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return normalized or "127.0.0.1"


def normalize_base_url(raw_base_url: str) -> str:
    base_url = raw_base_url.strip().rstrip("/")
    if not base_url:
        return ""
    if not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
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


def build_openai_client(endpoint: JudgeEndpoint) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("The 'openai' package is required. Install dependencies with: pip install -r requirements.txt")

    if is_loopback_base_url(endpoint.base_url):
        with suspended_proxy_env():
            return OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, timeout=endpoint.timeout)

    return OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url, timeout=endpoint.timeout)


def wait_for_endpoint_ready(endpoint: JudgeEndpoint) -> None:
    if not endpoint.healthcheck:
        return

    base_url = endpoint.base_url.rstrip("/")
    health_urls = [
        base_url[:-3] + "/health" if base_url.endswith("/v1") else base_url + "/health",
        base_url[:-3] + "/ping" if base_url.endswith("/v1") else base_url + "/ping",
        base_url + "/models",
    ]

    last_error: Optional[str] = None
    for _ in range(max(1, endpoint.healthcheck_retries)):
        for url in health_urls:
            try:
                with urllib_request.urlopen(url, timeout=min(10.0, endpoint.timeout)) as response:
                    if response.status == 200:
                        return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(max(0.1, endpoint.healthcheck_interval))

    raise RuntimeError(f"Endpoint not ready. Tried {health_urls}. Last error: {last_error}")


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


def resolve_vllm_endpoints(args: argparse.Namespace) -> list[JudgeEndpoint]:
    endpoints: list[JudgeEndpoint] = []
    host = normalize_client_host(args.vllm_host)
    for port in args.vllm_ports:
        base_url = normalize_base_url(f"http://{host}:{int(port)}/v1")
        endpoint = JudgeEndpoint(
            name=args.vllm_judge_name,
            backend="vllm",
            base_url=base_url,
            model=args.vllm_model,
            api_key=args.vllm_api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            healthcheck=args.healthcheck,
            healthcheck_retries=args.healthcheck_retries,
            healthcheck_interval=args.healthcheck_interval,
        )
        wait_for_endpoint_ready(endpoint)
        model = endpoint.model or (fetch_model_id(endpoint.base_url, endpoint.timeout) or "")
        if not model:
            raise ValueError(f"Unable to resolve vLLM model from {endpoint.base_url}/models. Set --vllm-model.")
        endpoints.append(
            JudgeEndpoint(
                name=endpoint.name,
                backend=endpoint.backend,
                base_url=endpoint.base_url,
                model=model,
                api_key=endpoint.api_key,
                temperature=endpoint.temperature,
                max_tokens=endpoint.max_tokens,
                timeout=endpoint.timeout,
                max_retries=endpoint.max_retries,
                retry_delay=endpoint.retry_delay,
                healthcheck=endpoint.healthcheck,
                healthcheck_retries=endpoint.healthcheck_retries,
                healthcheck_interval=endpoint.healthcheck_interval,
            )
        )
    return endpoints


def resolve_openai_endpoint(args: argparse.Namespace) -> JudgeEndpoint:
    api_key = args.openai_api_key.strip() or os.getenv(args.openai_api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Missing OpenAI API key. Set --openai-api-key or env {args.openai_api_key_env}.")

    base_url = normalize_base_url(args.openai_base_url or os.getenv("OPENAI_BASE_URL", "") or DEFAULT_OPENAI_BASE_URL)
    model = args.openai_model or os.getenv("OPENAI_MODEL", "") or DEFAULT_OPENAI_MODEL

    return JudgeEndpoint(
        name=args.openai_judge_name,
        backend="openai",
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )


def resolve_judge_groups(args: argparse.Namespace) -> list[list[JudgeEndpoint]]:
    groups: list[list[JudgeEndpoint]] = []
    if args.backend in {"vllm", "both"}:
        groups.append(resolve_vllm_endpoints(args))
    if args.backend in {"openai", "both"}:
        groups.append([resolve_openai_endpoint(args)])
    return groups


def result_key(sample: dict[str, Any]) -> str:
    key = sample["key"]
    return f"{key[0]}::{key[1]}"


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return existing

    for row in iter_jsonl(path):
        key = row.get("comparison_key")
        if isinstance(key, str) and key:
            existing[key] = row
    return existing


def append_jsonl(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def judge_single(
    sample: dict[str, Any],
    endpoint: JudgeEndpoint,
    client: "OpenAI",
    target_file: Path,
) -> dict[str, Any]:
    last_error: Optional[str] = None

    for attempt in range(endpoint.max_retries):
        try:
            completion = client.chat.completions.create(
                model=endpoint.model,
                messages=sample["messages"],
                temperature=endpoint.temperature,
                max_tokens=endpoint.max_tokens,
            )
            text = completion.choices[0].message.content or ""
            result, reason, parse_error = parse_result(text)

            return {
                "comparison_key": result_key(sample),
                "sample_id": sample.get("sample_id"),
                "index": sample.get("index"),
                "target_file": str(target_file),
                "target_name": target_file.stem,
                "target_judge_name": sample.get("target_judge_name"),
                "target_judge_model": sample.get("target_judge_model"),
                "baseline_judge_name": sample.get("baseline_judge_name"),
                "baseline_judge_model": sample.get("baseline_judge_model"),
                "evaluation_dimension": sample.get("evaluation_dimension"),
                "criteria": sample.get("criteria"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "target_modified_answer": sample.get("target_modified_answer"),
                "baseline_modified_answer": sample.get("baseline_modified_answer"),
                "judge_backend": endpoint.backend,
                "judge_name": endpoint.name,
                "judge_model": endpoint.model,
                "judge_base_url": endpoint.base_url,
                "result": result,
                "reason": reason,
                "raw_output": text.strip(),
                "parse_error": parse_error,
                "request_error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < endpoint.max_retries - 1:
                time.sleep(endpoint.retry_delay * (attempt + 1))
            else:
                break

    return {
        "comparison_key": result_key(sample),
        "sample_id": sample.get("sample_id"),
        "index": sample.get("index"),
        "target_file": str(target_file),
        "target_name": target_file.stem,
        "target_judge_name": sample.get("target_judge_name"),
        "target_judge_model": sample.get("target_judge_model"),
        "baseline_judge_name": sample.get("baseline_judge_name"),
        "baseline_judge_model": sample.get("baseline_judge_model"),
        "evaluation_dimension": sample.get("evaluation_dimension"),
        "criteria": sample.get("criteria"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "target_modified_answer": sample.get("target_modified_answer"),
        "baseline_modified_answer": sample.get("baseline_modified_answer"),
        "judge_backend": endpoint.backend,
        "judge_name": endpoint.name,
        "judge_model": endpoint.model,
        "judge_base_url": endpoint.base_url,
        "result": None,
        "reason": "",
        "raw_output": "",
        "parse_error": None,
        "request_error": last_error,
    }


def collect_samples(
    *,
    baseline_rows: dict[tuple[str, Any], dict[str, Any]],
    target_path: Path,
    sample_size: Optional[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    target_rows = load_rows_by_key(target_path)
    common_keys = sorted(set(baseline_rows) & set(target_rows), key=sort_key)
    if sample_size is not None:
        common_keys = common_keys[: max(0, sample_size)]

    stats = {
        "baseline_rows": len(baseline_rows),
        "target_rows": len(target_rows),
        "common_samples": len(common_keys),
        "baseline_only_samples": len(baseline_rows) - len(common_keys),
        "target_only_samples": len(target_rows) - len(common_keys),
        "skipped_missing_rewrite": 0,
    }

    samples: list[dict[str, Any]] = []
    for key in common_keys:
        sample = build_comparison_sample(
            baseline_row=baseline_rows[key],
            target_row=target_rows[key],
            target_path=target_path,
        )
        if sample is None:
            stats["skipped_missing_rewrite"] += 1
            continue
        samples.append(sample)

    return samples, stats


def safe_file_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unknown"


def detail_output_path(output_dir: Path, endpoint_group: list[JudgeEndpoint], target_path: Path) -> Path:
    primary = endpoint_group[0]
    judge_id = safe_file_part(f"{primary.backend}_{primary.name}_{primary.model}")
    return output_dir / "judgments" / judge_id / f"{target_path.stem}.jsonl"


def run_target_file(
    *,
    baseline_rows: dict[tuple[str, Any], dict[str, Any]],
    target_path: Path,
    endpoint_group: list[JudgeEndpoint],
    output_dir: Path,
    sample_size: Optional[int],
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    samples, stats = collect_samples(
        baseline_rows=baseline_rows,
        target_path=target_path,
        sample_size=sample_size,
    )

    output_path = detail_output_path(output_dir, endpoint_group, target_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume and output_path.exists():
        output_path.unlink()

    existing = load_existing_results(output_path) if resume else {}
    samples_to_run = [sample for sample in samples if result_key(sample) not in existing]

    clients = {endpoint.base_url: build_openai_client(endpoint) for endpoint in endpoint_group}
    write_lock = threading.Lock()
    progress = tqdm(total=len(samples), desc=f"{target_path.stem}", ncols=100) if tqdm is not None else None
    if progress is not None and existing:
        progress.update(min(len(existing), len(samples)))

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [
                executor.submit(
                    judge_single,
                    sample,
                    endpoint_group[index % len(endpoint_group)],
                    clients[endpoint_group[index % len(endpoint_group)].base_url],
                    target_path,
                )
                for index, sample in enumerate(samples_to_run)
            ]

            for future in as_completed(futures):
                result = future.result()
                existing[result["comparison_key"]] = result
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

    ordered_results = [existing[result_key(sample)] for sample in samples if result_key(sample) in existing]
    wins = sum(1 for row in ordered_results if row.get("result") == "win")
    ties = sum(1 for row in ordered_results if row.get("result") == "tie")
    losses = sum(1 for row in ordered_results if row.get("result") == "lose")
    request_failures = sum(1 for row in ordered_results if row.get("request_error"))
    parse_failures = sum(1 for row in ordered_results if row.get("parse_error"))
    judged = wins + ties + losses

    primary = endpoint_group[0]
    return {
        "target_name": target_path.stem,
        "target_file": str(target_path),
        "judge_backend": primary.backend,
        "judge_name": primary.name,
        "judge_model": primary.model,
        "judge_base_urls": [endpoint.base_url for endpoint in endpoint_group],
        "detail_file": str(output_path),
        "total_comparison_candidates": len(samples),
        "completed_rows": len(ordered_results),
        "judged_rows": judged,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": round(wins / judged, 6) if judged else None,
        "tie_rate": round(ties / judged, 6) if judged else None,
        "loss_rate": round(losses / judged, 6) if judged else None,
        "win_rate_excluding_ties": round(wins / (wins + losses), 6) if wins + losses else None,
        "net_win_rate": round((wins - losses) / judged, 6) if judged else None,
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        **stats,
    }


def discover_prediction_files(predictions_dir: Path, baseline_path: Path, explicit_inputs: list[Path]) -> list[Path]:
    if explicit_inputs:
        paths = [path.resolve() for path in explicit_inputs]
    else:
        paths = sorted(predictions_dir.glob("*.jsonl"))

    baseline_resolved = baseline_path.resolve()
    filtered = [path for path in paths if path.resolve() != baseline_resolved]
    if not filtered:
        raise ValueError("No target prediction files found.")
    return filtered


def write_summary_files(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = summaries_dir / "rewrite_win_rate_summary.json"
    summary_csv_path = summaries_dir / "rewrite_win_rate_summary.csv"

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summaries": rows,
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "judge_backend",
        "judge_name",
        "judge_model",
        "target_name",
        "judged_rows",
        "wins",
        "ties",
        "losses",
        "win_rate",
        "tie_rate",
        "loss_rate",
        "win_rate_excluding_ties",
        "net_win_rate",
        "completed_rows",
        "total_comparison_candidates",
        "common_samples",
        "baseline_only_samples",
        "target_only_samples",
        "skipped_missing_rewrite",
        "request_failures",
        "parse_failures",
        "target_file",
        "detail_file",
        "judge_base_urls",
    ]

    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["judge_backend"], -(item["win_rate"] or -1), item["target_name"])):
            csv_row = {field: row.get(field) for field in fieldnames}
            csv_row["judge_base_urls"] = json.dumps(row.get("judge_base_urls", []), ensure_ascii=False)
            writer.writerow(csv_row)

    return summary_json_path, summary_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate rewrite ability by LLM-judging each prediction file's "
            "predicted_modified_answer against the cortex-5 baseline rewrite."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory containing prediction JSONL files. Default: {DEFAULT_PREDICTIONS_DIR}",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR / DEFAULT_BASELINE_NAME,
        help=f"Baseline JSONL path. Default: {DEFAULT_PREDICTIONS_DIR / DEFAULT_BASELINE_NAME}",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=[],
        help="Optional subset of prediction files. Defaults to all JSONL files in --predictions-dir except baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for detailed judgments and summary CSV. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--backend", choices=("vllm", "openai", "both"), default="vllm")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sorted sample limit per target file.")
    parser.add_argument("--workers", type=int, default=96, help="Worker threads per target file.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing detailed judgment JSONL.")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=3.0)

    parser.add_argument("--vllm-host", type=str, default=DEFAULT_VLLM_HOST)
    parser.add_argument("--vllm-ports", nargs="+", type=int, default=list(DEFAULT_VLLM_PORTS))
    parser.add_argument("--vllm-model", type=str, default=os.getenv("VLLM_MODEL", ""))
    parser.add_argument("--vllm-api-key", type=str, default=DEFAULT_VLLM_API_KEY)
    parser.add_argument("--vllm-judge-name", type=str, default="vllm_rewrite_judge")
    parser.add_argument("--healthcheck", action="store_true", default=True)
    parser.add_argument("--no-healthcheck", dest="healthcheck", action="store_false")
    parser.add_argument("--healthcheck-retries", type=int, default=60)
    parser.add_argument("--healthcheck-interval", type=float, default=2.0)

    parser.add_argument("--openai-base-url", type=str, default="")
    parser.add_argument("--openai-model", type=str, default="")
    parser.add_argument("--openai-api-key", type=str, default="")
    parser.add_argument("--openai-api-key-env", type=str, default=DEFAULT_OPENAI_API_KEY_ENV)
    parser.add_argument("--openai-judge-name", type=str, default="gpt_rewrite_judge")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_dir = args.predictions_dir.resolve()
    baseline_path = args.baseline.resolve()
    output_dir = args.output_dir.resolve()

    if not predictions_dir.is_dir():
        raise NotADirectoryError(f"Predictions directory not found: {predictions_dir}")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    target_paths = discover_prediction_files(predictions_dir, baseline_path, args.inputs)
    endpoint_groups = resolve_judge_groups(args)

    print(f"[Predictions Dir] {predictions_dir}")
    print(f"[Baseline] {baseline_path}")
    print(f"[Targets] {len(target_paths)}")
    for group in endpoint_groups:
        primary = group[0]
        print(f"[Judge] backend={primary.backend}, name={primary.name}, model={primary.model}, endpoints={len(group)}")

    baseline_rows = load_rows_by_key(baseline_path)
    summary_rows: list[dict[str, Any]] = []

    for endpoint_group in endpoint_groups:
        for target_path in target_paths:
            print(f"[Run] {target_path.name} with {endpoint_group[0].backend}:{endpoint_group[0].name}")
            summary_rows.append(
                run_target_file(
                    baseline_rows=baseline_rows,
                    target_path=target_path,
                    endpoint_group=endpoint_group,
                    output_dir=output_dir,
                    sample_size=args.sample_size,
                    workers=args.workers,
                    resume=not args.no_resume,
                )
            )

    summary_json_path, summary_csv_path = write_summary_files(output_dir, summary_rows)
    print(f"Summary saved to: {summary_json_path}")
    print(f"CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
