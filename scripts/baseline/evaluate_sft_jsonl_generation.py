#!/usr/bin/env python
"""Evaluate judge models on score/reason/revision SFT JSONL data.

Input format is the chat SFT format produced by scripts/create_jsonl.py:

{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"score\": ..., ...}"}
  ]
}

For each judge model, this script calls the model with the non-assistant
messages and writes:
- predictions/<judge>.jsonl: detailed rows with reference/predicted fields
- sft_outputs/<judge>.jsonl: SFT-like rows whose assistant content is the
  model prediction in strict JSON format
- summaries/summary.json and summaries/summary.csv: score-level metrics
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = ROOT / "scripts" / "evaluation_pipeline"
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import (  # noqa: E402
    DEFAULT_API_KEY,
    append_jsonl,
    call_chat_with_retries,
    load_jsonl,
    models_endpoint_ready,
    normalize_text,
    resolve_model,
    safe_unlink,
    write_jsonl,
)
from score_reason_rewrite_local import parse_model_output  # noqa: E402

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


DEFAULT_INPUT_PATH = Path("datasets/train/final_score_reason_revision_sft/final_score_reason_revision_sft_test.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/train/final_score_reason_revision_sft_eval_outputs")
DEFAULT_JUDGE_CONFIG_PATH = Path("scripts/baseline/judge_model_config.example.json")

REQUIRED_LABEL_KEYS = ("score", "reason", "revision_suggestions", "modified_answer")


@dataclass
class JudgeConfig:
    name: str
    base_urls: tuple[str, ...]
    model: str = ""
    api_key: str = DEFAULT_API_KEY
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 120.0
    max_retries: int = 5
    retry_delay: float = 3.0
    system_override: str | None = None
    auto_fetch_model: bool = True
    healthcheck: bool = True
    healthcheck_retries: int = 60
    healthcheck_interval: float = 2.0


def normalize_client_host(host: str) -> str:
    value = str(host or "").strip()
    if value in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return value or "127.0.0.1"


def normalize_base_url(value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    if not base_url:
        return ""
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return base_url


def dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def is_loopback_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def parse_ports(raw_ports: Any) -> tuple[int, ...]:
    if raw_ports is None or raw_ports == "":
        return ()
    if not isinstance(raw_ports, list):
        raise ValueError("'ports' must be a JSON array.")
    return tuple(int(port) for port in raw_ports)


def resolve_config_base_urls(item: dict[str, Any]) -> tuple[str, ...]:
    host = normalize_client_host(str(item.get("host", "127.0.0.1")))
    raw_base_urls = item.get("base_urls")
    if raw_base_urls is not None:
        if not isinstance(raw_base_urls, list):
            raise ValueError("'base_urls' must be a JSON array.")
        return tuple(dedupe_preserve_order([normalize_base_url(str(url)) for url in raw_base_urls]))

    ports = parse_ports(item.get("ports"))
    if ports:
        return tuple(f"http://{host}:{port}/v1" for port in ports)

    raw_base_url = normalize_base_url(str(item.get("base_url", "")))
    if raw_base_url:
        return (raw_base_url,)

    port = item.get("port")
    if port is not None:
        return (f"http://{host}:{int(port)}/v1",)

    return ()


def load_judge_configs(path: Path) -> list[JudgeConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    judges_payload = payload.get("judges", []) if isinstance(payload, dict) else payload
    if not isinstance(judges_payload, list):
        raise ValueError("Judge config must be a list or an object with key 'judges'.")

    judges: list[JudgeConfig] = []
    for index, item in enumerate(judges_payload):
        if not isinstance(item, dict):
            raise ValueError(f"Judge entry {index} is not a JSON object.")

        name = normalize_text(item.get("name"))
        base_urls = resolve_config_base_urls(item)
        if not name or not base_urls:
            raise ValueError(f"Judge entry {index} must contain name and endpoint config.")

        api_key = item.get("api_key")
        api_key_env = item.get("api_key_env")
        if not api_key and api_key_env:
            api_key = os.getenv(str(api_key_env), "")

        judges.append(
            JudgeConfig(
                name=name,
                base_urls=base_urls,
                model=normalize_text(item.get("model")),
                api_key=str(api_key or DEFAULT_API_KEY),
                temperature=float(item.get("temperature", 0.0)),
                max_tokens=int(item.get("max_tokens", 1024)),
                timeout=float(item.get("timeout", 120.0)),
                max_retries=int(item.get("max_retries", 5)),
                retry_delay=float(item.get("retry_delay", 3.0)),
                system_override=(
                    str(item.get("system_override")).strip()
                    if item.get("system_override") is not None
                    else None
                ),
                auto_fetch_model=bool(item.get("auto_fetch_model", True)),
                healthcheck=bool(item.get("healthcheck", True)),
                healthcheck_retries=int(item.get("healthcheck_retries", 60)),
                healthcheck_interval=float(item.get("healthcheck_interval", 2.0)),
            )
        )
    return judges


def wait_for_ready_base_urls(judge: JudgeConfig, *, skip_health_check: bool) -> None:
    if skip_health_check or not judge.healthcheck:
        return

    timeout = max(1, int(min(10.0, judge.timeout)))
    for base_url in judge.base_urls:
        for attempt in range(max(1, judge.healthcheck_retries)):
            if models_endpoint_ready(base_url, timeout):
                break
            if attempt < judge.healthcheck_retries - 1:
                time.sleep(judge.healthcheck_interval)
        else:
            raise RuntimeError(f"Endpoint is not ready: {base_url.rstrip('/')}/models")


def resolve_judge_model(judge: JudgeConfig, *, health_check_timeout: int) -> str:
    if judge.model:
        return judge.model
    if not judge.auto_fetch_model:
        raise ValueError(f"Judge '{judge.name}' has empty model and auto_fetch_model=false.")
    return resolve_model("", list(judge.base_urls), health_check_timeout)


def sample_id_from_record(record: dict[str, Any], index: int) -> str:
    explicit = normalize_text(record.get("sample_id") or record.get("id") or record.get("unique_id"))
    if explicit:
        return explicit
    raw = json.dumps(record.get("messages", record), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(f"{index}||{raw}".encode("utf-8")).hexdigest()


def split_messages(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Record has no valid messages list.")

    prompt_messages: list[dict[str, str]] = []
    assistant_contents: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = str(message.get("content") or "")
        if role == "assistant":
            assistant_contents.append(content)
        elif role in {"system", "user"}:
            prompt_messages.append({"role": role, "content": content})

    if not prompt_messages:
        raise ValueError("Record has no system/user prompt messages.")
    if not assistant_contents:
        raise ValueError("Record has no assistant reference label.")
    return prompt_messages, assistant_contents[-1]


def parse_label(text: str) -> dict[str, Any]:
    parsed = parse_model_output(text)
    return {
        "score": parsed["score"],
        "reason": parsed["reason"],
        "revision_suggestions": parsed["revision_suggestions"],
        "modified_answer": parsed["modified_answer"],
        "raw_output": parsed["raw_output"],
        "parse_error": parsed["parse_error"],
        "strict_json_ok": parsed["strict_json_ok"],
    }


def prepare_samples(records: list[dict[str, Any]], *, skip_bad_records: bool) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(records):
        try:
            prompt_messages, reference_output = split_messages(record)
            reference = parse_label(reference_output)
            sample_id = sample_id_from_record(record, index)
            samples.append(
                {
                    "sample_id": sample_id,
                    "index": index,
                    "prompt_messages": prompt_messages,
                    "reference_output": reference_output,
                    "reference": reference,
                }
            )
        except Exception:
            skipped += 1
            if not skip_bad_records:
                raise
    if skipped:
        print(f"[load] skipped_bad_records={skipped}")
    return samples


def apply_system_override(messages: list[dict[str, str]], judge: JudgeConfig) -> list[dict[str, str]]:
    if not judge.system_override:
        return messages
    out: list[dict[str, str]] = []
    replaced = False
    for message in messages:
        if message["role"] == "system" and not replaced:
            out.append({"role": "system", "content": judge.system_override})
            replaced = True
        else:
            out.append(message)
    if not replaced:
        out.insert(0, {"role": "system", "content": judge.system_override})
    return out


def assistant_json(label: dict[str, Any]) -> str:
    payload = {
        "score": label.get("score"),
        "reason": label.get("reason") or "",
        "revision_suggestions": label.get("revision_suggestions") or "",
        "modified_answer": label.get("modified_answer") or "",
    }
    return json.dumps(payload, ensure_ascii=False)


def build_sft_row(sample: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    label = {
        "score": row.get("predicted_score"),
        "reason": row.get("predicted_reason"),
        "revision_suggestions": row.get("predicted_revision_suggestions"),
        "modified_answer": row.get("predicted_modified_answer"),
    }
    return {
        "messages": [
            *sample["prompt_messages"],
            {"role": "assistant", "content": assistant_json(label)},
        ]
    }


def run_one(
    sample: dict[str, Any],
    task_index: int,
    judge: JudgeConfig,
    model: str,
) -> dict[str, Any]:
    started = time.time()
    messages = apply_system_override(sample["prompt_messages"], judge)
    reference = sample["reference"]
    row: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "index": sample["index"],
        "judge_name": judge.name,
        "judge_model": model,
        "judge_base_urls": list(judge.base_urls),
        "reference_score": reference.get("score"),
        "reference_reason": reference.get("reason"),
        "reference_revision_suggestions": reference.get("revision_suggestions"),
        "reference_modified_answer": reference.get("modified_answer"),
        "reference_output": sample["reference_output"],
        "reference_strict_json_ok": reference.get("strict_json_ok"),
        "reference_parse_error": reference.get("parse_error"),
        "predicted_score": None,
        "predicted_reason": "",
        "predicted_revision_suggestions": "",
        "predicted_modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "latency_sec": None,
        "strict_json_ok": False,
        "messages": [],
    }

    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=list(judge.base_urls),
            task_index=task_index,
            api_key=judge.api_key,
            model=model,
            messages=messages,
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
            timeout_seconds=int(judge.timeout),
            retries=max(0, judge.max_retries - 1),
            retry_sleep=judge.retry_delay,
        )
        parsed = parse_label(raw_text)
        ok = (
            parsed["score"] is not None
            and bool(parsed["reason"])
            and bool(parsed["revision_suggestions"])
            and bool(parsed["modified_answer"])
        )
        row.update(
            {
                "predicted_score": parsed["score"],
                "predicted_reason": parsed["reason"],
                "predicted_revision_suggestions": parsed["revision_suggestions"],
                "predicted_modified_answer": parsed["modified_answer"],
                "raw_output": parsed["raw_output"],
                "ok": ok,
                "parse_error": parsed["parse_error"] if not ok else None,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
                "strict_json_ok": parsed["strict_json_ok"],
            }
        )
    except Exception as exc:
        row.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})

    if row["ok"]:
        row["messages"] = build_sft_row(sample, row)["messages"]
    return row


def load_completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for row in load_jsonl(path):
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            rows[sample_id] = row
    return rows


def format_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    den_x = sum((a - x_mean) ** 2 for a in x)
    den_y = sum((b - y_mean) ** 2 for b in y)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / math.sqrt(den_x * den_y)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(row["reference_score"]), float(row["predicted_score"]))
        for row in rows
        if isinstance(row.get("reference_score"), int)
        and isinstance(row.get("predicted_score"), int)
    ]
    refs = [a for a, _ in pairs]
    preds = [b for _, b in pairs]
    abs_errors = [abs(a - b) for a, b in pairs]
    sq_errors = [(a - b) ** 2 for a, b in pairs]
    exact = sum(1 for a, b in pairs if a == b)
    off_by_1 = sum(1 for a, b in pairs if abs(a - b) <= 1)
    return {
        "count": len(pairs),
        "reference_mean": format_float(average(refs)),
        "predicted_mean": format_float(average(preds)),
        "exact_match_accuracy": format_float(exact / len(pairs)) if pairs else None,
        "off_by_1_accuracy": format_float(off_by_1 / len(pairs)) if pairs else None,
        "mae": format_float(average(abs_errors)),
        "rmse": format_float(math.sqrt(average(sq_errors))) if sq_errors else None,
        "pearson": format_float(pearson(refs, preds)),
    }


def run_judge(
    *,
    samples: list[dict[str, Any]],
    judge: JudgeConfig,
    model: str,
    output_dir: Path,
    workers: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    predictions_dir = output_dir / "predictions"
    sft_dir = output_dir / "sft_outputs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = predictions_dir / f"{judge.name}.jsonl"
    sft_path = sft_dir / f"{judge.name}.jsonl"
    safe_unlink(prediction_path, overwrite)
    safe_unlink(sft_path, overwrite)

    completed = load_completed_rows(prediction_path) if resume else {}
    pending = [sample for sample in samples if sample["sample_id"] not in completed or not completed[sample["sample_id"]].get("ok")]

    print(
        f"[judge] {judge.name} model={model} endpoints={list(judge.base_urls)} "
        f"total={len(samples)} completed={len(completed)} pending={len(pending)}"
    )

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc=judge.name, ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(run_one, sample, index, judge, model): sample
                for index, sample in enumerate(pending)
            }
            for future in as_completed(futures):
                row = future.result()
                completed[row["sample_id"]] = row
                with lock:
                    append_jsonl(prediction_path, row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    all_rows_by_id = load_completed_rows(prediction_path)
    ordered_rows = [
        all_rows_by_id[sample["sample_id"]]
        for sample in sorted(samples, key=lambda item: item["index"])
        if sample["sample_id"] in all_rows_by_id
    ]
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    sft_rows = [
        build_sft_row(sample_by_id[row["sample_id"]], row)
        for row in ordered_rows
        if row.get("ok") is True and row["sample_id"] in sample_by_id
    ]
    write_jsonl(sft_path, sft_rows)

    request_failures = sum(1 for row in ordered_rows if row.get("request_error"))
    parse_failures = sum(1 for row in ordered_rows if row.get("parse_error"))
    strict_json_ok = sum(1 for row in ordered_rows if row.get("strict_json_ok"))
    ok_count = sum(1 for row in ordered_rows if row.get("ok") is True)

    return {
        "judge": {
            **asdict(judge),
            "model": model,
            "base_urls": list(judge.base_urls),
        },
        "prediction_file": str(prediction_path),
        "sft_output_file": str(sft_path),
        "total_samples": len(samples),
        "completed_samples": len(ordered_rows),
        "ok_samples": ok_count,
        "sft_rows": len(sft_rows),
        "request_failures": request_failures,
        "parse_failures": parse_failures,
        "strict_json_ok": strict_json_ok,
        "strict_json_success_rate": format_float(strict_json_ok / len(ordered_rows)) if ordered_rows else None,
        "metrics": compute_metrics(ordered_rows),
    }


def filter_judges(judges: list[JudgeConfig], only_models: list[str]) -> list[JudgeConfig]:
    if not only_models:
        return judges
    wanted = {name.strip() for name in only_models if name.strip()}
    return [judge for judge in judges if judge.name in wanted]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate judge models on constructed score/reason/revision SFT JSONL data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--judge-config", type=Path, default=DEFAULT_JUDGE_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--only-models", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-bad-records", action="store_true", default=True)
    parser.add_argument(
        "--strict-records",
        action="store_true",
        help="Fail on malformed input records instead of skipping them.",
    )
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-timeout", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.judge_config.is_file():
        raise FileNotFoundError(f"Judge config file not found: {args.judge_config}")

    records = load_jsonl(args.input)
    if args.sample_size is not None:
        records = records[: max(0, args.sample_size)]
    samples = prepare_samples(records, skip_bad_records=args.skip_bad_records and not args.strict_records)
    judges = filter_judges(load_judge_configs(args.judge_config), args.only_models)
    if not judges:
        raise ValueError("No judge models selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = args.output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    judge_summaries: list[dict[str, Any]] = []
    for judge in judges:
        wait_for_ready_base_urls(judge, skip_health_check=args.skip_health_check)
        model = resolve_judge_model(judge, health_check_timeout=args.health_check_timeout)
        summary = run_judge(
            samples=samples,
            judge=judge,
            model=model,
            output_dir=args.output_dir,
            workers=args.workers,
            resume=not args.no_resume,
            overwrite=args.overwrite,
        )
        judge_summaries.append(summary)

    summary = {
        "input_path": str(args.input),
        "judge_config_path": str(args.judge_config),
        "sample_count": len(samples),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_format": "sft_messages_with_predicted_assistant_json",
        "required_label_keys": list(REQUIRED_LABEL_KEYS),
        "judge_summaries": judge_summaries,
    }
    summary_path = summaries_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = summaries_dir / "summary.csv"
    csv_lines = [
        "judge_name,total_samples,ok_samples,sft_rows,valid_score_pairs,exact_match_accuracy,off_by_1_accuracy,mae,rmse,pearson,strict_json_success_rate,request_failures,parse_failures"
    ]
    for item in judge_summaries:
        metrics = item["metrics"]
        csv_lines.append(
            ",".join(
                [
                    item["judge"]["name"],
                    str(item["total_samples"]),
                    str(item["ok_samples"]),
                    str(item["sft_rows"]),
                    str(metrics["count"]),
                    str(metrics["exact_match_accuracy"]),
                    str(metrics["off_by_1_accuracy"]),
                    str(metrics["mae"]),
                    str(metrics["rmse"]),
                    str(metrics["pearson"]),
                    str(item["strict_json_success_rate"]),
                    str(item["request_failures"]),
                    str(item["parse_failures"]),
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary["judge_summaries"], ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
