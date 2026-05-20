#!/usr/bin/env python3
"""Build score/reason/revision SFT data with local vLLM endpoints.

Generates train/test JSONL in the same messages format as:
  datasets/train/final_score_reason_revision_sft/final_score_reason_revision_sft_test.jsonl

Pipeline:
1) For target scores < 5, generate a candidate answer at the target score.
2) Score + reason + revision + rewrite the candidate answer into strict JSON.
3) Write SFT rows with system/user/assistant messages.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
EVAL_PIPELINE_DIR = ROOT / "scripts" / "evaluation_pipeline"
sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    append_jsonl,
    build_base_urls,
    call_chat_with_retries,
    extract_json_object,
    format_score_criteria,
    iter_jsonl,
    models_endpoint_ready,
    normalize_text,
    parse_int_score,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
)
from regenerate_sft_score_reason_revision_local import sample_from_record  # noqa: E402
from score_reason_rewrite_local import parse_model_output  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


LOW_SCORE_SYSTEM_PROMPT = (
    "You are a data generation assistant for evaluation training. "
    "You will receive a question, one evaluation dimension, and the complete 0-5 score criteria. "
    "Generate a candidate answer that would receive exactly the target score on the given dimension. "
    "The weakness must appear in the answer itself, not only in the reason. "
    "Return strict JSON only with keys: score, reason, generated_answer."
)

LOW_SCORE_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Score Criteria (0-5):\n{criteria_text}\n\n"
    "Target Score:\n{target_score}\n\n"
    "Return JSON only with keys:\n"
    "{\"score\": <int>, \"reason\": \"...\", \"generated_answer\": \"...\"}"
)


DEFAULT_INPUT = Path("datasets/0-5/score_criteria_0_5.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/train/final_score_reason_revision_sft_new")


def build_base_id(record: Dict[str, Any]) -> str:
    parts = [
        str(record.get("question", "")).strip(),
        str(record.get("answer", record.get("reference_answer", ""))).strip(),
        str(record.get("dimension_name", record.get("evaluation_dimension", ""))).strip(),
        str(record.get("full_score_criteria", "")).strip(),
    ]
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return digest


def parse_target_scores(values: Optional[List[str]]) -> List[int]:
    if not values:
        return [0, 1, 2, 3, 4, 5]
    scores: List[int] = []
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if "," in text:
            for piece in text.split(","):
                piece = piece.strip()
                if piece:
                    scores.append(int(piece))
            continue
        if "-" in text:
            start_text, end_text = text.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            scores.extend(list(range(start, end + step, step)))
        else:
            scores.append(int(text))
    deduped: List[int] = []
    seen: set[int] = set()
    for score in scores:
        if 0 <= score <= 5 and score not in seen:
            seen.add(score)
            deduped.append(score)
    return deduped


def build_task_id(base_id: str, target_score: int) -> str:
    return f"{base_id}::score={target_score}"


def load_completed_task_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.is_file():
        return completed
    for row in iter_jsonl(path):
        task_id = normalize_text(row.get("task_id"))
        if task_id:
            completed.add(task_id)
    return completed


def sample_records(path: Path, count: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if count <= 0:
        return []
    if not shuffle:
        rows: List[Dict[str, Any]] = []
        for row in iter_jsonl(path):
            rows.append(row)
            if len(rows) >= count:
                break
        return rows
    reservoir: List[Dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(path)):
        if len(reservoir) < count:
            reservoir.append(row)
        else:
            j = rng.randint(0, index)
            if j < count:
                reservoir[j] = row
    rng.shuffle(reservoir)
    return reservoir


def resolve_question(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("question"))


def resolve_dimension(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("dimension_name") or record.get("evaluation_dimension"))


def resolve_reference_answer(record: Dict[str, Any]) -> str:
    return normalize_text(record.get("answer") or record.get("reference_answer") or record.get("model_answer"))


def resolve_criteria_text(record: Dict[str, Any]) -> str:
    criteria = record.get("score_criteria") or record.get("criteria") or record.get("full_score_criteria")
    return format_score_criteria(criteria, record)


def build_low_score_messages(
    *,
    question: str,
    dimension_name: str,
    criteria_text: str,
    target_score: int,
) -> List[Dict[str, str]]:
    user_content = LOW_SCORE_USER_TEMPLATE.format(
        question=question,
        dimension_name=dimension_name,
        criteria_text=criteria_text,
        target_score=target_score,
    )
    return [
        {"role": "system", "content": LOW_SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_low_score_output(text: str) -> Dict[str, Any]:
    parsed = extract_json_object(text)
    if not isinstance(parsed, dict):
        return {
            "score": None,
            "reason": "",
            "generated_answer": "",
        }
    score = parse_int_score(parsed.get("score") or parsed.get("Score"))
    reason = normalize_text(parsed.get("reason") or parsed.get("Reason"))
    generated_answer = normalize_text(
        parsed.get("generated_answer")
        or parsed.get("Generated Answer")
        or parsed.get("answer")
        or parsed.get("Answer")
    )
    return {
        "score": score,
        "reason": reason,
        "generated_answer": generated_answer,
    }


def build_sft_row(sample: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "score": parsed["score"],
        "reason": parsed["reason"],
        "revision_suggestions": parsed["revision_suggestions"],
        "modified_answer": parsed["modified_answer"],
    }
    return {
        "messages": [
            sample["messages"][0],
            sample["messages"][1],
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    }


def filter_ready_base_urls(
    base_urls: List[str],
    *,
    timeout_seconds: int,
    workers: int,
) -> List[str]:
    ready_urls: List[str] = []
    lock = threading.Lock()

    def _probe(base_url: str) -> None:
        if models_endpoint_ready(base_url, timeout_seconds):
            with lock:
                ready_urls.append(base_url)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_probe, base_url) for base_url in base_urls]
        for future in as_completed(futures):
            future.result()

    ready_set = set(ready_urls)
    return [base_url for base_url in base_urls if base_url in ready_set]


def resolve_runtime(args: argparse.Namespace) -> Tuple[List[str], str]:
    base_urls = build_base_urls(args.base_url_template, parse_ports(args.ports))
    if args.skip_health_check:
        pass
    elif args.wait_all_ports:
        wait_for_servers(base_urls, args.health_check_timeout, args.health_check_interval)
    else:
        before = len(base_urls)
        base_urls = filter_ready_base_urls(
            base_urls,
            timeout_seconds=args.health_check_timeout,
            workers=args.probe_workers,
        )
        print(f"[ports] ready={len(base_urls)} / requested={before}")
        if not base_urls:
            raise RuntimeError(
                "No ready vLLM endpoints found. Check --ports/--base-url-template, "
                "or pass --wait-all-ports if servers are still starting."
            )
    model = args.model.strip() or resolve_model("", base_urls, args.health_check_timeout)
    return base_urls, model


def run_task(
    task: Dict[str, Any],
    *,
    args: argparse.Namespace,
    base_urls: List[str],
    model: str,
) -> Dict[str, Any]:
    started = time.time()
    result: Dict[str, Any] = {
        "task_id": task["task_id"],
        "split": task["split"],
        "target_score": task["target_score"],
        "sample_id": task["sample_id"],
        "question": task["question"],
        "dimension_name": task["dimension_name"],
        "criteria_text": task["criteria_text"],
        "candidate_answer": "",
        "score": None,
        "reason": "",
        "revision_suggestions": "",
        "modified_answer": "",
        "raw_output": "",
        "ok": False,
        "parse_error": None,
        "request_error": None,
        "endpoint": "",
        "model": model,
        "latency_sec": None,
        "strict_json_ok": False,
        "sft_row": None,
    }

    for attempt in range(max(1, args.max_attempts)):
        try:
            if task["target_score"] == 5 and args.use_reference_for_score5:
                candidate_answer = task["reference_answer"]
            else:
                messages = build_low_score_messages(
                    question=task["question"],
                    dimension_name=task["dimension_name"],
                    criteria_text=task["criteria_text"],
                    target_score=task["target_score"],
                )
                raw_text, endpoint = call_chat_with_retries(
                    base_urls=base_urls,
                    task_index=task["index"],
                    api_key=args.api_key,
                    model=model,
                    messages=messages,
                    temperature=args.gen_temperature,
                    max_tokens=args.gen_max_tokens,
                    timeout_seconds=args.request_timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                parsed = parse_low_score_output(raw_text)
                candidate_answer = parsed.get("generated_answer", "")
                if not candidate_answer:
                    result["parse_error"] = "Empty generated_answer from low-score generation."
                    continue

            record = {
                "sample_id": task["sample_id"],
                "question": task["question"],
                "answer": candidate_answer,
                "dimension_name": task["dimension_name"],
                "score_criteria": task["score_criteria"],
                "full_score_criteria": task["full_score_criteria"],
            }
            record.update(task.get("criteria_fields") or {})
            sample = sample_from_record(record, task["index"], task["split"])
            if sample is None:
                result["parse_error"] = "Failed to build sample from record."
                continue

            result["sample_id"] = sample.get("sample_id", result["sample_id"])

            raw_text, endpoint = call_chat_with_retries(
                base_urls=base_urls,
                task_index=task["index"],
                api_key=args.api_key,
                model=model,
                messages=sample["messages"],
                temperature=args.score_temperature,
                max_tokens=args.score_max_tokens,
                timeout_seconds=args.request_timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            parsed = parse_model_output(raw_text)
            ok = (
                parsed["score"] is not None
                and 0 <= int(parsed["score"]) <= 5
                and bool(parsed["reason"])
                and bool(parsed["revision_suggestions"])
                and bool(parsed["modified_answer"])
            )
            if ok and args.enforce_target_score and parsed["score"] != task["target_score"]:
                ok = False
                parsed["parse_error"] = f"Score mismatch: got {parsed['score']} vs target {task['target_score']}"

            result.update(
                {
                    "candidate_answer": candidate_answer,
                    "score": parsed["score"],
                    "reason": parsed["reason"],
                    "revision_suggestions": parsed["revision_suggestions"],
                    "modified_answer": parsed["modified_answer"],
                    "raw_output": parsed["raw_output"],
                    "ok": ok,
                    "parse_error": parsed["parse_error"] if not ok else None,
                    "endpoint": endpoint,
                    "strict_json_ok": parsed["strict_json_ok"],
                    "latency_sec": round(time.time() - started, 4),
                    "sft_row": build_sft_row(sample, parsed) if ok else None,
                }
            )
            return result
        except Exception as exc:
            result.update({
                "request_error": str(exc),
                "latency_sec": round(time.time() - started, 4),
            })
            time.sleep(args.retry_sleep)

    return result


def build_tasks(
    *,
    base_records: List[Dict[str, Any]],
    split: str,
    target_scores: List[int],
    max_count: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    index = 0
    for record in base_records:
        question = resolve_question(record)
        dimension = resolve_dimension(record)
        reference_answer = resolve_reference_answer(record)
        criteria_text = resolve_criteria_text(record)
        score_criteria = record.get("score_criteria") or record.get("criteria")
        full_score_criteria = record.get("full_score_criteria")
        criteria_fields = {k: record[k] for k in record if str(k).startswith("criteria_")}
        if not (question and dimension and criteria_text and reference_answer):
            continue
        base_id = record.get("sample_id") or build_base_id(record)

        scores = list(target_scores)
        rng.shuffle(scores)
        for target_score in scores:
            task_id = build_task_id(str(base_id), int(target_score))
            tasks.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "target_score": int(target_score),
                    "sample_id": task_id,
                    "question": question,
                    "dimension_name": dimension,
                    "reference_answer": reference_answer,
                    "criteria_text": criteria_text,
                    "score_criteria": score_criteria,
                    "full_score_criteria": full_score_criteria,
                    "criteria_fields": criteria_fields,
                    "index": index,
                }
            )
            index += 1
            if max_count and len(tasks) >= max_count:
                return tasks[:max_count]
    return tasks[:max_count]


def generate_split(
    *,
    split: str,
    tasks: List[Dict[str, Any]],
    output_path: Path,
    cache_path: Path,
    args: argparse.Namespace,
    base_urls: List[str],
    model: str,
) -> None:
    safe_unlink(cache_path, args.overwrite)
    safe_unlink(output_path, args.overwrite)

    completed = load_completed_task_ids(cache_path)
    pending = [task for task in tasks if task["task_id"] not in completed]

    print(f"[run] split={split} output={output_path}")
    print(f"[run] split={split} cache={cache_path}")
    print(f"[run] split={split} total={len(tasks)} completed={len(completed)} pending={len(pending)}")

    lock = threading.Lock()
    progress = tqdm(total=len(pending), desc=f"{split}_build", ncols=100) if tqdm is not None else None
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_task, task, args=args, base_urls=base_urls, model=model): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                sft_row = row.pop("sft_row", None)
                with lock:
                    append_jsonl(cache_path, row)
                    if row.get("ok") is True and sft_row is not None:
                        append_jsonl(output_path, sft_row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    print(f"[done] split={split} sft_rows_written={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build score/reason/revision SFT data with local vLLM endpoints."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-output-name", default="final_score_reason_revision_sft_train.jsonl")
    parser.add_argument("--test-output-name", default="final_score_reason_revision_sft_test.jsonl")
    parser.add_argument("--train-cache-name", default="final_score_reason_revision_sft_train_cache.jsonl")
    parser.add_argument("--test-cache-name", default="final_score_reason_revision_sft_test_cache.jsonl")

    parser.add_argument("--train-count", type=int, default=500000)
    parser.add_argument("--test-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--target-scores", nargs="*", default=["0-5"])

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8000-8003"], help="Ports like: 8000-8003 or 8000 8001.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--gen-temperature", type=float, default=0.3)
    parser.add_argument("--score-temperature", type=float, default=0.0)
    parser.add_argument("--gen-max-tokens", type=int, default=512)
    parser.add_argument("--score-max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--enforce-target-score", action="store_true", default=False)
    parser.add_argument("--use-reference-for-score5", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--wait-all-ports", action="store_true")
    parser.add_argument("--probe-workers", type=int, default=32)
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_scores = parse_target_scores(args.target_scores)
    if not target_scores:
        raise ValueError("No valid --target-scores provided.")

    total_count = int(args.train_count) + int(args.test_count)
    per_record = max(1, len(target_scores))
    base_train = (int(args.train_count) + per_record - 1) // per_record
    base_test = (int(args.test_count) + per_record - 1) // per_record
    base_total = base_train + base_test

    rng = random.Random(args.seed)
    base_records = sample_records(args.input, base_total, args.seed, args.shuffle)
    if len(base_records) < base_total:
        raise ValueError(
            f"Not enough base records. Need {base_total}, got {len(base_records)} from {args.input}."
        )

    base_train_records = base_records[:base_train]
    base_test_records = base_records[base_train : base_train + base_test]

    train_tasks = build_tasks(
        base_records=base_train_records,
        split="train",
        target_scores=target_scores,
        max_count=int(args.train_count),
        rng=rng,
    )
    test_tasks = build_tasks(
        base_records=base_test_records,
        split="test",
        target_scores=target_scores,
        max_count=int(args.test_count),
        rng=rng,
    )

    base_urls, model = resolve_runtime(args)
    print(f"[base_urls] {base_urls}")
    print(f"[model] {model}")
    print(f"[targets] {target_scores}")

    if train_tasks:
        generate_split(
            split="train",
            tasks=train_tasks,
            output_path=output_dir / args.train_output_name,
            cache_path=output_dir / args.train_cache_name,
            args=args,
            base_urls=base_urls,
            model=model,
        )

    if test_tasks:
        generate_split(
            split="test",
            tasks=test_tasks,
            output_path=output_dir / args.test_output_name,
            cache_path=output_dir / args.test_cache_name,
            args=args,
            base_urls=base_urls,
            model=model,
        )


if __name__ == "__main__":
    main()
