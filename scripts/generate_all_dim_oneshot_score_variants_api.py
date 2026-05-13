#!/usr/bin/env python
"""Generate all-dimension one-shot 0-5 raw samples via multi-endpoint API."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import create_jsonl as base
import create_jsonl_0_5 as score05
import generate_score_variants_api as api_base


DEFAULT_INPUT = Path("datasets/0-5/full_score_dimension_seeds.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/all_dim_oneshot_score_variants")
DEFAULT_TARGET_SCORES = (0, 1, 2, 3, 4)
DEFAULT_PORTS = (8000, 8001, 8002, 8003)
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_MODEL = "Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_DISPATCH_POLL_INTERVAL = 0.05
DEFAULT_SCORE_5_OUTPUT = "score_5.jsonl"
FULL_SCORE_REASON_ROOT = Path("datasets/filled_iteration_train_outputs/datasets")

SYSTEM_PROMPT = (
    "You are a data generation assistant for one-shot answer revision training. "
    "You will receive a question, a high-quality reference answer, all evaluation "
    "dimensions with their full-score criteria, and a target overall score from 0 to 4. "
    "Generate one realistic answer that would receive exactly that overall score when "
    "judged jointly across all dimensions. Then produce one concise overall reason and "
    "executable revision_suggestions for fixing the weaknesses, plus "
    "one revised answer that clearly fixes the weaknesses and satisfies the dimensions "
    "well. Return strict JSON only."
)

USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "High-Quality Reference Answer:\n{answer}\n\n"
    "Evaluation Dimensions:\n{evaluation_dimensions}\n\n"
    "Target Overall Score:\n{target_score}\n\n"
    "Requirements:\n"
    "1. Generate a plausible answer that would score exactly the target overall score.\n"
    "2. Judge the answer jointly across all dimensions, not just one dimension in isolation.\n"
    "3. The weaknesses should be coherent and realistic for that target score.\n"
    "4. Keep the answer relevant to the question.\n"
    "5. Write one concise overall reason explaining why the generated answer deserves the target score.\n"
    "6. The reason must integrate all important dimensions instead of listing disconnected fragments.\n"
    "7. Generate revision_suggestions as concrete edit instructions for moving the generated answer to full score.\n"
    "8. Also generate a modified answer that clearly improves the generated answer and aims to satisfy the dimensions jointly.\n"
    "9. The modified answer must stay on-task, complete the reasoning, and avoid mentioning the rubric, target score, or evaluation process.\n"
    "10. Return JSON only with this schema:\n"
    "{{\"average_score\": {target_score}, \"reason\": \"...\", \"revision_suggestions\": \"...\", \"generated_answer\": \"...\", \"modified_answer\": \"...\"}}"
)


def _iter_records(path: Path) -> list[dict[str, Any]]:
    return api_base.load_records(path)


def build_grouped_seed_id(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("question", "")).strip(),
        str(record.get("answer", "")).strip(),
        json.dumps(record.get("evaluation_dimensions", []), ensure_ascii=False, sort_keys=True),
    ]
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def group_full_score_seeds(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_records(path):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        dim_name = str(row.get("dimension_name") or row.get("name") or "").strip()
        criteria = str(
            row.get("full_score_criteria")
            or row.get("criteria")
            or row.get("dimension_criteria")
            or ""
        ).strip()
        if base._is_empty(question) or base._is_empty(answer):
            continue
        if base._is_empty(dim_name) or base._is_empty(criteria):
            continue
        grouped[(question, answer)].append(
            {
                "dimension_name": dim_name,
                "full_score_criteria": criteria,
            }
        )

    seeds: list[dict[str, Any]] = []
    for (question, answer), dims in grouped.items():
        seen: set[str] = set()
        evaluation_dimensions: list[dict[str, Any]] = []
        for item in dims:
            key = base._normalize_key(item["dimension_name"])
            if key in seen:
                continue
            seen.add(key)
            evaluation_dimensions.append(item)

        seed = {
            "question": question,
            "answer": answer,
            "evaluation_dimensions": evaluation_dimensions,
        }
        seed["sample_id"] = build_grouped_seed_id(seed)
        seeds.append(seed)

    seeds.sort(key=lambda x: (x["question"], x["answer"]))
    return seeds


def build_messages(record: dict[str, Any], target_score: int) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        question=str(record.get("question", "")).strip(),
        answer=str(record.get("answer", "")).strip(),
        evaluation_dimensions=json.dumps(
            record.get("evaluation_dimensions", []), ensure_ascii=False, indent=2
        ),
        target_score=target_score,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_exact_score(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    text = _normalize_text(value)
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return int(numeric) if numeric.is_integer() else numeric


def normalize_generation_payload(
    parsed: dict[str, Any],
    *,
    target_score: int,
) -> dict[str, Any] | None:
    parsed_score = _coerce_exact_score(
        api_base.get_payload_value(
            parsed,
            "average_score",
            "AverageScore",
            "score",
            "Score",
        )
    )
    if parsed_score != target_score:
        return None

    generated_answer = _normalize_text(
        api_base.get_payload_value(parsed, "generated_answer", "answer")
    )
    reason = _normalize_text(api_base.get_payload_value(parsed, "reason", "Reason"))
    revision_suggestions = _normalize_text(
        api_base.get_payload_value(
            parsed,
            "revision_suggestions",
            "edit_intent",
            "Revision Suggestions",
            "Edit Intent",
        )
    )
    modified_answer = _normalize_text(
        api_base.get_payload_value(
            parsed,
            "modified_answer",
            "revised_answer",
            "corrected_answer",
        )
    )
    if (
        base._is_empty(generated_answer)
        or base._is_empty(reason)
        or base._is_empty(revision_suggestions)
        or base._is_empty(modified_answer)
    ):
        return None

    return {
        "average_score": parsed_score,
        "reason": reason,
        "revision_suggestions": revision_suggestions,
        "generated_answer": generated_answer,
        "modified_answer": modified_answer,
    }


def build_generation_tasks(
    *,
    seeds: list[dict[str, Any]],
    targets: list[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for target_score in targets:
        output_path = output_dir / f"score_{target_score}.jsonl"
        completed_ids = api_base.load_completed_ids(output_path)
        remaining = 0
        for seed in seeds:
            sample_id = str(seed["sample_id"])
            if sample_id in completed_ids:
                continue
            remaining += 1
            tasks.append(
                {
                    "record": seed,
                    "sample_id": sample_id,
                    "target_score": target_score,
                    "output_path": output_path,
                }
            )
        print(
            f"[Target {target_score}] output={output_path} total={len(seeds)} "
            f"completed={len(completed_ids)} remaining={remaining}"
        )
    return tasks


def build_score5_reason(
    *,
    seed: dict[str, Any],
    reason_lookup: score05.FullScoreReasonLookup | None,
) -> str:
    chunks: list[str] = []
    question = str(seed.get("question", "")).strip()
    answer = str(seed.get("answer", "")).strip()
    for item in seed.get("evaluation_dimensions", []):
        if not isinstance(item, dict):
            continue
        dimension_name = str(item.get("dimension_name") or "").strip()
        if base._is_empty(dimension_name):
            continue

        resolved_reason = ""
        if reason_lookup is not None:
            resolved_reason, _ = reason_lookup.resolve(
                question=question,
                answer=answer,
                dimension_name=dimension_name,
            )
        if base._is_empty(resolved_reason):
            resolved_reason = f"The answer satisfies the full-score criteria for {dimension_name}."
        chunks.append(f"{dimension_name}: {' '.join(str(resolved_reason).split())}")
    return " | ".join(chunks)


def write_score5_rows(
    *,
    seeds: list[dict[str, Any]],
    output_path: Path,
    reason_lookup: score05.FullScoreReasonLookup | None,
) -> None:
    completed_ids = api_base.load_completed_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8") as f:
        for seed in seeds:
            sample_id = str(seed["sample_id"])
            if sample_id in completed_ids:
                continue
            row = {
                "sample_id": sample_id,
                "question": seed["question"],
                "answer": seed["answer"],
                "evaluation_dimensions": seed["evaluation_dimensions"],
                "average_score": 5,
                "reason": build_score5_reason(seed=seed, reason_lookup=reason_lookup),
                "revision_suggestions": "No revision is needed because the answer is already treated as the full-score reference.",
                "edit_intent": "No revision is needed because the answer is already treated as the full-score reference.",
                "modified_answer": seed["answer"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"[Target 5] output={output_path} total={len(seeds)} newly_written={written}")


async def worker(
    *,
    base_url: str,
    api_key: str,
    model: str,
    input_queue: asyncio.Queue[dict[str, Any] | None],
    writer_queue: asyncio.Queue[tuple[Path, dict[str, Any]] | None],
    max_tokens: int,
    temperature: float,
    retries: int,
    request_timeout: int,
    progress_bar: Any,
    http_session: "api_base.aiohttp.ClientSession | None",
    request_executor: ThreadPoolExecutor | None,
) -> None:
    while True:
        task = await input_queue.get()
        if task is None:
            input_queue.task_done()
            break

        record = task["record"]
        sample_id = task["sample_id"]
        target_score = task["target_score"]
        output_path = task["output_path"]
        messages = build_messages(record, target_score)
        parsed, raw_text = await api_base.call_model(
            base_url,
            model=model,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            timeout_seconds=request_timeout,
            http_session=http_session,
            request_executor=request_executor,
        )
        if parsed is not None:
            normalized_payload = normalize_generation_payload(
                parsed,
                target_score=target_score,
            )
            if normalized_payload is not None:
                output_row = {
                    "sample_id": sample_id,
                    "question": record.get("question"),
                    "answer": normalized_payload["generated_answer"],
                    "evaluation_dimensions": record.get("evaluation_dimensions"),
                    "average_score": normalized_payload["average_score"],
                    "reason": normalized_payload["reason"],
                    "revision_suggestions": normalized_payload["revision_suggestions"],
                    "edit_intent": normalized_payload["revision_suggestions"],
                    "modified_answer": normalized_payload["modified_answer"],
                    "source_answer": record.get("answer"),
                    # "raw_response": raw_text,
                }
                await writer_queue.put((output_path, output_row))
            else:
                preview = " ".join(str(raw_text).split())[:240]
                print(
                    f"[INVALID] target={target_score} sample_id={sample_id} "
                    f"endpoint={base_url} response={preview}"
                )
        else:
            preview = " ".join(str(raw_text).split())[:240]
            print(
                f"[FAILED] target={target_score} sample_id={sample_id} "
                f"endpoint={base_url} response={preview}"
            )

        input_queue.task_done()
        if progress_bar is not None:
            progress_bar.update(1)


async def run_generation(
    *,
    base_urls: list[str],
    api_key: str,
    model: str,
    tasks: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    retries: int,
    concurrency_per_endpoint: int,
    request_timeout: int,
    dispatch_poll_interval: float,
) -> None:
    if not tasks:
        print("No pending tasks. Nothing to generate.")
        return

    progress_bar = None
    if api_base.tqdm is not None:
        progress_bar = api_base.tqdm(
            total=len(tasks),
            desc="Generating",
            unit="task",
            dynamic_ncols=True,
        )

    queue_capacity = max(1, concurrency_per_endpoint)
    endpoint_queues = [asyncio.Queue(maxsize=queue_capacity) for _ in base_urls]
    writer_queue: asyncio.Queue[tuple[Path, dict[str, Any]] | None] = asyncio.Queue()
    writer_task = asyncio.create_task(api_base.multi_writer(writer_queue))
    dispatcher_task = asyncio.create_task(
        api_base.dispatch_tasks_round_robin(
            tasks,
            endpoint_queues,
            poll_interval=dispatch_poll_interval,
        )
    )

    workers: list[asyncio.Task[Any]] = []
    worker_count = max(1, concurrency_per_endpoint)
    total_worker_count = max(1, len(base_urls) * worker_count)

    async def launch_workers(
        *,
        http_session: "api_base.aiohttp.ClientSession | None",
        request_executor: ThreadPoolExecutor | None,
    ) -> None:
        for base_url, input_queue in zip(base_urls, endpoint_queues):
            for _ in range(worker_count):
                workers.append(
                    asyncio.create_task(
                        worker(
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            input_queue=input_queue,
                            writer_queue=writer_queue,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            retries=retries,
                            request_timeout=request_timeout,
                            progress_bar=progress_bar,
                            http_session=http_session,
                            request_executor=request_executor,
                        )
                    )
                )

        await dispatcher_task
        await asyncio.gather(*workers)

    try:
        if api_base.aiohttp is not None:
            connector = api_base.aiohttp.TCPConnector(limit=max(64, total_worker_count * 2))
            timeout = api_base.aiohttp.ClientTimeout(total=request_timeout)
            async with api_base.aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
                await launch_workers(http_session=http_session, request_executor=None)
        else:
            with ThreadPoolExecutor(max_workers=max(32, total_worker_count)) as request_executor:
                await launch_workers(http_session=None, request_executor=request_executor)
    finally:
        await writer_queue.put(None)
        await writer_queue.join()
        await writer_task
        if progress_bar is not None:
            progress_bar.close()


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    seeds = group_full_score_seeds(input_path)
    if args.limit is not None:
        seeds = seeds[: max(0, args.limit)]
    print(f"Loaded grouped full-score seeds: {len(seeds)}")

    reason_lookup: score05.FullScoreReasonLookup | None = None
    full_score_reason_root = Path(args.full_score_reason_root)
    if args.write_score_5 and full_score_reason_root.is_dir():
        reason_lookup = score05.FullScoreReasonLookup(full_score_reason_root)
        print(
            f"[INFO] full-score reason lookup loaded | files={reason_lookup.files_loaded}, "
            f"rows={reason_lookup.rows_loaded}, reasons={reason_lookup.reasons_loaded}"
        )

    if args.write_score_5:
        write_score5_rows(
            seeds=seeds,
            output_path=output_dir / args.score_5_output_name,
            reason_lookup=reason_lookup,
        )

    ports = api_base.parse_ports(args.ports)
    base_urls = api_base.build_base_urls(args.base_url_template, ports)
    if not args.skip_health_check:
        await api_base.wait_for_servers(base_urls, timeout_seconds=args.health_check_timeout)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    tasks = build_generation_tasks(
        seeds=seeds,
        targets=args.targets,
        output_dir=output_dir,
    )
    await run_generation(
        base_urls=base_urls,
        api_key=api_key,
        model=args.model,
        tasks=tasks,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        concurrency_per_endpoint=args.concurrency_per_endpoint,
        request_timeout=args.request_timeout,
        dispatch_poll_interval=args.dispatch_poll_interval,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-dimension one-shot 0-5 raw answer variants from grouped "
            "full-score seeds via multiple OpenAI-compatible endpoints."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Grouped seed source json/jsonl path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store score_0.jsonl ... score_5.jsonl",
    )
    parser.add_argument(
        "--base-url-template",
        default=DEFAULT_BASE_URL_TEMPLATE,
        help="Base URL template, e.g. http://localhost:{port}/v1",
    )
    parser.add_argument(
        "--ports",
        nargs="*",
        default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"],
        help="Port list or ranges, e.g. --ports 8000-8003 or --ports 8000 8001 8002 8003",
    )
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by the API.")
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_SCORES),
        help="Target scores to generate, e.g. --targets 0 1 2 3 4",
    )
    parser.add_argument(
        "--write-score-5",
        action="store_true",
        help="Also materialize grouped full-score rows to score_5.jsonl.",
    )
    parser.add_argument(
        "--score-5-output-name",
        default=DEFAULT_SCORE_5_OUTPUT,
        help="Output filename used when --write-score-5 is enabled.",
    )
    parser.add_argument(
        "--full-score-reason-root",
        default=str(FULL_SCORE_REASON_ROOT),
        help="Root folder used to backfill integrated reasons for score_5 rows.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per sample.")
    parser.add_argument(
        "--concurrency-per-endpoint",
        type=int,
        default=16,
        help="Worker count per endpoint.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N grouped seeds.")
    parser.add_argument(
        "--dispatch-poll-interval",
        type=float,
        default=DEFAULT_DISPATCH_POLL_INTERVAL,
        help="Polling interval in seconds when all endpoint queues are temporarily full.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip polling /models before generation.",
    )
    parser.add_argument(
        "--health-check-timeout",
        type=int,
        default=10,
        help="HTTP timeout in seconds for endpoint health checks.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for each generation request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
