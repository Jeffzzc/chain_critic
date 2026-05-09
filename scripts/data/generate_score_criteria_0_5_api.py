#!/usr/bin/env python
"""Generate per-dimension 0-5 score criteria from full-score seeds via multi-endpoint API."""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from pathlib import Path
from typing import Any

import generate_score_variants_api as api_base


DEFAULT_INPUT = Path("datasets/full_score_dimension_seeds.jsonl")
DEFAULT_OUTPUT = Path("datasets/0-5/score_criteria_0_5.jsonl")
DEFAULT_PORTS = tuple(range(8000, 8004))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_MODEL = "Qwen3.5-27B"
DEFAULT_DISPATCH_POLL_INTERVAL = 0.05
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 640
SCORE_RANGE = tuple(range(6))


SYSTEM_PROMPT = (
    "You are a rubric generation assistant for answer evaluation training.\n\n"
    "You will receive:\n"
    "- a question,\n"
    "- a high-quality reference answer,\n"
    "- one evaluation dimension,\n"
    "- and the full-score criteria for score 5 on that dimension.\n\n"
    "Your task is to generate a complete 0-5 scoring rubric for that SAME dimension.\n\n"
    "Hard constraints:\n"
    "1. Evaluate only the specified dimension. Do not mix in other dimensions.\n"
    "2. Each criterion must describe observable properties of the answer itself.\n"
    "3. The six score levels must form a clear monotonic progression from 0 to 5.\n"
    "4. Adjacent score levels must be meaningfully distinguishable in realistic grading.\n"
    "5. Score 5 must preserve the meaning of the provided full-score criteria.\n"
    "6. Score 0 should represent near-complete failure on the specified dimension.\n"
    "7. Avoid meta commentary, rubric-writing commentary, JSON commentary, or process commentary.\n"
    "8. Keep each criterion concise, concrete, and directly usable for downstream answer generation and evaluation.\n"
    "9. Return strict JSON only.\n"
)


REPAIR_SYSTEM_PROMPT = (
    "You are a rubric repair assistant.\n\n"
    "You will receive a previously generated 0-5 rubric that failed validation.\n"
    "Your job is to rewrite it so that:\n"
    "1. it evaluates ONLY the specified dimension,\n"
    "2. the six score levels form a clean monotonic progression,\n"
    "3. adjacent levels are distinguishable,\n"
    "4. score 5 preserves the meaning of the provided full-score criteria,\n"
    "5. the criteria are concise, concrete, and self-contained,\n"
    "6. there is no meta commentary.\n\n"
    "Return strict JSON only."
)


DIMENSION_POLICIES: dict[str, dict[str, Any]] = {
    "factual correctness": {
        "focus": (
            "Focus only on whether the stated facts, quantities, values, and claims in the answer "
            "match the information in the question."
        ),
        "avoid": (
            "Do not grade answer completeness, reasoning completeness, internal coherence, "
            "language fluency, expression naturalness, comprehensibility, or output formatting."
        ),
        "forbidden_keywords": [
            "boxed",
            "unit",
            "grammar",
            "spelling",
            "natural",
            "conversational",
            "robotic",
            "coherent",
            "readable",
        ],
    },
    "answer completeness": {
        "focus": (
            "Focus only on whether the answer covers all required parts or components requested by the question."
        ),
        "avoid": (
            "Do not grade factual correctness, calculation correctness, internal coherence, "
            "reasoning quality, language fluency, expression naturalness, comprehensibility, or formatting."
        ),
        "forbidden_keywords": [
            "correct",
            "incorrect",
            "accurate",
            "inaccurate",
            "calculation error",
            "factual",
            "grammar",
            "spelling",
            "natural",
            "conversational",
            "robotic",
            "boxed",
            "unit",
        ],
    },
    "reasoning chain completeness": {
        "focus": (
            "Focus only on whether the answer includes the full chain of reasoning steps needed to connect "
            "the given information to the final result."
        ),
        "avoid": (
            "Do not grade factual correctness, final-answer formatting, language fluency, "
            "expression naturalness, or general readability."
        ),
        "forbidden_keywords": [
            "boxed",
            "unit",
            "grammar",
            "spelling",
            "natural",
            "conversational",
            "robotic",
        ],
    },
    "internal coherence": {
        "focus": (
            "Focus only on whether each step logically supports the next without contradiction, disconnect, or missing links."
        ),
        "avoid": (
            "Do not grade completeness, correctness, language fluency, expression naturalness, "
            "comprehensibility, or formatting."
        ),
        "forbidden_keywords": [
            "boxed",
            "unit",
            "grammar",
            "spelling",
            "natural",
            "conversational",
            "robotic",
        ],
    },
    "language fluency": {
        "focus": (
            "Focus only on grammar, spelling, syntax, and how smooth and natural the English is at the sentence level."
        ),
        "avoid": (
            "Do not grade factual correctness, answer completeness, reasoning completeness, "
            "internal coherence, formatting, or final-answer correctness."
        ),
        "forbidden_keywords": [
            "boxed",
            "unit",
            "correct final answer",
            "incorrect final answer",
            "complete answer",
            "all required parts",
        ],
    },
    "expression naturalness": {
        "focus": (
            "Focus only on whether the explanation sounds human, intuitive, and non-robotic in phrasing and flow."
        ),
        "avoid": (
            "Do not grade factual correctness, completeness, step correctness, grammar/spelling as primary criteria, "
            "or formatting requirements."
        ),
        "forbidden_keywords": [
            "boxed",
            "unit",
            "correct final answer",
            "incorrect final answer",
            "grammar",
            "spelling",
            "syntax",
        ],
    },
    "comprehensibility": {
        "focus": (
            "Focus only on whether a reader with basic background knowledge can understand the explanation and follow the steps."
        ),
        "avoid": (
            "Do not grade factual correctness, completeness, expression naturalness, fluency alone, or formatting."
        ),
        "forbidden_keywords": [
            "boxed",
            "unit",
            "correct final answer",
            "incorrect final answer",
            "natural human speech",
            "robotic",
        ],
    },
}


META_PATTERNS = [
    r"\bjson\b",
    r"\brubric\b",
    r"\bmeta\b",
    r"\bcriterion\s+for\s+score\b",
    r"\bscore\s*[0-5]\s*[:：]",
    r"\bthis criterion\b",
    r"\bthe rubric\b",
    r"\bdesign process\b",
]


def normalize_dimension_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def get_dimension_policy(dimension_name: str) -> dict[str, Any] | None:
    name = normalize_dimension_name(dimension_name)
    if name in DIMENSION_POLICIES:
        return DIMENSION_POLICIES[name]

    for key, policy in DIMENSION_POLICIES.items():
        if key in name or name in key:
            return policy
    return None


def build_dimension_guidance(record: dict[str, Any]) -> str:
    dimension_name = str(record.get("dimension_name", "")).strip()
    policy = get_dimension_policy(dimension_name)
    if not policy:
        return (
            "Dimension-specific guidance:\n"
            "- Keep the rubric tightly focused on the named dimension only.\n"
            "- Do not mix in correctness, completeness, coherence, fluency, naturalness, comprehensibility, or formatting unless they are the named dimension.\n"
        )

    return (
        "Dimension-specific guidance:\n"
        f"- Focus: {policy['focus']}\n"
        f"- Avoid: {policy['avoid']}\n"
    )


USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "High-Quality Reference Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Full-Score Criteria For Score 5:\n{full_score_criteria}\n\n"
    "{dimension_guidance}\n"
    "Requirements:\n"
    "1. Generate score criteria for scores 0, 1, 2, 3, 4, and 5 for this dimension.\n"
    "2. Preserve the semantics of the provided score-5 criteria.\n"
    "3. Keep the rubric self-contained and directly usable for evaluating answers to this question.\n"
    "4. Do not mention JSON, rubric-writing, prompt-writing, or any meta process inside the criteria.\n"
    "5. Do not define a score level by referencing another score level.\n"
    "6. Keep each criterion concise and concrete.\n"
    "7. Return JSON only with this schema:\n"
    "{{\"score_criteria\": {{\"0\": \"...\", \"1\": \"...\", \"2\": \"...\", \"3\": \"...\", \"4\": \"...\", \"5\": \"...\"}}}}"
)


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    user_content = USER_TEMPLATE.format(
        question=str(record.get("question", "")).strip(),
        answer=str(record.get("answer", "")).strip(),
        dimension_name=str(record.get("dimension_name", "")).strip(),
        full_score_criteria=str(record.get("full_score_criteria", "")).strip(),
        dimension_guidance=build_dimension_guidance(record),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_repair_messages(
    *,
    record: dict[str, Any],
    parsed: dict[str, Any],
    failure_reasons: list[str],
) -> list[dict[str, str]]:
    repair_user = (
        f"Question:\n{str(record.get('question', '')).strip()}\n\n"
        f"High-Quality Reference Answer:\n{str(record.get('answer', '')).strip()}\n\n"
        f"Evaluation Dimension:\n{str(record.get('dimension_name', '')).strip()}\n\n"
        f"Full-Score Criteria For Score 5:\n{str(record.get('full_score_criteria', '')).strip()}\n\n"
        f"{build_dimension_guidance(record)}\n"
        "The previous rubric failed validation for these reasons:\n"
        f"{json.dumps(failure_reasons, ensure_ascii=False, indent=2)}\n\n"
        "Previous invalid rubric JSON:\n"
        f"{json.dumps(parsed, ensure_ascii=False, indent=2)}\n\n"
        "Rewrite it into valid strict JSON with schema:\n"
        "{\"score_criteria\": {\"0\": \"...\", \"1\": \"...\", \"2\": \"...\", \"3\": \"...\", \"4\": \"...\", \"5\": \"...\"}}"
    )
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": repair_user},
    ]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _cleanup_criterion_text(text: str) -> str:
    text = _normalize_text(text)
    text = re.sub(r'^\s*["“”]+|["“”]+\s*$', "", text)
    text = re.sub(r"^\s*(score\s*)?[0-5]\s*[:：-]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_score_criteria_map(parsed: dict[str, Any]) -> dict[str, Any]:
    candidate = api_base.get_payload_value(
        parsed,
        "score_criteria",
        "criteria_by_score",
        "criteria_map",
        "rubric",
        default={},
    )
    if isinstance(candidate, dict):
        return candidate
    return {}


def normalize_generation_payload(
    parsed: dict[str, Any],
    *,
    record: dict[str, Any],
) -> dict[str, Any] | None:
    criteria_map = _extract_score_criteria_map(parsed)
    normalized: dict[str, str] = {}

    for score in SCORE_RANGE:
        value = criteria_map.get(str(score))
        if value is None:
            value = criteria_map.get(score)
        if value is None:
            value = parsed.get(f"criteria_{score}")
        if value is None:
            value = parsed.get(f"score_{score}")

        # Keep score 5 anchored to the seed whenever possible.
        if score == 5 and _normalize_text(record.get("full_score_criteria")):
            value = record.get("full_score_criteria")

        text = _cleanup_criterion_text(str(value or ""))
        if not text:
            return None
        normalized[str(score)] = text

    return {
        "score_criteria": normalized,
        **{f"criteria_{score}": normalized[str(score)] for score in SCORE_RANGE},
    }


def detect_seed_scope_issues(record: dict[str, Any]) -> list[str]:
    dimension_name = str(record.get("dimension_name", "")).strip()
    full_score = str(record.get("full_score_criteria", "")).strip().lower()
    policy = get_dimension_policy(dimension_name)
    if not policy or not full_score:
        return []

    hits: list[str] = []
    for kw in policy.get("forbidden_keywords", []):
        if kw.lower() in full_score:
            hits.append(kw)
    return sorted(set(hits))


def validate_generation_payload(
    normalized_payload: dict[str, Any],
    *,
    record: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    score_criteria = normalized_payload.get("score_criteria")
    if not isinstance(score_criteria, dict):
        return False, ["missing score_criteria map"]

    # Basic presence / content checks
    values: dict[str, str] = {}
    for score in SCORE_RANGE:
        key = str(score)
        text = _normalize_text(score_criteria.get(key))
        if not text:
            reasons.append(f"missing text for score {score}")
            continue
        if len(text) < 12:
            reasons.append(f"criterion too short for score {score}")
        if len(text) > 320:
            reasons.append(f"criterion too long for score {score}")
        values[key] = text

    if reasons:
        return False, reasons

    # Distinctness checks
    unique_values = {re.sub(r"\s+", " ", v.strip().lower()) for v in values.values()}
    if len(unique_values) < 5:
        reasons.append("too many duplicate or near-duplicate score criteria")

    for left, right in zip(SCORE_RANGE[:-1], SCORE_RANGE[1:]):
        if values[str(left)].strip().lower() == values[str(right)].strip().lower():
            reasons.append(f"scores {left} and {right} are identical")

    if values["0"].strip().lower() == values["5"].strip().lower():
        reasons.append("score 0 and score 5 are identical")

    # Meta commentary checks
    for score, text in values.items():
        lower_text = text.lower()
        for pattern in META_PATTERNS:
            if re.search(pattern, lower_text, flags=re.I):
                reasons.append(f"meta wording detected in score {score}")
                break

    # Dimension leakage checks (only strongly enforce on 0-4; score 5 is anchored to seed)
    dimension_name = str(record.get("dimension_name", "")).strip()
    policy = get_dimension_policy(dimension_name)
    seed_score_5 = _normalize_text(record.get("full_score_criteria")).lower()

    if policy is not None:
        forbidden_keywords = policy.get("forbidden_keywords", [])
        for score in ["0", "1", "2", "3", "4"]:
            text = values[score].lower()
            for kw in forbidden_keywords:
                kw_l = kw.lower()
                if kw_l in text and kw_l not in seed_score_5:
                    reasons.append(
                        f"dimension leakage in score {score}: contains forbidden keyword '{kw}'"
                    )
                    break

    return len(reasons) == 0, reasons


async def maybe_repair_payload(
    *,
    base_url: str,
    api_key: str,
    model: str,
    parsed: dict[str, Any],
    record: dict[str, Any],
    max_tokens: int,
    temperature: float,
    retries: int,
    request_timeout: int,
    http_session: "api_base.aiohttp.ClientSession | None",
    request_executor: ThreadPoolExecutor | None,
    repair_attempts: int,
) -> tuple[dict[str, Any] | None, str]:
    normalized = normalize_generation_payload(parsed, record=record)
    if normalized is None:
        failure_reasons = ["normalization failed: missing one or more score criteria"]
    else:
        is_valid, failure_reasons = validate_generation_payload(normalized, record=record)
        if is_valid:
            return normalized, "ok"

    last_parsed = parsed
    for attempt in range(repair_attempts):
        repair_messages = build_repair_messages(
            record=record,
            parsed=last_parsed,
            failure_reasons=failure_reasons,
        )
        repaired_parsed, _ = await api_base.call_model(
            base_url,
            model=model,
            api_key=api_key,
            messages=repair_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
            timeout_seconds=request_timeout,
            http_session=http_session,
            request_executor=request_executor,
        )
        if repaired_parsed is None:
            continue

        repaired_normalized = normalize_generation_payload(repaired_parsed, record=record)
        if repaired_normalized is None:
            last_parsed = repaired_parsed
            failure_reasons = ["repair normalization failed"]
            continue

        is_valid, failure_reasons = validate_generation_payload(repaired_normalized, record=record)
        if is_valid:
            return repaired_normalized, f"repaired_{attempt + 1}"

        last_parsed = repaired_parsed

    return None, "failed_validation"


def build_generation_tasks(
    *,
    records: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    completed_ids = api_base.load_completed_ids(output_path)
    tasks: list[dict[str, Any]] = []
    suspicious_seed_count = 0

    for record in records:
        sample_id = api_base.build_sample_id(record)
        if sample_id in completed_ids:
            continue

        scope_hits = detect_seed_scope_issues(record)
        if scope_hits:
            suspicious_seed_count += 1

        tasks.append(
            {
                "record": record,
                "sample_id": sample_id,
                "output_path": output_path,
                "seed_scope_hits": scope_hits,
            }
        )

    print(
        f"[Rubric Generation] output={output_path} total={len(records)} "
        f"completed={len(completed_ids)} remaining={len(tasks)} "
        f"suspicious_seed_5={suspicious_seed_count}"
    )
    return tasks


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
    repair_attempts: int,
    verbose_failures: bool,
) -> None:
    while True:
        try:
            task = await input_queue.get()
        except asyncio.CancelledError:
            break

        try:
            if task is None:
                break

            record = task["record"]
            sample_id = task["sample_id"]
            output_path = task["output_path"]
            seed_scope_hits = task.get("seed_scope_hits", [])
            messages = build_messages(record)

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

            if parsed is None:
                if verbose_failures:
                    print(f"[WARN] parse failed sample_id={sample_id}")
                continue

            normalized, generation_status = await maybe_repair_payload(
                base_url=base_url,
                api_key=api_key,
                model=model,
                parsed=parsed,
                record=record,
                max_tokens=max_tokens,
                temperature=temperature,
                retries=retries,
                request_timeout=request_timeout,
                http_session=http_session,
                request_executor=request_executor,
                repair_attempts=repair_attempts,
            )

            if normalized is None:
                if verbose_failures:
                    print(
                        f"[WARN] validation failed sample_id={sample_id} "
                        f"dimension={record.get('dimension_name')} "
                        f"seed_scope_hits={seed_scope_hits}"
                    )
                continue

            output_row = {
                "sample_id": sample_id,
                "question": record.get("question"),
                "answer": record.get("answer"),
                "dimension_name": record.get("dimension_name"),
                "full_score_criteria": record.get("full_score_criteria"),
                **normalized,
                "generation_status": generation_status,
            }

            if seed_scope_hits:
                output_row["seed_scope_warning"] = seed_scope_hits

            # Uncomment if you want debugging fields in output.
            # output_row["raw_response"] = raw_text

            await writer_queue.put((output_path, output_row))

        except Exception as exc:
            if verbose_failures:
                print(f"[ERROR] worker failed: {exc}")
        finally:
            input_queue.task_done()
            if task is not None and progress_bar is not None:
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
    repair_attempts: int,
    verbose_failures: bool,
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
    else:
        print("tqdm is not installed; progress bar disabled.")

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
    workers = []
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
                            repair_attempts=repair_attempts,
                            verbose_failures=verbose_failures,
                        )
                    )
                )

        await dispatcher_task
        await asyncio.gather(*workers)

    try:
        if api_base.aiohttp is not None:
            connector = api_base.aiohttp.TCPConnector(limit=max(64, total_worker_count * 2))
            timeout = api_base.aiohttp.ClientTimeout(total=request_timeout)
            async with api_base.aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            ) as http_session:
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
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = api_base.load_records(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    ports = api_base.parse_ports(args.ports)
    base_urls = api_base.build_base_urls(args.base_url_template, ports)
    if not args.skip_health_check:
        await api_base.wait_for_servers(base_urls, timeout_seconds=args.health_check_timeout)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    tasks = build_generation_tasks(records=records, output_path=output_path)
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
        repair_attempts=args.repair_attempts,
        verbose_failures=args.verbose_failures,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-dimension score criteria for scores 0 through 5 from "
            "full-score seed data via multiple OpenAI-compatible endpoints."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Seed json/jsonl path.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSONL path to store generated 0-5 score criteria.",
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
        help="Port list or ranges, e.g. --ports 8000-8007 or --ports 8000 8001 8002",
    )
    parser.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by the API.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max generation tokens.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per sample.")
    parser.add_argument(
        "--concurrency-per-endpoint",
        type=int,
        default=32,
        help="Worker count per endpoint. Use 1 to keep one active generator per GPU.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N seeds.")
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
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="How many automatic repair attempts to run after local validation fails.",
    )
    parser.add_argument(
        "--verbose-failures",
        action="store_true",
        help="Print validation / parse failures for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()