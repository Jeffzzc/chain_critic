#!/usr/bin/env python
"""Clean score/reason/revision SFT rows with local vLLM endpoints.

Default target:
datasets/train/final_score_reason_revision_sft/final_score_reason_revision_sft_train.jsonl

The script keeps already-valid rows, sends suspicious rows to local
OpenAI-compatible vLLM endpoints, and writes a cleaned SFT JSONL where the
assistant content is strict JSON with exactly:
score, reason, revision_suggestions, modified_answer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any

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
    iter_jsonl,
    models_endpoint_ready,
    normalize_text,
    parse_int_score,
    parse_ports,
    resolve_model,
    safe_unlink,
    wait_for_servers,
    write_jsonl,
)
from regenerate_sft_score_reason_revision_local import sample_from_record  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DEFAULT_INPUT = Path("datasets/train/final_score_reason_revision_sft/final_score_reason_revision_sft_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/train/final_score_reason_revision_sft_cleaned")

SYSTEM_PROMPT = (
    "You are a strict SFT label cleaner for answer evaluation and revision data. "
    "Your job is to repair only the assistant label for one training sample. "
    "Use only the provided question, candidate answer, evaluation dimension, scoring criteria, "
    "and current label. Do not add unsupported facts. Return strict JSON only."
)

CLEAN_USER_TEMPLATE = (
    "Clean the current assistant label for this SFT sample.\n\n"
    "Required output schema:\n"
    "{{\"score\": <int 0-5>, \"reason\": \"...\", \"revision_suggestions\": \"...\", "
    "\"modified_answer\": \"...\"}}\n\n"
    "Hard requirements:\n"
    "1. Keep the score unchanged if the current score is a valid integer from 0 to 5. "
    "Only infer a score from the criteria if the current score is missing or invalid.\n"
    "2. The reason must be concrete and grounded in the evaluation dimension and criteria. "
    "For scores 0-4, it must include at least one specific defect marked with the exact prefix "
    "\"error:\". The error must name the concrete wrong step, missing element, contradiction, "
    "unsupported claim, or rubric failure in the candidate answer.\n"
    "3. The revision_suggestions must be actionable edit instructions derived from the reason. "
    "They must say what to remove, add, correct, rewrite, reorder, clarify, or verify. "
    "Do not write vague advice such as 'make it better', 'improve clarity', or "
    "'be more accurate' unless it is paired with a concrete edit.\n"
    "4. For subjective dimensions such as coherence, clarity, relevance, completeness, style, "
    "tone, persuasiveness, or helpfulness, the revision_suggestions must still be executable "
    "editing actions tied to concrete text-level problems, not broad preferences.\n"
    "5. The modified_answer must follow the revision_suggestions, answer the original question, "
    "optimize only for the same evaluation dimension, and avoid unsupported facts.\n"
    "6. Do not include markdown, code fences, comments, extra keys, or explanatory text outside JSON.\n\n"
    "Detected issues in the current label:\n{issues}\n\n"
    "Question:\n{question}\n\n"
    "Candidate Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Score Criteria (0-5):\n{criteria_text}\n\n"
    "Current assistant label:\n{current_label}"
)

ACTION_VERBS = (
    "add",
    "remove",
    "delete",
    "replace",
    "rewrite",
    "revise",
    "correct",
    "fix",
    "change",
    "clarify",
    "specify",
    "include",
    "state",
    "explain",
    "reorder",
    "separate",
    "combine",
    "calculate",
    "recalculate",
    "verify",
    "align",
    "avoid",
    "drop",
    "补充",
    "删除",
    "移除",
    "替换",
    "改写",
    "修正",
    "纠正",
    "明确",
    "说明",
    "重排",
    "计算",
    "重新计算",
)

VAGUE_REVISION_PATTERNS = (
    r"\bmake (?:it|the answer) better\b",
    r"\bimprove (?:the )?(?:answer|response|quality|clarity|coherence)\b",
    r"\bbe more (?:clear|accurate|specific|coherent|relevant)\b",
    r"\bneeds? improvement\b",
    r"\bshould be improved\b",
)

SUBJECTIVE_DIMENSION_HINTS = (
    "coherence",
    "consistency",
    "clarity",
    "relevance",
    "completeness",
    "helpfulness",
    "usefulness",
    "tone",
    "style",
    "fluency",
    "persuasiveness",
    "specificity",
    "depth",
    "organization",
    "conciseness",
    "readability",
    "主观",
    "清晰",
    "相关",
    "完整",
    "有用",
    "语气",
    "风格",
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def assistant_content(record: dict[str, Any]) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""
    contents: list[str] = []
    for message in messages:
        if isinstance(message, dict) and normalize_text(message.get("role")).lower() == "assistant":
            contents.append(str(message.get("content") or ""))
    return contents[-1] if contents else ""


def parse_assistant_label(text: str) -> tuple[dict[str, Any], bool]:
    raw = str(text or "").strip()
    strict_json_ok = False
    parsed = None
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            parsed = loaded
            strict_json_ok = True
    except Exception:
        parsed = extract_json_object(raw)

    if not isinstance(parsed, dict):
        return {
            "score": None,
            "reason": "",
            "revision_suggestions": "",
            "modified_answer": "",
        }, False

    label = {
        "score": parse_int_score(parsed.get("score") or parsed.get("Score")),
        "reason": normalize_text(parsed.get("reason") or parsed.get("Reason")),
        "revision_suggestions": normalize_text(
            parsed.get("revision_suggestions")
            or parsed.get("Revision Suggestions")
            or parsed.get("edit_intent")
            or parsed.get("Edit Intent")
        ),
        "modified_answer": normalize_text(
            parsed.get("modified_answer")
            or parsed.get("Modified Answer")
            or parsed.get("revised_answer")
            or parsed.get("Revised Answer")
        ),
    }
    expected_keys = {"score", "reason", "revision_suggestions", "modified_answer"}
    strict_json_ok = strict_json_ok and set(parsed.keys()) == expected_keys
    return label, strict_json_ok


def label_to_json(label: dict[str, Any]) -> str:
    payload = {
        "score": int(label["score"]),
        "reason": normalize_text(label["reason"]),
        "revision_suggestions": normalize_text(label["revision_suggestions"]),
        "modified_answer": normalize_text(label["modified_answer"]),
    }
    return json.dumps(payload, ensure_ascii=False)


def is_subjective_dimension(dimension_name: str) -> bool:
    text = normalize_text(dimension_name).lower()
    return any(hint in text for hint in SUBJECTIVE_DIMENSION_HINTS)


def has_actionable_revision(text: str) -> bool:
    revision = normalize_text(text).lower()
    if len(revision) < 40:
        return False
    if any(re.search(pattern, revision) for pattern in VAGUE_REVISION_PATTERNS):
        has_concrete_action = any(verb in revision for verb in ACTION_VERBS)
        if not has_concrete_action:
            return False
    return any(verb in revision for verb in ACTION_VERBS)


def validate_label(
    label: dict[str, Any],
    *,
    strict_json_ok: bool,
    dimension_name: str,
    require_strict_json: bool,
    require_error_for_score5: bool,
) -> list[str]:
    issues: list[str] = []
    score = label.get("score")
    reason = normalize_text(label.get("reason"))
    revision = normalize_text(label.get("revision_suggestions"))
    modified = normalize_text(label.get("modified_answer"))

    if require_strict_json and not strict_json_ok:
        issues.append("assistant content is not strict JSON with exactly the required four keys")
    if score is None or not (0 <= int(score) <= 5):
        issues.append("score is missing or not an integer from 0 to 5")
    if not reason:
        issues.append("reason is empty")
    if not revision:
        issues.append("revision_suggestions is empty")
    if not modified:
        issues.append("modified_answer is empty")

    if reason and len(reason) < 50:
        issues.append("reason is too short to identify a concrete shortcoming")
    if score is not None and (int(score) < 5 or require_error_for_score5) and "error:" not in reason.lower():
        issues.append('reason does not mark a concrete defect with the prefix "error:"')
    if revision and not has_actionable_revision(revision):
        issues.append("revision_suggestions are vague or not actionable edit instructions")
    if revision and is_subjective_dimension(dimension_name) and not has_actionable_revision(revision):
        issues.append("subjective-dimension revision_suggestions are not executable text-level edits")
    return issues


def make_row_id(index: int, record: dict[str, Any]) -> str:
    messages = record.get("messages") if isinstance(record.get("messages"), list) else []
    user_text = ""
    assistant_text = assistant_content(record)
    for message in messages:
        if isinstance(message, dict) and normalize_text(message.get("role")).lower() == "user":
            user_text = str(message.get("content") or "")
            break
    digest = hashlib.sha1(
        json.dumps(
            {"index": index, "user": user_text, "assistant": assistant_text},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return f"row:{index}:{digest}"


def build_clean_messages(record: dict[str, Any], label: dict[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return [{"role": "assistant", "content": label_to_json(label)}]

    cleaned: list[dict[str, str]] = []
    skipped_final_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = str(message.get("content") or "")
        if role == "assistant":
            skipped_final_assistant = True
            continue
        cleaned.append({"role": role, "content": content})
    if not skipped_final_assistant and len(cleaned) == len(messages):
        cleaned = cleaned[:2]
    cleaned.append({"role": "assistant", "content": label_to_json(label)})
    return cleaned


def build_cleaning_messages(
    *,
    sample: dict[str, Any],
    current_label: dict[str, Any],
    issues: list[str],
) -> list[dict[str, str]]:
    current_label_json = json.dumps(
        {
            "score": current_label.get("score"),
            "reason": current_label.get("reason") or "",
            "revision_suggestions": current_label.get("revision_suggestions") or "",
            "modified_answer": current_label.get("modified_answer") or "",
        },
        ensure_ascii=False,
    )
    user_content = CLEAN_USER_TEMPLATE.format(
        issues="\n".join(f"- {issue}" for issue in issues) if issues else "- forced LLM cleaning requested",
        question=sample["question"],
        answer=sample["answer"],
        dimension_name=sample["dimension_name"],
        criteria_text=sample["criteria_text"],
        current_label=current_label_json,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_clean_output(text: str) -> dict[str, Any]:
    label, _ = parse_assistant_label(text)
    return label


def run_one(
    task: dict[str, Any],
    task_index: int,
    args: argparse.Namespace,
    base_urls: list[str],
    model: str,
) -> dict[str, Any]:
    started = time.time()
    result = {
        "row_id": task["row_id"],
        "source_index": task["source_index"],
        "ok": False,
        "issues_before": task["issues"],
        "issues_after": [],
        "label": {},
        "raw_output": "",
        "endpoint": "",
        "request_error": None,
        "latency_sec": None,
    }
    try:
        raw_text, endpoint = call_chat_with_retries(
            base_urls=base_urls,
            task_index=task_index,
            api_key=args.api_key,
            model=model,
            messages=task["messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        label = parse_clean_output(raw_text)
        strict_json_ok = extract_json_object(raw_text) is not None
        issues_after = validate_label(
            label,
            strict_json_ok=strict_json_ok,
            dimension_name=task["dimension_name"],
            require_strict_json=False,
            require_error_for_score5=args.require_error_for_score5,
        )
        result.update(
            {
                "ok": not issues_after,
                "issues_after": issues_after,
                "label": label,
                "raw_output": raw_text,
                "endpoint": endpoint,
                "latency_sec": round(time.time() - started, 4),
            }
        )
    except Exception as exc:
        result.update({"request_error": str(exc), "latency_sec": round(time.time() - started, 4)})
    return result


def filter_ready_base_urls(base_urls: list[str], *, timeout_seconds: int, workers: int) -> list[str]:
    ready_urls: list[str] = []
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


def resolve_runtime(args: argparse.Namespace) -> tuple[list[str], str]:
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
            raise RuntimeError("No ready vLLM endpoints found.")
    model = args.model.strip() or resolve_model("", base_urls, args.health_check_timeout)
    return base_urls, model


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    for row in iter_jsonl(path):
        row_id = normalize_text(row.get("row_id"))
        if row_id:
            completed[row_id] = row
    return completed


def analyze_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    stats = {
        "rows": 0,
        "valid_kept_without_llm": 0,
        "queued_for_llm": 0,
        "bad_sample": 0,
        "blank_or_invalid_label": 0,
        "non_actionable_revision": 0,
        "missing_error_prefix": 0,
    }

    input_path = resolve_path(args.input)
    for index, record in enumerate(iter_jsonl(input_path)):
        if args.limit is not None and index >= max(0, args.limit):
            break
        stats["rows"] += 1
        sample = sample_from_record(record, index, "train")
        row_id = make_row_id(index, record)
        if sample is None:
            stats["bad_sample"] += 1
            rows.append({"row_id": row_id, "source_index": index, "record": record, "sample": None, "label": None, "issues": ["cannot parse sample fields"]})
            continue

        label, strict_json_ok = parse_assistant_label(assistant_content(record))
        issues = validate_label(
            label,
            strict_json_ok=strict_json_ok,
            dimension_name=sample["dimension_name"],
            require_strict_json=args.require_strict_json,
            require_error_for_score5=args.require_error_for_score5,
        )
        if any("empty" in issue or "missing" in issue or "score" in issue for issue in issues):
            stats["blank_or_invalid_label"] += 1
        if any("actionable" in issue or "executable" in issue for issue in issues):
            stats["non_actionable_revision"] += 1
        if any("error:" in issue for issue in issues):
            stats["missing_error_prefix"] += 1

        row_info = {
            "row_id": row_id,
            "source_index": index,
            "record": record,
            "sample": sample,
            "label": label,
            "issues": issues,
        }
        rows.append(row_info)

        if args.llm_all or issues:
            tasks.append(
                {
                    "row_id": row_id,
                    "source_index": index,
                    "dimension_name": sample["dimension_name"],
                    "issues": issues,
                    "messages": build_cleaning_messages(sample=sample, current_label=label, issues=issues),
                }
            )
        else:
            stats["valid_kept_without_llm"] += 1

    stats["queued_for_llm"] = len(tasks)
    return rows, tasks, stats


def run(args: argparse.Namespace) -> None:
    args.input = resolve_path(args.input)
    args.output_dir = resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = resolve_path(args.output) if args.output else args.output_dir / args.output_name
    cache_path = args.output_dir / args.cache_name
    rejected_path = args.output_dir / args.rejected_name

    safe_unlink(output_path, args.overwrite)
    safe_unlink(cache_path, args.overwrite)
    safe_unlink(rejected_path, args.overwrite)

    rows, tasks, stats = analyze_rows(args)
    completed = load_cache(cache_path)
    pending = [task for task in tasks if task["row_id"] not in completed]

    print(f"[input] {args.input}")
    print(f"[output] {output_path}")
    print(f"[cache] {cache_path}")
    print(f"[rejected] {rejected_path}")
    print(
        f"[scan] rows={stats['rows']} queued_for_llm={stats['queued_for_llm']} "
        f"valid_kept_without_llm={stats['valid_kept_without_llm']} "
        f"bad_sample={stats['bad_sample']} blank_or_invalid_label={stats['blank_or_invalid_label']} "
        f"non_actionable_revision={stats['non_actionable_revision']} "
        f"missing_error_prefix={stats['missing_error_prefix']}"
    )

    if args.scan_only:
        print("[scan-only] no output files were written and no LLM requests were sent")
        return

    if pending:
        base_urls, model = resolve_runtime(args)
        print(f"[base_urls] {base_urls}")
        print(f"[model] {model}")

        lock = threading.Lock()
        progress = tqdm(total=len(pending), desc="clean_sft", ncols=100) if tqdm is not None else None
        try:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(run_one, task, task_index, args, base_urls, model): task
                    for task_index, task in enumerate(pending)
                }
                for future in as_completed(futures):
                    result = future.result()
                    completed[result["row_id"]] = result
                    with lock:
                        append_jsonl(cache_path, result)
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
                progress.close()
    else:
        print("[run] no pending LLM tasks")

    clean_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    fixed_count = 0
    kept_count = 0
    dropped_count = 0

    for row_info in rows:
        record = row_info["record"]
        sample = row_info["sample"]
        row_id = row_info["row_id"]
        original_issues = row_info["issues"]
        cached = completed.get(row_id)

        if cached and cached.get("ok") is True:
            clean_rows.append({"messages": build_clean_messages(record, cached["label"])})
            fixed_count += 1
            continue

        if sample is not None and not original_issues and not args.llm_all:
            clean_rows.append({"messages": build_clean_messages(record, row_info["label"])})
            kept_count += 1
            continue

        reject = {
            "row_id": row_id,
            "source_index": row_info["source_index"],
            "issues_before": original_issues,
            "clean_attempt": cached or {},
            "record": record,
        }
        rejected_rows.append(reject)
        if args.keep_unfixed and row_info["label"] and sample is not None:
            clean_rows.append({"messages": build_clean_messages(record, row_info["label"])})
            kept_count += 1
        else:
            dropped_count += 1

    write_jsonl(output_path, clean_rows)
    write_jsonl(rejected_path, rejected_rows)
    print(
        f"[done] clean_rows={len(clean_rows)} kept_without_llm={kept_count} "
        f"fixed_by_llm={fixed_count} rejected={len(rejected_rows)} dropped={dropped_count}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean score/reason/revision SFT train data with local vLLM services."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None, help="Optional exact output JSONL path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="final_score_reason_revision_sft_train_cleaned.jsonl")
    parser.add_argument("--cache-name", default="final_score_reason_revision_sft_train_clean_cache.jsonl")
    parser.add_argument("--rejected-name", default="final_score_reason_revision_sft_train_rejected.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only scan/process the first N rows.")
    parser.add_argument("--scan-only", action="store_true", help="Only report validation statistics; do not call LLM or write outputs.")
    parser.add_argument("--keep-unfixed", action="store_true", help="Keep rows that still fail validation after LLM cleaning.")
    parser.add_argument("--llm-all", action="store_true", help="Send every row to the LLM, not only suspicious rows.")
    parser.add_argument("--require-strict-json", action="store_true", default=True)
    parser.add_argument(
        "--allow-nonstrict-source-json",
        dest="require_strict_json",
        action="store_false",
        help="Do not queue rows solely because the source assistant content is non-strict JSON.",
    )
    parser.add_argument(
        "--require-error-for-score5",
        action="store_true",
        help='Also require "error:" in reason when score is 5. Default only requires it for scores 0-4.',
    )

    parser.add_argument("--base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--ports", nargs="*", default=["8001-8007"], help="Ports like: 8001-8007 or 8001 8002.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--workers", type=int, default=56)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument(
        "--wait-all-ports",
        action="store_true",
        help="Wait until every requested port is ready. By default, only ready ports are used.",
    )
    parser.add_argument("--probe-workers", type=int, default=32)
    parser.add_argument("--health-check-timeout", type=int, default=10)
    parser.add_argument("--health-check-interval", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
