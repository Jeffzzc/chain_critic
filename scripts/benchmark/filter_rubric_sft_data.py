#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter high-quality rubric-generation SFT data from a JSONL file.

Upgrades in this version:
1) Incremental persistence
   - Each finished sample is appended to kept.jsonl / dropped.jsonl / audit.jsonl immediately.
   - Files are flushed on every write, so processed results are not lost on interruption.
2) Resume support
   - On startup, existing audit.jsonl is loaded.
   - Samples whose sample_id already exists in audit.jsonl are skipped automatically.
3) Periodic summary snapshots
   - summary.json is refreshed during processing.
4) Safer write ordering
   - kept/dropped is written first, audit last.
   - audit.jsonl is treated as the source of truth for resume.

Target task:
    Input  : Question + Answer
    Output : One evaluation dimension + complete 0-5 scoring criteria
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import random
import re
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib import request as urllib_request

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =============================
# Default config
# =============================
DEFAULT_INPUT_PATH = Path("datasets/train_rubric/final_train.jsonl")
DEFAULT_OUTPUT_DIR = Path("datasets/train_rubric/rubric_sft_filtered")
DEFAULT_MODEL_NAME = "Qwen3.5-27B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORTS = [8000, 8001, 8002, 8003]
DEFAULT_WORKERS = 96
DEFAULT_NUM_JUDGES = 1
DEFAULT_MIN_OVERALL = 4.3
DEFAULT_MIN_DIM = 4.0
DEFAULT_SAVE_EVERY = 50


# -----------------------------
# Regex helpers
# -----------------------------
RUBRIC_DIM_RE = re.compile(
    r"^\s*Evaluation_dimension\s*:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

RUBRIC_SCORE_LINE_RE = re.compile(
    r"^\s*([0-5])\s*:\s*(.*?)\s*$",
    re.MULTILINE,
)

QUESTION_BLOCK_RE = re.compile(
    r"Question\s*:\s*(.*?)\n\s*Answer\s*:\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class ParsedRubric:
    dimension: str
    criteria: Dict[str, str]


@dataclass
class HardCheckResult:
    passed: bool
    issues: List[str]
    parsed_rubric: Optional[ParsedRubric]


@dataclass
class JudgeResult:
    format_score: float
    qa_relevance_score: float
    specificity_score: float
    consistency_score: float
    gradient_score: float
    overall_score: float
    decision: str
    reason: str
    raw_response: str


@dataclass
class AuditRecord:
    line_no: int
    sample_id: str
    hard_passed: bool
    hard_issues: List[str]
    extracted_question: str
    extracted_answer: str
    extracted_rubric: str
    judge: Optional[Dict[str, Any]]
    final_decision: str


# -----------------------------
# JSONL helpers
# -----------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                items.append(
                    {
                        "__broken_json__": True,
                        "__line_no__": line_no,
                        "__raw_line__": line,
                        "__error__": str(exc),
                    }
                )
                continue
            obj["__line_no__"] = line_no
            items.append(obj)
    return items


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {
                    "__broken_json__": True,
                    "__line_no__": line_no,
                    "__raw_line__": line,
                    "__error__": str(exc),
                }
                continue
            obj["__line_no__"] = line_no
            yield obj


class JsonlAppendWriter:
    def __init__(self, path: Path, mode: str = "a") -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = path.open(mode, encoding="utf-8")

    def write(self, row: Dict[str, Any]) -> None:
        self.fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()


# -----------------------------
# Sample extraction
# -----------------------------
def first_non_empty(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_from_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    user_texts = []
    assistant_texts = []

    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = msg.get("content")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        parts.append(text)
            content = "\n".join(parts)

        if not isinstance(content, str):
            continue

        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)

    question = None
    answer = None
    rubric = None

    if assistant_texts:
        rubric = assistant_texts[-1].strip()

    for text in reversed(user_texts):
        m = QUESTION_BLOCK_RE.search(text)
        if m:
            question = m.group(1).strip()
            answer = m.group(2).strip()
            break

    return question, answer, rubric


def extract_sample_fields(sample: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if sample.get("__broken_json__"):
        return None, None, None

    question = first_non_empty(
        sample.get("question"),
        sample.get("prompt"),
        sample.get("input_question"),
    )
    answer = first_non_empty(
        sample.get("answer"),
        sample.get("response"),
        sample.get("candidate_answer"),
        sample.get("input_answer"),
    )
    rubric = first_non_empty(
        sample.get("rubric"),
        sample.get("target"),
        sample.get("output"),
        sample.get("assistant"),
        sample.get("label"),
    )

    if (not question or not answer or not rubric) and isinstance(sample.get("messages"), list):
        mq, ma, mr = extract_from_messages(sample["messages"])
        question = question or mq
        answer = answer or ma
        rubric = rubric or mr

    if not rubric:
        dim = first_non_empty(sample.get("evaluation_dimension"), sample.get("dimension"))
        criteria = sample.get("criteria")
        if dim and isinstance(criteria, dict):
            ordered = []
            ok = True
            for i in range(6):
                key = str(i)
                if key not in criteria or not str(criteria[key]).strip():
                    ok = False
                    break
                ordered.append(f"{i}: {str(criteria[key]).strip()}")
            if ok:
                rubric = "Evaluation_dimension: " + dim + "\nCriteria (0-5):\n" + "\n".join(ordered)

    return question, answer, rubric


# -----------------------------
# Rubric parsing and hard checks
# -----------------------------
def parse_rubric_text(text: str) -> Optional[ParsedRubric]:
    if not text or not isinstance(text, str):
        return None

    dim_match = RUBRIC_DIM_RE.search(text)
    if not dim_match:
        return None
    dimension = dim_match.group(1).strip()
    if not dimension:
        return None

    score_lines = RUBRIC_SCORE_LINE_RE.findall(text)
    if not score_lines:
        return None

    criteria: Dict[str, str] = {}
    for score, criterion in score_lines:
        score = str(score)
        criterion = criterion.strip()
        if score in criteria:
            return None
        criteria[score] = criterion

    if set(criteria.keys()) != {str(i) for i in range(6)}:
        return None

    return ParsedRubric(dimension=dimension, criteria=criteria)


def hard_validate(question: Optional[str], answer: Optional[str], rubric: Optional[str]) -> HardCheckResult:
    issues: List[str] = []

    if not question or not question.strip():
        issues.append("missing_question")
    if not answer or not answer.strip():
        issues.append("missing_answer")
    if not rubric or not rubric.strip():
        issues.append("missing_rubric")

    if issues:
        return HardCheckResult(False, issues, None)

    parsed = parse_rubric_text(rubric)
    if parsed is None:
        issues.append("rubric_parse_failed")
        return HardCheckResult(False, issues, None)

    if len(parsed.dimension) < 3:
        issues.append("dimension_too_short")
    if len(parsed.dimension) > 80:
        issues.append("dimension_too_long")

    normalized_criteria = []
    for i in range(6):
        item = parsed.criteria[str(i)].strip()
        if len(item) < 8:
            issues.append(f"criterion_{i}_too_short")
        normalized_criteria.append(re.sub(r"\s+", " ", item.lower()))

    if len(set(normalized_criteria)) < 4:
        issues.append("criteria_overly_duplicated")

    dim_norm = re.sub(r"\s+", " ", parsed.dimension.lower())
    if dim_norm in set(normalized_criteria):
        issues.append("dimension_repeated_in_criteria")

    passed = len(issues) == 0
    return HardCheckResult(passed, issues, parsed)


# -----------------------------
# Endpoint pool
# -----------------------------
class EndpointPool:
    def __init__(self, ports: List[int], host: str = "127.0.0.1", path: str = "/v1/chat/completions") -> None:
        if not ports:
            raise ValueError("ports must not be empty")
        self.endpoints = [f"http://{host}:{port}{path}" for port in ports]
        self._lock = threading.Lock()
        self._idx = 0

    def next(self) -> str:
        with self._lock:
            endpoint = self.endpoints[self._idx]
            self._idx = (self._idx + 1) % len(self.endpoints)
            return endpoint


# -----------------------------
# LLM judge
# -----------------------------
JUDGE_SYSTEM_PROMPT = """You are a strict data-quality judge for an SFT dataset.
The dataset trains a model to do this task:
Given a question and an answer, generate exactly one evaluation dimension and its complete 0-5 scoring criteria.

Your job is to judge whether the candidate rubric is high-quality training data.
A high-quality rubric must satisfy ALL of the following:
1. The evaluation dimension is clearly defined and suitable for judging THIS specific answer.
2. The criteria 0-5 are complete, specific, and easy to distinguish.
3. The criteria levels increase consistently from very poor (0) to excellent (5).
4. The rubric is internally consistent and does not contradict itself.
5. The rubric is not overly generic; it should be grounded in the given QA pair.
6. The rubric should help supervise a model well, rather than just look formally correct.

Return ONLY valid JSON with these fields:
{
  "format_score": 0-5,
  "qa_relevance_score": 0-5,
  "specificity_score": 0-5,
  "consistency_score": 0-5,
  "gradient_score": 0-5,
  "overall_score": 0-5,
  "decision": "keep" | "review" | "drop",
  "reason": "one concise sentence"
}
Do not output markdown or any extra text."""


JUDGE_USER_TEMPLATE = """[Question]
{question}

[Answer]
{answer}

[Candidate Rubric]
{rubric}

Judge whether this rubric is suitable as HIGH-QUALITY SFT training data for the rubric-generation task.
Be strict. Prefer "drop" when the rubric is generic, weakly grounded, internally inconsistent, or poorly graded across 0-5.
"""


def build_payload(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def extract_text_from_chat_response(data: Dict[str, Any]) -> str:
    return data["choices"][0]["message"]["content"]


def parse_judge_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"failed to parse judge JSON: {text[:500]}")


def clamp_score(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(5.0, x))


def normalize_decision(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"keep", "review", "drop"}:
        return s
    return "drop"


def call_chat_completion(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def judge_once(
    endpoint_pool: EndpointPool,
    model: str,
    question: str,
    answer: str,
    rubric: str,
    retries: int = 3,
    temperature: float = 0.0,
) -> JudgeResult:
    last_error: Optional[Exception] = None
    user_prompt = JUDGE_USER_TEMPLATE.format(question=question, answer=answer, rubric=rubric)
    payload = build_payload(model=model, system_prompt=JUDGE_SYSTEM_PROMPT, user_prompt=user_prompt, temperature=temperature)

    for _ in range(retries):
        endpoint = endpoint_pool.next()
        try:
            response = call_chat_completion(endpoint, payload)
            text = extract_text_from_chat_response(response)
            data = parse_judge_json(text)
            return JudgeResult(
                format_score=clamp_score(data.get("format_score")),
                qa_relevance_score=clamp_score(data.get("qa_relevance_score")),
                specificity_score=clamp_score(data.get("specificity_score")),
                consistency_score=clamp_score(data.get("consistency_score")),
                gradient_score=clamp_score(data.get("gradient_score")),
                overall_score=clamp_score(data.get("overall_score")),
                decision=normalize_decision(data.get("decision")),
                reason=str(data.get("reason", "")).strip(),
                raw_response=text,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(0.8)

    raise RuntimeError(f"judge failed after retries: {last_error}")


def aggregate_judgements(results: List[JudgeResult]) -> JudgeResult:
    if not results:
        raise ValueError("no judge results")

    def avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 4)

    format_score = avg([r.format_score for r in results])
    qa_relevance_score = avg([r.qa_relevance_score for r in results])
    specificity_score = avg([r.specificity_score for r in results])
    consistency_score = avg([r.consistency_score for r in results])
    gradient_score = avg([r.gradient_score for r in results])
    overall_score = avg([r.overall_score for r in results])

    votes = [r.decision for r in results]
    keep_votes = votes.count("keep")
    review_votes = votes.count("review")
    drop_votes = votes.count("drop")

    if keep_votes >= max(review_votes, drop_votes):
        decision = "keep"
    elif drop_votes >= max(keep_votes, review_votes):
        decision = "drop"
    else:
        decision = "review"

    reason = " | ".join([r.reason for r in results if r.reason][:3]).strip()

    return JudgeResult(
        format_score=format_score,
        qa_relevance_score=qa_relevance_score,
        specificity_score=specificity_score,
        consistency_score=consistency_score,
        gradient_score=gradient_score,
        overall_score=overall_score,
        decision=decision,
        reason=reason,
        raw_response="\n\n---\n\n".join(r.raw_response for r in results),
    )


# -----------------------------
# Decision rule
# -----------------------------
def final_decision(hard: HardCheckResult, judge: Optional[JudgeResult], min_overall: float, min_dim: float) -> str:
    if not hard.passed:
        return "drop"
    if judge is None:
        return "drop"

    dim_scores = [
        judge.format_score,
        judge.qa_relevance_score,
        judge.specificity_score,
        judge.consistency_score,
        judge.gradient_score,
    ]

    if judge.overall_score < min_overall:
        return "drop"
    if min(dim_scores) < min_dim:
        return "drop"
    if judge.decision == "drop":
        return "drop"
    if judge.decision == "review":
        return "review"
    return "keep"


# -----------------------------
# Main worker
# -----------------------------
def make_sample_id(sample: Dict[str, Any], question: Optional[str], answer: Optional[str], rubric: Optional[str]) -> str:
    explicit = first_non_empty(sample.get("id"), sample.get("sample_id"), sample.get("task_id"))
    if explicit:
        return explicit

    raw = json.dumps(
        {
            "question": question or "",
            "answer": answer or "",
            "rubric": rubric or "",
            "line_no": sample.get("__line_no__", -1),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def process_one_sample(
    sample: Dict[str, Any],
    endpoint_pool: EndpointPool,
    model: str,
    num_judges: int,
    min_overall: float,
    min_dim: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], AuditRecord]:
    line_no = int(sample.get("__line_no__", -1))
    question, answer, rubric = extract_sample_fields(sample)
    sample_id = make_sample_id(sample, question, answer, rubric)

    hard = hard_validate(question, answer, rubric)

    judge_agg: Optional[JudgeResult] = None
    if hard.passed and question and answer and rubric:
        judge_results = []
        for _ in range(num_judges):
            judge_results.append(judge_once(endpoint_pool, model, question, answer, rubric))
        judge_agg = aggregate_judgements(judge_results)

    decision = final_decision(hard, judge_agg, min_overall=min_overall, min_dim=min_dim)

    audit = AuditRecord(
        line_no=line_no,
        sample_id=sample_id,
        hard_passed=hard.passed,
        hard_issues=hard.issues,
        extracted_question=question or "",
        extracted_answer=answer or "",
        extracted_rubric=rubric or "",
        judge=asdict(judge_agg) if judge_agg else None,
        final_decision=decision,
    )

    enriched = dict(sample)
    enriched["sample_id"] = sample_id
    if question is not None:
        enriched["question"] = question
    if answer is not None:
        enriched["answer"] = answer
    if rubric is not None:
        enriched["rubric"] = rubric

    if hard.parsed_rubric is not None:
        enriched["evaluation_dimension"] = hard.parsed_rubric.dimension
        enriched["criteria"] = {str(i): hard.parsed_rubric.criteria[str(i)] for i in range(6)}

    if judge_agg is not None:
        enriched["quality_judge"] = asdict(judge_agg)

    if decision == "keep":
        return enriched, None, audit
    return None, enriched, audit


# -----------------------------
# Resume helpers
# -----------------------------
def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def load_processed_state(audit_path: Path) -> Tuple[Set[str], Dict[str, int]]:
    processed_ids: Set[str] = set()
    stats = {
        "processed": 0,
        "kept": 0,
        "dropped": 0,
        "review": 0,
    }

    if not audit_path.exists():
        return processed_ids, stats

    for row in iter_jsonl(audit_path):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        if sample_id in processed_ids:
            continue
        processed_ids.add(sample_id)
        stats["processed"] += 1
        decision = str(row.get("final_decision", "")).strip().lower()
        if decision == "keep":
            stats["kept"] += 1
        elif decision == "review":
            stats["review"] += 1
            stats["dropped"] += 1
        else:
            stats["dropped"] += 1

    return processed_ids, stats


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    total_input: int,
    already_processed: int,
    pending_total: int,
    newly_done: int,
    kept_count: int,
    dropped_count: int,
    review_count: int,
    start_time: float,
    interrupted: bool = False,
) -> None:
    elapsed = time.time() - start_time
    processed_total = already_processed + newly_done
    summary = {
        "input_path": str(args.input),
        "output_dir": str(args.output_dir),
        "total_input": total_input,
        "already_processed_before_run": already_processed,
        "pending_total_this_run": pending_total,
        "processed_total": processed_total,
        "processed_this_run": newly_done,
        "remaining_estimated": max(total_input - processed_total, 0),
        "kept": kept_count,
        "dropped": dropped_count,
        "review": review_count,
        "keep_rate_over_processed": round(kept_count / processed_total, 6) if processed_total else 0.0,
        "model": args.model,
        "ports": args.ports,
        "workers": args.workers,
        "num_judges": args.num_judges,
        "min_overall": args.min_overall,
        "min_dim": args.min_dim,
        "resume": not args.overwrite,
        "elapsed_seconds_this_run": round(elapsed, 2),
        "samples_per_second_this_run": round(newly_done / elapsed, 4) if elapsed > 0 else 0.0,
        "interrupted": interrupted,
        "updated_at": int(time.time()),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter high-quality rubric SFT data with local vLLM judge endpoints.")

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input JSONL path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save filtered outputs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Served model name for the local vLLM endpoint")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="vLLM host")
    parser.add_argument("--ports", type=int, nargs="+", default=DEFAULT_PORTS, help="vLLM ports")

    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Thread workers")
    parser.add_argument("--num-judges", type=int, default=DEFAULT_NUM_JUDGES, help="How many independent judge calls per sample")
    parser.add_argument("--min-overall", type=float, default=DEFAULT_MIN_OVERALL, help="Minimum overall score to keep")
    parser.add_argument("--min-dim", type=float, default=DEFAULT_MIN_DIM, help="Minimum per-dimension score to keep")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="Refresh summary.json every N completed samples")

    parser.add_argument("--max-samples", type=int, default=0, help="If > 0, only process the first N input rows before resume filtering")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle pending samples before filtering")
    parser.add_argument("--overwrite", action="store_true", help="Ignore old outputs and start from scratch")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kept_path = args.output_dir / "kept.jsonl"
    dropped_path = args.output_dir / "dropped.jsonl"
    audit_path = args.output_dir / "audit.jsonl"
    summary_path = args.output_dir / "summary.json"

    if args.overwrite:
        remove_if_exists(kept_path)
        remove_if_exists(dropped_path)
        remove_if_exists(audit_path)
        remove_if_exists(summary_path)

    rows = load_jsonl(args.input)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    total_input = len(rows)

    processed_ids, old_stats = load_processed_state(audit_path)
    already_processed = old_stats["processed"]
    kept_count = old_stats["kept"]
    dropped_count = old_stats["dropped"]
    review_count = old_stats["review"]

    pending_rows: List[Dict[str, Any]] = []
    for row in rows:
        q, a, r = extract_sample_fields(row)
        sample_id = make_sample_id(row, q, a, r)
        if sample_id in processed_ids:
            continue
        pending_rows.append(row)

    if args.shuffle:
        random.shuffle(pending_rows)

    pending_total = len(pending_rows)
    start_time = time.time()

    endpoint_pool = EndpointPool(ports=args.ports, host=args.host)
    kept_writer = JsonlAppendWriter(kept_path, mode="a")
    dropped_writer = JsonlAppendWriter(dropped_path, mode="a")
    audit_writer = JsonlAppendWriter(audit_path, mode="a")

    newly_done = 0

    print(f"[Input ] {args.input}")
    print(f"[Output] {args.output_dir}")
    print(f"[Model ] {args.model}")
    print(f"[Ports ] {args.ports}")
    print(f"[Workers] {args.workers}")
    print(f"[Judges ] {args.num_judges}")
    print(f"[Total input] {total_input}")
    print(f"[Already processed] {already_processed}")
    print(f"[Pending this run] {pending_total}")

    write_summary(
        summary_path,
        args=args,
        total_input=total_input,
        already_processed=already_processed,
        pending_total=pending_total,
        newly_done=newly_done,
        kept_count=kept_count,
        dropped_count=dropped_count,
        review_count=review_count,
        start_time=start_time,
        interrupted=False,
    )

    executor = futures.ThreadPoolExecutor(max_workers=args.workers)
    pbar = None

    try:
        future_to_sample = {
            executor.submit(
                process_one_sample,
                sample=row,
                endpoint_pool=endpoint_pool,
                model=args.model,
                num_judges=args.num_judges,
                min_overall=args.min_overall,
                min_dim=args.min_dim,
            ): row
            for row in pending_rows
        }

        iterator = futures.as_completed(future_to_sample)
        if tqdm is not None:
            pbar = tqdm(iterator, total=pending_total, desc="Filtering", ncols=100)
            iterator = pbar

        for fut in iterator:
            sample = future_to_sample[fut]
            try:
                kept, dropped, audit = fut.result()
            except Exception as exc:
                question, answer, rubric = extract_sample_fields(sample)
                sample_id = make_sample_id(sample, question, answer, rubric)
                dropped = dict(sample)
                dropped["sample_id"] = sample_id
                dropped["filter_error"] = str(exc)
                audit = AuditRecord(
                    line_no=int(sample.get("__line_no__", -1)),
                    sample_id=sample_id,
                    hard_passed=False,
                    hard_issues=[f"exception:{exc}"],
                    extracted_question=question or "",
                    extracted_answer=answer or "",
                    extracted_rubric=rubric or "",
                    judge=None,
                    final_decision="drop",
                )
                kept = None

            if kept is not None:
                kept_writer.write(kept)
                kept_count += 1
            if dropped is not None:
                dropped_writer.write(dropped)
                dropped_count += 1
            if audit.final_decision == "review":
                review_count += 1

            # Important: audit is written LAST so it becomes the resume source-of-truth
            # only after kept/dropped has been safely appended.
            audit_writer.write(asdict(audit))

            newly_done += 1

            if args.save_every > 0 and (newly_done % args.save_every == 0 or newly_done == pending_total):
                write_summary(
                    summary_path,
                    args=args,
                    total_input=total_input,
                    already_processed=already_processed,
                    pending_total=pending_total,
                    newly_done=newly_done,
                    kept_count=kept_count,
                    dropped_count=dropped_count,
                    review_count=review_count,
                    start_time=start_time,
                    interrupted=False,
                )

    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        write_summary(
            summary_path,
            args=args,
            total_input=total_input,
            already_processed=already_processed,
            pending_total=pending_total,
            newly_done=newly_done,
            kept_count=kept_count,
            dropped_count=dropped_count,
            review_count=review_count,
            start_time=start_time,
            interrupted=True,
        )
        print("\nInterrupted. Partial results have been flushed. Re-run the same command to resume.")
        return
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if pbar is not None:
            pbar.close()
        kept_writer.close()
        dropped_writer.close()
        audit_writer.close()

    write_summary(
        summary_path,
        args=args,
        total_input=total_input,
        already_processed=already_processed,
        pending_total=pending_total,
        newly_done=newly_done,
        kept_count=kept_count,
        dropped_count=dropped_count,
        review_count=review_count,
        start_time=start_time,
        interrupted=False,
    )

    print("\nDone.")
    print(f"Kept    : {kept_path}")
    print(f"Dropped : {dropped_path}")
    print(f"Audit   : {audit_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
