#!/usr/bin/env python
"""
功能：清洗评估 rubric 生成用的满分标准数据 + 审计生成后 rubric 的逻辑矛盾
核心行为：
1. 读取种子数据 → 清洗不合格的 full_score_criteria 字段 → 输出清洗后数据
2. 清洗后仍不合格的数据直接删除，不新增字段
3. 支持 audit-rubrics 模式：审计 rubric 逻辑矛盾
"""

# ------------------------------ 导入依赖 ------------------------------
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
from typing import Any
import generate_score_variants_api as api_base

# ------------------------------ 全局默认配置 ------------------------------
# 文件路径
DEFAULT_INPUT = Path("datasets/full_score_dimension_seeds.jsonl")
DEFAULT_OUTPUT = Path("datasets/full_score_dimension_seeds_cleaned.jsonl")
DEFAULT_AUDIT_OUTPUT = Path("datasets/0-5/score_criteria_0_5_audit.jsonl")

# API 服务配置
DEFAULT_PORTS = tuple(range(8000, 8004))
DEFAULT_BASE_URL_TEMPLATE = "http://localhost:{port}/v1"
DEFAULT_MODEL = "Qwen3.5-27B"

# 请求参数
DEFAULT_DISPATCH_POLL_INTERVAL = 0.05
DEFAULT_MAX_TOKENS = 320
DEFAULT_TEMPERATURE = 0.0
SCORE_RANGE = tuple(range(6))

# ------------------------------ 系统提示词 ------------------------------
CLEAN_SYSTEM_PROMPT = (
    "You are a seed-cleaning assistant for evaluation rubric generation.\n\n"
    "You will receive:\n"
    "- a question,\n"
    "- a high-quality answer,\n"
    "- an evaluation dimension,\n"
    "- and a score-5 full-score criterion.\n\n"
    "Your task is to determine whether the provided score-5 criterion is clean and reasonable for that dimension only. "
    "If not, rewrite it into a clean score-5 criterion.\n\n"
    "Hard constraints:\n"
    "1. Evaluate ONLY the named dimension. Do not mix in unrelated dimensions.\n"
    "2. The criterion must describe observable properties of the answer itself.\n"
    "3. The criterion must be a true full-score description, not a near-full-score description.\n"
    "4. Avoid concessions such as 'but', 'though', 'except', 'mostly', 'slightly', 'minor'.\n"
    "5. Avoid meta wording, rubric wording, score labels, or JSON commentary inside the criterion.\n"
    "6. Keep the cleaned criterion concise, concrete, self-contained, and directly usable.\n"
    "7. Return strict JSON only."
)

REPAIR_SYSTEM_PROMPT = (
    "You are a criterion repair assistant.\n\n"
    "You will receive a previously cleaned score-5 criterion that failed validation. "
    "Rewrite it so that it is dimension-pure, concise, observable, and suitable as a true full-score criterion.\n\n"
    "Return strict JSON only."
)

# ------------------------------ 维度清洗规则 ------------------------------
DIMENSION_POLICIES: dict[str, dict[str, Any]] = {
    "factual correctness": {
        "focus": "Focus only on whether stated facts, quantities, values, and claims match the information in the question.",
        "avoid": "Do not mention completeness, reasoning quality, coherence, fluency, naturalness, comprehensibility, or formatting.",
        "forbidden_keywords": [
            "boxed", "unit", "grammar", "spelling", "syntax", "natural",
            "conversational", "robotic", "coherent", "readable", "complete answer", "complete response"
        ],
    },
    "answer completeness": {
        "focus": "Focus only on whether all required parts of the answer are present.",
        "avoid": "Do not mention correctness, arithmetic accuracy, reasoning quality, coherence, fluency, naturalness, comprehensibility, or formatting.",
        "forbidden_keywords": [
            "correct", "incorrect", "accurate", "inaccurate", "arithmetic",
            "calculation", "factual", "grammar", "spelling", "syntax", "natural",
            "conversational", "robotic", "boxed", "unit"
        ],
    },
    "reasoning chain completeness": {
        "focus": "Focus only on whether the full chain of reasoning steps is present from the given information to the conclusion.",
        "avoid": "Do not mention final-answer formatting, language fluency, expression naturalness, or general readability.",
        "forbidden_keywords": ["boxed", "unit", "grammar", "spelling", "syntax", "natural", "conversational", "robotic"],
    },
    "internal coherence": {
        "focus": "Focus only on whether the explanation is internally consistent from step to step, with no contradictions or broken links.",
        "avoid": "Do not mention completeness, final correctness, fluency, naturalness, comprehensibility, or formatting.",
        "forbidden_keywords": ["boxed", "unit", "grammar", "spelling", "syntax", "natural", "conversational", "robotic"],
    },
    "logical consistency": {
        "focus": "Focus only on whether the reasoning follows logically without contradiction or invalid assumptions.",
        "avoid": "Do not mention fluency, naturalness, formatting, or unrelated quality dimensions.",
        "forbidden_keywords": ["boxed", "unit", "grammar", "spelling", "syntax", "natural", "conversational", "robotic"],
    },
    "comprehensibility": {
        "focus": "Focus only on whether a basic reader can understand the explanation and follow the steps.",
        "avoid": "Do not mention factual correctness, completeness, naturalness, or formatting.",
        "forbidden_keywords": ["boxed", "unit", "correct final answer", "incorrect final answer", "natural human speech", "robotic"],
    },
    "language fluency": {
        "focus": "Focus only on grammar, spelling, syntax, and sentence-level fluency.",
        "avoid": "Do not mention final-answer correctness, completeness, reasoning quality, coherence, or formatting.",
        "forbidden_keywords": ["boxed", "unit", "correct final answer", "incorrect final answer", "complete answer", "all required parts"],
    },
    "expression naturalness": {
        "focus": "Focus only on whether the explanation sounds human, intuitive, and non-robotic in phrasing and flow.",
        "avoid": "Do not mention factual correctness, completeness, arithmetic correctness, formatting, or grammar/spelling as primary criteria.",
        "forbidden_keywords": ["boxed", "unit", "correct final answer", "incorrect final answer", "grammar", "spelling", "syntax"],
    },
    "data accuracy": {
        "focus": "Focus only on whether numerical values, arithmetic operations, and calculations are correct and correctly applied.",
        "avoid": "Do not mention fluency, naturalness, completeness, or formatting.",
        "forbidden_keywords": ["boxed", "unit", "natural", "conversational", "robotic", "grammar", "spelling", "syntax"],
    },
    "argument rigor": {
        "focus": "Focus only on whether the conclusion is supported by explicit, valid, and sufficiently precise reasoning.",
        "avoid": "Do not mention language fluency, naturalness, formatting, or unrelated presentation qualities.",
        "forbidden_keywords": ["boxed", "unit", "grammar", "spelling", "syntax", "natural", "conversational", "robotic"],
    },
}

# ------------------------------ 违规检测规则 ------------------------------
META_PATTERNS = [r"\bjson\b", r"\brubric\b", r"\bmeta\b", r"\bscore\s*[0-5]\b", r"\bcriterion\b", r"\bdesign process\b", r"\bprompt\b"]
CONCESSION_WORDS = ["but", "though", "however", "except", "mostly", "slightly", "minor", "nearly", "almost", "generally", "somewhat", "occasionally", "may", "might"]

# ------------------------------ 文本归一化工具函数 ------------------------------
def normalize_dimension_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)

def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())

def cleanup_criterion_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r'^\s*["“”]+|["“”]+\s*$', "", text)
    text = re.sub(r"^\s*(score\s*)?[0-5]\s*[:：-]\s*", "", text, flags=re.I)
    return text.strip()

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
            "- Keep the criterion tightly focused on the named dimension only.\n"
            "- Avoid mixing correctness, completeness, coherence, fluency, naturalness, comprehensibility, or formatting unless they are the named dimension.\n"
        )
    return (
        "Dimension-specific guidance:\n"
        f"- Focus: {policy['focus']}\n"
        f"- Avoid: {policy['avoid']}\n"
    )

# ------------------------------ 种子质量检测函数 ------------------------------
def detect_meta_issues(text: str) -> list[str]:
    issues = []
    lower = text.lower()
    for pattern in META_PATTERNS:
        if re.search(pattern, lower, flags=re.I):
            issues.append(f"meta wording detected: /{pattern}/")
    return issues

def detect_concession_issues(text: str) -> list[str]:
    issues = []
    lower = text.lower()
    for word in CONCESSION_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            issues.append(f"full-score criterion contains concession word '{word}'")
    return issues

def detect_dimension_leakage(*, dimension_name: str, criterion_text: str) -> list[str]:
    policy = get_dimension_policy(dimension_name)
    if not policy:
        return []
    issues = []
    lower = criterion_text.lower()
    for kw in policy.get("forbidden_keywords", []):
        if kw.lower() in lower:
            issues.append(f"dimension leakage: contains forbidden keyword '{kw}'")
    return sorted(set(issues))

def detect_reasonableness_issues(record: dict[str, Any], criterion_text: str) -> list[str]:
    text = cleanup_criterion_text(criterion_text)
    issues = []
    dimension_name = str(record.get("dimension_name", "")).strip()
    
    if not text:
        return ["empty full-score criterion"]
    if len(text) < 18:
        issues.append("criterion too short")
    if len(text) > 260:
        issues.append("criterion too long")

    issues.extend(detect_meta_issues(text))
    issues.extend(detect_concession_issues(text))
    issues.extend(detect_dimension_leakage(dimension_name=dimension_name, criterion_text=text))

    near_full_markers = [r"\bminor\b", r"\bslight\w*\b", r"\bmostly\b", r"\bgenerally\b", r"\bsomewhat\b", r"\bmay\b", r"\bmight\b", r"\boccasionally\b", r"\bbut\b", r"\bthough\b"]
    lower = text.lower()
    for pattern in near_full_markers:
        if re.search(pattern, lower):
            issues.append(f"full-score criterion sounds less than perfect: /{pattern}/")

    if not re.search(r"\b(answer|response|explanation|reasoning)\b", lower):
        issues.append("criterion does not explicitly refer to the answer/response/explanation")

    dim = normalize_dimension_name(dimension_name)
    if "completeness" in dim and re.search(r"\b(correct|accurate|incorrect|inaccurate)\b", lower):
        issues.append("completeness criterion is contaminated by correctness wording")
    if "fluency" in dim and re.search(r"\b(correct final answer|all required parts|complete answer)\b", lower):
        issues.append("fluency criterion is contaminated by content-quality wording")
    if ("coherence" in dim or "consistency" in dim) and re.search(r"\b(grammar|spelling|syntax|natural)\b", lower):
        issues.append("logic/coherence criterion is contaminated by language-quality wording")
    
    return sorted(set(issues))

# ------------------------------ Rubric 矛盾检测函数 ------------------------------
def detect_internal_contradictions(text: str) -> list[str]:
    t = normalize_text(text).lower()
    issues = []
    if re.search(r"\bexactly\s+\w+\b.*\bcorrect\b", t) and re.search(r"\b(one|two|three|four|a)\b.*\b(error|incorrect|wrong|mistake|slip)\b", t):
        issues.append("counting contradiction: says an exact number are correct but also says there is an error")
    if re.search(r"\ball\b.*\b(correct|accurate|consistent|complete|present)\b", t) and re.search(r"\bexcept\b.*\b(error|incorrect|wrong|missing|omits?|lack|contradiction|inconsisten\w*)\b", t):
        issues.append("absolute-vs-exception contradiction")
    if re.search(r"\bno\b.*\bcontradiction", t) and re.search(r"\bbut\b.*\bcontradiction", t):
        issues.append("contradiction contradiction")
    if re.search(r"\bno\b.*\bmissing\b", t) and re.search(r"\bbut\b.*\b(omit|omits|omitting|missing|lacks?|fails to include)\b", t):
        issues.append("completeness contradiction")
    if re.search(r"\bno\b.*\b(grammar|spelling|syntax)\b", t) and re.search(r"\bbut\b.*\b(grammar|spelling|syntax)\b.*\b(error|errors)\b", t):
        issues.append("language-quality contradiction")
    if re.search(r"\b(all|fully|entirely)\b.*\b(correct|accurate)\b", t) and re.search(r"\bbut\b.*\b(incorrect|wrong)\b", t):
        issues.append("correctness contradiction")
    return sorted(set(issues))

def validate_rubric_record_strict(record: dict[str, Any]) -> tuple[bool, list[str]]:
    issues = []
    dimension_name = str(record.get("dimension_name", "")).strip()
    criteria_map = record.get("score_criteria")
    
    if not isinstance(criteria_map, dict):
        criteria_map = {str(i): record.get(f"criteria_{i}") for i in SCORE_RANGE}
    
    texts = {}
    for score in SCORE_RANGE:
        key = str(score)
        text = cleanup_criterion_text(str(criteria_map.get(key) or ""))
        if not text:
            issues.append(f"missing criterion for score {score}")
            continue
        texts[key] = text
        internal = detect_internal_contradictions(text)
        for item in internal:
            issues.append(f"score {score}: {item}")

    if len(texts) == 6:
        normalized_set = {normalize_text(v).lower() for v in texts.values()}
        if len(normalized_set) < 5:
            issues.append("too many duplicate or near-duplicate score criteria")
        for left, right in zip(SCORE_RANGE[:-1], SCORE_RANGE[1:]):
            if normalize_text(texts[str(left)]).lower() == normalize_text(texts[str(right)]).lower():
                issues.append(f"scores {left} and {right} are identical")

    policy = get_dimension_policy(dimension_name)
    if policy:
        forbidden_keywords = policy.get("forbidden_keywords", [])
        for score in ["0", "1", "2", "3", "4"]:
            text = texts.get(score, "").lower()
            for kw in forbidden_keywords:
                if kw.lower() in text:
                    issues.append(f"score {score}: dimension leakage via '{kw}'")
                    break
    return len(issues) == 0, sorted(set(issues))

# ------------------------------ 模型提示词构建 ------------------------------
def build_clean_messages(record: dict[str, Any], heuristic_issues: list[str]) -> list[dict[str, str]]:
    user_content = (
        f"Question:\n{str(record.get('question', '')).strip()}\n\n"
        f"High-Quality Reference Answer:\n{str(record.get('answer', '')).strip()}\n\n"
        f"Evaluation Dimension:\n{str(record.get('dimension_name', '')).strip()}\n\n"
        f"Original Full-Score Criterion:\n{str(record.get('full_score_criteria', '')).strip()}\n\n"
        f"{build_dimension_guidance(record)}\n"
        "Heuristic issues already detected:\n"
        f"{json.dumps(heuristic_issues, ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        ' "is_reasonable": true,\n'
        ' "issues": ["..."],\n'
        ' "cleaned_full_score_criteria": "...",\n'
        ' "dimension_focus_summary": "..."\n'
        "}\n\n"
        "Requirements for cleaned_full_score_criteria:\n"
        "1. Keep only the intended dimension.\n"
        "2. Make it a true score-5 statement.\n"
        "3. Keep it concise, concrete, and self-contained.\n"
        "4. Do not mention scores, JSON, rubrics, or prompt-writing."
    )
    return [{"role": "system", "content": CLEAN_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

def build_repair_messages(*, record: dict[str, Any], previous_payload: dict[str, Any], validation_issues: list[str]) -> list[dict[str, str]]:
    user_content = (
        f"Question:\n{str(record.get('question', '')).strip()}\n\n"
        f"High-Quality Reference Answer:\n{str(record.get('answer', '')).strip()}\n\n"
        f"Evaluation Dimension:\n{str(record.get('dimension_name', '')).strip()}\n\n"
        f"Current invalid payload:\n{json.dumps(previous_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Validation issues:\n{json.dumps(validation_issues, ensure_ascii=False, indent=2)}\n\n"
        f"{build_dimension_guidance(record)}\n"
        "Rewrite into strict JSON with this schema:\n"
        "{\n"
        ' "is_reasonable": true,\n'
        ' "issues": ["..."],\n'
        ' "cleaned_full_score_criteria": "...",\n'
        ' "dimension_focus_summary": "..."\n'
        "}"
    )
    return [{"role": "system", "content": REPAIR_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

def parse_clean_payload(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    cleaned_text = api_base.get_payload_value(parsed, "cleaned_full_score_criteria", "full_score_criteria", "rewritten_full_score_criteria", "criterion", default="")
    cleaned_text = cleanup_criterion_text(str(cleaned_text or ""))
    is_reasonable = api_base.get_payload_value(parsed, "is_reasonable", default=True)
    
    if isinstance(is_reasonable, str):
        is_reasonable = is_reasonable.strip().lower() in {"1", "true", "yes"}
    else:
        is_reasonable = bool(is_reasonable)

    issues = api_base.get_payload_value(parsed, "issues", default=[])
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    
    focus_summary = api_base.get_payload_value(parsed, "dimension_focus_summary", default="")
    if not cleaned_text:
        return None
    
    return {
        "is_reasonable": is_reasonable,
        "issues": [normalize_text(x) for x in issues if normalize_text(x)],
        "cleaned_full_score_criteria": cleaned_text,
        "dimension_focus_summary": normalize_text(focus_summary),
    }

def validate_clean_payload(cleaned_payload: dict[str, Any], record: dict[str, Any]) -> tuple[bool, list[str]]:
    cleaned_text = cleanup_criterion_text(cleaned_payload.get("cleaned_full_score_criteria", ""))
    issues = detect_reasonableness_issues(record, cleaned_text)
    return len(issues) == 0, issues

# ------------------------------ 异步写入工具 ------------------------------
async def clean_writer(writer_queue: asyncio.Queue[tuple[Path, dict[str, Any]] | None]) -> None:
    handles = {}
    try:
        while True:
            item = await writer_queue.get()
            try:
                if item is None:
                    break
                output_path, row = item
                if output_path not in handles:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    handles[output_path] = output_path.open("a", encoding="utf-8")
                fout = handles[output_path]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
            finally:
                writer_queue.task_done()
    finally:
        for fout in handles.values():
            fout.close()

# ------------------------------ 任务构建 ------------------------------
def build_seed_tasks(*, records: list[dict[str, Any]], rewrite_all: bool) -> list[dict[str, Any]]:
    tasks = []
    suspicious_count = 0
    for record in records:
        original = str(record.get("full_score_criteria", "")).strip()
        heuristic_issues = detect_reasonableness_issues(record, original)
        if heuristic_issues:
            suspicious_count += 1
        tasks.append({
            "record": record,
            "heuristic_issues": heuristic_issues,
            "needs_rewrite": rewrite_all or bool(heuristic_issues),
        })
    print(f"[Seed Clean] total={len(records)} suspicious={suspicious_count} rewrite_all={rewrite_all} pending={len(tasks)}")
    return tasks

# ------------------------------ 核心清洗工作协程 ------------------------------
async def clean_worker(*, base_url: str, api_key: str, model: str, input_queue: asyncio.Queue, writer_queue: asyncio.Queue, output_path: Path, max_tokens: int, temperature: float, retries: int, request_timeout: int, progress_bar: Any, http_session, request_executor: ThreadPoolExecutor | None, repair_attempts: int, verbose_failures: bool) -> None:
    while True:
        try:
            task = await input_queue.get()
        except asyncio.CancelledError:
            break
        try:
            if task is None:
                break
            record = task["record"]
            heuristic_issues = list(task.get("heuristic_issues", []))
            needs_rewrite = bool(task.get("needs_rewrite", False))
            original_full = str(record.get("full_score_criteria", "")).strip()
            cleaned_full = original_full

            # 无需重写 → 直接校验
            if not needs_rewrite:
                final_issues = detect_reasonableness_issues(record, cleaned_full)
                if not final_issues:
                    await writer_queue.put((output_path, record))
                continue

            # 第一次清洗
            messages = build_clean_messages(record, heuristic_issues)
            parsed, _ = await api_base.call_model(
                base_url, model=model, api_key=api_key, messages=messages,
                max_tokens=max_tokens, temperature=temperature, retries=retries,
                timeout_seconds=request_timeout, http_session=http_session, request_executor=request_executor
            )
            if parsed is None:
                continue
            payload = parse_clean_payload(parsed)
            if payload is None:
                continue

            # 校验失败 → 修复尝试
            is_valid, validation_issues = validate_clean_payload(payload, record)
            if not is_valid:
                repaired = False
                last_payload = payload
                last_issues = validation_issues
                for _ in range(max(0, repair_attempts)):
                    repair_messages = build_repair_messages(record=record, previous_payload=last_payload, validation_issues=last_issues)
                    repaired_parsed, _ = await api_base.call_model(
                        base_url, model=model, api_key=api_key, messages=repair_messages,
                        max_tokens=max_tokens, temperature=temperature, retries=retries,
                        timeout_seconds=request_timeout, http_session=http_session, request_executor=request_executor
                    )
                    if not repaired_parsed:
                        continue
                    repaired_payload = parse_clean_payload(repaired_parsed)
                    if not repaired_payload:
                        continue
                    is_valid, repair_validation_issues = validate_clean_payload(repaired_payload, record)
                    if is_valid:
                        payload = repaired_payload
                        repaired = True
                        break
                    last_payload = repaired_payload
                    last_issues = repair_validation_issues
                if not repaired and not is_valid:
                    continue

            # 最终校验
            cleaned_full = payload["cleaned_full_score_criteria"]
            final_issues = detect_reasonableness_issues(record, cleaned_full)
            if final_issues:
                continue

            # 输出结果（仅替换 full_score_criteria）
            output_row = dict(record)
            output_row["full_score_criteria"] = cleaned_full
            await writer_queue.put((output_path, output_row))
        except Exception as exc:
            if verbose_failures:
                print(f"[ERROR] clean_worker failed: {exc}")
        finally:
            input_queue.task_done()
            if task is not None and progress_bar is not None:
                progress_bar.update(1)

# ------------------------------ 种子清洗主流程 ------------------------------
async def run_seed_clean(*, base_urls: list[str], api_key: str, model: str, tasks: list[dict[str, Any]], output_path: Path, max_tokens: int, temperature: float, retries: int, concurrency_per_endpoint: int, request_timeout: int, dispatch_poll_interval: float, repair_attempts: int, verbose_failures: bool) -> None:
    if not tasks:
        print("No pending tasks. Nothing to clean.")
        return

    # 清空输出文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    # 进度条
    progress_bar = api_base.tqdm(total=len(tasks), desc="Cleaning seeds", unit="task", dynamic_ncols=True) if api_base.tqdm else None
    queue_capacity = max(1, concurrency_per_endpoint)
    endpoint_queues = [asyncio.Queue(maxsize=queue_capacity) for _ in base_urls]
    writer_queue = asyncio.Queue()
    writer_task = asyncio.create_task(clean_writer(writer_queue))
    dispatcher_task = asyncio.create_task(api_base.dispatch_tasks_round_robin(tasks, endpoint_queues, poll_interval=dispatch_poll_interval))
    workers = []
    worker_count = max(1, concurrency_per_endpoint)

    async def launch_workers(http_session, request_executor):
        for base_url, input_queue in zip(base_urls, endpoint_queues):
            for _ in range(worker_count):
                workers.append(asyncio.create_task(clean_worker(
                    base_url=base_url, api_key=api_key, model=model, input_queue=input_queue,
                    writer_queue=writer_queue, output_path=output_path, max_tokens=max_tokens,
                    temperature=temperature, retries=retries, request_timeout=request_timeout,
                    progress_bar=progress_bar, http_session=http_session, request_executor=request_executor,
                    repair_attempts=repair_attempts, verbose_failures=verbose_failures
                )))
        await dispatcher_task
        await asyncio.gather(*workers)

    try:
        if api_base.aiohttp:
            connector = api_base.aiohttp.TCPConnector(limit=max(64, len(base_urls)*worker_count*2))
            timeout = api_base.aiohttp.ClientTimeout(total=request_timeout)
            async with api_base.aiohttp.ClientSession(connector=connector, timeout=timeout) as http_session:
                await launch_workers(http_session, None)
        else:
            with ThreadPoolExecutor(max_workers=max(32, len(base_urls)*worker_count)) as executor:
                await launch_workers(None, executor)
    finally:
        await writer_queue.put(None)
        await writer_queue.join()
        await writer_task
        if progress_bar:
            progress_bar.close()

# ------------------------------ Rubric 审计流程 ------------------------------
def audit_rubric_file(input_path: Path, output_path: Path) -> None:
    records = api_base.load_records(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total, invalid = 0, 0
    with output_path.open("w", encoding="utf-8") as fout:
        for record in records:
            total += 1
            is_valid, issues = validate_rubric_record_strict(record)
            row = {
                "sample_id": record.get("sample_id") or api_base.build_sample_id(record),
                "dimension_name": record.get("dimension_name"),
                "is_valid": is_valid,
                "issues": issues
            }
            if not is_valid:
                invalid += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[Rubric Audit] input={input_path} total={total} invalid={invalid} output={output_path}")

# ------------------------------ CLI 命令行参数 ------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean full-score seed criteria and audit rubric contradictions.")
    parser.add_argument("--mode", default="clean-seeds", choices=["clean-seeds", "audit-rubrics"], help="运行模式：清洗种子/审计rubric")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="输入文件路径")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="清洗后输出路径")
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT_OUTPUT), help="审计报告输出路径")
    parser.add_argument("--base-url-template", default=DEFAULT_BASE_URL_TEMPLATE, help="API地址模板")
    parser.add_argument("--ports", nargs="*", default=[f"{DEFAULT_PORTS[0]}-{DEFAULT_PORTS[-1]}"], help="API端口")
    parser.add_argument("--api-key", default=None, help="API密钥")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="最大生成token")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="采样温度")
    parser.add_argument("--retries", type=int, default=3, help="请求重试次数")
    parser.add_argument("--concurrency-per-endpoint", type=int, default=32, help="单端口并发数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前N条数据")
    parser.add_argument("--dispatch-poll-interval", type=float, default=DEFAULT_DISPATCH_POLL_INTERVAL, help="任务调度间隔")
    parser.add_argument("--skip-health-check", action="store_true", help="跳过API健康检查")
    parser.add_argument("--health-check-timeout", type=int, default=10, help="健康检查超时时间")
    parser.add_argument("--request-timeout", type=int, default=180, help="请求超时时间")
    parser.add_argument("--rewrite-all", action="store_true", help="强制重写所有样本")
    parser.add_argument("--repair-attempts", type=int, default=1, help="修复尝试次数")
    parser.add_argument("--verbose-failures", action="store_true", help="打印失败详情")
    return parser.parse_args()

# ------------------------------ 主入口 ------------------------------
async def async_main(args: argparse.Namespace) -> None:
    if args.mode == "audit-rubrics":
        audit_rubric_file(Path(args.input), Path(args.audit_output))
        return

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = api_base.load_records(input_path)
    if args.limit:
        records = records[:max(0, args.limit)]

    ports = api_base.parse_ports(args.ports)
    base_urls = api_base.build_base_urls(args.base_url_template, ports)
    
    if not args.skip_health_check:
        await api_base.wait_for_servers(base_urls, timeout_seconds=args.health_check_timeout)
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    tasks = build_seed_tasks(records=records, rewrite_all=args.rewrite_all)
    await run_seed_clean(
        base_urls=base_urls, api_key=api_key, model=args.model, tasks=tasks, output_path=output_path,
        max_tokens=args.max_tokens, temperature=args.temperature, retries=args.retries,
        concurrency_per_endpoint=args.concurrency_per_endpoint, request_timeout=args.request_timeout,
        dispatch_poll_interval=args.dispatch_poll_interval, repair_attempts=args.repair_attempts,
        verbose_failures=args.verbose_failures
    )

def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()