#!/usr/bin/env python
"""Train a score+reason+rewrite model with GRPO.

This script is designed for the repository's single-dimension evaluation setup:

Input:
  {question + answer + dimension + score_criteria}

Output:
  Score: <0-5>
  Reason: <rubric-grounded explanation>
  Modified Answer: <improved answer for the same dimension>

Reward design
-------------
The total reward is a weighted combination of four normalized components:

1. format_reward
   Parses the completion and checks whether all required fields exist.

2. score_reward
   Sample-level proxy for "correlation with GPT". Correlation itself is global,
   so for RL we optimize a per-sample agreement surrogate against the teacher
   score:

     score_reward = 0.7 * (1 - abs(pred_score - teacher_score) / 5)
                  + 0.3 * exact_match(pred_score, teacher_score)

3. reason_reward
   Embedding cosine similarity between the generated reason and the rubric
   criterion corresponding to the generated score. If a teacher reason exists,
   an optional extra similarity term can be blended in.

4. rewrite_reward
   Improvement of the rewritten answer over the original answer under the same
   dimension/rubric. By default it re-scores the rewritten answer with a frozen
   judge endpoint and compares it with the original-answer teacher score:

     rewrite_reward = max(0, judge_score(rewrite) - base_score) / max(1, 5 - base_score)

   If the original teacher score is missing, the script can judge the original
   answer on the fly as a fallback.

Total reward:

  total = weighted_mean(
      format_reward,
      score_reward,
      reason_reward,
      rewrite_reward,
  ) - penalties

Expected input JSONL
--------------------
The script reads flat JSONL rows and auto-detects common fields such as:
  question, answer, dimension_name/evaluation_dimension, score_criteria,
  full_score_criteria, reference_score/Score/score, reference_reason/Reason,
  reference_modified_answer/modified_answer.

Examples that work well:
  - datasets/baseline_new/predictions/openai_cortex-5_judge_outputs_with_reference.jsonl
  - datasets/0-5/score_0.jsonl ... score_4.jsonl
  - datasets/0-5/full_score_dimension_seeds.jsonl

Dependencies
------------
This script expects TRL + Transformers and optionally PEFT:
  pip install trl transformers accelerate peft
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional

from datasets import Dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers is required for this script.") from exc

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("trl is required for this script. Please install trl first.") from exc

try:
    from peft import LoraConfig
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_PIPELINE_DIR = SCRIPT_DIR.parent / "evaluation_pipeline"
if str(EVAL_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE_DIR))

from pipeline_common import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL_TEMPLATE,
    build_base_urls,
    call_chat_with_retries,
    cosine_similarity,
    embed_texts_with_retries,
    normalize_score_criteria,
    normalize_text,
    parse_ports,
    resolve_model,
    wait_for_servers,
)


SYSTEM_PROMPT = """You are an AI evaluator-and-rewriter.
Evaluate the given answer strictly using ONLY the provided evaluation dimension and its complete score criteria.
Then revise the answer to better satisfy ONLY that dimension.
Do not add unsupported facts. If an assumption is necessary, state it minimally and explicitly.
Output plain text in exactly 3 lines, with exactly these prefixes and no numbering:
Score: <an integer from 0 to 5>
Reason: <one-line concise explanation strictly based on the given dimension criteria>
Modified Answer: <one-line revised answer optimized only for the given dimension criteria>
Do not include any extra text, JSON, markdown, bullets, or extra line breaks inside any field."""

USER_TEMPLATE = """###Task Description:
You are given a question, a response to evaluate, and one evaluation dimension with its complete 0-5 score criteria.
1. Write a score that reflects how well the response satisfies the criteria.
2. Write feedback that assesses the quality of the response strictly based on the dimension criteria.
3. Then rewrite the answer so it better satisfies the criteria, without adding unsupported facts.
4. The output format must be exactly:
Score: <score>
Reason: <feedback>
Modified Answer: <rewritten answer>
5. Do not generate any other opening, closing, JSON, or explanations.

Question:
{question}

Answer:
{answer}

Evaluation Dimension:
{dimension_name}

Score Criteria (0-5):
{criteria_text}"""

JUDGE_SCORE_SYSTEM_PROMPT = """You are a strict answer evaluator.
Evaluate the candidate answer using only the question, evaluation dimension, and 0-5 score criteria.
Return exactly one line:
Score: <integer 0-5>
Do not output anything else."""

JUDGE_SCORE_USER_TEMPLATE = """Question:
{question}

Candidate Answer:
{candidate_answer}

Evaluation Dimension:
{dimension_name}

Score Criteria (0-5):
{criteria_text}

Return only the score line."""

SCORE_LINE_RE = re.compile(r"(?im)^\s*score\s*[:\uFF1A]\s*([0-5])\s*$")
REASON_RE = re.compile(
    r"(?is)reason\s*[:\uFF1A]\s*(.*?)\s*(?:(?:\n\s*)?(?:modified answer|revised answer)\s*[:\uFF1A]|$)"
)
MODIFIED_RE = re.compile(r"(?is)(?:modified answer|revised answer)\s*[:\uFF1A]\s*(.*)$")


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def format_score_criteria_text(row: dict[str, Any]) -> str:
    criteria = normalize_score_criteria(row, first_non_empty(row.get("full_score_criteria"), row.get("criteria_5")))
    pieces = []
    for score in range(6):
        text = normalize_text(criteria.get(str(score)))
        if text:
            pieces.append(f"Score {score}: {text}")
    return "\n".join(pieces)


def parse_int_score(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            rounded = round(number)
            if abs(number - rounded) < 1e-6 and 0 <= rounded <= 5:
                return int(rounded)
        return None
    text = str(value).strip()
    match = re.search(r"\b([0-5])\b", text)
    return int(match.group(1)) if match else None


def parse_completion(text: str) -> dict[str, Any]:
    raw = str(text or "").strip().replace("\r\n", "\n")
    score_match = SCORE_LINE_RE.search(raw)
    reason_match = REASON_RE.search(raw)
    modified_match = MODIFIED_RE.search(raw)
    score = int(score_match.group(1)) if score_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    modified_answer = normalize_text(modified_match.group(1)) if modified_match else ""
    is_valid = score is not None and bool(reason) and bool(modified_answer)
    return {
        "score": score,
        "reason": reason,
        "modified_answer": modified_answer,
        "is_valid": is_valid,
        "raw_text": raw,
    }


def extract_teacher_score(row: dict[str, Any]) -> Optional[int]:
    for key in ("reference_score", "Score", "score", "predicted_score", "target_score"):
        score = parse_int_score(row.get(key))
        if score is not None:
            return score
    return None


def extract_teacher_reason(row: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("reference_reason"),
        row.get("Reason"),
        row.get("reason"),
        row.get("predicted_reason"),
        row.get("target_reason"),
    )


def extract_teacher_modified_answer(row: dict[str, Any]) -> str:
    return first_non_empty(
        row.get("reference_modified_answer"),
        row.get("modified_answer"),
        row.get("Modified Answer"),
        row.get("predicted_modified_answer"),
        row.get("target_modified_answer"),
    )


def extract_dimension_name(row: dict[str, Any]) -> str:
    return first_non_empty(row.get("dimension_name"), row.get("evaluation_dimension"), row.get("name"))


def build_prompt(row: dict[str, Any]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        + USER_TEMPLATE.format(
            question=row["question"],
            answer=row["answer"],
            dimension_name=row["dimension_name"],
            criteria_text=row["criteria_text"],
        )
    )


def prepare_row(row: dict[str, Any], index: int) -> Optional[dict[str, Any]]:
    question = first_non_empty(
        row.get("question"),
        row.get("prompt"),
        row.get("instruction"),
        row.get("query"),
        row.get("problem"),
    )
    answer = first_non_empty(
        row.get("answer"),
        row.get("response"),
        row.get("output"),
        row.get("candidate_answer"),
        row.get("model_answer"),
    )
    dimension_name = extract_dimension_name(row)
    criteria_text = format_score_criteria_text(row)
    if not question or not answer or not dimension_name or not criteria_text:
        return None

    prepared = {
        "sample_id": first_non_empty(row.get("sample_id"), row.get("id"), row.get("unique_id"), f"row:{index}"),
        "question": question,
        "answer": answer,
        "dimension_name": dimension_name,
        "criteria_text": criteria_text,
        "target_score": extract_teacher_score(row),
        "target_reason": extract_teacher_reason(row),
        "target_modified_answer": extract_teacher_modified_answer(row),
    }
    prepared["prompt"] = build_prompt(prepared)
    return prepared


def load_training_dataset(path: Path, limit: Optional[int]) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if limit is not None and len(rows) >= limit:
                break
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                continue
            prepared = prepare_row(payload, index)
            if prepared is not None:
                rows.append(prepared)
    if not rows:
        raise ValueError(f"No usable rows found in {path}")
    return Dataset.from_list(rows)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class RewardRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.judge_base_urls = build_base_urls(args.judge_base_url_template, parse_ports(args.judge_ports))
        self.embedding_base_urls = build_base_urls(
            args.embedding_base_url_template,
            parse_ports(args.embedding_ports),
        )
        self.api_key = args.api_key or DEFAULT_API_KEY
        if not args.skip_server_wait:
            wait_for_servers(self.judge_base_urls, timeout_seconds=args.request_timeout, interval_seconds=1.0)
            wait_for_servers(self.embedding_base_urls, timeout_seconds=args.request_timeout, interval_seconds=1.0)
        self.judge_model = resolve_model(args.judge_model, self.judge_base_urls, args.request_timeout)
        self.embedding_model = resolve_model(args.embedding_model, self.embedding_base_urls, args.request_timeout)

    def judge_answer_score(self, sample: dict[str, Any], candidate_answer: str, task_index: int) -> Optional[int]:
        if not candidate_answer:
            return None
        prompt = JUDGE_SCORE_USER_TEMPLATE.format(
            question=sample["question"],
            candidate_answer=candidate_answer,
            dimension_name=sample["dimension_name"],
            criteria_text=sample["criteria_text"],
        )
        try:
            raw_text, _ = call_chat_with_retries(
                base_urls=self.judge_base_urls,
                task_index=task_index,
                api_key=self.api_key,
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SCORE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=16,
                timeout_seconds=self.args.request_timeout,
                retries=self.args.request_retries,
                retry_sleep=self.args.retry_sleep,
            )
        except Exception:
            return None
        return parse_int_score(raw_text)

    def embed_pairs(
        self,
        left_texts: list[str],
        right_texts: list[str],
    ) -> list[Optional[float]]:
        if not left_texts:
            return []
        embeddings = embed_texts_with_retries(
            base_url=self.embedding_base_urls[0],
            api_key=self.api_key,
            model=self.embedding_model,
            texts=left_texts + right_texts,
            timeout_seconds=self.args.request_timeout,
            retries=self.args.request_retries,
            retry_sleep=self.args.retry_sleep,
        )
        midpoint = len(left_texts)
        left_vectors = embeddings[:midpoint]
        right_vectors = embeddings[midpoint:]
        sims: list[Optional[float]] = []
        for left, right in zip(left_vectors, right_vectors):
            sims.append(cosine_similarity(left, right))
        return sims


class WeightedReward:
    def __init__(self, runtime: RewardRuntime, args: argparse.Namespace) -> None:
        self.runtime = runtime
        self.args = args

    def _format_reward(self, parsed: dict[str, Any]) -> float:
        if not parsed["is_valid"]:
            return 0.0
        raw_text = parsed["raw_text"]
        lines = [line for line in raw_text.splitlines() if line.strip()]
        strict_prefixes = [
            lines[0].startswith("Score:") if len(lines) > 0 else False,
            lines[1].startswith("Reason:") if len(lines) > 1 else False,
            lines[2].startswith("Modified Answer:") if len(lines) > 2 else False,
        ]
        strict_bonus = sum(1 for ok in strict_prefixes if ok) / 3.0
        return 0.7 + 0.3 * strict_bonus

    def _score_reward(self, predicted_score: Optional[int], target_score: Optional[int]) -> Optional[float]:
        if predicted_score is None or target_score is None:
            return None
        distance_term = 1.0 - (abs(predicted_score - target_score) / 5.0)
        exact_match = 1.0 if predicted_score == target_score else 0.0
        return clamp01(0.7 * distance_term + 0.3 * exact_match)

    def _reason_reward(
        self,
        parsed_items: list[dict[str, Any]],
        samples: list[dict[str, Any]],
    ) -> list[Optional[float]]:
        left_texts: list[str] = []
        right_texts: list[str] = []
        pair_metadata: list[tuple[int, float, str]] = []

        for index, (parsed, sample) in enumerate(zip(parsed_items, samples)):
            if not parsed["reason"]:
                continue

            criterion_map = normalize_score_criteria(
                {"score_criteria": self._criteria_dict_from_text(sample["criteria_text"])},
                "",
            )
            criterion_score = parsed["score"]
            if criterion_score is None and sample.get("target_score") is not None:
                criterion_score = int(sample["target_score"])
            criterion_text = normalize_text(criterion_map.get(str(criterion_score))) if criterion_score is not None else ""
            if not criterion_text:
                continue

            left_texts.append(parsed["reason"])
            right_texts.append(f"Score {criterion_score}: {criterion_text}")
            pair_metadata.append((index, 1.0 - self.args.reason_reference_weight, "criterion"))

            teacher_reason = normalize_text(sample.get("target_reason"))
            if teacher_reason and self.args.reason_reference_weight > 0.0:
                left_texts.append(parsed["reason"])
                right_texts.append(teacher_reason)
                pair_metadata.append((index, self.args.reason_reference_weight, "teacher_reason"))

        similarities = self.runtime.embed_pairs(left_texts, right_texts)
        per_sample_weighted_sum: dict[int, float] = {}
        per_sample_total_weight: dict[int, float] = {}
        for meta, similarity in zip(pair_metadata, similarities):
            sample_index, weight, _ = meta
            if similarity is None:
                continue
            normalized = clamp01((float(similarity) + 1.0) / 2.0)
            per_sample_weighted_sum[sample_index] = per_sample_weighted_sum.get(sample_index, 0.0) + weight * normalized
            per_sample_total_weight[sample_index] = per_sample_total_weight.get(sample_index, 0.0) + weight

        rewards: list[Optional[float]] = [None] * len(samples)
        for index in range(len(samples)):
            total_weight = per_sample_total_weight.get(index, 0.0)
            if total_weight > 0:
                rewards[index] = per_sample_weighted_sum[index] / total_weight
        return rewards

    def _rewrite_rewards(
        self,
        parsed_items: list[dict[str, Any]],
        samples: list[dict[str, Any]],
    ) -> list[Optional[float]]:
        rewards: list[Optional[float]] = [None] * len(samples)

        def _judge_single(index: int) -> tuple[int, Optional[int], Optional[int]]:
            sample = samples[index]
            parsed = parsed_items[index]
            if not parsed["modified_answer"]:
                return index, None, None

            base_score = parse_int_score(sample.get("target_score"))
            if base_score is None and self.args.judge_original_if_missing:
                base_score = self.runtime.judge_answer_score(sample, sample["answer"], index * 2)
            rewritten_score = self.runtime.judge_answer_score(sample, parsed["modified_answer"], index * 2 + 1)
            return index, base_score, rewritten_score

        with ThreadPoolExecutor(max_workers=self.args.reward_workers) as executor:
            futures = [executor.submit(_judge_single, index) for index in range(len(samples))]
            for future in as_completed(futures):
                index, base_score, rewritten_score = future.result()
                if base_score is None or rewritten_score is None:
                    continue
                if base_score >= 5:
                    rewards[index] = 1.0 if rewritten_score >= 5 else 0.0
                    continue
                gain = max(0.0, float(rewritten_score - base_score))
                headroom = max(1.0, float(5 - base_score))
                rewards[index] = clamp01(gain / headroom)
        return rewards

    @staticmethod
    def _criteria_dict_from_text(criteria_text: str) -> dict[str, str]:
        criteria: dict[str, str] = {}
        pattern = re.compile(r"(?ms)^\s*(?:Score\s*)?([0-5])\s*[:\uFF1A]\s*(.*?)(?=^\s*(?:Score\s*)?[0-5]\s*[:\uFF1A]|\Z)")
        for match in pattern.finditer(criteria_text):
            criteria[match.group(1)] = normalize_text(match.group(2))
        return criteria

    @staticmethod
    def _completion_to_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return normalize_text(completion.get("content"))
        if isinstance(completion, list):
            pieces: list[str] = []
            for item in completion:
                if isinstance(item, dict):
                    pieces.append(str(item.get("content") or ""))
                else:
                    pieces.append(str(item))
            return "\n".join(piece for piece in pieces if piece).strip()
        return str(completion or "")

    def __call__(
        self,
        prompts: list[str],
        completions: list[Any],
        question: list[str],
        answer: list[str],
        dimension_name: list[str],
        criteria_text: list[str],
        target_score: list[Any],
        target_reason: list[str],
        target_modified_answer: list[str],
        sample_id: list[str],
        **_: Any,
    ) -> list[float]:
        samples = [
            {
                "prompt": prompts[index],
                "question": question[index],
                "answer": answer[index],
                "dimension_name": dimension_name[index],
                "criteria_text": criteria_text[index],
                "target_score": target_score[index],
                "target_reason": target_reason[index],
                "target_modified_answer": target_modified_answer[index],
                "sample_id": sample_id[index],
            }
            for index in range(len(completions))
        ]

        parsed_items = [parse_completion(self._completion_to_text(text)) for text in completions]
        format_rewards = [self._format_reward(parsed) for parsed in parsed_items]
        score_rewards = [
            self._score_reward(parsed["score"], parse_int_score(sample["target_score"]))
            for parsed, sample in zip(parsed_items, samples)
        ]
        reason_rewards = self._reason_reward(parsed_items, samples)
        rewrite_rewards = self._rewrite_rewards(parsed_items, samples)

        final_rewards: list[float] = []
        for index, parsed in enumerate(parsed_items):
            if not parsed["is_valid"]:
                final_rewards.append(self.args.invalid_output_penalty)
                continue

            weighted_components: list[tuple[float, float]] = [(self.args.format_weight, format_rewards[index])]

            if score_rewards[index] is not None:
                weighted_components.append((self.args.score_weight, float(score_rewards[index])))
            if reason_rewards[index] is not None:
                weighted_components.append((self.args.reason_weight, float(reason_rewards[index])))
            if rewrite_rewards[index] is not None:
                weighted_components.append((self.args.rewrite_weight, float(rewrite_rewards[index])))

            total_weight = sum(weight for weight, _ in weighted_components)
            if total_weight <= 0:
                reward = 0.0
            else:
                reward = sum(weight * value for weight, value in weighted_components) / total_weight

            if normalize_text(parsed["modified_answer"]) == normalize_text(samples[index]["answer"]):
                reward -= self.args.noop_rewrite_penalty
            if len(parsed["modified_answer"]) > max(1, len(samples[index]["answer"])) * self.args.max_rewrite_ratio:
                reward -= self.args.verbosity_penalty

            final_rewards.append(float(reward))
        return final_rewards


def build_peft_config(args: argparse.Namespace) -> Any:
    if not args.use_lora:
        return None
    if LoraConfig is None:
        raise ValueError("peft is required for --use-lora.")
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def make_grpo_config(args: argparse.Namespace) -> Any:
    config_kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "beta": args.beta,
    }
    signature = inspect.signature(GRPOConfig.__init__)
    filtered = {key: value for key, value in config_kwargs.items() if key in signature.parameters and value is not None}
    return GRPOConfig(**filtered)


def make_trainer(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    args: argparse.Namespace,
    reward_fn: WeightedReward,
    peft_config: Any,
) -> Any:
    trainer_kwargs = {
        "model": model,
        "args": make_grpo_config(args),
        "train_dataset": train_dataset,
        "reward_funcs": [reward_fn],
        "peft_config": peft_config,
    }
    signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    return GRPOTrainer(**trainer_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a score/reason/rewrite model with GRPO.")
    parser.add_argument("--input", type=Path, required=True, help="Flat JSONL training file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Base causal LM.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=None, help="Optional KL coefficient for GRPO if supported.")

    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report-to", type=str, default="none")

    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--judge-base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--judge-ports", nargs="+", default=["8000-8003"])
    parser.add_argument("--embedding-base-url-template", type=str, default=DEFAULT_BASE_URL_TEMPLATE)
    parser.add_argument("--embedding-ports", nargs="+", default=["8004"])
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--judge-model", type=str, default="")
    parser.add_argument("--embedding-model", type=str, default="")
    parser.add_argument("--skip-server-wait", action="store_true")
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--judge-original-if-missing", action="store_true")

    parser.add_argument("--format-weight", type=float, default=0.10)
    parser.add_argument("--score-weight", type=float, default=0.40)
    parser.add_argument("--reason-weight", type=float, default=0.20)
    parser.add_argument("--rewrite-weight", type=float, default=0.30)
    parser.add_argument("--reason-reference-weight", type=float, default=0.0)
    parser.add_argument("--invalid-output-penalty", type=float, default=-1.0)
    parser.add_argument("--noop-rewrite-penalty", type=float, default=0.10)
    parser.add_argument("--verbosity-penalty", type=float, default=0.05)
    parser.add_argument("--max-rewrite-ratio", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset(args.input, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    runtime = RewardRuntime(args)
    reward_fn = WeightedReward(runtime, args)
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
        reward_fn=reward_fn,
        peft_config=build_peft_config(args),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
