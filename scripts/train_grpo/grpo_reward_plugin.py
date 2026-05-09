"""ms-swift custom reward plugin for ChainCritic GRPO.

Use with:

  swift rlhf --rlhf_type grpo \
    --external_plugins scripts/train/grpo_reward_plugin.py \
    --reward_funcs chaincritic_weighted

Reward inputs are expected to come from scripts/train/prepare_grpo_dataset.py.
The implementation keeps all hyperparameters in environment variables so the
shell training script can tune weights without editing Python.

Reward components:
- score_reward: generated Score vs target_score.
- reason_reward: generated Reason vs the rubric item for the generated Score.
- rewrite_reward: generated Modified Answer vs gt_answer embedding similarity.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any, Optional
from urllib import request as urllib_request

try:
    from swift.rewards import ORM, orms
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"Failed to import ms-swift reward API: {exc}") from exc

SCORE_LINE_RE = re.compile(r"(?im)^\s*score\s*[:\uFF1A]\s*([0-5])\s*$")
REASON_RE = re.compile(
    r"(?is)reason\s*[:\uFF1A]\s*(.*?)\s*(?:(?:\n\s*)?(?:modified answer|revised answer)\s*[:\uFF1A]|$)"
)
MODIFIED_RE = re.compile(r"(?is)(?:modified answer|revised answer)\s*[:\uFF1A]\s*(.*)$")
CRITERION_RE = re.compile(
    r"(?ms)(?:^|\n|\s)(?:Score\s*)?([0-5])\s*[:\uFF1A]\s*(.*?)(?=(?:^|\n|\s)(?:Score\s*)?[0-5]\s*[:\uFF1A]|\Z)"
)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    return int(value)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


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
    match = re.search(r"\b([0-5])\b", str(value))
    return int(match.group(1)) if match else None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_completion(text: Any) -> dict[str, Any]:
    raw = completion_to_text(text).strip().replace("\r\n", "\n")
    score_match = SCORE_LINE_RE.search(raw)
    reason_match = REASON_RE.search(raw)
    modified_match = MODIFIED_RE.search(raw)
    score = int(score_match.group(1)) if score_match else None
    reason = normalize_text(reason_match.group(1)) if reason_match else ""
    modified_answer = normalize_text(modified_match.group(1)) if modified_match else ""
    return {
        "score": score,
        "reason": reason,
        "modified_answer": modified_answer,
        "raw_text": raw,
        "is_valid": score is not None and bool(reason) and bool(modified_answer),
    }


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return normalize_text(completion.get("content"))
    if isinstance(completion, list):
        pieces = []
        for item in completion:
            if isinstance(item, dict):
                pieces.append(str(item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(completion or "")


def parse_criteria(criteria_text: Any) -> dict[str, str]:
    criteria: dict[str, str] = {}
    for match in CRITERION_RE.finditer(str(criteria_text or "")):
        criteria[match.group(1)] = normalize_text(match.group(2))
    return criteria


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    req = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout: float, retries: int, retry_sleep: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                response = post_json(f"{self.base_url}/embeddings", payload, self.api_key, self.timeout)
                data = sorted(response.get("data") or [], key=lambda item: int(item.get("index", 0)))
                embeddings = [item.get("embedding") for item in data]
                if len(embeddings) == len(texts) and all(isinstance(item, list) for item in embeddings):
                    return embeddings  # type: ignore[return-value]
            except Exception as exc:
                last_error = exc
            time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"Embedding request failed: {last_error}")


def cosine_similarity(left: list[float], right: list[float]) -> Optional[float]:
    if len(left) != len(right) or not left:
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for lv, rv in zip(left, right):
        left_value = float(lv)
        right_value = float(rv)
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return dot / math.sqrt(left_norm * right_norm)


class ChainCriticWeightedORM(ORM):
    def __init__(self, args=None, **kwargs) -> None:
        super().__init__()
        self.args = args
        
        self.format_weight = env_float("CHAINCRITIC_FORMAT_WEIGHT", 0.10)
        self.score_weight = env_float("CHAINCRITIC_SCORE_WEIGHT", 0.40)
        self.reason_weight = env_float("CHAINCRITIC_REASON_WEIGHT", 0.20)
        self.rewrite_weight = env_float("CHAINCRITIC_REWRITE_WEIGHT", 0.30)
        self.invalid_output_penalty = env_float("CHAINCRITIC_INVALID_OUTPUT_PENALTY", -1.0)
        self.noop_rewrite_penalty = env_float("CHAINCRITIC_NOOP_REWRITE_PENALTY", 0.10)
        self.verbosity_penalty = env_float("CHAINCRITIC_VERBOSITY_PENALTY", 0.05)
        self.max_rewrite_ratio = env_float("CHAINCRITIC_MAX_REWRITE_RATIO", 3.0)

        api_key = os.getenv("CHAINCRITIC_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
        timeout = env_float("CHAINCRITIC_REQUEST_TIMEOUT", 120.0)
        retries = env_int("CHAINCRITIC_REQUEST_RETRIES", 2)
        retry_sleep = env_float("CHAINCRITIC_RETRY_SLEEP", 1.0)
        self.embedding = OpenAICompatibleClient(
            os.getenv("CHAINCRITIC_EMBEDDING_BASE_URL", "http://127.0.0.1:8004/v1"),
            os.getenv("CHAINCRITIC_EMBEDDING_MODEL", ""),
            api_key,
            timeout,
            retries,
            retry_sleep,
        )

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        parsed_items = [parse_completion(completion) for completion in completions]
        samples = self._build_samples(kwargs, len(parsed_items))
        score_rewards = [self._score_reward(item["score"], parse_int_score(sample.get("target_score"))) for item, sample in zip(parsed_items, samples)]
        reason_rewards = self._reason_rewards(parsed_items, samples)
        rewrite_rewards = self._rewrite_rewards(parsed_items, samples)

        rewards: list[float] = []
        for index, parsed in enumerate(parsed_items):
            if not parsed["is_valid"]:
                rewards.append(self.invalid_output_penalty)
                continue

            components = [(self.format_weight, self._format_reward(parsed))]
            if score_rewards[index] is not None:
                components.append((self.score_weight, float(score_rewards[index])))
            if reason_rewards[index] is not None:
                components.append((self.reason_weight, float(reason_rewards[index])))
            if rewrite_rewards[index] is not None:
                components.append((self.rewrite_weight, float(rewrite_rewards[index])))

            total_weight = sum(weight for weight, _ in components)
            reward = sum(weight * value for weight, value in components) / total_weight if total_weight > 0 else 0.0

            if normalize_text(parsed["modified_answer"]) == normalize_text(samples[index].get("answer")):
                reward -= self.noop_rewrite_penalty
            if len(parsed["modified_answer"]) > max(1, len(str(samples[index].get("answer") or ""))) * self.max_rewrite_ratio:
                reward -= self.verbosity_penalty
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def _build_samples(kwargs: dict[str, Any], count: int) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index in range(count):
            sample = {}
            for key, value in kwargs.items():
                if isinstance(value, list) and len(value) == count:
                    sample[key] = value[index]
                else:
                    sample[key] = value
            samples.append(sample)
        return samples

    @staticmethod
    def _format_reward(parsed: dict[str, Any]) -> float:
        lines = [line for line in parsed["raw_text"].splitlines() if line.strip()]
        strict = [
            len(lines) > 0 and lines[0].startswith("Score:"),
            len(lines) > 1 and lines[1].startswith("Reason:"),
            len(lines) > 2 and lines[2].startswith("Modified Answer:"),
        ]
        return 0.7 + 0.3 * (sum(1 for item in strict if item) / 3.0)

    @staticmethod
    def _score_reward(predicted: Optional[int], target: Optional[int]) -> Optional[float]:
        if predicted is None or target is None:
            return None
        distance_term = 1.0 - abs(predicted - target) / 5.0
        exact = 1.0 if predicted == target else 0.0
        return clamp01(0.7 * distance_term + 0.3 * exact)

    def _reason_rewards(self, parsed_items: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[Optional[float]]:
        left_texts: list[str] = []
        right_texts: list[str] = []
        row_indices: list[int] = []

        for index, (parsed, sample) in enumerate(zip(parsed_items, samples)):
            score = parsed["score"]
            reason = parsed["reason"]
            criterion = parse_criteria(sample.get("criteria_text")).get(str(score))
            if score is None or not reason or not criterion:
                continue
            left_texts.append(reason)
            right_texts.append(f"Score {score}: {criterion}")
            row_indices.append(index)

        rewards: list[Optional[float]] = [None] * len(samples)
        if not left_texts:
            return rewards

        similarities = self._embed_pair_similarities(left_texts, right_texts)
        for index, similarity in zip(row_indices, similarities):
            if similarity is not None:
                rewards[index] = clamp01((similarity + 1.0) / 2.0)
        return rewards

    def _embed_pair_similarities(self, left_texts: list[str], right_texts: list[str]) -> list[Optional[float]]:
        embeddings = self.embedding.embed(left_texts + right_texts)
        split = len(left_texts)
        return [cosine_similarity(left, right) for left, right in zip(embeddings[:split], embeddings[split:])]

    def _rewrite_rewards(self, parsed_items: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[Optional[float]]:
        left_texts: list[str] = []
        right_texts: list[str] = []
        row_indices: list[int] = []
        rewards: list[Optional[float]] = [None] * len(samples)

        for index, (parsed, sample) in enumerate(zip(parsed_items, samples)):
            modified_answer = parsed["modified_answer"]
            gt_answer = normalize_text(sample.get("gt_answer") or sample.get("target_modified_answer"))
            if not modified_answer or not gt_answer:
                continue
            left_texts.append(modified_answer)
            right_texts.append(gt_answer)
            row_indices.append(index)

        if not left_texts:
            return rewards

        similarities = self._embed_pair_similarities(left_texts, right_texts)
        for index, similarity in zip(row_indices, similarities):
            if similarity is not None:
                rewards[index] = clamp01((similarity + 1.0) / 2.0)
        return rewards


orms["chaincritic_weighted"] = ChainCriticWeightedORM
