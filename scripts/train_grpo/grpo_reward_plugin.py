"""ms-swift custom reward plugin for ChainCritic GRPO.

Use with:

  swift rlhf --rlhf_type grpo \
    --external_plugins scripts/train_grpo/grpo_reward_plugin.py \
    --reward_funcs chaincritic_weighted

The reward intentionally uses only two components:
1. revision_suggestions vs ground-truth revision_suggestions embedding
   similarity. Reward is 1 when cosine similarity >= threshold, else 0.
2. A verifier LLM judges whether the generated modified_answer is worse than
   the original answer on the given dimension/rubric. WORSE -> 0, BETTER/SAME -> 1.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import re
import time
from typing import Any, Optional
from urllib import request as urllib_request

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from swift.rewards import ORM, orms
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"Failed to import ms-swift reward API: {exc}") from exc


REVISION_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:revision[_ ]suggestions?|suggestions?)\s*[:\uFF1A]\s*"
    r"(.*?)\s*(?=(?:\n\s*)?(?:modified[_ ]answer|revised[_ ]answer)\s*[:\uFF1A]|\Z)"
)
MODIFIED_RE = re.compile(r"(?is)(?:^|\n)\s*(?:modified[_ ]answer|revised[_ ]answer)\s*[:\uFF1A]\s*(.*)$")
SECTION_PATTERNS = {
    "question": re.compile(r"(?im)^\s*Question\s*:\s*"),
    "answer": re.compile(r"(?im)^\s*(?:Candidate\s+Answer|Original\s+Answer|Answer)\s*:\s*"),
    "evaluation_dimension": re.compile(r"(?im)^\s*Evaluation(?:[_ ]+Dimension)?\s*:\s*"),
    "criteria": re.compile(r"(?im)^\s*(?:Score\s+Criteria|Criteria)\s*(?:\(\s*0\s*-\s*5\s*\))?\s*:\s*"),
}

VERIFIER_SYSTEM_PROMPT = """You are a strict evaluator.
Compare the revised answer against the original answer only on the given evaluation dimension and rubric.
Judge whether the revised answer is better, the same, or worse than the original answer.

Important rules:
1. Evaluate only with respect to the provided evaluation dimension and score criteria.
2. Do not reward verbosity unless it improves performance on the rubric.
3. Penalize unsupported claims, incorrect reasoning, or rewrites that drift away from the question.
4. If the revised answer does not clearly improve the original answer on this dimension, return SAME or WORSE.
5. Output exactly one label and nothing else:
BETTER
SAME

WORSE"""


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        pieces = []
        for item in completion:
            if isinstance(item, dict):
                pieces.append(str(item.get("content") or ""))
            else:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(completion or "")


def parse_json_object(raw: str) -> Optional[dict[str, Any]]:
    text = raw.strip()
    candidates = [text]
    fenced = re.search(r"(?is)```(?:json)?\s*(\{.*?\})\s*```", text)
    if fenced:
        candidates.append(fenced.group(1))
    object_match = re.search(r"(?is)\{.*\}", text)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def first_non_empty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def parse_completion(text: Any) -> dict[str, Any]:
    raw = completion_to_text(text).strip().replace("\r\n", "\n")
    payload = parse_json_object(raw)
    if payload is not None:
        revision_suggestions = first_non_empty(
            payload.get("revision_suggestions"),
            payload.get("revision suggestion"),
            payload.get("revision"),
            payload.get("suggestions"),
            payload.get("edit_intent"),
        )
        modified_answer = first_non_empty(
            payload.get("modified_answer"),
            payload.get("modified answer"),
            payload.get("revised_answer"),
            payload.get("revised answer"),
        )
    else:
        revision_match = REVISION_RE.search(raw)
        modified_match = MODIFIED_RE.search(raw)
        revision_suggestions = normalize_text(revision_match.group(1)) if revision_match else ""
        modified_answer = normalize_text(modified_match.group(1)) if modified_match else ""

    return {
        "revision_suggestions": revision_suggestions,
        "modified_answer": modified_answer,
        "raw_text": raw,
        "is_valid": bool(revision_suggestions) and bool(modified_answer),
    }


def extract_messages(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return "", "", ""

    system_content = ""
    user_content = ""
    assistant_content = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role")).lower()
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        content = str(content or "")
        if role == "system" and not system_content:
            system_content = content
        elif role == "user" and not user_content:
            user_content = content
        elif role == "assistant" and not assistant_content:
            assistant_content = content
    return system_content, user_content, assistant_content


def parse_labeled_sections(text: str) -> dict[str, str]:
    matches: list[tuple[int, int, str]] = []
    for key, pattern in SECTION_PATTERNS.items():
        match = pattern.search(text or "")
        if match:
            matches.append((match.start(), match.end(), key))
    matches.sort()

    sections: dict[str, str] = {}
    for index, (_, start_content, key) in enumerate(matches):
        end_content = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        sections[key] = normalize_text(text[start_content:end_content])
    return sections


def enrich_sample_from_messages(sample: dict[str, Any]) -> dict[str, Any]:
    _, user_content, assistant_content = extract_messages(sample)
    if user_content:
        sections = parse_labeled_sections(user_content)
        for key, value in sections.items():
            if value and not normalize_text(sample.get(key)):
                sample[key] = value
        if sections.get("evaluation_dimension") and not normalize_text(sample.get("dimension_name")):
            sample["dimension_name"] = sections["evaluation_dimension"]
        if sections.get("criteria") and not normalize_text(sample.get("criteria_text")):
            sample["criteria_text"] = sections["criteria"]

    if assistant_content:
        parsed = parse_completion(assistant_content)
        if parsed["revision_suggestions"] and not ChainCriticWeightedORM._extract_gt_revision_suggestions(sample):
            sample["target_revision_suggestions"] = parsed["revision_suggestions"]
        if parsed["modified_answer"] and not normalize_text(sample.get("target_modified_answer")):
            sample["target_modified_answer"] = parsed["modified_answer"]

    return sample


def post_json(url: str, payload: dict[str, Any], api_key: str, timeout: float) -> dict[str, Any]:
    req = urllib_request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def batched_cosine_diagonal(left_vectors: list[list[float]], right_vectors: list[list[float]]) -> list[Optional[float]]:
    if len(left_vectors) != len(right_vectors):
        raise ValueError("Embedding batch size mismatch.")
    if not left_vectors:
        return []

    if np is None:
        return [cosine_similarity(left, right) for left, right in zip(left_vectors, right_vectors)]

    try:
        left = np.asarray(left_vectors, dtype=np.float32)
        right = np.asarray(right_vectors, dtype=np.float32)
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            return [cosine_similarity(lv, rv) for lv, rv in zip(left_vectors, right_vectors)]

        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        valid = (left_norm[:, 0] > 0) & (right_norm[:, 0] > 0)
        left = left / np.maximum(left_norm, 1e-12)
        right = right / np.maximum(right_norm, 1e-12)
        similarity_matrix = left @ right.T
        diagonal = np.diag(similarity_matrix)
        return [float(value) if bool(is_valid) else None for value, is_valid in zip(diagonal, valid)]
    except Exception:
        return [cosine_similarity(left, right) for left, right in zip(left_vectors, right_vectors)]


def build_user_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    answer = str(row.get("answer", "")).strip()
    dimension = str(row.get("evaluation_dimension", "")).strip()
    criteria = str(row.get("criteria", "")).strip()
    revised = str(row.get("predicted_modified_answer", "")).strip()

    return f"""Question:
{question}

Original Answer:
{answer}

Evaluation Dimension:
{dimension}

Score Criteria:
{criteria}

Revised Answer:
{revised}

Compare the Revised Answer against the Original Answer for this dimension only.
Return exactly one label:
BETTER
SAME
WORSE"""


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
        response = self._post_with_retries(f"{self.base_url}/embeddings", payload)
        data = sorted(response.get("data") or [], key=lambda item: int(item.get("index", 0)))
        embeddings = [item.get("embedding") for item in data]
        if len(embeddings) != len(texts) or not all(isinstance(item, list) for item in embeddings):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
        return embeddings  # type: ignore[return-value]

    def chat_label(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": 4,
        }
        response = self._post_with_retries(f"{self.base_url}/chat/completions", payload)
        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return normalize_text(message.get("content"))

    def _post_with_retries(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                return post_json(url, payload, self.api_key, self.timeout)
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_sleep * (attempt + 1))
        raise RuntimeError(f"Request failed for {url}: {last_error}")


class ChainCriticWeightedORM(ORM):
    def __init__(self, args=None, **kwargs) -> None:
        super().__init__()
        self.args = args

        self.rs_weight = env_float("CHAINCRITIC_RS_WEIGHT", 0.5)
        self.verifier_weight = env_float("CHAINCRITIC_VERIFIER_WEIGHT", 0.5)
        self.similarity_threshold = env_float("CHAINCRITIC_RS_SIMILARITY_THRESHOLD", 0.8)
        self.invalid_output_penalty = env_float("CHAINCRITIC_INVALID_OUTPUT_PENALTY", -1.0)
        self.verifier_workers = env_int("CHAINCRITIC_VERIFIER_WORKERS", 8)

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
        self.verifier = OpenAICompatibleClient(
            os.getenv("CHAINCRITIC_VERIFIER_BASE_URL", "http://127.0.0.1:8005/v1"),
            os.getenv("CHAINCRITIC_VERIFIER_MODEL", ""),
            api_key,
            timeout,
            retries,
            retry_sleep,
        )
        self.gt_revision_embedding_cache: dict[str, list[float]] = {}

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        parsed_items = [parse_completion(completion) for completion in completions]
        samples = self._build_samples(kwargs, len(parsed_items))

        rs_rewards = self._revision_suggestion_rewards(parsed_items, samples)
        verifier_rewards = self._verifier_rewards(parsed_items, samples)

        rewards: list[float] = []
        for index, parsed in enumerate(parsed_items):
            if not parsed["is_valid"]:
                rewards.append(self.invalid_output_penalty)
                continue

            components: list[tuple[float, float]] = []
            if rs_rewards[index] is not None:
                components.append((self.rs_weight, float(rs_rewards[index])))
            if verifier_rewards[index] is not None:
                components.append((self.verifier_weight, float(verifier_rewards[index])))

            if not components:
                rewards.append(self.invalid_output_penalty)
                continue

            total_weight = sum(weight for weight, _ in components)
            reward = sum(weight * value for weight, value in components) / total_weight if total_weight > 0 else 0.0
            rewards.append(float(clamp01(reward)))
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
            sample = enrich_sample_from_messages(sample)
            samples.append(sample)
        return samples

    def _revision_suggestion_rewards(
        self, parsed_items: list[dict[str, Any]], samples: list[dict[str, Any]]
    ) -> list[Optional[float]]:
        predicted_texts: list[str] = []
        gt_texts: list[str] = []
        row_indices: list[int] = []
        rewards: list[Optional[float]] = [None] * len(samples)

        for index, (parsed, sample) in enumerate(zip(parsed_items, samples)):
            predicted = parsed["revision_suggestions"]
            gt_revision = self._extract_gt_revision_suggestions(sample)
            if not predicted or not gt_revision:
                continue
            predicted_texts.append(predicted)
            gt_texts.append(gt_revision)
            row_indices.append(index)

        if not predicted_texts:
            return rewards

        predicted_embeddings = self.embedding.embed(predicted_texts)
        gt_embeddings = self._get_cached_gt_embeddings(gt_texts)
        similarities = batched_cosine_diagonal(predicted_embeddings, gt_embeddings)

        for index, similarity in zip(row_indices, similarities):
            if similarity is not None:
                rewards[index] = 1.0 if similarity >= self.similarity_threshold else 0.0
        return rewards

    def _get_cached_gt_embeddings(self, gt_texts: list[str]) -> list[list[float]]:
        missing = []
        seen = set()
        for text in gt_texts:
            if text not in self.gt_revision_embedding_cache and text not in seen:
                missing.append(text)
                seen.add(text)

        if missing:
            embeddings = self.embedding.embed(missing)
            for text, embedding in zip(missing, embeddings):
                self.gt_revision_embedding_cache[text] = embedding

        return [self.gt_revision_embedding_cache[text] for text in gt_texts]

    @staticmethod
    def _extract_gt_revision_suggestions(sample: dict[str, Any]) -> str:
        return first_non_empty(
            sample.get("gt_revision_suggestions"),
            sample.get("target_revision_suggestions"),
            sample.get("reference_revision_suggestions"),
            sample.get("revision_suggestions"),
            sample.get("edit_intent"),
            sample.get("target_edit_intent"),
        )

    def _verifier_rewards(
        self, parsed_items: list[dict[str, Any]], samples: list[dict[str, Any]]
    ) -> list[Optional[float]]:
        rewards: list[Optional[float]] = [None] * len(samples)
        tasks: list[tuple[int, str]] = []

        for index, (parsed, sample) in enumerate(zip(parsed_items, samples)):
            modified_answer = parsed["modified_answer"]
            question = first_non_empty(sample.get("question"), sample.get("prompt"), sample.get("instruction"))
            original_answer = first_non_empty(sample.get("answer"), sample.get("candidate_answer"), sample.get("response"))
            dimension = first_non_empty(sample.get("evaluation_dimension"), sample.get("dimension_name"), sample.get("name"))
            criteria = first_non_empty(sample.get("criteria"), sample.get("criteria_text"), sample.get("score_criteria"))

            if not modified_answer or not question or not original_answer or not dimension or not criteria:
                continue

            row = {
                "question": question,
                "answer": original_answer,
                "evaluation_dimension": dimension,
                "criteria": criteria,
                "predicted_modified_answer": modified_answer,
            }
            tasks.append((index, build_user_prompt(row)))

        if not tasks:
            return rewards

        max_workers = max(1, min(self.verifier_workers, len(tasks)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.verifier.chat_label, VERIFIER_SYSTEM_PROMPT, user_prompt): index
                for index, user_prompt in tasks
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    label = self._parse_verifier_label(future.result())
                except Exception:
                    label = None
                if label is None:
                    continue
                rewards[index] = 0.0 if label == "WORSE" else 1.0

        return rewards

    @staticmethod
    def _parse_verifier_label(text: Any) -> Optional[str]:
        normalized = normalize_text(text).upper()
        match = re.search(r"\b(BETTER|SAME|WORSE)\b", normalized)
        return match.group(1) if match else None


orms["chaincritic_weighted"] = ChainCriticWeightedORM
