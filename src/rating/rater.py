"""LLM Rating Module - Score answers based on evaluation dimensions."""

from typing import Optional, Dict, Any, List
import json
import re
import time

try:
    from ..common.llm_client import ask_llm
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.llm_client import ask_llm


class LLMRater:
    """LLM rater - Score answers with reasons based on evaluation dimensions."""

    def __init__(self, provider: Optional[str] = None, max_score: int = 5):
        """Initialize LLM rater.

        Args:
            provider: LLM provider name. If not provided, uses default env provider.
            max_score: Maximum score for each dimension.
        """
        self.provider = provider
        self.max_score = max_score

    def rate_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Rate a single sample.

        Args:
            sample: Sample containing 'question', 'answer', and 'evaluation_dimensions'.

        Returns:
            Parsed rating result dict.
        """
        dimensions = sample.get("evaluation_dimensions", [])
        if not dimensions:
            return {"ratings": []}

        prompt = self._build_single_sample_prompt(sample, dimensions)
        result_text = ask_llm(prompt=prompt, provider=self.provider)
        return self._parse_rating_result(result_text)

    def rate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        sample_size: Optional[int] = None,
        delay_between_requests: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Rate a dataset of samples.

        Args:
            dataset: Dataset list with evaluation_dimensions for each sample.
            sample_size: Number of samples to rate. If None, rate all.
            delay_between_requests: Delay in seconds between requests.

        Returns:
            List of rating results with scores and reasons.
        """
        if sample_size is not None:
            samples = dataset[:min(sample_size, len(dataset))]
        else:
            samples = dataset

        results_with_samples = []

        for idx, sample in enumerate(samples, 1):
            try:
                dimensions = sample.get("evaluation_dimensions", [])
                if not dimensions:
                    results_with_samples.append({
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "evaluation_dimensions": [],
                        "ratings": [],
                        "overall_score": None
                    })
                    continue

                rating_result = self.rate_single_sample(sample)

                if "parse_error" in rating_result or "raw_output" in rating_result:
                    results_with_samples.append({
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "evaluation_dimensions": dimensions,
                        "ratings": [],
                        "overall_score": None,
                        "error": rating_result.get("parse_error", "Failed to parse LLM output"),
                        "raw_output": rating_result.get("raw_output", "")
                    })
                    continue

                ratings = self._extract_ratings_from_result(rating_result, dimensions)
                overall_score = self._compute_overall_score(ratings)

                results_with_samples.append({
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "evaluation_dimensions": dimensions,
                    "ratings": ratings,
                    "overall_score": overall_score
                })

                print(f"  Sample {idx}/{len(samples)} rating completed, {len(ratings)} dimensions scored")

                if idx < len(samples) and delay_between_requests > 0:
                    time.sleep(delay_between_requests)

            except Exception as e:
                print(f"Warning: Sample {idx} rating failed: {e}")
                results_with_samples.append({
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "evaluation_dimensions": sample.get("evaluation_dimensions", []),
                    "ratings": [],
                    "overall_score": None,
                    "error": str(e)
                })
                continue

        return results_with_samples

    def _build_single_sample_prompt(
        self,
        sample: Dict[str, Any],
        dimensions: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for a single sample rating."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        dimension_lines = []
        # Build dimension lines with full score criteria
        for idx, dim in enumerate(dimensions, 1):
            dim_name = dim.get("dimension_name", "")
            category = dim.get("category", "")
            criteria = dim.get("full_score_criteria", "")
            dimension_lines.append(
                f"{idx}. {dim_name} (category: {category})\n"
                f"   Full-score criteria: {criteria}"
            )

        dimensions_text = "\n".join(dimension_lines) if dimension_lines else "(No dimensions provided)"

        json_example = """{
  "ratings": [
    {
      "dimension_name": "Dimension Name",
      "score": 4,
      "reason": "Brief reason for the score",
      "category": "subjective"
    }
  ]
}"""

        prompt = f"""You are a mind chain assessment expert. Please evaluate each answer that includes a mind chain based on the provided dimensions and the full score criteria for that dimension.

Question: {question}

Answer: {answer}

Evaluation Dimensions:
{dimensions_text}

Requirements:
- Score each dimension from 0 to {self.max_score}
- The score should be an integer or a decimal number within the range
- Provide a concise reason for each score based on the full-score criteria
- If a dimension is not applicable, give a low score and explain why
- Use the provided dimension names exactly

Output JSON format:
{json_example}

Output only JSON, do not add any other text.
"""
        return prompt

    def _parse_rating_result(self, result_text: str) -> Dict[str, Any]:
        """Parse rating result returned by LLM."""
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        list_match = re.search(r'\[[\s\S]*\]', result_text)
        if list_match:
            try:
                return {"ratings": json.loads(list_match.group())}
            except json.JSONDecodeError:
                pass

        return {
            "raw_output": result_text,
            "parse_error": "Failed to parse JSON, returning raw output"
        }

    def _extract_ratings_from_result(
        self,
        rating_result: Dict[str, Any],
        dimensions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract ratings list from result and align with dimensions."""
        if "parse_error" in rating_result or "raw_output" in rating_result:
            return []

        ratings = rating_result.get("ratings", [])
        if not isinstance(ratings, list):
            return []

        dim_by_name = {}
        dim_by_lower = {}
        for dim in dimensions:
            name = dim.get("dimension_name", "")
            if name:
                dim_by_name[name] = dim
                dim_by_lower[name.lower()] = dim

        extracted = []
        seen = set()

        for item in ratings:
            if not isinstance(item, dict):
                continue
            dim_name = item.get("dimension_name") or item.get("name") or ""
            if not dim_name:
                continue

            dim_info = dim_by_name.get(dim_name) or dim_by_lower.get(dim_name.lower())
            category = item.get("category", "") or (dim_info.get("category", "") if dim_info else "")
            criteria = dim_info.get("full_score_criteria", "") if dim_info else ""
            score = self._normalize_score(item.get("score"))
            reason = item.get("reason", "")

            extracted.append({
                "dimension_name": dim_name,
                "score": score,
                "reason": reason,
                "category": category,
                "full_score_criteria": criteria
            })

            seen.add(dim_name)

        for dim in dimensions:
            dim_name = dim.get("dimension_name", "")
            if dim_name and dim_name not in seen:
                extracted.append({
                    "dimension_name": dim_name,
                    "score": None,
                    "reason": "No rating returned by LLM. Please provide reasoning for low score.",
                    "category": dim.get("category", ""),
                    "full_score_criteria": dim.get("full_score_criteria", "")
                })

        return extracted

    def _normalize_score(self, value: Any) -> Optional[float]:
        """Normalize score value to float if possible."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _compute_overall_score(self, ratings: List[Dict[str, Any]]) -> Optional[float]:
        """Compute average score across dimensions."""
        scores: List[float] = []
        for r in ratings:
            s = r.get("score")
            if isinstance(s, (int, float)):
                scores.append(float(s))

        if not scores:
            return None
        return sum(scores) / len(scores)
