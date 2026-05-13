#!/usr/bin/env python
"""Build SFT jsonl for score+reason+revision training.

Target message format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

Each input sample is expanded by dimension when per-dimension results exist:
- one row with N dimensions -> N training items
- user: instruction + Q + A + single dimension + complete 0-5 score criteria
- assistant: score + reason + revision_suggestions + modified answer

Edit prompts below directly when you need a new dataset prompt style.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ===================== Editable config =====================
INPUT_ROOT = Path("datasets/filled_iteration_train_outputs/datasets")
DIMENSIONS_ROOT = Path("datasets")
OUTPUT_JSONL = Path("datasets/filled_iteration_train_outputs/score_reason_revision_sft.jsonl")
OUTPUT_TRAIN_JSONL = Path("datasets/filled_iteration_train_outputs/score_reason_revision_sft_train.jsonl")
OUTPUT_TEST_JSONL = Path("datasets/filled_iteration_train_outputs/score_reason_revision_sft_test.jsonl")
SINGLE_DIM_DIR_NAME = "single_dim"
DIMENSIONS_DIR_NAME = "dimensions"
# test : train = 1 : 50
TEST_RATIO = 1 / 51

SPLIT_SEED = 42
SHUFFLE_BEFORE_SPLIT = True

SYSTEM_PROMPT = (
    "You are an expert evaluator and answer rewriter. Evaluate a candidate answer "
    "to a given question under the specified evaluation dimension and 0-5 scoring "
    "criteria. Identify its strengths, errors, missing parts, and why it does or does "
    "not meet the highest-score criterion. Then provide actionable revision suggestions "
    "and rewrite the answer into a stronger response optimized for the same dimension. "
    "Use only the provided question, candidate answer, evaluation dimension, and scoring "
    "criteria. Do not add unsupported facts. Return strict JSON only, with exactly these "
    "keys: score, reason, revision_suggestions, modified_answer."
)

USER_TASK_INSTRUCTION = (
    "### Task Description:\n"
    "You are given a question, a candidate answer, one evaluation dimension, "
    "and the complete 0-5 scoring criteria for that dimension.\n"
    "\n"
    "Your task is to evaluate and improve the candidate answer using only the provided "
    "question, candidate answer, evaluation dimension, and scoring criteria.\n"
    "\n"
    "Follow these steps:\n"
    "1. Assign one integer score from 0 to 5 according to the provided scoring criteria.\n"
    "2. Provide a concise but concrete reason for the score. The reason should explain "
    "the candidate answer's strengths, weaknesses, missing elements, unsupported claims, "
    "or failures to satisfy the evaluation dimension. When identifying a specific "
    "problem, explicitly mark it with the prefix \"error:\".\n"
    "3. Based on the scoring reason, provide actionable revision suggestions explaining "
    "how the candidate answer should be revised to better satisfy the score-5 criterion.\n"
    "4. Rewrite the candidate answer into a stronger modified_answer for the same "
    "question and the same evaluation dimension. The modified_answer should address the "
    "identified errors, follow the revision suggestions, and avoid adding unsupported "
    "facts.\n"
    "5. Return strict JSON only with exactly this schema:\n"
    "{\"score\": <int>, \"reason\": \"...\", \"revision_suggestions\": \"...\", "
    "\"modified_answer\": \"...\"}"
)

USER_TEMPLATE = (
    "{instruction}\n\n"
    "Question:\n{question}\n\n"
    "Candidate Answer:\n{answer}\n\n"
    "Evaluation Dimension:\n{dimension_name}\n\n"
    "Score Criteria (0-5):\n{score_criteria}"
)

ASSISTANT_FORMAT = "json"

# If True, put existing reason into user prompt as extra context.
# Default False to avoid leaking label target into input.
INCLUDE_REASON_IN_USER = False
USER_REASON_TEMPLATE = "\n\nReference Information (Optional):\n{input_reason}"

# If True, skip samples with empty score/reason/modified_answer.
REQUIRE_NON_EMPTY_LABELS = True
REQUIRE_NON_EMPTY_REVISION_SUGGESTIONS = False

# Force complete 0-5 score criteria by default. This matches the new SFT input
# contract: {Q+A+D+Cs}.
REQUIRE_NON_EMPTY_CRITERIA = True
REQUIRE_COMPLETE_SCORE_CRITERIA = True
# ==========================================================

CANDIDATE_LIST_KEYS = ("items", "data", "samples", "records", "results")

# New: support multiple possible criteria field names.
CRITERIA_FIELD_CANDIDATES = (
    "full_score_criteria",
    "criteria",
    "dimension_criteria",
    "full_criteria",
    "scoring_criteria",
    "rubric",
    "dimension_rubric",
    "description",
)

SCORE_CRITERIA_FIELD_CANDIDATES = (
    "score_criteria",
    "0-5_Criteria",
    "criteria_by_score",
    "criteria_map",
    "rubric_by_score",
    "scoring_rubric",
)

SCORE_FIELD_CANDIDATES = (
    "modified_score",
    "predicted_score",
    "Score",
    "score",
)

REASON_FIELD_CANDIDATES = (
    "predicted_reason",
    "Reason",
    "reason",
    "feedback",
    "rationale",
)

REVISION_SUGGESTIONS_FIELD_CANDIDATES = (
    "revision_suggestions",
    "edit_intent",
    "edit_suggestion",
    "modification_suggestion",
    "modification_suggestions",
    "rewrite_suggestions",
)

MODIFIED_ANSWER_FIELD_CANDIDATES = (
    "predicted_modified_answer",
    "modified_answer",
    "Modify_ans",
    "modify_ans",
    "rewritten_answer",
    "rewrite",
    "revised_answer",
    "better_answer",
)

SINGLE_TO_DIMENSIONS_SUFFIX_RULES = (
    (
        "_dimensions_score_correct_items_iterative_score_single_dim.json",
        "_dimensions.json",
    ),
    (
        "_dimensions_score_correct_items_iteration_score_direct.json",
        "_dimensions.json",
    ),
    (
        "_dimensions_rating_correct_items_iteration_score_direct.json",
        "_dimensions.json",
    ),
    (
        "_score_correct_items_iterative_score_single_dim.json",
        "_dimensions.json",
    ),
    (
        "_score_correct_items_iterative_score_single_dim.json",
        ".json",
    ),
    (
        "_score_correct_items_iteration_score_direct.json",
        "_dimensions.json",
    ),
    (
        "_score_correct_items_iteration_score_direct.json",
        ".json",
    ),
    (
        "_rating_correct_items_iteration_score_direct.json",
        "_dimensions.json",
    ),
    (
        "_rating_correct_items_iteration_score_direct.json",
        ".json",
    ),
)


@dataclass
class FileStats:
    file: str
    rows_total: int = 0
    rows_skipped_no_per_dim: int = 0
    dims_total: int = 0
    dims_emitted: int = 0
    dims_skipped_missing_label: int = 0
    dims_skipped_missing_name: int = 0
    dims_skipped_missing_criteria: int = 0
    dims_skipped_incomplete_score_criteria: int = 0


@dataclass
class DatasetDimensionsIndex:
    by_file_qa: dict[str, dict[tuple[str, str], dict[str, str]]]
    by_file_q: dict[str, dict[str, dict[str, str]]]
    by_qa: dict[tuple[str, str], dict[str, str]]
    by_q: dict[str, dict[str, str]]


class ExternalCriteriaLookup:
    def __init__(self, *, dimensions_root: Path, dimensions_dir_name: str) -> None:
        self.dimensions_root = dimensions_root
        self.dimensions_dir_name = dimensions_dir_name
        self._dataset_cache: dict[str, DatasetDimensionsIndex] = {}

    def resolve(
        self,
        *,
        dataset_name: str,
        single_dim_file_name: str,
        question: str,
        answer: str,
        dimension_name: str,
    ) -> tuple[str, str]:
        if _is_empty(dataset_name) or _is_empty(question) or _is_empty(dimension_name):
            return "", "missing"

        index = self._load_dataset_index(dataset_name)
        dim_key = _normalize_key(dimension_name)
        question_key = str(question or "").strip()
        answer_key = str(answer or "").strip()
        qa_key = (question_key, answer_key)

        for candidate_name in _candidate_dimension_file_names(single_dim_file_name):
            file_qa_map = index.by_file_qa.get(candidate_name)
            if file_qa_map:
                value = file_qa_map.get(qa_key, {}).get(dim_key)
                if not _is_empty(value):
                    return value, f"external.file_qa:{candidate_name}"

            file_q_map = index.by_file_q.get(candidate_name)
            if file_q_map:
                value = file_q_map.get(question_key, {}).get(dim_key)
                if not _is_empty(value):
                    return value, f"external.file_q:{candidate_name}"

        value = index.by_qa.get(qa_key, {}).get(dim_key)
        if not _is_empty(value):
            return value, "external.dataset_qa"

        value = index.by_q.get(question_key, {}).get(dim_key)
        if not _is_empty(value):
            return value, "external.dataset_q"

        return "", "missing"

    def _load_dataset_index(self, dataset_name: str) -> DatasetDimensionsIndex:
        cached = self._dataset_cache.get(dataset_name)
        if cached is not None:
            return cached

        index = DatasetDimensionsIndex(by_file_qa={}, by_file_q={}, by_qa={}, by_q={})
        dataset_dir = self.dimensions_root / dataset_name / self.dimensions_dir_name
        if not dataset_dir.is_dir():
            self._dataset_cache[dataset_name] = index
            return index

        for path in sorted(dataset_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            try:
                rows = _load_records(path)
            except Exception as exc:
                print(f"[WARN] skip dimensions file {path} | {exc}")
                continue

            file_qa_map = index.by_file_qa.setdefault(path.name, {})
            file_q_map = index.by_file_q.setdefault(path.name, {})

            for row in rows:
                if not isinstance(row, dict):
                    continue

                question = str(row.get("question") or "").strip()
                answer = str(row.get("answer") or "").strip()
                if _is_empty(question):
                    continue

                dim_meta_map = _build_dimension_meta_map(row)
                for dim_key, meta in dim_meta_map.items():
                    dim_name = str(meta.get("dimension_name") or "").strip()
                    dim_stub = {"dimension_name": dim_name}
                    criteria, _ = _resolve_dimension_criteria(dim_stub, meta, row)
                    if _is_empty(criteria):
                        continue

                    qa_key = (question, answer)
                    file_qa_map.setdefault(qa_key, {}).setdefault(dim_key, criteria)
                    file_q_map.setdefault(question, {}).setdefault(dim_key, criteria)
                    index.by_qa.setdefault(qa_key, {}).setdefault(dim_key, criteria)
                    index.by_q.setdefault(question, {}).setdefault(dim_key, criteria)

        self._dataset_cache[dataset_name] = index
        return index


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _candidate_dimension_file_names(single_dim_file_name: str) -> list[str]:
    candidates: list[str] = []

    def _add_candidate(name: str) -> None:
        if _is_empty(name):
            return
        if name not in candidates:
            candidates.append(name)

        path = Path(name)
        suffix = path.suffix.lower()
        if suffix == ".json":
            alt_name = f"{path.stem}.jsonl"
            if alt_name not in candidates:
                candidates.append(alt_name)
        elif suffix == ".jsonl":
            alt_name = f"{path.stem}.json"
            if alt_name not in candidates:
                candidates.append(alt_name)

    _add_candidate(single_dim_file_name)
    for old_suffix, new_suffix in SINGLE_TO_DIMENSIONS_SUFFIX_RULES:
        if single_dim_file_name.endswith(old_suffix):
            _add_candidate(
                single_dim_file_name[: -len(old_suffix)] + new_suffix
            )
    return candidates


def _infer_dataset_name(path: Path, input_root: Path) -> str:
    try:
        relative_path = path.relative_to(input_root)
    except ValueError:
        return ""
    if not relative_path.parts:
        return ""
    return relative_path.parts[0]


def _extract_rows(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in CANDIDATE_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return None


def _load_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _extract_rows(payload)
    if rows is None:
        raise ValueError("Unsupported JSON structure: cannot find record list.")
    return rows


def _iter_single_dim_files(root: Path, single_dim_dir_name: str) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.parent.name != single_dim_dir_name:
            continue
        if p.suffix.lower() in {".json", ".jsonl"}:
            files.append(p)
    return sorted(files)


def _first_non_empty_from_dict(data: dict[str, Any], field_names: tuple[str, ...]) -> tuple[Any, str | None]:
    for field in field_names:
        if field in data and not _is_empty(data.get(field)):
            return data.get(field), field
    return None, None


def _build_dimension_meta_map(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collect per-dimension auxiliary fields from multiple possible locations."""
    dim_map: dict[str, dict[str, Any]] = {}

    def _ingest(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            dim_name = item.get("dimension_name") or item.get("name")
            key = _normalize_key(dim_name)
            if not key:
                continue
            meta = dim_map.setdefault(key, {})
            if dim_name and "dimension_name" not in meta:
                meta["dimension_name"] = dim_name

            for field in CRITERIA_FIELD_CANDIDATES:
                if field in item and field not in meta and not _is_empty(item.get(field)):
                    meta[field] = item.get(field)

            for field in SCORE_CRITERIA_FIELD_CANDIDATES:
                if field in item and field not in meta and not _is_empty(item.get(field)):
                    meta[field] = item.get(field)

            for field in SCORE_FIELD_CANDIDATES:
                if field in item and field not in meta and not _is_empty(item.get(field)):
                    meta[field] = item.get(field)

            for field in REASON_FIELD_CANDIDATES:
                if field in item and field not in meta and not _is_empty(item.get(field)):
                    meta[field] = item.get(field)

            for field in REVISION_SUGGESTIONS_FIELD_CANDIDATES:
                if field in item and field not in meta and not _is_empty(item.get(field)):
                    meta[field] = item.get(field)

            for field in MODIFIED_ANSWER_FIELD_CANDIDATES:
                if field in item and field not in meta and not _is_empty(item.get(field)):
                    meta[field] = item.get(field)

    _ingest(sample.get("per_dimension_results"))
    _ingest(sample.get("modified_dimension_scores"))
    _ingest(sample.get("evaluation_dimensions"))
    _ingest(sample.get("ratings"))
    _ingest(sample.get("final_ratings"))
    _ingest(sample.get("dimension_definitions"))
    _ingest(sample.get("dimensions"))
    return dim_map


def _format_score(score: Any) -> str:
    if isinstance(score, float):
        return str(score)
    return str(score)


def _parse_score_criteria_text(text: str) -> dict[str, str]:
    criteria: dict[str, str] = {}
    current_score: str | None = None
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:score\s*)?([0-5])\s*[:：]\s*(.+)$", line, flags=re.I)
        if match:
            current_score = match.group(1)
            criteria[current_score] = match.group(2).strip()
        elif current_score is not None:
            criteria[current_score] = f"{criteria[current_score]} {line}".strip()
    return criteria


def _normalize_score_criteria_from_value(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        criteria: dict[str, str] = {}
        for score in range(6):
            score_key = str(score)
            text = value.get(score_key)
            if text is None:
                text = value.get(score)
            if not _is_empty(text):
                criteria[score_key] = str(text).strip()
        return criteria

    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return _normalize_score_criteria_from_value(parsed)
        return _parse_score_criteria_text(value)

    return {}


def _normalize_score_criteria_from_source(
    source: dict[str, Any],
    *,
    full_score_criteria: str = "",
) -> dict[str, str]:
    criteria: dict[str, str] = {}

    for field in SCORE_CRITERIA_FIELD_CANDIDATES:
        if field not in source or _is_empty(source.get(field)):
            continue
        criteria.update(_normalize_score_criteria_from_value(source.get(field)))

    for score in range(6):
        score_key = str(score)
        value, _ = _first_non_empty_from_dict(
            source,
            (
                f"criteria_{score}",
                f"score_{score}",
                f"score_{score}_criteria",
                f"criterion_{score}",
            ),
        )
        if not _is_empty(value):
            criteria[score_key] = str(value).strip()

    if not _is_empty(full_score_criteria) and _is_empty(criteria.get("5")):
        criteria["5"] = str(full_score_criteria).strip()

    return {str(score): criteria.get(str(score), "") for score in range(6)}


def _has_complete_score_criteria(criteria: dict[str, str]) -> bool:
    return all(not _is_empty(criteria.get(str(score))) for score in range(6))


def _format_score_criteria(criteria: dict[str, str]) -> str:
    return "\n".join(
        f"Score {score}: {str(criteria.get(str(score), '')).strip()}"
        for score in range(6)
        if not _is_empty(criteria.get(str(score)))
    )


def _resolve_score_criteria(
    dim: dict[str, Any],
    meta: dict[str, Any],
    row: dict[str, Any],
    *,
    full_score_criteria: str,
) -> tuple[dict[str, str], str]:
    for source_name, source in (("dim", dim), ("meta", meta), ("row", row)):
        criteria = _normalize_score_criteria_from_source(
            source,
            full_score_criteria=full_score_criteria if source_name == "row" else "",
        )
        if any(not _is_empty(v) for v in criteria.values()):
            return criteria, source_name

    if not _is_empty(full_score_criteria):
        return _normalize_score_criteria_from_source(
            {},
            full_score_criteria=full_score_criteria,
        ), "full_score_only"

    return {}, "missing"


def _build_revision_suggestions(
    *,
    explicit_value: Any,
    reason: str,
    dimension_name: str,
    score_criteria_text: str,
) -> str:
    if not _is_empty(explicit_value):
        return str(explicit_value).strip()
    if _is_empty(reason):
        return ""
    return (
        "Revise the answer to address the specific issue identified in the reason: "
        f"{reason} Ensure the revision satisfies the score-5 requirement for "
        f"'{dimension_name}' under the provided score criteria."
    )


def _build_user_content(
    *,
    question: str,
    answer: str,
    dimension_name: str,
    score_criteria_text: str,
    input_reason: str,
    include_reason_in_user: bool,
) -> str:
    user_text = USER_TEMPLATE.format(
        instruction=USER_TASK_INSTRUCTION,
        question=question,
        answer=answer,
        dimension_name=dimension_name,
        score_criteria=score_criteria_text,
    )
    if include_reason_in_user and not _is_empty(input_reason):
        user_text += USER_REASON_TEMPLATE.format(input_reason=input_reason)
    return user_text


def _build_assistant_content(
    *,
    score: Any,
    reason: str,
    revision_suggestions: str,
    modified_answer: str,
) -> str:
    payload = {
        "score": int(score) if str(score).strip().isdigit() else _format_score(score),
        "reason": reason,
        "revision_suggestions": revision_suggestions,
        "modified_answer": modified_answer,
    }
    if ASSISTANT_FORMAT == "json":
        return json.dumps(payload, ensure_ascii=False)

    return (
        f"Score: {payload['score']}\n"
        f"Reason: {reason}\n"
        f"Revision Suggestions: {revision_suggestions}\n"
        f"Modified Answer: {modified_answer}"
    )



def _resolve_dimension_criteria(
    dim: dict[str, Any],
    meta: dict[str, Any],
    row: dict[str, Any],
) -> tuple[str, str]:
    """
    Resolve criteria and return:
    - criteria_text
    - source_tag
    """
    value, field = _first_non_empty_from_dict(dim, CRITERIA_FIELD_CANDIDATES)
    if not _is_empty(value):
        return str(value).strip(), f"dim.{field}"

    value, field = _first_non_empty_from_dict(meta, CRITERIA_FIELD_CANDIDATES)
    if not _is_empty(value):
        return str(value).strip(), f"meta.{field}"

    # Optional fallback: row-level criteria maps / definitions
    for row_field in (
        "criteria_map",
        "dimension_criteria_map",
        "rubric_map",
        "dimension_rubric_map",
    ):
        obj = row.get(row_field)
        if isinstance(obj, dict):
            dim_name = str(dim.get("dimension_name") or dim.get("name") or "").strip()
            if dim_name:
                # direct key
                if dim_name in obj and not _is_empty(obj[dim_name]):
                    return str(obj[dim_name]).strip(), f"row.{row_field}[direct]"
                # normalized key match
                dim_key = _normalize_key(dim_name)
                for k, v in obj.items():
                    if _normalize_key(k) == dim_key and not _is_empty(v):
                        return str(v).strip(), f"row.{row_field}[normalized]"

    return "", "missing"


def _convert_one_file(
    path: Path,
    *,
    dataset_name: str,
    include_reason_in_user: bool,
    require_non_empty_labels: bool,
    require_non_empty_revision_suggestions: bool,
    require_non_empty_criteria: bool,
    require_complete_score_criteria: bool,
    missing_criteria_sources: Counter[str],
    missing_criteria_examples: dict[str, list[str]],
    found_criteria_sources: Counter[str],
    external_criteria_lookup: ExternalCriteriaLookup | None,
) -> tuple[list[dict[str, Any]], FileStats]:
    rows = _load_records(path)
    stats = FileStats(file=str(path), rows_total=len(rows))
    out: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        per_dim = row.get("per_dimension_results")
        if not isinstance(per_dim, list) or not per_dim:
            if first_non_empty := str(row.get("dimension_name") or row.get("evaluation_dimension") or "").strip():
                per_dim = [{**row, "dimension_name": first_non_empty}]
            else:
                stats.rows_skipped_no_per_dim += 1
                continue

        dim_meta_map = _build_dimension_meta_map(row)

        for dim in per_dim:
            if not isinstance(dim, dict):
                continue
            stats.dims_total += 1

            dim_name = str(dim.get("dimension_name") or dim.get("name") or "").strip()
            if not dim_name:
                stats.dims_skipped_missing_name += 1
                continue

            dim_key = _normalize_key(dim_name)
            meta = dim_meta_map.get(dim_key, {})

            dim_criteria, criteria_source = _resolve_dimension_criteria(dim, meta, row)
            if _is_empty(dim_criteria) and external_criteria_lookup is not None:
                dim_criteria, criteria_source = external_criteria_lookup.resolve(
                    dataset_name=dataset_name,
                    single_dim_file_name=path.name,
                    question=question,
                    answer=answer,
                    dimension_name=dim_name,
                )
            score_criteria, score_criteria_source = _resolve_score_criteria(
                dim,
                meta,
                row,
                full_score_criteria=dim_criteria,
            )
            if _is_empty(dim_criteria) and not _is_empty(score_criteria.get("5")):
                dim_criteria = score_criteria["5"]
                criteria_source = f"score_criteria_5:{score_criteria_source}"

            if _is_empty(dim_criteria):
                missing_criteria_sources[criteria_source] += 1
                if len(missing_criteria_examples[path.name]) < 10:
                    missing_criteria_examples[path.name].append(dim_name)
            else:
                found_criteria_sources[criteria_source] += 1

            score_criteria_text = _format_score_criteria(score_criteria)

            score, _ = _first_non_empty_from_dict(dim, SCORE_FIELD_CANDIDATES)
            if _is_empty(score):
                score, _ = _first_non_empty_from_dict(meta, SCORE_FIELD_CANDIDATES)
            if _is_empty(score):
                score, _ = _first_non_empty_from_dict(row, SCORE_FIELD_CANDIDATES)

            reason, _ = _first_non_empty_from_dict(dim, REASON_FIELD_CANDIDATES)
            if _is_empty(reason):
                reason, _ = _first_non_empty_from_dict(meta, REASON_FIELD_CANDIDATES)
            if _is_empty(reason):
                reason, _ = _first_non_empty_from_dict(row, REASON_FIELD_CANDIDATES)
            reason = str(reason or "").strip()

            revision_suggestions, _ = _first_non_empty_from_dict(dim, REVISION_SUGGESTIONS_FIELD_CANDIDATES)
            if _is_empty(revision_suggestions):
                revision_suggestions, _ = _first_non_empty_from_dict(meta, REVISION_SUGGESTIONS_FIELD_CANDIDATES)
            if _is_empty(revision_suggestions):
                revision_suggestions, _ = _first_non_empty_from_dict(row, REVISION_SUGGESTIONS_FIELD_CANDIDATES)
            revision_suggestions = _build_revision_suggestions(
                explicit_value=revision_suggestions,
                reason=reason,
                dimension_name=dim_name,
                score_criteria_text=score_criteria_text,
            )

            modified_answer, _ = _first_non_empty_from_dict(dim, MODIFIED_ANSWER_FIELD_CANDIDATES)
            if _is_empty(modified_answer):
                modified_answer, _ = _first_non_empty_from_dict(meta, MODIFIED_ANSWER_FIELD_CANDIDATES)
            if _is_empty(modified_answer):
                modified_answer, _ = _first_non_empty_from_dict(row, MODIFIED_ANSWER_FIELD_CANDIDATES)
            if _is_empty(modified_answer):
                modified_answer = row.get("modified_answer")
            if _is_empty(modified_answer):
                modified_answer = answer
            modified_answer = str(modified_answer or "").strip()

            if require_non_empty_labels and (
                _is_empty(score) or _is_empty(reason) or _is_empty(modified_answer)
            ):
                stats.dims_skipped_missing_label += 1
                continue

            if require_non_empty_revision_suggestions and _is_empty(revision_suggestions):
                stats.dims_skipped_missing_label += 1
                continue

            if require_non_empty_criteria and _is_empty(score_criteria_text):
                stats.dims_skipped_missing_criteria += 1
                continue

            if require_complete_score_criteria and not _has_complete_score_criteria(score_criteria):
                stats.dims_skipped_incomplete_score_criteria += 1
                missing_criteria_sources[f"incomplete_score_criteria:{score_criteria_source}"] += 1
                continue

            user_content = _build_user_content(
                question=question,
                answer=answer,
                dimension_name=dim_name,
                score_criteria_text=score_criteria_text,
                input_reason=reason,
                include_reason_in_user=include_reason_in_user,
            )
            assistant_content = _build_assistant_content(
                score=score,
                reason=reason,
                revision_suggestions=revision_suggestions,
                modified_answer=modified_answer,
            )

            out.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )
            stats.dims_emitted += 1

    return out, stats


def _write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_train_test(
    records: list[dict[str, Any]],
    *,
    test_ratio: float,
    seed: int,
    shuffle: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    items = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    total = len(items)
    test_size = int(total * test_ratio)
    if test_size <= 0:
        test_size = 1 if total > 1 else 0
    if test_size >= total:
        test_size = max(total - 1, 0)

    split_at = total - test_size
    train_records = items[:split_at]
    test_records = items[split_at:]
    return train_records, test_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert single_dim iteration data into chat-format SFT jsonl."
    )
    parser.add_argument("--input-root", default=str(INPUT_ROOT), help="Root folder to scan.")
    parser.add_argument(
        "--dimensions-root",
        default=str(DIMENSIONS_ROOT),
        help="Root folder that contains per-dataset dimensions folders.",
    )
    parser.add_argument("--output", default=str(OUTPUT_JSONL), help="Merged output jsonl path.")
    parser.add_argument(
        "--train-output",
        default=str(OUTPUT_TRAIN_JSONL),
        help="Train split output jsonl path.",
    )
    parser.add_argument(
        "--test-output",
        default=str(OUTPUT_TEST_JSONL),
        help="Test split output jsonl path.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Test split ratio in (0,1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="Random seed for shuffling before split.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before train/test split.",
    )
    parser.add_argument(
        "--single-dim-dir-name",
        default=SINGLE_DIM_DIR_NAME,
        help="Only files under folders with this name are used.",
    )
    parser.add_argument(
        "--dimensions-dir-name",
        default=DIMENSIONS_DIR_NAME,
        help="Per-dataset folder name that stores source dimensions files.",
    )
    parser.add_argument(
        "--include-reason-in-user",
        action="store_true",
        default=INCLUDE_REASON_IN_USER,
        help="Append source reason into user content.",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Do not skip rows with empty score/reason/modified_answer.",
    )
    parser.add_argument(
        "--require-revision-suggestions",
        action="store_true",
        default=REQUIRE_NON_EMPTY_REVISION_SUGGESTIONS,
        help="Skip rows that do not have explicit revision_suggestions/edit_intent.",
    )
    parser.add_argument(
        "--allow-empty-criteria",
        action="store_true",
        help="Do not skip rows with empty criteria.",
    )
    parser.add_argument(
        "--allow-incomplete-score-criteria",
        action="store_true",
        help="Allow samples without a complete 0-5 score criteria map.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Debug mode: process at most N files.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    dimensions_root = Path(args.dimensions_root)
    output_path = Path(args.output)
    train_output_path = Path(args.train_output)
    test_output_path = Path(args.test_output)
    require_non_empty_labels = not args.allow_empty_labels
    require_non_empty_criteria = not args.allow_empty_criteria
    require_complete_score_criteria = not args.allow_incomplete_score_criteria

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    external_criteria_lookup: ExternalCriteriaLookup | None = None
    if dimensions_root.is_dir():
        external_criteria_lookup = ExternalCriteriaLookup(
            dimensions_root=dimensions_root,
            dimensions_dir_name=args.dimensions_dir_name,
        )
    else:
        print(f"[WARN] dimensions root not found, external lookup disabled: {dimensions_root}")

    files = _iter_single_dim_files(input_root, args.single_dim_dir_name)
    if args.max_files is not None:
        files = files[: max(0, args.max_files)]

    if not files:
        print("No single_dim files found.")
        return

    all_records: list[dict[str, Any]] = []
    all_stats: list[FileStats] = []

    missing_criteria_sources: Counter[str] = Counter()
    found_criteria_sources: Counter[str] = Counter()
    missing_criteria_examples: dict[str, list[str]] = defaultdict(list)

    for fp in files:
        try:
            dataset_name = _infer_dataset_name(fp, input_root)
            records, stats = _convert_one_file(
                fp,
                dataset_name=dataset_name,
                include_reason_in_user=args.include_reason_in_user,
                require_non_empty_labels=require_non_empty_labels,
                require_non_empty_revision_suggestions=args.require_revision_suggestions,
                require_non_empty_criteria=require_non_empty_criteria,
                require_complete_score_criteria=require_complete_score_criteria,
                missing_criteria_sources=missing_criteria_sources,
                missing_criteria_examples=missing_criteria_examples,
                found_criteria_sources=found_criteria_sources,
                external_criteria_lookup=external_criteria_lookup,
            )
            all_records.extend(records)
            all_stats.append(stats)
            print(
                f"[OK] {fp.name} | rows={stats.rows_total}, dims={stats.dims_total}, "
                f"emitted={stats.dims_emitted}, missing_label={stats.dims_skipped_missing_label}, "
                f"missing_criteria={stats.dims_skipped_missing_criteria}, "
                f"incomplete_score_criteria={stats.dims_skipped_incomplete_score_criteria}"
            )
        except Exception as exc:
            print(f"[SKIP] {fp} | {exc}")

    _write_jsonl(all_records, output_path)
    train_records, test_records = _split_train_test(
        all_records,
        test_ratio=args.test_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    _write_jsonl(train_records, train_output_path)
    _write_jsonl(test_records, test_output_path)

    total_rows = sum(s.rows_total for s in all_stats)
    total_dims = sum(s.dims_total for s in all_stats)
    total_emitted = sum(s.dims_emitted for s in all_stats)
    total_no_per_dim = sum(s.rows_skipped_no_per_dim for s in all_stats)
    total_missing_label = sum(s.dims_skipped_missing_label for s in all_stats)
    total_missing_name = sum(s.dims_skipped_missing_name for s in all_stats)
    total_missing_criteria = sum(s.dims_skipped_missing_criteria for s in all_stats)
    total_incomplete_score_criteria = sum(s.dims_skipped_incomplete_score_criteria for s in all_stats)

    print("\n===== Summary =====")
    print(f"files={len(all_stats)}")
    print(f"rows={total_rows}")
    print(f"rows_no_per_dim={total_no_per_dim}")
    print(f"dims_total={total_dims}")
    print(f"dims_emitted={total_emitted}")
    print(f"dims_missing_label={total_missing_label}")
    print(f"dims_missing_name={total_missing_name}")
    print(f"dims_missing_criteria={total_missing_criteria}")
    print(f"dims_incomplete_score_criteria={total_incomplete_score_criteria}")
    print(f"merged_output={output_path}")
    print(f"train_output={train_output_path} | train_count={len(train_records)}")
    print(f"test_output={test_output_path} | test_count={len(test_records)}")

    print("\n===== Criteria Source Stats (Found) =====")
    if found_criteria_sources:
        for source, cnt in found_criteria_sources.most_common():
            print(f"{source}: {cnt}")
    else:
        print("No non-empty criteria found from any source.")

    print("\n===== Criteria Source Stats (Missing) =====")
    if missing_criteria_sources:
        for source, cnt in missing_criteria_sources.most_common():
            print(f"{source}: {cnt}")
    else:
        print("No missing criteria cases detected.")

    print("\n===== Missing Criteria Examples (first up to 10 dims per file) =====")
    if missing_criteria_examples:
        for file_name, dims in sorted(missing_criteria_examples.items()):
            print(f"{file_name}: {dims}")
    else:
        print("No missing criteria examples recorded.")


if __name__ == "__main__":
    main()
