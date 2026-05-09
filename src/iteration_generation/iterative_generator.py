# """Iterative CoT generation module - Improve answers dimension by dimension."""

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Any, Callable, Dict, List, Optional, Tuple
# import json
# import os
# import re
# import threading
# import time
# from pathlib import Path
# from urllib.error import URLError
# from urllib.request import Request, urlopen

# from openai import OpenAI

# try:
#     from ..common.llm_client import (
#         DEFAULT_SYSTEM_MESSAGE,
#         PROVIDER_CONFIGS,
#         ask_llm,
#     )
#     from ..common.data_loader import DatasetLoader
#     from ..rating.rater import LLMRater
# except ImportError:
#     import sys
#     from pathlib import Path

#     sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#     from src.common.llm_client import (
#         DEFAULT_SYSTEM_MESSAGE,
#         PROVIDER_CONFIGS,
#         ask_llm,
#     )
#     from src.common.data_loader import DatasetLoader
#     from src.rating.rater import LLMRater


# class IterativeGenerator:
#     """Iteratively improve chain-of-thought answers using dimension ratings."""

#     def __init__(
#         self,
#         provider: Optional[str] = None,
#         max_score: float = 5.0,
#         local_host: str = "localhost",
#         local_ports: Optional[List[int]] = None,
#         local_model: Optional[str] = None,
#         request_timeout: float = 120.0,
#         prefer_local: bool = True,
#     ):
#         """Initialize the generator.

#         Args:
#             provider: LLM provider name. If not provided, auto-detects from env.
#             max_score: Maximum score for each dimension.
#             local_host: Host of local OpenAI-compatible servers.
#             local_ports: Port list for local servers, e.g. [8000, 8001, 8002, 8003].
#             local_model: Override local model id. If None, fetch from /v1/models.
#             request_timeout: Timeout for local requests in seconds.
#             prefer_local: Try local endpoints first if configured.
#         """
#         self.provider = provider
#         self.max_score = max_score
#         self.local_host = local_host
#         self.local_ports = list(local_ports or [])
#         self.local_model = local_model
#         self.request_timeout = request_timeout
#         self.prefer_local = prefer_local

#         self._local_clients: Dict[int, OpenAI] = {}
#         self._local_model_by_port: Dict[int, str] = {}
#         self._active_ports: List[int] = []
#         self._rr_idx = 0
#         self._rr_lock = threading.Lock()
#         self._local_ready = False
#         self._local_init_lock = threading.Lock()
#         self._local_fallback_notice_printed = False

#     def generate_single_sample(
#         self,
#         sample: Dict[str, Any],
#         min_score_to_improve: Optional[float] = None,
#         max_dimensions: Optional[int] = None,
#         order: str = "score_asc",
#         include_no_score: bool = False,
#         force_all_dimensions: bool = False,
#         rewrite_full_score_dimensions: bool = False,
#         rewrite_no_score_dimensions: bool = False,
#         temperature: float = 0.3,
#         delay_between_requests: float = 0.5,
#         keep_intermediate: bool = False,
#         rerate_after_iteration: bool = False,
#         rating_temperature: Optional[float] = None,
#         enable_dimension_reflection: bool = True,
#         reflection_temperature: Optional[float] = None,
#     ) -> Dict[str, Any]:
#         """Iteratively improve a single sample."""
#         question = sample.get("question", "")
#         answer = sample.get("answer", "")
#         evaluation_dimensions = sample.get("evaluation_dimensions", [])
#         effective_rating_temperature = rating_temperature if rating_temperature is not None else temperature
#         effective_reflection_temperature = (
#             reflection_temperature if reflection_temperature is not None else effective_rating_temperature
#         )
#         effective_force_all_dimensions = force_all_dimensions or enable_dimension_reflection

#         dims = self._collect_dimension_ratings(sample)
#         dims_to_apply = self._select_dimensions(
#             dims,
#             min_score_to_improve=min_score_to_improve,
#             max_dimensions=max_dimensions,
#             order=order,
#             include_no_score=include_no_score,
#             force_all_dimensions=effective_force_all_dimensions,
#         )

#         result = dict(sample)
#         if not question or not answer:
#             result["iterated_answer"] = answer
#             result["final_answer"] = answer
#             result["iteration_trace"] = []
#             result["dimension_results"] = []
#             result["iteration_status"] = "invalid_input"
#             result["iteration_error"] = "Missing question or answer."
#             self._attach_final_scores(
#                 result=result,
#                 question=question,
#                 final_answer=answer,
#                 evaluation_dimensions=evaluation_dimensions,
#                 rerate_after_iteration=rerate_after_iteration,
#                 rating_temperature=effective_rating_temperature,
#             )
#             return result

#         if not dims_to_apply:
#             result["iterated_answer"] = answer
#             result["final_answer"] = answer
#             result["iteration_trace"] = []
#             result["dimension_results"] = []
#             result["iteration_status"] = "skipped_no_dimensions"
#             self._attach_final_scores(
#                 result=result,
#                 question=question,
#                 final_answer=answer,
#                 evaluation_dimensions=evaluation_dimensions,
#                 rerate_after_iteration=rerate_after_iteration,
#                 rating_temperature=effective_rating_temperature,
#             )
#             return result

#         base_answer = answer
#         trace: List[Dict[str, Any]] = []
#         representative_answer = answer

#         for idx, dim in enumerate(dims_to_apply, 1):
#             original_dim_score = self._normalize_score(dim.get("score"))
#             original_dim_reason = (dim.get("reason") or "").strip()
#             effective_dim = dict(dim)
#             reflection_result: Dict[str, Any] = {}
#             revision_generation_error: str = ""
#             should_rewrite = False
#             revised = base_answer
#             modified_score = original_dim_score
#             modified_reason_raw = original_dim_reason

#             if enable_dimension_reflection:
#                 try:
#                     reflection_result = self._reflect_dimension_rating(
#                         question=question,
#                         answer=base_answer,
#                         dim=effective_dim,
#                         temperature=effective_reflection_temperature,
#                     )
#                     if reflection_result.get("is_reasonable") is False:
#                         revised_answer = str(reflection_result.get("revised_answer", "") or "").strip()
#                         revised_score = self._normalize_score(reflection_result.get("revised_score"))
#                         revised_reason = str(reflection_result.get("revised_reason", "") or "").strip()

#                         if revised_score is None:
#                             revised_score = original_dim_score
#                         if not revised_reason:
#                             revised_reason = original_dim_reason

#                         should_rewrite = (
#                             bool(revised_answer)
#                             or revised_score != original_dim_score
#                             or revised_reason != original_dim_reason
#                         )
#                         if should_rewrite:
#                             revised = revised_answer or base_answer
#                             representative_answer = revised
#                             modified_score = revised_score
#                             modified_reason_raw = revised_reason
#                             if modified_score is not None:
#                                 effective_dim["score"] = modified_score
#                             effective_dim["reason"] = modified_reason_raw
#                         else:
#                             revision_generation_error = (
#                                 "Reflection marked score as unreasonable but returned no usable revised payload."
#                             )
#                 except Exception as exc:
#                     reflection_result = {"error": str(exc)}

#             trace_item = {
#                 "dimension_name": effective_dim.get("dimension_name", ""),
#                 "score": original_dim_score,
#                 "reason": original_dim_reason,
#                 "category": effective_dim.get("category", ""),
#                 "full_score_criteria": effective_dim.get("full_score_criteria", ""),
#                 "input_answer": base_answer,
#                 "input_score": original_dim_score,
#                 "input_reason": original_dim_reason,
#                 "modified_answer": revised,
#                 "modified_score": modified_score,
#                 "modified_reason": "",
#                 "modified_category": effective_dim.get("category", ""),
#                 "modified_full_score_criteria": effective_dim.get("full_score_criteria", ""),
#                 "revision_applied": should_rewrite,
#                 "reflection_enabled": enable_dimension_reflection,
#                 "reflection_is_reasonable": reflection_result.get("is_reasonable"),
#                 "reflection_reason": reflection_result.get("reason", ""),
#                 "reflection_suggested_score": self._normalize_score(reflection_result.get("revised_score")),
#                 "reflection_suggested_reason": reflection_result.get("revised_reason", ""),
#                 "reflection_rerated_input_score": None,
#                 "reflection_rerated_input_reason": "",
#                 "force_rewrite_due_reflection": (
#                     should_rewrite and reflection_result.get("is_reasonable") is False
#                 ),
#             }
#             if reflection_result.get("error"):
#                 trace_item["reflection_error"] = reflection_result.get("error")
#             if revision_generation_error:
#                 trace_item["revision_generation_error"] = revision_generation_error

#             trace_item["modified_reason_raw"] = modified_reason_raw
#             trace_item["modified_reason"] = self._build_progress_reason(
#                 original_score=original_dim_score,
#                 original_reason=original_dim_reason,
#                 modified_score=modified_score,
#                 modified_reason=modified_reason_raw,
#                 reflection_is_reasonable=reflection_result.get("is_reasonable"),
#                 reflection_reason=(reflection_result.get("reason") or "").strip(),
#                 revision_applied=should_rewrite,
#             )
#             if revision_generation_error:
#                 trace_item["modified_reason"] = (
#                     f"{trace_item['modified_reason']} "
#                     f"Revision generation failed: {revision_generation_error}."
#                 ).strip()

#             if keep_intermediate:
#                 trace_item["before"] = base_answer
#                 trace_item["after"] = revised

#             trace.append(trace_item)

#             if idx < len(dims_to_apply) and delay_between_requests > 0 and enable_dimension_reflection:
#                 time.sleep(delay_between_requests)

#         # Independent per-dimension mode has no single global merged answer.
#         result["iterated_answer"] = answer
#         result["final_answer"] = answer
#         result["representative_answer"] = representative_answer
#         result["iteration_trace"] = trace
#         result["dimension_results"] = trace
#         result["iteration_status"] = "iterated"
#         result["iteration_meta"] = {
#             "min_score_to_improve": self.max_score if min_score_to_improve is None else min_score_to_improve,
#             "max_dimensions": max_dimensions,
#             "order": order,
#             "include_no_score": include_no_score,
#             "force_all_dimensions": effective_force_all_dimensions,
#             "rewrite_full_score_dimensions": rewrite_full_score_dimensions,
#             "rewrite_no_score_dimensions": rewrite_no_score_dimensions,
#             "enable_dimension_reflection": enable_dimension_reflection,
#             "reflection_temperature": effective_reflection_temperature,
#             "provider": self.provider or "auto",
#             "local_ports": self.local_ports,
#             "revision_mode": "independent_per_dimension",
#             "input_answer_policy": "always_original_answer",
#             "aggregate_answer_mode": "no_aggregation_use_per_dimension_results",
#         }
#         if rerate_after_iteration:
#             result["final_rating_note"] = (
#                 "rerate_after_iteration is ignored in independent_per_dimension mode; "
#                 "final_ratings come from per-dimension results."
#             )
#         self._attach_trace_final_scores(result=result, trace=trace)

#         return result

#     def generate_dataset(
#         self,
#         input_file: str,
#         output_file: Optional[str] = None,
#         sample_size: Optional[int] = None,
#         min_score_to_improve: Optional[float] = None,
#         max_dimensions: Optional[int] = None,
#         order: str = "score_asc",
#         include_no_score: bool = False,
#         force_all_dimensions: bool = False,
#         rewrite_full_score_dimensions: bool = False,
#         rewrite_no_score_dimensions: bool = False,
#         temperature: float = 0.3,
#         delay_between_requests: float = 0.5,
#         keep_intermediate: bool = False,
#         num_workers: int = 1,
#         rerate_after_iteration: bool = False,
#         rating_temperature: Optional[float] = None,
#         enable_dimension_reflection: bool = False,
#         reflection_temperature: Optional[float] = None,
#         output_mode: str = "simplified",
#         progress_cb: Optional[Callable[[int, int], None]] = None,
#     ) -> str:
#         """Iteratively improve a dataset."""
#         loader = DatasetLoader()
#         dataset = loader.load_from_json(input_file)

#         if sample_size is not None:
#             dataset = dataset[: min(sample_size, len(dataset))]

#         total = len(dataset)
#         print(f"Loaded {total} samples from {input_file}")
#         if progress_cb:
#             progress_cb(0, total)

#         if output_file is None:
#             input_path = Path(input_file)
#             provider_suffix = f"_{self.provider}" if self.provider else "_auto"
#             output_file = str(input_path.parent / f"{input_path.stem}_iterated{provider_suffix}{input_path.suffix}")

#         if self.prefer_local and self.local_ports:
#             self._ensure_local_clients()
#             print(
#                 f"Local endpoint mode: host={self.local_host}, "
#                 f"configured_ports={self.local_ports}, active_ports={self._active_ports}"
#             )

#         if num_workers <= 1:
#             results, successful, failed = self._run_sequential(
#                 dataset=dataset,
#                 min_score_to_improve=min_score_to_improve,
#                 max_dimensions=max_dimensions,
#                 order=order,
#                 include_no_score=include_no_score,
#                 force_all_dimensions=force_all_dimensions,
#                 rewrite_full_score_dimensions=rewrite_full_score_dimensions,
#                 rewrite_no_score_dimensions=rewrite_no_score_dimensions,
#                 temperature=temperature,
#                 delay_between_requests=delay_between_requests,
#                 keep_intermediate=keep_intermediate,
#                 rerate_after_iteration=rerate_after_iteration,
#                 rating_temperature=rating_temperature,
#                 enable_dimension_reflection=enable_dimension_reflection,
#                 reflection_temperature=reflection_temperature,
#                 progress_cb=progress_cb,
#             )
#         else:
#             results, successful, failed = self._run_parallel(
#                 dataset=dataset,
#                 min_score_to_improve=min_score_to_improve,
#                 max_dimensions=max_dimensions,
#                 order=order,
#                 include_no_score=include_no_score,
#                 force_all_dimensions=force_all_dimensions,
#                 rewrite_full_score_dimensions=rewrite_full_score_dimensions,
#                 rewrite_no_score_dimensions=rewrite_no_score_dimensions,
#                 temperature=temperature,
#                 delay_between_requests=delay_between_requests,
#                 keep_intermediate=keep_intermediate,
#                 num_workers=num_workers,
#                 rerate_after_iteration=rerate_after_iteration,
#                 rating_temperature=rating_temperature,
#                 enable_dimension_reflection=enable_dimension_reflection,
#                 reflection_temperature=reflection_temperature,
#                 progress_cb=progress_cb,
#             )

#         formatted_results = self._format_output_items(results, output_mode=output_mode)

#         output_path = Path(output_file)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         self._save_json_or_jsonl(output_path, formatted_results)

#         print("\n" + "=" * 60)
#         print("Iteration completed!")
#         print(f"  Successful: {successful}")
#         print(f"  Failed: {failed}")
#         print(f"  Output file: {output_file}")
#         print("=" * 60)

#         return output_file

#     def _format_output_items(
#         self,
#         items: List[Dict[str, Any]],
#         output_mode: str,
#     ) -> List[Dict[str, Any]]:
#         """Format output records before saving."""
#         mode = (output_mode or "simplified").strip().lower()
#         if mode == "full":
#             return items
#         if mode != "simplified":
#             raise ValueError(f"Unsupported output_mode: {output_mode}. Use 'simplified' or 'full'.")
#         return [self._to_simplified_item(item) for item in items]

#     def _to_simplified_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
#         """Keep only question/answer plus per-dimension scoring summary."""
#         per_dimension_results: List[Dict[str, Any]] = []
#         trace = item.get("iteration_trace", [])
#         if isinstance(trace, list):
#             for step in trace:
#                 if not isinstance(step, dict):
#                     continue
#                 step_score = self._normalize_score(step.get("modified_score"))
#                 if step_score is None:
#                     step_score = self._normalize_score(step.get("score"))
#                 step_reason = (
#                     step.get("modified_reason")
#                     or step.get("reason")
#                     or step.get("modified_rating_error")
#                     or ""
#                 ).strip()
#                 per_dimension_results.append({
#                     "dimension_name": step.get("dimension_name", ""),
#                     "modified_answer": step.get("modified_answer") or step.get("after") or "",
#                     "modified_score": step_score,
#                     "reason": step_reason,
#                     "full_score_criteria": (
#                         step.get("modified_full_score_criteria", "")
#                         or step.get("full_score_criteria", "")
#                         or ""
#                     ),
#                 })

#         if not per_dimension_results:
#             final_ratings = item.get("final_ratings")
#             if not isinstance(final_ratings, list):
#                 final_ratings = item.get("ratings", [])
#             is_failed_item = str(item.get("iteration_status", "")).lower() == "failed"
#             fallback_answer = item.get("iterated_answer") or item.get("answer") or ""
#             fallback_error = (item.get("iteration_error") or "").strip()
#             if isinstance(final_ratings, list):
#                 for rating in final_ratings:
#                     if not isinstance(rating, dict):
#                         continue
#                     reason = str(rating.get("reason", "") or "").strip()
#                     if is_failed_item and fallback_error:
#                         reason = f"{reason} [iteration_error: {fallback_error}]".strip()
#                     per_dimension_results.append({
#                         "dimension_name": rating.get("dimension_name") or rating.get("name") or "",
#                         "modified_answer": fallback_answer if is_failed_item else "",
#                         "modified_score": self._normalize_score(rating.get("score")),
#                         "reason": reason,
#                         "full_score_criteria": rating.get("full_score_criteria", ""),
#                     })

#         return {
#             "question": item.get("question", ""),
#             "answer": item.get("answer", ""),
#             "per_dimension_results": per_dimension_results,
#         }

#     def _run_sequential(
#         self,
#         dataset: List[Dict[str, Any]],
#         min_score_to_improve: Optional[float],
#         max_dimensions: Optional[int],
#         order: str,
#         include_no_score: bool,
#         force_all_dimensions: bool,
#         rewrite_full_score_dimensions: bool,
#         rewrite_no_score_dimensions: bool,
#         temperature: float,
#         delay_between_requests: float,
#         keep_intermediate: bool,
#         rerate_after_iteration: bool,
#         rating_temperature: Optional[float],
#         enable_dimension_reflection: bool = False,
#         reflection_temperature: Optional[float] = None,
#         progress_cb: Optional[Callable[[int, int], None]] = None,
#     ) -> Tuple[List[Dict[str, Any]], int, int]:
#         results: List[Dict[str, Any]] = []
#         successful = 0
#         failed = 0
#         total = len(dataset)
#         done = 0

#         for idx, sample in enumerate(dataset, 1):
#             try:
#                 updated = self.generate_single_sample(
#                     sample=sample,
#                     min_score_to_improve=min_score_to_improve,
#                     max_dimensions=max_dimensions,
#                     order=order,
#                     include_no_score=include_no_score,
#                     force_all_dimensions=force_all_dimensions,
#                     rewrite_full_score_dimensions=rewrite_full_score_dimensions,
#                     rewrite_no_score_dimensions=rewrite_no_score_dimensions,
#                     temperature=temperature,
#                     delay_between_requests=delay_between_requests,
#                     keep_intermediate=keep_intermediate,
#                     rerate_after_iteration=rerate_after_iteration,
#                     rating_temperature=rating_temperature,
#                     enable_dimension_reflection=enable_dimension_reflection,
#                     reflection_temperature=reflection_temperature,
#                 )
#                 results.append(updated)
#                 successful += 1
#             except Exception as exc:
#                 failed += 1
#                 results.append(self._build_error_item(sample, str(exc)))
#             finally:
#                 done += 1
#                 if progress_cb:
#                     progress_cb(done, total)

#         return results, successful, failed

#     def _run_parallel(
#         self,
#         dataset: List[Dict[str, Any]],
#         min_score_to_improve: Optional[float],
#         max_dimensions: Optional[int],
#         order: str,
#         include_no_score: bool,
#         force_all_dimensions: bool,
#         rewrite_full_score_dimensions: bool,
#         rewrite_no_score_dimensions: bool,
#         temperature: float,
#         delay_between_requests: float,
#         keep_intermediate: bool,
#         num_workers: int,
#         rerate_after_iteration: bool,
#         rating_temperature: Optional[float],
#         enable_dimension_reflection: bool = False,
#         reflection_temperature: Optional[float] = None,
#         progress_cb: Optional[Callable[[int, int], None]] = None,
#     ) -> Tuple[List[Dict[str, Any]], int, int]:
#         total = len(dataset)
#         results: List[Optional[Dict[str, Any]]] = [None] * total
#         successful = 0
#         failed = 0

#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = {
#                 executor.submit(
#                     self.generate_single_sample,
#                     sample,
#                     min_score_to_improve,
#                     max_dimensions,
#                     order,
#                     include_no_score,
#                     force_all_dimensions,
#                     rewrite_full_score_dimensions,
#                     rewrite_no_score_dimensions,
#                     temperature,
#                     delay_between_requests,
#                     keep_intermediate,
#                     rerate_after_iteration,
#                     rating_temperature,
#                     enable_dimension_reflection,
#                     reflection_temperature,
#                 ): idx
#                 for idx, sample in enumerate(dataset)
#             }

#             completed = 0
#             for future in as_completed(futures):
#                 idx = futures[future]
#                 sample = dataset[idx]
#                 try:
#                     results[idx] = future.result()
#                     successful += 1
#                 except Exception as exc:
#                     results[idx] = self._build_error_item(sample, str(exc))
#                     failed += 1
#                 completed += 1
#                 if progress_cb:
#                     progress_cb(completed, total)

#         finalized = [item for item in results if item is not None]
#         return finalized, successful, failed

#     def _build_error_item(self, sample: Dict[str, Any], error_message: str) -> Dict[str, Any]:
#         error_item = dict(sample)
#         error_item["iterated_answer"] = sample.get("answer", "")
#         error_item["final_answer"] = sample.get("answer", "")
#         error_item["iteration_trace"] = []
#         error_item["dimension_results"] = []
#         error_item["iteration_status"] = "failed"
#         error_item["iteration_error"] = error_message
#         return error_item

#     def _attach_trace_final_scores(self, result: Dict[str, Any], trace: List[Dict[str, Any]]) -> None:
#         """Attach final scores from per-dimension independent trace."""
#         final_ratings: List[Dict[str, Any]] = []
#         numeric_scores: List[float] = []

#         for step in trace:
#             if not isinstance(step, dict):
#                 continue
#             score = self._normalize_score(step.get("modified_score"))
#             if score is not None:
#                 numeric_scores.append(score)
#             final_ratings.append({
#                 "dimension_name": step.get("dimension_name", ""),
#                 "score": score,
#                 "reason": step.get("modified_reason", ""),
#                 "category": step.get("modified_category", "") or step.get("category", ""),
#                 "full_score_criteria": (
#                     step.get("modified_full_score_criteria", "") or step.get("full_score_criteria", "")
#                 ),
#             })

#         result["final_ratings"] = final_ratings
#         result["final_overall_score"] = (
#             sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
#         )
#         result["final_rating_mode"] = "from_per_dimension_results"

#     def _attach_final_scores(
#         self,
#         result: Dict[str, Any],
#         question: str,
#         final_answer: str,
#         evaluation_dimensions: Any,
#         rerate_after_iteration: bool,
#         rating_temperature: float,
#     ) -> None:
#         """Attach final per-dimension scores after iteration."""
#         if not isinstance(evaluation_dimensions, list) or not evaluation_dimensions:
#             result["final_ratings"] = []
#             result["final_overall_score"] = None
#             result["final_rating_mode"] = "no_dimensions"
#             return

#         if rerate_after_iteration:
#             try:
#                 final_ratings, final_overall = self._rerate_single_answer(
#                     question=question,
#                     answer=final_answer,
#                     evaluation_dimensions=evaluation_dimensions,
#                     temperature=rating_temperature,
#                 )
#                 result["final_ratings"] = final_ratings
#                 result["final_overall_score"] = final_overall
#                 result["final_rating_mode"] = "rerated"
#                 return
#             except Exception as exc:
#                 result["final_ratings"] = []
#                 result["final_overall_score"] = None
#                 result["final_rating_mode"] = "rerated_failed"
#                 result["final_rating_error"] = str(exc)
#                 return

#         existing_ratings = result.get("ratings", [])
#         existing_overall = result.get("overall_score")
#         if isinstance(existing_ratings, list):
#             result["final_ratings"] = existing_ratings
#         else:
#             result["final_ratings"] = []
#         result["final_overall_score"] = existing_overall if isinstance(existing_overall, (int, float)) else None
#         result["final_rating_mode"] = "copied_original"

#     def _rerate_single_answer(
#         self,
#         question: str,
#         answer: str,
#         evaluation_dimensions: List[Dict[str, Any]],
#         temperature: float,
#     ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
#         """Re-score one iterated answer with the same dimensions."""
#         max_score = int(self.max_score) if self.max_score >= 1 else 5
#         rater = LLMRater(provider=self.provider, max_score=max_score)
#         sample = {"question": question, "answer": answer}
#         prompt = rater._build_single_sample_prompt(sample, evaluation_dimensions)
#         result_text = self._call_llm(prompt=prompt, temperature=temperature)
#         parsed = rater._parse_rating_result(result_text)
#         if "parse_error" in parsed:
#             raw_output = parsed.get("raw_output", "")
#             parse_error = parsed.get("parse_error", "Failed to parse rating JSON")
#             raise RuntimeError(f"{parse_error}. raw_output={raw_output[:300]}")
#         ratings = rater._extract_ratings_from_result(parsed, evaluation_dimensions)
#         overall_score = rater._compute_overall_score(ratings)
#         return ratings, overall_score

#     def _rerate_single_dimension_answer(
#         self,
#         question: str,
#         answer: str,
#         dim: Dict[str, Any],
#         temperature: float,
#     ) -> Dict[str, Any]:
#         """Re-score answer on one target dimension only."""
#         name = dim.get("dimension_name") or dim.get("name") or ""
#         if not name:
#             return {}

#         single_dimension = {
#             "dimension_name": name,
#             "category": dim.get("category", ""),
#             "full_score_criteria": dim.get("full_score_criteria", ""),
#         }
#         ratings, _ = self._rerate_single_answer(
#             question=question,
#             answer=answer,
#             evaluation_dimensions=[single_dimension],
#             temperature=temperature,
#         )
#         if ratings and isinstance(ratings[0], dict):
#             return ratings[0]
#         return {}

#     def _reflect_dimension_rating(
#         self,
#         question: str,
#         answer: str,
#         dim: Dict[str, Any],
#         temperature: float,
#     ) -> Dict[str, Any]:
#         """One-step audit+revision in a single prompt."""
#         dim_name = str(dim.get("dimension_name") or dim.get("name") or "").strip()
#         dim_cat = str(dim.get("category") or "").strip()
#         dim_score = self._normalize_score(dim.get("score"))
#         dim_reason = str(dim.get("reason") or "").strip() or "(no reason provided)"
#         criteria = str(dim.get("full_score_criteria") or "").strip() or "(missing)"
#         score_text = "null" if dim_score is None else str(dim_score)

#         name_l = dim_name.lower()
#         cat_l = dim_cat.lower()
#         math_critical = any(k in name_l for k in [
#             "accuracy", "calculation", "correct", "method", "logic", "consisten", "reasoning", "valid"
#         ]) or any(k in cat_l for k in [
#             "accuracy", "calculation", "correct", "logic", "reasoning", "math"
#         ])

#         if math_critical:
#             revision_policy = (
#                 "When revision is needed, you may rewrite from scratch, but every numeric statement must be correct and "
#                 "internally consistent. If the original answer has any mathematical inconsistency, fix it fully."
#             )
#         else:
#             revision_policy = (
#                 "When revision is needed, prefer minimal edits that improve this target dimension while preserving the "
#                 "correct final result and overall structure."
#             )

#         prompt = (
#             "Instruction:\n"
#             "You are a strict evaluation auditor and answer editor.\n"
#             "In ONE step, decide whether the current score/reason is reasonable for this dimension, and if not, revise the answer and re-score.\n\n"
#             f"Question (Q):\n{question}\n\n"
#             f"Answer (A):\n{answer}\n\n"
#             "Dimension:\n"
#             f"- dimension_name: {dim_name}\n"
#             f"- category: {dim_cat}\n"
#             f"- current_score: {score_text}\n"
#             f"- current_reason: {dim_reason}\n"
#             f"- full_score_criteria: {criteria}\n\n"
#             "Revision policy:\n"
#             f"- {revision_policy}\n\n"
#             "Output JSON only with exactly these fields:\n"
#             "{\n"
#             '  "is_reasonable": true or false,\n'
#             '  "reason": "Audit explanation referencing criteria and current answer",\n'
#             '  "revised_score": number or null,\n'
#             '  "revised_reason": "Corrected scoring reason",\n'
#             '  "revised_answer": "Revised answer text"\n'
#             "}\n"
#             "Strict rules:\n"
#             "1) If is_reasonable=true, keep score/reason/answer unchanged from current input.\n"
#             "2) If is_reasonable=false, provide corrected revised_score, revised_reason, and revised_answer.\n"
#             f"3) revised_score must be within [0, {self.max_score}].\n"
#             "4) revised_answer must be mathematically correct and logically consistent.\n"
#             "5) Do not output markdown fences or extra text."
#         )
#         result_text = self._call_llm(prompt=prompt, temperature=temperature)
#         parsed = self._parse_json_object(result_text)
#         if not isinstance(parsed, dict):
#             raise RuntimeError("Reflection output is not a JSON object.")

#         is_reasonable = parsed.get("is_reasonable")
#         if isinstance(is_reasonable, str):
#             lowered = is_reasonable.strip().lower()
#             if lowered in {"true", "yes", "1"}:
#                 is_reasonable = True
#             elif lowered in {"false", "no", "0"}:
#                 is_reasonable = False
#         if not isinstance(is_reasonable, bool):
#             raise RuntimeError("Reflection JSON missing boolean field: is_reasonable")

#         revised_score = self._normalize_score(parsed.get("revised_score"))
#         if revised_score is None:
#             revised_score = self._normalize_score(parsed.get("suggested_score"))
#         if revised_score is not None:
#             revised_score = max(0.0, min(float(self.max_score), revised_score))

#         revised_reason = str(
#             parsed.get("revised_reason")
#             or parsed.get("suggested_reason")
#             or ""
#         ).strip()
#         revised_answer = str(parsed.get("revised_answer", "") or "").strip()

#         if is_reasonable is True:
#             if revised_score is None:
#                 revised_score = dim_score
#             if not revised_reason:
#                 revised_reason = dim_reason
#             if not revised_answer:
#                 revised_answer = answer

#         if is_reasonable is False and revised_score is None and not revised_reason and not revised_answer:
#             raise RuntimeError(
#                 "Reflection JSON marked score as unreasonable but missing revised_score/revised_reason/revised_answer."
#             )

#         return {
#             "is_reasonable": is_reasonable,
#             "reason": str(parsed.get("reason", "")).strip(),
#             "revised_score": revised_score,
#             "revised_reason": revised_reason,
#             "revised_answer": revised_answer,
#             # Keep compatibility with historical field names.
#             "suggested_score": revised_score,
#             "suggested_reason": revised_reason,
#         }

#     def _parse_json_object(self, text: str) -> Dict[str, Any]:
#         """Extract the first JSON object from text."""
#         raw = str(text or "").strip()
#         if not raw:
#             raise RuntimeError("Empty LLM output.")

#         try:
#             obj = json.loads(raw)
#             if isinstance(obj, dict):
#                 return obj
#         except Exception:
#             pass

#         match = re.search(r"\{[\s\S]*\}", raw)
#         if not match:
#             raise RuntimeError(f"No JSON object found. raw_output={raw[:300]}")
#         try:
#             obj = json.loads(match.group(0))
#         except Exception as exc:
#             raise RuntimeError(f"Failed to parse reflection JSON: {exc}. raw_output={raw[:300]}") from exc
#         if not isinstance(obj, dict):
#             raise RuntimeError("Parsed reflection JSON is not an object.")
#         return obj

#     def _call_llm(self, prompt: str, temperature: float) -> str:
#         if self.prefer_local and self.local_ports:
#             self._ensure_local_clients()
#             if self._active_ports:
#                 try:
#                     return self._call_local_round_robin(prompt=prompt, temperature=temperature)
#                 except Exception as exc:
#                     if not self._local_fallback_notice_printed:
#                         print(f"Warning: local endpoints unavailable, fallback to API provider. detail={exc}")
#                         self._local_fallback_notice_printed = True
#                     if not self._has_api_provider():
#                         raise RuntimeError(f"Local call failed and no API provider is configured: {exc}") from exc
#             elif not self._has_api_provider():
#                 raise RuntimeError(
#                     f"No reachable local endpoints on {self.local_host}:{self.local_ports} "
#                     "and no API provider is configured."
#                 )

#         return ask_llm(
#             prompt=prompt,
#             provider=self.provider,
#             temperature=temperature,
#         )

#     def _has_api_provider(self) -> bool:
#         if self.provider:
#             return True
#         if os.getenv("LLM_PROVIDER"):
#             return True
#         for config in PROVIDER_CONFIGS.values():
#             api_key_env = config.get("api_key_env")
#             if api_key_env and os.getenv(api_key_env):
#                 return True
#         return False

#     def _ensure_local_clients(self) -> None:
#         if self._local_ready:
#             return
#         with self._local_init_lock:
#             if self._local_ready:
#                 return

#             clients: Dict[int, OpenAI] = {}
#             model_by_port: Dict[int, str] = {}

#             for port in self.local_ports:
#                 base_url = f"http://{self.local_host}:{int(port)}/v1"
#                 model_name = self.local_model or self._fetch_model_id(base_url=base_url)
#                 if not model_name:
#                     continue
#                 clients[int(port)] = OpenAI(api_key="EMPTY", base_url=base_url, timeout=self.request_timeout)
#                 model_by_port[int(port)] = model_name

#             self._local_clients = clients
#             self._local_model_by_port = model_by_port
#             self._active_ports = sorted(clients.keys())
#             self._local_ready = True

#     def _fetch_model_id(self, base_url: str) -> Optional[str]:
#         url = f"{base_url.rstrip('/')}/models"
#         req = Request(url, method="GET")
#         try:
#             with urlopen(req, timeout=min(5.0, self.request_timeout)) as response:
#                 payload = json.loads(response.read().decode("utf-8"))
#         except (URLError, ValueError, json.JSONDecodeError):
#             return None

#         if not isinstance(payload, dict):
#             return None
#         data = payload.get("data", [])
#         if not data or not isinstance(data, list) or not isinstance(data[0], dict):
#             return None
#         model_id = data[0].get("id")
#         if isinstance(model_id, str) and model_id:
#             return model_id
#         return None

#     def _next_port(self) -> int:
#         with self._rr_lock:
#             if not self._active_ports:
#                 raise RuntimeError("No active local ports available.")
#             port = self._active_ports[self._rr_idx % len(self._active_ports)]
#             self._rr_idx = (self._rr_idx + 1) % len(self._active_ports)
#             return port

#     def _call_local_round_robin(self, prompt: str, temperature: float) -> str:
#         if not self._active_ports:
#             raise RuntimeError("No available local endpoint.")

#         last_error: Optional[Exception] = None
#         for _ in range(len(self._active_ports)):
#             port = self._next_port()
#             client = self._local_clients[port]
#             model_name = self._local_model_by_port[port]
#             try:
#                 completion = client.chat.completions.create(
#                     model=model_name,
#                     messages=[
#                         {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
#                         {"role": "user", "content": prompt},
#                     ],
#                     temperature=temperature,
#                 )
#                 return completion.choices[0].message.content or ""
#             except Exception as exc:
#                 last_error = exc
#                 continue

#         raise RuntimeError(f"All local endpoints failed. last_error={last_error}")

#     def _collect_dimension_ratings(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Collect dimension info with scores from sample."""
#         evaluation_dimensions = sample.get("evaluation_dimensions", [])
#         dim_index = self._index_dimensions(evaluation_dimensions)

#         ratings = sample.get("ratings", [])
#         if isinstance(ratings, list) and ratings:
#             return self._merge_ratings_with_dimensions(ratings, dim_index, evaluation_dimensions)

#         if isinstance(evaluation_dimensions, list) and evaluation_dimensions:
#             return self._extract_from_dimensions(evaluation_dimensions)

#         return []

#     def _index_dimensions(self, dimensions: Any) -> Dict[str, Dict[str, Any]]:
#         """Index evaluation dimensions by lowercased name."""
#         index: Dict[str, Dict[str, Any]] = {}
#         if not isinstance(dimensions, list):
#             return index
#         for dim in dimensions:
#             if not isinstance(dim, dict):
#                 continue
#             name = dim.get("dimension_name") or dim.get("name") or ""
#             if not name:
#                 continue
#             index[name.lower()] = dim
#         return index

#     def _merge_ratings_with_dimensions(
#         self,
#         ratings: List[Dict[str, Any]],
#         dim_index: Dict[str, Dict[str, Any]],
#         evaluation_dimensions: Any,
#     ) -> List[Dict[str, Any]]:
#         """Merge rating records with evaluation dimension metadata."""
#         merged: List[Dict[str, Any]] = []
#         seen_names: set[str] = set()
#         for rating in ratings:
#             if not isinstance(rating, dict):
#                 continue
#             name = rating.get("dimension_name") or rating.get("name") or ""
#             if not name:
#                 continue
#             lowered_name = name.lower()
#             dim_info = dim_index.get(lowered_name, {})
#             merged.append({
#                 "dimension_name": name,
#                 "score": self._normalize_score(rating.get("score")),
#                 "reason": rating.get("reason", ""),
#                 "category": rating.get("category", "") or dim_info.get("category", ""),
#                 "full_score_criteria": rating.get("full_score_criteria", "") or dim_info.get("full_score_criteria", ""),
#             })
#             seen_names.add(lowered_name)

#         # Backfill dimensions missing from ratings to avoid dropping dimensions silently.
#         if isinstance(evaluation_dimensions, list):
#             for dim in evaluation_dimensions:
#                 if not isinstance(dim, dict):
#                     continue
#                 name = dim.get("dimension_name") or dim.get("name") or ""
#                 if not name:
#                     continue
#                 lowered_name = name.lower()
#                 if lowered_name in seen_names:
#                     continue
#                 merged.append({
#                     "dimension_name": name,
#                     "score": self._normalize_score(dim.get("score")),
#                     "reason": dim.get("reason", ""),
#                     "category": dim.get("category", ""),
#                     "full_score_criteria": dim.get("full_score_criteria", ""),
#                 })
#                 seen_names.add(lowered_name)
#         return merged

#     def _extract_from_dimensions(self, dimensions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Extract score info directly from evaluation_dimensions if present."""
#         extracted: List[Dict[str, Any]] = []
#         for dim in dimensions:
#             if not isinstance(dim, dict):
#                 continue
#             name = dim.get("dimension_name") or dim.get("name") or ""
#             if not name:
#                 continue
#             extracted.append({
#                 "dimension_name": name,
#                 "score": self._normalize_score(dim.get("score")),
#                 "reason": dim.get("reason", ""),
#                 "category": dim.get("category", ""),
#                 "full_score_criteria": dim.get("full_score_criteria", ""),
#             })
#         return extracted

#     def _select_dimensions(
#         self,
#         dimensions,
#         min_score_to_improve,
#         max_dimensions,
#         order,
#         include_no_score,
#         force_all_dimensions=False,
#     ):
#         if force_all_dimensions:
#             filtered = list(dimensions)
#         else:
#             threshold = self.max_score if min_score_to_improve is None else min_score_to_improve

#             filtered = []
#             for dim in dimensions:
#                 score = dim.get("score")

#                 if score is None:
#                     if include_no_score:
#                         filtered.append(dim)
#                     continue

#                 if score < threshold:
#                     filtered.append(dim)

#         if order == "score_desc":
#             filtered.sort(key=self._score_sort_key_desc)
#         elif order == "score_asc":
#             filtered.sort(key=self._score_sort_key_asc)

#         if max_dimensions is not None and max_dimensions > 0:
#             filtered = filtered[:max_dimensions]

#         return filtered

#     def _score_sort_key_asc(self, dim: Dict[str, Any]) -> Tuple[int, float]:
#         score = dim.get("score")
#         if score is None:
#             return (1, float("inf"))
#         return (0, float(score))

#     def _score_sort_key_desc(self, dim: Dict[str, Any]) -> Tuple[int, float]:
#         score = dim.get("score")
#         if score is None:
#             return (1, float("-inf"))
#         return (0, -float(score))
    
#     def _format_score(self, score: Optional[float]) -> str:
#         if score is None:
#             return "None"
#         try:
#             normalized = float(score)
#         except (TypeError, ValueError):
#             return str(score)
#         if normalized.is_integer():
#             return str(int(normalized))
#         return f"{normalized:.2f}".rstrip("0").rstrip(".")

#     def _build_progress_reason(
#         self,
#         original_score: Optional[float],
#         original_reason: str,
#         modified_score: Optional[float],
#         modified_reason: str,
#         reflection_is_reasonable: Any,
#         reflection_reason: str,
#         revision_applied: bool,
#     ) -> str:
#         """Compose human-readable reason showing score evolution and improvement."""
#         chunks: List[str] = []
#         chunks.append(
#             f"Original score={self._format_score(original_score)}; "
#             # f"original reason={original_reason or '(none)'}."
#         )

#         if reflection_is_reasonable is True:
#             chunks.append(
#                 f"Reflection judged the original score reasonable. "
#                 f"Audit note:{reflection_reason or '(none)'}."
#             )
#         elif reflection_is_reasonable is False:
#             chunks.append(
#                 f"Reflection judged the original score unreasonable. "
#                 f"Audit note:{reflection_reason or '(none)'}."
#             )
#         elif reflection_reason:
#             chunks.append(f"Reflection note:{reflection_reason}.")

#         if revision_applied:
#             chunks.append(
#                 f"After one-step revision, score={self._format_score(modified_score)}; "
#                 f"reason:{modified_reason or '(none)'}."
#             )
#             if original_score is not None and modified_score is not None:
#                 delta = float(modified_score) - float(original_score)
#                 if delta > 0:
#                     chunks.append(f"Score improved by {self._format_score(delta)} after the fix.")
#                 elif delta < 0:
#                     chunks.append(f"Score dropped by {self._format_score(abs(delta))} after the fix.")
#                 else:
#                     chunks.append("Score unchanged after the fix.")
#         else:
#             chunks.append(
#                 "No revision applied for this dimension; original score/reason kept."
#             )

#         return " ".join(chunks)

#     def _normalize_score(self, value: Any) -> Optional[float]:
#         """Normalize score value to float if possible."""
#         if isinstance(value, (int, float)):
#             return float(value)
#         if isinstance(value, str):
#             try:
#                 return float(value.strip())
#             except ValueError:
#                 return None
#         return None

#     def _is_full_score(self, score: Any) -> bool:
#         """Whether score reaches full score (e.g., 5)."""
#         normalized = self._normalize_score(score)
#         if normalized is None:
#             return False
#         return normalized >= float(self.max_score)

#     def _save_json_or_jsonl(self, path: Path, items: List[Dict[str, Any]]) -> None:
#         if path.suffix.lower() == ".jsonl":
#             with path.open("w", encoding="utf-8") as f:
#                 for item in items:
#                     f.write(json.dumps(item, ensure_ascii=False) + "\n")
#             return

#         with path.open("w", encoding="utf-8") as f:
#             json.dump(items, f, ensure_ascii=False, indent=2)


"""Iterative CoT generation module - Improve answers dimension by dimension."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import os
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from openai import OpenAI

try:
    from ..common.llm_client import (
        DEFAULT_SYSTEM_MESSAGE,
        PROVIDER_CONFIGS,
        ask_llm,
    )
    from ..common.data_loader import DatasetLoader
    from ..rating.rater import LLMRater
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.llm_client import (
        DEFAULT_SYSTEM_MESSAGE,
        PROVIDER_CONFIGS,
        ask_llm,
    )
    from src.common.data_loader import DatasetLoader
    from src.rating.rater import LLMRater


class IterativeGenerator:
    """Iteratively improve chain-of-thought answers using dimension ratings."""

    def __init__(
        self,
        provider: Optional[str] = None,
        max_score: float = 5.0,
        local_host: str = "localhost",
        local_ports: Optional[List[int]] = None,
        local_model: Optional[str] = None,
        request_timeout: float = 120.0,
        prefer_local: bool = True,
    ):
        """Initialize the generator.

        Args:
            provider: LLM provider name. If not provided, auto-detects from env.
            max_score: Maximum score for each dimension.
            local_host: Host of local OpenAI-compatible servers.
            local_ports: Port list for local servers, e.g. [8000, 8001, 8002, 8003].
            local_model: Override local model id. If None, fetch from /v1/models.
            request_timeout: Timeout for local requests in seconds.
            prefer_local: Try local endpoints first if configured.
        """
        self.provider = provider
        self.max_score = max_score
        self.local_host = local_host
        self.local_ports = list(local_ports or [])
        self.local_model = local_model
        self.request_timeout = request_timeout
        self.prefer_local = prefer_local

        self._local_clients: Dict[int, OpenAI] = {}
        self._local_model_by_port: Dict[int, str] = {}
        self._active_ports: List[int] = []
        self._rr_idx = 0
        self._rr_lock = threading.Lock()
        self._local_ready = False
        self._local_init_lock = threading.Lock()
        self._local_fallback_notice_printed = False

    def generate_single_sample(
        self,
        sample: Dict[str, Any],
        min_score_to_improve: Optional[float] = None,
        max_dimensions: Optional[int] = None,
        order: str = "score_asc",
        include_no_score: bool = False,
        force_all_dimensions: bool = False,
        rewrite_full_score_dimensions: bool = False,
        rewrite_no_score_dimensions: bool = False,
        temperature: float = 0.3,
        delay_between_requests: float = 0.5,
        keep_intermediate: bool = False,
        rerate_after_iteration: bool = False,
        rating_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Iteratively improve a single sample."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        evaluation_dimensions = sample.get("evaluation_dimensions", [])
        effective_rating_temperature = rating_temperature if rating_temperature is not None else temperature

        dims = self._collect_dimension_ratings(sample)
        dims_to_apply = self._select_dimensions(
            dims,
            min_score_to_improve=min_score_to_improve,
            max_dimensions=max_dimensions,
            order=order,
            include_no_score=include_no_score,
            force_all_dimensions=force_all_dimensions,
        )

        result = dict(sample)
        if not question or not answer:
            result["iterated_answer"] = answer
            result["final_answer"] = answer
            result["iteration_trace"] = []
            result["dimension_results"] = []
            result["iteration_status"] = "invalid_input"
            result["iteration_error"] = "Missing question or answer."
            self._attach_final_scores(
                result=result,
                question=question,
                final_answer=answer,
                evaluation_dimensions=evaluation_dimensions,
                rerate_after_iteration=rerate_after_iteration,
                rating_temperature=effective_rating_temperature,
            )
            return result

        if not dims_to_apply:
            result["iterated_answer"] = answer
            result["final_answer"] = answer
            result["iteration_trace"] = []
            result["dimension_results"] = []
            result["iteration_status"] = "skipped_no_dimensions"
            self._attach_final_scores(
                result=result,
                question=question,
                final_answer=answer,
                evaluation_dimensions=evaluation_dimensions,
                rerate_after_iteration=rerate_after_iteration,
                rating_temperature=effective_rating_temperature,
            )
            return result

        base_answer = answer
        trace: List[Dict[str, Any]] = []
        representative_answer = answer

        for idx, dim in enumerate(dims_to_apply, 1):
            dim_score = self._normalize_score(dim.get("score"))
            skip_revision_for_full_score = (
                self._is_full_score(dim_score) and not rewrite_full_score_dimensions
            )
            skip_revision_for_no_score = (
                dim_score is None and not rewrite_no_score_dimensions
            )
            should_rewrite = not (skip_revision_for_full_score or skip_revision_for_no_score)

            if not should_rewrite:
                revised = base_answer
            else:
                prompt = self._build_revision_prompt(question, base_answer, dim)
                revised = self._call_llm(prompt=prompt, temperature=temperature).strip()
                representative_answer = revised

            trace_item = {
                "dimension_name": dim.get("dimension_name", ""),
                "score": dim_score,
                "reason": dim.get("reason", ""),
                "category": dim.get("category", ""),
                "full_score_criteria": dim.get("full_score_criteria", ""),
                "input_answer": base_answer,
                "modified_answer": revised,
                "modified_score": None,
                "modified_reason": "",
                "modified_category": dim.get("category", ""),
                "modified_full_score_criteria": dim.get("full_score_criteria", ""),
                "revision_applied": should_rewrite,
            }

            if not should_rewrite:
                trace_item["modified_score"] = dim_score
                trace_item["modified_reason"] = dim.get("reason", "")
            else:
                try:
                    modified_rating = self._rerate_single_dimension_answer(
                        question=question,
                        answer=revised,
                        dim=dim,
                        temperature=effective_rating_temperature,
                    )
                    trace_item["modified_score"] = self._normalize_score(modified_rating.get("score"))
                    trace_item["modified_reason"] = modified_rating.get("reason", "")
                    trace_item["modified_category"] = (
                        modified_rating.get("category", "") or trace_item["modified_category"]
                    )
                    trace_item["modified_full_score_criteria"] = (
                        modified_rating.get("full_score_criteria", "") or trace_item["modified_full_score_criteria"]
                    )
                except Exception as exc:
                    trace_item["modified_rating_error"] = str(exc)

            if keep_intermediate:
                trace_item["before"] = base_answer
                trace_item["after"] = revised

            trace.append(trace_item)

            if idx < len(dims_to_apply) and delay_between_requests > 0 and should_rewrite:
                time.sleep(delay_between_requests)

        # Independent per-dimension mode has no single global merged answer.
        result["iterated_answer"] = answer
        result["final_answer"] = answer
        result["representative_answer"] = representative_answer
        result["iteration_trace"] = trace
        result["dimension_results"] = trace
        result["iteration_status"] = "iterated"
        result["iteration_meta"] = {
            "min_score_to_improve": self.max_score if min_score_to_improve is None else min_score_to_improve,
            "max_dimensions": max_dimensions,
            "order": order,
            "include_no_score": include_no_score,
            "force_all_dimensions": force_all_dimensions,
            "rewrite_full_score_dimensions": rewrite_full_score_dimensions,
            "rewrite_no_score_dimensions": rewrite_no_score_dimensions,
            "provider": self.provider or "auto",
            "local_ports": self.local_ports,
            "revision_mode": "independent_per_dimension",
            "input_answer_policy": "always_original_answer",
            "aggregate_answer_mode": "no_aggregation_use_per_dimension_results",
        }
        if rerate_after_iteration:
            result["final_rating_note"] = (
                "rerate_after_iteration is ignored in independent_per_dimension mode; "
                "final_ratings come from per-dimension results."
            )
        self._attach_trace_final_scores(result=result, trace=trace)

        return result

    def generate_dataset(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        sample_size: Optional[int] = None,
        min_score_to_improve: Optional[float] = None,
        max_dimensions: Optional[int] = None,
        order: str = "score_asc",
        include_no_score: bool = False,
        force_all_dimensions: bool = False,
        rewrite_full_score_dimensions: bool = False,
        rewrite_no_score_dimensions: bool = False,
        temperature: float = 0.3,
        delay_between_requests: float = 0.5,
        keep_intermediate: bool = False,
        num_workers: int = 1,
        rerate_after_iteration: bool = False,
        rating_temperature: Optional[float] = None,
        output_mode: str = "simplified",
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """Iteratively improve a dataset."""
        loader = DatasetLoader()
        dataset = loader.load_from_json(input_file)

        if sample_size is not None:
            dataset = dataset[: min(sample_size, len(dataset))]

        total = len(dataset)
        print(f"Loaded {total} samples from {input_file}")
        if progress_cb:
            progress_cb(0, total)

        if output_file is None:
            input_path = Path(input_file)
            provider_suffix = f"_{self.provider}" if self.provider else "_auto"
            output_file = str(input_path.parent / f"{input_path.stem}_iterated{provider_suffix}{input_path.suffix}")

        if self.prefer_local and self.local_ports:
            self._ensure_local_clients()
            print(
                f"Local endpoint mode: host={self.local_host}, "
                f"configured_ports={self.local_ports}, active_ports={self._active_ports}"
            )

        if num_workers <= 1:
            results, successful, failed = self._run_sequential(
                dataset=dataset,
                min_score_to_improve=min_score_to_improve,
                max_dimensions=max_dimensions,
                order=order,
                include_no_score=include_no_score,
                force_all_dimensions=force_all_dimensions,
                rewrite_full_score_dimensions=rewrite_full_score_dimensions,
                rewrite_no_score_dimensions=rewrite_no_score_dimensions,
                temperature=temperature,
                delay_between_requests=delay_between_requests,
                keep_intermediate=keep_intermediate,
                rerate_after_iteration=rerate_after_iteration,
                rating_temperature=rating_temperature,
                progress_cb=progress_cb,
            )
        else:
            results, successful, failed = self._run_parallel(
                dataset=dataset,
                min_score_to_improve=min_score_to_improve,
                max_dimensions=max_dimensions,
                order=order,
                include_no_score=include_no_score,
                force_all_dimensions=force_all_dimensions,
                rewrite_full_score_dimensions=rewrite_full_score_dimensions,
                rewrite_no_score_dimensions=rewrite_no_score_dimensions,
                temperature=temperature,
                delay_between_requests=delay_between_requests,
                keep_intermediate=keep_intermediate,
                num_workers=num_workers,
                rerate_after_iteration=rerate_after_iteration,
                rating_temperature=rating_temperature,
                progress_cb=progress_cb,
            )

        formatted_results = self._format_output_items(results, output_mode=output_mode)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_json_or_jsonl(output_path, formatted_results)

        print("\n" + "=" * 60)
        print("Iteration completed!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output file: {output_file}")
        print("=" * 60)

        return output_file

    def _format_output_items(
        self,
        items: List[Dict[str, Any]],
        output_mode: str,
    ) -> List[Dict[str, Any]]:
        """Format output records before saving."""
        mode = (output_mode or "simplified").strip().lower()
        if mode == "full":
            return items
        if mode != "simplified":
            raise ValueError(f"Unsupported output_mode: {output_mode}. Use 'simplified' or 'full'.")
        return [self._to_simplified_item(item) for item in items]

    def _to_simplified_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only question/answer plus per-dimension scoring summary."""
        per_dimension_results: List[Dict[str, Any]] = []
        trace = item.get("iteration_trace", [])
        if isinstance(trace, list):
            for step in trace:
                if not isinstance(step, dict):
                    continue
                step_score = self._normalize_score(step.get("modified_score"))
                if step_score is None:
                    step_score = self._normalize_score(step.get("score"))
                step_reason = (
                    step.get("modified_reason")
                    or step.get("reason")
                    or step.get("modified_rating_error")
                    or ""
                ).strip()
                per_dimension_results.append({
                    "dimension_name": step.get("dimension_name", ""),
                    "modified_answer": step.get("modified_answer") or step.get("after") or "",
                    "modified_score": step_score,
                    "reason": step_reason,
                    "full_score_criteria": (
                        step.get("modified_full_score_criteria", "")
                        or step.get("full_score_criteria", "")
                        or ""
                    ),
                })

        if not per_dimension_results:
            final_ratings = item.get("final_ratings")
            if not isinstance(final_ratings, list):
                final_ratings = item.get("ratings", [])
            if isinstance(final_ratings, list):
                for rating in final_ratings:
                    if not isinstance(rating, dict):
                        continue
                    per_dimension_results.append({
                        "dimension_name": rating.get("dimension_name") or rating.get("name") or "",
                        "modified_answer": "",
                        "modified_score": self._normalize_score(rating.get("score")),
                        "reason": rating.get("reason", ""),
                        "full_score_criteria": rating.get("full_score_criteria", ""),
                    })

        return {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "per_dimension_results": per_dimension_results,
        }

    def _run_sequential(
        self,
        dataset: List[Dict[str, Any]],
        min_score_to_improve: Optional[float],
        max_dimensions: Optional[int],
        order: str,
        include_no_score: bool,
        force_all_dimensions: bool,
        rewrite_full_score_dimensions: bool,
        rewrite_no_score_dimensions: bool,
        temperature: float,
        delay_between_requests: float,
        keep_intermediate: bool,
        rerate_after_iteration: bool,
        rating_temperature: Optional[float],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        results: List[Dict[str, Any]] = []
        successful = 0
        failed = 0
        total = len(dataset)
        done = 0

        for idx, sample in enumerate(dataset, 1):
            try:
                updated = self.generate_single_sample(
                    sample=sample,
                    min_score_to_improve=min_score_to_improve,
                    max_dimensions=max_dimensions,
                    order=order,
                    include_no_score=include_no_score,
                    force_all_dimensions=force_all_dimensions,
                    rewrite_full_score_dimensions=rewrite_full_score_dimensions,
                    rewrite_no_score_dimensions=rewrite_no_score_dimensions,
                    temperature=temperature,
                    delay_between_requests=delay_between_requests,
                    keep_intermediate=keep_intermediate,
                    rerate_after_iteration=rerate_after_iteration,
                    rating_temperature=rating_temperature,
                )
                results.append(updated)
                successful += 1
            except Exception as exc:
                failed += 1
                results.append(self._build_error_item(sample, str(exc)))
            finally:
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return results, successful, failed

    def _run_parallel(
        self,
        dataset: List[Dict[str, Any]],
        min_score_to_improve: Optional[float],
        max_dimensions: Optional[int],
        order: str,
        include_no_score: bool,
        force_all_dimensions: bool,
        rewrite_full_score_dimensions: bool,
        rewrite_no_score_dimensions: bool,
        temperature: float,
        delay_between_requests: float,
        keep_intermediate: bool,
        num_workers: int,
        rerate_after_iteration: bool,
        rating_temperature: Optional[float],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        total = len(dataset)
        results: List[Optional[Dict[str, Any]]] = [None] * total
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self.generate_single_sample,
                    sample,
                    min_score_to_improve,
                    max_dimensions,
                    order,
                    include_no_score,
                    force_all_dimensions,
                    rewrite_full_score_dimensions,
                    rewrite_no_score_dimensions,
                    temperature,
                    delay_between_requests,
                    keep_intermediate,
                    rerate_after_iteration,
                    rating_temperature,
                ): idx
                for idx, sample in enumerate(dataset)
            }

            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                sample = dataset[idx]
                try:
                    results[idx] = future.result()
                    successful += 1
                except Exception as exc:
                    results[idx] = self._build_error_item(sample, str(exc))
                    failed += 1
                completed += 1
                if progress_cb:
                    progress_cb(completed, total)

        finalized = [item for item in results if item is not None]
        return finalized, successful, failed

    def _build_error_item(self, sample: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        error_item = dict(sample)
        error_item["iterated_answer"] = sample.get("answer", "")
        error_item["final_answer"] = sample.get("answer", "")
        error_item["iteration_trace"] = []
        error_item["dimension_results"] = []
        error_item["iteration_status"] = "failed"
        error_item["iteration_error"] = error_message
        return error_item

    def _attach_trace_final_scores(self, result: Dict[str, Any], trace: List[Dict[str, Any]]) -> None:
        """Attach final scores from per-dimension independent trace."""
        final_ratings: List[Dict[str, Any]] = []
        numeric_scores: List[float] = []

        for step in trace:
            if not isinstance(step, dict):
                continue
            score = self._normalize_score(step.get("modified_score"))
            if score is not None:
                numeric_scores.append(score)
            final_ratings.append({
                "dimension_name": step.get("dimension_name", ""),
                "score": score,
                "reason": step.get("modified_reason", ""),
                "category": step.get("modified_category", "") or step.get("category", ""),
                "full_score_criteria": (
                    step.get("modified_full_score_criteria", "") or step.get("full_score_criteria", "")
                ),
            })

        result["final_ratings"] = final_ratings
        result["final_overall_score"] = (
            sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        )
        result["final_rating_mode"] = "from_per_dimension_results"

    def _attach_final_scores(
        self,
        result: Dict[str, Any],
        question: str,
        final_answer: str,
        evaluation_dimensions: Any,
        rerate_after_iteration: bool,
        rating_temperature: float,
    ) -> None:
        """Attach final per-dimension scores after iteration."""
        if not isinstance(evaluation_dimensions, list) or not evaluation_dimensions:
            result["final_ratings"] = []
            result["final_overall_score"] = None
            result["final_rating_mode"] = "no_dimensions"
            return

        if rerate_after_iteration:
            try:
                final_ratings, final_overall = self._rerate_single_answer(
                    question=question,
                    answer=final_answer,
                    evaluation_dimensions=evaluation_dimensions,
                    temperature=rating_temperature,
                )
                result["final_ratings"] = final_ratings
                result["final_overall_score"] = final_overall
                result["final_rating_mode"] = "rerated"
                return
            except Exception as exc:
                result["final_ratings"] = []
                result["final_overall_score"] = None
                result["final_rating_mode"] = "rerated_failed"
                result["final_rating_error"] = str(exc)
                return

        existing_ratings = result.get("ratings", [])
        existing_overall = result.get("overall_score")
        if isinstance(existing_ratings, list):
            result["final_ratings"] = existing_ratings
        else:
            result["final_ratings"] = []
        result["final_overall_score"] = existing_overall if isinstance(existing_overall, (int, float)) else None
        result["final_rating_mode"] = "copied_original"

    def _rerate_single_answer(
        self,
        question: str,
        answer: str,
        evaluation_dimensions: List[Dict[str, Any]],
        temperature: float,
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """Re-score one iterated answer with the same dimensions."""
        max_score = int(self.max_score) if self.max_score >= 1 else 5
        rater = LLMRater(provider=self.provider, max_score=max_score)
        sample = {"question": question, "answer": answer}
        prompt = rater._build_single_sample_prompt(sample, evaluation_dimensions)
        result_text = self._call_llm(prompt=prompt, temperature=temperature)
        parsed = rater._parse_rating_result(result_text)
        if "parse_error" in parsed:
            raw_output = parsed.get("raw_output", "")
            parse_error = parsed.get("parse_error", "Failed to parse rating JSON")
            raise RuntimeError(f"{parse_error}. raw_output={raw_output[:300]}")
        ratings = rater._extract_ratings_from_result(parsed, evaluation_dimensions)
        overall_score = rater._compute_overall_score(ratings)
        return ratings, overall_score

    def _rerate_single_dimension_answer(
        self,
        question: str,
        answer: str,
        dim: Dict[str, Any],
        temperature: float,
    ) -> Dict[str, Any]:
        """Re-score answer on one target dimension only."""
        name = dim.get("dimension_name") or dim.get("name") or ""
        if not name:
            return {}

        single_dimension = {
            "dimension_name": name,
            "category": dim.get("category", ""),
            "full_score_criteria": dim.get("full_score_criteria", ""),
        }
        ratings, _ = self._rerate_single_answer(
            question=question,
            answer=answer,
            evaluation_dimensions=[single_dimension],
            temperature=temperature,
        )
        if ratings and isinstance(ratings[0], dict):
            return ratings[0]
        return {}

    def _call_llm(self, prompt: str, temperature: float) -> str:
        if self.prefer_local and self.local_ports:
            self._ensure_local_clients()
            if self._active_ports:
                try:
                    return self._call_local_round_robin(prompt=prompt, temperature=temperature)
                except Exception as exc:
                    if not self._local_fallback_notice_printed:
                        print(f"Warning: local endpoints unavailable, fallback to API provider. detail={exc}")
                        self._local_fallback_notice_printed = True
                    if not self._has_api_provider():
                        raise RuntimeError(f"Local call failed and no API provider is configured: {exc}") from exc
            elif not self._has_api_provider():
                raise RuntimeError(
                    f"No reachable local endpoints on {self.local_host}:{self.local_ports} "
                    "and no API provider is configured."
                )

        return ask_llm(
            prompt=prompt,
            provider=self.provider,
            temperature=temperature,
        )

    def _has_api_provider(self) -> bool:
        if self.provider:
            return True
        if os.getenv("LLM_PROVIDER"):
            return True
        for config in PROVIDER_CONFIGS.values():
            api_key_env = config.get("api_key_env")
            if api_key_env and os.getenv(api_key_env):
                return True
        return False

    def _ensure_local_clients(self) -> None:
        if self._local_ready:
            return
        with self._local_init_lock:
            if self._local_ready:
                return

            clients: Dict[int, OpenAI] = {}
            model_by_port: Dict[int, str] = {}

            for port in self.local_ports:
                base_url = f"http://{self.local_host}:{int(port)}/v1"
                model_name = self.local_model or self._fetch_model_id(base_url=base_url)
                if not model_name:
                    continue
                clients[int(port)] = OpenAI(api_key="EMPTY", base_url=base_url, timeout=self.request_timeout)
                model_by_port[int(port)] = model_name

            self._local_clients = clients
            self._local_model_by_port = model_by_port
            self._active_ports = sorted(clients.keys())
            self._local_ready = True

    def _fetch_model_id(self, base_url: str) -> Optional[str]:
        url = f"{base_url.rstrip('/')}/models"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=min(5.0, self.request_timeout)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, ValueError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        data = payload.get("data", [])
        if not data or not isinstance(data, list) or not isinstance(data[0], dict):
            return None
        model_id = data[0].get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
        return None

    def _next_port(self) -> int:
        with self._rr_lock:
            if not self._active_ports:
                raise RuntimeError("No active local ports available.")
            port = self._active_ports[self._rr_idx % len(self._active_ports)]
            self._rr_idx = (self._rr_idx + 1) % len(self._active_ports)
            return port

    def _call_local_round_robin(self, prompt: str, temperature: float) -> str:
        if not self._active_ports:
            raise RuntimeError("No available local endpoint.")

        last_error: Optional[Exception] = None
        for _ in range(len(self._active_ports)):
            port = self._next_port()
            client = self._local_clients[port]
            model_name = self._local_model_by_port[port]
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"All local endpoints failed. last_error={last_error}")

    def _collect_dimension_ratings(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect dimension info with scores from sample."""
        evaluation_dimensions = sample.get("evaluation_dimensions", [])
        dim_index = self._index_dimensions(evaluation_dimensions)

        ratings = sample.get("ratings", [])
        if isinstance(ratings, list) and ratings:
            return self._merge_ratings_with_dimensions(ratings, dim_index, evaluation_dimensions)

        if isinstance(evaluation_dimensions, list) and evaluation_dimensions:
            return self._extract_from_dimensions(evaluation_dimensions)

        return []

    def _index_dimensions(self, dimensions: Any) -> Dict[str, Dict[str, Any]]:
        """Index evaluation dimensions by lowercased name."""
        index: Dict[str, Dict[str, Any]] = {}
        if not isinstance(dimensions, list):
            return index
        for dim in dimensions:
            if not isinstance(dim, dict):
                continue
            name = dim.get("dimension_name") or dim.get("name") or ""
            if not name:
                continue
            index[name.lower()] = dim
        return index

    def _merge_ratings_with_dimensions(
        self,
        ratings: List[Dict[str, Any]],
        dim_index: Dict[str, Dict[str, Any]],
        evaluation_dimensions: Any,
    ) -> List[Dict[str, Any]]:
        """Merge rating records with evaluation dimension metadata."""
        merged: List[Dict[str, Any]] = []
        seen_names: set[str] = set()
        for rating in ratings:
            if not isinstance(rating, dict):
                continue
            name = rating.get("dimension_name") or rating.get("name") or ""
            if not name:
                continue
            lowered_name = name.lower()
            dim_info = dim_index.get(lowered_name, {})
            merged.append({
                "dimension_name": name,
                "score": self._normalize_score(rating.get("score")),
                "reason": rating.get("reason", ""),
                "category": rating.get("category", "") or dim_info.get("category", ""),
                "full_score_criteria": rating.get("full_score_criteria", "") or dim_info.get("full_score_criteria", ""),
            })
            seen_names.add(lowered_name)

        # Backfill dimensions missing from ratings to avoid dropping dimensions silently.
        if isinstance(evaluation_dimensions, list):
            for dim in evaluation_dimensions:
                if not isinstance(dim, dict):
                    continue
                name = dim.get("dimension_name") or dim.get("name") or ""
                if not name:
                    continue
                lowered_name = name.lower()
                if lowered_name in seen_names:
                    continue
                merged.append({
                    "dimension_name": name,
                    "score": self._normalize_score(dim.get("score")),
                    "reason": dim.get("reason", ""),
                    "category": dim.get("category", ""),
                    "full_score_criteria": dim.get("full_score_criteria", ""),
                })
                seen_names.add(lowered_name)
        return merged

    def _extract_from_dimensions(self, dimensions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract score info directly from evaluation_dimensions if present."""
        extracted: List[Dict[str, Any]] = []
        for dim in dimensions:
            if not isinstance(dim, dict):
                continue
            name = dim.get("dimension_name") or dim.get("name") or ""
            if not name:
                continue
            extracted.append({
                "dimension_name": name,
                "score": self._normalize_score(dim.get("score")),
                "reason": dim.get("reason", ""),
                "category": dim.get("category", ""),
                "full_score_criteria": dim.get("full_score_criteria", ""),
            })
        return extracted

    def _select_dimensions(
        self,
        dimensions,
        min_score_to_improve,
        max_dimensions,
        order,
        include_no_score,
        force_all_dimensions=False,
    ):
        if force_all_dimensions:
            filtered = list(dimensions)
        else:
            threshold = self.max_score if min_score_to_improve is None else min_score_to_improve

            filtered = []
            for dim in dimensions:
                score = dim.get("score")

                if score is None:
                    if include_no_score:
                        filtered.append(dim)
                    continue

                if score < threshold:
                    filtered.append(dim)

        if order == "score_desc":
            filtered.sort(key=self._score_sort_key_desc)
        elif order == "score_asc":
            filtered.sort(key=self._score_sort_key_asc)

        if max_dimensions is not None and max_dimensions > 0:
            filtered = filtered[:max_dimensions]

        return filtered

    def _score_sort_key_asc(self, dim: Dict[str, Any]) -> Tuple[int, float]:
        score = dim.get("score")
        if score is None:
            return (1, float("inf"))
        return (0, float(score))

    def _score_sort_key_desc(self, dim: Dict[str, Any]) -> Tuple[int, float]:
        score = dim.get("score")
        if score is None:
            return (1, float("-inf"))
        return (0, -float(score))
    
    def _build_revision_prompt(self, question: str, answer: str, dim: Dict[str, Any]) -> str:
        """Build revision prompt for one dimension.

        Strategy:
        - For non-math-critical dimensions (clarity, structure, engagement, etc.): do light editing only.
        - For math-critical dimensions (accuracy, correctness, calculation, logic): allow rewrite-from-scratch with checks.
        """
        score = dim.get("score")
        score_text = "None" if score is None else str(score)
        reason = (dim.get("reason") or "").strip() or "(no reason provided)"
        dim_name = (dim.get("dimension_name") or dim.get("name") or "").strip()
        dim_cat = (dim.get("category") or "").strip()
        criteria = (dim.get("full_score_criteria") or "").strip()

        # Heuristic: treat these as math-critical dimensions (you can tune keywords)
        name_l = dim_name.lower()
        cat_l = dim_cat.lower()
        math_critical = any(k in name_l for k in [
            "accuracy", "calculation", "correct", "method", "logic", "consisten", "reasoning", "valid"
        ]) or any(k in cat_l for k in [
            "accuracy", "calculation", "correct", "logic", "reasoning", "math"
        ])

        common_header = (
            "Instruction:\n"
            "You are an expert math-solution verifier and editor.\n"
            "Edit the answer for ONE target dimension while preserving mathematical correctness.\n"
            "If you introduce intermediate quantities (totals, parts, ratios, etc.), they must be correct and consistent.\n"
            "Do not add meta commentary or extra sections.\n\n"
            f"Question (Q):\n{question}\n\n"
            f"Answer (A):\n{answer}\n\n"
            "Target Dimension:\n"
            f"- dimension_name: {dim_name}\n"
            f"- category: {dim_cat}\n"
            f"- current_score: {score_text}\n"
            f"- current_reason: {reason}\n"
            f"- full_score_criteria: {criteria}\n\n"
        )

        # Light edit prompt (default): minimal edits, keep final answer unless it's clearly wrong
        light_prompt = (
            common_header
            + "Task:\n"
            "Revise the current answer with MINIMAL changes to better satisfy the target dimension.\n"
            "Preserve the current structure and the final numeric answer UNLESS it is clearly wrong.\n\n"
            "Hard constraints:\n"
            "1) Do NOT re-solve from scratch if the current answer is already correct.\n"
            "2) Do NOT introduce new numbers unless necessary. If you do, they must be correct and consistent with the final answer.\n"
            "3) If you detect a mathematical error, fix it, and ensure every stated number is consistent.\n"
            "4) Output ONLY the revised answer. No meta commentary.\n"
        )

        # Strict prompt: allow rewrite-from-scratch, but enforce consistency and checks
        strict_prompt = (
            common_header
            + "Task:\n"
            "Ensure the answer is mathematically correct and logically consistent, then improve it to satisfy the target dimension.\n\n"
            "Hard constraints (must follow):\n"
            "A) Answer exactly what the question asks (quantity and format).\n"
            "B) No unjustified approximations/guessing.\n"
            "C) Every numeric statement must be correct. Do not include any incorrect intermediate totals/ratios.\n"
            "D) If the current answer is wrong/inconsistent, rewrite it so it is fully correct.\n\n"
            "Required procedure:\n"
            "1) Quickly verify the current answer. If it is correct and consistent, do NOT rewrite from scratch; do a light edit only.\n"
            "2) If there is any error or inconsistency, solve correctly and include at least one explicit check.\n"
            "3) After correctness is secured, revise wording/structure to meet the target dimension.\n"
            "4) Output ONLY the revised answer.\n"
        )

        return strict_prompt if math_critical else light_prompt

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

    def _is_full_score(self, score: Any) -> bool:
        """Whether score reaches full score (e.g., 5)."""
        normalized = self._normalize_score(score)
        if normalized is None:
            return False
        return normalized >= float(self.max_score)

    def _save_json_or_jsonl(self, path: Path, items: List[Dict[str, Any]]) -> None:
        if path.suffix.lower() == ".jsonl":
            with path.open("w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            return

        with path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
