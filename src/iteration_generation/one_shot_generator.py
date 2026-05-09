"""One-shot CoT generation module - Improve answers based on all dimensions simultaneously."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable
import json
import os
import threading
import time
from pathlib import Path
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


class OneShotGenerator:
    """Improve chain-of-thought answers by consolidating all dimension ratings into a single prompt."""

    def __init__(
        self,
        provider: Optional[str] = None,
        max_score: float = 5.0,
        local_host: str = "localhost",
        local_ports: Optional[List[int]] = None,
        local_model: Optional[str] = None,
        request_timeout: float = 120.0,
        prefer_local: bool = True,
        auto_workers_per_port: int = 4,
        local_max_inflight_per_port: int = 0,
        local_max_retries: int = 2,
        local_retry_backoff: float = 0.2,
    ):
        """Initialize the generator.

        Args:
            provider: LLM provider name. If not provided, auto-detects from env.
            max_score: Maximum score for each dimension.
            local_host: Host of local OpenAI-compatible servers.
            local_ports: Port list for local servers, e.g. [8000, 8001].
            local_model: Override local model id.
            request_timeout: Timeout for local requests in seconds.
            prefer_local: Try local endpoints first if configured.
            auto_workers_per_port: Auto mode workers per active local port.
            local_max_inflight_per_port: Max in-flight requests per port. <=0 means unlimited.
            local_max_retries: Retry rounds for local calls.
            local_retry_backoff: Sleep seconds between local retries.
        """
        self.provider = provider
        self.max_score = max_score
        self.local_host = local_host
        self.local_ports = list(local_ports or [])
        self.local_model = local_model
        self.request_timeout = request_timeout
        self.prefer_local = prefer_local
        self.auto_workers_per_port = max(1, int(auto_workers_per_port))
        self.local_max_inflight_per_port = int(local_max_inflight_per_port)
        self.local_max_retries = max(1, int(local_max_retries))
        self.local_retry_backoff = max(0.0, float(local_retry_backoff))

        self._local_clients: Dict[int, OpenAI] = {}
        self._local_model_by_port: Dict[int, str] = {}
        self._active_ports: List[int] = []
        self._rr_idx = 0
        self._rr_lock = threading.Lock()
        self._port_state_cv = threading.Condition()
        self._inflight_by_port: Dict[int, int] = {}
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
        temperature: float = 0.3,
        delay_between_requests: float = 0.0,  # Less relevant in one-shot but kept for interface consistency
        keep_intermediate: bool = False,      # Kept for consistency, stores the single hop
        rerate_after_iteration: bool = False,
        rating_temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Improve a single sample in one shot."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        evaluation_dimensions = sample.get("evaluation_dimensions", [])

        # 1. Collect and Filter Dimensions
        all_dims = self._collect_dimension_ratings(sample)
        dims_to_apply = self._select_dimensions(
            all_dims,
            min_score_to_improve=min_score_to_improve,
            max_dimensions=max_dimensions,
            order=order,
            include_no_score=include_no_score,
        )

        result = dict(sample)

        # Validation
        if not question or not answer:
            result["iterated_answer"] = answer
            result["final_answer"] = answer
            result["iteration_trace"] = []
            result["iteration_status"] = "invalid_input"
            result["iteration_error"] = "Missing question or answer."
            self._attach_final_scores(
                result, question, answer, evaluation_dimensions,
                rerate_after_iteration, rating_temperature or temperature
            )
            return result

        # No dimensions to fix -> Skip
        if not dims_to_apply:
            result["iterated_answer"] = answer
            result["final_answer"] = answer
            result["iteration_trace"] = []
            result["iteration_status"] = "skipped_no_dimensions"
            self._attach_final_scores(
                result, question, answer, evaluation_dimensions,
                rerate_after_iteration, rating_temperature or temperature
            )
            return result

        # 2. Build ONE Consolidated Prompt
        # Prompt input contains all dimensions and criteria, while targeted dimensions
        # are listed explicitly to guide priority during correction.
        prompt = self._build_consolidated_prompt(
            question=question,
            answer=answer,
            all_dims=all_dims,
            targeted_dims=dims_to_apply,
        )

        # 3. Call LLM Once
        try:
            model_output = self._call_llm(prompt=prompt, temperature=temperature).strip()
            revised_answer, integrated_reason = self._extract_rewrite_and_reason(
                raw_text=model_output,
                fallback_answer=answer,
            )
            if not revised_answer:
                revised_answer = answer

            # Record trace (Single Step)
            trace_item = {
                "step": "one_shot_consolidation",
                "targeted_dimensions": [d.get("dimension_name") for d in dims_to_apply],
                "all_dimensions": [d.get("dimension_name") for d in all_dims],
                "dimension_details": dims_to_apply,
                "consolidated_reason": integrated_reason,
            }
            if keep_intermediate:
                trace_item["before"] = answer
                trace_item["after"] = revised_answer

            result["iterated_answer"] = revised_answer
            result["final_answer"] = revised_answer
            result["one_shot_reason"] = integrated_reason
            result["reason"] = integrated_reason
            result["iteration_trace"] = [trace_item]
            result["iteration_status"] = "one_shot_success"

        except Exception as e:
            result["iterated_answer"] = answer
            result["final_answer"] = answer
            result["iteration_trace"] = []
            result["iteration_status"] = "failed"
            result["iteration_error"] = str(e)
            return result

        # 4. Meta Info & Rerating
        result["iteration_meta"] = {
            "mode": "one_shot",
            "min_score_to_improve": self.max_score if min_score_to_improve is None else min_score_to_improve,
            "dimension_count": len(dims_to_apply),
            "provider": self.provider or "auto",
        }

        self._attach_final_scores(
            result=result,
            question=question,
            final_answer=revised_answer,
            evaluation_dimensions=evaluation_dimensions,
            rerate_after_iteration=rerate_after_iteration,
            rating_temperature=rating_temperature if rating_temperature is not None else temperature,
        )
        if not str(result.get("reason", "")).strip():
            result["reason"] = self._compose_aggregate_reason(result.get("final_ratings", []))
            result["one_shot_reason"] = result["reason"]

        return result

    def _build_consolidated_prompt(
        self,
        question: str,
        answer: str,
        all_dims: List[Dict[str, Any]],
        targeted_dims: List[Dict[str, Any]],
    ) -> str:
        """Build a one-shot prompt with all dimensions and a structured output contract."""
        all_dims_text = ""
        for i, dim in enumerate(all_dims, 1):
            name = dim.get("dimension_name", "Unknown Dimension")
            score = dim.get("score")
            score_str = str(score) if score is not None else "N/A"
            reason = dim.get("reason", "No specific reason provided.")
            criteria = dim.get("full_score_criteria", "No criteria provided.")

            all_dims_text += (
                f"{i}. {name}\n"
                f"   - current_score: {score_str}\n"
                f"   - current_reason: {reason}\n"
                f"   - full_score_criteria: {criteria}\n\n"
            )

        targeted_names = [d.get("dimension_name", "") for d in targeted_dims if isinstance(d, dict)]
        targeted_text = ", ".join([name for name in targeted_names if name]) or "(none)"

        prompt = (
            "Instruction:\n"
            "You are a precise chain-of-thought editor and evaluator.\n"
            "Use the question, original answer, and all dimensions with full-score criteria to produce a corrected answer.\n"
            "Mathematical correctness is highest priority.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Original Answer:\n"
            f"{answer}\n\n"
            "All Dimensions (with score and full-score criteria):\n"
            f"{all_dims_text}"
            "Priority Dimensions to improve first:\n"
            f"{targeted_text}\n\n"
            "Output requirements:\n"
            "1) Return valid JSON only.\n"
            "2) The JSON must include:\n"
            "   - corrected_answer: corrected full answer text\n"
            "   - integrated_reason: one integrated reason covering all dimensions and trade-offs\n"
            "3) Do not output markdown fences.\n"
            "4) Do not add extra fields.\n\n"
            "JSON schema example:\n"
            "{\"corrected_answer\": \"...\", \"integrated_reason\": \"...\"}"
        )
        return prompt

    def generate_dataset(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        sample_size: Optional[int] = None,
        min_score_to_improve: Optional[float] = None,
        max_dimensions: Optional[int] = None,
        order: str = "score_asc",
        include_no_score: bool = False,
        temperature: float = 0.3,
        delay_between_requests: float = 0.0,
        keep_intermediate: bool = False,
        num_workers: int = 1,
        rerate_after_iteration: bool = False,
        rating_temperature: Optional[float] = None,
        output_mode: str = "simplified",
        progress_cb: Optional[Callable[[int, int], None]] = None,  # NEW
    ) -> str:
        """Process a dataset using the one-shot strategy.

        progress_cb: callback(done, total) called from the main thread.
        """
        loader = DatasetLoader()
        dataset = loader.load_from_json(input_file)

        if sample_size is not None:
            dataset = dataset[: min(sample_size, len(dataset))]

        total = len(dataset)
        print(f"Loaded {total} samples from {input_file} (One-Shot Mode)")

        if progress_cb:
            progress_cb(0, total)

        if output_file is None:
            input_path = Path(input_file)
            provider_suffix = f"_{self.provider}" if self.provider else "_auto"
            output_file = str(input_path.parent / f"{input_path.stem}_oneshot{provider_suffix}{input_path.suffix}")

        if self.prefer_local and self.local_ports:
            self._ensure_local_clients()
            print(f"Local endpoint mode: {self.local_host}, ports={self._active_ports}")

        resolved_workers = self._resolve_num_workers(num_workers=num_workers, total_samples=total)
        if total > 0:
            print(
                f"Execution config: workers={resolved_workers}, "
                f"auto_workers_per_port={self.auto_workers_per_port}, "
                f"max_inflight_per_port={self.local_max_inflight_per_port}"
            )

        if resolved_workers <= 1:
            results, successful, failed, skipped = self._run_sequential(
                dataset,
                min_score_to_improve,
                max_dimensions,
                order,
                include_no_score,
                temperature,
                delay_between_requests,
                keep_intermediate,
                rerate_after_iteration,
                rating_temperature,
                progress_cb,  # NEW
            )
        else:
            results, successful, failed, skipped = self._run_parallel(
                dataset,
                min_score_to_improve,
                max_dimensions,
                order,
                include_no_score,
                temperature,
                delay_between_requests,
                keep_intermediate,
                resolved_workers,
                rerate_after_iteration,
                rating_temperature,
                progress_cb,  # NEW
            )

        formatted_results = self._format_output_items(results, output_mode=output_mode)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_json_or_jsonl(output_path, formatted_results)

        print(f"\nOne-Shot Iteration completed! Success: {successful}, Failed: {failed}, Skipped: {skipped}")
        print(f"Output: {output_file}")
        return output_file

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _format_output_items(self, items: List[Dict[str, Any]], output_mode: str) -> List[Dict[str, Any]]:
        mode = (output_mode or "simplified").strip().lower()
        if mode == "full":
            return items
        return [self._to_simplified_item(item) for item in items]

    def _to_simplified_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        final_ratings = item.get("final_ratings")
        if not isinstance(final_ratings, list):
            final_ratings = item.get("ratings", [])

        dimension_scores: List[Dict[str, Any]] = []
        if isinstance(final_ratings, list):
            for rating in final_ratings:
                if not isinstance(rating, dict):
                    continue
                dimension_scores.append({
                    "dimension_name": rating.get("dimension_name") or rating.get("name") or "",
                    "score": self._normalize_score(rating.get("score")),
                })
        integrated_reason = (
            str(item.get("reason", "")).strip()
            or str(item.get("one_shot_reason", "")).strip()
            or self._compose_aggregate_reason(final_ratings)
        )

        return {
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "modified_answer": item.get("final_answer") or item.get("iterated_answer") or item.get("answer", ""),
            "modified_dimension_scores": dimension_scores,
            "reason": integrated_reason,
        }

    def _extract_rewrite_and_reason(self, raw_text: str, fallback_answer: str) -> Tuple[str, str]:
        """Parse one-shot model output and return (corrected_answer, integrated_reason)."""
        text = str(raw_text or "").strip()
        if not text:
            return fallback_answer, ""

        candidates: List[str] = [text]
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if not block:
                    continue
                if block.lower().startswith("json"):
                    block = block[4:].strip()
                candidates.append(block)

        for candidate in candidates:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end <= start:
                continue
            snippet = candidate[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue

            corrected_answer = str(
                parsed.get("corrected_answer")
                or parsed.get("revised_answer")
                or parsed.get("modified_answer")
                or parsed.get("final_answer")
                or parsed.get("answer")
                or ""
            ).strip()
            integrated_reason = str(
                parsed.get("integrated_reason")
                or parsed.get("overall_reason")
                or parsed.get("summary_reason")
                or parsed.get("reason")
                or ""
            ).strip()
            if corrected_answer:
                return corrected_answer, integrated_reason

        return text, ""

    def _compose_aggregate_reason(self, ratings: Any) -> str:
        """Compose a single integrated reason from dimension ratings as fallback."""
        if not isinstance(ratings, list):
            return ""
        chunks: List[str] = []
        for item in ratings:
            if not isinstance(item, dict):
                continue
            name = str(item.get("dimension_name") or item.get("name") or "").strip()
            score = self._normalize_score(item.get("score"))
            reason = str(item.get("reason") or "").strip()
            if not name and not reason:
                continue
            score_text = "N/A" if score is None else str(score)
            if reason:
                chunks.append(f"[{name}] score={score_text}; {reason}")
            else:
                chunks.append(f"[{name}] score={score_text}")
        return " || ".join(chunks)

    def _resolve_num_workers(self, num_workers: int, total_samples: int) -> int:
        if total_samples <= 0:
            return 1

        if num_workers > 0:
            return min(num_workers, total_samples)

        if self.prefer_local and self._active_ports:
            auto_workers = len(self._active_ports) * self.auto_workers_per_port
            return min(max(1, auto_workers), total_samples)

        return 1

    def _run_sequential(
        self,
        dataset,
        min_score,
        max_dims,
        order,
        inc_no_score,
        temp,
        delay,
        keep,
        rerate,
        rate_temp,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        results: List[Dict[str, Any]] = []
        success, failed, skipped = 0, 0, 0
        total = len(dataset)
        done = 0

        for idx, sample in enumerate(dataset, 1):
            try:
                print(f"Processing {idx}/{total}...", end=" ", flush=True)
                res = self.generate_single_sample(
                    sample,
                    min_score,
                    max_dims,
                    order,
                    inc_no_score,
                    temp,
                    delay,
                    keep,
                    rerate,
                    rate_temp,
                )
                results.append(res)
                status = str(res.get("iteration_status", "")).strip().lower()
                if status in {"failed", "invalid_input"}:
                    failed += 1
                else:
                    success += 1
                    if status == "skipped_no_dimensions":
                        skipped += 1
                print("ok")
            except Exception as e:
                print(f"err: {e}")
                failed += 1
                results.append(self._build_error_item(sample, str(e)))
            finally:
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return results, success, failed, skipped

    def _run_parallel(
        self,
        dataset,
        min_score,
        max_dims,
        order,
        inc_no_score,
        temp,
        delay,
        keep,
        workers,
        rerate,
        rate_temp,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        results: List[Optional[Dict[str, Any]]] = [None] * len(dataset)
        success, failed, skipped = 0, 0, 0
        total = len(dataset)
        done = 0

        print(f"Parallel execution with {workers} workers")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self.generate_single_sample,
                    sample,
                    min_score,
                    max_dims,
                    order,
                    inc_no_score,
                    temp,
                    delay,
                    keep,
                    rerate,
                    rate_temp,
                ): i
                for i, sample in enumerate(dataset)
            }

            # �?杩欓噷鏄€滀富绾跨▼鏀堕泦缁撴灉鈥濈殑鍦版柟锛氭瘡瀹屾垚涓€涓?future锛宒one += 1 骞跺洖璋?progress_cb
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    result = fut.result()
                    results[i] = result
                    status = str(result.get("iteration_status", "")).strip().lower()
                    if status in {"failed", "invalid_input"}:
                        failed += 1
                    else:
                        success += 1
                        if status == "skipped_no_dimensions":
                            skipped += 1
                except Exception as e:
                    results[i] = self._build_error_item(dataset[i], str(e))
                    failed += 1
                finally:
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)

        return [r for r in results if r], success, failed, skipped

    def _build_error_item(self, sample, msg):
        item = dict(sample)
        item.update({
            "iteration_status": "failed",
            "iteration_error": msg,
            "final_answer": sample.get("answer", ""),
        })
        return item

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
        temperature: float
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        """Re-score one iterated answer with the same dimensions."""
        rater = LLMRater(provider=self.provider, max_score=int(self.max_score))
        sample = {"question": question, "answer": answer}
        prompt = rater._build_single_sample_prompt(sample, evaluation_dimensions)
        res_text = self._call_llm(prompt, temperature)
        parsed = rater._parse_rating_result(res_text)
        ratings = rater._extract_ratings_from_result(parsed, evaluation_dimensions)
        return ratings, rater._compute_overall_score(ratings)

    # --- Infrastructure (Network/Local) ---

    def _call_llm(self, prompt: str, temperature: float) -> str:
        if self.prefer_local and self.local_ports:
            self._ensure_local_clients()
            if self._active_ports:
                try:
                    return self._call_local_round_robin(prompt, temperature)
                except Exception as e:
                    if not self._local_fallback_notice_printed:
                        print(f"Warn: Local failed ({e}), using API.")
                        self._local_fallback_notice_printed = True
                    if not self._has_api_provider():
                        raise
            elif not self._has_api_provider():
                raise RuntimeError("No local ports active and no API provider.")
        return ask_llm(prompt=prompt, provider=self.provider, temperature=temperature)

    def _call_local_round_robin(self, prompt: str, temperature: float) -> str:
        if not self._active_ports:
            raise RuntimeError("No active ports")
        last_err = None
        total_attempts = max(1, len(self._active_ports) * self.local_max_retries)
        for _ in range(total_attempts):
            port = self._acquire_port()
            try:
                client = self._local_clients[port]
                model = self._local_model_by_port[port]
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if self.local_retry_backoff > 0:
                    time.sleep(self.local_retry_backoff)
            finally:
                self._release_port(port)

        raise RuntimeError(f"All local ports failed after {total_attempts} attempts. Last: {last_err}")

    def _acquire_port(self) -> int:
        if not self._active_ports:
            raise RuntimeError("No active ports")
        max_inflight = self.local_max_inflight_per_port

        while True:
            with self._port_state_cv:
                for _ in range(len(self._active_ports)):
                    port = self._next_port()
                    inflight = self._inflight_by_port.get(port, 0)
                    if max_inflight <= 0 or inflight < max_inflight:
                        self._inflight_by_port[port] = inflight + 1
                        return port
                self._port_state_cv.wait(timeout=0.01)

    def _release_port(self, port: int) -> None:
        with self._port_state_cv:
            inflight = self._inflight_by_port.get(port, 0)
            if inflight > 0:
                self._inflight_by_port[port] = inflight - 1
            self._port_state_cv.notify()

    def _ensure_local_clients(self):
        if self._local_ready:
            return
        with self._local_init_lock:
            if self._local_ready:
                return
            clients, models = {}, {}
            for p in self.local_ports:
                base = f"http://{self.local_host}:{p}/v1"
                m_id = self.local_model or self._fetch_model_id(base)
                if m_id:
                    clients[p] = OpenAI(api_key="EMPTY", base_url=base, timeout=self.request_timeout)
                    models[p] = m_id
            self._local_clients, self._local_model_by_port = clients, models
            self._active_ports = sorted(clients.keys())
            self._inflight_by_port = {p: 0 for p in self._active_ports}
            self._local_ready = True

    def _fetch_model_id(self, base):
        try:
            with urlopen(Request(f"{base.rstrip('/')}/models"), timeout=5) as r:
                data = json.loads(r.read())["data"]
                return data[0]["id"]
        except Exception:
            return None

    def _next_port(self):
        with self._rr_lock:
            p = self._active_ports[self._rr_idx % len(self._active_ports)]
            self._rr_idx = (self._rr_idx + 1) % len(self._active_ports)
            return p

    def _has_api_provider(self):
        if self.provider or os.getenv("LLM_PROVIDER"):
            return True
        return any(os.getenv(cfg.get("api_key_env", "")) for cfg in PROVIDER_CONFIGS.values())

    def _collect_dimension_ratings(self, sample):
        dims = sample.get("evaluation_dimensions", [])
        ratings = sample.get("ratings", [])
        dim_list = dims if isinstance(dims, list) else []
        index = {
            (d.get("dimension_name") or d.get("name") or "").lower(): d
            for d in dim_list
            if isinstance(d, dict) and (d.get("dimension_name") or d.get("name"))
        }

        merged = []
        seen = set()

        if isinstance(ratings, list) and ratings:
            for item in ratings:
                if not isinstance(item, dict):
                    continue
                name = item.get("dimension_name") or item.get("name")
                if not name:
                    continue

                lowered = name.lower()
                ref = index.get(lowered, {})
                merged.append({
                    "dimension_name": name,
                    "score": self._normalize_score(item.get("score")),
                    "reason": item.get("reason", ""),
                    "category": item.get("category") or ref.get("category", ""),
                    "full_score_criteria": item.get("full_score_criteria") or ref.get("full_score_criteria", ""),
                })
                seen.add(lowered)

            # Backfill dimensions missing from ratings.
            for dim in dim_list:
                if not isinstance(dim, dict):
                    continue
                name = dim.get("dimension_name") or dim.get("name")
                if not name:
                    continue
                lowered = name.lower()
                if lowered in seen:
                    continue
                merged.append({
                    "dimension_name": name,
                    "score": self._normalize_score(dim.get("score")),
                    "reason": dim.get("reason", ""),
                    "category": dim.get("category", ""),
                    "full_score_criteria": dim.get("full_score_criteria", ""),
                })
                seen.add(lowered)
        else:
            for dim in dim_list:
                if not isinstance(dim, dict):
                    continue
                name = dim.get("dimension_name") or dim.get("name")
                if not name:
                    continue
                merged.append({
                    "dimension_name": name,
                    "score": self._normalize_score(dim.get("score")),
                    "reason": dim.get("reason", ""),
                    "category": dim.get("category", ""),
                    "full_score_criteria": dim.get("full_score_criteria", ""),
                })
        return merged

    def _select_dimensions(self, dims, min_score_to_improve, max_dimensions, order, include_no_score):
        threshold = self.max_score if min_score_to_improve is None else min_score_to_improve
        filtered = []
        for d in dims:
            s = d.get("score")
            if s is None:
                if include_no_score:
                    filtered.append(d)
            elif s < threshold:
                filtered.append(d)

        if order == "score_asc":
            key_func = lambda x: (0, x.get("score") if x.get("score") is not None else float("inf"))
        elif order == "score_desc":
            key_func = lambda x: (0, -(x.get("score") if x.get("score") is not None else float("-inf")))
        else:
            key_func = lambda x: 0

        filtered.sort(key=key_func)
        if max_dimensions is not None and max_dimensions > 0:
            filtered = filtered[:max_dimensions]
        return filtered

    def _normalize_score(self, val):
        try:
            return float(val)
        except Exception:
            return None

    def _save_json_or_jsonl(self, path, items):
        with path.open("w", encoding="utf-8") as f:
            if path.suffix == ".jsonl":
                for i in items:
                    f.write(json.dumps(i, ensure_ascii=False) + "\n")
            else:
                json.dump(items, f, ensure_ascii=False, indent=2)
