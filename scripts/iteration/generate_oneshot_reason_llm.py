#!/usr/bin/env python
"""Generate one aggregated reason for one-shot iteration outputs using LLM.

For each sample in files under iteration Api:
- Input to LLM:
  instruction + question + modified_answer + all dimensions + each score + full_score_criteria
- Output:
  one plain-text response stored as a single field (default: "reason")

Supports two backends:
1) local    : OpenAI-compatible local API endpoints
2) provider : deployed provider via src.common.llm_client.ask_llm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ONESHOT_SUFFIXES = ("_one-shot_score_direct", "_one_shot_score_direct")
SYSTEM_MESSAGE = (
    "You are an expert evaluator for math answer quality. "
    "Given dimension scores and criteria, explain why the revised answer deserves those scores."
)


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_dim(value: Any) -> str:
    return _normalize_text(value).lower()


def _records_from_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "samples", "records", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


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
    return _records_from_payload(payload)


def _save_records(path: Path, rows: list[Any]) -> None:
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_oneshot_suffix(stem: str) -> str:
    for suffix in ONESHOT_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _pick_filter_source(iter_file: Path, filter_dir: Path) -> Path | None:
    base = _strip_oneshot_suffix(iter_file.stem)
    candidates = [filter_dir / f"{base}.json", filter_dir / f"{base}.jsonl"]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(filter_dir.rglob(f"{base}.json")):
        if path.is_file():
            return path
    for path in sorted(filter_dir.rglob(f"{base}.jsonl")):
        if path.is_file():
            return path
    return None


def _build_question_index(rows: list[Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = _normalize_text(row.get("question"))
        if question and question not in index:
            index[question] = row
    return index


def _extract_dim_criteria_map(source_row: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def absorb(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            name = _normalize_text(item.get("dimension_name") or item.get("name"))
            key = _normalize_dim(name)
            if not key:
                continue
            criteria = _normalize_text(item.get("full_score_criteria"))
            if criteria and key not in mapping:
                mapping[key] = criteria

    absorb(source_row.get("evaluation_dimensions"))
    absorb(source_row.get("ratings"))
    absorb(source_row.get("final_ratings"))
    return mapping


def _build_prompt(question: str, modified_answer: str, dims: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, dim in enumerate(dims, start=1):
        name = _normalize_text(dim.get("dimension_name") or dim.get("name"))
        score = dim.get("score")
        criteria = _normalize_text(dim.get("full_score_criteria"))
        lines.append(
            f"{idx}. dimension={name}; score={score}; full_score_criteria={criteria or '(missing)'}"
        )

    dims_text = "\n".join(lines) if lines else "(no dimensions provided)"

    return (
        "Task:\n"
        "Write ONE integrated reason explaining why the revised answer gets the given dimension scores.\n"
        "Requirements:\n"
        "1) Use all provided dimensions and criteria.\n"
        "2) Reflect score levels faithfully (full scores indicate strengths; lower scores indicate remaining issues).\n"
        "3) Keep it concise and factual (2-6 sentences).\n"
        "4) Output plain text only.\n\n"
        f"Question:\n{question}\n\n"
        f"Revised Answer:\n{modified_answer}\n\n"
        f"Dimensions:\n{dims_text}\n"
    )


@dataclass
class LocalEndpoint:
    port: int
    client: Any
    model: str


class LocalBackend:
    def __init__(
        self,
        host: str,
        ports: list[int],
        model: str | None,
        timeout: float,
    ) -> None:
        self._endpoints: list[LocalEndpoint] = []
        self._rr = 0
        api_key = os.getenv("LOCAL_API_KEY", "EMPTY")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "backend=local requires package 'openai'. Please install it first."
            ) from exc

        for port in ports:
            base_url = f"http://{host}:{port}/v1"
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            model_name = model or self._fetch_model_name(host=host, port=port, timeout=timeout)
            self._endpoints.append(LocalEndpoint(port=port, client=client, model=model_name))

        if not self._endpoints:
            raise RuntimeError("No valid local endpoints available.")

    @staticmethod
    def _fetch_model_name(host: str, port: int, timeout: float) -> str:
        url = f"http://{host}:{port}/v1/models"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        model_list = data.get("data", [])
        if not model_list:
            raise RuntimeError(f"No models returned by {url}")
        model_id = model_list[0].get("id")
        if not model_id:
            raise RuntimeError(f"Invalid model payload from {url}")
        return str(model_id)

    def _next(self) -> LocalEndpoint:
        ep = self._endpoints[self._rr % len(self._endpoints)]
        self._rr += 1
        return ep

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_retries: int,
        retry_delay: float,
    ) -> str:
        last_err: Exception | None = None
        total_attempts = max(1, max_retries)
        for attempt in range(total_attempts):
            ep = self._next()
            try:
                completion = ep.client.chat.completions.create(
                    model=ep.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                )
                return _normalize_text(completion.choices[0].message.content or "")
            except Exception as exc:  # pragma: no cover
                last_err = exc
                if attempt < total_attempts - 1:
                    time.sleep(retry_delay)
        raise RuntimeError(f"Local backend failed after retries: {last_err}")


class ProviderBackend:
    def __init__(self, provider: str | None, model: str | None) -> None:
        self._ask_llm = None
        self._provider = provider
        self._model = model

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_retries: int,
        retry_delay: float,
    ) -> str:
        if self._ask_llm is None:
            from src.common.llm_client import ask_llm  # lazy import after repo path is ensured

            self._ask_llm = ask_llm

        kwargs: dict[str, Any] = {}
        if self._model:
            kwargs["model"] = self._model
        text = self._ask_llm(
            prompt=prompt,
            provider=self._provider,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            **kwargs,
        )
        return _normalize_text(text)


def _resolve_paths(repo_root: Path, raw_paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        out.append(path)
    return out


def _collect_files(repo_root: Path, iter_dir: Path, files: list[str], pattern: str, recursive: bool) -> list[Path]:
    if files:
        return _resolve_paths(repo_root, files)
    if recursive:
        return sorted(iter_dir.rglob(pattern))
    return sorted(iter_dir.glob(pattern))


def _parse_ports(raw: str) -> list[int]:
    ports: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        ports.append(int(chunk))
    return ports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aggregated reason for one-shot iteration outputs via LLM."
    )
    parser.add_argument("--iter-dir", default="datasets/GSM8K/iteration/Api", help="Iteration output folder.")
    parser.add_argument("--filter-dir", default="datasets/GSM8K/filter", help="Filter folder with criteria source.")
    parser.add_argument("--files", nargs="*", default=[], help="Explicit files to process.")
    parser.add_argument(
        "--pattern",
        default="*_one-shot_score_direct.json",
        help="File pattern under iter-dir when --files is empty.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan iter-dir.")
    parser.add_argument(
        "--backend",
        choices=["local", "provider"],
        default="provider",
        help="Reason generation backend.",
    )
    parser.add_argument("--provider", default=None, help="Provider name when backend=provider (e.g. zyuncs).")
    parser.add_argument("--provider-model", default=None, help="Override provider model id.")
    parser.add_argument("--local-host", default="localhost", help="Local API host when backend=local.")
    parser.add_argument(
        "--local-ports",
        default="8000,8001,8002,8003",
        help="Comma-separated local API ports when backend=local.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Optional fixed local model name. If omitted, auto-fetch from /v1/models.",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for LLM calls.")
    parser.add_argument("--retry-delay", type=float, default=1.5, help="Delay between retries in seconds.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between samples in seconds.")
    parser.add_argument("--output-field", default="reason", help="Field name to store generated reason.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output-field.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit samples per file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write processed files. "
            "Default: <iter-dir>/one-shot. Ignored when --inplace is set."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Write back to original input files instead of output-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Small-batch mode: number of samples per batch in each file.",
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Small-batch mode: batch index (0-based) in each file.",
    )
    parser.add_argument("--write", action="store_true", help="Write output files. Default dry-run.")
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.batch_index < 0:
        raise ValueError("--batch-index must be >= 0")
    if args.max_items is not None and args.max_items < 0:
        raise ValueError("--max-items must be >= 0")

    repo_root = _ensure_repo_on_path()
    iter_dir = Path(args.iter_dir)
    if not iter_dir.is_absolute():
        iter_dir = repo_root / iter_dir
    filter_dir = Path(args.filter_dir)
    if not filter_dir.is_absolute():
        filter_dir = repo_root / filter_dir
    output_dir = None
    if not args.inplace:
        if args.output_dir:
            output_dir = Path(args.output_dir)
            if not output_dir.is_absolute():
                output_dir = repo_root / output_dir
        else:
            output_dir = iter_dir / "one-shot"

    if not filter_dir.is_dir():
        raise FileNotFoundError(f"Filter dir not found: {filter_dir}")
    if not iter_dir.is_dir() and not args.files:
        raise FileNotFoundError(f"Iteration dir not found: {iter_dir}")
    if output_dir is not None and args.write:
        output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_files(
        repo_root=repo_root,
        iter_dir=iter_dir,
        files=args.files,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    if not files:
        print("No files found.")
        return

    if args.backend == "local":
        backend = LocalBackend(
            host=args.local_host,
            ports=_parse_ports(args.local_ports),
            model=args.local_model,
            timeout=args.request_timeout,
        )
    else:
        backend = ProviderBackend(provider=args.provider, model=args.provider_model)

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"Mode: {mode}; backend={args.backend}")

    for file_path in files:
        if not file_path.is_file():
            print(f"[SKIP] not a file: {file_path}")
            continue

        source_file = _pick_filter_source(file_path, filter_dir)
        if source_file is None:
            print(f"[SKIP] source not found for: {file_path.name}")
            continue

        rows = _load_records(file_path)
        source_index = _build_question_index(_load_records(source_file))

        total_rows = len(rows)
        if args.batch_size is not None:
            batch_start = args.batch_index * args.batch_size
            batch_end = min(total_rows, batch_start + args.batch_size)
            if batch_start >= total_rows:
                print(
                    f"[SKIP] batch out of range for {file_path.name}: "
                    f"start={batch_start}, total={total_rows}"
                )
                continue
            selected_indices = list(range(batch_start, batch_end))
        else:
            batch_start = 0
            batch_end = total_rows
            selected_indices = list(range(total_rows))

        if args.max_items is not None:
            selected_indices = selected_indices[: args.max_items]

        stats = {
            "rows_total": total_rows,
            "rows_selected": len(selected_indices),
            "batch_start": batch_start,
            "batch_end": batch_end,
            "samples_total": 0,
            "generated": 0,
            "kept": 0,
            "unmatched_question": 0,
            "llm_failed": 0,
        }

        for idx in selected_indices:
            row = rows[idx]
            if not isinstance(row, dict):
                continue
            stats["samples_total"] += 1

            existing = _normalize_text(row.get(args.output_field))
            if existing and not args.overwrite:
                stats["kept"] += 1
                continue

            question = _normalize_text(row.get("question"))
            modified_answer = _normalize_text(row.get("modified_answer") or row.get("answer"))
            if not question or not modified_answer:
                row[args.output_field] = "Insufficient input to generate reason."
                stats["generated"] += 1
                continue

            source_row = source_index.get(question)
            if source_row is None:
                stats["unmatched_question"] += 1
                row[args.output_field] = "No matched source question for criteria lookup."
                stats["generated"] += 1
                continue

            dim_scores = row.get("modified_dimension_scores")
            if not isinstance(dim_scores, list):
                dim_scores = []

            criteria_map = _extract_dim_criteria_map(source_row)
            dims_for_prompt: list[dict[str, Any]] = []
            for dim_row in dim_scores:
                if not isinstance(dim_row, dict):
                    continue
                name = _normalize_text(dim_row.get("dimension_name") or dim_row.get("name"))
                if not name:
                    continue
                key = _normalize_dim(name)
                criteria = _normalize_text(dim_row.get("full_score_criteria")) or criteria_map.get(key, "")
                dims_for_prompt.append(
                    {
                        "dimension_name": name,
                        "score": dim_row.get("score"),
                        "full_score_criteria": criteria,
                    }
                )

            prompt = _build_prompt(
                question=question,
                modified_answer=modified_answer,
                dims=dims_for_prompt,
            )

            try:
                reason_text = backend.generate(
                    prompt=prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
            except Exception as exc:  # pragma: no cover
                stats["llm_failed"] += 1
                reason_text = f"LLM generation failed: {exc}"

            row[args.output_field] = reason_text
            stats["generated"] += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"[OK] {file_path.name}")
        print(f"     source={source_file}")
        print(
            "     rows={rows_total}, selected={rows_selected}, "
            "batch_range=[{batch_start}, {batch_end}), "
            "processed={samples_total}, generated={generated}, kept={kept}, "
            "unmatched_q={unmatched_question}, llm_failed={llm_failed}".format(**stats)
        )

        if args.write:
            if args.inplace:
                out_file = file_path
            else:
                try:
                    relative = file_path.relative_to(iter_dir)
                    out_file = output_dir / relative  # type: ignore[operator]
                except ValueError:
                    out_file = output_dir / file_path.name  # type: ignore[operator]
                out_file.parent.mkdir(parents=True, exist_ok=True)
            _save_records(out_file, rows)
            print(f"     output={out_file}")


if __name__ == "__main__":
    main()
