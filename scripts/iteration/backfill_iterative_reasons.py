#!/usr/bin/env python
"""Backfill reason fields for GSM8K iteration outputs without regeneration.

Interfaces:
1) oneshot:
   Add ONE aggregated `reason` per sample for files under:
   datasets/GSM8K/iteration/Api/*.json
   (excluding single_dim)

2) single-dim:
   Add PER-DIMENSION `reason` for files under:
   datasets/GSM8K/iteration/Api/single_dim/*.json
   by writing reason into:
   - per_dimension_results[].reason
   - modified_dimension_scores[].reason

3) cleanup:
   Remove previously injected reason fields from output files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ONESHOT_SUFFIX = "_one-shot_score_direct"
SINGLE_DIM_SUFFIX = "_iterative_score_single_dim"


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
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_dim(value: Any) -> str:
    return _normalize_text(value).lower()


def _dataset_root_from_output(output_file: Path) -> Path | None:
    parts = list(output_file.parts)
    for idx, name in enumerate(parts):
        if name == "iteration" and idx > 0:
            return Path(*parts[:idx])
    return None


def _infer_base_stem(output_file: Path, mode: str) -> str:
    stem = output_file.stem
    if mode == "oneshot" and stem.endswith(ONESHOT_SUFFIX):
        return stem[: -len(ONESHOT_SUFFIX)]
    if mode == "single-dim" and stem.endswith(SINGLE_DIM_SUFFIX):
        return stem[: -len(SINGLE_DIM_SUFFIX)]
    return stem


def _candidate_sources(output_file: Path, mode: str) -> list[Path]:
    root = _dataset_root_from_output(output_file)
    if root is None:
        return []

    base = _infer_base_stem(output_file, mode)
    candidates = [
        root / "filter" / f"{base}.json",
        root / "filter" / f"{base}.jsonl",
        root / f"{base}.json",
        root / f"{base}.jsonl",
    ]
    if not any(path.exists() for path in candidates):
        candidates.extend(sorted(root.rglob(f"{base}.json")))
        candidates.extend(sorted(root.rglob(f"{base}.jsonl")))
    return candidates


def _pick_source(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _extract_dim_reason_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def append_items(items: Any, prefer_reason_only: bool) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            name = _normalize_text(item.get("dimension_name") or item.get("name"))
            key = _normalize_dim(name)
            if not key or key in seen:
                continue
            reason = _normalize_text(item.get("reason"))
            if not reason and not prefer_reason_only:
                criteria = _normalize_text(item.get("full_score_criteria"))
                if criteria:
                    reason = f"(fallback from criteria) {criteria}"
            if not reason:
                reason = "No explicit reason provided in source ratings."
            seen.add(key)
            pairs.append((name, reason))

    append_items(row.get("ratings"), prefer_reason_only=False)
    append_items(row.get("final_ratings"), prefer_reason_only=False)
    append_items(row.get("evaluation_dimensions"), prefer_reason_only=False)
    return pairs


def _build_question_index(rows: list[Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = _normalize_text(row.get("question"))
        if question and question not in index:
            index[question] = row
    return index


def _compose_aggregate_reason(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "No dimension-level reason found in source sample."
    chunks = [f"[{name}] {reason}" for name, reason in pairs]
    return " || ".join(chunks)


def _reason_map_from_pairs(pairs: list[tuple[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, reason in pairs:
        key = _normalize_dim(name)
        if key and key not in mapping:
            mapping[key] = reason
    return mapping


def _to_abs_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _collect_output_files(
    repo_root: Path,
    files: list[str],
    dirs: list[str],
    default_glob: str,
    dir_file_pattern: str,
) -> list[Path]:
    collected: list[Path] = []
    seen: set[str] = set()

    # Priority 1: explicit files
    if files:
        for raw in files:
            path = _to_abs_path(repo_root, raw)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                collected.append(path)
        return collected

    # Priority 2: explicit directories (recursive)
    if dirs:
        for raw_dir in dirs:
            base_dir = _to_abs_path(repo_root, raw_dir)
            if not base_dir.is_dir():
                continue
            for path in sorted(base_dir.rglob(dir_file_pattern)):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                collected.append(path)
        return collected

    # Priority 3: default glob
    for path in sorted(repo_root.glob(default_glob)):
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        collected.append(path)
    return collected


def _apply_oneshot(output_file: Path, source_file: Path, overwrite: bool) -> tuple[list[Any], dict[str, int]]:
    out_rows = _load_records(output_file)
    src_index = _build_question_index(_load_records(source_file))

    stats = {
        "samples_total": 0,
        "samples_unmatched": 0,
        "reason_written": 0,
        "reason_kept": 0,
    }

    for row in out_rows:
        if not isinstance(row, dict):
            continue
        stats["samples_total"] += 1

        if _normalize_text(row.get("reason")) and not overwrite:
            stats["reason_kept"] += 1
            continue

        q = _normalize_text(row.get("question"))
        src = src_index.get(q)
        if src is None:
            stats["samples_unmatched"] += 1
            row["reason"] = "No matched source sample by question."
            stats["reason_written"] += 1
            continue

        pairs = _extract_dim_reason_pairs(src)
        row["reason"] = _compose_aggregate_reason(pairs)
        stats["reason_written"] += 1

    return out_rows, stats


def _apply_single_dim(output_file: Path, source_file: Path, overwrite: bool) -> tuple[list[Any], dict[str, int]]:
    out_rows = _load_records(output_file)
    src_index = _build_question_index(_load_records(source_file))

    stats = {
        "samples_total": 0,
        "samples_unmatched": 0,
        "per_dim_total": 0,
        "per_dim_reason_written": 0,
        "per_dim_reason_kept": 0,
        "score_reason_written": 0,
    }

    for row in out_rows:
        if not isinstance(row, dict):
            continue
        stats["samples_total"] += 1
        q = _normalize_text(row.get("question"))
        src = src_index.get(q)
        if src is None:
            stats["samples_unmatched"] += 1
            reason_map: dict[str, str] = {}
        else:
            reason_map = _reason_map_from_pairs(_extract_dim_reason_pairs(src))

        per_dim = row.get("per_dimension_results")
        local_reason_map: dict[str, str] = {}
        if isinstance(per_dim, list):
            for dim_row in per_dim:
                if not isinstance(dim_row, dict):
                    continue
                stats["per_dim_total"] += 1
                dim_name = _normalize_text(dim_row.get("dimension_name")) or "unknown_dimension"
                key = _normalize_dim(dim_name)
                existing = _normalize_text(dim_row.get("reason"))
                if existing and not overwrite:
                    stats["per_dim_reason_kept"] += 1
                    local_reason_map[key] = existing
                    continue

                reason = reason_map.get(key)
                if not reason:
                    reason = f"No matched source reason for dimension '{dim_name}'."
                dim_row["reason"] = reason
                local_reason_map[key] = reason
                stats["per_dim_reason_written"] += 1

        score_rows = row.get("modified_dimension_scores")
        if isinstance(score_rows, list):
            for score_row in score_rows:
                if not isinstance(score_row, dict):
                    continue
                existing = _normalize_text(score_row.get("reason"))
                if existing and not overwrite:
                    continue
                dim_name = _normalize_text(score_row.get("dimension_name"))
                key = _normalize_dim(dim_name)
                if not key:
                    continue
                reason = local_reason_map.get(key) or reason_map.get(key)
                if reason:
                    score_row["reason"] = reason
                    stats["score_reason_written"] += 1

    return out_rows, stats


def _cleanup_reasons(output_file: Path) -> tuple[list[Any], dict[str, int]]:
    rows = _load_records(output_file)
    stats = {
        "samples_total": 0,
        "top_level_removed": 0,
        "per_dim_removed": 0,
        "score_removed": 0,
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        stats["samples_total"] += 1

        if "reason" in row:
            del row["reason"]
            stats["top_level_removed"] += 1

        per_dim = row.get("per_dimension_results")
        if isinstance(per_dim, list):
            for dim_row in per_dim:
                if isinstance(dim_row, dict) and "reason" in dim_row:
                    del dim_row["reason"]
                    stats["per_dim_removed"] += 1

        scores = row.get("modified_dimension_scores")
        if isinstance(scores, list):
            for score_row in scores:
                if isinstance(score_row, dict) and "reason" in score_row:
                    del score_row["reason"]
                    stats["score_removed"] += 1

    return rows, stats


def _run_oneshot(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    files = _collect_output_files(
        repo_root=repo_root,
        files=args.files,
        dirs=args.dirs,
        default_glob="datasets/GSM8K/iteration/Api/*_one-shot_score_direct.json",
        dir_file_pattern="*_one-shot_score_direct.json",
    )
    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"Mode: {mode}")

    for output_file in files:
        if not output_file.is_file():
            print(f"[SKIP] Not a file: {output_file}")
            continue
        source_file = _pick_source(_candidate_sources(output_file, mode="oneshot"))
        if source_file is None:
            print(f"[SKIP] Source not found: {output_file.name}")
            continue
        rows, stats = _apply_oneshot(output_file, source_file, overwrite=args.overwrite)
        print(f"[OK] {output_file.name}")
        print(f"     source={source_file}")
        print(
            "     samples={samples_total}, unmatched={samples_unmatched}, "
            "reason_written={reason_written}, reason_kept={reason_kept}".format(**stats)
        )
        if args.write:
            _save_records(output_file, rows)


def _run_single_dim(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    files = _collect_output_files(
        repo_root=repo_root,
        files=args.files,
        dirs=args.dirs,
        default_glob="datasets/GSM8K/iteration/Api/single_dim/*_iterative_score_single_dim.json",
        dir_file_pattern="*_iterative_score_single_dim.json",
    )
    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"Mode: {mode}")

    for output_file in files:
        if not output_file.is_file():
            print(f"[SKIP] Not a file: {output_file}")
            continue
        source_file = _pick_source(_candidate_sources(output_file, mode="single-dim"))
        if source_file is None:
            print(f"[SKIP] Source not found: {output_file.name}")
            continue
        rows, stats = _apply_single_dim(output_file, source_file, overwrite=args.overwrite)
        print(f"[OK] {output_file.name}")
        print(f"     source={source_file}")
        print(
            "     samples={samples_total}, unmatched={samples_unmatched}, per_dim={per_dim_total}, "
            "per_dim_written={per_dim_reason_written}, per_dim_kept={per_dim_reason_kept}, "
            "score_written={score_reason_written}".format(**stats)
        )
        if args.write:
            _save_records(output_file, rows)


def _run_cleanup(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    files = _collect_output_files(
        repo_root=repo_root,
        files=args.files,
        dirs=args.dirs,
        default_glob="datasets/GSM8K/iteration/Api/**/*.json",
        dir_file_pattern="*.json",
    )
    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"Mode: {mode}")

    for output_file in files:
        if not output_file.is_file():
            print(f"[SKIP] Not a file: {output_file}")
            continue
        rows, stats = _cleanup_reasons(output_file)
        removed = stats["top_level_removed"] + stats["per_dim_removed"] + stats["score_removed"]
        if removed == 0:
            continue
        print(f"[OK] {output_file.name}")
        print(
            "     samples={samples_total}, top_removed={top_level_removed}, "
            "per_dim_removed={per_dim_removed}, score_removed={score_removed}".format(**stats)
        )
        if args.write:
            _save_records(output_file, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill/cleanup reason fields for GSM8K iteration outputs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("oneshot", help="Generate ONE aggregate reason per sample for one-shot outputs.")
    p1.add_argument("--files", nargs="*", default=[], help="Specific files to process.")
    p1.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help="Directories to scan recursively for matching output files.",
    )
    p1.add_argument("--write", action="store_true", help="Write changes to files.")
    p1.add_argument("--overwrite", action="store_true", help="Overwrite existing reason.")
    p1.set_defaults(func=_run_oneshot)

    p2 = sub.add_parser("single-dim", help="Generate per-dimension reasons for iterative single-dim outputs.")
    p2.add_argument("--files", nargs="*", default=[], help="Specific files to process.")
    p2.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help="Directories to scan recursively for matching output files.",
    )
    p2.add_argument("--write", action="store_true", help="Write changes to files.")
    p2.add_argument("--overwrite", action="store_true", help="Overwrite existing reason.")
    p2.set_defaults(func=_run_single_dim)

    p3 = sub.add_parser("cleanup", help="Remove injected reason fields from iteration outputs.")
    p3.add_argument("--files", nargs="*", default=[], help="Specific files to process.")
    p3.add_argument(
        "--dirs",
        nargs="*",
        default=[],
        help="Directories to scan recursively for matching output files.",
    )
    p3.add_argument("--write", action="store_true", help="Write changes to files.")
    p3.set_defaults(func=_run_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
