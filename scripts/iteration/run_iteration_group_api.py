"""Run batch answer generation over a folder using API provider only.

Supports both iterative and one-shot strategies. For iterative mode, output is now
single-dimension independent revisions (each dimension starts from original QA).
"""

import sys
from pathlib import Path

from tqdm import tqdm  # pip install tqdm

# ===== Manual Config =====
STRATEGY = "iterative"  # "one-shot" | "iterative"
PROVIDER = "deepseek_v31"  # API provider name

# Force API-only mode (disable local OpenAI-compatible servers)
FORCE_API_ONLY = True

# Input/Output
INPUT_DIR = "datasets\\NuminaMath-CoT\\filter"
OUTPUT_DIR = "datasets\\NuminaMath-CoT\\iteration\\Api\\single_dim"
INPUT_GLOBS = ["*.json", "*.jsonl"]

# Process controls
SAMPLE_SIZE = 1
MAX_SCORE = 5.0
MIN_SCORE = None
MAX_DIMENSIONS = None
ORDER = "score_asc"
INCLUDE_NO_SCORE = True
# If True (iterative mode), force processing every available dimension in each sample.
FORCE_ALL_DIMENSIONS = True
# Legacy gates. In one-step reflection mode, an "unreasonable" judgment will still trigger revision.
REWRITE_FULL_SCORE_DIMENSIONS = False
REWRITE_NO_SCORE_DIMENSIONS = True

# LLM settings
TEMPERATURE = 0.3
DELAY = 0.1
KEEP_INTERMEDIATE = False
NUM_WORKERS = 10
# Independent per-dimension mode uses per_dimension_results as final scores; no extra rerate needed.
RERATE_AFTER_ITERATION = False
RATING_TEMPERATURE = 0.0
ENABLE_DIMENSION_REFLECTION = True
REFLECTION_TEMPERATURE = 0.0
# In iterative mode, simplified output includes: per_dimension_results
OUTPUT_MODE = "simplified"

# Runtime protections
SKIP_IF_OUTPUT_EXISTS = True
MIN_OUTPUT_BYTES_TO_SKIP = 200
# =========================


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "src").exists():
        raise RuntimeError(f"repo_root seems wrong: {repo_root} (no ./src found)")
    sys.path.insert(0, str(repo_root))
    return repo_root


def _iter_input_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in INPUT_GLOBS:
        files.extend(sorted(input_dir.rglob(pattern)))

    uniq: list[Path] = []
    seen = set()
    for file_path in files:
        resolved = str(file_path.resolve())
        if resolved in seen or not file_path.is_file():
            continue
        seen.add(resolved)
        uniq.append(file_path)
    return uniq


def _build_output_path(repo_root: Path, input_file: Path) -> Path:
    out_dir = repo_root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "single_dim" if STRATEGY == "iterative" else "direct"
    out_name = f"{input_file.stem}_{STRATEGY}_score_{suffix}.json"
    return out_dir / out_name


def _should_skip_output(out_file: Path) -> bool:
    if not SKIP_IF_OUTPUT_EXISTS:
        return False
    if not out_file.exists():
        return False
    try:
        return out_file.stat().st_size >= MIN_OUTPUT_BYTES_TO_SKIP
    except OSError:
        return False


def main() -> None:
    repo_root = _ensure_repo_on_path()

    from src.common.llm_client import list_providers
    from src.iteration_generation.iterative_generator import IterativeGenerator
    from src.iteration_generation.one_shot_generator import OneShotGenerator

    supported_providers = set(list_providers())
    if PROVIDER is not None and PROVIDER not in supported_providers:
        raise ValueError(
            f"Unsupported PROVIDER={PROVIDER!r}. "
            f"Supported providers: {sorted(supported_providers)}. "
            "Set PROVIDER to a supported value or None to use LLM_PROVIDER from environment."
        )

    input_dir = repo_root / INPUT_DIR
    if not input_dir.exists():
        print(f"Input dir does not exist: {input_dir}")
        return

    input_files = _iter_input_files(input_dir)
    if not input_files:
        print(f"No input files found under: {input_dir} with patterns {INPUT_GLOBS}")
        return

    init_args = {
        "provider": PROVIDER,
        "max_score": MAX_SCORE,
    }

    if FORCE_API_ONLY:
        init_args.update(
            {
                "prefer_local": False,
                "local_host": "localhost",
                "local_ports": [],
                "local_model": None,
                "request_timeout": 120.0,
            }
        )

    if STRATEGY == "iterative":
        generator = IterativeGenerator(**init_args)
    else:
        generator = OneShotGenerator(**init_args)

    if STRATEGY == "iterative" and FORCE_ALL_DIMENSIONS:
        print(
            "[Config] FORCE_ALL_DIMENSIONS is ON: all dimensions will be iterated; "
            f"rewrite_full_score={REWRITE_FULL_SCORE_DIMENSIONS}, "
            f"rewrite_no_score={REWRITE_NO_SCORE_DIMENSIONS}"
        )

    file_pbar = tqdm(input_files, desc="Files", unit="file", dynamic_ncols=True)

    for in_file in file_pbar:
        out_file = _build_output_path(repo_root, in_file)
        file_pbar.set_postfix_str(in_file.name)

        if _should_skip_output(out_file):
            tqdm.write(f"[SKIP] {in_file.name} -> {out_file.name} (exists)")
            continue

        sample_pbar = tqdm(
            total=0,
            desc=f"Samples ({in_file.name})",
            unit="sample",
            dynamic_ncols=True,
            leave=False,
        )
        last_done = 0

        def progress_cb(done: int, total: int) -> None:
            nonlocal last_done
            if sample_pbar.total == 0 and total > 0:
                sample_pbar.total = total
                sample_pbar.refresh()
            inc = done - last_done
            if inc > 0:
                sample_pbar.update(inc)
                last_done = done

        try:
            # force_all_dimensions = STRATEGY == "iterative" and FORCE_ALL_DIMENSIONS
            # effective_min_score = MIN_SCORE
            # effective_include_no_score = True if force_all_dimensions else INCLUDE_NO_SCORE

            # one-step reflection 开启时，强制让所有维度进入 per-dim loop（否则部分维度不会被审计）
            force_all_dimensions = (
                STRATEGY == "iterative" and (FORCE_ALL_DIMENSIONS or ENABLE_DIMENSION_REFLECTION)
            )

            effective_min_score = None if force_all_dimensions else MIN_SCORE

            # force_all_dimensions 开启时，include_no_score 必须为 True，避免无分维度被静默丢掉
            effective_include_no_score = True if force_all_dimensions else INCLUDE_NO_SCORE
            call_kwargs = {
                "input_file": str(in_file),
                "output_file": str(out_file),
                "sample_size": SAMPLE_SIZE,
                "min_score_to_improve": effective_min_score, # None
                "max_dimensions": MAX_DIMENSIONS, # None
                "order": ORDER,
                "include_no_score": effective_include_no_score, # True
                "temperature": TEMPERATURE,
                "delay_between_requests": DELAY,
                "keep_intermediate": KEEP_INTERMEDIATE,
                "num_workers": max(1, NUM_WORKERS),
                "rerate_after_iteration": RERATE_AFTER_ITERATION, # False
                "rating_temperature": RATING_TEMPERATURE,
                "enable_dimension_reflection": ENABLE_DIMENSION_REFLECTION, # True
                "reflection_temperature": REFLECTION_TEMPERATURE, # 0.0
                "output_mode": OUTPUT_MODE, # "simplified"
                "progress_cb": progress_cb,
            }
            if STRATEGY == "iterative":
                call_kwargs["force_all_dimensions"] = force_all_dimensions
                call_kwargs["rewrite_full_score_dimensions"] = REWRITE_FULL_SCORE_DIMENSIONS # False
                call_kwargs["rewrite_no_score_dimensions"] = REWRITE_NO_SCORE_DIMENSIONS # True

            generator.generate_dataset(**call_kwargs)
        finally:
            sample_pbar.close()

    print("=== All done ===")


if __name__ == "__main__":
    main()
