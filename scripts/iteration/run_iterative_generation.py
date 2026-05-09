"""Run answer generation with manual config (Iterative or One-Shot)."""

import os
import sys
from pathlib import Path

# ===== Manual Config =====
# Strategy: "iterative" (fix one dimension at a time) or "one-shot" (fix all dimensions at once)
STRATEGY = "iterative"

# LLM provider (None for auto-detect, or "openai", "deepseek_v31", "doubao_seed_1.6", etc.)
PROVIDER = "zyuncs"

# Local multi-GPU OpenAI-compatible servers
LOCAL_HOST = "localhost"
LOCAL_PORTS = [8000, 8001, 8002, 8003]
LOCAL_MODEL = None  # e.g. "Qwen3-0.6B"; None => auto fetch from /v1/models
PREFER_LOCAL = True  # True => use local endpoints first
REQUEST_TIMEOUT = 120.0

# Input/Output files
INPUT_FILE = "datasets/LIMO/score/Qwen3-0.6B_result_score.json"
OUTPUT_FILE = f"datasets/LIMO/iteration/single_dim/Qwen3-0.6B_result_{STRATEGY}_score_prompt1_direct.json"

# Iteration settings
SAMPLE_SIZE = 4
MAX_SCORE = 5.0
MIN_SCORE = None  # Improve dimensions with score < MIN_SCORE (None => use MAX_SCORE)
MAX_DIMENSIONS = None  # Max dimensions per sample
ORDER = "score_asc"  # "score_asc" | "score_desc" | "original"
INCLUDE_NO_SCORE = True

# LLM settings
TEMPERATURE = 0.3
DELAY = 0.5
KEEP_INTERMEDIATE = False
NUM_WORKERS = 8  # Parallel samples; tune by GPU throughput
RERATE_AFTER_ITERATION = True  # Output repaired per-dimension scores
RATING_TEMPERATURE = 0.0  # None => use TEMPERATURE
OUTPUT_MODE = "simplified"  # Keep minimal output fields only

# =========================


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    return repo_root


def main() -> None:
    repo_root = _ensure_repo_on_path()

    from src.iteration_generation.iterative_generator import IterativeGenerator

    from src.iteration_generation.one_shot_generator import OneShotGenerator


    input_path = repo_root / INPUT_FILE
    output_path = repo_root / OUTPUT_FILE if OUTPUT_FILE else None

    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        return

    if STRATEGY == "one-shot" and OneShotGenerator is None:
        print("one-shot strategy requested, but src/iteration_generation/one_shot_generator.py is missing.")
        print("Switch STRATEGY to 'iterative' or add one_shot_generator.py first.")
        return

    print("=== Starting Generation ===")
    print(f"Strategy: {STRATEGY}")
    print(f"Provider: {PROVIDER}")
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")

    init_args = {
        "provider": PROVIDER,
        "max_score": MAX_SCORE,
        "local_host": LOCAL_HOST,
        "local_ports": LOCAL_PORTS,
        "local_model": LOCAL_MODEL,
        "request_timeout": REQUEST_TIMEOUT,
        "prefer_local": PREFER_LOCAL,
    }

    if STRATEGY == "one-shot":
        generator = OneShotGenerator(**init_args)
    else:
        generator = IterativeGenerator(**init_args)

    generator.generate_dataset(
        input_file=str(input_path),
        output_file=str(output_path) if output_path else None,
        sample_size=SAMPLE_SIZE,
        min_score_to_improve=MIN_SCORE,
        max_dimensions=MAX_DIMENSIONS,
        order=ORDER,
        include_no_score=INCLUDE_NO_SCORE,
        temperature=TEMPERATURE,
        delay_between_requests=DELAY,
        keep_intermediate=KEEP_INTERMEDIATE,
        num_workers=max(1, NUM_WORKERS),
        rerate_after_iteration=RERATE_AFTER_ITERATION,
        rating_temperature=RATING_TEMPERATURE,
        output_mode=OUTPUT_MODE,
    )


if __name__ == "__main__":
    main()
