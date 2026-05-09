# """Run batch answer generation over a folder using local OpenAI-compatible servers."""

# import os
# import socket
# import sys
# from pathlib import Path

# from tqdm import tqdm  # pip install tqdm

# # ===== Manual Config =====
# STRATEGY = "iterative"  # "one-shot" | "iterative"
# PROVIDER = None  # In local-only mode, keep provider empty to avoid unsupported fallback ids.

# # Force local-only mode (disable API fallback)
# PREFER_LOCAL = True
# FORCE_LOCAL_ONLY = True

# # Remote/local endpoint config:
# # - LOCAL_HOST: target host/IP where vLLM servers are deployed.
# # - LOCAL_PORTS: comma-separated ports, e.g. "8000,8001,8002".
# LOCAL_HOST = os.getenv("LOCAL_HOST", "localhost").strip()
# _ports_env = os.getenv("LOCAL_PORTS", "").strip()
# if _ports_env:
#     LOCAL_PORTS = [int(p.strip()) for p in _ports_env.split(",") if p.strip()]
# else:
#     LOCAL_PORTS = list(range(8000, 8008))
# GPU_COUNT = min(len(LOCAL_PORTS), int(os.getenv("GPU_COUNT", str(len(LOCAL_PORTS)))))  # max ports to use
# LOCAL_MODEL = "Qwen3-Omni-30B-A3B-Instruct"
# REQUEST_TIMEOUT = 120.0

# # Input/Output
# INPUT_DIR = "datasets/NuminaMath-CoT/filter"
# OUTPUT_DIR = "datasets/NuminaMath-CoT/iteration/Api/single_dim"
# INPUT_GLOBS = ["*.json", "*.jsonl"]

# # Process controls
# SAMPLE_SIZE = None
# MAX_SCORE = 5.0
# MIN_SCORE = None
# MAX_DIMENSIONS = None
# ORDER = "score_asc"
# INCLUDE_NO_SCORE = True
# FORCE_ALL_DIMENSIONS = True
# REWRITE_FULL_SCORE_DIMENSIONS = False
# REWRITE_NO_SCORE_DIMENSIONS = True

# # LLM settings
# TEMPERATURE = 0.2
# DELAY = 0.0
# KEEP_INTERMEDIATE = False
# NUM_WORKERS = 192
# AUTO_WORKERS_PER_GPU = 24

# # Independent per-dimension mode uses per_dimension_results as final scores.
# RERATE_AFTER_ITERATION = False
# RATING_TEMPERATURE = 0.0
# ENABLE_DIMENSION_REFLECTION = True
# REFLECTION_TEMPERATURE = 0.0
# OUTPUT_MODE = "simplified"

# # Runtime protections
# SKIP_IF_OUTPUT_EXISTS = True
# MIN_OUTPUT_BYTES_TO_SKIP = 200
# # =========================


# TEMPERATURE = 0.3
# DELAY = 0.3
# KEEP_INTERMEDIATE = False

# # 并发控制：8卡 * 24 workers = 192
# NUM_WORKERS = 192
# AUTO_WORKERS_PER_GPU = 24

# # one-shot 才用到的 inflight / retries（这里保留，不影响 iteration）
# MAX_INFLIGHT_PER_GPU = 24
# LOCAL_MAX_RETRIES = 1
# LOCAL_RETRY_BACKOFF = 0.2

# RERATE_AFTER_ITERATION = True
# RATING_TEMPERATURE = 0.0
# OUTPUT_MODE = "simplified"

# # ✅ 流式落盘开关（每个 sample 完成就写一行 jsonl）
# STREAM_WRITE = True
# STREAM_FSYNC_EVERY = 20  # 每 20 条做一次 os.fsync；0 表示只 flush 不 fsync

# SKIP_IF_OUTPUT_EXISTS = True
# MIN_OUTPUT_BYTES_TO_SKIP = 200
# # =========================


# def _ensure_repo_on_path() -> Path:
#     repo_root = Path(__file__).resolve().parents[2]
#     if not (repo_root / "src").exists():
#         raise RuntimeError(f"repo_root seems wrong: {repo_root} (no ./src found)")
#     sys.path.insert(0, str(repo_root))
#     return repo_root


# def _iter_input_files(input_dir: Path) -> list[Path]:
#     files: list[Path] = []
#     for pattern in INPUT_GLOBS:
#         files.extend(sorted(input_dir.rglob(pattern)))

#     uniq: list[Path] = []
#     seen = set()
#     for file_path in files:
#         resolved = str(file_path.resolve())
#         if resolved in seen or not file_path.is_file():
#             continue
#         seen.add(resolved)
#         uniq.append(file_path)
#     return uniq


# def _build_output_path(repo_root: Path, input_file: Path) -> Path:
#     out_dir = repo_root / OUTPUT_DIR
#     out_dir.mkdir(parents=True, exist_ok=True)
#     suffix = "single_dim" if STRATEGY == "iterative" else "direct"
#     out_name = f"{input_file.stem}_{STRATEGY}_score_{suffix}.json"
#     return out_dir / out_name


# def _should_skip_output(out_file: Path) -> bool:
#     if not SKIP_IF_OUTPUT_EXISTS:
#         return False
#     if not out_file.exists():
#         return False
#     try:
#         return out_file.stat().st_size >= MIN_OUTPUT_BYTES_TO_SKIP
#     except OSError:
#         return False


# def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
#     try:
#         with socket.create_connection((host, port), timeout=timeout):
#             return True
#     except OSError:
#         return False


# def _validate_local_config() -> list[int]:
#     selected_ports = LOCAL_PORTS[:GPU_COUNT]

#     if not PREFER_LOCAL:
#         raise RuntimeError("PREFER_LOCAL must be True when FORCE_LOCAL_ONLY=True.")
#     if not LOCAL_HOST or not selected_ports:
#         raise RuntimeError("LOCAL_HOST / LOCAL_PORTS must be set for local inference.")
#     if GPU_COUNT <= 0:
#         raise RuntimeError("GPU_COUNT must be positive.")
#     if len(selected_ports) < GPU_COUNT:
#         raise RuntimeError(
#             f"GPU_COUNT={GPU_COUNT} but only {len(selected_ports)} local ports configured: {selected_ports}."
#         )
#     if not LOCAL_MODEL:
#         raise RuntimeError("LOCAL_MODEL must be set to served-model-name to avoid fallback.")
#     return selected_ports


# def _discover_active_ports(selected_ports: list[int]) -> list[int]:
#     active = [p for p in selected_ports if _port_open(LOCAL_HOST, p)]
#     if not active:
#         raise RuntimeError(
#             f"No reachable local ports from configured set: {selected_ports}. "
#             f"Please make sure vLLM servers are running on {LOCAL_HOST}:<port>."
#         )
#     inactive = [p for p in selected_ports if p not in active]
#     if inactive:
#         print(f"[Config] Skip unreachable local ports: {inactive}. Use active ports: {active}")
#     return active


# def _disable_api_fallback_env() -> None:
#     """Best effort: keep this process in pure local mode."""
#     from src.common.llm_client import PROVIDER_CONFIGS

#     removed = []
#     if os.environ.pop("LLM_PROVIDER", None):
#         removed.append("LLM_PROVIDER")
#     for config in PROVIDER_CONFIGS.values():
#         api_key_env = config.get("api_key_env")
#         if api_key_env and os.environ.pop(api_key_env, None):
#             removed.append(api_key_env)
#     if removed:
#         print(f"[Config] FORCE_LOCAL_ONLY: cleared API env keys for this process: {sorted(set(removed))}")


# def main() -> None:
#     repo_root = _ensure_repo_on_path()

#     selected_ports = LOCAL_PORTS[:GPU_COUNT]
#     print(f"[Config] target_host={LOCAL_HOST}, configured_ports={selected_ports}")
#     if FORCE_LOCAL_ONLY:
#         selected_ports = _validate_local_config()
#         selected_ports = _discover_active_ports(selected_ports)

#     from src.iteration_generation.iterative_generator import IterativeGenerator
#     from src.iteration_generation.one_shot_generator import OneShotGenerator

#     if FORCE_LOCAL_ONLY:
#         _disable_api_fallback_env()

#     effective_workers = (len(selected_ports) * AUTO_WORKERS_PER_GPU) if NUM_WORKERS <= 0 else NUM_WORKERS

#     input_dir = repo_root / INPUT_DIR
#     if not input_dir.exists():
#         print(f"Input dir does not exist: {input_dir}")
#         return

#     input_files = _iter_input_files(input_dir)
#     if not input_files:
#         print(f"No input files found under: {input_dir} with patterns {INPUT_GLOBS}")
#         return

#     init_args = {
#         "provider": PROVIDER,
#         "max_score": MAX_SCORE,
#         "prefer_local": PREFER_LOCAL,
#         "local_host": LOCAL_HOST,
#         "local_ports": selected_ports,
#         "local_model": LOCAL_MODEL,
#         "request_timeout": REQUEST_TIMEOUT,
#     }

#     if STRATEGY == "iterative":
#         generator = IterativeGenerator(**init_args)
#     else:
#         generator = OneShotGenerator(**init_args)

#     if STRATEGY == "iterative" and FORCE_ALL_DIMENSIONS:
#         print(
#             "[Config] FORCE_ALL_DIMENSIONS is ON: all dimensions will be iterated; "
#             f"rewrite_full_score={REWRITE_FULL_SCORE_DIMENSIONS}, "
#             f"rewrite_no_score={REWRITE_NO_SCORE_DIMENSIONS}, "
#             f"enable_reflection={ENABLE_DIMENSION_REFLECTION}, "
#             f"reflection_temperature={REFLECTION_TEMPERATURE}, "
#             f"rerate_after_iteration={RERATE_AFTER_ITERATION}, "
#             f"active_ports={selected_ports}, "
#             f"num_workers={max(1, effective_workers)}"
#         )

#     file_pbar = tqdm(input_files, desc="Files", unit="file", dynamic_ncols=True)

#     for in_file in file_pbar:
#         out_file = _build_output_path(repo_root, in_file)
#         file_pbar.set_postfix_str(in_file.name)

#         if _should_skip_output(out_file):
#             tqdm.write(f"[SKIP] {in_file.name} -> {out_file.name} (exists)")
#             continue

#         sample_pbar = tqdm(
#             total=0,
#             desc=f"Samples ({in_file.name})",
#             unit="sample",
#             dynamic_ncols=True,
#             leave=False,
#         )
#         last_done = 0

#         def progress_cb(done: int, total: int) -> None:
#             nonlocal last_done
#             if sample_pbar.total == 0 and total > 0:
#                 sample_pbar.total = total
#                 sample_pbar.refresh()
#             inc = done - last_done
#             if inc > 0:
#                 sample_pbar.update(inc)
#                 last_done = done

#         try:
#             force_all_dimensions = (
#                 STRATEGY == "iterative" and (FORCE_ALL_DIMENSIONS or ENABLE_DIMENSION_REFLECTION)
#             )
#             effective_min_score = None if force_all_dimensions else MIN_SCORE
#             effective_include_no_score = True if force_all_dimensions else INCLUDE_NO_SCORE

#             call_kwargs = {
#                 "input_file": str(in_file),
#                 "output_file": str(out_file),
#                 "sample_size": SAMPLE_SIZE,
#                 "min_score_to_improve": effective_min_score,
#                 "max_dimensions": MAX_DIMENSIONS,
#                 "order": ORDER,
#                 "include_no_score": effective_include_no_score,
#                 "temperature": TEMPERATURE,
#                 "delay_between_requests": DELAY,
#                 "keep_intermediate": KEEP_INTERMEDIATE,
#                 "num_workers": max(1, effective_workers),
#                 "rerate_after_iteration": RERATE_AFTER_ITERATION,
#                 "rating_temperature": RATING_TEMPERATURE,
#                 "enable_dimension_reflection": ENABLE_DIMENSION_REFLECTION,
#                 "reflection_temperature": REFLECTION_TEMPERATURE,
#                 "output_mode": OUTPUT_MODE,
#                 "progress_cb": progress_cb,
#             }

#             if STRATEGY == "iterative":
#                 call_kwargs["force_all_dimensions"] = force_all_dimensions
#                 call_kwargs["rewrite_full_score_dimensions"] = REWRITE_FULL_SCORE_DIMENSIONS
#                 call_kwargs["rewrite_no_score_dimensions"] = REWRITE_NO_SCORE_DIMENSIONS

#             generator.generate_dataset(**call_kwargs)
#         finally:
#             sample_pbar.close()

#     print("=== All done ===")


# if __name__ == "__main__":
#     main()



"""Run answer generation over a folder (Iterative or One-Shot) with tqdm (file-level + per-file sample-level).
Force using local vLLM OpenAI-compatible servers only.
"""

import sys
import socket
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

# ===== Manual Config =====
STRATEGY = "iteration"

# Force local-only mode (disable any remote/provider fallback)
PROVIDER = "local"

LOCAL_HOST = "localhost"
LOCAL_PORTS = list(range(8000, 8008))  # 8000-8007
GPU_COUNT = 8

# Force served-model-name to avoid auto-selection / fallback
LOCAL_MODEL = "Qwen3-Omni-30B-A3B-Instruct"

PREFER_LOCAL = True
FORCE_LOCAL_ONLY = True

REQUEST_TIMEOUT = 60.0

INPUT_DIR = "datasets/QwQ-LongCoT-130K/filter"
OUTPUT_DIR = "datasets/QwQ-LongCoT-130K/iteration/Api/single_dim"
INPUT_GLOBS = ["*.json", "*.jsonl"]

SAMPLE_SIZE = None
MAX_SCORE = 5.0
MIN_SCORE = None
MAX_DIMENSIONS = None
ORDER = "score_asc"
INCLUDE_NO_SCORE = True

TEMPERATURE = 0.3
DELAY = 0.2
KEEP_INTERMEDIATE = False

# 并发控制：8卡 * 24 workers = 192
NUM_WORKERS = 192
AUTO_WORKERS_PER_GPU = 24

# max in-flight requests per local port; <=0 means unlimited
MAX_INFLIGHT_PER_GPU = 24
LOCAL_MAX_RETRIES = 1
LOCAL_RETRY_BACKOFF = 0.2

RERATE_AFTER_ITERATION = True
RATING_TEMPERATURE = 0.0
OUTPUT_MODE = "simplified"

SKIP_IF_OUTPUT_EXISTS = True
MIN_OUTPUT_BYTES_TO_SKIP = 200
# =========================


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    return repo_root


def _iter_input_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pat in INPUT_GLOBS:
        files.extend(sorted(input_dir.rglob(pat)))

    uniq: list[Path] = []
    seen = set()
    for p in files:
        rp = str(p.resolve())
        if rp not in seen and p.is_file():
            seen.add(rp)
            uniq.append(p)
    return uniq


def _build_output_path(repo_root: Path, input_file: Path) -> Path:
    out_dir = repo_root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{input_file.stem}_{STRATEGY}_score_direct.json"
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


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _assert_local_ready() -> None:
    selected_ports = LOCAL_PORTS[:GPU_COUNT]

    if not PREFER_LOCAL:
        raise RuntimeError("PREFER_LOCAL must be True when FORCE_LOCAL_ONLY=True.")

    if not LOCAL_HOST or not selected_ports:
        raise RuntimeError("LOCAL_HOST / LOCAL_PORTS must be set for local inference.")

    if GPU_COUNT <= 0:
        raise RuntimeError("GPU_COUNT must be positive.")
    if len(selected_ports) < GPU_COUNT:
        raise RuntimeError(
            f"GPU_COUNT={GPU_COUNT} but only {len(selected_ports)} local ports configured: {selected_ports}."
        )

    if not LOCAL_MODEL:
        raise RuntimeError("LOCAL_MODEL must be set to served-model-name to avoid fallback.")

    bad = [p for p in selected_ports if not _port_open(LOCAL_HOST, p)]
    if bad:
        raise RuntimeError(
            f"FORCE_LOCAL_ONLY=True, but local ports are not reachable: {bad}. "
            f"Please make sure vLLM servers are running on {LOCAL_HOST}:{selected_ports}."
        )


def main() -> None:
    repo_root = _ensure_repo_on_path()

    # Hard check: all local ports must be reachable when FORCE_LOCAL_ONLY=True
    if FORCE_LOCAL_ONLY:
        _assert_local_ready()

    from src.iteration_generation.iterative_generator import IterativeGenerator
    from src.iteration_generation.one_shot_generator import OneShotGenerator

    input_dir = repo_root / INPUT_DIR
    if not input_dir.exists():
        print(f"Input dir does not exist: {input_dir}")
        return

    input_files = _iter_input_files(input_dir)
    if not input_files:
        print(f"No input files found under: {input_dir} with patterns {INPUT_GLOBS}")
        return

    if STRATEGY == "one-shot" and OneShotGenerator is None:
        print("one-shot strategy requested, but src/iteration_generation/one_shot_generator.py is missing.")
        return

    init_args = {
        "provider": PROVIDER,
        "max_score": MAX_SCORE,
        "local_host": LOCAL_HOST,
        "local_ports": LOCAL_PORTS[:GPU_COUNT],  # 8000-8007
        "local_model": LOCAL_MODEL,
        "request_timeout": REQUEST_TIMEOUT,
        "prefer_local": PREFER_LOCAL,
    }

    if STRATEGY == "one-shot":
        init_args.update({
            "auto_workers_per_port": AUTO_WORKERS_PER_GPU,
            "local_max_inflight_per_port": MAX_INFLIGHT_PER_GPU,
            "local_max_retries": LOCAL_MAX_RETRIES,
            "local_retry_backoff": LOCAL_RETRY_BACKOFF,
        })

    generator = OneShotGenerator(**init_args) if STRATEGY == "one-shot" else IterativeGenerator(**init_args)

    if STRATEGY == "one-shot":
        configured_workers = "auto" if NUM_WORKERS <= 0 else str(NUM_WORKERS)
        print(
            "Parallel setup: "
            f"gpus={GPU_COUNT}, ports={init_args['local_ports']}, "
            f"num_workers={configured_workers}, "
            f"auto_workers_per_gpu={AUTO_WORKERS_PER_GPU}, "
            f"max_inflight_per_gpu={MAX_INFLIGHT_PER_GPU}"
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
            generator.generate_dataset(
                input_file=str(in_file),
                output_file=str(out_file),
                sample_size=SAMPLE_SIZE,
                min_score_to_improve=MIN_SCORE,
                max_dimensions=MAX_DIMENSIONS,
                order=ORDER,
                include_no_score=INCLUDE_NO_SCORE,
                temperature=TEMPERATURE,
                delay_between_requests=DELAY,
                keep_intermediate=KEEP_INTERMEDIATE,
                num_workers=NUM_WORKERS if STRATEGY == "one-shot" else max(1, NUM_WORKERS),
                rerate_after_iteration=RERATE_AFTER_ITERATION,
                rating_temperature=RATING_TEMPERATURE,
                output_mode=OUTPUT_MODE,
                progress_cb=progress_cb,
            )
        finally:
            sample_pbar.close()

    print("=== All done ===")


if __name__ == "__main__":
    main()