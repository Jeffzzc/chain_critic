#!/usr/bin/env python
"""Small Gradio monitor for ms-swift GRPO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def tail_file(path: Path, max_lines: int) -> str:
    if not path.exists():
        return f"Waiting for log file: {path}"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return f"Could not read {path}: {exc}"
    return "".join(lines[-max_lines:]).rstrip() or f"{path} is empty."


def latest_trainer_state(output_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = [output_dir / "trainer_state.json"]
    candidates.extend(output_dir.glob("checkpoint-*/trainer_state.json"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None, None

    latest = max(existing, key=lambda path: path.stat().st_mtime)
    try:
        return latest, json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return latest, None


def gpu_status() -> list[list[str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return [["N/A", "nvidia-smi unavailable", "N/A", "N/A", "N/A"]]

    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            rows.append([parts[0], parts[1], f"{parts[2]}%", f"{parts[3]} / {parts[4]} MiB", f"{parts[5]} C"])
    return rows or [["N/A", "No GPU rows returned", "N/A", "N/A", "N/A"]]


def build_summary(output_dir: Path) -> tuple[str, list[list[Any]]]:
    state_path, state = latest_trainer_state(output_dir)
    if not state:
        source = str(state_path) if state_path else "not created yet"
        return f"### Training State\nWaiting for trainer state: `{source}`", []

    rows: list[list[Any]] = []
    for item in state.get("log_history", [])[-20:]:
        if isinstance(item, dict):
            rows.append(
                [
                    item.get("step", ""),
                    item.get("epoch", ""),
                    item.get("loss", ""),
                    item.get("reward", item.get("rewards/chosen", "")),
                    item.get("learning_rate", ""),
                ]
            )

    summary = [
        "### Training State",
        f"State file: `{state_path}`",
        f"Global step: `{state.get('global_step', 'N/A')}`",
        f"Epoch: `{state.get('epoch', 'N/A')}`",
        f"Best metric: `{state.get('best_metric', 'N/A')}`",
        f"Best checkpoint: `{state.get('best_model_checkpoint', 'N/A')}`",
    ]
    return "\n\n".join(summary), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Gradio monitor for GRPO training.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--embedding-log", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--refresh-seconds", type=float, default=5.0)
    parser.add_argument("--tail-lines", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("gradio is required for the training monitor. Install it with: pip install gradio") from exc

    args = parse_args()

    def refresh() -> tuple[str, list[list[Any]], list[list[str]], str, str]:
        summary, history = build_summary(args.output_dir)
        return (
            summary,
            history,
            gpu_status(),
            tail_file(args.train_log, args.tail_lines),
            tail_file(args.embedding_log, args.tail_lines),
        )

    with gr.Blocks(title="ChainCritic GRPO Monitor") as app:
        gr.Markdown("# ChainCritic GRPO Monitor")
        gr.Markdown(f"Output directory: `{args.output_dir}`")
        refresh_button = gr.Button("Refresh now")
        summary_box = gr.Markdown()
        history_table = gr.Dataframe(
            headers=["step", "epoch", "loss", "reward", "learning_rate"],
            label="Recent trainer metrics",
            interactive=False,
        )
        gpu_table = gr.Dataframe(
            headers=["index", "name", "utilization", "memory", "temperature"],
            label="GPU status",
            interactive=False,
        )
        train_log_box = gr.Textbox(label="Training log tail", lines=24, max_lines=40)
        embedding_log_box = gr.Textbox(label="Embedding server log tail", lines=12, max_lines=24)

        outputs = [summary_box, history_table, gpu_table, train_log_box, embedding_log_box]
        app.load(refresh, outputs=outputs, every=args.refresh_seconds)
        refresh_button.click(refresh, outputs=outputs)

    app.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
