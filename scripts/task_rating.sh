#!/usr/bin/env bash
set -euo pipefail

# Paths
DIMENSIONS_DIR="/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic/datasets/GSM8K/dimensions"
OUTPUT_DIR="/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic/datasets/GSM8K/score"
OUTPUT_SUFFIX="_scores.json"

# Parallel config
PORTS="8000,8001,8002,8003,8004,8005,8006,8007"
NUM_WORKERS="${NUM_WORKERS:-32}"
GPU_COUNT=8

if [[ ! -d "$DIMENSIONS_DIR" ]]; then
  echo "Dimensions directory not found: $DIMENSIONS_DIR" >&2
  exit 1
fi

count_samples() {
  local input_file="$1"
  if [[ "$input_file" == *.jsonl ]]; then
    wc -l < "$input_file"
  else
    INPUT_FILE="$input_file" python - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["INPUT_FILE"])
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data))
PY
  fi
}

merge_shards() {
  local input_file="$1"
  local merged_output="$2"
  INPUT_FILE="$input_file" OUTPUT_DIR="$OUTPUT_DIR" OUTPUT_SUFFIX="$OUTPUT_SUFFIX" MERGED_OUTPUT="$merged_output" python - <<'PY'
import json
import os
import re
from pathlib import Path

input_file = Path(os.environ["INPUT_FILE"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_suffix = os.environ.get("OUTPUT_SUFFIX", "_scores.json")
merged_output = Path(os.environ["MERGED_OUTPUT"])

stem = input_file.stem
pattern = re.compile(rf"^{re.escape(stem)}_lines_(\d+)_(\d+){re.escape(output_suffix)}$")

shard_files = []
for path in output_dir.iterdir():
    if not path.is_file():
        continue
    match = pattern.match(path.name)
    if match:
        shard_files.append((int(match.group(1)), int(match.group(2)), path))

if not shard_files:
    print(f"No shard files found to merge in: {output_dir}")
    raise SystemExit(0)

shard_files.sort(key=lambda x: x[0])

merged = []
empty_files = 0
deleted_files = 0
for _, _, path in shard_files:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            empty_files += 1
            continue
        data = json.loads(text)
        if isinstance(data, list) and data:
            merged.extend(data)
        elif isinstance(data, list):
            empty_files += 1
    except Exception as exc:
        print(f"Skipping {path} due to error: {exc}")

merged_output.parent.mkdir(parents=True, exist_ok=True)
with merged_output.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Merged {len(merged)} records into {merged_output}")
if empty_files:
    print(f"Skipped {empty_files} empty shard files.")

# Remove shard files after successful merge
for _, _, path in shard_files:
    try:
        path.unlink()
        deleted_files += 1
    except Exception as exc:
        print(f"Failed to delete {path}: {exc}")
print(f"Deleted {deleted_files} shard files.")
PY
}

shopt -s nullglob
input_files=("$DIMENSIONS_DIR"/*.json "$DIMENSIONS_DIR"/*.jsonl)
if [[ ${#input_files[@]} -eq 0 ]]; then
  echo "No JSON/JSONL files found in: $DIMENSIONS_DIR" >&2
  exit 1
fi

for INPUT_FILE in "${input_files[@]}"; do
  echo "Processing input: $INPUT_FILE"

  TOTAL_LINES=$(count_samples "$INPUT_FILE")
  if [[ "$TOTAL_LINES" -le 0 ]]; then
    echo "Input file is empty: $INPUT_FILE" >&2
    continue
  fi

  TASKS_PER_GPU=$(( (TOTAL_LINES + GPU_COUNT - 1) / GPU_COUNT ))
  if [[ "$TASKS_PER_GPU" -le 0 ]]; then
    TASKS_PER_GPU=1
  fi

  for ((i=0; i<GPU_COUNT; i++)); do
    START_LINE=$((i * TASKS_PER_GPU))
    if [[ "$START_LINE" -ge "$TOTAL_LINES" ]]; then
      break
    fi

    END_LINE=$(((i + 1) * TASKS_PER_GPU - 1))
    if [[ "$END_LINE" -ge $((TOTAL_LINES - 1)) ]]; then
      END_LINE=$((TOTAL_LINES - 1))
    fi

    echo "Starting task for GPU $i (lines $START_LINE to $END_LINE)"
    CUDA_VISIBLE_DEVICES=$i python -m scripts.parallel_rating \
      --input "$INPUT_FILE" \
      --out-dir "$OUTPUT_DIR" \
      --ports "$PORTS" \
      --num-workers "$NUM_WORKERS" \
      --output-suffix "$OUTPUT_SUFFIX" \
      --start-line "$START_LINE" \
      --end-line "$END_LINE" \
      --gpu-count "$GPU_COUNT" &

    echo "GPU $i is processing lines $START_LINE to $END_LINE"
  done

  wait
  echo "All GPU tasks completed for: $INPUT_FILE"

  MERGED_OUTPUT="${OUTPUT_DIR}/$(basename "${INPUT_FILE%.*}")_score.json"
  merge_shards "$INPUT_FILE" "$MERGED_OUTPUT"
done
