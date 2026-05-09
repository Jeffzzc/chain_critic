#!/usr/bin/env bash
set -euo pipefail

# ========== Config ==========
# Input dataset directory (JSON/JSONL). Adjust to your real path.
INPUT_DIR="/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic/datasets/FinCoT/out"

# Output directory for merged dimensions
OUTPUT_DIR="/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic/datasets/FinCoT/dimensions"
OUTPUT_SUFFIX="_dimensions.json"

# Your python module for dimension generation (MUST exist): python -m <DIM_PY_MODULE>
# Examples: "scripts.parallel_analysis" / "scripts.parallel_dimensions"
DIM_PY_MODULE="scripts.dimensions.parallel_analysis"

# Local LLM servers
PORTS="${PORTS:-8000,8001,8002,8003,8004,8005,8006,8007}"
NUM_WORKERS="${NUM_WORKERS:-32}"
GPU_COUNT="${GPU_COUNT:-8}"

# Extra args passthrough (optional), e.g.:
# DIM_EXTRA_ARGS='--temperature 0.3 --timeout 120 --max-retries 5 --retry-delay 3'
DIM_EXTRA_ARGS="${DIM_EXTRA_ARGS:-}"

# ========== Checks ==========
if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
SHARD_DIR="${OUTPUT_DIR}/.shards"
mkdir -p "$SHARD_DIR"

# ========== Helpers ==========

# Detect whether the file is:
# - json: a single JSON list/dict
# - jsonl: many JSON objects (jsonl) OR json.load raises "Extra data"
# Prints: "<format> <count>"
detect_and_count() {
  local input_file="$1"
  INPUT_FILE="$input_file" python - <<'PY'
import os, json
from pathlib import Path
from json import JSONDecodeError

path = Path(os.environ["INPUT_FILE"])
text = path.read_text(encoding="utf-8", errors="replace")
s = text.lstrip()

def count_jsonl(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                n += 1
    return n

# Fast-path: if looks like json array and json.load works
try:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        print("json", len(data))
    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        print("json", len(data["data"]))
    else:
        # If it's a dict but not {"data": [...]}, treat as 1 sample (or change if you want)
        print("json", 1)
except JSONDecodeError as e:
    # "Extra data" usually means JSONL in a .json file
    # Also treat other decode errors as jsonl if line-based JSON objects are present
    fmt = "jsonl"
    cnt = count_jsonl(path)
    print(fmt, cnt)
PY
}

# Create a shard JSONL file containing records [start_line, end_line] (inclusive).
# This works for both JSON and JSONL inputs.
make_shard_jsonl() {
  local input_file="$1"
  local shard_file="$2"
  local start_line="$3"
  local end_line="$4"

  INPUT_FILE="$input_file" SHARD_FILE="$shard_file" START_LINE="$start_line" END_LINE="$end_line" python - <<'PY'
import os, json
from pathlib import Path
from json import JSONDecodeError

in_path = Path(os.environ["INPUT_FILE"])
out_path = Path(os.environ["SHARD_FILE"])
start_line = int(os.environ["START_LINE"])
end_line = int(os.environ["END_LINE"])

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_json(p: Path):
    with p.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    # fallback
    return [data]

items = None
try:
    # Try as standard JSON first
    items = load_json(in_path)
    slice_items = items[start_line:end_line+1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for obj in slice_items:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
except JSONDecodeError:
    # Treat as JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    idx = -1
    with out_path.open("w", encoding="utf-8") as w:
        for obj in iter_jsonl(in_path):
            idx += 1
            if idx < start_line:
                continue
            if idx > end_line:
                break
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
PY
}

# Merge shard outputs (each is a JSON list) into merged_output.
merge_shards() {
  local input_file="$1"
  local merged_output="$2"

  INPUT_FILE="$input_file" OUTPUT_DIR="$OUTPUT_DIR" OUTPUT_SUFFIX="$OUTPUT_SUFFIX" MERGED_OUTPUT="$merged_output" python - <<'PY'
import os, json, re
from pathlib import Path

input_file = Path(os.environ["INPUT_FILE"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_suffix = os.environ.get("OUTPUT_SUFFIX", "_dimensions.json")
merged_output = Path(os.environ["MERGED_OUTPUT"])

stem = input_file.stem
pattern = re.compile(rf"^{re.escape(stem)}_lines_(\d+)_(\d+){re.escape(output_suffix)}$")

shard_files = []
for p in output_dir.iterdir():
    if not p.is_file():
        continue
    m = pattern.match(p.name)
    if m:
        shard_files.append((int(m.group(1)), int(m.group(2)), p))

if not shard_files:
    print(f"No shard files found to merge in: {output_dir}")
    raise SystemExit(0)

shard_files.sort(key=lambda x: x[0])

merged = []
empty_files = 0
bad_files = 0
for _, _, p in shard_files:
    try:
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if not txt:
            empty_files += 1
            continue
        data = json.loads(txt)
        if isinstance(data, list):
            merged.extend(data)
        else:
            bad_files += 1
    except Exception:
        bad_files += 1

merged_output.parent.mkdir(parents=True, exist_ok=True)
with merged_output.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Merged {len(merged)} records into {merged_output}")
if empty_files:
    print(f"Skipped {empty_files} empty shard files.")
if bad_files:
    print(f"Skipped {bad_files} bad shard files.")

# Remove shard outputs after successful merge
deleted = 0
for _, _, p in shard_files:
    try:
        p.unlink()
        deleted += 1
    except Exception as exc:
        print(f"Failed to delete {p}: {exc}")
print(f"Deleted {deleted} shard output files.")
PY
}

# ========== Main ==========
shopt -s nullglob
input_files=("$INPUT_DIR"/*.json "$INPUT_DIR"/*.jsonl)
if [[ ${#input_files[@]} -eq 0 ]]; then
  echo "No JSON/JSONL files found in: $INPUT_DIR" >&2
  exit 1
fi

for INPUT_FILE in "${input_files[@]}"; do
  echo "Processing input: $INPUT_FILE"

  DETECT_RES="$(detect_and_count "$INPUT_FILE")"
  INPUT_FMT="$(echo "$DETECT_RES" | awk '{print $1}')"
  TOTAL_LINES="$(echo "$DETECT_RES" | awk '{print $2}')"

  if [[ "${TOTAL_LINES}" -le 0 ]]; then
    echo "Input file is empty: $INPUT_FILE" >&2
    continue
  fi

  echo "Detected format: ${INPUT_FMT}, samples: ${TOTAL_LINES}"

  TASKS_PER_GPU=$(( (TOTAL_LINES + GPU_COUNT - 1) / GPU_COUNT ))
  if [[ "$TASKS_PER_GPU" -le 0 ]]; then
    TASKS_PER_GPU=1
  fi

  # Launch per-GPU shard jobs
  for ((i=0; i<GPU_COUNT; i++)); do
    START_LINE=$((i * TASKS_PER_GPU))
    if [[ "$START_LINE" -ge "$TOTAL_LINES" ]]; then
      break
    fi

    END_LINE=$(((i + 1) * TASKS_PER_GPU - 1))
    if [[ "$END_LINE" -ge $((TOTAL_LINES - 1)) ]]; then
      END_LINE=$((TOTAL_LINES - 1))
    fi

    echo "Preparing shard for GPU $i (lines $START_LINE to $END_LINE)"
    SHARD_INPUT="${SHARD_DIR}/$(basename "${INPUT_FILE%.*}")_lines_${START_LINE}_${END_LINE}.jsonl"
    make_shard_jsonl "$INPUT_FILE" "$SHARD_INPUT" "$START_LINE" "$END_LINE"

    SHARD_OUTPUT="${OUTPUT_DIR}/$(basename "${INPUT_FILE%.*}")_lines_${START_LINE}_${END_LINE}${OUTPUT_SUFFIX}"
    echo "Starting task for GPU $i (lines $START_LINE to $END_LINE)"
    CUDA_VISIBLE_DEVICES=$i python -m "$DIM_PY_MODULE" --input "$SHARD_INPUT" --output "$SHARD_OUTPUT" --ports "$PORTS" --num-workers "$NUM_WORKERS" $DIM_EXTRA_ARGS &

    echo "GPU $i is processing shard: $SHARD_INPUT -> $SHARD_OUTPUT"
  done

  wait
  echo "All GPU tasks completed for: $INPUT_FILE"

  MERGED_OUTPUT="${OUTPUT_DIR}/$(basename "${INPUT_FILE%.*}")${OUTPUT_SUFFIX}"
  merge_shards "$INPUT_FILE" "$MERGED_OUTPUT"

  # Cleanup shard inputs
  rm -f "${SHARD_DIR}/$(basename "${INPUT_FILE%.*}")_lines_"*".jsonl" || true
  echo "Cleaned shard inputs under: $SHARD_DIR"
done
