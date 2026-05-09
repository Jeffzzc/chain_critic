#!/bin/bash
set -euo pipefail

# ================= ⚙️ 配置区域 =================
PROJECT_ROOT="/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic"

INPUT_DIR="${PROJECT_ROOT}/datasets/NuminaMath-CoT/dimensions"  # 输入目录
OUTPUT_DIR="${PROJECT_ROOT}/datasets/NuminaMath-CoT/score"      # 输出目录

PROVIDER="zyuncs"
SAMPLE_SIZE=9999999

SCRIPT_PATH="src/rating/main.py"  # 使用评分脚本路径
# ===========================================

mkdir -p "$OUTPUT_DIR"

# 0) 依赖当前 shell 环境（你已手动 conda activate）
#    可选：确保 python / pip / which 都来自你当前环境
echo "=================================================="
echo "🔎 Using current shell environment"
echo "which python: $(which python || true)"
echo "python -V   : $(python -V 2>/dev/null || true)"
echo "=================================================="

# 1) 收集任务列表（只收集 *.json 文件）
TASKS=()
shopt -s nullglob
for input_file in "$INPUT_DIR"/*.json; do  # 修改为 *.json 文件
  filename="$(basename "$input_file")"
  TASKS+=("$input_file")  # 直接加入任务列表，不进行过滤
done
shopt -u nullglob

echo "TASKS array: ${TASKS[@]}"

NUM_TASKS=${#TASKS[@]}
if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "❌ No tasks found under: $INPUT_DIR"
  echo "   (Make sure *.json files exist in the input directory)"
  exit 1
fi

echo "=================================================="
echo "📺 Initializing LIMO Rating Runner"
echo "📂 Input Dir : $INPUT_DIR"
echo "💾 Output Dir: $OUTPUT_DIR"
echo "🤖 Provider  : $PROVIDER"
echo "🧩 Script    : $SCRIPT_PATH"
echo "🧮 Tasks     : $NUM_TASKS"
echo "=================================================="

# 2) 进入项目目录，设置 PYTHONPATH（避免相对导入问题）
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# 3) 顺序执行每个任务
for i in "${!TASKS[@]}"; do
  input_file="${TASKS[$i]}"
  filename="$(basename "$input_file")"
  filename_no_ext="${filename%.*}"
  output_file="${OUTPUT_DIR}/${filename_no_ext}_rating.json"  # 输出结果的文件名修改为 '_rating.json'

  echo "Configuring task $i for: $filename"

  echo "Full path: $input_file"
  if [[ ! -f "$input_file" ]]; then
    echo "❌ File does not exist: $input_file"
    continue
  fi

  PY_CMD=(python "$SCRIPT_PATH" --provider "$PROVIDER" --input "$input_file" --output "$output_file")
  if [[ -n "${SAMPLE_SIZE:-}" ]]; then
    PY_CMD+=(--sample-size "$SAMPLE_SIZE")
  fi

  echo "🚀 Running task: $filename"
  echo "IN : $input_file"
  echo "OUT: $output_file"
  echo "CMD: ${PY_CMD[*]}"
  echo "----------------"
  "${PY_CMD[@]}"
  echo "----------------"
  echo "✅ Done with $filename"
done

echo "---------------------------------------------------"
echo "✅ All $NUM_TASKS tasks completed."
echo "---------------------------------------------------"
