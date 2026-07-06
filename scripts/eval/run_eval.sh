#!/bin/bash
# Evaluate a model on math benchmarks using SC@21 (self-consistency with 21 samples)
# Usage: bash scripts/eval/run_eval.sh <model_path> [benchmark|all] [prompt_type]
#
# Examples:
#   bash scripts/eval/run_eval.sh Qwen/Qwen2.5-Math-1.5B gsm8k
#   bash scripts/eval/run_eval.sh outputs/core_cr/global_step_24/actor/huggingface all
#   bash scripts/eval/run_eval.sh outputs/core_cr/global_step_24/actor/huggingface math qwen-boxed

set -e

MODEL_PATH=${1:?"Usage: $0 <model_path> [benchmark|all] [prompt_type]"}
BENCHMARK=${2:-"all"}
PROMPT_TYPE=${3:-"qwen-boxed"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
EVAL_SCRIPT="${SCRIPT_DIR}/../../evaluation/sh/eval_selfconsistent.sh"

if [ "$BENCHMARK" = "all" ]; then
    exec bash "${EVAL_SCRIPT}" "${PROMPT_TYPE}" "${MODEL_PATH}"
fi

# --- Single benchmark mode ---
# Determine per-benchmark parameters
EXTRA_ARGS=""
case "$BENCHMARK" in
    # group 1: English open-ended (no extra args)
    gsm8k|math|svamp|asdiv|mawps|tabmwp|minerva_math|gaokao2023en|olympiadbench)
        EXTRA_ARGS=""
        ;;
    # group 2: English multiple-choice (5-shot)
    mmlu_stem)
        EXTRA_ARGS="--num_shots 5"
        ;;
    # group 3: Chinese benchmarks (adaptive few-shot, lower GPU memory)
    gaokao_math_qa|cmath)
        EXTRA_ARGS="--adapt_few_shot --gpu_memory_utilization 0.3"
        ;;
    *)
        echo "Error: Unknown benchmark '$BENCHMARK'"
        echo "Supported benchmarks: gsm8k math asdiv mawps tabmwp svamp mmlu_stem gaokao2023en gaokao_math_qa cmath minerva_math olympiadbench"
        echo "Use 'all' to run all 12 benchmarks."
        exit 1
        ;;
esac

# Set up environment and paths (same as eval_selfconsistent.sh)
EVAL_DIR=$(cd -- "${SCRIPT_DIR}/../../evaluation" &> /dev/null && pwd)
DATA_DIR="${EVAL_DIR}/data"
CACHE_DIR="${EVAL_DIR}/sh/.cache"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

OUTPUT_DIR="${MODEL_PATH}/math_eval_selfconsistent_n21"

echo "Running single benchmark: ${BENCHMARK}"
echo "    model path: ${MODEL_PATH}"
echo "    prompt type: ${PROMPT_TYPE}"
echo ""

TOKENIZERS_PARALLELISM=false \
PYTHONPATH="${EVAL_DIR}:$PYTHONPATH" \
python3 -u "${EVAL_DIR}/math_eval.py" \
    --model_name_or_path "$MODEL_PATH" \
    --data_name "$BENCHMARK" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split "test" \
    --prompt_type "$PROMPT_TYPE" \
    --num_test_sample -1 \
    --seed 0 \
    --temperature 0.7 \
    --n_sampling 21 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --self_consistent \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.9 \
    --enforce-eager \
    $EXTRA_ARGS
