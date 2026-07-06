#!/bin/bash
# SFT training using embedded LLaMA-Factory
# Usage: bash run_sft.sh [config_yaml] [num_gpus]
#
# Examples:
#   bash run_sft.sh configs/qwen2_math_7b_sft.yaml
#   bash run_sft.sh configs/deepseek_1_5b_sft.yaml

set -e
cd "$(dirname "$0")"

CONFIG=${1:-"configs/qwen2_math_7b_sft.yaml"}
NUM_GPUS=${2:-2}

echo "Starting SFT training..."
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"

# Check LLaMA-Factory installation
if ! command -v llamafactory-cli &> /dev/null; then
    echo "Error: llamafactory-cli not found. Please install (core_sft env, see README):"
    echo "  pip install -r requirements/requirements_sft.txt && cd fine-tune/LLaMA-Factory && pip install -e . --no-deps"
    exit 1
fi

# respect a caller-provided CUDA_VISIBLE_DEVICES (shared machines); default to 0..N-1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
fi
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"

llamafactory-cli train ${CONFIG}

echo "SFT training complete!"
