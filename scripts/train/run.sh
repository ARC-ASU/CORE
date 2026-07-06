#!/bin/bash
# ============================================================================
# CORE: Unified RL Training Script
# ============================================================================
# Usage:
#   bash scripts/train/run.sh <config>
#
# Configs (in scripts/train/configs/):
#   core_cr           CORE-CR on Qwen2-Math-7B       (main result, Table 2)
#   core_kl           CORE-KL on Qwen2-Math-7B       (Table 2)
#   core_base         CORE-Base on Qwen2-Math-7B     (Table 2)
#   core_cr_deepseek  CORE-CR on DeepSeek-R1-DQ-1.5B (Table 4)
#   core_cr_qwen25    CORE-CR on Qwen2.5-Math-1.5B   (Table 4)
#   core_cr_llama     CORE-CR on Llama-3-8B-Instruct (Table 4)
#   core_cr_ppo       CORE-CR with PPO backbone       (Table 8)
#
# Examples:
#   bash scripts/train/run.sh core_cr
#   bash scripts/train/run.sh core_kl
#   bash scripts/train/run.sh core_cr_deepseek
#
# Override parameters:
#   EPOCHS=5 bash scripts/train/run.sh core_cr
#   NUM_GPUS=4 bash scripts/train/run.sh core_cr
# ============================================================================

set -e
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)

# ---- Load config ----
CONFIG_NAME=${1:-"core_cr"}
CONFIG_FILE="${PROJECT_ROOT}/scripts/train/configs/${CONFIG_NAME}.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config '${CONFIG_NAME}' not found."
    echo "Available configs:"
    ls scripts/train/configs/*.sh 2>/dev/null | xargs -I{} basename {} .sh | sed 's/^/  /'
    exit 1
fi

# ---- Defaults (can be overridden by config or environment) ----
TRAINER_MODULE="verl.trainer.main_ppo"
ADV_ESTIMATOR="grpo"
REWARD_MANAGER="naive"
USE_CONCEPT_FILE=false
MODEL_PATH="Qwen/Qwen2-Math-7B"
EPOCHS=${EPOCHS:-3}
NUM_GPUS=${NUM_GPUS:-2}
OUTPUT_DIR=""

MAX_RESPONSE_LENGTH=1024
TRAIN_BATCH_SIZE=128
MICRO_BATCH_SIZE=8
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.3
LOG_PROB_MICRO_BATCH=16

USE_KL_LOSS=True
REF_PARAM_OFFLOAD=True
GRAD_OFFLOAD=False
OPTIMIZER_OFFLOAD=False

# Extended vLLM settings (for long-sequence models)
USE_EXTENDED_VLLM=false
SWAP_SPACE=""
MAX_NUM_BATCHED_TOKENS=""
MAX_NUM_SEQS=""

# KL enhancement (CORE-KL only)
USE_KL_ENHANCEMENT=false
KL_BASE_LAMBDA=""
KL_EFFECTIVE_MULTIPLIER=""
KL_INEFFECTIVE_MULTIPLIER=""

# PPO settings (PPO variant only)
USE_PPO=false
PPO_CLIP_RATIO=""
PPO_ENTROPY_COEFF=""
PPO_GAMMA=""
PPO_LAM=""
CRITIC_LR=""

SAVE_INFERENCE_ONLY=""
EXTRA_ENV=""
PROJECT_NAME=""

# ---- Source config (overrides defaults) ----
source "$CONFIG_FILE"

# ---- Derived values ----
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/${CONFIG_NAME}"}
PROJECT_NAME=${PROJECT_NAME:-"${CONFIG_NAME}"}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ---- Environment ----
export VLLM_ATTENTION_BACKEND=XFORMERS
# W&B defaults to offline so training never blocks on `wandb login`;
# set WANDB_MODE=online after logging in to sync runs to the cloud.
export WANDB_MODE=${WANDB_MODE:-offline}
# Leave RAY_TMPDIR at Ray's short default (/tmp/ray): AF_UNIX socket paths are
# capped at 107 bytes, so pointing it into a deep project path breaks Ray startup.
# Export RAY_TMPDIR yourself (short path) to relocate it.
mkdir -p $OUTPUT_DIR

if [ -n "$EXTRA_ENV" ]; then
    export $EXTRA_ENV
fi

echo "============================================"
echo "CORE Training: ${CONFIG_NAME}"
echo "  Model:          ${MODEL_PATH}"
echo "  Method:         ${ADV_ESTIMATOR} + ${REWARD_MANAGER}"
echo "  GPUs:           ${NUM_GPUS}"
echo "  Epochs:         ${EPOCHS}"
echo "  Output:         ${OUTPUT_DIR}"
echo "============================================"

# ---- Build command ----
CMD="python -m ${TRAINER_MODULE}"
CMD+=" algorithm.adv_estimator=${ADV_ESTIMATOR}"
CMD+=" data.train_files=${PROJECT_ROOT}/data/train.parquet"
CMD+=" data.val_files=${PROJECT_ROOT}/data/val.parquet"
CMD+=" data.train_batch_size=${TRAIN_BATCH_SIZE}"
CMD+=" data.val_batch_size=${TRAIN_BATCH_SIZE}"
CMD+=" data.max_prompt_length=1024"
CMD+=" data.max_response_length=${MAX_RESPONSE_LENGTH}"
CMD+=" reward_model.reward_manager=${REWARD_MANAGER}"

if [ "$USE_CONCEPT_FILE" = true ]; then
    CMD+=" +reward_model.concept_file=${PROJECT_ROOT}/data/quizzes/concept_quizzes.jsonl"
fi

CMD+=" actor_rollout_ref.model.path=${MODEL_PATH}"
CMD+=" actor_rollout_ref.actor.optim.lr=1e-6"
CMD+=" actor_rollout_ref.model.use_remove_padding=True"
CMD+=" actor_rollout_ref.actor.ppo_mini_batch_size=32"
CMD+=" actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
CMD+=" actor_rollout_ref.actor.use_dynamic_bsz=True"
CMD+=" actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288"
CMD+=" actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"

if [ "$USE_KL_LOSS" = "True" ]; then
    CMD+=" actor_rollout_ref.actor.kl_loss_coef=0.001"
    CMD+=" actor_rollout_ref.actor.kl_loss_type=low_var_kl"
fi

if [ "$USE_PPO" = true ]; then
    CMD+=" actor_rollout_ref.actor.clip_ratio=${PPO_CLIP_RATIO}"
    CMD+=" actor_rollout_ref.actor.entropy_coeff=${PPO_ENTROPY_COEFF}"
    CMD+=" algorithm.gamma=${PPO_GAMMA}"
    CMD+=" algorithm.lam=${PPO_LAM}"
fi

CMD+=" actor_rollout_ref.model.enable_gradient_checkpointing=True"
CMD+=" actor_rollout_ref.actor.fsdp_config.param_offload=False"

if [ -n "$GRAD_OFFLOAD" ] && [ "$GRAD_OFFLOAD" != "" ]; then
    CMD+=" +actor_rollout_ref.actor.fsdp_config.grad_offload=${GRAD_OFFLOAD}"
fi

CMD+=" actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}"
CMD+=" actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE}"
CMD+=" actor_rollout_ref.rollout.name=vllm"
CMD+=" actor_rollout_ref.rollout.temperature=0.7"
CMD+=" actor_rollout_ref.rollout.top_p=0.9"
CMD+=" actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTIL}"

if [ "$USE_EXTENDED_VLLM" = true ]; then
    CMD+=" +actor_rollout_ref.rollout.swap_space=${SWAP_SPACE}"
    CMD+=" actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    CMD+=" actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
fi

CMD+=" actor_rollout_ref.rollout.n=4"
CMD+=" +actor_rollout_ref.rollout.n_val=1"
CMD+=" actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH}"
CMD+=" actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
CMD+=" actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH}"
CMD+=" algorithm.kl_ctrl.kl_coef=0.001"

# PPO critic settings
if [ "$USE_PPO" = true ]; then
    CMD+=" algorithm.kl_ctrl.type=fixed"
    CMD+=" critic.model.path=${MODEL_PATH}"
    CMD+=" critic.optim.lr=${CRITIC_LR}"
    CMD+=" critic.model.use_remove_padding=True"
    CMD+=" critic.model.enable_gradient_checkpointing=True"
    CMD+=" critic.ppo_micro_batch_size_per_gpu=8"
    CMD+=" critic.use_dynamic_bsz=True"
    CMD+=" critic.ppo_max_token_len_per_gpu=24576"
    CMD+=" critic.model.fsdp_config.param_offload=False"
    CMD+=" critic.model.fsdp_config.optimizer_offload=True"
    CMD+=" critic.grad_clip=1.0"
    CMD+=" critic.cliprange_value=0.5"
fi

CMD+=" trainer.critic_warmup=0"
CMD+=" trainer.logger=[console,wandb]"
CMD+=" trainer.project_name=${PROJECT_NAME}"
CMD+=" trainer.experiment_name=${PROJECT_NAME}-${TIMESTAMP}"
CMD+=" trainer.checkpoints_dir=${OUTPUT_DIR}"
CMD+=" trainer.n_gpus_per_node=${NUM_GPUS}"
CMD+=" trainer.nnodes=1"
CMD+=" trainer.save_freq=${SAVE_FREQ:-8}"
CMD+=" trainer.test_freq=10"
CMD+=" trainer.default_hdfs_dir=null"

if [ -n "$SAVE_INFERENCE_ONLY" ]; then
    CMD+=" +trainer.save_inference_only=${SAVE_INFERENCE_ONLY}"
fi

CMD+=" trainer.total_epochs=${EPOCHS}"

# KL enhancement parameters (CORE-KL only)
if [ "$USE_KL_ENHANCEMENT" = true ]; then
    CMD+=" +kl_enhancement.base_lambda=${KL_BASE_LAMBDA}"
    CMD+=" +kl_enhancement.effective_multiplier=${KL_EFFECTIVE_MULTIPLIER}"
    CMD+=" +kl_enhancement.ineffective_multiplier=${KL_INEFFECTIVE_MULTIPLIER}"
fi

# ---- Run ----
echo "Command: ${CMD}"
echo ""
eval $CMD

echo "Training complete. Checkpoints saved to ${OUTPUT_DIR}"
