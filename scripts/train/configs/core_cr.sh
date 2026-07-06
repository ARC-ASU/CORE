# CORE-CR: Concept-Guided Trajectory Replacement (Table 2, Row 4)
# Model: Qwen2-Math-7B | r_bonus=0.4 | Epochs: 3 | GPUs: 2x H200
# batch=64 -> 51 optimization steps over 3 epochs; the reported checkpoint is step 50.

MODEL_PATH="Qwen/Qwen2-Math-7B"
TRAINER_MODULE="verl.trainer.main_ppo"
ADV_ESTIMATOR="grpo"
REWARD_MANAGER="concept_aug"
USE_CONCEPT_FILE=true

NUM_GPUS=2
MAX_RESPONSE_LENGTH=1024
TRAIN_BATCH_SIZE=64
SAVE_FREQ=25
MICRO_BATCH_SIZE=8
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.3
LOG_PROB_MICRO_BATCH=16

USE_KL_LOSS=True
REF_PARAM_OFFLOAD=True
GRAD_OFFLOAD=False
OPTIMIZER_OFFLOAD=False

PROJECT_NAME="core-cr"
