set -ex

# Set a local, writable cache directory to avoid download/permission issues
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
CACHE_DIR="${SCRIPT_DIR}/.cache"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"

# allow vLLM to use max_model_len longer than the model's default length
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval_selfconsistent_n21

SPLIT="test"
NUM_TEST_SAMPLE=-1

# function: check if the dataset is completed
check_dataset_completed() {
    local datasets=$1
    local output_dir=$2
    local prompt_type=$3

    IFS=',' read -ra DATASETS <<< "$datasets"
    local remaining_datasets=()
    local completed_datasets=()

    for dataset in "${DATASETS[@]}"; do
        actual_output_dir="${output_dir}/${dataset}"
        result_file="${actual_output_dir}/test_${prompt_type}_-1_seed0_t0.7_s0_e-1_${prompt_type}_metrics.json"

        if [[ -f "$result_file" ]]; then
            # further check if the file is complete (contains the acc field)
            if grep -q '"acc"' "$result_file" 2>/dev/null; then
                completed_datasets+=("$dataset")
                echo "  dataset $dataset is completed, skip (file: $result_file)" >&2
            else
                remaining_datasets+=("$dataset")
                echo "  dataset $dataset result file exists but is incomplete, rerun" >&2
            fi
        else
            remaining_datasets+=("$dataset")
            echo "  dataset $dataset is not completed, need to run (find: $result_file)" >&2
        fi
    done

    # if there are completed datasets, display the statistics
    if [[ ${#completed_datasets[@]} -gt 0 ]]; then
        echo "  completed datasets: ${completed_datasets[*]}" >&2
    fi

    # return the datasets need to run (separated by commas)
    if [[ ${#remaining_datasets[@]} -gt 0 ]]; then
        local IFS=','
        echo "${remaining_datasets[*]}"
    else
        echo ""
    fi
}

# function: run the dataset evaluation
run_evaluation() {
    local datasets=$1
    local extra_args=$2

    if [[ -z "$datasets" ]]; then
        echo "all datasets in this group are completed, skip"
        return 0
    fi

    echo "run datasets: $datasets"

    # get the actual directory of the script
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
    # evaluation/ directory (the parent directory of the script)
    EVAL_DIR=$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)

    # data directory
    DATA_DIR="${EVAL_DIR}/data"

    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="${EVAL_DIR}:$PYTHONPATH" \
    python3 -u "${EVAL_DIR}/math_eval.py" \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --data_name "$datasets" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --split "$SPLIT" \
        --prompt_type "$PROMPT_TYPE" \
        --num_test_sample "$NUM_TEST_SAMPLE" \
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
        $extra_args
}

echo "start Self-Consistent evaluation (smart recovery mode)"
echo "    model path: ${MODEL_NAME_OR_PATH}"
echo "    prompt type: ${PROMPT_TYPE}"
echo "    output directory: ${OUTPUT_DIR}"
echo "    check completed datasets..."
echo ""

# group 1: English open-ended benchmarks (no extra args)
echo "group 1: check English open-ended benchmarks..."
DATA_NAME="gsm8k,math,svamp,asdiv,mawps,tabmwp,minerva_math,gaokao2023en,olympiadbench"
REMAINING_DATASETS=$(check_dataset_completed "$DATA_NAME" "$OUTPUT_DIR" "$PROMPT_TYPE")
run_evaluation "$REMAINING_DATASETS" ""

# group 2: English multiple-choice benchmarks (5-shot)
echo "group 2: check English multiple-choice benchmarks..."
DATA_NAME="mmlu_stem"
REMAINING_DATASETS=$(check_dataset_completed "$DATA_NAME" "$OUTPUT_DIR" "$PROMPT_TYPE")
run_evaluation "$REMAINING_DATASETS" "--num_shots 5"

# group 3: Chinese benchmarks (adaptive few-shot, lower GPU memory)
echo "group 3: check Chinese benchmarks..."
DATA_NAME="gaokao_math_qa,cmath"
REMAINING_DATASETS=$(check_dataset_completed "$DATA_NAME" "$OUTPUT_DIR" "$PROMPT_TYPE")
run_evaluation "$REMAINING_DATASETS" "--adapt_few_shot --gpu_memory_utilization 0.3"

echo ""
echo "all datasets evaluation completed!"
echo "results saved in: ${OUTPUT_DIR}"
