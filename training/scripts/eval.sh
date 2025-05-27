#!/usr/bin/env bash
# =========================================================================
# Model Evaluation Script
# 
# Usage: 
#   ./eval.sh MODEL SUITE TASKS [OUTPUT_DIR] [MAX_NEW_TOKENS] [FEW_SHOTS] [TENSOR_PARALLEL] [TEMPERATURE] [IN_TASK_LEARNING_EXAMPLES] [IN_TASK_LEARNING_MODE] [TEST_TIME_REPEATS] [VERIFIER_BEST_OF_N] [Majority_Vote]
#
# Example:
#   ./eval.sh Qwen/Qwen2.5-7B custom gsm8k,math_500,gpqa_diamond ./evals/qwen_results 28762 0 1
# =========================================================================

set -e  
cd /home/jixuanl/verl
source ~/.bashrc
source r1/bin/activate


export HF_HOME=/mnt/localssd/jixuanl/hf_cache/hub
export HF_DATASETS_CACHE=/mnt/localssd/jixuanl/hf_cache/hf_datasets
export HF_HUB_CACHE=/mnt/localssd/jixuanl/hub

export VLLM_WORKER_MULTIPROC_METHOD=spawn

show_usage() {
    echo "Usage: $0 MODEL SUITE TASKS [OUTPUT_DIR] [MAX_NEW_TOKENS] [FEW_SHOTS] [TENSOR_PARALLEL]"
    echo ""
    echo "Required arguments:"
    echo "  MODEL              Model name or path"
    echo "  SUITE              Evaluation suite name (e.g., 'custom')"
    echo "  TASKS              Comma-separated list of tasks (e.g., 'gsm8k,math_500')"
    echo ""
    echo "Optional arguments:"
    echo "  OUTPUT_DIR                      Directory for evaluation results (default: ./evals)"
    echo "  MAX_NEW_TOKENS                  Maximum number of new tokens (default: 16384)"
    echo "  FEW_SHOTS                       Number of few-shot examples (default: 0)"
    echo "  TENSOR_PARALLEL                 Tensor parallel size (default: 1)"
    echo "  TEMPERATURE                     Sampling temperature (default: 0.7)"
    echo "  IN_TASK_LEARNING_EXAMPLES       Number of in-task learning examples (default: 0)"
    echo "  IN_TASK_LEARNING_MODE           In-task learning mode (default: 0)"
    echo "  TEST_TIME_REPEATS               Number of test time repeats (default: 0)"
    echo "  AUDIT_SELF_CONSISTENCY            Use audit self-consistency (default: 0)"
    echo "  BASELINE_SELF_CONSISTENCY         Use baseline self-consistency (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 Qwen/Qwen2.5-7B custom gsm8k,math_500,gpqa_diamond ./evals/base_all"
}

if [ $# -lt 3 ]; then
    echo "Error: Missing required arguments."
    show_usage
    exit 1
fi

MODEL="$1"
SUITE="$2"
TASKS="$3"
DEFAULT_OUTPUT_DIR="./evals"
OUTPUT_DIR="${4:-$DEFAULT_OUTPUT_DIR}"
MAX_NEW_TOKENS="${5:-32768}"
FEW_SHOTS="${6:-0}"
TENSOR_PARALLEL="${7:-1}"
TEMPERATURE="${8:-0}"
IN_TASK_LEARNING_EXAMPLES="${9:-0}"
IN_TASK_LEARNING_MODE="${10:-0}"
TEST_TIME_REPEATS="${11:-0}"
AUDIT_SELF_CONSISTENCY="${12:-0}"
BASELINE_SELF_CONSISTENCY="${13:-0}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$MODEL" ]]; then
    echo "Error: MODEL cannot be empty."
    exit 1
fi

if [[ -z "$SUITE" ]]; then
    echo "Error: SUITE cannot be empty."
    exit 1
fi

if [[ -z "$TASKS" ]]; then
    echo "Error: TASKS cannot be empty."
    exit 1
fi

IFS=',' read -r -a TASK_LIST <<< "$TASKS"

echo "==================== Evaluation Configuration ====================="
echo "MODEL:                            $MODEL"
echo "SUITE:                            $SUITE"
echo "TASKS:                            ${TASKS}"
echo "OUTPUT_DIR:                       $OUTPUT_DIR"
echo "MAX_NEW_TOKENS:                   $MAX_NEW_TOKENS"
echo "FEW_SHOTS:                        $FEW_SHOTS"
echo "TENSOR_PARALLEL:                  $TENSOR_PARALLEL"
echo "TEMPERATURE:                      $TEMPERATURE"
echo "IN_TASK_LEARNING_EXAMPLES:        $IN_TASK_LEARNING_EXAMPLES"
echo "IN_TASK_LEARNING_MODE:            $IN_TASK_LEARNING_MODE"
echo "TEST_TIME_REPEATS:                $TEST_TIME_REPEATS"
echo "AUDIT SELF CONSISTENCY:           $AUDIT_SELF_CONSISTENCY"
echo "BASELINE SELF CONSISTENCY:        $BASELINE_SELF_CONSISTENCY"
echo "Time started:         $(date)"
echo "=================================================================="

# Create a text file explaining the filename format
# in case i forget in the future :)
create_format_guide() {
    local output_dir="$1"
    local all_tasks="$2"
    
    local base_dir=$(echo "$output_dir" | grep -o "^\./[^/]*")
    base_dir=${base_dir:-"./evals"}
    
    local guide_file="${base_dir}/filename_format_guide.txt"
    
    echo "Creating filename format guide at: ${guide_file}"
    
    cat > "$guide_file" << EOF
Filename Format Guide
=====================

The results are saved in JSON files with the following naming pattern:
{suite}_{tasks}_{few_shot}_{temperature}_{in_task_learning_examples}_{test_time_scaling_repeats}

Where:
- suite: The evaluation suite (e.g., 'custom')
- tasks: The list of tasks evaluated
- few_shot: Number of few-shot examples used (${FEW_SHOTS})
- temperature: Sampling temperature (${TEMPERATURE})
- in_task_learning_examples: Number of in-task learning examples (${IN_TASK_LEARNING_EXAMPLES})
- in_task_learning_mode: In-task learning mode (${IN_TASK_LEARNING_MODE})
- test_time_scaling_repeats: Number of test time repeats (${TEST_TIME_REPEATS})

Example for this run:
${SUITE}_${all_tasks}_${FEW_SHOTS}_${TEMPERATURE}_${IN_TASK_LEARNING_EXAMPLES}_${IN_TASK_LEARNING_MODE}_${TEST_TIME_REPEATS}

Created: $(date)
EOF
}

{
    echo "Starting evaluation at $(date)"
    
    create_format_guide "$OUTPUT_DIR" "$TASKS"
    
    python ./scripts/evaluate.py \
        --model "$MODEL" \
        --task "${TASK_LIST[@]}" \
        --suite "$SUITE" \
        --use_chat_template \
        --tensor_parallel "$TENSOR_PARALLEL" \
        --few_shot "$FEW_SHOTS" \
        --output_dir "$OUTPUT_DIR" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --in_task_learning_examples "$IN_TASK_LEARNING_EXAMPLES" \
        --in_task_learning_mode "$IN_TASK_LEARNING_MODE" \
        --test_time_scaling_repeats "$TEST_TIME_REPEATS" \
        --audit_self_consistency "$AUDIT_SELF_CONSISTENCY" \
        --baseline_self_consistency "$BASELINE_SELF_CONSISTENCY" \
 
    echo "Evaluation completed at $(date)"
}

