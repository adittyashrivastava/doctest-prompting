#!/bin/bash

# Simple bash script to run evaluation with PTP enabled for 2 examples
# 
# This script automatically chooses the right evaluation script:
# - job_util.py for local models (with attention analysis)
# - run_eval.py for API models (no attention analysis)
# - PTP (Prompt Template Programming) enabled (default behavior)
# - Processing 2 examples (indices 0-1)
#
# Usage: ./run_job_util.sh [TASK_NAME] [MODEL_CONFIG]
# 
# Examples:
#   ./run_job_util.sh medcalc_rules qwen2.5-1.5b     # Use small local Qwen model  
#   ./run_job_util.sh medcalc_rules qwen2.5-7b       # Use larger local Qwen model
#   ./run_job_util.sh medcalc_rules sonnet3          # Use Claude Sonnet via API

set -e  # Exit on any error

# Default values
TASK_NAME=${1:-"medcalc_rules"}      # Default task if none provided
MODEL_CONFIG=${2:-"qwen2.5-1.5b"}       # Default to qwen2.5-1.5b model if none provided

# Available tasks (from medcalc/examples/test/):
# medcalc_formulas, medcalc_formulas_rel, medcalc_rules, medcalc_rules_rel

# Available model configs (from conf.d/):
# local models: qwen2.5-1.5b, qwen2.5-7b, qwen2.5-7b-together, coder, deepseekv3, deepseekR1
# API models: sonnet3, sonnet3-5, haiku, haiku3-5, gpt4o, flash

echo "🚀 Running evaluation with PTP enabled"
echo "📋 Task: $TASK_NAME"
echo "🤖 Model config: $MODEL_CONFIG"
echo "📊 Processing 2 examples (indices 0-1)"
echo "🎯 PTP (Prompt Template Programming): ENABLED (default)"

# Check model type and report capabilities
if grep -q "service = local" "$CONFIG_FILE"; then
    echo "🔍 Attention analysis: ENABLED (local model)"
else
    echo "🔍 Attention analysis: NOT AVAILABLE (API model)"
fi
echo ""

# Check if config file exists
CONFIG_FILE="conf.d/${MODEL_CONFIG}.conf"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found!"
    echo "Available configs:"
    ls conf.d/*.conf | sed 's/conf.d\///g' | sed 's/\.conf//g' | sort
    exit 1
fi

# Check if task exists
TASK_FILE="medcalc/examples/test/${TASK_NAME}.json"
if [ ! -f "$TASK_FILE" ]; then
    echo "❌ Error: Task file $TASK_FILE not found!"
    echo "Available tasks:"
    ls medcalc/examples/test/*.json | sed 's/medcalc\/examples\/test\///g' | sed 's/\.json//g' | sort
    exit 1
fi

echo "✅ Config file: $CONFIG_FILE"
echo "✅ Task file: $TASK_FILE"
echo ""

# Set up environment - try to fix the multiprocessing issue
echo "🔧 Setting up environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

# Try to fix multiprocessing issue by setting environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH="${PYTHONPATH}:."

# Create output directory if it doesn't exist
mkdir -p ../doctest-prompting-data/logs2

# Check if this is a local model or API model
if grep -q "service = local" "$CONFIG_FILE"; then
    echo "🔧 Running job_util.py with local model..."
    python job_util.py \
        --config conf.d/medcalc.conf \
        --config2 "$CONFIG_FILE" \
        --lo 0 \
        --hi 2 \
        --enable_attention_analysis \
        --task_dir tasks \
        --variant _rel \
        "$TASK_NAME"
else
    echo "🔧 Running run_eval.py with API model..."
    echo "⚠️  Note: Attention analysis not available with API models"
    python run_eval.py \
        --config conf.d/medcalc.conf \
        --config2 "$CONFIG_FILE" \
        --lo 0 \
        --hi 2 \
        --variant _rel \
        "$TASK_NAME"
fi

echo ""
echo "✅ Job completed!"
echo "📁 Check the logs in ../doctest-prompting-data/logs2/ for results"
echo "🔍 Attention analysis results will be in the attention_analysis subdirectory" 