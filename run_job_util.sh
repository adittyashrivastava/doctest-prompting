#!/bin/bash

# Simple bash script to run job_util.py with PTP and attention enabled for 2 examples
# 
# This script demonstrates running job_util.py with:
# - PTP (Prompt Template Programming) enabled (default behavior)
# - Attention analysis enabled 
# - Processing 2 examples (indices 0-1)
# - Local models support
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
MODEL_CONFIG=${2:-"qwen2.5-7b"}         # Default to qwen2.5-7b model if none provided

# Available tasks (from medcalc/examples/test/):
# medcalc_formulas, medcalc_formulas_rel, medcalc_rules, medcalc_rules_rel

# Available model configs (from conf.d/):
# local models: qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-7b, qwen2.5-7b-together, coder, deepseekv3, deepseekR1
# API models: sonnet3, sonnet3-5, haiku, haiku3-5, gpt4o, flash

echo "🚀 Running job_util.py with PTP and attention analysis"
echo "📋 Task: $TASK_NAME"
echo "🤖 Model config: $MODEL_CONFIG"
echo "📊 Processing 2 examples (indices 0-1)"
echo "🔍 Attention analysis: ENABLED"
echo "🎯 PTP (Prompt Template Programming): ENABLED (default)"
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

# Optimize for GPU usage
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="${PYTHONPATH}:."

# Create output directory if it doesn't exist
mkdir -p ../doctest-prompting-data/logs2

echo "🔧 Running job_util.py..."
python job_util.py \
    --config conf.d/medcalc.conf \
    --config2 "$CONFIG_FILE" \
    --lo 0 \
    --hi 2 \
    --enable_attention_analysis \
    --task_dir tasks \
    --variant _rel \
    "$TASK_NAME"

echo ""
echo "✅ Job completed!"
echo "📁 Check the logs in ../doctest-prompting-data/logs2/ for results"
echo "🔍 Attention analysis results will be in the attention_analysis subdirectory" 