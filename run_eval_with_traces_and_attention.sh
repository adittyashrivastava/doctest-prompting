#!/bin/bash

# Script to run run_eval.py with program traces and attention analysis
# 
# This script demonstrates running run_eval.py with:
# - Program traces enabled (using templates/predict_output_with_traces.txt)
# - Attention analysis enabled for local models
# - Processing examples with both simulation traces and top-k attention facts
#
# Usage: ./run_eval_with_traces_and_attention.sh [TASK_NAME] [MODEL_CONFIG] [LO] [HI] [--enable-attention]
# 
# Examples:
#   ./run_eval_with_traces_and_attention.sh medcalc_rules qwen2.5-7b 0 2                    # Traces only
#   ./run_eval_with_traces_and_attention.sh medcalc_rules qwen2.5-7b 0 2 --enable-attention # Traces + Attention
#   ./run_eval_with_traces_and_attention.sh medcalc_rules qwen2.5-1.5b 0 5 --enable-attention # Smaller model

set -e  # Exit on any error

# Parse arguments
TASK_NAME=${1:-"medcalc_rules"}      # Default task if none provided
MODEL_CONFIG=${2:-"qwen2.5-7b"}     # Default to qwen2.5-7b model if none provided
LO=${3:-0}                          # Start index (default 0)
HI=${4:-2}                          # End index (default 2)
ENABLE_ATTENTION=${5:-""}           # Attention analysis flag

# Determine if attention analysis should be enabled
ATTENTION_FLAG=""
if [ "$ENABLE_ATTENTION" = "--enable-attention" ]; then
    ATTENTION_FLAG="--enable_attention_analysis"
    echo "🔍 Attention analysis: ENABLED"
else
    echo "🔍 Attention analysis: DISABLED (use --enable-attention to enable)"
fi

# Determine configuration file based on model and whether we need traces
if [ "$MODEL_CONFIG" = "qwen2.5-7b" ]; then
    CONFIG_FILE="conf.d/medcalc_traces_7b.conf"
elif [ "$MODEL_CONFIG" = "qwen2.5-1.5b" ]; then
    CONFIG_FILE="conf.d/medcalc_traces_local.conf"
else
    # For other models, construct config filename
    CONFIG_FILE="conf.d/medcalc_traces_${MODEL_CONFIG}.conf"
    # If that doesn't exist, fall back to creating a temporary config
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠️  Warning: Traces config for $MODEL_CONFIG not found, using generic traces config"
        CONFIG_FILE="conf.d/medcalc_traces.conf"
    fi
fi

echo "🚀 Running run_eval.py with program traces and attention analysis"
echo "📋 Task: $TASK_NAME"
echo "🤖 Model config: $MODEL_CONFIG"
echo "📊 Processing examples $LO to $HI"
echo "📝 Program traces: ENABLED"
echo "🎯 Configuration: $CONFIG_FILE"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file $CONFIG_FILE not found!"
    echo "Available traces configs:"
    ls conf.d/medcalc_traces*.conf 2>/dev/null | sed 's/conf.d\///g' | sort || echo "No traces configs found"
    echo ""
    echo "Available model configs:"
    ls conf.d/*.conf | sed 's/conf.d\///g' | sed 's/\.conf//g' | sort
    exit 1
fi

# Check if task exists
TASK_FILE="medcalc/examples/train/${TASK_NAME}.json"
if [ ! -f "$TASK_FILE" ]; then
    echo "❌ Error: Task file $TASK_FILE not found!"
    echo "Available tasks:"
    ls medcalc/examples/train/*.json 2>/dev/null | sed 's/medcalc\/examples\/train\///g' | sed 's/\.json//g' | sort || echo "No tasks found"
    exit 1
fi

echo "✅ Config file: $CONFIG_FILE"
echo "✅ Task file: $TASK_FILE"
echo ""

# Set up environment
echo "🔧 Setting up environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || echo "⚠️  Conda not found, using system Python"

# Try to activate conda environment
if conda info --envs | grep -q "medCalcEnv"; then
    conda activate medCalcEnv
    echo "✅ Activated medCalcEnv"
elif conda info --envs | grep -q "doctest-env"; then
    conda activate doctest-env
    echo "✅ Activated doctest-env"
else
    echo "⚠️  No conda environment found, using current Python environment"
fi

# Optimize for GPU usage
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH="${PYTHONPATH}:."

# Create output directory if it doesn't exist
mkdir -p ../doctest-prompting-data/logs2

echo "🔧 Running run_eval.py with traces..."
echo "Command: python run_eval.py $TASK_NAME --config $CONFIG_FILE --lo $LO --hi $HI $ATTENTION_FLAG"
echo ""

python run_eval.py \
    "$TASK_NAME" \
    --config "$CONFIG_FILE" \
    --lo "$LO" \
    --hi "$HI" \
    $ATTENTION_FLAG

echo ""
echo "✅ Job completed!"
echo "📁 Check the logs in ../doctest-prompting-data/logs2/ for results"
echo "📝 Program traces are included in the output"
if [ "$ENABLE_ATTENTION" = "--enable-attention" ]; then
    echo "🔍 Attention analysis results will be in the attention_analysis subdirectory"
fi
echo ""
echo "📋 Summary:"
echo "   Task: $TASK_NAME"
echo "   Model: $MODEL_CONFIG"
echo "   Examples: $LO to $HI"
echo "   Traces: ✅ ENABLED"
echo "   Attention: $([ "$ENABLE_ATTENTION" = "--enable-attention" ] && echo '✅ ENABLED' || echo '❌ DISABLED')" 