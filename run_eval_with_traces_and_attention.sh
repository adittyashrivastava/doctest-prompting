#!/bin/bash
# Enhanced script to run evaluation with traces and attention analysis
# Usage: ./run_eval_with_traces_and_attention.sh <task> <model_size> <lo> <hi> [--enable-attention]

set -e  # Exit on any error

# Parse arguments
TASK=${1:-medcalc_rules}
MODEL_SIZE=${2:-qwen2.5-1.5b}  # Default to smaller model
LO=${3:-0}
HI=${4:-2}
ENABLE_ATTENTION=""

# Check for attention flag
if [[ "$5" == "--enable-attention" ]]; then
    ENABLE_ATTENTION="--enable_attention_analysis"
fi

# Determine config file based on model size
case $MODEL_SIZE in
    "qwen2.5-1.5b")
        CONFIG="conf.d/medcalc_traces_1.5b.conf"
        echo "🚀 Using Qwen 2.5 1.5B model (GPU memory friendly)"
        ;;
    "qwen2.5-7b")
        CONFIG="conf.d/medcalc_traces_7b.conf"
        echo "🚀 Using Qwen 2.5 7B model (requires more GPU memory)"
        ;;
    "qwen2.5-1.5b-stable")
        CONFIG="conf.d/medcalc_traces_1.5b_stable.conf"
        echo "🚀 Using Qwen 2.5 1.5B model (stable configuration)"
        ;;
    *)
        echo "❌ Unknown model size: $MODEL_SIZE"
        echo "Available options: qwen2.5-1.5b, qwen2.5-7b, qwen2.5-1.5b-stable"
        exit 1
        ;;
esac

# Check if config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "❌ Configuration file not found: $CONFIG"
    exit 1
fi

echo "📋 Configuration: $CONFIG"
echo "📊 Range: examples $LO to $HI"
echo "🧠 Attention analysis: $([ -n "$ENABLE_ATTENTION" ] && echo "ENABLED" || echo "DISABLED")"
echo ""

# Set up environment
echo "🔧 Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if attention_viz is available when attention analysis is requested
if [[ -n "$ENABLE_ATTENTION" ]]; then
    echo "🔍 Checking attention_viz availability..."
    python -c "
try:
    from attention_viz import AttentionExtractor, AttrievelRetriever, AttrievelConfig
    print('✅ attention_viz module is available')
except ImportError as e:
    print(f'❌ attention_viz not available: {e}')
    print('Please install attention_viz or run without --enable-attention flag')
    exit(1)
" || exit 1
fi

# Run the evaluation
echo "🚀 Running evaluation..."
echo ""

CMD="python run_eval.py $TASK --config $CONFIG --lo $LO --hi $HI $ENABLE_ATTENTION"
echo "Executing: $CMD"
echo ""

eval $CMD

EXITCODE=$?

if [[ $EXITCODE -eq 0 ]]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📁 Results saved to: logs/${TASK}/"
    
    if [[ -n "$ENABLE_ATTENTION" ]]; then
        echo "🧠 Attention analysis results: logs/${TASK}/*/attention_analysis/"
        echo ""
        echo "📊 Example attention files:"
        find logs/${TASK}/ -name "top_facts.json" -type f | head -3 | while read file; do
            echo "   $file"
        done
    fi
    
    echo ""
    echo "🎯 Usage examples:"
    echo "   # Run with 1.5B model (recommended for GPU memory)"
    echo "   $0 $TASK qwen2.5-1.5b $LO $HI --enable-attention"
    echo ""
    echo "   # Run with 7B model (needs more GPU memory)"
    echo "   $0 $TASK qwen2.5-7b $LO $HI --enable-attention"
    echo ""
    echo "   # Run without attention analysis (faster)"
    echo "   $0 $TASK qwen2.5-1.5b $LO $HI"
    
else
    echo ""
    echo "❌ Evaluation failed with exit code: $EXITCODE"
    exit $EXITCODE
fi 