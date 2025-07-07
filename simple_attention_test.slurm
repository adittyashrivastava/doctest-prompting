#!/bin/bash
#SBATCH --job-name=attention_test_simple
#SBATCH --partition=debug
#SBATCH --qos=debug_qos
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output=logs/attention_simple_%j.out
#SBATCH --error=logs/attention_simple_%j.err

echo "🚀 Starting Simple Attention Analysis Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate doctest-env

# Create logs directory
mkdir -p logs

# Check environment
echo "✅ Environment check:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

try:
    from attention_viz import AttentionExtractor, AttrievelRetriever
    print('✅ attention_viz imported successfully')
except ImportError as e:
    print(f'❌ attention_viz import failed: {e}')
"

# Set HuggingFace cache to use our local directory
export HF_HOME="/data/user_data/ashriva3"
export TRANSFORMERS_CACHE="/data/user_data/ashriva3"

# Run simple test with boolean_expressions task (2 examples)
echo "🔍 Running attention analysis test..."
python job_util.py \
    boolean_expressions \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --service "huggingface" \
    --lo 0 \
    --hi 2 \
    --log_dir "../doctest-prompting-data" \
    --example_dir "bbh/examples/train" \
    --test_example_dir "bbh/examples/test/" \
    --partial_program_dir "bbh/mocks/partialprograms" \
    --cot_prompt_dir "bbh/modified-cot-prompts/" \
    --variant "" \
    --enable_attention_analysis

echo "✅ Test completed at $(date)"

# Show results
echo "📊 Results:"
if [ -d "doctest-prompting-data" ]; then
    find doctest-prompting-data -name "attention_analysis" -type d
    find doctest-prompting-data -name "top_facts.json"
else
    echo "No results directory found"
fi
