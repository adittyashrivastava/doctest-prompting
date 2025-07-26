#!/bin/bash
#SBATCH --job-name=medcalc_ptp_attention_vllm
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/medcalc_ptp_attention_vllm_%j.out
#SBATCH --error=logs/medcalc_ptp_attention_vllm_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "========================================="
echo "MedCalc PTP Attention Analysis Job (vLLM)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================="

# Change to the project directory
cd /home/ashriva3/codebase/cohen_repo/doctest-prompting

# Set up conda environment
export PATH="/home/ashriva3/miniconda3/bin:$PATH"
source /home/ashriva3/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
echo "Activating doctest-env..."
conda activate doctest-env

# Install vLLM if not already installed
echo "Checking vLLM installation..."
python -c "import vllm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing vLLM..."
    pip install vllm>=0.8.3
    pip install torch>=2.0.0
    pip install transformers>=4.0.0
    pip install accelerate>=0.20.0
    pip install xformers>=0.0.22
    pip install flash-attn>=2.0.0
    pip install ninja>=1.10.0
else
    echo "vLLM already installed"
fi

# Set up environment variables for model cache
export HF_HOME="/data/user_data/ashriva3"
export TRANSFORMERS_CACHE="/data/user_data/ashriva3"

# Set vLLM environment variables for optimal performance
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0

# Set up NLTK data
echo "Setting up NLTK data..."
python setup_nltk.py

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Current directory: $(pwd)"

# Check GPU availability and vLLM installation
echo "GPU Information:"
nvidia-smi

echo "vLLM Installation Check:"
python -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Run the PTP system with vLLM and attention analysis
echo "Starting MedCalc PTP system with vLLM and attention analysis..."
echo "Command: python3 job_util_vllm.py medcalc_rules --config1 conf.d/medcalc.conf --config2 conf.d/qwen2.5-7b.conf --variant _rel --lo 0 --hi 2 --enable_attention_analysis --tensor_parallel_size 1 --gpu_memory_utilization 0.8 --temperature 0.7 --top_p 0.6"

python3 job_util_vllm.py medcalc_rules \
    --config1 "conf.d/medcalc.conf" \
    --config2 "conf.d/qwen2.5-7b.conf" \
    --variant "_rel" \
    --lo 0 \
    --hi 2 \
    --enable_attention_analysis \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    --temperature 0.7 \
    --top_p 0.6 \
    --max_model_len 4096

# Capture exit status
EXIT_STATUS=$?

echo "Job completed with exit status: $EXIT_STATUS"
echo "End Time: $(date)"

# Print generated files
echo "Generated log files:"
find ../doctest-prompting-data/logs2/ -name "*medcalc_rules*" -type f | head -10

echo "Generated attention analysis files:"
find ../doctest-prompting-data/logs2/ -name "*.json" -path "*/attention_analysis/*" | head -10

# Print vLLM performance summary
echo "vLLM Performance Summary:"
echo "========================================="
echo "Inference backend: vLLM"
echo "GPU Memory Utilization: 80%"
echo "Tensor Parallel Size: 1"
echo "Temperature: 0.7"
echo "Top-p: 0.6"
echo "Max Model Length: 4096"
echo "========================================="

echo "Job completed!"
exit $EXIT_STATUS