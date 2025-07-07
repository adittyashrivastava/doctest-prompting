#!/bin/bash
#SBATCH --job-name=medcalc_ptp_attention
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/medcalc_ptp_attention_%j.out
#SBATCH --error=logs/medcalc_ptp_attention_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "========================================="
echo "MedCalc PTP Attention Analysis Job"
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

# Set up environment variables for model cache
export HF_HOME="/data/user_data/ashriva3"
export TRANSFORMERS_CACHE="/data/user_data/ashriva3"

# Set up NLTK data
echo "Setting up NLTK data..."
python setup_nltk.py

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Current directory: $(pwd)"

# Check GPU availability
echo "GPU Information:"
nvidia-smi

# Run the PTP system with attention analysis
echo "Starting MedCalc PTP system with attention analysis..."
echo "Command: python3 job_util.py medcalc_rules --config1 conf.d/medcalc.conf --config2 conf.d/qwen2.5-7b.conf --variant _rel --lo 0 --hi 2 --enable_attention_analysis"

python3 job_util.py medcalc_rules \
    --config1 "conf.d/medcalc.conf" \
    --config2 "conf.d/qwen2.5-7b.conf" \
    --variant "_rel" \
    --lo 0 \
    --hi 2 \
    --enable_attention_analysis

# Capture exit status
EXIT_STATUS=$?

echo "Job completed with exit status: $EXIT_STATUS"
echo "End Time: $(date)"

# Print generated files
echo "Generated log files:"
find ../doctest-prompting-data/logs2/ -name "*medcalc_rules*" -type f | head -10

echo "Generated attention analysis files:"
find ../doctest-prompting-data/logs2/ -name "*.json" -path "*/attention_analysis/*" | head -10

echo "Job completed!"
exit $EXIT_STATUS