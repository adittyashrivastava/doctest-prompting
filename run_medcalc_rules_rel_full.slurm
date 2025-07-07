#!/bin/bash
#SBATCH --job-name=medcalc_rules_rel_full
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/ashriva3/logs/medcalc_rules_rel_full_%j.out
#SBATCH --error=/home/ashriva3/logs/medcalc_rules_rel_full_%j.err

# Create logs directory if it doesn't exist
mkdir -p /home/ashriva3/logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Change to the project directory
cd /home/ashriva3/codebase/cohen_repo/doctest-prompting

# Activate conda environment
source /home/ashriva3/miniconda3/etc/profile.d/conda.sh
conda activate doctest-env

# Export API key for Together AI
export TOGETHER_API_KEY="f27acbd8d438bafc0298c78677f5b33a99fecba8d37859269dde9794bbdd5043"

# Print environment info
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Current directory: $(pwd)"

# Run the full medcalc_rules evaluation with _rel variant
echo "Starting medcalc_rules evaluation with _rel variant..."
echo "Command: python3 run_eval_range.py medcalc_rules --variant _rel --config conf.d/medcalc.conf --config2 conf.d/qwen2.5-7b-together.conf"

python3 run_eval_range.py medcalc_rules \
    --variant _rel \
    --config conf.d/medcalc.conf \
    --config2 conf.d/qwen2.5-7b-together.conf

# Capture exit status
EXIT_STATUS=$?

echo "Job completed with exit status: $EXIT_STATUS"
echo "End Time: $(date)"

# Print some final statistics
if [ -d "../doctest-prompting-data/logs2/together-Qwen/Qwen2.5-7B-Instruct-Turbo/medcalc_rules" ]; then
    echo "Generated log files:"
    ls -la ../doctest-prompting-data/logs2/together-Qwen/Qwen2.5-7B-Instruct-Turbo/medcalc_rules/*_rel*.log
fi

exit $EXIT_STATUS