#!/bin/bash

# Setup script for separate vLLM and verl environments
# This avoids the numpy version conflict

# Initialize conda
source /home/hrangara/miniconda3/etc/profile.d/conda.sh

echo "🚀 Setting up separate environments for vLLM and verl..."

# Create vLLM environment (with numpy 2.x)
echo "📦 Creating vLLM environment..."
conda create -n vllm-env python=3.10 -y

echo "📦 Installing vLLM dependencies..."
conda activate vllm-env
pip install vllm==0.8.3
pip install torch transformers accelerate
pip install matplotlib plotly pandas seaborn scikit-learn
pip install configargparse nltk

# Install attention_viz in vLLM environment
echo "📦 Installing attention_viz in vLLM environment..."
cd /home/hrangara/MedCalc/MedCalc-Bench/attention_viz
pip install -e .

echo "✅ vLLM environment ready!"

# Create verl environment (with numpy 1.x)
echo "📦 Creating verl environment..."
conda create -n verl-env python=3.10 -y

echo "📦 Installing verl dependencies..."
conda activate verl-env
echo "📦 Activated verl environment"
pip install "numpy<2.0.0"
pip install verl
pip install torch transformers accelerate
pip install matplotlib plotly pandas seaborn scikit-learn
pip install configargparse nltk

# Install attention_viz in verl environment
echo "📦 Installing attention_viz in verl environment..."
cd /home/hrangara/MedCalc/MedCalc-Bench/attention_viz
pip install -e .

echo "✅ verl environment ready!"

# Test both environments
echo "🧪 Testing environments..."

echo "Testing vLLM environment:"
conda activate vllm-env
python -c "
import vllm
print('✅ vLLM working')
import numpy
print(f'✅ numpy version: {numpy.__version__}')
from attention_viz import AttentionExtractor
print('✅ attention_viz working in vLLM env')
"

echo "Testing verl environment:"
conda activate ex
python -c "
import verl
print('✅ verl working')
import numpy
print(f'✅ numpy version: {numpy.__version__}')
from attention_viz import AttentionExtractor
print('✅ attention_viz working in verl env')
"

echo "🎉 Environment setup complete!"
echo ""
echo "Usage:"
echo "  # Run the training pipeline:"
echo "  python verl_training_pipeline_v2.py"
echo ""
echo "  # Or with custom settings:"
echo "  python verl_training_pipeline_v2.py --max-samples 20 --episodes 10"