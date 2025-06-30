#!/bin/bash

echo "🛡️ Starting LatentSync A100 Safe Mode..."
echo "✅ Basic optimizations only - guaranteed to work!"

# Set basic CUDA optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo "🚀 Launching Safe Mode Gradio Interface..."
python gradio_app_safe.py 