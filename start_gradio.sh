#!/bin/bash

echo "🌐 Starting LatentSync Gradio Interface on A100..."

# Activate conda environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Set environment variables for A100 optimization
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture
export CUDA_LAUNCH_BLOCKING=0

# Enable memory optimization for A100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device()}'); print(f'Device name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No CUDA available')"

# Check if checkpoints exist
if [ ! -f "checkpoints/latentsync_unet.pt" ]; then
    echo "❌ Checkpoint not found! Please run setup_runpod_a100.sh first."
    exit 1
fi

# Create public URL for RunPod
echo "🚀 Starting Gradio with public sharing..."
echo "📱 The interface will be available at the RunPod public URL"
echo "🔗 Look for the 'Running on public URL' message below"

# Start Gradio with optimized settings for A100
python gradio_app_a100.py

echo "👋 Gradio interface stopped." 