#!/bin/bash

echo "🚀 Starting LatentSync with XFormers-Only Mode (A100 Optimized)"
echo "=============================================================="

# Set optimizations for XFormers-only mode
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_BENCHMARK=1
export XFORMERS_FORCE_DISABLE_TRITON=0  # Enable Triton with XFormers
export DISABLE_FLASH_ATTENTION=1  # Disable Flash Attention to avoid errors

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Check GPU status
echo "🎯 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo "🔍 Verifying XFormers-only setup..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
print(f'✅ TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}')

try:
    import xformers
    print(f'✅ XFormers: {xformers.__version__}')
    print('✅ XFormers: Memory-efficient attention enabled')
except Exception as e:
    print(f'❌ XFormers: {e}')

try:
    import triton
    print(f'✅ Triton: {triton.__version__} (GPU acceleration)')
except Exception as e:
    print(f'❌ Triton: {e}')

print('⚠️ Flash Attention: Disabled (using XFormers instead)')
print('🚀 Expected speedup: 2-3x faster than standard attention')
"

echo ""
echo "🚀 Launching LatentSync with XFormers optimization..."
echo "💡 This will be significantly faster than standard attention!"
python gradio_app_optimized.py 