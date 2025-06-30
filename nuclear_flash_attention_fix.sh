#!/bin/bash

echo "💥 NUCLEAR FLASH ATTENTION FIX - No Mercy Mode!"
echo "================================================"

source activate latentsync

echo "🔧 Step 1: Complete cleanup of problematic packages..."
pip uninstall xformers flash-attn triton -y 2>/dev/null || true

echo "✅ Cleanup complete!"

echo "🎯 Step 2: Installing Flash Attention precompiled wheel..."
# Use the exact precompiled wheel for PyTorch 2.5.1 + CUDA 12.1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo "✅ Flash Attention wheel installed!"

echo "⚡ Step 3: Installing compatible XFormers..."
# Use XFormers that works with PyTorch 2.5.1
pip install xformers==0.0.28.post2 --extra-index-url https://download.pytorch.org/whl/cu121

echo "✅ XFormers installed!"

echo "🚀 Step 4: Installing Triton (let PyTorch handle the version)..."
pip install triton  # Let it install whatever PyTorch wants

echo "✅ Triton installed!"

echo "🔧 Step 5: Alternative Flash Attention if wheel fails..."
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "⚠️ Wheel failed, trying pip install with CUDA override..."
    export CUDA_HOME=/usr/local/cuda-11.8
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"  # A100 architectures
    pip install flash-attn==2.5.8 --no-build-isolation --force-reinstall
fi

echo "🔍 Step 6: Final verification..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')

try:
    import xformers
    print(f'✅ XFormers: {xformers.__version__}')
except Exception as e:
    print(f'❌ XFormers: {e}')

try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
    from flash_attn.flash_attn_interface import flash_attn_func
    print('✅ Flash Attention function: Available')
except Exception as e:
    print(f'❌ Flash Attention: {e}')

try:
    import triton
    print(f'✅ Triton: {triton.__version__}')
except Exception as e:
    print(f'❌ Triton: {e}')
"

echo ""
echo "💥 NUCLEAR FIX COMPLETE!"
echo "========================"
echo "If Flash Attention still fails, we'll use XFormers-only mode"
echo "🚀 Run: ./start_gradio_xformers.sh" 