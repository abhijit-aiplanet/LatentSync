#!/bin/bash

echo "🔧 FIXING MISSING PACKAGES & XFORMERS..."
echo "🎯 Installing missing dependencies..."

# Activate environment
export PATH="/root/miniconda/bin:$PATH"
source activate latentsync

echo "📦 Installing missing packages..."

# Install missing packages
pip install imageio imageio-ffmpeg
pip install einops kornia
pip install scipy scikit-image
pip install tensorboard wandb
pip install onnxruntime-gpu

echo "✅ Missing packages installed"

echo "🔧 Fixing XFormers for CUDA 11.8..."

# Uninstall current XFormers and install CUDA 11.8 compatible version
pip uninstall -y xformers

# Install XFormers compatible with CUDA 11.8
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu118

echo "✅ XFormers fixed for CUDA 11.8"

echo "🧪 Testing installations..."

python -c "
print('🔍 Testing all packages...')

try:
    import imageio
    print('✅ imageio imported successfully')
    
    import einops
    print('✅ einops imported successfully')
    
    import kornia
    print('✅ kornia imported successfully')
    
    import xformers
    print(f'✅ XFormers {xformers.__version__} imported successfully')
    
    import flash_attn
    print(f'✅ Flash Attention {flash_attn.__version__} working')
    
    import torch
    print(f'✅ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')
    
    # Test GPU
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    
    print('🎉 All packages working correctly!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎉 PACKAGE FIX COMPLETE!"
echo "🚀 Now try: ./start_gradio_optimized.sh" 