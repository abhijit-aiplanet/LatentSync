#!/bin/bash

echo "🔧 FIXING NUMPY VERSION & INSTALLING DECORD..."
echo "🎯 Downgrading NumPy and installing missing packages..."

# Activate environment
export PATH="/root/miniconda/bin:$PATH"
source activate latentsync

echo "📦 Fixing NumPy version compatibility..."

# Downgrade NumPy to be compatible with PyTorch 2.1.0
pip install "numpy<2.0" --force-reinstall

echo "✅ NumPy downgraded to compatible version"

echo "📦 Installing missing packages..."

# Install decord for video/audio reading
pip install decord

# Install other potentially missing packages
pip install av ffmpeg-python

echo "✅ Missing packages installed"

echo "🧪 Testing installation..."

python -c "
print('🔍 Testing NumPy and package compatibility...')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
    
    import torch
    print(f'✅ PyTorch: {torch.__version__} (should load without NumPy warnings)')
    
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ CUDA version: {torch.version.cuda}')
    
    # Test decord
    import decord
    print(f'✅ Decord: {decord.__version__}')
    
    # Test other video packages
    try:
        import av
        print('✅ PyAV imported successfully')
    except:
        print('⚠️ PyAV not available (not critical)')
    
    # Test Flash Attention
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
    
    # Test XFormers
    import xformers
    print(f'✅ XFormers: {xformers.__version__}')
    
    # Test other core packages
    import diffusers, transformers, gradio
    print(f'✅ Diffusers: {diffusers.__version__}')
    print(f'✅ Transformers: {transformers.__version__}')
    print(f'✅ Gradio: {gradio.__version__}')
    
    print('🎉 All packages working correctly!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎉 NUMPY & DECORD FIX COMPLETE!"
echo "🚀 Now try: ./start_gradio_optimized.sh" 