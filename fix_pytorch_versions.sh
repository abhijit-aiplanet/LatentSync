#!/bin/bash

echo "🔧 FIXING PYTORCH/TORCHVISION VERSION MISMATCH..."
echo "🎯 Installing compatible versions..."

# Activate environment
export PATH="/root/miniconda/bin:$PATH"
source activate latentsync

echo "🗑️ Uninstalling conflicting versions..."

# Uninstall current versions
pip uninstall -y torch torchvision torchaudio

echo "📦 Installing compatible PyTorch ecosystem..."

# Install compatible versions for CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "✅ PyTorch ecosystem reinstalled"

echo "🔧 Reinstalling XFormers for new PyTorch version..."

# Reinstall XFormers for the new PyTorch version
pip uninstall -y xformers
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

echo "✅ XFormers reinstalled"

echo "🧪 Testing installation..."

python -c "
print('🔍 Testing PyTorch/torchvision compatibility...')

try:
    import torch
    import torchvision
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ Torchvision: {torchvision.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ CUDA version: {torch.version.cuda}')
    
    # Test torchvision operations
    from torchvision.transforms import InterpolationMode
    print('✅ Torchvision transforms working')
    
    # Test XFormers
    import xformers
    print(f'✅ XFormers: {xformers.__version__}')
    
    # Test Flash Attention
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
    
    print('🎉 All versions compatible!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🎉 VERSION FIX COMPLETE!"
echo "🚀 Now try: ./start_gradio_optimized.sh" 