#!/bin/bash

echo "🔧 Fixing PyTorch/XFormers/Triton version compatibility..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Stop any running processes
pkill -f gradio_app 2>/dev/null || true

echo "🗑️ Removing incompatible packages..."
pip uninstall -y torch torchvision torchaudio xformers triton flash-attn

echo "📦 Installing compatible PyTorch ecosystem..."

# Install specific compatible versions
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install compatible XFormers
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121

# Install compatible Triton
pip install triton==3.1.0

# Install Flash Attention compatible version
pip install flash-attn==2.6.3 --no-build-isolation

echo "🔍 Verifying installations..."

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test XFormers
python -c "
try:
    import xformers
    import xformers.ops
    print(f'XFormers: {xformers.__version__} ✅')
except Exception as e:
    print(f'XFormers error: {e}')
"

# Test Triton
python -c "
try:
    import triton
    print(f'Triton: {triton.__version__} ✅')
except Exception as e:
    print(f'Triton error: {e}')
"

# Test Flash Attention
python -c "
try:
    import flash_attn
    print('Flash Attention: Available ✅')
except Exception as e:
    print(f'Flash Attention error: {e}')
"

echo "🎯 Installing additional optimizations..."

# Reinstall other dependencies that might be affected
pip install --upgrade diffusers==0.32.2 accelerate kornia

echo "✅ Version compatibility fix complete!"
echo "🚀 Try starting Gradio again: ./start_gradio_optimized.sh" 