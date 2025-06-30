#!/bin/bash

echo "⚡ Installing Flash Attention using pre-built wheels..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Try different pre-built wheel sources
echo "🔍 Trying pre-built wheels..."

# Option 1: Try PyPI with specific CUDA version
pip install flash-attn==2.6.3 --find-links https://download.pytorch.org/whl/cu121

# If that fails, try alternative sources
if [ $? -ne 0 ]; then
    echo "⚡ Trying alternative wheel source..."
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --force-reinstall
fi

# If still fails, try without build isolation with modified CUDA path
if [ $? -ne 0 ]; then
    echo "🔧 Setting up minimal CUDA environment..."
    
    # Create symbolic link to existing CUDA if available
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d "/opt/conda/pkgs/cuda-toolkit*" ]; then
        export CUDA_HOME=$(ls -d /opt/conda/pkgs/cuda-toolkit* | head -n1)
    fi
    
    # Try compilation with existing CUDA
    pip install flash-attn==2.6.3 --no-build-isolation --no-cache-dir
fi

# Verify installation
echo "✅ Testing Flash Attention..."
python -c "
try:
    import flash_attn
    print('✅ Flash Attention: Successfully installed!')
    from flash_attn import flash_attn_func
    print('✅ Flash Attention functions: Available!')
except Exception as e:
    print(f'❌ Flash Attention error: {e}')
    print('⚠️ Continuing without Flash Attention - XFormers will provide major speedup!')
"

echo "🚀 Installation attempt complete!" 