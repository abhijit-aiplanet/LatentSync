#!/bin/bash

echo "🔧 CUDA RUNTIME HEADERS FIX"
echo "🎯 Installing CUDA runtime and headers directly from NVIDIA..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Install system dependencies
echo "📦 Installing system build tools..."
apt-get update -qq
apt-get install -y build-essential wget curl

echo "📦 Adding NVIDIA CUDA repository..."
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb

# Update package list
apt-get update -qq

echo "🔧 Installing CUDA 12.1 runtime and development headers..."
# Install specific CUDA 12.1 components
apt-get install -y \
    cuda-runtime-12-1 \
    cuda-toolkit-12-1 \
    libcudnn8-dev \
    libcublas-dev-12-1 \
    libcufft-dev-12-1 \
    libcurand-dev-12-1 \
    libcusolver-dev-12-1 \
    libcusparse-dev-12-1

echo "🔧 Setting up CUDA environment..."
# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH

# Create symlink to expected location
ln -sf /usr/local/cuda-12.1 /usr/local/cuda

echo "🔍 Verifying CUDA installation..."
nvcc --version
ls -la /usr/local/cuda/include/cuda_runtime.h 2>/dev/null || echo "❌ cuda_runtime.h still missing"

echo "🔧 Installing ninja and build tools..."
pip install ninja packaging setuptools wheel

echo "🔥 Now trying Flash Attention installation..."
# Set proper environment for compilation
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4

# Try installation with proper CUDA paths
pip install --no-build-isolation --no-cache-dir flash-attn==2.6.3

# Verify
python -c "
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print('🎉 SUCCESS: Flash Attention is working!')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo "🚀 CUDA runtime fix complete!" 