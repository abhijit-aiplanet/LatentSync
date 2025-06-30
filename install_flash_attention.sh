#!/bin/bash

echo "🔥 Installing Flash Attention with CUDA support..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Install CUDA development toolkit
echo "📦 Installing CUDA development toolkit..."
conda install -y -c nvidia cuda-toolkit=12.1 cuda-nvcc=12.1

# Verify nvcc installation
echo "🔍 Checking nvcc..."
which nvcc
nvcc --version

# Set CUDA environment variables
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo "🔥 Installing Flash Attention..."
pip install flash-attn==2.6.3 --no-build-isolation

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
"

echo "🚀 Flash Attention installation complete!" 