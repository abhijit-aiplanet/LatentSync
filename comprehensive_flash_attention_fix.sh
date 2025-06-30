#!/bin/bash

echo "🔥 COMPREHENSIVE FLASH ATTENTION FIX"
echo "🎯 Fixing CUDA version mismatch, headers, and environment..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Stop any conflicting processes
pkill -f gradio_app 2>/dev/null || true

echo "🗑️ Cleaning up previous installations..."
pip uninstall -y flash-attn 2>/dev/null || true
conda clean -a -y

echo "📦 Step 1: Install complete CUDA 12.1 toolkit with headers..."
# Remove conflicting CUDA versions first
conda remove -y cuda-toolkit cuda-nvcc cuda-compiler --force 2>/dev/null || true

# Install complete CUDA 12.1 with all development components
conda install -y -c nvidia \
    cuda-toolkit=12.1 \
    cuda-nvcc=12.1 \
    cuda-compiler=12.1 \
    cuda-libraries-dev=12.1 \
    cuda-runtime=12.1 \
    cuda-cupti=12.1

echo "🔧 Step 2: Install ninja build system..."
conda install -y ninja

echo "🔧 Step 3: Fix CUDA environment variables..."
# Set correct CUDA paths
export CUDA_HOME=$CONDA_PREFIX
export CUDA_ROOT=$CONDA_PREFIX  
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

# Add to conda environment
echo "export CUDA_HOME=$CONDA_PREFIX" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
echo "export CUDA_ROOT=$CONDA_PREFIX" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
echo "export PATH=$CUDA_HOME/bin:\$PATH" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
echo "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
echo "export CPATH=$CUDA_HOME/include:\$CPATH" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh
echo "export LIBRARY_PATH=$CUDA_HOME/lib64:\$LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/cuda.sh

# Create directories if missing
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/

echo "🔍 Step 4: Verify CUDA installation..."
which nvcc
nvcc --version
echo "CUDA_HOME: $CUDA_HOME"
ls -la $CUDA_HOME/include/ | grep cuda 2>/dev/null || echo "No CUDA headers found"

echo "🔧 Step 5: Install build dependencies..."
pip install ninja setuptools wheel packaging

echo "🔥 Step 6: Try multiple Flash Attention installation methods..."

# Method 1: Pre-built wheel for exact versions
echo "Method 1: Trying pre-built wheels..."
pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu121torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" 2>/dev/null

# Check if successful
python -c "import flash_attn; print('✅ Method 1: Success!')" 2>/dev/null && {
    echo "🎉 Flash Attention installed successfully via pre-built wheel!"
    exit 0
}

# Method 2: Alternative wheel source
echo "Method 2: Alternative wheel sources..."
pip install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" 2>/dev/null

python -c "import flash_attn; print('✅ Method 2: Success!')" 2>/dev/null && {
    echo "🎉 Flash Attention installed successfully via alternative wheel!"
    exit 0
}

# Method 3: Compilation with proper environment
echo "Method 3: Compilation from source with fixed environment..."

# Set compilation flags
export TORCH_CUDA_ARCH_LIST="8.0;9.0"  # A100 and H100 architectures
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=4

# Install from source with proper flags
pip install --no-build-isolation --no-cache-dir \
    flash-attn==2.6.3 \
    --global-option="build_ext" \
    --global-option="--include-dirs=$CUDA_HOME/include" \
    --global-option="--library-dirs=$CUDA_HOME/lib64"

python -c "import flash_attn; print('✅ Method 3: Success!')" 2>/dev/null && {
    echo "🎉 Flash Attention compiled successfully from source!"
    exit 0
}

# Method 4: Install specific compatible version
echo "Method 4: Installing compatible version..."
pip install --no-cache-dir flash-attn==2.5.8

python -c "import flash_attn; print('✅ Method 4: Success!')" 2>/dev/null && {
    echo "🎉 Flash Attention installed (compatible version)!"
    exit 0
}

# Method 5: Direct GitHub installation
echo "Method 5: Direct GitHub installation..."
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

python -c "import flash_attn; print('✅ Method 5: Success!')" 2>/dev/null && {
    echo "🎉 Flash Attention installed from GitHub!"
    exit 0
}

echo "❌ All methods failed. Checking final status..."

# Final verification and diagnostics
echo "🔍 Final Diagnostics:"
echo "PyTorch CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "System CUDA version: $(nvcc --version | grep 'release' || echo 'Not found')"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA headers: $(ls $CUDA_HOME/include/cuda*.h 2>/dev/null | wc -l) files"

python -c "
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print('🎉 SUCCESS: Flash Attention is working!')
except ImportError as e:
    print(f'❌ Flash Attention still not available: {e}')
    print('⚠️ Will continue with XFormers optimization')
except Exception as e:
    print(f'⚠️ Flash Attention imported but has issues: {e}')
"

echo "🚀 Flash Attention installation process complete!"
echo "ℹ️ If all methods failed, XFormers will still provide 3-4x speedup" 