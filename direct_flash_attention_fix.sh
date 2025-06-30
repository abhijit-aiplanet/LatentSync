#!/bin/bash

echo "⚡ DIRECT FLASH ATTENTION FIX - FASTEST METHOD"
echo "🎯 Trying direct wheel installations first..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Clean previous attempts
pip uninstall -y flash-attn 2>/dev/null || true

echo "🔥 Method 1: Direct wheel installation (most likely to work)..."

# Try the most common working wheels first
WHEELS=(
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu121torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu122torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
)

for wheel in "${WHEELS[@]}"; do
    echo "🔄 Trying: $(basename $wheel)"
    pip install --force-reinstall --no-deps --no-cache-dir "$wheel" 2>/dev/null
    
    # Test if it works
    if python -c "import flash_attn; from flash_attn import flash_attn_func; print('✅ SUCCESS!')" 2>/dev/null; then
        echo "🎉 Flash Attention installed successfully!"
        echo "🔥 Wheel used: $wheel"
        python -c "import flash_attn; print(f'Version: {flash_attn.__version__}')"
        exit 0
    fi
done

echo "🔧 Method 2: PyPI with force flags..."
pip install --force-reinstall --no-cache-dir flash-attn==2.6.3 --find-links https://download.pytorch.org/whl/cu121 2>/dev/null

if python -c "import flash_attn; print('✅ PyPI Success!')" 2>/dev/null; then
    echo "🎉 Flash Attention installed via PyPI!"
    exit 0
fi

echo "🔧 Method 3: Compatible version fallback..."
# Try older compatible versions
for version in "2.5.8" "2.5.7" "2.5.6"; do
    echo "Trying version $version..."
    pip install --force-reinstall --no-cache-dir flash-attn==$version 2>/dev/null
    
    if python -c "import flash_attn; print('✅ Version $version works!')" 2>/dev/null; then
        echo "🎉 Flash Attention $version installed successfully!"
        exit 0
    fi
done

echo "❌ Direct methods failed. Running comprehensive fix..."
exec ./comprehensive_flash_attention_fix.sh 