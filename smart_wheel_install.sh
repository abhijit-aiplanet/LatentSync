#!/bin/bash

echo "🎯 SMART FLASH ATTENTION WHEEL INSTALLER"
echo "🔍 Finding compatible wheels automatically..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Clean previous attempts
pip uninstall -y flash-attn 2>/dev/null || true

# Get system info
PYTHON_VERSION="cp310"
PLATFORM="linux_x86_64"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")

echo "🔍 System Info:"
echo "Python: $PYTHON_VERSION"
echo "PyTorch: $TORCH_VERSION"  
echo "CUDA: $CUDA_VERSION"
echo "Platform: $PLATFORM"

echo "🔍 Searching for compatible wheels..."

# List of potential wheel patterns to try
BASE_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3"

# Generate possible wheel names
WHEEL_PATTERNS=(
    "flash_attn-2.6.3+cu121torch2.5cxx11abiFALSE-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}.whl"
    "flash_attn-2.6.3+cu118torch2.5cxx11abiFALSE-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}.whl"
    "flash_attn-2.6.3+cu122torch2.5cxx11abiFALSE-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}.whl"
    "flash_attn-2.6.3+cu121torch2.4cxx11abiFALSE-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}.whl"
    "flash_attn-2.6.3-${PYTHON_VERSION}-${PYTHON_VERSION}-${PLATFORM}.whl"
)

# Function to test URL
test_url() {
    local url=$1
    echo "🔄 Testing: $(basename $url)"
    if curl --output /dev/null --silent --head --fail "$url"; then
        echo "✅ Found: $url"
        return 0
    else
        echo "❌ Not found: $(basename $url)"
        return 1
    fi
}

# Function to install and test wheel
install_and_test() {
    local url=$1
    echo "⬇️ Installing: $(basename $url)"
    
    if pip install --force-reinstall --no-deps --no-cache-dir "$url"; then
        if python -c "import flash_attn; from flash_attn import flash_attn_func; print('✅ Working!')" 2>/dev/null; then
            echo "🎉 SUCCESS: Flash Attention installed and working!"
            python -c "import flash_attn; print(f'Version: {flash_attn.__version__}')"
            return 0
        else
            echo "❌ Installed but not working"
            return 1
        fi
    else
        echo "❌ Installation failed"
        return 1
    fi
}

# Try each wheel pattern
for pattern in "${WHEEL_PATTERNS[@]}"; do
    wheel_url="$BASE_URL/$pattern"
    
    if test_url "$wheel_url"; then
        if install_and_test "$wheel_url"; then
            exit 0
        fi
    fi
done

echo "🔍 Trying alternative sources..."

# Try alternative repositories/sources
ALT_SOURCES=(
    "https://download.pytorch.org/whl/cu121/flash_attn-2.6.3%2Bcu121-cp310-cp310-linux_x86_64.whl"
    "https://files.pythonhosted.org/packages/flash_attn/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl"
)

for source in "${ALT_SOURCES[@]}"; do
    if test_url "$source"; then
        if install_and_test "$source"; then
            exit 0
        fi
    fi
done

echo "🔧 Trying version-specific installations..."

# Try specific versions that are known to work better
for version in "2.5.8" "2.5.7" "2.5.6" "2.4.2"; do
    echo "🔄 Trying Flash Attention v$version..."
    
    if pip install --force-reinstall --no-cache-dir "flash-attn==$version" 2>/dev/null; then
        if python -c "import flash_attn; print(f'✅ Version {version} works!')" 2>/dev/null; then
            echo "🎉 SUCCESS: Flash Attention v$version installed!"
            exit 0
        fi
    fi
done

echo "❌ Smart wheel installation failed."
echo "🔧 Running comprehensive fix..."
exec ./comprehensive_flash_attention_fix.sh 