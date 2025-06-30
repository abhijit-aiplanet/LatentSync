#!/bin/bash

# Installs Flash Attention precompiled cu121 wheel for PyTorch 2.5.1 + Python 3.10
# Works on A100 GPUs in RunPod environment (driver CUDA 11.8 runtime is fine)

set -e

echo "🚀 Installing Flash Attention (cu121) for PyTorch 2.5.1..."

source activate latentsync

echo "🧹 Removing conflicting packages..."
pip uninstall -y flash-attn xformers || true

# Step 1: Ensure correct PyTorch/TorchVision

echo "📦 Installing PyTorch 2.5.1 + cu121 ..."
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Step 2: Install matching XFormers

echo "⚡ Installing XFormers 0.0.28.post3 ..."
pip install xformers==0.0.28.post3 --extra-index-url https://download.pytorch.org/whl/cu121

# Step 3: Install precompiled Flash Attention wheel (cu121, torch2.5, py310)

WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu121torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "🌐 Downloading Flash Attention wheel from: $WHEEL_URL"
pip install "$WHEEL_URL"

echo "🔍 Verifying installation..."
python - << 'PY'
import torch, subprocess, os, sys
print(f"✅ PyTorch: {torch.__version__}")
try:
    import flash_attn, xformers, triton
    from flash_attn.flash_attn_interface import flash_attn_func
    print(f"✅ Flash Attention: {flash_attn.__version__}")
    print(f"✅ XFormers: {xformers.__version__}")
    print(f"✅ Triton: {triton.__version__}")
    print("🎉 Flash Attention kernel ready!")
except Exception as e:
    print("❌ Verification failed:", e)
    sys.exit(1)
PY

echo "🎉 Flash Attention cu121 installation SUCCESS!" 