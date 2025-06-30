#!/bin/bash

# LatentSync Final Setup Script - Nuclear Option
# This script completely wipes and reinstalls everything for maximum performance

set -e  # Exit on any error

echo "🚀 LatentSync Final Setup - Nuclear Clean Install"
echo "=================================================="

# Step 1: Nuclear cleanup
echo "💥 Step 1: Nuclear cleanup - removing everything..."
conda deactivate 2>/dev/null || true
conda env remove -n latentsync -y 2>/dev/null || true
conda clean --all -y 2>/dev/null || true
pip cache purge 2>/dev/null || true

# Clean any leftover pip packages in base environment
pip uninstall torch torchvision torchaudio xformers flash-attn -y 2>/dev/null || true

echo "✅ Cleanup complete!"

# Step 2: Create fresh conda environment
echo "🔧 Step 2: Creating fresh conda environment..."
conda create -n latentsync python=3.10 -y
source activate latentsync

echo "✅ Fresh environment created!"

# Step 3: Install CUDA-enabled PyTorch (latest stable)
echo "🔥 Step 3: Installing optimized PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "✅ PyTorch installed!"

# Step 4: Install XFormers for memory optimization
echo "⚡ Step 4: Installing XFormers for memory optimization..."
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

echo "✅ XFormers installed!"

# Step 5: Install Flash Attention for speed
echo "🚀 Step 5: Installing Flash Attention for maximum speed..."
pip install packaging ninja
pip install flash-attn==2.3.3 --no-build-isolation

echo "✅ Flash Attention installed!"

# Step 6: Install core AI/ML packages with specific versions
echo "🧠 Step 6: Installing core AI/ML packages..."
pip install numpy==1.24.3
pip install scipy==1.11.3
pip install scikit-learn==1.3.0
pip install pillow==10.0.1
pip install matplotlib==3.7.2

echo "✅ Core packages installed!"

# Step 7: Install computer vision packages
echo "👁️ Step 7: Installing computer vision packages..."
pip install opencv-contrib-python==4.8.1.78
pip install insightface==0.7.3
pip install face-alignment==1.3.5
pip install albumentations==1.3.1

echo "✅ Computer vision packages installed!"

# Step 8: Install audio processing packages
echo "🎵 Step 8: Installing audio processing packages..."
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install torchaudio==2.1.0

echo "✅ Audio packages installed!"

# Step 9: Install ONNX for model optimization
echo "🔧 Step 9: Installing ONNX for model optimization..."
pip install onnx==1.14.1
pip install onnxruntime-gpu==1.16.1

echo "✅ ONNX packages installed!"

# Step 10: Install video processing packages
echo "🎬 Step 10: Installing video processing packages..."
pip install moviepy==1.0.3
pip install decord==0.6.0
pip install av==10.0.0

echo "✅ Video packages installed!"

# Step 11: Install deep learning utilities
echo "🔧 Step 11: Installing deep learning utilities..."
pip install timm==0.9.7
pip install transformers==4.34.0
pip install diffusers==0.21.4
pip install accelerate==0.23.0

echo "✅ Deep learning utilities installed!"

# Step 12: Install web interface packages
echo "🌐 Step 12: Installing web interface packages..."
pip install gradio==3.50.2
pip install fastapi==0.104.1
pip install uvicorn==0.24.0

echo "✅ Web interface packages installed!"

# Step 13: Install development utilities
echo "🛠️ Step 13: Installing development utilities..."
pip install tqdm==4.66.1
pip install rich==13.6.0
pip install pydantic==2.4.2
pip install omegaconf==2.3.0
pip install tensorboard==2.14.1

echo "✅ Development utilities installed!"

# Step 14: Install remaining LatentSync specific dependencies
echo "🎯 Step 14: Installing LatentSync specific dependencies..."
pip install einops==0.7.0
pip install kornia==0.7.0
pip install lpips==0.1.4
pip install cleanfid==0.1.35

echo "✅ LatentSync dependencies installed!"

# Step 15: Set optimal environment variables
echo "⚙️ Step 15: Setting optimal environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1
export XFORMERS_FORCE_DISABLE_TRITON=1

# Make these permanent
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" >> ~/.bashrc
echo "export TORCH_CUDNN_V8_API_ENABLED=1" >> ~/.bashrc
echo "export XFORMERS_FORCE_DISABLE_TRITON=1" >> ~/.bashrc

echo "✅ Environment variables set!"

# Step 16: Verify installation
echo "🔍 Step 16: Verifying installation..."
python -c "
import torch
import xformers
import flash_attn
import cv2
import librosa
import insightface
import onnxruntime
print('🎉 ALL PACKAGES IMPORTED SUCCESSFULLY!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'XFormers version: {xformers.__version__}')
print(f'Flash Attention available: {hasattr(flash_attn, \"flash_attn_func\")}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "🎉🎉🎉 INSTALLATION COMPLETE! 🎉🎉🎉"
echo "=================================================="
echo "✅ All dependencies installed with optimal versions"
echo "✅ XFormers enabled for memory optimization"
echo "✅ Flash Attention enabled for speed optimization"
echo "✅ CUDA optimizations enabled"
echo "✅ Environment variables set for maximum performance"
echo ""
echo "🚀 Your LatentSync setup is now FULLY OPTIMIZED!"
echo "Run: source activate latentsync && python gradio_app_optimized.py"
echo "==================================================" 