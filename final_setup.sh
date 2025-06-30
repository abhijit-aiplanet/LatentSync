#!/bin/bash

# LatentSync Final Setup Script - Nuclear Option
# This script completely wipes and reinstalls everything for maximum performance
# Uses exact versions from requirements.txt and requirements_a100.txt

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

# Step 3: Install CUDA-enabled PyTorch (from requirements.txt - CUDA 12.1)
echo "🔥 Step 3: Installing optimized PyTorch with CUDA 12.1..."
pip install torch==2.5.1 torchvision==0.20.1 --extra-index-url https://download.pytorch.org/whl/cu121

echo "✅ PyTorch installed!"

# Step 4: Install core requirements from requirements.txt
echo "📦 Step 4: Installing core requirements from requirements.txt..."
pip install diffusers==0.32.2
pip install transformers==4.48.0
pip install decord==0.6.0
pip install accelerate==0.26.1
pip install einops==0.7.0
pip install omegaconf==2.3.0
pip install numpy==1.26.4

echo "✅ Core requirements installed!"

# Step 5: Install computer vision packages
echo "👁️ Step 5: Installing computer vision packages..."
pip install opencv-python==4.9.0.80
pip install mediapipe==0.10.11
pip install face-alignment==1.4.1
pip install insightface==0.7.3
pip install kornia==0.8.0

echo "✅ Computer vision packages installed!"

# Step 6: Install audio processing packages
echo "🎵 Step 6: Installing audio processing packages..."
pip install librosa==0.10.1
pip install python_speech_features==0.6

echo "✅ Audio packages installed!"

# Step 7: Install video processing packages
echo "🎬 Step 7: Installing video processing packages..."
pip install scenedetect==0.6.1
pip install ffmpeg-python==0.2.0
pip install imageio==2.31.1
pip install imageio-ffmpeg==0.5.1

echo "✅ Video packages installed!"

# Step 8: Install ONNX for model optimization
echo "🔧 Step 8: Installing ONNX for model optimization..."
pip install onnxruntime-gpu==1.21.0

echo "✅ ONNX packages installed!"

# Step 9: Install web interface packages
echo "🌐 Step 9: Installing web interface packages..."
pip install gradio==5.24.0
pip install huggingface-hub==0.30.2

echo "✅ Web interface packages installed!"

# Step 10: Install remaining LatentSync specific dependencies
echo "🎯 Step 10: Installing LatentSync specific dependencies..."
pip install lpips==0.1.4
pip install DeepCache==0.1.1

echo "✅ LatentSync dependencies installed!"

# Step 11: Install A100 specific optimizations
echo "⚡ Step 11: Installing A100 specific optimizations..."
pip install packaging ninja
pip install "xformers>=0.0.23"
pip install "flash-attn>=2.0.0" --no-build-isolation
pip install "triton>=2.0.0"

echo "✅ A100 optimizations installed!"

# Step 12: Install additional useful packages
echo "🛠️ Step 12: Installing additional useful packages..."
pip install tqdm
pip install rich
pip install soundfile
pip install pillow
pip install matplotlib

echo "✅ Additional packages installed!"

# Step 13: Set optimal environment variables
echo "⚙️ Step 13: Setting optimal environment variables..."
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1
export XFORMERS_FORCE_DISABLE_TRITON=0  # Enable Triton for A100

# Make these permanent
echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" >> ~/.bashrc
echo "export TORCH_CUDNN_V8_API_ENABLED=1" >> ~/.bashrc
echo "export XFORMERS_FORCE_DISABLE_TRITON=0" >> ~/.bashrc

echo "✅ Environment variables set!"

# Step 14: Verify installation
echo "🔍 Step 14: Verifying installation..."
python -c "
import torch
import xformers
import flash_attn
import cv2
import librosa
import insightface
import onnxruntime
import diffusers
import transformers
import gradio
print('🎉 ALL PACKAGES IMPORTED SUCCESSFULLY!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'XFormers version: {xformers.__version__}')
print(f'Flash Attention available: {hasattr(flash_attn, \"flash_attn_func\")}')
print(f'Diffusers version: {diffusers.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Gradio version: {gradio.__version__}')
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