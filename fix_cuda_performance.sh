#!/bin/bash

echo "🔧 Fixing CUDA Performance Issues..."

# Activate environment
source $HOME/miniconda/bin/activate
source activate latentsync

# Install missing CUDA libraries
echo "📦 Installing missing CUDA libraries..."
conda install -y -c nvidia cuda-nvrtc=12.1 cuda-toolkit=12.1

# Fix ONNX Runtime CUDA
echo "⚡ Reinstalling ONNX Runtime with proper CUDA support..."
pip uninstall -y onnxruntime-gpu onnxruntime
pip install onnxruntime-gpu==1.21.0

# Install optimized InsightFace
echo "👤 Reinstalling InsightFace with GPU support..."
pip uninstall -y insightface
pip install insightface==0.7.3 --no-deps
pip install onnx onnxruntime-gpu

# Set environment variables
echo "🌍 Setting CUDA environment variables..."
export CUDA_HOME=/opt/conda/envs/latentsync
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# Add to bashrc for persistence
echo 'export CUDA_HOME=/opt/conda/envs/latentsync' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc

echo "✅ CUDA performance fix completed!"
echo "🔄 Please restart Gradio: ./start_gradio.sh" 