#!/bin/bash

echo "🚀 Setting up LatentSync on RunPod A100..."

# Update system packages
echo "📦 Updating system packages..."
apt update && apt install -y libgl1-mesa-glx libglib2.0-0 wget git curl unzip

# Install conda if not present
if ! command -v conda &> /dev/null; then
    echo "🐍 Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    rm miniconda.sh
fi

# Initialize conda
source $HOME/miniconda/bin/activate
conda init bash
source ~/.bashrc

# Create conda environment
echo "🌟 Creating conda environment..."
conda create -y -n latentsync python=3.10.13
source activate latentsync

# Install ffmpeg
echo "🎥 Installing ffmpeg..."
conda install -y -c conda-forge ffmpeg

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install additional dependencies for better A100 performance
echo "⚡ Installing A100 optimizations..."
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation

# Create checkpoints directory
echo "📁 Creating checkpoints directory..."
mkdir -p checkpoints

# Download model checkpoints
echo "⬇️ Downloading model checkpoints..."
pip install huggingface-hub
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# Set up auxiliary models directory
echo "🔗 Setting up auxiliary models..."
mkdir -p ~/.cache/torch/hub/checkpoints

# Create temp directory for outputs
mkdir -p temp

# Set proper permissions
chmod -R 755 checkpoints
chmod -R 755 temp

echo "✅ Setup complete! LatentSync is ready to use on A100."
echo "🌐 To start Gradio interface, run: ./start_gradio.sh" 