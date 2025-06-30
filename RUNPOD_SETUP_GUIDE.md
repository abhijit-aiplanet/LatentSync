# 🚀 LatentSync A100 RunPod Setup Guide

## 📋 Prerequisites & RunPod Credentials

### RunPod Account Setup
1. **Create RunPod Account**: Go to [runpod.io](https://runpod.io) and sign up
2. **Add Credits**: Add credits to your account ($0.50-$2.00/hour for A100)
3. **No API Keys Required**: RunPod works through their web interface
4. **Optional**: Set up SSH keys for easier access

### Required Information
- **RunPod Username/Password**: For logging into the platform
- **Credit Balance**: Ensure sufficient credits for your usage
- **SSH Key (Optional)**: For terminal access

## 🎯 RunPod Instance Setup

### Step 1: Create GPU Instance
1. Log into RunPod dashboard
2. Click **"Deploy"** → **"GPU Cloud"**
3. Select **A100 SXM4 40GB** or **A100 SXM4 80GB**
4. Choose **PyTorch 2.1** template or **Ubuntu 22.04**
5. Set **Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
6. **Volume Size**: At least 50GB for models and outputs
7. **Network Volume**: Optional, for persistent storage
8. **Expose Ports**: 7860 (for Gradio), 22 (for SSH)

### Step 2: Instance Configuration
- **Instance Type**: A100 SXM4 (40GB or 80GB)
- **vCPU**: 16+ cores recommended
- **RAM**: 64GB+ recommended
- **Network**: High-speed network for model downloads

## 🛠️ Installation Process

### Option A: Quick Setup (Recommended)
```bash
# 1. Connect to your RunPod instance terminal
# 2. Navigate to workspace
cd /workspace

# 3. Clone LatentSync repository
git clone https://github.com/bytedance/LatentSync.git
cd LatentSync

# 4. Make setup script executable
chmod +x setup_runpod_a100.sh

# 5. Run the setup (takes ~10-15 minutes)
./setup_runpod_a100.sh

# 6. Start Gradio interface
chmod +x start_gradio.sh
./start_gradio.sh
```

### Option B: Manual Setup
```bash
# Install system dependencies
apt update && apt install -y libgl1-mesa-glx libglib2.0-0 wget git

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="$HOME/miniconda/bin:$PATH"

# Create environment
conda create -y -n latentsync python=3.10.13
conda activate latentsync

# Install dependencies
pip install -r requirements_a100.txt

# Download models
huggingface-cli download ByteDance/LatentSync-1.6 --local-dir checkpoints
```

## 🌐 Accessing Gradio Interface

### Method 1: Public URL (Recommended)
1. Run `./start_gradio.sh`
2. Look for the **"Running on public URL"** message
3. Click the provided link (e.g., `https://xxxxx.gradio.live`)
4. The interface will be accessible worldwide

### Method 2: RunPod Port Forwarding
1. Go to RunPod dashboard
2. Find your instance and click **"Connect"**
3. Use **HTTP [7860]** link to access Gradio interface

## 🎬 Using the Interface

### Upload Files
- **Video**: Upload MP4 files (recommended resolution: 256x256 to 1024x1024)
- **Audio**: Upload WAV files (16kHz recommended)

### Optimal A100 Settings
- **Guidance Scale**: 2.0-2.5 for best results
- **Inference Steps**: 25-35 (higher = better quality)
- **DeepCache**: Enable for faster processing
- **Resolution**: Up to 512×512 for maximum quality

### Processing Times (A100)
- **5-second video**: ~30-60 seconds
- **10-second video**: ~1-2 minutes
- **30-second video**: ~3-5 minutes

## 💰 Cost Optimization

### A100 Pricing (Approximate)
- **A100 40GB**: $1.20-$2.00/hour
- **A100 80GB**: $2.00-$3.00/hour
- **Data Transfer**: Usually included

### Tips to Save Money
1. **Stop instances** when not in use
2. **Use Spot instances** for 50% discount (may be interrupted)
3. **Batch process** multiple videos in one session
4. **Use Network Volumes** for persistent storage

## 🔧 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use smaller videos
2. **Model Download Fails**: Check internet connection
3. **Gradio Not Accessible**: Ensure port 7860 is exposed
4. **Slow Processing**: Verify A100 is being used (`nvidia-smi`)

### Verification Commands
```bash
# Check GPU
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check models
ls -la checkpoints/

# Test Gradio
curl -I http://localhost:7860
```

## 📊 Performance Expectations

### A100 40GB
- **Max Resolution**: 512×512
- **Batch Size**: 1-2
- **VRAM Usage**: ~20-30GB
- **Processing Speed**: ~2-3x faster than RTX 3090

### A100 80GB  
- **Max Resolution**: 512×512+
- **Batch Size**: 2-4
- **VRAM Usage**: ~30-50GB
- **Processing Speed**: ~3-4x faster than RTX 3090

## 🔒 Security Notes

- RunPod instances are **temporary** by default
- Use **Network Volumes** for persistent data
- **Don't store sensitive data** on temporary instances
- **Stop instances** when not in use to prevent unauthorized access

## 📞 Support

### RunPod Support
- **Discord**: [RunPod Community](https://discord.gg/runpod)
- **Documentation**: [docs.runpod.io](https://docs.runpod.io)
- **Email**: support@runpod.io

### LatentSync Issues
- **GitHub**: [LatentSync Issues](https://github.com/bytedance/LatentSync/issues)
- **Paper**: [ArXiv](https://arxiv.org/abs/2412.09262)

## 🎉 Quick Start Checklist

- [ ] RunPod account created with credits
- [ ] A100 instance launched
- [ ] Repository cloned
- [ ] `setup_runpod_a100.sh` executed successfully
- [ ] Gradio interface accessible
- [ ] Test video processed successfully
- [ ] Instance stopped when not in use

---

**Estimated Total Setup Time**: 15-20 minutes  
**Estimated Cost**: $2-5 for setup and testing  
**Ready to Process**: High-quality lip-sync videos! 🎬 