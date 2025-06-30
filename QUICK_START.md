# 🚀 Quick Start: LatentSync on RunPod A100

## 📋 What You'll Need for RunPod

### RunPod Credentials & Setup
1. **Account**: Go to [runpod.io](https://runpod.io) and create account
2. **Credits**: Add $10-20 credits (A100 costs ~$1.50-2.50/hour)
3. **No API Keys**: Just username/password needed
4. **SSH Key**: Optional, but recommended for easier access

## 🎯 5-Minute Setup

### Step 1: Launch RunPod Instance
1. Login to RunPod → Click **"Deploy"** → **"GPU Cloud"**
2. Select **A100 SXM4 40GB** or **A100 SXM4 80GB**
3. Template: **PyTorch 2.1** or **Ubuntu 22.04**
4. Container: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
5. Expose ports: **7860** (Gradio), **22** (SSH)
6. Volume: **50GB** minimum
7. Click **Deploy**

### Step 2: Setup LatentSync (10-15 minutes)
```bash
# Connect to your RunPod terminal
cd /workspace

# Clone and setup
git clone https://github.com/bytedance/LatentSync.git
cd LatentSync

# Run automated setup
./setup_runpod_a100.sh

# Verify setup
python test_setup.py
```

### Step 3: Launch Gradio Interface
```bash
# Start the web interface
./start_gradio.sh

# Look for the public URL in the output:
# "Running on public URL: https://xxxxx.gradio.live"
```

## 🎬 Ready to Use!

### Upload & Process
1. **Upload Video**: MP4 format, any resolution
2. **Upload Audio**: WAV format (16kHz recommended)
3. **Settings**: Use defaults or optimize:
   - Guidance Scale: 2.0-2.5
   - Inference Steps: 25-35
   - Enable DeepCache: ✅
4. **Click Process**: Wait for magic! ✨

### Performance Expectations
- **5-sec video**: ~30-60 seconds
- **10-sec video**: ~1-2 minutes  
- **Quality**: 512×512 high-resolution output
- **VRAM Usage**: ~20-30GB (well within A100 limits)

## 💰 Cost Breakdown

### Estimated Costs
- **Setup**: $2-5 (one-time)
- **Processing**: $0.50-1.00 per 10-second video
- **Idle Time**: $0 (stop instance when not using)

### Money-Saving Tips
- Stop instance when not in use
- Use Spot instances (50% cheaper, may be interrupted)
- Batch process multiple videos

## 🔧 Files Created

Your setup includes these optimized files:
- `setup_runpod_a100.sh` - Automated installation
- `start_gradio.sh` - Launch web interface  
- `gradio_app_a100.py` - A100-optimized Gradio app
- `requirements_a100.txt` - A100-specific dependencies
- `test_setup.py` - Verify installation
- `RUNPOD_SETUP_GUIDE.md` - Detailed documentation

## 🆘 Troubleshooting

### Quick Fixes
```bash
# GPU not detected
nvidia-smi

# Models missing
ls -la checkpoints/

# Gradio not accessible
curl -I http://localhost:7860

# Re-run setup if needed
./setup_runpod_a100.sh
```

### Common Issues
- **Out of memory**: Use smaller videos or reduce batch size
- **Slow processing**: Verify A100 is being used
- **Network issues**: Check RunPod connection
- **Model download fails**: Check internet connection

## 🎉 Success Checklist

- [ ] RunPod A100 instance running
- [ ] LatentSync repository cloned
- [ ] Setup script completed successfully
- [ ] Test script passes all checks
- [ ] Gradio interface accessible via public URL
- [ ] First video processed successfully
- [ ] Instance stopped when not in use

---

**Total Time**: ~20 minutes from start to first video  
**Total Cost**: ~$3-5 for setup and testing  
**Result**: Professional-quality lip-sync videos! 🎬

Ready to create amazing lip-sync content? Let's go! 🚀 