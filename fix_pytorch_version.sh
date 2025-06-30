#!/bin/bash

echo "🔧 Fixing PyTorch version compatibility..."
source activate latentsync

echo "📦 Reinstalling correct PyTorch version..."
pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121 --force-reinstall

echo "✅ PyTorch version fixed!"

python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
" 