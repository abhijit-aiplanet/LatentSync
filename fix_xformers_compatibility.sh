#!/bin/bash

echo "🔧 Fixing XFormers and Flash Attention Compatibility Issues..."
echo "============================================================"

# Make sure we're in the right environment
source activate latentsync

echo "💥 Step 1: Removing incompatible packages..."
pip uninstall xformers flash-attn triton -y 2>/dev/null || true

echo "✅ Cleanup complete!"

echo "🔧 Step 2: Installing compatible XFormers for PyTorch 2.5.1..."
pip install xformers==0.0.28.post3 --extra-index-url https://download.pytorch.org/whl/cu121

echo "✅ XFormers installed!"

echo "🚀 Step 3: Installing compatible Flash Attention..."
pip install packaging ninja
pip install flash-attn==2.5.9.post1 --no-build-isolation

echo "✅ Flash Attention installed!"

echo "⚡ Step 4: Installing Triton for GPU acceleration..."
pip install triton==2.3.1

echo "✅ Triton installed!"

echo "🔍 Step 5: Verifying installations..."
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')

try:
    import xformers
    print(f'✅ XFormers: {xformers.__version__}')
    # Test XFormers functionality
    import xformers.ops
    print('✅ XFormers ops: Available')
except Exception as e:
    print(f'❌ XFormers error: {e}')

try:
    import flash_attn
    print(f'✅ Flash Attention: Available')
    from flash_attn.flash_attn_interface import flash_attn_func
    print('✅ Flash Attention func: Available')
except Exception as e:
    print(f'❌ Flash Attention error: {e}')

try:
    import triton
    print(f'✅ Triton: {triton.__version__}')
except Exception as e:
    print(f'❌ Triton error: {e}')
"

echo ""
echo "🎉 COMPATIBILITY FIX COMPLETE!"
echo "=============================="
echo "✅ XFormers reinstalled for PyTorch 2.5.1"
echo "✅ Flash Attention reinstalled for Python 3.10"
echo "✅ Triton installed for GPU acceleration"
echo ""
echo "🚀 Now run: ./start_gradio_xformers.sh" 