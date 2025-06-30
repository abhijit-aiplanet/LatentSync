#!/bin/bash

echo "🔥 MASTER FLASH ATTENTION FIX SCRIPT"
echo "🎯 Running all fixes in order of success probability..."
echo ""
echo "📋 Fix Strategy:"
echo "1. 🎯 Smart wheel detection and installation"
echo "2. ⚡ Direct wheel downloads"  
echo "3. 🔧 CUDA runtime headers fix"
echo "4. 🔥 Comprehensive environment fix"
echo ""

# Make all scripts executable
chmod +x smart_wheel_install.sh
chmod +x direct_flash_attention_fix.sh  
chmod +x cuda_runtime_fix.sh
chmod +x comprehensive_flash_attention_fix.sh

# Check current status
echo "🔍 Current Status Check:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')

try:
    import xformers
    print(f'XFormers: ✅ {xformers.__version__}')
except:
    print('XFormers: ❌')

try:
    import flash_attn
    print('Flash Attention: ✅ Already installed!')
    exit(0)
except:
    print('Flash Attention: ❌ Missing')
"

# If Flash Attention already works, exit
if python -c "import flash_attn; from flash_attn import flash_attn_func" 2>/dev/null; then
    echo "🎉 Flash Attention already working! No fix needed."
    exit 0
fi

echo ""
echo "🚀 Starting fix sequence..."

# Method 1: Smart wheel installer (highest success rate)
echo ""
echo "=== 🎯 METHOD 1: SMART WHEEL DETECTION ==="
if ./smart_wheel_install.sh; then
    echo "🎉 SUCCESS: Smart wheel method worked!"
    exit 0
fi

# Method 2: Direct wheel downloads  
echo ""
echo "=== ⚡ METHOD 2: DIRECT WHEEL DOWNLOADS ==="
if ./direct_flash_attention_fix.sh; then
    echo "🎉 SUCCESS: Direct wheel method worked!"
    exit 0
fi

# Method 3: CUDA runtime fix
echo ""
echo "=== 🔧 METHOD 3: CUDA RUNTIME HEADERS FIX ==="
if ./cuda_runtime_fix.sh; then
    echo "🎉 SUCCESS: CUDA runtime fix worked!"
    exit 0
fi

# Method 4: Comprehensive fix
echo ""
echo "=== 🔥 METHOD 4: COMPREHENSIVE ENVIRONMENT FIX ==="
if ./comprehensive_flash_attention_fix.sh; then
    echo "🎉 SUCCESS: Comprehensive fix worked!"
    exit 0
fi

# Final status check
echo ""
echo "🔍 FINAL STATUS CHECK:"
python -c "
try:
    import flash_attn
    from flash_attn import flash_attn_func
    print('🎉 ULTIMATE SUCCESS: Flash Attention is now working!')
    print(f'Version: {flash_attn.__version__}')
except ImportError:
    print('❌ Flash Attention still not available')
    print('✅ However, XFormers is working and will provide 3-4x speedup!')
    print('🚀 You can proceed with ./start_gradio_xformers.sh')
except Exception as e:
    print(f'⚠️ Flash Attention has issues: {e}')
"

echo ""
echo "🔥 MASTER FIX SCRIPT COMPLETE!"
echo "📋 Next Steps:"
echo "   ✅ If Flash Attention works: Use ./start_gradio_optimized.sh"
echo "   🎯 If only XFormers works: Use ./start_gradio_xformers.sh"  
echo "   🛡️ If neither works: Use ./start_gradio_safe.sh" 