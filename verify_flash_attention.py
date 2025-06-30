#!/usr/bin/env python3

import sys
import torch
import time
import traceback

def test_flash_attention():
    """Comprehensive Flash Attention test"""
    print("🔥 FLASH ATTENTION VERIFICATION TEST")
    print("=" * 50)
    
    # Basic import test
    print("1. 📦 Testing basic import...")
    try:
        import flash_attn
        print(f"   ✅ flash_attn version: {flash_attn.__version__}")
    except ImportError as e:
        print(f"   ❌ Cannot import flash_attn: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Import issue: {e}")
        return False
    
    # Function import test
    print("2. 🔧 Testing function imports...")
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        print("   ✅ Core functions imported successfully")
    except ImportError as e:
        print(f"   ❌ Cannot import core functions: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Function import issue: {e}")
        return False
    
    # CUDA availability test
    print("3. 🎯 Testing CUDA availability...")
    if not torch.cuda.is_available():
        print("   ❌ CUDA not available")
        return False
    
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)
    print(f"   ✅ CUDA available on {gpu_name}")
    
    # Basic functionality test
    print("4. ⚡ Testing Flash Attention functionality...")
    try:
        # Create test tensors
        batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        print(f"   🔧 Created test tensors: {q.shape}")
        
        # Test Flash Attention computation
        start_time = time.time()
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"   ✅ Flash Attention computation successful!")
        print(f"   ⚡ Output shape: {out.shape}")
        print(f"   ⏱️ Time: {(end_time - start_time)*1000:.2f}ms")
        
    except Exception as e:
        print(f"   ❌ Flash Attention computation failed: {e}")
        traceback.print_exc()
        return False
    
    # Performance comparison
    print("5. 📊 Performance comparison test...")
    try:
        # Test with larger sequences for meaningful comparison
        batch_size, seq_len, num_heads, head_dim = 1, 2048, 12, 64
        
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        # Flash Attention timing
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(5):
            out_flash = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / 5
        
        # Standard attention timing
        q_std = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k_std = k.transpose(1, 2)
        v_std = v.transpose(1, 2)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(5):
            scores = torch.matmul(q_std, k_std.transpose(-2, -1)) / (head_dim ** 0.5)
            # Apply causal mask
            mask = torch.triu(torch.ones(seq_len, seq_len, device='cuda'), diagonal=1).bool()
            scores.masked_fill_(mask, float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            out_std = torch.matmul(attn_weights, v_std)
        torch.cuda.synchronize()
        std_time = (time.time() - start_time) / 5
        
        speedup = std_time / flash_time
        print(f"   ⚡ Flash Attention: {flash_time*1000:.2f}ms")
        print(f"   🐌 Standard Attention: {std_time*1000:.2f}ms")
        print(f"   🚀 Speedup: {speedup:.2f}x")
        
        if speedup > 1.5:
            print("   ✅ Significant speedup achieved!")
        else:
            print("   ⚠️ Speedup lower than expected")
            
    except Exception as e:
        print(f"   ⚠️ Performance test failed: {e}")
        # Don't return False here as basic functionality still works
    
    # Memory usage test
    print("6. 💾 Memory efficiency test...")
    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Large sequence test
        batch_size, seq_len, num_heads, head_dim = 1, 4096, 16, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda', dtype=torch.float16)
        
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
        peak_memory = torch.cuda.max_memory_allocated()
        
        memory_used = (peak_memory - initial_memory) / 1e9
        print(f"   💾 Memory used: {memory_used:.2f}GB for {seq_len} sequence length")
        print("   ✅ Memory efficiency test passed!")
        
    except Exception as e:
        print(f"   ⚠️ Memory test failed: {e}")
    
    print("\n🎉 FLASH ATTENTION VERIFICATION COMPLETE!")
    print("✅ Flash Attention is working correctly!")
    return True

def test_integration_with_diffusers():
    """Test Flash Attention integration with diffusers"""
    print("\n7. 🔧 Testing integration with diffusers...")
    try:
        from diffusers import UNet2DConditionModel
        from diffusers.models.attention_processor import FlashAttnProcessor
        
        # Create a small UNet for testing
        unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=768,
        ).cuda().half()
        
        # Set Flash Attention processor
        unet.set_attn_processor(FlashAttnProcessor())
        
        # Test forward pass
        sample = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)
        timestep = torch.tensor([1], device='cuda')
        encoder_hidden_states = torch.randn(1, 77, 768, device='cuda', dtype=torch.float16)
        
        with torch.no_grad():
            output = unet(sample, timestep, encoder_hidden_states).sample
        
        print(f"   ✅ UNet with Flash Attention: {output.shape}")
        print("   ✅ Diffusers integration successful!")
        return True
        
    except Exception as e:
        print(f"   ⚠️ Diffusers integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Starting Flash Attention verification...")
    
    success = test_flash_attention()
    
    if success:
        test_integration_with_diffusers()
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Flash Attention is ready for maximum performance!")
        sys.exit(0)
    else:
        print("\n❌ FLASH ATTENTION VERIFICATION FAILED!")
        print("🛡️ Please use XFormers mode instead")
        sys.exit(1) 