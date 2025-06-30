#!/usr/bin/env python3
"""
Test script to verify LatentSync A100 setup on RunPod
"""

import sys
import os
from pathlib import Path
import subprocess

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m"
    }
    print(f"{colors.get(status, colors['INFO'])}[{status}] {message}{colors['RESET']}")

def check_gpu():
    """Check GPU availability and specifications"""
    print_status("Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print_status(f"GPU Count: {gpu_count}", "SUCCESS")
            print_status(f"GPU Name: {gpu_name}", "SUCCESS")
            print_status(f"GPU Memory: {gpu_memory:.1f}GB", "SUCCESS")
            
            if "A100" in gpu_name:
                print_status("A100 GPU detected - Optimal performance expected!", "SUCCESS")
            else:
                print_status(f"Non-A100 GPU detected: {gpu_name}", "WARNING")
                
            return True
        else:
            print_status("No CUDA GPU available!", "ERROR")
            return False
    except ImportError:
        print_status("PyTorch not installed!", "ERROR")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print_status("Checking dependencies...")
    
    required_packages = [
        "torch", "torchvision", "diffusers", "transformers", 
        "gradio", "omegaconf", "librosa", "opencv-python"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print_status(f"✓ {package}", "SUCCESS")
        except ImportError:
            print_status(f"✗ {package}", "ERROR")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_models():
    """Check if model checkpoints are available"""
    print_status("Checking model checkpoints...")
    
    required_files = [
        "checkpoints/latentsync_unet.pt",
        "checkpoints/whisper/tiny.pt"
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
            print_status(f"✓ {file_path} ({file_size:.1f}MB)", "SUCCESS")
        else:
            print_status(f"✗ {file_path}", "ERROR")
            all_present = False
    
    return all_present

def check_gradio_setup():
    """Check if Gradio app files are present"""
    print_status("Checking Gradio setup...")
    
    required_files = [
        "gradio_app_a100.py",
        "start_gradio.sh",
        "configs/unet/stage2_512.yaml"
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"✓ {file_path}", "SUCCESS")
        else:
            print_status(f"✗ {file_path}", "ERROR")
            all_present = False
    
    return all_present

def test_basic_functionality():
    """Test basic PyTorch and CUDA functionality"""
    print_status("Testing basic functionality...")
    
    try:
        import torch
        
        # Test tensor operations
        x = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            x = x.cuda()
            y = torch.matmul(x, x.t())
            print_status("✓ CUDA tensor operations working", "SUCCESS")
        else:
            y = torch.matmul(x, x.t())
            print_status("✓ CPU tensor operations working", "WARNING")
        
        return True
    except Exception as e:
        print_status(f"✗ Basic functionality test failed: {e}", "ERROR")
        return False

def run_system_checks():
    """Run system-level checks"""
    print_status("Running system checks...")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print_status(f"Python version: {python_version}", "INFO")
    
    # Check available disk space
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        print_status("Disk space:", "INFO")
        print(result.stdout)
    except:
        print_status("Could not check disk space", "WARNING")
    
    # Check memory
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print_status("Memory usage:", "INFO")
        print(result.stdout)
    except:
        print_status("Could not check memory", "WARNING")

def main():
    """Main test function"""
    print_status("="*60, "INFO")
    print_status("LatentSync A100 Setup Verification", "INFO")
    print_status("="*60, "INFO")
    
    tests = [
        ("System Checks", run_system_checks),
        ("GPU Detection", check_gpu),
        ("Dependencies", check_dependencies),
        ("Model Checkpoints", check_models),
        ("Gradio Setup", check_gradio_setup),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print_status(f"\n--- {test_name} ---", "INFO")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status(f"Test {test_name} failed with exception: {e}", "ERROR")
            results[test_name] = False
    
    # Summary
    print_status("\n" + "="*60, "INFO")
    print_status("SUMMARY", "INFO")
    print_status("="*60, "INFO")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "SUCCESS" if result else "ERROR"
        print_status(f"{test_name}: {'PASSED' if result else 'FAILED'}", status)
    
    print_status(f"\nOverall: {passed}/{total} tests passed", 
                "SUCCESS" if passed == total else "WARNING")
    
    if passed == total:
        print_status("🎉 All tests passed! LatentSync is ready to use.", "SUCCESS")
        print_status("Run './start_gradio.sh' to start the interface.", "INFO")
    else:
        print_status("❌ Some tests failed. Please check the setup.", "ERROR")
        print_status("Refer to RUNPOD_SETUP_GUIDE.md for troubleshooting.", "INFO")

if __name__ == "__main__":
    main() 