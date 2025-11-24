#!/usr/bin/env python3
"""
CUDA and GPU Test Script for WSL
Tests CUDA availability and GPU computational capabilities
"""

import subprocess
import sys

def check_nvidia_smi():
    """Check if nvidia-smi is accessible"""
    print("=" * 60)
    print("Testing nvidia-smi access...")
    print("=" * 60)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(result.stdout)
        print("✓ nvidia-smi is accessible\n")
        return True
    except Exception as e:
        print(f"✗ nvidia-smi failed: {e}\n")
        return False

def check_cuda_available():
    """Check if CUDA is available through PyTorch"""
    print("=" * 60)
    print("Testing CUDA availability through PyTorch...")
    print("=" * 60)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available!")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
                
                # Get memory info
                mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"    Total Memory: {mem_total:.2f} GB")
                print(f"    Allocated Memory: {mem_allocated:.2f} GB")
                print(f"    Reserved Memory: {mem_reserved:.2f} GB")
            
            print()
            return True
        else:
            print("✗ CUDA is not available through PyTorch")
            print("  This might mean:")
            print("  - PyTorch CPU-only version is installed")
            print("  - CUDA drivers are not properly configured")
            print()
            return False
            
    except ImportError:
        print("✗ PyTorch is not installed")
        print("  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print()
        return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}\n")
        return False

def run_gpu_computation_test():
    """Run a simple GPU computation test"""
    print("=" * 60)
    print("Running GPU computation test...")
    print("=" * 60)
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            print("✗ Skipping computation test - CUDA not available\n")
            return False
        
        # Create large tensors
        size = 5000
        print(f"Creating {size}x{size} matrices...")
        
        # CPU computation
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start
        
        # GPU computation
        device = torch.device('cuda:0')
        a_gpu = torch.randn(size, size, device=device)
        b_gpu = torch.randn(size, size, device=device)
        
        # Warm-up
        torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"\n  CPU Time: {cpu_time:.4f} seconds")
        print(f"  GPU Time: {gpu_time:.4f} seconds")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
        if gpu_time < cpu_time:
            print(f"\n✓ GPU is working correctly and faster than CPU!")
        else:
            print(f"\n⚠ GPU is slower than CPU (might be overhead for small operations)")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ GPU computation test failed: {e}\n")
        return False

def check_cupy():
    """Check if CuPy is available (useful for Qiskit GPU)"""
    print("=" * 60)
    print("Testing CuPy availability...")
    print("=" * 60)
    try:
        import cupy as cp
        print(f"✓ CuPy is installed (version {cp.__version__})")
        
        # Simple CuPy test
        x = cp.array([1, 2, 3])
        y = cp.sum(x)
        print(f"  Simple CuPy test: sum([1,2,3]) = {y}")
        print(f"  CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        print()
        return True
        
    except ImportError:
        print("✗ CuPy is not installed")
        print("  CuPy is needed for Qiskit GPU acceleration")
        print("  Install with: pip install cupy-cuda12x")
        print()
        return False
    except Exception as e:
        print(f"✗ CuPy error: {e}\n")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CUDA and GPU Configuration Test for WSL")
    print("=" * 60 + "\n")
    
    results = {
        'nvidia-smi': check_nvidia_smi(),
        'cuda': check_cuda_available(),
        'gpu_computation': False,
        'cupy': False
    }
    
    if results['cuda']:
        results['gpu_computation'] = run_gpu_computation_test()
    
    results['cupy'] = check_cupy()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! GPU is ready for use.")
    else:
        print("⚠ Some tests failed. Please check the output above.")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
