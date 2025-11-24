#!/usr/bin/env python3
"""
Qiskit GPU Acceleration Test Script
Tests if Qiskit can utilize GPU for quantum circuit simulations
"""

import sys
import time
import numpy as np

def check_qiskit_installation():
    """Check if Qiskit is properly installed"""
    print("=" * 70)
    print("Checking Qiskit Installation")
    print("=" * 70)
    try:
        import qiskit
        print(f"✓ Qiskit version: {qiskit.__version__}")
        
        # Check for Qiskit Aer
        try:
            from qiskit_aer import AerSimulator
            print(f"✓ Qiskit Aer is installed")
            return True
        except ImportError:
            print("✗ Qiskit Aer is not installed")
            print("  Install with: pip install qiskit-aer-gpu")
            return False
            
    except ImportError:
        print("✗ Qiskit is not installed")
        print("  Install with: pip install qiskit==1.3.1")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_gpu_backend_availability():
    """Check if GPU backend is available in Qiskit Aer"""
    print("\n" + "=" * 70)
    print("Checking GPU Backend Availability")
    print("=" * 70)
    try:
        from qiskit_aer import AerSimulator
        
        # Check available devices
        simulator = AerSimulator()
        available_devices = simulator.available_devices()
        
        print(f"Available devices: {available_devices}")
        
        if 'GPU' in available_devices:
            print("✓ GPU backend is available!")
            return True
        else:
            print("✗ GPU backend is NOT available")
            print("  Available backends:", available_devices)
            print("\n  Troubleshooting:")
            print("  1. Make sure CUDA is properly installed")
            print("  2. Install qiskit-aer-gpu: pip install qiskit-aer-gpu")
            print("  3. Check CuPy installation: pip install cupy-cuda12x")
            return False
            
    except Exception as e:
        print(f"✗ Error checking GPU backend: {e}")
        return False

def benchmark_cpu_vs_gpu():
    """Benchmark quantum circuit simulation on CPU vs GPU"""
    print("\n" + "=" * 70)
    print("Benchmarking CPU vs GPU Performance")
    print("=" * 70)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        # Create a quantum circuit with multiple qubits
        num_qubits = 20
        depth = 10
        
        print(f"\nCreating quantum circuit:")
        print(f"  Qubits: {num_qubits}")
        print(f"  Depth: {depth}")
        print(f"  Shots: 1000")
        
        # Build a random quantum circuit
        qc = QuantumCircuit(num_qubits)
        
        for layer in range(depth):
            # Add Hadamard gates
            for qubit in range(num_qubits):
                qc.h(qubit)
            
            # Add CNOT gates
            for qubit in range(0, num_qubits - 1, 2):
                qc.cx(qubit, qubit + 1)
            
            # Add rotation gates
            for qubit in range(num_qubits):
                qc.rz(np.pi / 4, qubit)
        
        # Add measurements
        qc.measure_all()
        
        print(f"\nCircuit created with {qc.size()} gates")
        
        # CPU Simulation
        print("\n" + "-" * 70)
        print("Running CPU Simulation...")
        print("-" * 70)
        
        cpu_simulator = AerSimulator(method='statevector', device='CPU')
        
        start_time = time.time()
        cpu_job = cpu_simulator.run(qc, shots=1000)
        cpu_result = cpu_job.result()
        cpu_time = time.time() - start_time
        
        print(f"CPU Time: {cpu_time:.4f} seconds")
        cpu_counts = cpu_result.get_counts()
        print(f"Result states: {len(cpu_counts)}")
        
        # GPU Simulation
        print("\n" + "-" * 70)
        print("Running GPU Simulation...")
        print("-" * 70)
        
        try:
            gpu_simulator = AerSimulator(method='statevector', device='GPU')
            
            start_time = time.time()
            gpu_job = gpu_simulator.run(qc, shots=1000)
            gpu_result = gpu_job.result()
            gpu_time = time.time() - start_time
            
            print(f"GPU Time: {gpu_time:.4f} seconds")
            gpu_counts = gpu_result.get_counts()
            print(f"Result states: {len(gpu_counts)}")
            
            # Calculate speedup
            speedup = cpu_time / gpu_time
            print("\n" + "=" * 70)
            print(f"SPEEDUP: {speedup:.2f}x")
            print("=" * 70)
            
            if speedup > 1.2:
                print("✓ GPU acceleration is working effectively!")
            elif speedup > 0.8:
                print("⚠ GPU speedup is marginal (might be expected for smaller circuits)")
            else:
                print("⚠ GPU is slower than CPU (might be overhead for this circuit size)")
            
            return True
            
        except Exception as e:
            print(f"✗ GPU simulation failed: {e}")
            print("\nNote: GPU simulation requires:")
            print("  - qiskit-aer-gpu package")
            print("  - CUDA-capable GPU")
            print("  - Proper CUDA installation")
            return False
            
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_specific_features():
    """Test GPU-specific features in Qiskit"""
    print("\n" + "=" * 70)
    print("Testing GPU-Specific Features")
    print("=" * 70)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        # Test tensor network method (GPU-accelerated)
        print("\nTesting tensor network method on GPU...")
        
        qc = QuantumCircuit(15)
        for i in range(15):
            qc.h(i)
        for i in range(14):
            qc.cx(i, i + 1)
        qc.measure_all()
        
        try:
            simulator = AerSimulator(method='tensor_network', device='GPU')
            job = simulator.run(qc, shots=100)
            result = job.result()
            print("✓ Tensor network method works on GPU")
            print(f"  Execution time: {result.time_taken:.4f} seconds")
            return True
        except Exception as e:
            print(f"⚠ Tensor network on GPU failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ GPU-specific features test failed: {e}")
        return False

def run_practical_example():
    """Run a practical quantum machine learning example"""
    print("\n" + "=" * 70)
    print("Running Practical Example: Quantum State Preparation")
    print("=" * 70)
    
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        
        # Create a quantum circuit for state preparation
        qc = QuantumCircuit(12)
        
        # Create superposition
        for i in range(12):
            qc.h(i)
        
        # Add entanglement
        for i in range(11):
            qc.cx(i, i + 1)
        
        # Add parametrized rotations (common in QML)
        angles = np.random.rand(12) * np.pi
        for i, angle in enumerate(angles):
            qc.ry(angle, i)
            qc.rz(angle, i)
        
        qc.measure_all()
        
        print(f"\nCircuit details:")
        print(f"  Qubits: {qc.num_qubits}")
        print(f"  Gates: {qc.size()}")
        print(f"  Depth: {qc.depth()}")
        
        # Simulate on GPU
        simulator = AerSimulator(device='GPU')
        
        print("\nExecuting on GPU...")
        start = time.time()
        job = simulator.run(qc, shots=2000)
        result = job.result()
        elapsed = time.time() - start
        
        counts = result.get_counts()
        print(f"\n✓ Simulation completed successfully!")
        print(f"  Execution time: {elapsed:.4f} seconds")
        print(f"  Unique states measured: {len(counts)}")
        print(f"  Top 3 states:")
        
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_counts[:3]:
            print(f"    |{state}⟩: {count} times ({count/2000*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Practical example failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Qiskit GPU tests"""
    print("\n" + "=" * 70)
    print("QISKIT GPU ACCELERATION TEST SUITE")
    print("=" * 70 + "\n")
    
    results = {}
    
    # Test 1: Installation check
    results['installation'] = check_qiskit_installation()
    if not results['installation']:
        print("\n⚠ Please install Qiskit before proceeding.")
        return 1
    
    # Test 2: GPU backend availability
    results['gpu_backend'] = check_gpu_backend_availability()
    
    # Test 3: CPU vs GPU benchmark
    if results['gpu_backend']:
        results['benchmark'] = benchmark_cpu_vs_gpu()
        results['gpu_features'] = test_gpu_specific_features()
        results['practical'] = run_practical_example()
    else:
        print("\n⚠ Skipping GPU tests - GPU backend not available")
        results['benchmark'] = False
        results['gpu_features'] = False
        results['practical'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    gpu_tests_passed = results.get('gpu_backend', False) and \
                       results.get('benchmark', False)
    
    print("\n" + "=" * 70)
    if gpu_tests_passed:
        print("✓ Qiskit GPU acceleration is working correctly!")
        print("  You can now train quantum models on GPU.")
    else:
        print("⚠ GPU acceleration is not fully functional.")
        print("  Check the output above for specific issues.")
    print("=" * 70 + "\n")
    
    return 0 if gpu_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())
