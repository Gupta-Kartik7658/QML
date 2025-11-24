#!/usr/bin/env python3
"""
GPU Monitoring Script for Quantum Feature Selection
Run this in a separate terminal to monitor GPU usage during execution
"""

import subprocess
import time
import sys

def monitor_gpu(interval=2):
    """Monitor GPU usage continuously"""
    print("=" * 80)
    print("GPU MONITORING - Press Ctrl+C to stop")
    print("=" * 80)
    print("\nMonitoring GPU usage every {} seconds...\n".format(interval))
    
    try:
        while True:
            # Clear previous output
            print("\n" + "=" * 80)
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Run nvidia-smi
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 7:
                        gpu_id, name, temp, gpu_util, mem_util, mem_used, mem_total = parts[:7]
                        
                        print(f"\nGPU {gpu_id}: {name}")
                        print(f"  Temperature: {temp}°C")
                        print(f"  GPU Utilization: {gpu_util}%")
                        print(f"  Memory Utilization: {mem_util}%")
                        print(f"  Memory Used: {mem_used} MB / {mem_total} MB")
                        
                        # Visual bar for utilization
                        gpu_bar = '█' * (int(gpu_util) // 5) + '░' * (20 - int(gpu_util) // 5)
                        mem_bar = '█' * (int(mem_util) // 5) + '░' * (20 - int(mem_util) // 5)
                        print(f"  GPU: [{gpu_bar}] {gpu_util}%")
                        print(f"  MEM: [{mem_bar}] {mem_util}%")
                
                # Also show process information
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                if result.stdout.strip():
                    print("\n  Active GPU Processes:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"    {line}")
                else:
                    print("\n  No active GPU processes")
                
            except subprocess.CalledProcessError as e:
                print(f"Error running nvidia-smi: {e}")
            except Exception as e:
                print(f"Error: {e}")
            
            # Wait before next update
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor GPU usage')
    parser.add_argument('--interval', type=float, default=2,
                       help='Monitoring interval in seconds (default: 2)')
    
    args = parser.parse_args()
    monitor_gpu(args.interval)
