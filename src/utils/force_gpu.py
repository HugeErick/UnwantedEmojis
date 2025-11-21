#!/usr/bin/env python3
"""
GPU-Enabled Training Wrapper
This script forces GPU detection and provides detailed diagnostics
"""

import os
import sys

# CRITICAL: Remove the environment variable that's hiding the GPU
if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
    print("⚠️  WARNING: CUDA_VISIBLE_DEVICES was set to -1 (hiding GPU)")
    print("⚠️  Removing it to enable GPU detection...")
    del os.environ['CUDA_VISIBLE_DEVICES']
    print("✓ CUDA_VISIBLE_DEVICES removed\n")

# Set it to use GPU 0 explicitly
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_gpu_setup():
    """Check GPU detection and provide diagnostics"""
    print("=" * 70)
    print("GPU DETECTION CHECK")
    print("=" * 70)
    
    try:
        import tensorflow as tf
        print(f"\n✓ TensorFlow version: {tf.__version__}")
        
        # Check for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"\nGPUs detected: {len(gpus)}")
        
        if gpus:
            print("\n✓ SUCCESS! GPU(s) available:")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                try:
                    # Get memory info
                    details = tf.config.experimental.get_device_details(gpu)
                    if details:
                        print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
                except:  
                    pass
            
            # Configure GPU memory growth to avoid OOM errors
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("\n✓ GPU memory growth enabled (prevents OOM errors)")
            except RuntimeError as e:
                print(f"\n⚠️  Note: {e}")
            
            return True
        else:
            print("\n❌ NO GPUs DETECTED")
            print("\nDebugging information:")
            
            # Check CUDA availability
            print(f"  CUDA available: {tf.test.is_built_with_cuda()}")
            print(f"  GPU support: {tf.test.is_built_with_gpu_support()}")
            
            # Check environment variables
            print("\nEnvironment variables:")
            print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'not set')}")
            
            return False
            
    except ImportError as e:
        print(f"\n❌ Error importing TensorFlow: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 70)
    print("TRAINING WITH GPU (if available)")
    print("=" * 70 + "\n")
    
    # Check GPU
    gpu_available = check_gpu_setup()
    
    if not gpu_available:
        print("\n" + "=" * 70)
        print("⚠️  WARNING: No GPU detected, training will use CPU")
        print("=" * 70)
        response = input("\nContinue with CPU? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("✓ GPU ENABLED - Training will be much faster!")
        print("=" * 70)
    
    input("\nPress Enter to start training...")
    
    # Import and run the training script
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    try:
        # Import the training module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Try to import the training script
        try:
            import trained_emotions
            trained_emotions.main()
        except ImportError:
            # If import fails, try running as script
            import subprocess
            script_path = os.path.join('src', 'trained_emotions.py')
            if os.path.exists(script_path):
                result = subprocess.run([sys.executable, script_path])
                sys.exit(result.returncode)
            else:
                print(f"\n❌ Error: Could not find training script at {script_path}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()