# final_gpu_check.py
import os
import sys

# Set environment variables BEFORE importing tensorflow
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add CUDA libraries to path if needed
cuda_paths = [
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cuda_cupti', 'bin'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cudnn', 'bin'),
    os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', 'cublas', 'bin'),
]

for path in cuda_paths:
    if os.path.exists(path):
        os.environ['PATH'] = path + ';' + os.environ['PATH']

print("=" * 70)
print("COMPREHENSIVE GPU DETECTION TEST")
print("=" * 70)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check build info
    try:
        from tensorflow.python.platform import build_info
        print(f"Build info: {build_info.build_info}")
    except:
        print("Could not get build info")
    
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Built with GPU support: {tf.test.is_built_with_gpu_support()}")
    
    # List all physical devices
    print(f"\nAll physical devices:")
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f"  - {device}")
    
    # Specifically check GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs detected: {len(gpus)}")
    
    if gpus:
        print("\nSUCCESS! GPU DETECTED!")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    print(f"    Compute Capability: {details.get('compute_capability', 'N/A')}")
                    print(f"    Device Name: {details.get('device_name', 'N/A')}")
            except Exception as e:
                print(f"    Could not get details: {e}")
        
        # Test actual GPU computation
        print("\nTesting GPU computation...")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("GPU computation successful!")
                print(f"Result:\n{c.numpy()}")
                
                # Test larger computation to verify it's actually using GPU
                print("\nTesting larger computation...")
                large_a = tf.random.normal([1000, 1000])
                large_b = tf.random.normal([1000, 1000])
                large_c = tf.matmul(large_a, large_b)
                print("Large GPU computation successful!")
                
        except Exception as e:
            print(f"GPU computation failed: {e}")
            
    else:
        print("\nNo GPUs detected")
        print("\nEnvironment variables:")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"  PATH contains CUDA: {'nvidia' in os.environ.get('PATH', '')}")
        
        # Check if CUDA libraries are accessible
        print("\nChecking CUDA library accessibility:")
        try:
            import nvidia.cudnn
            print("nvidia.cudnn is accessible")
        except ImportError as e:
            print(f"nvidia.cudnn not accessible: {e}")
            
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)