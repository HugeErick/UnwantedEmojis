# check_tf_build.py
import tensorflow as tf
from tensorflow.python.platform import build_info

print("=" * 60)
print("TENSORFLOW BUILD INFORMATION")
print("=" * 60)

print("Build info:", build_info.build_info)

# Check what compute capabilities are supported
print("\nSupported compute capabilities:")
try:
    from tensorflow.python.framework import test_util
    print("Available:", test_util.get_gpu_compilation_usage())
except:
    print("Could not get compute capabilities")

print("\nCUDA and GPU support:")
print(f"is_built_with_cuda: {tf.test.is_built_with_cuda()}")
print(f"is_built_with_gpu_support: {tf.test.is_built_with_gpu_support()}")
print(f"is_built_with_rocm: {tf.test.is_built_with_rocm()}")
print(f"is_built_with_xla: {tf.test.is_built_with_xla()}")

print("\nAvailable devices:")
print(tf.config.list_physical_devices())