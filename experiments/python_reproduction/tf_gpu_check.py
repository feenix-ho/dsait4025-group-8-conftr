import tensorflow as tf
import os

# Print environment information
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Print built config info
print("\nBuild configuration:")
for key, value in tf.sysconfig.get_build_info().items():
    print(f"  {key}: {value}")

# List available physical devices
print("\nAvailable physical devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Check if TensorFlow can see any GPUs
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"\nTensorFlow can see {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  {gpu}")

    # Try to configure memory growth
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\nSuccessfully configured memory growth for all GPUs")
    except RuntimeError as e:
        print(f"\nError configuring memory growth: {e}")
else:
    print("\nNo GPUs visible to TensorFlow")
    print("\nPossible reasons:")
    print("1. CUDA/cuDNN not installed or incompatible with this TensorFlow version")
    print("2. GPU is being used by another process")
    print("3. TensorFlow 2.4.1 requires CUDA 11.0 and cuDNN 8.0")
    print("4. Environment variables not properly configured")

# Try a simple operation
print("\nTrying a simple TensorFlow operation...")
try:
    with tf.device("/GPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Result: {c}")
        print("Operation completed successfully on GPU")
except RuntimeError as e:
    print(f"Error running operation on GPU: {e}")
    print("Trying on CPU instead...")
    with tf.device("/CPU:0"):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Result: {c}")
        print("Operation completed successfully on CPU")
