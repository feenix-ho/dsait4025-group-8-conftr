import jax
import jax.numpy as jnp
import os
import sys

# Add TensorFlow import
try:
    import tensorflow as tf

    # Configure TensorFlow to use memory growth and only allocate memory as needed
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Memory growth enabled for {device}")
            except Exception as e:
                print(f"Failed to set memory growth: {e}")
    tensorflow_available = True
except ImportError:
    tensorflow_available = False

# Print environment details
print(f"Python version: {sys.version}")
print(f"JAX version: {jax.__version__}")

# Check CUDA environment
print("CUDA environment variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

# Try to get CUDA version
try:
    if tensorflow_available:
        print(f"CUDA version for TF: {tf.sysconfig.get_build_info()['cuda_version']}")
        print(f"cuDNN version for TF: {tf.sysconfig.get_build_info()['cudnn_version']}")
except Exception as e:
    print(f"Could not get CUDA/cuDNN version from TensorFlow: {e}")

# Check if jax was built with CUDA support
try:
    # Check JAX's device count to indicate CUDA support
    gpu_count = jax.device_count("gpu")
    print(
        f"JAX GPU device count: {gpu_count} (CUDA is {'supported' if gpu_count > 0 else 'not supported'})"
    )
except Exception as e:
    print(f"Could not check JAX CUDA support: {e}")

# Check and print available devices
print("\nDevice Information:")
devices = jax.devices()
print(f"Available JAX devices: {devices}")

# Check if TensorFlow can see GPUs
if tensorflow_available:
    print("\nTensorFlow GPU Information:")
    print(f"TensorFlow version: {tf.__version__}")
    tf_gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow physical GPUs: {tf_gpus}")
    if tf_gpus:
        print(f"Found {len(tf_gpus)} TensorFlow GPU device(s)")
        # Print more details about each GPU
        for i, gpu in enumerate(tf_gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                print(
                    f"  GPU {i}: {details.get('device_name', 'Unknown')} - {gpu.name}"
                )
            except Exception as e:
                print(f"  GPU {i}: {gpu.name} (Could not get details: {e})")
    else:
        print("No TensorFlow GPU devices found")
else:
    print("\nTensorFlow is not available in this environment")

# Check if GPU is available for JAX
gpu_devices = [d for d in devices if d.platform == "gpu"]
if gpu_devices:
    print(f"\nJAX GPU is available! Found {len(gpu_devices)} GPU device(s).")

    # Run a simple computation on GPU
    print("\nRunning test computation on GPU...")
    x = jnp.ones((1000, 1000))
    result = jnp.dot(x, x).block_until_ready()
    print(f"Test computation completed successfully on {jax.default_device()}")
else:
    print("\nNo JAX GPU devices found. JAX is using CPU only.")
    print("\nPossible reasons:")
    print("1. CUDA is not properly installed or not compatible with this JAX version")
    print(
        "2. Environment variables like CUDA_VISIBLE_DEVICES might be restricting GPU access"
    )
    print("3. The JAX installation doesn't include GPU support")
