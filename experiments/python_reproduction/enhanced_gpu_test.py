import jax
import jax.numpy as jnp
import os
import sys

# Print environment details
print(f"Python version: {sys.version}")
print(f"JAX version: {jax.__version__}")
print("CUDA environment variables:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"  XLA_FLAGS: {os.environ.get('XLA_FLAGS', 'Not set')}")

# Check and print available devices
print("\nDevice Information:")
devices = jax.devices()
print(f"Available devices: {devices}")

# Check if GPU is available
gpu_devices = [d for d in devices if d.platform == "gpu"]
if gpu_devices:
    print(f"\nGPU is available! Found {len(gpu_devices)} GPU device(s).")

    # Run a simple computation on GPU
    print("\nRunning test computation on GPU...")
    x = jnp.ones((1000, 1000))
    result = jnp.dot(x, x).block_until_ready()
    print(f"Test computation completed successfully on {jax.default_device()}")
else:
    print("\nNo GPU devices found. JAX is using CPU only.")
    print("\nPossible reasons:")
    print("1. CUDA is not properly installed or not compatible with this JAX version")
    print(
        "2. Environment variables like CUDA_VISIBLE_DEVICES might be restricting GPU access"
    )
    print("3. The JAX installation doesn't include GPU support")
