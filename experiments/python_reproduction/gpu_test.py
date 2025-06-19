import jax

# List JAX devices
print("JAX sees:", jax.devices())

# Run a trivial computation
import jax.numpy as jnp

x = jnp.ones((1000, 1000))
y = jnp.dot(x, x).block_until_ready()  # block to ensure it executes now
