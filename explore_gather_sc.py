"""Run this and get the HLO and LLO for the gather operation.
"""

import jax
import jax.numpy as jnp

@jax.jit
def gather_direct(indices, x):
    """Gather using direct indexing."""
    return x[indices]

def main():
    num_tokens = 8192
    hidden_size = 6144
    num_indices = 65536

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
    indices = jax.random.randint(k2, (num_indices,), 0, num_tokens)

    gather_direct(indices, hidden_states).block_until_ready()
    
    # profile_path = "/tmp/sort2_tokens_profile"
    # jax.profiler.start_trace(profile_path)
    for _ in range(3):
      gather_direct(indices, hidden_states).block_until_ready()
    # jax.profiler.stop_trace()
        
if __name__ == "__main__":
    main()
