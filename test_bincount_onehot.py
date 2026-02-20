"""
This script demonstrates how to optimize jnp.bincount.

jnp.bincount is slow because it lowers to the scatter-add. And scatter operation is known to be slow on TPU.

one-hot convert the scatter operation into dense operation.
The one-hot matrix will be jax.nn.one_hot(topk_indices_flat, global_number_experts, dtype=jnp.int32): [num_tokens*topk, global_number_experts]. It has a size of num_tokens*topk*global_number_experts*4.
If num_tokens=8192, topk=8, global_number_experts=160, then the one-hot matrix will be 8192*8*160*4/1024/1024=40MB
"""

import jax
import jax.numpy as jnp
import numpy as np

T = 8192
k = 8
E = 160

np.random.seed(42)

topk_indices = jnp.array(
    np.random.randint(0, E, size=(T, k), dtype=np.int32)
)
topk_indices_flat = topk_indices.flatten()  # [T*k]

# Original: bincount
@jax.jit(static_argnames=("E",))
def bincount(topk_indices_flat, E):
  return jnp.bincount(topk_indices_flat, length=E)

# Proposed: one_hot + sum
@jax.jit(static_argnames=("E",))
def one_hot_sum(topk_indices_flat, E):
  return jax.nn.one_hot(topk_indices_flat, E, dtype=jnp.int32).sum(axis=0)

group_sizes_bincount = bincount(topk_indices_flat, E)
group_sizes_onehot = one_hot_sum(topk_indices_flat, E)
match = jnp.array_equal(group_sizes_bincount, group_sizes_onehot)
print(f"Exact match: {match}")

profile_path = "/tmp/my_tpu_profile"
jax.profiler.start_trace(profile_path)
for _ in range(2):
  bincount(topk_indices_flat, E).block_until_ready()

for _ in range(2):
  one_hot_sum(topk_indices_flat, E).block_until_ready()
jax.profiler.stop_trace()
