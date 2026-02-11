import dataclasses
import functools
from typing import Callable, Tuple
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def kernel(
    # Prefetch
    topk_indices_ref: jax.Array,
    # In
    x_ref: jax.Array,
    # Out
    x_sorted_ref: jax.Array,
    group_sizes_ref: jax.Array,
    topk_argsort_revert_indices_ref: jax.Array,
    # Scratch
    group_offset_ref: jax.Array,
):
  num_tokens, topk = topk_indices_ref.shape
  num_experts = group_sizes_ref.shape[0]

  for i in range(num_experts):
    group_sizes_ref[i] = 0

  for t in range(num_tokens):
    for k in range(topk):
      expert_idx = topk_indices_ref[t, k]
      group_size = group_sizes_ref[expert_idx]
      group_sizes_ref[expert_idx] = group_size + 1

  curr_offset = 0
  for e in range(num_experts):
    next_offset = curr_offset + group_sizes_ref[e]
    group_offset_ref[e] = next_offset
    curr_offset = next_offset

  for t in range(num_tokens):
    for k in range(topk):
      expert_idx = topk_indices_ref[t, k]
      group_offset = group_offset_ref[expert_idx]
      topk_argsort_revert_indices_ref[group_offset] = t
      group_offset_ref[expert_idx] = group_offset + 1

  num_sublanes = 16
  for start in range(0, x_sorted_ref.shape[0], num_sublanes):
    end = start + num_sublanes

    vals = []
    for i in range(start, end):
      vals.append(x_ref[topk_argsort_revert_indices_ref[i], :])
    vals_orig = jnp.concat(vals, axis=0).astype(x_sorted_ref.dtype)
    x_sorted_ref[start:end, :] = vals_orig


@jax.jit(static_argnames=["num_experts"])
def sort_tokens(
    x: jax.Array, topk_indices: jax.Array, num_experts: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.8)
  num_tokens, hidden_size = x.shape
  topk = topk_indices.shape[1]
  out_shape = (num_tokens * topk, hidden_size)

  orig_dtype = x.dtype
  x = x.astype(jnp.float32)
  x = x.reshape(num_tokens, 1, hidden_size)

  scope_name = f"sort_tokens-m_{num_tokens}-k_{hidden_size}-topk_{topk}"
  return pl.pallas_call(
      kernel,
      out_shape=[
          jax.ShapeDtypeStruct(out_shape, orig_dtype),
          jax.ShapeDtypeStruct((num_experts,), jnp.int32),
          jax.ShapeDtypeStruct((topk_indices.size,), jnp.int32),
      ],
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
          out_specs=[
              pl.BlockSpec(memory_space=pltpu.VMEM),
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(memory_space=pltpu.SMEM),
          ],
          scratch_shapes=[
              pltpu.SMEM((num_experts,), jnp.int32),
          ],
      ),
      compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
      name=scope_name,
  )(topk_indices, x)
