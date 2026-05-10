import jax
import jax.numpy as jnp 
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
# Implement GMM 
# Input: lhs [b, d], rhs [g, d, h]
# Output: [b, h]

# fake inputs 

b = 128
d = 2048 
g = 2
h = 4096
tokens_per_expert = b // g


key = jax.random.PRNGKey(0)
lhs = jax.random.normal(key, (b, d))
group_sizes = jnp.asarray([tokens_per_expert for _ in range(g)])
rhs = jax.random.normal(key, (g, d, h))


expected_out = jnp.einsum("gbd,gdh->gbh", lhs.reshape(g, -1, d), rhs).reshape(b, h)
assert expected_out.shape == (b, h)
print("expected_out shape: ", expected_out.shape)

tile_b = 128 
tile_d = 128 
tile_h = 128 

def kernel(
    lhs_ref,
    rhs_ref,
    group_sizes_ref,
    out_ref,
):
    b_idx = pl.program_id(0)
    h_idx = pl.program_id(1)

    row = b_idx * tile_b + jnp.arange(tile_b)[:, None]  # [tile_b, 1]

    acc = jnp.zeros((tile_b, tile_h), dtype=lhs_ref.dtype)
    for g_idx in range(g):
        group_size = group_sizes_ref[g_idx]
        start_idx = g_idx * tokens_per_expert
        mask = (row >= start_idx) & (row < start_idx + group_size)  # [tile_b, 1]
        res = jnp.zeros((tile_b, tile_h), dtype=lhs_ref.dtype)
        for d_idx in range(0, d, tile_d):
            res += lhs_ref[:, d_idx:d_idx+tile_d] @ rhs_ref[g_idx, d_idx:d_idx+tile_d, :]  # [tile_b, tile_h]
        
        acc = acc + jnp.where(mask, res, 0)
    out_ref[...] = acc


def gmm_native(lhs, rhs, group_sizes):
    # assume lhs is already sorted by experts

    # Each output tile (b_tile, h_tile) is fully computed by one kernel
    # invocation; experts are iterated inside via fori_loop.
    grid = (b // tile_b, h // tile_h)

    in_specs = [
        pl.BlockSpec(block_shape=(tile_b, d), index_map=lambda b_idx, h_idx: (b_idx, 0)),
        pl.BlockSpec(block_shape=(g, d, tile_h), index_map=lambda b_idx, h_idx: (0, 0, h_idx)),
        pl.BlockSpec(memory_space=pltpu.SMEM),
    ]

    out_specs = pl.BlockSpec(block_shape=(tile_b, tile_h), index_map=lambda b_idx, h_idx: (b_idx, h_idx))
    
    return pl.pallas_call(kernel, 
                   grid = grid, 
                   in_specs = in_specs, 
                   out_specs = out_specs, 
                   out_shape = jax.ShapeDtypeStruct((b, h), lhs.dtype), 
                   )(lhs, rhs, group_sizes) 
    

gmm_out = gmm_native(lhs, rhs, group_sizes)

print(gmm_out)
print(expected_out.ravel()[:10])

assert jnp.allclose(gmm_out, expected_out, atol = 1e-3)

