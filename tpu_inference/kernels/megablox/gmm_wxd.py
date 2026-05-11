import jax
import jax.numpy as jnp 
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import time 

# Implement GMM 
# Input: lhs [b, d], rhs [g, d, h]
# Output: [b, h]
# fake inputs 

b = 128
d = 2048 
g = 2
h = 4096
tokens_per_expert = b // g

tile_b = 128 
tile_d = 1024 
tile_h = 1024 


key = jax.random.PRNGKey(0)
lhs = jax.random.normal(key, (b, d)).astype(jnp.bfloat16)
rhs = jax.random.normal(key, (g, d, h)).astype(jnp.bfloat16)


def compute_expected(lhs, rhs, group_sizes):
    """Reference implementation: tile K identically to the kernel so float32
    accumulation order matches and bf16 rounding is bit-exact."""
    offsets = jnp.concatenate([jnp.array([0]), jnp.cumsum(group_sizes)])
    out = jnp.zeros((b, h), dtype=lhs.dtype)
    for g_idx in range(g):
        start = int(offsets[g_idx])
        end = int(offsets[g_idx + 1])
        if end > start:
            # Accumulate in float32 over K-tiles, matching the kernel's acc_ref pattern.
            acc = jnp.zeros((end - start, h), dtype=jnp.float32)
            for k in range(0, d, tile_d):
                acc += jnp.matmul(
                    lhs[start:end, k:k + tile_d],
                    rhs[g_idx, k:k + tile_d, :],
                    preferred_element_type=jnp.float32,
                )
            out = out.at[start:end].set(acc.astype(lhs.dtype))
    return out


def run_test(group_sizes_list, label):
    
    group_sizes = jnp.asarray(group_sizes_list, dtype=jnp.int32)
    expected_out = compute_expected(lhs, rhs, group_sizes)
    gmm_out = gmm_native(lhs, rhs, group_sizes)
    print("expected_out", expected_out)
    print("gmm_out", gmm_out)
    max_diff = jnp.max(jnp.abs(gmm_out - expected_out))
    assert jnp.allclose(gmm_out, expected_out, atol=0.02), \
        f"FAILED [{label}]: max diff = {max_diff}"
    print(f"PASSED [{label}]")

def kernel(
    gm_start_ref,
    gm_end_ref,
    gm_expert_ids_ref,
    lhs_ref,
    rhs_ref,
    out_ref,
    acc_ref,  # scratch: [tile_b, tile_h] float32
):
    h_idx = pl.program_id(0)
    gm_idx = pl.program_id(1)
    d_idx = pl.program_id(2)

    is_first_k = d_idx == 0
    is_last_k = d_idx == (d // tile_d - 1)

    gm_start = gm_start_ref[gm_idx]
    gm_end = gm_end_ref[gm_idx]
    row = (gm_start // tile_b) * tile_b + jnp.arange(tile_b)[:, None]  # [tile_b, 1]
    mask = (row >= gm_start) & (row < gm_end)  # [tile_b, 1]
    
    # On the first K tile, zero out the accumulator.
    @pl.when(is_first_k)
    def _():
        acc_ref[...] = jnp.where(mask, jnp.zeros_like(acc_ref[...]), acc_ref[...])
    
    res = jnp.matmul(lhs_ref[...], rhs_ref[0, ...], preferred_element_type=jnp.float32)
    res = jnp.where(mask, res, 0)
    acc_ref[...] = acc_ref[...] + res

    # Only write to output on the last K tile.
    @pl.when(is_last_k)
    def _():
        out_ref[...] = jnp.where(mask, acc_ref[...].astype(out_ref.dtype), out_ref[...])

def compute_metadata(group_sizes):
    lhs_pointer = 0 
    gm_expert_ids, gm_lhs_start, gm_lhs_end = [], [], []
    for expert_id, gs in enumerate(group_sizes):
        if gs == 0:
            continue 
        for tile_start in range(lhs_pointer, lhs_pointer + gs, tile_b):
            
            gm_expert_ids.append(expert_id)
            gm_lhs_start.append( tile_start)
            gm_lhs_end.append(min(lhs_pointer + gs, tile_start + tile_b))
        
        lhs_pointer += gs
    return gm_expert_ids, gm_lhs_start, gm_lhs_end

@jax.jit
def gmm_native(lhs, rhs, group_sizes, gm_expert_ids, gm_lhs_start, gm_lhs_end):
    # assume lhs is already sorted by experts
  
    grid = ( h // tile_h, len(gm_expert_ids), d // tile_d) 

    in_specs = [
        pl.BlockSpec(block_shape=(tile_b, tile_d), index_map=lambda  h_idx, gm_idx, d_idx, lhs_start_ref, lhs_end_ref, gm_expert_ids_ref: (lhs_start_ref[gm_idx] // tile_b, d_idx)),
        pl.BlockSpec(block_shape=(1, tile_d, tile_h), index_map=lambda  h_idx, gm_idx, d_idx, lhs_start_ref, lhs_end_ref, gm_expert_ids_ref: (gm_expert_ids_ref[gm_idx], d_idx, h_idx)),
    ]

    out_specs = pl.BlockSpec(block_shape=(tile_b, tile_h), index_map=lambda  h_idx, gm_idx,  d_idx, lhs_start_ref, lhs_end_ref, gm_expert_ids_ref: (lhs_start_ref[gm_idx] // tile_b, h_idx))
    
    return pl.pallas_call(kernel, 
                   
                   out_shape = jax.ShapeDtypeStruct((b, h), lhs.dtype),
                   grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=3,          # how many leading args are scalar prefetch
                        in_specs = in_specs, 
                        out_specs = out_specs, 
                        grid=grid,
                        scratch_shapes=[pltpu.VMEM((tile_b, tile_h), jnp.float32)],
                    )
                   )(gm_lhs_start, gm_lhs_end, gm_expert_ids, lhs, rhs) 
    


# --- Tests ---

## Accuracy 

# run_test([128, 0],   "one empty expert (128, 0)")
# run_test([0, 128],   "one empty expert (0, 128)")
# run_test([64, 64],   "even routing (64, 64)")
# run_test([96, 32],   "uneven routing (96, 32)")
# run_test([32, 96],   "uneven routing (32, 96)")
# run_test([1, 127],   "extreme skew (1, 127)")


import collections
timings = collections.defaultdict(list)

def run_test_timed(group_sizes_list, label):
    group_sizes = jnp.asarray(group_sizes_list, dtype=jnp.int32)
    gm_expert_ids, gm_lhs_start, gm_lhs_end = compute_metadata(group_sizes)
    
    gm_expert_ids_arr = jnp.asarray(gm_expert_ids, dtype=jnp.int32)
    gm_lhs_start_arr  = jnp.asarray(gm_lhs_start,  dtype=jnp.int32)
    gm_lhs_end_arr    = jnp.asarray(gm_lhs_end,    dtype=jnp.int32)

    start_time = time.time()
    gmm_out = gmm_native(lhs, rhs, group_sizes, gm_expert_ids_arr, gm_lhs_start_arr, gm_lhs_end_arr)
    gmm_out.block_until_ready()
    timings[label].append(time.time() - start_time)
N = 10
for _ in range(N):
    run_test_timed([64, 64],   "even routing (64, 64)")
    run_test_timed([96, 32],   "uneven routing (96, 32)")
    run_test_timed([32, 96],   "uneven routing (32, 96)")
    run_test_timed([128, 0],   "one empty expert (128, 0)")
    run_test_timed([0, 128],   "one empty expert (0, 128)")
    run_test_timed([1, 127],   "extreme skew (1, 127)")

print(f"\n--- Average over {N} runs ---")
for label, times in timings.items():
    avg = sum(times) / len(times)
    print(f"PASSED [{label}] avg {avg*1000:.2f} ms")


# v1. all tokens go through matmul for each export 


# PASSED [even routing (64, 64)] in 0.1136634349822998
# PASSED [uneven routing (96, 32)] in 0.11358451843261719
# PASSED [uneven routing (32, 96)] in 0.10309863090515137
# PASSED [one empty expert (128, 0)] in 0.10653448104858398
# PASSED [one empty expert (0, 128)] in 0.1001133918762207
# PASSED [extreme skew (1, 127)] in 0.10481739044189453



# v2. use jax.lax.cond to skip matmul for non-overlapping experts. s

# PASSED [even routing (64, 64)] avg 111.47 ms
# PASSED [uneven routing (96, 32)] avg 110.56 ms
# PASSED [uneven routing (32, 96)] avg 111.72 ms
# PASSED [one empty expert (128, 0)] avg 109.14 ms
# PASSED [one empty expert (0, 128)] avg 110.48 ms
# PASSED [extreme skew (1, 127)] avg 127.90 ms


# v3. use gm 

# --- Average over 10 runs ---
# PASSED [even routing (64, 64)] avg 238.79 ms
# PASSED [uneven routing (96, 32)] avg 205.71 ms
# PASSED [uneven routing (32, 96)] avg 200.84 ms
# PASSED [one empty expert (128, 0)] avg 204.95 ms
# PASSED [one empty expert (0, 128)] avg 197.88 ms
# PASSED [extreme skew (1, 127)] avg 198.32 ms

# v4 . jitted 

# PASSED [even routing (64, 64)] avg 23.03 ms
# PASSED [uneven routing (96, 32)] avg 0.21 ms
# PASSED [uneven routing (32, 96)] avg 0.22 ms
# PASSED [one empty expert (128, 0)] avg 27.42 ms
# PASSED [one empty expert (0, 128)] avg 0.23 ms
# PASSED [extreme skew (1, 127)] avg 0.20 ms