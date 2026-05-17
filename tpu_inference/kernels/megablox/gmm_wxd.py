import functools
import jax
import jax.numpy as jnp 
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import time 

# Implement GMM 
# Input: lhs [b, d], rhs [g, d, h]
# Output: [b, h]
# fake inputs 

b = 256
d = 2048 
g = 4
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
    assert jnp.allclose(gmm_out, expected_out, atol=0.4), \
        f"FAILED [{label}]: max diff = {max_diff}"
    print(f"PASSED [{label}]")

def kernel_inner(
    lhs_ref,
    rhs_ref,
    out_ref,
    # Scratch
    acc_ref,  # scratch: [tile_b, tile_h] float32
    gm_lhs_start_ref,
):
    h_idx = pl.program_id(0)
    gm_idx = pl.program_id(1)
    d_idx = pl.program_id(2)

    is_first_k = d_idx == 0
    is_last_k = d_idx == (d // tile_d - 1)

    gm_start = gm_lhs_start_ref[gm_idx]
    gm_end = gm_lhs_start_ref[gm_idx + 1]
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

def fill_metadata(group_sizes_ref, gm_expert_ids_ref, gm_lhs_start_ref):
    
    gm_lhs_start_ref[0] = 0
    
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m, group_id):
        tile_size = jnp.minimum(tile_b, end_m - curr_m_offset)
        gm_expert_ids_ref[tm_id] = group_id
        gm_lhs_start_ref[tm_id] = curr_m_offset
        gm_lhs_start_ref[tm_id + 1] = curr_m_offset + tile_size
        return curr_m_offset + tile_size
    
    def outer_group_loop(group_id, carry):
        num_gm, m_offset = carry
        group_size = group_sizes_ref[group_id]
        end_m = m_offset + group_size

        num_tiles = pl.cdiv(group_size, tile_b)
        num_tiles = jnp.where(group_size == 0, 0, num_tiles)
        next_num_gm = num_gm + num_tiles

        tm_fn = functools.partial(inner_tm_loop, end_m=end_m, group_id=group_id)
        jax.lax.fori_loop(num_gm, next_num_gm, tm_fn, m_offset)

        return next_num_gm, end_m    
    
    num_gm, _ = jax.lax.fori_loop(0, g, outer_group_loop, (0, 0))
    return num_gm

def kernel_outer(group_sizes_ref, 
                 lhs_ref, 
                 rhs_ref, 
                 out_ref, 
                 acc_ref,
                 gm_expert_ids_ref, 
                 gm_lhs_start_ref):
    
    
    num_gm = fill_metadata(group_sizes_ref, gm_expert_ids_ref, gm_lhs_start_ref)
    
    grid = ( h // tile_h, num_gm, d // tile_d) 
    in_specs = [
        pl.BlockSpec(block_shape=(tile_b, tile_d), index_map=lambda  h_idx, gm_idx, d_idx: (gm_lhs_start_ref[gm_idx] // tile_b, d_idx)),
        pl.BlockSpec(block_shape=(1, tile_d, tile_h), index_map=lambda  h_idx, gm_idx, d_idx: (gm_expert_ids_ref[gm_idx], d_idx, h_idx),pipeline_mode=pl.Buffered(buffer_count=3)),
    ]
    out_specs = pl.BlockSpec(block_shape=(tile_b, tile_h), index_map=lambda  h_idx, gm_idx,  d_idx: (gm_lhs_start_ref[gm_idx] // tile_b, h_idx))

    return pltpu.emit_pipeline(
        kernel_inner, 
        grid = grid, 
        in_specs = in_specs, 
        out_specs = out_specs
    )(lhs_ref, rhs_ref, out_ref, scratches = [acc_ref, gm_lhs_start_ref]) 

@jax.jit
def gmm_native(lhs, rhs, group_sizes):
    # assume lhs is already sorted by experts
  
    in_specs = [
        pl.BlockSpec(memory_space = pltpu.HBM), #lhs
        pl.BlockSpec(memory_space = pltpu.HBM), #rhs
    ]
    out_specs = pl.BlockSpec(memory_space = pltpu.HBM) #out
    max_number_of_gm = len(group_sizes) - 1 + b//tile_b
    scope_name = (
        f"gmm_wxd-g_{g}-m_{b}-k_{d}-act_None-n_{h}"
        f"-tm_{tile_b}-tk_{tile_d}-tn_{tile_h}"
    )
    return pl.pallas_call(kernel_outer,
                   out_shape = jax.ShapeDtypeStruct((b, h), lhs.dtype),
                   name = scope_name,
                   grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=1,          
                        in_specs = in_specs, 
                        out_specs = out_specs, 
                        grid=(),
                        scratch_shapes=[pltpu.VMEM((tile_b, tile_h), jnp.float32), # acc 
                                        pltpu.SMEM((max_number_of_gm, ), jnp.int32), # gm_expert_ids, 
                                        pltpu.SMEM((max_number_of_gm + 1, ), jnp.int32), #gm_lhs_start
                                        ],
                    )
                   )(group_sizes, lhs, rhs) 
    


# --- Tests ---

## Accuracy 

# run_test([128, 0],   "one empty expert (128, 0)")
# run_test([0, 128],   "one empty expert (0, 128)")
# run_test([64, 64],   "even routing (64, 64)")
# run_test([96, 32],   "uneven routing (96, 32)")
# run_test([32, 96],   "uneven routing (32, 96)")

if __name__ == "__main__":
    run_test([1, 127, 128, 0],   "[1, 127, 128]")
    run_test([64, 64, 64, 64],   "[64, 64, 64, 64]")


    import collections
    timings = collections.defaultdict(list)

    WARMUP = 5
    N = 5

    def run_test_timed(group_sizes_list, label, warmup=False):
        group_sizes = jnp.asarray(group_sizes_list, dtype=jnp.int32)

        start_time = time.time()
        gmm_out = gmm_native(lhs, rhs, group_sizes)
        gmm_out.block_until_ready()
        if not warmup:
            timings[label].append(time.time() - start_time)

    for _ in range(WARMUP):
        run_test_timed([64, 64, 64, 64],   "even routing (64, 64, 64, 64)", warmup=True)
        run_test_timed([96, 32, 96, 32],   "uneven routing (96, 32, 96, 32)", warmup=True)
        run_test_timed([32, 96, 32, 96],   "uneven routing (32, 96, 32, 96)", warmup=True)
        run_test_timed([128, 0, 128, 0],   "one empty expert (128, 0, 128, 0)", warmup=True)
        run_test_timed([0, 128, 0, 128],   "one empty expert (0, 128, 0, 128)", warmup=True)
        run_test_timed([1, 127, 1, 127],   "extreme skew (1, 127, 1, 127)", warmup=True)

    for _ in range(N):
        run_test_timed([64, 64, 64, 64],   "even routing (64, 64, 64, 64)")
        run_test_timed([96, 32, 96, 32],   "uneven routing (96, 32, 96, 32)")
        run_test_timed([32, 96, 32, 96],   "uneven routing (32, 96, 32, 96)")
        run_test_timed([128, 0, 128, 0],   "one empty expert (128, 0, 128, 0)")
        run_test_timed([0, 128, 0, 128],   "one empty expert (0, 128, 0, 128)")
        run_test_timed([1, 127, 1, 127],   "extreme skew (1, 127, 1, 127)")


    print(f"\n--- Average over {N} runs ---")
    for label, times in timings.items():
        avg = sum(times) / len(times)
        print(f"PASSED [{label}] avg {avg*1000:.2f} ms")
