import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import os
import sys

sys.path.append('.')
from tpu_inference.layers.common.fused_moe_gmm import moe_gmm_local
from tpu_inference.layers.common.sharding import ShardingAxisName

def test():
    mesh = Mesh(np.array(jax.devices()).reshape(8), axis_names=(ShardingAxisName.EXPERT,))

    batch_size = 1024
    topk = 8
    local_num_experts = 20
    hidden_size = 6144
    intermediate_size = 128
    
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    
    x = 8.0 * (jax.random.bernoulli(k1, 0.005, (batch_size * topk, hidden_size)).astype(jnp.bfloat16))
    w1 = 1.0 * (jax.random.bernoulli(k2, 0.005, (local_num_experts, hidden_size, intermediate_size * 2)).astype(jnp.bfloat16))
    w2 = 1.0 * (jax.random.bernoulli(k3, 0.005, (local_num_experts, intermediate_size, hidden_size)).astype(jnp.bfloat16))
    
    group_sizes = jnp.full((local_num_experts,), batch_size * topk // local_num_experts, dtype=jnp.int32)
    group_offset = jnp.array([0], dtype=jnp.int32)
    topk_argsort_revert_indices = jax.random.permutation(k4, jnp.arange(batch_size * topk, dtype=jnp.int32))
    
    topk_weights = jnp.ones((batch_size, topk), dtype=jnp.bfloat16)
    
    def run_gmm(sc_thresh):
        return jax.shard_map(
            lambda x_l, w1_l, w2_l, gs_l, go_l, tari_l, tw_l: moe_gmm_local(
                x=x_l, w1=w1_l, w1_scale=None, w1_bias=None,
                w2=w2_l, w2_scale=None, w2_bias=None,
                group_sizes=gs_l, group_offset=go_l,
                topk_argsort_revert_indices=tari_l, topk_weights=tw_l,
                activation='silu',
                topk=topk, parallelism='ep',
                sc_kernel_threshold=sc_thresh, sc_kernel_col_chunk_size=3072, sc_psum_num_chunks=4
            ),
            mesh=mesh, in_specs=(P(), P(), P(), P(), P(), P(), P()), out_specs=P(), check_vma=False
        )(x, w1, w2, group_sizes, group_offset, topk_argsort_revert_indices, topk_weights)

    print('=== Warming up (compiling) ===')
    out_base = run_gmm(1000000)
    out_base.block_until_ready()
    out_sc = run_gmm(0)
    out_sc.block_until_ready()
    
    diff = jnp.abs(out_sc - out_base)
    print(f'Max abs diff: {jnp.max(diff)}')
    
    dump_dir = '/mnt/disks/persist/dump_dir/v5_clean'
    os.makedirs(dump_dir, exist_ok=True)
    
    print('=== Profiling Baseline ===')
    jax.profiler.start_trace(dump_dir)
    for _ in range(5):
        out_base = run_gmm(1000000)
        out_base.block_until_ready()
    jax.profiler.stop_trace()

    print('=== Profiling SC ===')
    jax.profiler.start_trace(dump_dir)
    for _ in range(5):
        out_sc = run_gmm(0)
        out_sc.block_until_ready()
    jax.profiler.stop_trace()
    print(f'Traces saved to {dump_dir}')

if __name__ == '__main__':
    test()
