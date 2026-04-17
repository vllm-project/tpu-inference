# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

sys.path.append('.')
from tpu_inference.layers.common.fused_moe_gmm import moe_gmm_local
from tpu_inference.layers.common.sharding import ShardingAxisName


def test():
    mesh = Mesh(np.array(jax.devices()).reshape(8),
                axis_names=(ShardingAxisName.EXPERT, ))

    batch_size = 1024
    topk = 8
    local_num_experts = 4
    hidden_size = 6144
    intermediate_size = 128

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5, k_nan = jax.random.split(key, 6)

    # 1. Fix group sizes so the sum exactly matches (batch_size * topk)
    global_num_experts = local_num_experts * 2
    tokens_per_expert = (batch_size * topk) // global_num_experts

    group_sizes = jnp.full((global_num_experts, ),
                           tokens_per_expert,
                           dtype=jnp.int32)
    group_offset = jnp.array([0], dtype=jnp.int32)

    # 2. Generate clean inputs and weights
    x = jax.random.normal(k1, (batch_size * topk, hidden_size),
                          dtype=jnp.bfloat16)
    w1 = jax.random.normal(
        k2, (local_num_experts, hidden_size, intermediate_size * 2),
        dtype=jnp.bfloat16)
    w2 = jax.random.normal(k3,
                           (local_num_experts, intermediate_size, hidden_size),
                           dtype=jnp.bfloat16)
    topk_argsort_revert_indices = jax.random.permutation(
        k4, jnp.arange(batch_size * topk, dtype=jnp.int32))
    topk_weights = jax.random.normal(k5, (batch_size, topk),
                                     dtype=jnp.bfloat16)

    # 3. INJECT FAULTY VALUES (Poison the out-of-bounds tokens)
    # Tokens from index 0 to `local_token_count` belong to this shard.
    # Tokens after this index belong to experts on other shards.
    local_token_count = local_num_experts * tokens_per_expert

    # Set all tokens for remote experts to NaN.
    # This guarantees `gmm2_res` will have NaNs in the masked-out rows.
    x_wnan = x.at[local_token_count:].set(jnp.nan)
    print('cleme')
    print(jnp.isnan(x_wnan).sum())

    # 4. Grouped arguments
    args = (x_wnan, w1, w2, group_sizes, group_offset,
            topk_argsort_revert_indices, topk_weights)
    args_wnan = (x_wnan, w1, w2, group_sizes, group_offset,
                 topk_argsort_revert_indices, topk_weights)

    # 2. Refactored to return the function rather than executing it immediately
    def get_gmm_fn(sc_thresh):
        return jax.shard_map(
            lambda x_l, w1_l, w2_l, gs_l, go_l, tari_l, tw_l: moe_gmm_local(
                x=x_l,
                w1=w1_l,
                w1_scale=None,
                w1_bias=None,
                w2=w2_l,
                w2_scale=None,
                w2_bias=None,
                group_sizes=gs_l,
                group_offset=go_l,
                topk_argsort_revert_indices=tari_l,
                topk_weights=tw_l,
                activation='silu',
                topk=topk,
                parallelism='ep',
                sc_kernel_threshold=sc_thresh,
                sc_kernel_col_chunk_size=3072,
                sc_psum_num_chunks=4,
            ),
            mesh=mesh,
            in_specs=(P(), P(), P(), P(), P(), P(), P()),
            out_specs=P(),
            check_vma=False)

    print('=== Lowering and Compiling ===')
    base_fn = jax.jit(get_gmm_fn(1000000))
    base_compiled = base_fn.lower(*args).compile()

    sc_fn = jax.jit(get_gmm_fn(0))
    sc_compiled = sc_fn.lower(*args_wnan).compile()

    print('=== Warming up ===')
    out_base = base_compiled(*args)
    out_base.block_until_ready()
    out_sc = sc_compiled(*args_wnan)
    out_sc.block_until_ready()

    # Fixed `dprint` to `print`
    print('\n=== Difference Analysis ===')

    # Check if the unrouting op successfully swallowed the NaNs
    nan_count_base = jnp.isnan(out_base).sum()
    nan_count_sc = jnp.isnan(out_sc).sum()

    print(f"NaNs in Base output:       {int(nan_count_base)}")
    print(f"NaNs in SC output:         {int(nan_count_sc)}")

    if nan_count_sc > 0:
        print(
            "\n❌ FAILURE: NaN leakage detected! The unrouting op is not safely masking 0.0 * NaN."
        )
    else:
        print(
            "\n✅ SUCCESS: No NaNs detected. The unrouting op successfully masked out faulty values."
        )

    # 1. Absolute Differences
    abs_diff = jnp.abs(out_sc - out_base)
    max_abs_diff = jnp.max(abs_diff)
    mean_abs_diff = jnp.mean(abs_diff)

    # 2. Relative Differences (Add epsilon to avoid division by zero)
    epsilon = 1e-8
    rel_diff = abs_diff / (jnp.abs(out_base) + epsilon)
    max_rel_diff = jnp.max(rel_diff)
    mean_rel_diff = jnp.mean(rel_diff)

    # 3. Element Mismatch Counts
    total_elements = out_base.size

    # Exact mismatches
    exact_mismatches = jnp.sum(out_sc != out_base)
    exact_mismatch_pct = (exact_mismatches / total_elements) * 100.0

    # Tolerant mismatches (Useful because bfloat16 math can have tiny variations)
    # Using standard tolerances: rtol=1e-2, atol=1e-3
    is_close = jnp.isclose(out_base, out_sc, rtol=1e-2, atol=1e-3)
    tolerant_mismatches = jnp.sum(~is_close)
    tolerant_mismatch_pct = (tolerant_mismatches / total_elements) * 100.0

    print(f"Total elements:            {total_elements}")
    print(
        f"Exact mismatches:          {int(exact_mismatches)} ({float(exact_mismatch_pct):.4f}%)"
    )
    print(
        f"Mismatches (w/ tolerance): {int(tolerant_mismatches)} ({float(tolerant_mismatch_pct):.4f}%)"
    )
    print(f"Max absolute diff:         {float(max_abs_diff):.6f}")
    print(f"Mean absolute diff:        {float(mean_abs_diff):.6f}")
    print(f"Max relative diff:         {float(max_rel_diff):.6f}")
    print(f"Mean relative diff:        {float(mean_rel_diff):.6f}\n")

    dump_dir = '/mnt/disks/persist/dump_dir/v5_clean'
    os.makedirs(dump_dir, exist_ok=True)

    print('=== Profiling Baseline ===')
    jax.profiler.start_trace(dump_dir)
    for _ in range(5):
        out_base = base_compiled(*args)
        out_base.block_until_ready()
    jax.profiler.stop_trace()

    print('=== Profiling SC ===')
    jax.profiler.start_trace(dump_dir)
    for _ in range(5):
        out_sc = sc_compiled(*args)
        out_sc.block_until_ready()
    jax.profiler.stop_trace()
    print(f'Traces saved to {dump_dir}')


if __name__ == '__main__':
    test()
