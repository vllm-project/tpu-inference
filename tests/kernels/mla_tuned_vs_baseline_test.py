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

import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tests.kernels.mla_v2_test import generate_mla_inputs
from tpu_inference.kernels.mla.v2.kernel import mla_ragged_paged_attention
from tpu_inference.kernels.mla.v2.tuned_params import (TunableParams,
                                                       tuned_params_mapping)


def _get_tuned_test_cases():
    test_cases = []
    for key in tuned_params_mapping.keys():
        name = (f"tokens_{key.max_num_tokens}_heads_{key.actual_num_q_heads}_"
                f"pages_{key.total_num_pages}_"
                f"seqs_{key.max_num_seqs}_"
                f"pagesperseq_{key.pages_per_seq}")
        test_cases.append(dict(testcase_name=name, key=key))
    return test_cases


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MlaTunedVsBaselinePerformanceTest(jtu.JaxTestCase):

    @parameterized.named_parameters(*_get_tuned_test_cases())
    def test_tuned_vs_baseline_performance(self, key):
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Performance comparison requires TPUv4+")

        tuned_params = tuned_params_mapping[key]

        baseline_params = TunableParams(
            decode_batch_size=4,
            num_kv_pages_per_block=3,
            num_queries_per_block=1,
            vmem_limit_bytes=64 * 1024 * 1024,
        )

        kv_len = key.pages_per_seq * key.page_size_per_kv_packing * key.kv_packing
        rng = np.random.default_rng(1234)
        inputs = generate_mla_inputs(
            seq_lens=[[1, kv_len] for _ in range(key.max_num_seqs)],
            num_heads=key.actual_num_q_heads,
            lkv_dim=key.actual_lkv_dim,
            r_dim=key.actual_r_dim,
            page_size=key.page_size_per_kv_packing * key.kv_packing,
            q_dtype=jnp.dtype(key.q_dtype),
            kv_dtype=jnp.dtype(key.kv_dtype),
            num_pages=key.total_num_pages,
            rng=rng,
        )

        (ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices,
         cu_q_lens, distribution) = inputs
        ql_nope_transposed = jnp.transpose(ql_nope, (1, 0, 2))

        @jax.jit(static_argnames=[
            'decode_batch_size', 'num_kv_pages_per_block',
            'num_queries_per_block', 'vmem_limit_bytes'
        ])
        def run_kernel(decode_batch_size, num_kv_pages_per_block,
                       num_queries_per_block, vmem_limit_bytes):
            return mla_ragged_paged_attention(
                ql_nope=ql_nope_transposed,
                q_pe=q_pe,
                new_kv_c=new_kv_c,
                new_k_pe=new_k_pe,
                cache_kv=cache_kv.copy(),
                kv_lens=kv_lens,
                page_indices=page_indices,
                cu_q_lens=cu_q_lens,
                distribution=distribution,
                sliding_window=key.sliding_window,
                soft_cap=key.soft_cap,
                q_scale=None,
                k_scale=None,
                v_scale=None,
                chunk_prefill_size=key.chunk_prefill_size,
                s_dtype=key.s_dtype,
                p_same_dtype_as_v=key.p_same_dtype_as_v,
                decode_batch_size=decode_batch_size,
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
                vmem_limit_bytes=vmem_limit_bytes,
            )

        print(f"\nCompiling baseline kernel for: {key}...")
        jax.block_until_ready(
            run_kernel(baseline_params.decode_batch_size,
                       baseline_params.num_kv_pages_per_block,
                       baseline_params.num_queries_per_block,
                       baseline_params.vmem_limit_bytes))
        print(f"Compiling tuned kernel for: {key}...")
        jax.block_until_ready(
            run_kernel(tuned_params.decode_batch_size,
                       tuned_params.num_kv_pages_per_block,
                       tuned_params.num_queries_per_block,
                       tuned_params.vmem_limit_bytes))

        iters = 50
        start_ns = time.perf_counter_ns()
        for _ in range(iters):
            jax.block_until_ready(
                run_kernel(baseline_params.decode_batch_size,
                           baseline_params.num_kv_pages_per_block,
                           baseline_params.num_queries_per_block,
                           baseline_params.vmem_limit_bytes))
        baseline_latency = (time.perf_counter_ns() - start_ns) / iters

        start_ns = time.perf_counter_ns()
        for _ in range(iters):
            jax.block_until_ready(
                run_kernel(tuned_params.decode_batch_size,
                           tuned_params.num_kv_pages_per_block,
                           tuned_params.num_queries_per_block,
                           tuned_params.vmem_limit_bytes))
        tuned_latency = (time.perf_counter_ns() - start_ns) / iters

        speedup = (baseline_latency - tuned_latency) / baseline_latency * 100

        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON FOR KEY:")
        print(f"{key}")
        print("-" * 80)
        print(
            f"Baseline (BS={baseline_params.decode_batch_size}, Pages={baseline_params.num_kv_pages_per_block}): {baseline_latency / 1e3:.2f} us"
        )
        print(
            f"Tuned    (BS={tuned_params.decode_batch_size}, Pages={tuned_params.num_kv_pages_per_block}): {tuned_latency / 1e3:.2f} us"
        )
        print(f"Speedup: {speedup:+.2f}%")
        print("=" * 80 + "\n")

        self.assertLessEqual(
            tuned_latency, baseline_latency * 1.05,
            f"Regression detected! Tuned latency ({tuned_latency / 1e3:.2f} us) "
            f"is significantly slower than baseline ({baseline_latency / 1e3:.2f} us)"
        )


if __name__ == "__main__":
    absltest.main()
