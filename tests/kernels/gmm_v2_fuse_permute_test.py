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
"""Tests for the fused-permute path of gmm_v2.

`gmm_v2(..., gather_indices=idx)` gathers its LHS rows per-row from an
un-permuted source pool instead of reading a pre-permuted contiguous block.
We check it is (1) numerically identical to the unfused
`gmm_v2(lhs[idx], ...)` and (2) within a reasonable perf range of a standalone
permute + gmm_v2, benchmarked over a few representative GMM1 shapes (the
timings are printed).
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2

jax.config.parse_flags_with_absl()


def _balanced_group_sizes(size_m: int, num_groups: int) -> jax.Array:
    base = size_m // num_groups
    sizes = [base] * num_groups
    sizes[-1] += size_m - base * num_groups
    return jnp.asarray(sizes, jnp.int32)


def _make_inputs(size_m, k, n, num_groups, num_src, seed=0):
    k0, k1, k2 = jax.random.split(jax.random.key(seed), 3)
    # The gather needs an fp32 source pool (per-row addressability).
    lhs = jax.random.normal(k0, (num_src, k), jnp.float32) * 0.5
    rhs = jax.random.normal(k1, (num_groups, k, n), jnp.float32) * 0.05
    group_sizes = _balanced_group_sizes(size_m, num_groups)
    gather_idx = jax.random.randint(k2, (size_m, ), 0, num_src, jnp.int32)
    return lhs, rhs, group_sizes, gather_idx


def _median_us(fn, *args, warmup=3, iters=10):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(samples))


# Representative prefill GMM1 shapes: (size_m, k, n, num_groups, num_src).
_PERF_CASES = [
    (4096, 2048, 1536, 8, 1024),
    (8192, 4096, 2048, 8, 2048),
    (16384, 2048, 768, 16, 4096),
]


class GmmV2FusePermuteTest(jtu.JaxTestCase):

    @parameterized.named_parameters(
        ("no_act", None),
        ("silu", "silu"),
    )
    def test_gather_matches_unfused(self, fuse_act):
        lhs, rhs, group_sizes, idx = _make_inputs(512,
                                                  256,
                                                  512,
                                                  4,
                                                  num_src=512)
        kw = dict(group_offset=jnp.array([0], jnp.int32),
                  fuse_act=fuse_act,
                  preferred_element_type=jnp.float32,
                  maybe_quantize_lhs=False,
                  zero_initialize=False)
        fused = gmm_v2(lhs, rhs, group_sizes, gather_indices=idx, **kw)
        unfused = gmm_v2(lhs[idx], rhs, group_sizes, **kw)
        self.assertArraysEqual(fused, unfused)

    def test_fused_permute_perf(self):
        # Benchmark the fused gather against the two unfused permute paths at a few
        # representative prefill GMM1 shapes (bf16 matmul, silu) and print the
        # timings: an XLA gather (lhs[idx]) and the onehot matmul permute the MoE
        # layer uses for small batches. The assertion only checks fused vs the XLA
        # gather stay in the same ballpark -- this is a benchmark, not a win gate.
        print("\n  fused permute vs unfused permute + gmm_v2 (bf16, silu):")
        for size_m, k, n, num_groups, num_src in _PERF_CASES:
            lhs, rhs, group_sizes, idx = _make_inputs(size_m,
                                                      k,
                                                      n,
                                                      num_groups,
                                                      num_src=num_src)
            rhs = rhs.astype(jnp.bfloat16)
            kw = dict(group_offset=jnp.array([0], jnp.int32),
                      fuse_act="silu",
                      preferred_element_type=jnp.bfloat16,
                      maybe_quantize_lhs=False,
                      zero_initialize=False)
            fused = jax.jit(lambda src, i: gmm_v2(
                src, rhs, group_sizes, gather_indices=i, **kw))
            # XLA gather permute, then a contiguous gmm_v2.
            gather = jax.jit(lambda src, i: gmm_v2(src[i].astype(jnp.bfloat16),
                                                   rhs, group_sizes, **kw))

            # Onehot matmul permute (one_hot(i, num_src) @ src), the path the MoE
            # layer takes below onehot_moe_permute_threshold, then a contiguous gmm.
            def onehot_permute(src, i):
                oh = jax.nn.one_hot(i, src.shape[0], dtype=jnp.bfloat16)
                return gmm_v2(oh @ src.astype(jnp.bfloat16), rhs, group_sizes,
                              **kw)

            onehot = jax.jit(onehot_permute)
            t_fused = _median_us(fused, lhs, idx)
            t_gather = _median_us(gather, lhs, idx)
            t_onehot = _median_us(onehot, lhs, idx)
            print(f"    m={size_m:>6} k={k} n={n} E={num_groups}:  "
                  f"fused={t_fused:8.1f}  gather={t_gather:8.1f}  "
                  f"onehot={t_onehot:8.1f} us  "
                  f"(fused/gather={t_fused / t_gather:.2f}, "
                  f"fused/onehot={t_fused / t_onehot:.2f})")
            self.assertBetween(t_fused / t_gather, 0.5, 2.0)


if __name__ == "__main__":
    absltest.main()
