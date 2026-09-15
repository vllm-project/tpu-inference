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
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.megablox.gmm_v2 import gmm_v2
from tpu_inference.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

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


def _bf16_silu_kw():
    return dict(group_offset=jnp.array([0], jnp.int32),
                fuse_act="silu",
                preferred_element_type=jnp.bfloat16,
                maybe_quantize_lhs=False,
                zero_initialize=False)


def _permute_methods(rhs, group_sizes, kw):
    """The four GMM1 permute paths as jitted fns of (fp32 source pool, idx)."""
    rhs = rhs.astype(jnp.bfloat16)

    def tc_fused(src, i):  # gather fused into GMM1's LHS read (TensorCore)
        return gmm_v2(src, rhs, group_sizes, gather_indices=i, **kw)

    def tc_gather(src, i):  # XLA gather permute, then a contiguous gmm_v2 (TC)
        return gmm_v2(src[i].astype(jnp.bfloat16), rhs, group_sizes, **kw)

    def tc_onehot(src, i):  # onehot matmul permute (MoE small-batch path, TC)
        oh = jax.nn.one_hot(i, src.shape[0], dtype=jnp.bfloat16)
        return gmm_v2(oh @ src.astype(jnp.bfloat16), rhs, group_sizes, **kw)

    def sc_gather(src, i):  # SparseCore ragged_gather_v2 permute (prod path)
        # ragged_gather_v2 mixes int dtypes internally; allow standard dtype
        # promotion (the test runs under jtu's strict mode).
        with jax.numpy_dtype_promotion("standard"):
            permuted = ragged_gather_v2(src.astype(jnp.bfloat16), i,
                                        jnp.int32(0), jnp.int32(i.shape[0]))
        return gmm_v2(permuted, rhs, group_sizes, **kw)

    return {
        name: jax.jit(fn)
        for name, fn in (("tc_fused", tc_fused), ("tc_gather", tc_gather),
                         ("tc_onehot", tc_onehot), ("sc_gather", sc_gather))
    }


def _capture_xprof(prof_dir, case=_PERF_CASES[1], iters=20):
    """Trace each permute path into its own <prof_dir>/<method>/ subdir, so each
    produces a separate xplane.pb (one xprof per method). The subdirs form the
    phased layout the upload-profile-to-xprof helper expects."""
    size_m, k, n, num_groups, num_src = case
    lhs, rhs, group_sizes, idx = _make_inputs(size_m,
                                              k,
                                              n,
                                              num_groups,
                                              num_src=num_src)
    for name, fn in _permute_methods(rhs, group_sizes,
                                     _bf16_silu_kw()).items():
        jax.block_until_ready(fn(lhs, idx))  # compile before tracing
        method_dir = os.path.join(prof_dir, name)
        jax.profiler.start_trace(method_dir)
        for i in range(iters):
            with jax.profiler.StepTraceAnnotation(name, step_num=i):
                jax.block_until_ready(fn(lhs, idx))
        jax.profiler.stop_trace()
        print(f"  [xprof] {name} (m={size_m} k={k} n={n}) -> {method_dir}")


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
        # Benchmark the fused gather (tc_fused) against the three unfused permute
        # paths at a few representative prefill GMM1 shapes (bf16 matmul, silu):
        # tc_gather (XLA lhs[idx]), tc_onehot (onehot matmul), and sc_gather (the
        # SparseCore ragged_gather). The assertion only checks tc_fused vs
        # tc_gather stay in the same ballpark -- a benchmark, not a win gate.
        print("\n  fused permute vs unfused permute + gmm_v2 (bf16, silu):")
        for size_m, k, n, num_groups, num_src in _PERF_CASES:
            lhs, rhs, group_sizes, idx = _make_inputs(size_m,
                                                      k,
                                                      n,
                                                      num_groups,
                                                      num_src=num_src)
            m = _permute_methods(rhs, group_sizes, _bf16_silu_kw())
            t = {name: _median_us(fn, lhs, idx) for name, fn in m.items()}
            print(f"    m={size_m:>6} k={k} n={n} E={num_groups}:  "
                  f"tc_fused={t['tc_fused']:8.1f}  "
                  f"tc_gather={t['tc_gather']:8.1f}  "
                  f"tc_onehot={t['tc_onehot']:8.1f}  "
                  f"sc_gather={t['sc_gather']:8.1f} us  "
                  f"(tc_fused/tc_gather={t['tc_fused'] / t['tc_gather']:.2f}, "
                  f"tc_fused/tc_onehot={t['tc_fused'] / t['tc_onehot']:.2f}, "
                  f"tc_fused/sc_gather={t['tc_fused'] / t['sc_gather']:.2f})")
            self.assertBetween(t["tc_fused"] / t["tc_gather"], 0.5, 2.0)
        # Optional xprof capture (set FUSE_PERMUTE_XPROF_DIR) for kernel-level
        # attribution of the perf differences between the three paths.
        prof_dir = os.environ.get("FUSE_PERMUTE_XPROF_DIR")
        if prof_dir:
            _capture_xprof(prof_dir)


if __name__ == "__main__":
    absltest.main()
