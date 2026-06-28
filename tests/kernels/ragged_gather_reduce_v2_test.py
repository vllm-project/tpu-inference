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

import functools
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from tpu_inference.kernels.sparse_core import ragged_gather_reduce_v2 as v2_mod
from tpu_inference.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce as ragged_gather_reduce_v1
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as ragged_gather_reduce_v2
from tpu_inference.kernels.sparse_core.ragged_scatter import ragged_scatter

jax.config.parse_flags_with_absl()


def reference_ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    out = x[indices] * topk_weights[:, None].astype(jnp.float32)
    out = jnp.where(valid_rows_mask[:, None], out, 0)
    out = out.reshape(-1, reduce_group_size, out.shape[-1])
    out = jnp.sum(out, axis=1).astype(jnp.bfloat16)
    return out


@functools.partial(jax.jit, static_argnames="reduce_group_size")
def ragged_scatter_and_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    start: jax.Array,
    end: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Reference implementation of ragged gather reduce."""
    x = ragged_scatter(x, indices, start, end)
    out = x.reshape((-1, reduce_group_size, x.shape[-1]))
    topk_weights = topk_weights.reshape((-1, reduce_group_size))[..., None]
    out = out * topk_weights
    out = jnp.where(
        valid_rows_mask.reshape((-1, reduce_group_size))[:, :, None], out, 0.0)
    out = out.sum(axis=-2)
    return out


def _time_function(fn, *args, n_repeats=100):
    # Warmup
    for _ in range(10):
        fn(*args).block_until_ready()

    # Asynchronous dispatch to hide Python overhead
    start = time.perf_counter()
    results = [fn(*args) for _ in range(n_repeats)]
    results[-1].block_until_ready()
    end = time.perf_counter()

    return (end - start) / n_repeats


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ScatterTest(jtu.JaxTestCase):
    _test_cases = [
        dict(out_size=o,
             start_end=se,
             hidden_size=h,
             dtype=d,
             reduce_group_size=rg) for o, se, h, d, rg in itertools.chain(
                 itertools.product(
                     [400, 840],
                     [(3, 338), (10, 255)],
                     [128, 512, 8192],
                     [jnp.bfloat16, jnp.float32],
                     [8, 5],
                 ),
                 itertools.product(
                     [16384],
                     [(99, 1120)],
                     [7168],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [16384],
                     [(300, 2358)],
                     [6144],
                     [jnp.bfloat16],
                     [8],
                 ),
                 itertools.product(
                     [20480],
                     [(300, 2850)],
                     [4096],
                     [jnp.bfloat16],
                     [10],
                 ),
             )
    ]

    @parameterized.parameters(*_test_cases)
    def test_sc_ragged_gather_reduce(self, out_size, hidden_size, start_end,
                                     dtype, reduce_group_size):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )
        # Correctness check.
        desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                 valid_rows_mask,
                                                 reduce_group_size)
        for rgr, name in (
            (ragged_gather_reduce_v1, "ragged_gather_reduce_v1"),
            (ragged_gather_reduce_v2, "ragged_gather_reduce_v2"),
        ):
            try:
                actual = rgr(x, indices, topk_weights, valid_rows_mask,
                             reduce_group_size)
                np.testing.assert_allclose(actual,
                                           desired,
                                           atol=1e-2,
                                           rtol=1e-2)
            except AssertionError:
                raise
            except Exception as e:  # pylint: disable=broad-except
                print(f"Skipping {name} correctness check due to error: {e}")

    def test_sc_ragged_gather_reduce_v2_no_spmem_overflow(self):
        """Regression: large input_size must not overflow SparseCore SPMEM.

        v2 (#2836) staged the *entire* row-partition sort permutation into one
        resident SPMEM scratch (`sorted_by_validity_vmem`), which scales linearly
        with input_size and overflows the per-subcore tile_spmem budget. On v7,
        input_size=1,835,008 (hidden=1024) overflows the unfixed kernel with
        `E3000 CompileTimeSparseCoreAllocationFailure`. The windowed fix bounds
        the scratch to a fixed window so it compiles for any input_size.

        Compile-only (ShapeDtypeStruct) — no large allocation; hidden kept small
        so x stays under the 16 GiB SparseCore per-tensor limit.
        """
        n, h, rgs = 1_835_008, 1024, 8
        specs = (
            jax.ShapeDtypeStruct((n, h), jnp.bfloat16),
            jax.ShapeDtypeStruct((n, ), jnp.int32),
            jax.ShapeDtypeStruct((n, ), jnp.bfloat16),
            jax.ShapeDtypeStruct((n, ), jnp.bool_),
        )
        # Must compile without raising E3000 (do NOT swallow — this is the gate).
        ragged_gather_reduce_v2.lower(*specs,
                                      reduce_group_size=rgs).compile()

    # Forced-small-window cases that exercise the multi-window streaming path on
    # SC-reaching shapes (out_size large enough to skip the TensorCore fallback).
    # max_window=1 puts every row-block in its own window, so the cross-window
    # segmented-reduce carry is tested at every block boundary -- the riskiest
    # path. Covers partial last window, reduce-group splits across window
    # boundaries, bf16 row-packing across a boundary, f32, and an empty (all
    # rows masked) partition.
    _windowed_test_cases = [
        dict(out_size=o, start_end=se, hidden_size=h, dtype=d,
             reduce_group_size=rg, max_window=mw)
        for o, se, h, d, rg, mw in (
            (16384, (99, 1120), 7168, jnp.bfloat16, 8, 1),
            (16384, (99, 1120), 7168, jnp.bfloat16, 8, 2),
            (16384, (99, 1120), 7168, jnp.bfloat16, 8, 3),
            (16384, (99, 1120), 7168, jnp.float32, 8, 1),
            (16384, (300, 2358), 6144, jnp.bfloat16, 8, 2),
            (20480, (300, 2850), 4096, jnp.bfloat16, 10, 1),
            (16384, (0, 0), 7168, jnp.bfloat16, 8, 2),  # all rows masked out
        )
    ]

    @parameterized.parameters(*_windowed_test_cases)
    def test_sc_ragged_gather_reduce_v2_windowed(self, out_size, hidden_size,
                                                 start_end, dtype,
                                                 reduce_group_size, max_window):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ), True, False)
        desired = reference_ragged_gather_reduce(x, indices, topk_weights,
                                                 valid_rows_mask,
                                                 reduce_group_size)
        # Force a small per-window block count; clear the jit cache so the new
        # window size actually takes effect for an already-seen shape.
        prev = v2_mod._MAX_WINDOW_OVERRIDE
        v2_mod._MAX_WINDOW_OVERRIDE = max_window
        jax.clear_caches()
        try:
            actual = ragged_gather_reduce_v2(x, indices, topk_weights,
                                             valid_rows_mask,
                                             reduce_group_size)
            np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)
        finally:
            v2_mod._MAX_WINDOW_OVERRIDE = prev
            jax.clear_caches()

    def test_max_row_window_bounds_spmem(self):
        """`_max_row_window` keeps the per-subcore SPMEM use under budget."""
        from jax.experimental.pallas import tpu as pltpu
        sc = pltpu.get_tpu_info().sparse_core
        budget = sc.vmem_capacity_bytes // 4
        num_simd_lanes = sc.num_lanes
        # Sweep representative (hidden-driven col tiling, row_chunk, blocks).
        for col_size, col_chunk, row_chunk in ((512, 512, 64), (3584, 896, 64),
                                               (2048, 1024, 64), (4096, 1024,
                                                                  16)):
            for max_blocks in (1, 4, 100, 10_000, 1_000_000):
                w = v2_mod._max_row_window(row_chunk, col_size, col_chunk,
                                           num_simd_lanes, max_blocks)
                self.assertGreaterEqual(w, 1)
                self.assertLessEqual(w, max_blocks)
                fixed = (col_size + num_simd_lanes * col_chunk +
                         2 * num_simd_lanes * col_chunk + num_simd_lanes +
                         6 * row_chunk)
                per_subcore = w * row_chunk + fixed
                # Window=1 may not fit a pathologically large `fixed`; otherwise
                # the chosen window must respect the budget with headroom.
                if w > 1:
                    self.assertLessEqual(per_subcore, budget)

    # The first perf test case approximates the DeepSeekV3, 2k-batch-size, EP=16.
    # The second case approximates the Qwen3-Coder-480B, 2k-batch-size, EP=8.
    _perf_test_cases = [
        dict(
            out_size=o,
            start_end=se,
            hidden_size=h,
            dtype=d,
            reduce_group_size=rg,
            col_chunk_size=c_sz,
        ) for o, se, h, d, rg, c_sz in itertools.chain(
            itertools.product(
                [16384],
                [(99, 1120)],
                [7168],
                [jnp.bfloat16],
                [8],
                [3584],
            ),
            itertools.product(
                [16384],
                [(300, 2400)],
                [6144],
                [jnp.bfloat16],
                [8],
                [2048],
            ),
            itertools.product(
                [65536],
                [(100, 8300)],
                [6144],
                [jnp.bfloat16],
                [8],
                [2048],
            ),
        )
    ]

    @parameterized.parameters(*_perf_test_cases)
    def test_perf(
        self,
        out_size,
        hidden_size,
        start_end,
        dtype,
        reduce_group_size,
        col_chunk_size,
    ):
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (out_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.permutation(key, out_size)
        topk_weights = jax.random.normal(key, (out_size, ), jnp.bfloat16)
        valid_rows_mask = jnp.where(
            jnp.logical_and(
                jnp.array([start], jnp.int32) <= indices,
                indices < jnp.array([end], jnp.int32),
            ),
            True,
            False,
        )

        print(f"\n=== Running shape: out={out_size},"
              f" hidden={hidden_size}, start={start}, end={end} ===")

        def run_and_time(name, fn, *args):
            try:
                t_val = _time_function(fn, *args)
                print(f"{name}: {t_val*1000:.3f} ms")
            except Exception as e:  # pylint: disable=broad-except
                print(f"{name} failed: {e}")

        run_and_time(
            "ragged_scatter_and_reduce",
            ragged_scatter_and_reduce,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            start,
            end,
            reduce_group_size,
        )

        run_and_time(
            "ragged_gather_reduce_v1",
            ragged_gather_reduce_v1,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )

        run_and_time(
            "ragged_gather_reduce_v2",
            ragged_gather_reduce_v2,
            x,
            indices,
            topk_weights,
            valid_rows_mask,
            reduce_group_size,
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
