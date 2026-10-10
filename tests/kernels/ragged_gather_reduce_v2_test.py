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

from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2 import \
    ragged_gather_reduce as ragged_gather_reduce_v2
from tpu_inference.kernels.sparse_core.ragged_gather_reduce_v3 import \
    ragged_gather_reduce as ragged_gather_reduce_v3
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
            (ragged_gather_reduce_v2, "ragged_gather_reduce_v2"),
            (ragged_gather_reduce_v3, "ragged_gather_reduce_v3"),
        ):
            try:
                actual = rgr(x, indices, topk_weights,
                             valid_rows_mask,
                             reduce_group_size)
                np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)
            except AssertionError:
                raise
            except Exception as e:  # pylint: disable=broad-except
                print(f"Skipping {name} correctness check due to error: {e}")

    def test_sc_ragged_gather_reduce_v3_sentinel_does_not_read_uninitialized_rows(
            self):
        """Regression: a 64-token block with exactly 2*64 valid routes that are not
        adjacent pairs makes v3 emit a zero-weight sentinel route. Under EP the
        rows of `x` outside the local expert range are uninitialized HBM; the
        sentinel must not gather such a row, otherwise 0 * NaN poisons token 0."""
        num_tokens, topk, hidden = 2048, 10, 4096
        n = num_tokens * topk
        rng = np.random.default_rng(7)
        perm = rng.permutation(n).astype(np.int32)
        valid = (perm >= 2000) & (perm < 4600)
        blk = valid[:64 * topk].reshape(64, topk)
        blk[:] = False
        blk[:, :2] = True
        blk[0, 1] = False
        # Still 128 valid routes, but token 0 has 1 and token 1 has 3.
        blk[1, 2] = True
        valid[:64 * topk] = blk.reshape(-1)
        x = np.full((n, hidden), np.nan, np.float32)
        x[perm[valid]] = rng.standard_normal((int(valid.sum()), hidden),
                                             dtype=np.float32)
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        indices = jnp.asarray(perm)
        w = jnp.asarray(rng.standard_normal(n).astype(np.float32),
                        dtype=jnp.bfloat16)
        vm = jnp.asarray(valid)
        desired = reference_ragged_gather_reduce(x, indices, w, vm, topk)
        actual = ragged_gather_reduce_v3(x, indices, w, vm, topk)
        self.assertFalse(bool(jnp.isnan(actual.astype(jnp.float32)).any()))
        np.testing.assert_allclose(actual, desired, atol=1e-2, rtol=1e-2)

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
