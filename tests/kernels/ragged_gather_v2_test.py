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

from tpu_inference.kernels.sparse_core.ragged_gather import ragged_gather
from tpu_inference.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

jax.config.parse_flags_with_absl()


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class GatherTest(jtu.JaxTestCase):

    @parameterized.product(
        in_out_size=[(512, 400), (512, 1024)],
        start_end=[(3, 338), (10, 422)],
        hidden_size=[128, 512, 8192],
        dtype=[jnp.int4, jnp.int8, jnp.bfloat16, jnp.float32],
        kernel_version=[2],
    )
    def test_sc_gather(self, in_out_size, hidden_size, start_end, dtype,
                       kernel_version):
        in_size, out_size = in_out_size
        start, end = start_end
        start = min(start, out_size)
        end = min(end, out_size)
        key = jax.random.key(0)
        x = jax.random.normal(key, (in_size, hidden_size), jnp.float32)
        x = x.astype(dtype)
        indices = jax.random.randint(key, (out_size, ), 0, in_size, jnp.int32)

        start_arr = jnp.array([start], jnp.int32)
        end_arr = jnp.array([end], jnp.int32)

        kernel = ragged_gather if kernel_version == 1 else ragged_gather_v2

        actual = kernel(x, indices, start_arr, end_arr)
        actual.block_until_ready()

        # Correctness check.
        actual = actual[start:end]
        desired = x[indices][start:end]

        self.assertArraysEqual(actual, desired)

    def test_benchmark(self):
        benchmark_shapes = [
            # in_size, out_size, hidden_size
            (512, 1024, 8192),
            (1024, 2048, 8192),
            (2048, 4096, 8192),
        ]
        dtype = jnp.bfloat16

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

        for in_size, out_size, hidden_size in benchmark_shapes:

            print(f"\n=== Running shape: in={in_size}, out={out_size},"
                  f" hidden={hidden_size} ===")

            key = jax.random.key(0)
            x = jax.random.normal(key, (in_size, hidden_size),
                                  jnp.float32).astype(dtype)
            indices = jax.random.randint(key, (out_size, ), 0, in_size,
                                         jnp.int32)

            # For benchmark, use full coverage to test peak performance
            start = 0
            end = out_size
            start_arr = jnp.array([start], jnp.int32)
            end_arr = jnp.array([end], jnp.int32)

            @jax.jit
            def run_v1(x, indices, start_arr, end_arr):
                return ragged_gather(x, indices, start_arr, end_arr)

            @jax.jit
            def run_v2(x, indices, start_arr, end_arr):
                return ragged_gather_v2(x, indices, start_arr, end_arr)

            @jax.jit
            def run_jax(x, indices):
                return x[indices]

            # Run JAX baseline first
            t_jax = _time_function(run_jax, x, indices)
            res_jax = run_jax(x, indices)
            res_jax_sliced = res_jax[start:end]

            print(f"JAX:       {t_jax*1000:.3f} ms")

            # V1 Kernel
            t_v1_str = "FAILED"
            err_v1_str = "FAILED"
            try:
                t_v1 = _time_function(run_v1, x, indices, start_arr, end_arr)
                t_v1_str = f"{t_v1*1000:.3f} ms"
                res_v1 = run_v1(x, indices, start_arr, end_arr)
                res_v1_sliced = res_v1[start:end]
                err_v1 = jnp.max(jnp.abs(res_v1_sliced - res_jax_sliced))
                err_v1_str = f"{float(err_v1):.6f}"
                np.testing.assert_allclose(res_v1_sliced,
                                           res_jax_sliced,
                                           atol=1e-2,
                                           rtol=1e-2)
            except Exception:  # pylint: disable=broad-except
                print(
                    f"[Warning] V1 Kernel failed for shape ({in_size}, {out_size},"
                    f" {hidden_size})")

            # V2 Kernel
            t_v2_str = "FAILED"
            err_v2_str = "FAILED"
            try:
                t_v2 = _time_function(run_v2, x, indices, start_arr, end_arr)
                t_v2_str = f"{t_v2*1000:.3f} ms"
                res_v2 = run_v2(x, indices, start_arr, end_arr)
                res_v2_sliced = res_v2[start:end]
                err_v2 = jnp.max(jnp.abs(res_v2_sliced - res_jax_sliced))
                err_v2_str = f"{float(err_v2):.6f}"
                np.testing.assert_allclose(res_v2_sliced,
                                           res_jax_sliced,
                                           atol=1e-2,
                                           rtol=1e-2)
            except Exception:  # pylint: disable=broad-except
                print(
                    f"[Warning] V2 Kernel failed for shape ({in_size}, {out_size},"
                    f" {hidden_size})")

            print(f"V1 Kernel: {t_v1_str}")
            print(f"V2 Kernel: {t_v2_str}")
            print(f"V1 Kernel vs JAX Max Err: {err_v1_str}")
            print(f"V2 Kernel vs JAX Max Err: {err_v2_str}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
