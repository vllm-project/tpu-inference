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
"""Benchmark the matmul + reduce-scatter collective at Llama-70B TP shapes.

For each token count M in the sweep this measures

    reduce_scatter(a[M, N // tp] @ w[N // tp, K], axis=0)

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_matmul_reduce_scatter.py \
        --m 256,1024,8192
"""

import jax
import jax.numpy as jnp
from collective_bench_lib import AXIS, run
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_inputs(mesh, m, k, n, dtype):
    """a[M, N] and w[N, K] both contraction-sharded over the TP axis."""
    key_a, key_w = jax.random.split(jax.random.key(0), 2)
    a = jax.device_put(
        jax.random.normal(key_a, (m, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    w = jax.device_put(
        jax.random.normal(key_w, (n, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    return jax.block_until_ready((a, w))


def build_xla_baseline(mesh, inputs):
    """Auto-partitioned einsum: XLA inserts the reduce-scatter from the shards."""
    fn = jax.jit(lambda a, w: jnp.einsum("mn,nk->mk", a, w),
                 out_shardings=NamedSharding(mesh, P(AXIS, None)))
    return fn.lower(*inputs).compile()


IMPLEMENTATIONS = {"xla": build_xla_baseline}

if __name__ == "__main__":
    run(make_inputs=make_inputs,
        implementations=IMPLEMENTATIONS,
        default_n=28672)  # N = FFN intermediate dim (the contraction axis)
