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
"""Benchmark the all-gather + matmul collective at Llama-70B TP shapes.

For each token count M in the sweep this measures

    all_gather(x[M // tp, K], axis=0) @ y[K, N // tp]

Run on a TPU host:
    python scripts/benchmarking/kernels/benchmark_all_gather_matmul.py \
        --m 256,1024,8192
"""

import jax
import jax.numpy as jnp
from collective_bench_lib import AXIS, run
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def make_inputs(mesh, m, k, n, dtype):
    """x[M, K] token-sharded, y[K, N] column-sharded over the TP axis."""
    key_x, key_y = jax.random.split(jax.random.key(0), 2)
    x = jax.device_put(
        jax.random.normal(key_x, (m, k), dtype) * 0.1,
        NamedSharding(mesh, P(AXIS, None)))
    y = jax.device_put(
        jax.random.normal(key_y, (k, n), dtype) * 0.1,
        NamedSharding(mesh, P(None, AXIS)))
    return jax.block_until_ready((x, y))


def build_xla_baseline(mesh, inputs):
    """Auto-partitioned einsum: XLA inserts the all-gather from the shards."""
    fn = jax.jit(lambda x, y: jnp.einsum("mk,kn->mn", x, y),
                 out_shardings=NamedSharding(mesh, P(None, AXIS)))
    return fn.lower(*inputs).compile()


IMPLEMENTATIONS = {"xla": build_xla_baseline}

if __name__ == "__main__":
    run(make_inputs=make_inputs,
        implementations=IMPLEMENTATIONS,
        default_n=57344)  # N = fused gate+up FFN dim (2 x 28672)
