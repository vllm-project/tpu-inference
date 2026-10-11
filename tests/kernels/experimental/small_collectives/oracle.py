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
"""Inputs and the float64 oracle for the small-collective kernel tests."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

N = 8
AXIS = "d"


def blocks(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32).astype(jnp.bfloat16)


def run_sharded(mesh, body, per_core, ranks):
    """``body(x, rank)`` in a shard_map; ``per_core`` is [N, rows, hidden] and
    ``ranks`` the int32 index handed to each core as data."""
    n, rows, hidden = per_core.shape
    spec = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(jnp.asarray(per_core).reshape(n * rows, hidden), spec)
    rank = jax.device_put(jnp.asarray(ranks, dtype=jnp.int32), spec)
    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS), P(AXIS)),
            out_specs=P(AXIS),
            check_vma=False,
        ))
    out = np.asarray(fn(x, rank))
    return out.reshape(n, out.shape[0] // n, hidden)


def bf16_ulp(magnitude):
    """The bf16 rounding step at each magnitude (8 significand bits)."""
    _, exponent = np.frexp(np.maximum(magnitude, np.finfo(np.float32).tiny))
    return np.ldexp(1.0, exponent - 8)


def sum_tolerance(terms):
    """Bound on |kernel - bf16(exact sum)| over the leading axis of ``terms``.

    The kernel sums in float32 in its own order, then rounds once (or once per
    stage in a 2-stage hierarchical collective). The order costs at most (n - 1)
    float32 roundings of partial sums bounded by the sum of magnitudes; an
    intermediate bf16 stage rounds four 2-term sums (at most 2 bf16 steps); the
    final rounding costs one bf16 step at the result.
    """
    terms = terms.astype(np.float64)
    magnitudes = np.abs(terms).sum(axis=0)
    order_error = (terms.shape[0] - 1) * 2.0**-24 * magnitudes
    stage_error = 2.0 * bf16_ulp(magnitudes)
    return (bf16_ulp(np.abs(terms.sum(axis=0)) + order_error + stage_error) +
            order_error + stage_error)


def exact_sum(terms):
    """The float64 sum, rounded once to bf16, as float64."""
    exact = terms.astype(np.float64).sum(axis=0)
    return exact.astype(np.float32).astype(jnp.bfloat16).astype(np.float64)


def within(got, want, tolerance):
    return np.abs(got.astype(np.float64) - want) <= tolerance
