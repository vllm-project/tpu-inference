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
"""KV-cache outputs of the model step must keep their storage sharding.

`model_loader` builds the step jit with NO out_sharding on the kv-cache
output (None = propagate): the caches are donated inputs updated in place,
and each attention flavor produces its cache in the sharding it was
allocated with. Forcing one blanket PartitionSpec that puts
KV_HEAD=('model','expert') on dim 2 is a no-op on DP-attention meshes
(KV_HEAD product == 1 there) but on a TP mesh without DP attention it
reshards every cache at the jit boundary each step. XLA then budgets a
pool-sized resharding buffer plus an un-aliased pool copy PER LAYER --
HLO temporaries at pool scale, invariant to the token bucket, and the
donation of the cache is dead.

The two tests pin both halves on a 2-device 'model' mesh:
  1. With out_shardings=None, an in-place donated cache update keeps the
     input sharding, aliases fully (out == alias), and needs less than one
     pool of temporaries.
  2. Anti-vacuity control: the same program compiled with the old blanket
     spec produces an output sharding DIFFERENT from storage, so test 1's
     sharding assertion is measuring a real degree of freedom.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.sharding import ShardingAxisName

NUM_SLOTS, HEADS, K = 64, 8, 128
POOL_BYTES = NUM_SLOTS * HEADS * K * K * 4  # one f32 state pool


def _mesh():
    if len(jax.devices()) < 2:
        pytest.skip("needs >= 2 devices for a nontrivial 'model' axis")
    devs = np.asarray(jax.devices()[:2]).reshape(1, 1, 1, 1, 2, 1, 1)
    return Mesh(
        devs,
        ("data", "attn_dp", "attn_dp_expert", "expert", "model", "dcp", "pcp"))


def _pool(mesh):
    # A mamba-style recurrent state pool: heads sharded on ATTN_HEAD (dim 1),
    # exactly how initialize_kv_cache allocates it.
    sharding = NamedSharding(mesh,
                             P(None, ShardingAxisName.ATTN_HEAD, None, None))
    return jax.device_put(jnp.zeros((NUM_SLOTS, HEADS, K, K), jnp.float32),
                          sharding)


def _step(pool, update):
    # In-place per-slot update, the shape of every recurrent-state write.
    return pool.at[3].set(update)


def _compile(mesh, out_sharding):
    fn = jax.jit(_step, donate_argnums=(0, ), out_shardings=out_sharding)
    pool = _pool(mesh)
    update = jnp.ones((HEADS, K, K), jnp.float32)
    compiled = fn.lower(pool, update).compile()
    return pool, compiled


def test_unconstrained_output_keeps_storage_sharding_and_aliases():
    if jax.default_backend() == "cpu":
        pytest.skip("memory_analysis needs the TPU backend")
    mesh = _mesh()
    pool, compiled = _compile(mesh, out_sharding=None)

    out_sharding = compiled.output_shardings
    assert out_sharding.is_equivalent_to(
        pool.sharding,
        4), (f"propagated output sharding {out_sharding} != storage sharding "
             f"{pool.sharding}; the step reshards the cache every call")

    stats = compiled.memory_analysis()
    if stats is None:
        pytest.skip("memory_analysis unavailable")
    assert stats.alias_size_in_bytes >= stats.output_size_in_bytes, (
        "donated cache no longer aliases its output in place")
    assert stats.temp_size_in_bytes < POOL_BYTES, (
        f"temporaries {stats.temp_size_in_bytes} >= one pool {POOL_BYTES}")


def test_blanket_kv_head_spec_reshards_the_stored_pool():
    """Anti-vacuity control for the test above: the old blanket spec forces
    an output sharding DIFFERENT from storage, so test 1's "output keeps
    storage sharding" assertion is measuring a real degree of freedom. The
    downstream cost of that reshard (a pool-sized collective plus an
    un-aliased pool copy per layer per step, with no cross-layer reuse) only
    materializes in the full model program where the caches thread through
    opaque attention custom-calls; a single-op program reshards cheaply, so
    this control pins the resharded contract, not the copy bytes."""
    if jax.default_backend() == "cpu":
        pytest.skip("memory_analysis needs the TPU backend")
    mesh = _mesh()
    # The old model_loader contract: KV_HEAD forced onto dim 2 of every
    # cache, regardless of how it is stored.
    blanket = NamedSharding(
        mesh,
        P(ShardingAxisName.BATCH, ShardingAxisName.KV_CONTEXT,
          ShardingAxisName.KV_HEAD))
    pool, compiled = _compile(mesh, out_sharding=blanket)

    assert not compiled.output_shardings.is_equivalent_to(pool.sharding, 4), (
        "the blanket KV_HEAD spec now matches the storage sharding on a "
        "mesh with a nontrivial 'model' axis; if this became intentional, "
        "revisit whether the None out_sharding is still load-bearing")
