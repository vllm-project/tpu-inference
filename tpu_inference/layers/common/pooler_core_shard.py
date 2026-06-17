# Copyright 2025 Google LLC
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
"""2-way INTRA-CHIP sharding for the MoE pooler / unpooler on TPU v7x.

The Avocado MoE pooler/unpooler is normally replicated on every core (HBM waste +
full matmul redundantly on every core) or sharded over the full tensor-parallel
(``model``) axis (which on TP16 pays an expensive cross-torus all-gather at decode).

This module provides a third option: shard the pooler/unpooler weight across ONLY
the **2 TensorCores of each tpu7x chip**, exchanging the activation halves over the
~600 GB/s on-chip core-to-core link. Benchmarks (msl-tpu-kernel
``recipes/experimental/pooler-core-shard/RESULTS.md``) show this is a win in BOTH
regimes: at decode (memory-bound) it halves the per-core weight read with a
near-free 2-way intra-chip gather (~2.2x cheaper than the full-TP gather, topology
independent); at prefill (flops-bound) it halves the matmul.

Implementation is a **secondary mesh** over the same devices as the runtime mesh's
``model`` (TP) axis, reshaped to ``(n_chips, 2)`` with the inner axis = the 2
on-chip cores. The main mesh and every existing TP sharding rule are left
untouched, so this is purely additive: nothing changes unless a caller explicitly
builds this mesh and shards a weight on it.

Relies on the tpu7x device order listing the 2 on-chip cores adjacently
(``core_on_chip`` is the minor index; verified on hardware and asserted here), so a
``(n_chips, 2)`` reshape of the ``model``-axis devices maps the inner axis exactly
to each chip's 2-core pair.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Axis names of the secondary intra-chip pooler mesh. Distinct from the main mesh
# axes so the two never collide if composed.
POOL_CHIP_AXIS = "pool_chip"  # inter-chip (n_chips), replicated for the pooler
POOL_CORE_AXIS = "pool_core"  # the 2 cores of one chip — the shard axis


def _model_axis_devices(mesh: Mesh) -> np.ndarray:
    """The devices along the ``model`` (TP) axis of ``mesh``, in axis order.

    Requires every other mesh axis to be size 1 (pure tensor parallelism on the
    ``model`` axis), which is the Avocado T5 TP16 / dp*tp* serving layout. Raises
    otherwise — intra-chip pooler sharding is only defined when ``model`` owns the
    full set of TP cores.
    """
    names = list(mesh.axis_names)
    if "model" not in names:
        raise ValueError(f"mesh has no 'model' axis: {names}")
    arr = np.asarray(mesh.devices, dtype=object)
    moved = np.moveaxis(arr, names.index("model"), 0)
    tp = moved.shape[0]
    flat = moved.reshape(tp, -1)
    if flat.shape[1] != 1:
        raise ValueError(
            "intra-chip pooler sharding requires all non-'model' mesh axes to be "
            f"size 1 (pure TP on 'model'); mesh shape was {dict(mesh.shape)}"
        )
    return flat[:, 0]


def assert_cores_adjacent(devices: np.ndarray) -> int:
    """Assert consecutive device pairs are the 2 cores of one physical chip
    (identical ``coords``, ``core_on_chip`` == {0, 1}). Returns ``n_chips``.

    This is the load-bearing topology fact: it lets a ``(n_chips, 2)`` reshape put
    each chip's 2 cores on the inner axis. It holds for tpu7x single- and
    multi-host slices (``core_on_chip`` minor) but is asserted, not assumed.
    """
    n = len(devices)
    if n % 2 != 0:
        raise ValueError(f"TP size {n} must be even (2 cores/chip on tpu7x)")
    for k in range(0, n, 2):
        d0, d1 = devices[k], devices[k + 1]
        c0, c1 = getattr(d0, "coords", None), getattr(d1, "coords", None)
        k0, k1 = getattr(d0, "core_on_chip", None), getattr(d1, "core_on_chip", None)
        if c0 != c1 or {k0, k1} != {0, 1}:
            raise ValueError(
                "intra-chip pooler sharding needs the 'model' (TP) axis to list a "
                f"chip's 2 cores adjacently (core_on_chip minor); device pair "
                f"{k},{k + 1} has coords {c0}/{c1} core_on_chip {k0}/{k1}. The "
                "device order does not keep on-chip cores adjacent."
            )
    return n // 2


def intrachip_core_mesh(mesh: Mesh) -> Mesh:
    """Build the secondary ``(n_chips, 2)`` intra-chip mesh from ``mesh``'s TP axis.

    Axes: ``(POOL_CHIP_AXIS, POOL_CORE_AXIS)``; ``POOL_CORE_AXIS`` (size 2) = the 2
    cores of each chip. Same physical devices as the runtime mesh's ``model`` axis,
    same order, so a tensor sharded on ``POOL_CORE_AXIS`` lands on the on-chip pair.
    """
    devs = _model_axis_devices(mesh)
    n_chips = assert_cores_adjacent(devs)
    return Mesh(
        np.asarray(devs, dtype=object).reshape(n_chips, 2),
        (POOL_CHIP_AXIS, POOL_CORE_AXIS),
    )


def pooler_weight_sharding(core_mesh: Mesh) -> NamedSharding:
    """Column-parallel sharding for a 2-D pooler/unpooler weight ``[out, in]``:
    shard the output dim over the 2 on-chip cores, replicate across chips. Each
    core stores/reads half the weight."""
    return NamedSharding(core_mesh, P(POOL_CORE_AXIS, None))


def gather_cores(x: jax.Array, core_mesh: Mesh, axis: int = 1) -> jax.Array:
    """All-gather a ``POOL_CORE_AXIS``-sharded activation back to full over the 2
    on-chip cores (the ~600 GB/s core-to-core link), preserving dtype.

    Manual ``shard_map`` ``all_gather`` (not ``with_sharding_constraint``) so XLA
    cannot re-type the collective to the consumer dtype — the bf16 gather stays
    bf16 and any fp32 upcast happens after, in the residual add. ``x`` is the local
    ``[..., shard]`` shard; returns the full ``[..., full]`` tensor replicated
    within each chip.
    """
    ndim = x.ndim
    in_spec = [None] * ndim
    in_spec[axis] = POOL_CORE_AXIS

    def _g(local):
        return jax.lax.all_gather(local, POOL_CORE_AXIS, axis=axis, tiled=True)

    return jax.shard_map(
        _g,
        mesh=core_mesh,
        in_specs=P(*in_spec),
        out_specs=P(*([None] * ndim)),
        check_vma=False,
    )(x)
