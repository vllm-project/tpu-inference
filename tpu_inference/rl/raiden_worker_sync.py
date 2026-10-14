# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Binds a TPUWorker's live model weights to the Raiden transport, in-process.

Runs inside the EngineCore subprocess where the live TPU arrays live; only
plain-data metadata (`RaidenWorkerSync.metadata_dict`) crosses the
`collective_rpc` boundary back to `RLVllmSampler` (in Tunix).

Duplicates (rather than imports) tunix's
`raiden_synchronizer.RaidenSynchronizer` binding mechanics, since
`tpu-inference` has no Tunix dependency. Keep the two in sync by hand;
tunix's `weight_sync.dict_to_metadata` defines the metadata shape.
"""

from __future__ import annotations

import socket
import time
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_ws_lib: Any = None
_RAIDEN_IMPORT_ERROR: Optional[Exception] = None

try:
    from tpu_sync.api.jax import \
        weight_synchronizer as _ws_lib  # pylint: disable=g-import-not-at-top
except ImportError as exc:
    _RAIDEN_IMPORT_ERROR = exc


def local_ip() -> str:
    for family, probe in (
        (socket.AF_INET, ("8.8.8.8", 80)),
        (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
    ):
        try:
            s = socket.socket(family, socket.SOCK_DGRAM)
            try:
                s.connect(probe)
                ip = s.getsockname()[0]
            finally:
                s.close()
            return f"[{ip}]" if ":" in ip else ip
        except OSError:
            continue
    return "localhost"


def is_maxtext_model(model: Any) -> bool:
    """True if `model` is a MaxTextForCausalLM (checked by class name to
    avoid a maxtext dependency)."""
    return type(model).__name__ == "MaxTextForCausalLM"


def extract_weight_state(state: Any, model: Any) -> Any:
    """Returns the Param state ready for Raiden binding/inspection.

    For MaxText, params nest one level down under the wrapper's `model`
    key (not `base`, unlike the trainer side); unwrapped here either way.
    """
    maxtext = is_maxtext_model(model)
    if state is not None:
        if maxtext:
            try:
                return {"base": state["model"]}
            except (KeyError, TypeError):
                pass
        return state
    if model is not None:
        if maxtext:
            inner = getattr(model, "model", None)
            if inner is not None:
                from flax import nnx
                return {"base": nnx.state(inner, nnx.Param)}
            return None
        from flax import nnx
        return nnx.state(model, nnx.Param)
    return None


def flatten_weights(state: Any) -> Tuple[List[str], List[Any]]:
    """Returns (names, arrays) for every array leaf, in stable tree order."""
    names, arrays = [], []
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        arr = getattr(leaf, "value", leaf)
        if hasattr(arr, "shape") and hasattr(arr, "dtype"):
            names.append(jax.tree_util.keystr(path))
            arrays.append(arr)
    return names, arrays


def _bindable(arr: Any) -> bool:
    """True if the native layer can bind this leaf."""
    try:
        if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            return False
        if arr.ndim < 1:
            return False
        if not jnp.issubdtype(arr.dtype, jnp.floating):
            return False
        devices = arr.devices()
        if not devices:
            return False
        return all(getattr(d, "platform", "?") == "tpu" for d in devices)
    except Exception:
        return False


# KV-cache leaves are ordinary float arrays in the weight tree, so _bindable
# accepts them, but the trainer has no counterpart to pair them with.
_NON_WEIGHT_PATH_PARTS = ("['cache']", )


def _is_weight(name: str) -> bool:
    return not any(part in name for part in _NON_WEIGHT_PATH_PARTS)


def _filter_bindable(names: List[str],
                     arrays: List[Any]) -> Tuple[List[str], List[Any]]:
    """Drops leaves the native layer cannot bind, and non-weight state."""
    keep_names: List[str] = []
    keep_arrays: List[Any] = []
    dropped_non_weight = 0
    for name, arr in zip(names, arrays):
        if not _is_weight(name):
            dropped_non_weight += 1
            continue
        if _bindable(arr):
            if hasattr(arr, "block_until_ready"):
                arr.block_until_ready()
            keep_names.append(name)
            keep_arrays.append(arr)
    if dropped_non_weight:
        logger.info("skipped %d non-weight leaves (KV cache) when binding",
                    dropped_non_weight)
    return keep_names, keep_arrays


def _axis_name(axis: Any) -> str:
    if axis is None:
        return ""
    if isinstance(axis, str):
        return axis
    return ",".join(axis)


def _global_shard_index(
    name: str,
    slices: Tuple[slice, ...],
    shape: Tuple[int, ...],
    mesh_shape: Tuple[int, ...],
) -> int:
    """Row-major index of a device's slice in the variable's shard grid.

    The controller enumerates a variable's slices in row-major order over
    `mesh_shape` with tile `shape[i] // mesh_shape[i]` along each dimension
    (`nd_slice_math.compute_nd_shard_slices`); this returns the position of
    `slices` in that enumeration. A replicated dimension has mesh size 1 and
    coordinate 0, so replicated devices share an index.

    Args:
      name: Variable name, for error messages.
      slices: The device's slice per tensor dimension, as returned by
        `Sharding.devices_indices_map`. `start=None` means 0.
      shape: Global tensor shape.
      mesh_shape: The variable's logical mesh shape (shards per dimension).

    Returns:
      The flat row-major shard index in `[0, prod(mesh_shape))`.

    Raises:
      ValueError: If the ranks disagree, or a slice does not start on a tile
        boundary inside the grid (ragged shardings are not representable).
    """
    if not len(slices) == len(shape) == len(mesh_shape):
        raise ValueError(
            f"variable {name!r}: slices {slices}, shape {shape} and mesh_shape"
            f" {mesh_shape} must have the same rank")
    flat_index = 0
    for dim_slice, dim_size, dim_shards in zip(slices, shape, mesh_shape):
        tile = dim_size // dim_shards
        if tile == 0:
            raise ValueError(
                f"variable {name!r}: mesh_shape {mesh_shape} splits shape"
                f" {shape} into more shards than elements")
        start = 0 if dim_slice.start is None else int(dim_slice.start)
        coord = start // tile
        if start % tile != 0 or not 0 <= coord < dim_shards:
            raise ValueError(
                f"variable {name!r}: slice {slices} is not a tile of shape"
                f" {shape} sharded as mesh_shape {mesh_shape}; shard start"
                f" {start} must be a multiple of tile {tile} below"
                f" {dim_shards} tiles")
        flat_index = flat_index * dim_shards + coord
    return flat_index


def _tensor_metadata_dict(name: str, arr: Any, layer_idx: int,
                          shard_devices: List[Any]) -> dict:
    """Describes `arr` for registration.

    Args:
      name: Variable name.
      arr: The bound array.
      layer_idx: Stable batching ordinal.
      shard_devices: The devices behind the work unit's `shards`, in `shards`
        order; `global_shard_indices[j]` describes `shard_devices[j]`. Empty
        when the unit publishes no shards.

    Returns:
      The variable's metadata. `global_shard_indices` is populated only when
      `arr` carries a real sharding; otherwise it is left empty and the
      controller falls back to its own placement inference.
    """
    sharding: Any = getattr(arr, "sharding", None)
    spec = tuple(getattr(sharding, "spec", ()) or ())
    spec = (spec + (None, ) * arr.ndim)[:arr.ndim]
    shape = tuple(arr.shape)
    global_shard_indices: Tuple[int, ...] = ()
    try:
        local = sharding.shard_shape(shape)
        mesh_shape = tuple(g // s for g, s in zip(shape, local))
    except Exception:  # pylint: disable=broad-exception-caught
        mesh_shape = (1, ) * arr.ndim
    else:
        # Which slice of the shard grid each local shard holds. Derived from
        # the array's own sharding rather than from the host's position in
        # the device mesh: with expert parallelism or a non-identity device
        # order, the controller cannot recover it from `mesh_shape` alone.
        devices_indices_map = sharding.devices_indices_map(shape)
        global_shard_indices = tuple(
            _global_shard_index(name, devices_indices_map[d], shape,
                                mesh_shape) for d in shard_devices)
    return {
        "name": name,
        "shape": list(shape),
        "mesh_shape": list(mesh_shape),
        "layout": list(reversed(range(arr.ndim))),
        "item_size": arr.dtype.itemsize,
        "layer_idx": layer_idx,
        "sharding_spec": [_axis_name(a) for a in spec],
        "global_shard_indices": list(global_shard_indices),
    }


def _l1_norm(arrays: List[Any]) -> float:
    """L1 norm over every array, as one stacked device sync.

    Accumulated in float64 on the host: float32 loses the low digits at 35b
    scale, which blinds _wait_until_settled to a small tensor still landing.
    Not bit-comparable against tunix's source-side total, which sums
    sequentially in Python -- compare with a tolerance.
    """
    if not arrays:
        return 0.0
    per_tensor = jnp.stack(
        [jnp.sum(jnp.abs(a), dtype=jnp.float32) for a in arrays])
    return float(np.asarray(per_tensor, dtype=np.float64).sum())


class RaidenWorkerSync:
    """One TPUWorker's weights on the Raiden transport, plus wire-safe metadata.

    In-process counterpart to tunix's `RaidenSynchronizer`. Construct once
    per `TPUWorker`, rebind on every sync round via `bind()`.
    """

    def __init__(
        self,
        job_name: str,
        *,
        worker_index: int = 0,
        parallelism: int = 4,
        bind_ip: Optional[str] = None,
    ):
        self.job_name = job_name
        self.worker_index = worker_index
        self.names: List[str] = []
        self.arrays: List[Any] = []
        self.ip = bind_ip or local_ip()
        self._parallelism = parallelism
        self._sync: Any = None

    @property
    def bound(self) -> bool:
        return bool(self.names)

    def bind(self, state: Any) -> None:
        """Binds (or rebinds after a weight update) this worker's weights."""
        self.names, self.arrays = _filter_bindable(*flatten_weights(state))
        if _ws_lib is None:
            raise RuntimeError(
                f"{self.job_name}: tpu_sync is not importable, cannot bind "
                f"weight_synchronizer. Original error: {_RAIDEN_IMPORT_ERROR}")
        if self._sync is None:
            self._sync = _ws_lib.WeightSynchronizer(
                self.arrays,
                local_port=0,
                parallelism=self._parallelism,
                unsafe_skip_buffer_lock=True,
                listener_port=0,
                bind_ip=None,
                # Ingest each slice as it lands. At the default (False), h2d()
                # unpacks whatever is staged when it is called, installing
                # in-flight tensors torn -- silently, as checksums that are
                # partway between the initial and synced values. Matches how
                # tunix's in-process destination already binds.
                auto_h2d=True,
            )
        else:
            self._sync.bind_weights(self.arrays)

    def _require_sync(self, op: str) -> Any:
        if self._sync is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
        return self._sync

    def h2d(self, uuid: Optional[int] = None) -> None:
        # h2d() is async; block so a checksum/read right after sees the
        # transferred data.
        sync = self._require_sync("h2d()")
        if hasattr(sync, "wait_for_transfer_completion"):
            sync.wait_for_transfer_completion(uuid)
            jax.block_until_ready(self.arrays)
            return
        sync.h2d()
        jax.block_until_ready(self.arrays)
        # `block_until_ready` only orders JAX computations. Raiden's H2D is
        # documented as asynchronous and writes these buffers via DMA outside
        # the JAX graph, so it is not a completion barrier -- without the wait
        # below the rollout can resume generating from half-written weights.
        if envs.RAIDEN_H2D_SETTLE:
            self._wait_until_settled()

    def _wait_until_settled(self,
                            timeout_s: float = 180.0,
                            interval_s: float = 0.5,
                            stable_reads: int = 3) -> None:
        """Blocks until a digest over all bound arrays stops changing.

        Interim stand-in for a real completion signal; drop it once
        `WeightSynchronizer` exposes one (its API has no wait/join today).
        """
        prev = None
        stable = 0
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            cur = _l1_norm(self.arrays)
            if prev is not None and cur == prev:
                stable += 1
                if stable >= stable_reads:
                    return
            else:
                stable = 0
            prev = cur
            time.sleep(interval_s)
        logger.warning("raiden h2d did not settle within %.0fs", timeout_s)

    def metrics(self) -> dict:
        if self._sync is None:
            return {}
        getter = getattr(self._sync, "get_metrics",
                         getattr(self._sync, "metrics", None))
        return getter() if getter is not None else {}

    def checksums(self, sample: int = 3) -> dict:
        """Per-tensor float32 abs-sums for cross-process verification.

        `__grand_total__` covers every bound tensor: a three-tensor sample
        says nothing about how much of the model actually arrived.
        """

        out = {
            name: _l1_norm([arr])
            for name, arr in list(zip(self.names, self.arrays))[:sample]
        }
        out["__grand_total__"] = _l1_norm(self.arrays)
        # Totals only compare if both sides bound the same tensors.
        out["__tensor_count__"] = len(self.arrays)
        out["__element_count__"] = int(sum(a.size for a in self.arrays))
        return out

    def _shard_devices(self) -> List[Any]:
        """Devices behind `shards`, in the order `metadata_dict` lists them.

        Empty when the unit publishes no shards (nothing bound, or the
        transport is not up yet), so the variables' `global_shard_indices`
        stay empty too.

        Raises:
          RuntimeError: If the local shard count disagrees with the native
            synchronizer's `num_shards`.
        """
        if not self.arrays or self._sync is None:
            return []
        # The native layer stages local shard j from `addressable_shards[j]`,
        # and `shards` repeats this host's data address once per such shard.
        devices = [shard.device for shard in self.arrays[0].addressable_shards]
        if len(devices) != self._sync.num_shards:
            raise RuntimeError(
                f"{self.job_name}: {len(devices)} addressable shard(s) on"
                f" {self.names[0]!r} but the native synchronizer reports"
                f" num_shards={self._sync.num_shards}")
        return devices

    def metadata_dict(self) -> dict:
        """Wire-safe registration metadata, shaped for tunix's
        `weight_sync.dict_to_metadata`."""
        mesh_axes: tuple = ()
        mesh_shape = None
        for arr in self.arrays:
            mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
            if mesh is not None:
                mesh_axes = tuple(mesh.axis_names)
                mesh_shape = tuple(mesh.shape[a] for a in mesh.axis_names)
                break
        if mesh_shape is None:
            raise RuntimeError(
                f"{self.job_name}: no bound array carries a sharding mesh; "
                "cannot determine mesh_shape/mesh_axes for registration "
                "metadata.")

        shard_devices = self._shard_devices()
        # Positional index, not name-derived -- must match tunix's side.
        variables = [
            _tensor_metadata_dict(name, arr, idx, shard_devices)
            for idx, (name, arr) in enumerate(zip(self.names, self.arrays))
        ]

        data_addr = f"{self.ip}:{self._sync.local_port}" if self._sync else ""
        control_addr = (f"{self.ip}:{self._sync.listener_port}"
                        if self._sync and self._sync.listener_port else "")
        num_shards = self._sync.num_shards if self._sync else 1
        shards = [data_addr] * num_shards if data_addr else []
        # The controller reads `global_shard_indices[j]` as the slice behind
        # `shards[j]`; a length mismatch would silently misplace every shard.
        if shards:
            for variable in variables:
                if len(variable["global_shard_indices"]) != len(shards):
                    raise RuntimeError(
                        f"{self.job_name}: variable {variable['name']!r}"
                        f" publishes {len(variable['global_shard_indices'])}"
                        f" global_shard_indices for {len(shards)} shards")
        return {
            "unit": {
                "job_name":
                self.job_name,
                "job_replica_id":
                str(self.worker_index) if self.worker_index else "",
            },
            "shards": shards,
            "control_plane_rpc_address": control_addr,
            "mesh_shape": list(mesh_shape),
            "variables": variables,
            "mesh_axes": list(mesh_axes) if mesh_axes else None,
        }
