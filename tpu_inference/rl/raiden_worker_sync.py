# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Binds a TPUWorker's live model weights to the Raiden transport, in-process.

Runs inside the EngineCore subprocess where the live TPU arrays live; only
plain-data metadata (`RaidenWorkerSync.metadata_dict`) crosses the
`collective_rpc` boundary back to `RLVllmSampler`.

Duplicates (rather than imports) tunix's
`raiden_synchronizer.RaidenSynchronizer` binding mechanics, since
`vllm_sampler.py` has no Tunix dependency. Keep the two in sync by hand;
tunix's `weight_sync.dict_to_metadata` defines the metadata shape.
"""

from __future__ import annotations

import socket
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp

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
        from flax import nnx
        if maxtext:
            inner = getattr(model, "model", None)
            if inner is not None:
                return {"base": nnx.state(inner, nnx.Param)}
            return None
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


def _filter_bindable(names: List[str],
                     arrays: List[Any]) -> Tuple[List[str], List[Any]]:
    """Drops leaves the native layer cannot bind (e.g. RNG-key arrays)."""
    keep_names: List[str] = []
    keep_arrays: List[Any] = []
    for name, arr in zip(names, arrays):
        if _bindable(arr):
            if hasattr(arr, "block_until_ready"):
                arr.block_until_ready()
            keep_names.append(name)
            keep_arrays.append(arr)
    return keep_names, keep_arrays


def _axis_name(axis: Any) -> str:
    if axis is None:
        return ""
    if isinstance(axis, str):
        return axis
    return ",".join(axis)


def _tensor_metadata_dict(name: str, arr: Any, layer_idx: int) -> dict:
    sharding: Any = getattr(arr, "sharding", None)
    spec = tuple(getattr(sharding, "spec", ()) or ())
    spec = (spec + (None, ) * arr.ndim)[:arr.ndim]
    try:
        local = sharding.shard_shape(tuple(arr.shape))
        mesh_shape = tuple(g // s for g, s in zip(arr.shape, local))
    except Exception:  # pylint: disable=broad-exception-caught
        mesh_shape = (1, ) * arr.ndim
    return {
        "name": name,
        "shape": list(arr.shape),
        "mesh_shape": list(mesh_shape),
        "layout": list(reversed(range(arr.ndim))),
        "item_size": arr.dtype.itemsize,
        "layer_idx": layer_idx,
        "sharding_spec": [_axis_name(a) for a in spec],
    }


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

    def h2d(self) -> None:
        # h2d() is async; block so a checksum/read right after sees the
        # transferred data.
        self._require_sync("h2d()").h2d()
        jax.block_until_ready(self.arrays)

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

        def total(arr):
            return float(jnp.sum(jnp.abs(arr).astype(jnp.float32)))

        out = {
            name: total(arr)
            for name, arr in list(zip(self.names, self.arrays))[:sample]
        }
        out["__grand_total__"] = sum(total(arr) for arr in self.arrays)
        # Totals only compare if both sides bound the same tensors.
        out["__tensor_count__"] = len(self.arrays)
        out["__element_count__"] = int(sum(a.size for a in self.arrays))
        return out

    def metadata_dict(self) -> dict:
        """Wire-safe registration metadata, shaped for tunix's
        `weight_sync.dict_to_metadata`."""
        # Positional index, not name-derived -- must match tunix's side.
        variables = [
            _tensor_metadata_dict(name, arr, idx)
            for idx, (name, arr) in enumerate(zip(self.names, self.arrays))
        ]
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
        data_addr = f"{self.ip}:{self._sync.local_port}" if self._sync else ""
        control_addr = (f"{self.ip}:{self._sync.listener_port}"
                        if self._sync and self._sync.listener_port else "")
        num_shards = self._sync.num_shards if self._sync else 1
        return {
            "unit": {
                "job_name":
                self.job_name,
                "job_replica_id":
                str(self.worker_index) if self.worker_index else "",
            },
            "shards": [data_addr] * num_shards if data_addr else [],
            "control_plane_rpc_address": control_addr,
            "mesh_shape": list(mesh_shape),
            "variables": variables,
            "mesh_axes": list(mesh_axes) if mesh_axes else None,
        }
