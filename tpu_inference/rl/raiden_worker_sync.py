# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Binds a TPUWorker's live model weights to the Raiden transport, in-process.

This runs inside the EngineCore subprocess (where `TPUWorker` and its live
TPU-resident `jax.Array`s actually live), dispatched via
`collective_rpc`/`RLVllmSampler._call_worker_method`. Only plain-data
metadata (see `RaidenWorkerSync.metadata_dict`) crosses back over that RPC
boundary -- never the arrays themselves, which vLLM's msgpack-based RPC
serializer cannot carry anyway (`flax.nnx.statelib.State is not
serializable`), and which would be disconnected host copies rather than the
live device buffers if forced through pickle.

This intentionally duplicates the binding mechanics of tunix's
`tunix.experimental.worker.raiden_synchronizer.RaidenSynchronizer` (native
Raiden wheel calls, `nnx.State` flattening, `TensorMetadata` construction)
rather than importing it: `tpu_inference.rl.vllm_sampler` documents itself as
having no dependency on Tunix, and this code runs in the worker subprocess,
the wrong side of that boundary to relax it. The two copies should be kept in
sync by hand if the binding contract changes; the tunix side is the
canonical registration-metadata *shape* (see `weight_sync.dict_to_metadata`,
which this module's `metadata_dict()` output is built to satisfy) and the
trainer-side `RaidenSynchronizer.work_unit_metadata()` is the reference
implementation this ports from.
"""

from __future__ import annotations

import socket
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp

_ws_lib: Any = None
try:
    from tpu_raiden.api.jax import weight_synchronizer as _ws_lib  # pylint: disable=g-import-not-at-top
except ImportError:
    try:
        from tpu_sync.api.jax import weight_synchronizer as _ws_lib  # pylint: disable=g-import-not-at-top
    except ImportError:
        _ws_lib = None


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
    """True if `model` is `maxtext_vllm_adapter.adapter.MaxTextForCausalLM`.

    Checked by class name rather than an isinstance/import so this module
    (and its callers) never need a maxtext dependency, and so the check
    survives whatever wrapping tpu-inference's model loader does to the
    instance before it reaches `TPUWorker`/`RLVllmSampler` -- see
    `extract_weight_state`.
    """
    return type(model).__name__ == "MaxTextForCausalLM"


def extract_weight_state(state: Any, model: Any) -> Any:
    """Returns the Param state ready for Raiden binding/inspection.

    `model_loader.get_flax_model` sets `TPUModelRunner.state` to the
    *already-split* `nnx.State` of the model wrapper returned by
    `_get_nnx_model` (`ModelInterface(state=..., model=jit_model, ...)`,
    `tpu_runner.py`'s `load_model`: `self.state = model.state; self.model =
    model.model`) -- so `runner.state` is set for every model, and callers
    that check `runner.state is not None` before falling back to
    `nnx.state(runner.model, ...)` will always take the `state` branch. For
    `MaxTextForCausalLM`, that pre-split state nests the actual MaxText
    Transformer's params one level down, at the wrapper's own `model` key
    (`state["model"]`), not under `base` the way the trainer's
    `TunixMaxTextAdapter` wraps it. Both branches need the same unwrap-and-
    rename, so this is a single function called from both.

    Args:
      state: `runner.state` (may be None).
      model: `runner.model` (may be None) -- only used to type-check for
        MaxText; when `state` is None this is also state-extracted directly.

    Returns:
      The best available Param state, or None if neither `state` nor `model`
      is available (callers should fall back to `state_leaves`).
    """
    maxtext = is_maxtext_model(model)
    if state is not None:
        if maxtext:
            try:
                return {"base": state["model"]}
            except Exception:
                pass
        return state
    if model is not None:
        try:
            from flax import nnx
            if maxtext:
                inner = getattr(model, "model", None)
                if inner is not None:
                    return {"base": nnx.state(inner, nnx.Param)}
            return nnx.state(model, nnx.Param)
        except Exception:
            pass
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


def _filter_bindable(
    names: List[str], arrays: List[Any]
) -> Tuple[List[str], List[Any]]:
    """Drops leaves the native layer cannot bind; binding them is undefined
    behavior (observed: random RuntimeError or SIGSEGV on RNG-key arrays)."""
    keep_names: List[str] = []
    keep_arrays: List[Any] = []
    for name, arr in zip(names, arrays):
        if _bindable(arr):
            if hasattr(arr, "block_until_ready"):
                try:
                    arr.block_until_ready()
                except Exception:
                    pass
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
    spec = (spec + (None,) * arr.ndim)[: arr.ndim]
    try:
        local = sharding.shard_shape(tuple(arr.shape))
        mesh_shape = tuple(g // l for g, l in zip(arr.shape, local))
    except Exception:  # pylint: disable=broad-exception-caught
        mesh_shape = (1,) * arr.ndim
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

    In-process counterpart to tunix's `RaidenSynchronizer` -- see module
    docstring for why this is a separate implementation rather than a shared
    import. Meant to be constructed once per `TPUWorker` and reused (rebind
    on every sync round) via `bind()`.
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
            return
        if self._sync is None:
            self._sync = _ws_lib.WeightSynchronizer(
                self.arrays,
                local_port=0,
                parallelism=self._parallelism,
                unsafe_skip_buffer_lock=True,
                listener_port=0,
                bind_ip=None,
            )
        else:
            self._sync.bind_weights(self.arrays)

    def _require_sync(self, op: str) -> Any:
        if self._sync is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
        return self._sync

    def h2d(self) -> None:
        self._require_sync("h2d()").h2d()

    def d2h(self) -> None:
        self._require_sync("d2h()").d2h()

    def metrics(self) -> dict:
        if self._sync is None:
            return {}
        if hasattr(self._sync, "get_metrics"):
            return self._sync.get_metrics()
        if hasattr(self._sync, "metrics"):
            return self._sync.metrics()
        return {}

    def checksums(self, sample: int = 3) -> dict:
        """Per-tensor float32 abs-sums for cross-process verification."""

        def total(arr):
            return float(jnp.sum(jnp.abs(arr).astype(jnp.float32)))

        return {
            name: total(arr) for name, arr in list(zip(self.names, self.arrays))[:sample]
        }

    def metadata_dict(self) -> dict:
        """Wire-safe (plain-data-only) registration metadata.

        Shaped to satisfy `tunix.experimental.orchestrator.weight_sync.dict_to_metadata`
        on the receiving (tunix) side of the RPC boundary.
        """
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
            mesh_axes = ("fsdp",)
            mesh_shape = (1,)
        data_addr = f"{self.ip}:{self._sync.local_port}" if self._sync else ""
        control_addr = (
            f"{self.ip}:{self._sync.listener_port}"
            if self._sync and self._sync.listener_port
            else ""
        )
        num_shards = self._sync.num_shards if self._sync else 1
        return {
            "unit": {
                "job_name": self.job_name,
                "job_replica_id": str(self.worker_index) if self.worker_index else "",
            },
            "shards": [data_addr] * num_shards if data_addr else [],
            "control_plane_rpc_address": control_addr,
            "mesh_shape": list(mesh_shape),
            "variables": variables,
            "mesh_axes": list(mesh_axes) if mesh_axes else None,
        }
