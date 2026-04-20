# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""TP-selective weight loading for multi-host tpu-inference.  **WIP — see STATUS.**

Problem this solves:
  For pure-TP multi-host runs (e.g. v4-64 = 8 hosts x 4 chips = TP=32) of large
  MoE FP8 checkpoints, each host today reads the full 744 GB into CPU RAM
  before JAX shards it across its 4 local chips, wasting ~31/32 of the bytes
  and OOM-ing CPU RAM.

Strategy:
  - At load-model time, walk the materialized model and build a plan
    `disk_name -> (axis, tp_size, tp_rank)` from each destination
    param's `output_dim` / `input_dim` attribute.
  - **Resize** each TP-aware `param.data` to its local (1/num_hosts) slice
    before load, so CPU RAM usage drops to 1/num_hosts.
  - Replace the weight iterator so each host reads only its slice via
    `safetensors.safe_open(f).get_slice(name)[slice_obj]`.
  - Set `param.is_sharded_weight = True` on the destination param so vLLM's
    downstream `weight_loader` skips its own `.narrow(...)`.
  - Monkey-patch the v2 `_ColumnvLLMParameter` / `RowvLLMParameter` methods to
    honor `is_sharded_weight` (upstream v2 path unconditionally narrows).
  - Monkey-patch `FusedMoE._load_w13` / `_load_w2` to honor `is_sharded_weight`
    so pre-sliced expert weights flow through unchanged.
  - Register each resized param's original full shape so the downstream
    `general_device_put` path can pass `global_shape` into
    `jax.make_array_from_process_local_data` — JAX reconstructs the full
    multi-host array from each process's local slice.

Gated by env `TPU_TP_SELECTIVE_LOAD=1`. Only activates on **pure TP multi-host**
(PP=1, EP=1, JAX process_count>1); other configs stay no-ops because their own
PP/EP filters already handle per-host slicing.

STATUS (2026-04-17): End-to-end framework is implemented:
- loader + iterator + param-resize + registry
- ``general_device_put`` consults the registry via ``lookup_tp_full_shape``
- ``process_linear_weights`` captures ``tp_full_shape`` from the torch
  param.data registry before torch→JAX conversion, and propagates it
  through the ``LinearWeights`` dataclass
- ``shard_linear_weights`` passes it as ``global_shape=`` to
  ``general_device_put``, so JAX reconstructs the full multi-host array

Ready to test on pure-TP multi-host with ``TPU_TP_SELECTIVE_LOAD=1``.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# GCS direct Range-GET helpers (bypass gcsfuse for TP-selective load)
# ---------------------------------------------------------------------------

# Env var: set to "0" to force gcsfuse path even when gcsfuse mount detected.
_GCS_WEIGHT_LOAD_ENV = "TPU_GCS_WEIGHT_LOAD"

_ST_DTYPE_ELEM_SIZE: dict[str, int] = {
    "F32": 4, "F16": 2, "BF16": 2,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

_ST_NP_DTYPE: dict[str, Any] = {
    "F32": np.float32, "F16": np.float16,
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "BOOL": np.bool_,
    # BF16 and F8 have no numpy equivalent; handled specially.
    "BF16": np.uint16,
    "F8_E4M3": np.uint8,
    "F8_E5M2": np.uint8,
}

_gcs_client: Any = None
_gcs_client_lock = threading.Lock()


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is not None:
        return _gcs_client
    with _gcs_client_lock:
        if _gcs_client is None:
            from google.cloud import storage  # type: ignore[import]
            from requests.adapters import HTTPAdapter
            client = storage.Client()
            # Default urllib3 HTTPAdapter has pool_maxsize=10, which serializes
            # our 64-worker ThreadPoolExecutor to 10 effective parallel Range
            # GETs (observed in logs: "Connection pool is full, discarding
            # connection"). Bump to 128 for full utilization.
            adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128,
                                  max_retries=3)
            client._http.mount("https://", adapter)
            client._http.mount("http://", adapter)
            _gcs_client = client
    return _gcs_client


def _find_gcs_mount(file_path: str) -> Optional[tuple[str, str]]:
    """Return (bucket_name, mountpoint) if file_path is under a gcsfuse mount."""
    try:
        with open("/proc/mounts") as mf:
            best_mp = ""
            best_bucket = ""
            for line in mf:
                parts = line.split()
                if len(parts) < 3:
                    continue
                src, mp, fstype = parts[0], parts[1], parts[2]
                if ("gcsfuse" in fstype) and file_path.startswith(mp + "/") \
                        and len(mp) > len(best_mp):
                    best_mp = mp
                    best_bucket = src
            if best_mp:
                return best_bucket, best_mp
    except Exception:  # noqa: BLE001
        pass
    return None


def _parse_st_header(blob: Any) -> tuple[dict, int]:
    """Parse safetensors header via 2 GCS Range GETs.

    Returns (header_dict, data_section_byte_offset).
    """
    header_len_raw = blob.download_as_bytes(start=0, end=7)
    header_len = struct.unpack_from("<Q", header_len_raw)[0]
    header_json = blob.download_as_bytes(start=8, end=8 + header_len - 1)
    return json.loads(header_json), 8 + header_len


def _raw_to_tensor(raw: bytes, dtype_str: str, shape: list[int]) -> torch.Tensor:
    """Convert raw bytes from GCS to a torch.Tensor of the requested shape."""
    np_dtype = _ST_NP_DTYPE[dtype_str]
    arr = np.frombuffer(raw, dtype=np_dtype).copy()  # copy → writable
    t = torch.from_numpy(arr)
    if dtype_str == "BF16":
        t = t.view(torch.bfloat16)
    elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
        t = t.view(torch.float8_e4m3fn)
    elif dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
        t = t.view(torch.float8_e5m2)
    return t.reshape(shape)


def _fetch_tensor(
    blob: Any,
    name: str,
    meta: dict,
    data_offset: int,
    axis: Optional[int],
    tp_size: int,
    tp_rank: int,
) -> tuple[str, torch.Tensor]:
    """Fetch one tensor via GCS Range GET, applying TP slice if needed."""
    dtype_str: str = meta["dtype"]
    shape: list[int] = meta["shape"]
    t_start, t_end = meta["data_offsets"]  # t_end is exclusive (safetensors spec)
    elem_size = _ST_DTYPE_ELEM_SIZE[dtype_str]

    if axis is not None and shape and shape[axis] % tp_size == 0:
        per_rank = shape[axis] // tp_size
        sl_s = tp_rank * per_rank
        sl_e = sl_s + per_rank

        if axis == 0:
            # Rows sl_s:sl_e are contiguous on disk → single Range GET.
            row_elems = 1
            for d in shape[1:]:
                row_elems *= d
            b_start = t_start + sl_s * row_elems * elem_size
            b_end = t_start + sl_e * row_elems * elem_size  # exclusive
            raw = blob.download_as_bytes(
                start=data_offset + b_start,
                end=data_offset + b_end - 1,
            )
            new_shape = [per_rank] + list(shape[1:])
            return name, _raw_to_tensor(raw, dtype_str, new_shape)
        else:
            # Non-contiguous (axis ≥ 1): fetch full tensor, slice in Python.
            raw = blob.download_as_bytes(
                start=data_offset + t_start,
                end=data_offset + t_end - 1,
            )
            t = _raw_to_tensor(raw, dtype_str, shape)
            idx = [slice(None)] * len(shape)
            idx[axis] = slice(sl_s, sl_e)
            return name, t[tuple(idx)].contiguous()

    # Replicated / non-sliced tensor.
    raw = blob.download_as_bytes(
        start=data_offset + t_start,
        end=data_offset + t_end - 1,
    )
    return name, _raw_to_tensor(raw, dtype_str, shape)

# Module-level flag: have we already monkey-patched vLLM?
_PATCHES_APPLIED = False


def is_enabled() -> bool:
    return os.environ.get("TPU_TP_SELECTIVE_LOAD", "0") == "1"


# ---------------------------------------------------------------------------
# Monkey-patch vLLM's v2 Parameter methods to honor is_sharded_weight.
# Upstream's _ColumnvLLMParameter / RowvLLMParameter unconditionally narrow
# the loaded_weight by tp_rank; when our iterator pre-slices, we need them
# to skip that narrow.
# ---------------------------------------------------------------------------
def _apply_vllm_patches() -> None:
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from vllm.model_executor import parameter as vp

    # ---- _ColumnvLLMParameter.load_column_parallel_weight ----
    def _col_load(self, loaded_weight: torch.Tensor) -> None:
        if not getattr(self, "is_sharded_weight", False):
            shard_size = self.data.shape[self.output_dim]
            loaded_weight = loaded_weight.narrow(
                self.output_dim, self.tp_rank * shard_size, shard_size)
        assert self.data.shape == loaded_weight.shape, (
            f"is_sharded_weight load shape mismatch: "
            f"param={tuple(self.data.shape)} loaded={tuple(loaded_weight.shape)}")
        self.data.copy_(loaded_weight)

    # ---- _ColumnvLLMParameter.load_merged_column_weight ----
    def _merged_load(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        shard_offset: int = kwargs["shard_offset"]
        shard_size: int = kwargs["shard_size"]

        if (isinstance(self, (vp.PackedColumnParameter, vp.PackedvLLMParameter))
                and self.packed_dim == self.output_dim):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size)

        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)

        if not getattr(self, "is_sharded_weight", False):
            loaded_weight = loaded_weight.narrow(
                self.output_dim, self.tp_rank * shard_size, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # ---- _ColumnvLLMParameter.load_qkv_weight ----
    def _qkv_load(self, loaded_weight: torch.Tensor, **kwargs) -> None:
        shard_offset: int = kwargs["shard_offset"]
        shard_size: int = kwargs["shard_size"]
        shard_id: str = kwargs["shard_id"]
        num_heads: int = kwargs["num_heads"]

        if (isinstance(self, (vp.PackedColumnParameter, vp.PackedvLLMParameter))
                and self.output_dim == self.packed_dim):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size)

        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)

        if not getattr(self, "is_sharded_weight", False):
            shard_id_int = (self.tp_rank if shard_id == "q"
                            else self.tp_rank // num_heads)
            loaded_weight = loaded_weight.narrow(
                self.output_dim, shard_id_int * shard_size, shard_size)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    # ---- RowvLLMParameter.load_row_parallel_weight ----
    def _row_load(self, loaded_weight: torch.Tensor) -> None:
        if not getattr(self, "is_sharded_weight", False):
            shard_size = self.data.shape[self.input_dim]
            loaded_weight = loaded_weight.narrow(
                self.input_dim, self.tp_rank * shard_size, shard_size)
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    vp._ColumnvLLMParameter.load_column_parallel_weight = _col_load
    vp._ColumnvLLMParameter.load_merged_column_weight = _merged_load
    vp._ColumnvLLMParameter.load_qkv_weight = _qkv_load
    vp.RowvLLMParameter.load_row_parallel_weight = _row_load

    # ---- FusedMoE._load_w13 / _load_w2 ----
    from vllm.model_executor.layers.fused_moe import layer as moe_layer

    _orig_load_w13 = moe_layer.FusedMoE._load_w13
    _orig_load_w2 = moe_layer.FusedMoE._load_w2

    def _patched_load_w13(self, expert_data, shard_dim, shard_id,
                          loaded_weight, tp_rank, load_full=False):
        is_sharded = getattr(self.w13_weight, "is_sharded_weight", False) \
            if hasattr(self, "w13_weight") else False
        if is_sharded and not load_full and loaded_weight.ndim > 0:
            # Pre-sliced: don't narrow by tp_rank.
            if self.moe_config.is_act_and_mul:
                shard_size = expert_data.shape[shard_dim] // 2
            else:
                shard_size = expert_data.shape[shard_dim]
            if shard_id == "w1":
                expert_data = expert_data.narrow(shard_dim, 0, shard_size)
            else:
                expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
            hidden_dim = self._get_hidden_dim(shard_dim, expert_data.ndim)
            expert_data = self._narrow_expert_data_for_padding(
                expert_data, loaded_weight, hidden_dim=hidden_dim)
            expert_data.copy_(loaded_weight)
            return
        _orig_load_w13(self, expert_data=expert_data, shard_dim=shard_dim,
                       shard_id=shard_id, loaded_weight=loaded_weight,
                       tp_rank=tp_rank, load_full=load_full)

    def _patched_load_w2(self, expert_data, shard_dim, loaded_weight, tp_rank,
                         load_full=False):
        is_sharded = getattr(self.w2_weight, "is_sharded_weight", False) \
            if hasattr(self, "w2_weight") else False
        if is_sharded and not load_full and loaded_weight.ndim > 0:
            # Pre-sliced: don't narrow by tp_rank.
            hidden_dim = self._get_hidden_dim(shard_dim, expert_data.ndim)
            expert_data = self._narrow_expert_data_for_padding(
                expert_data, loaded_weight, hidden_dim=hidden_dim)
            expert_data.copy_(loaded_weight)
            return
        _orig_load_w2(self, expert_data=expert_data, shard_dim=shard_dim,
                      loaded_weight=loaded_weight, tp_rank=tp_rank,
                      load_full=load_full)

    moe_layer.FusedMoE._load_w13 = _patched_load_w13
    moe_layer.FusedMoE._load_w2 = _patched_load_w2

    _PATCHES_APPLIED = True
    logger.info("[TP-selective] vLLM v2 parameter + FusedMoE patches applied")


# ---------------------------------------------------------------------------
# Plan building
# ---------------------------------------------------------------------------
def _get_stacked_mapping(model: torch.nn.Module) -> list[tuple]:
    """Return (target_part, source_part, shard_id) mapping rules.

    vLLM models declare this inline in ``load_weights``; no runtime API. We
    detect via class attribute or fall back to a minimal rule set that works
    for the common LLaMA/GLM/DeepSeek MoE families.
    """
    stacked = getattr(type(model), "stacked_params_mapping", None)
    if stacked:
        return list(stacked)
    # Fallback: GLM-4 / LLaMA / DeepSeek-family rules.
    return [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        # DeepSeek MLA
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
    ]


# Registry: maps tensor.data_ptr() -> full (pre-slice) shape tuple.
# Consumed by the monkey-patched general_device_put to pass global_shape into
# jax.make_array_from_process_local_data so JAX reconstructs the full tensor
# across hosts from each host's local slice. Keying by data_ptr (not id()) is
# intentional: param.data can be reassigned during load, and both register and
# lookup see the new tensor's memory address consistently.
_TP_FULL_SHAPES: dict[int, tuple[int, ...]] = {}


def register_tp_full_shape(tensor: torch.Tensor, full_shape: tuple[int, ...]) -> None:
    _TP_FULL_SHAPES[tensor.data_ptr()] = tuple(full_shape)


def lookup_tp_full_shape(t) -> tuple[int, ...] | None:
    # Accept both torch.Tensor (during load) and anything with data_ptr.
    # Pop-on-lookup: each registered tensor is consumed exactly once by
    # the matching general_device_put call. Without the pop the registry
    # would grow unbounded over a full load, and freed tensors' data_ptr
    # values could be recycled by new allocations, returning a stale
    # shape for a coincident pointer.
    try:
        dp = t.data_ptr() if hasattr(t, "data_ptr") else None
    except Exception:  # noqa: BLE001
        return None
    if dp is None:
        return None
    return _TP_FULL_SHAPES.pop(dp, None)


def _resize_param_to_local(param: torch.nn.Parameter, axis: int,
                           tp_size: int, tp_rank: int) -> tuple[int, ...] | None:
    """Resize ``param.data`` to this host's 1/tp_size slice along ``axis``.

    Returns the original full shape for registry purposes, or ``None`` if the
    tensor cannot be evenly split (caller should skip slicing for this param).
    """
    full_shape = tuple(param.data.shape)
    full_axis_size = full_shape[axis]
    if full_axis_size % tp_size != 0:
        return None
    per_host = full_axis_size // tp_size
    new_shape = list(full_shape)
    new_shape[axis] = per_host
    # Allocate the smaller buffer. Keep the same dtype/device.
    new_data = torch.empty(new_shape, dtype=param.data.dtype,
                           device=param.data.device)
    param.data = new_data
    register_tp_full_shape(new_data, full_shape)
    return full_shape


def build_tp_plan(
    model: torch.nn.Module,
    tp_size: int,
    tp_rank: int,
) -> dict[str, tuple[int, int, int]]:
    """Build a ``disk_name -> (axis, tp_size, tp_rank)`` plan AND resize each
    TP-aware ``param.data`` in-place to 1/tp_size on its TP axis.

    Only entries present in the returned dict will be TP-pre-sliced by the
    iterator. Absent entries are loaded in full (norms, scalars, router
    weights, biases, replicated params, FP8 block scales not directly
    attached to TP-aware modules, etc).

    Also flips ``param.is_sharded_weight = True`` on every destination param
    we plan to feed pre-sliced, so vLLM's downstream weight_loader skips its
    own narrow.  Records the original full shape in a registry consumed by
    the monkey-patched ``general_device_put`` (via
    ``lookup_tp_full_shape``) so multi-host ``make_array_from_process_local_data``
    reconstructs the full tensor correctly.
    """
    if tp_size <= 1:
        return {}

    params_dict = dict(model.named_parameters())
    stacked = _get_stacked_mapping(model)
    plan: dict[str, tuple[int, int, int]] = {}
    resized = skipped_non_divisible = 0

    # Build set of param.data_ptr() for modules whose module-level
    # weight_loader methods are incompatible with our 1/coarse_tp resize:
    #   MergedColumnParallelLinear / QKVParallelLinear — narrow by tp_size=32
    #     using output_partition_sizes; breaks with resize-by-coarse_tp (8).
    #   VocabParallelEmbedding — asserts loaded_weight.shape[output_dim]
    #     equals the full org_vocab_size; breaks with pre-sliced tensor.
    # For these, skip TP-selective so the iterator falls back to full-load
    # and vLLM's default narrow path handles them correctly. Memory cost:
    # small (embedding + qkv + gate_up is ~5% of total weights).
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear, QKVParallelLinear)
    from vllm.model_executor.layers.vocab_parallel_embedding import \
        VocabParallelEmbedding
    # Use param full-name identity instead of data_ptr: FP8 create_weights may
    # register param objects under different Python ids than those seen via
    # model.named_parameters(), making data_ptr comparisons unreliable.
    _skip_names: set[str] = set()
    for _mname, _mod in model.named_modules():
        if isinstance(_mod, (MergedColumnParallelLinear, QKVParallelLinear,
                             VocabParallelEmbedding)):
            for _pname, _p in _mod.named_parameters(recurse=False):
                _full = f"{_mname}.{_pname}" if _mname else _pname
                _skip_names.add(_full)
    # kv_b_proj IS sliced by coarse_tp — the MLA absorb-path assert is
    # handled by our VllmMLAAttention.process_weights_after_loading override
    # which infers num_heads_local from the actual sliced shape. See
    # tpu_inference/layers/vllm/custom_ops/mla_attention.py.
    logger.info("[TP-selective] skip_names=%d (merged/qkv/embed params)",
                len(_skip_names))

    def _is_merged_linear(param_name: str) -> bool:
        return param_name in _skip_names

    def _register(disk_name: str, target_param: torch.nn.Parameter) -> None:
        nonlocal resized, skipped_non_divisible
        odim = getattr(target_param, "output_dim", None)
        idim = getattr(target_param, "input_dim", None)
        if odim is None and idim is None:
            return
        # Skip merged Linear params — vLLM weight_loader narrows by tp_size=32
        # which conflicts with our 1/coarse_tp resize. Let default path load
        # these in full.
        if _is_merged_linear(disk_name):
            return
        axis = int(odim if odim is not None else idim)
        if getattr(target_param, "is_sharded_weight", False):
            # Already registered (e.g., multiple disk names map to same fused
            # param).  Plan is keyed on disk name so still record it, but
            # don't resize again.
            plan[disk_name] = (axis, int(tp_size), int(tp_rank))
            return
        full_shape = _resize_param_to_local(target_param, axis, tp_size, tp_rank)
        if full_shape is None:
            skipped_non_divisible += 1
            return
        target_param.is_sharded_weight = True
        plan[disk_name] = (axis, int(tp_size), int(tp_rank))
        resized += 1

    # 1) Plain (non-stacked) params: on-disk name matches a model param name.
    for name, param in params_dict.items():
        _register(name, param)

    # 2) Stacked params: on-disk has split names (q_proj/k_proj/v_proj,
    #    gate_proj/up_proj) that collapse to a fused destination param.
    for target_part, source_part, _shard_id in stacked:
        # Find every model param whose name contains target_part -> record
        # the corresponding on-disk name with source_part instead.
        for tgt_name, tgt_param in params_dict.items():
            if target_part not in tgt_name:
                continue
            # Skip merged Linear params (see _is_merged_linear rationale).
            if _is_merged_linear(tgt_name):
                continue
            disk_name = tgt_name.replace(target_part, source_part)
            if disk_name in plan:
                continue
            odim = getattr(tgt_param, "output_dim", None)
            if odim is None:
                continue
            # Fused column: all three (q/k/v or gate/up) slice on output_dim.
            plan[disk_name] = (int(odim), int(tp_size), int(tp_rank))
            tgt_param.is_sharded_weight = True

    # 3) FusedMoE expert weights: disk has per-expert per-shard tensors
    #    (e.g. mlp.experts.N.gate_proj.weight), destination is a fused 3D
    #    param w13_weight (shape: [E, 2*I, H]) or w2_weight ([E, H, I]) on
    #    the FusedMoE module.  Pure-TP slicing happens on the feature axis
    #    (dim 1 for w13, dim 2 for w2), NOT on the expert axis (dim 0).
    from vllm.model_executor.layers.fused_moe import FusedMoE
    moe_resized = 0
    for mod_name, module in model.named_modules():
        if not isinstance(module, FusedMoE):
            continue
        # w13: output (intermediate*2) on dim 1.
        w13 = getattr(module, "w13_weight", None)
        if w13 is not None and not getattr(w13, "is_sharded_weight", False):
            if _resize_param_to_local(w13, axis=1, tp_size=tp_size,
                                      tp_rank=tp_rank) is not None:
                w13.is_sharded_weight = True
                moe_resized += 1
        # w2: input (intermediate) on dim 2.
        w2 = getattr(module, "w2_weight", None)
        if w2 is not None and not getattr(w2, "is_sharded_weight", False):
            if _resize_param_to_local(w2, axis=2, tp_size=tp_size,
                                      tp_rank=tp_rank) is not None:
                w2.is_sharded_weight = True
                moe_resized += 1
        # FP8 block scales follow the same feature-axis sharding as the
        # weights they scale. Best-effort: skip if shape doesn't divide; the
        # iterator will fall back to full-load for those names.
        for sname, sax in (("w13_weight_scale", 1), ("w2_weight_scale", 2),
                           ("w13_weight_scale_inv", 1),
                           ("w2_weight_scale_inv", 2)):
            sp = getattr(module, sname, None)
            if sp is None or getattr(sp, "is_sharded_weight", False):
                continue
            if sp.dim() < sax + 1:
                continue
            if _resize_param_to_local(sp, axis=sax, tp_size=tp_size,
                                      tp_rank=tp_rank) is not None:
                sp.is_sharded_weight = True
                moe_resized += 1

    logger.info(
        "[TP-selective] build_tp_plan: disk_entries=%d resized_linear=%d "
        "resized_moe=%d skipped_nondivisible=%d",
        len(plan), resized, moe_resized, skipped_non_divisible)
    return plan


# ---------------------------------------------------------------------------
# Iterator wrapper
# ---------------------------------------------------------------------------
_EXPERT_COL_KEYS = ("mlp.experts.", ".gate_proj.weight",
                    ".up_proj.weight")
_EXPERT_ROW_KEYS = ("mlp.experts.", ".down_proj.weight")
_EXPERT_SCALE_KEYS = (".weight_scale", ".weight_scale_inv")


def _expert_axis_for(name: str) -> int | None:
    """Return TP slice axis for a per-expert on-disk tensor, or None."""
    if "mlp.experts." not in name:
        return None
    if name.endswith(".gate_proj.weight") or name.endswith(".up_proj.weight"):
        return 0
    if name.endswith(".down_proj.weight"):
        return 1
    # Per-expert FP8 scales: same axis as the weight they scale.
    for scale_key in _EXPERT_SCALE_KEYS:
        if scale_key in name:
            if ".gate_proj" in name or ".up_proj" in name:
                return 0
            if ".down_proj" in name:
                return 1
    return None


_SHM_DIR_DEFAULT = "/dev/shm/tpu_loader"
_FULL_SHARD_THR_ENV = "TPU_FULL_SHARD_THR"   # 0.0-1.0; 0 disables Plan A (always per-tensor)
_FULL_SHARD_THR_DEFAULT = "0.5"
_CHUNK_SIZE_MB_ENV = "TPU_CHUNK_SIZE_MB"     # per-chunk MiB for parallel download (default 128)
_CHUNK_SIZE_MB_DEFAULT = "128"
_CHUNK_WORKERS_ENV = "TPU_CHUNK_WORKERS"     # concurrent chunk-download workers (default 16)
_CHUNK_WORKERS_DEFAULT = "16"


def _tensor_from_mmap_slice(
    raw_mv: memoryview,
    dtype_str: str,
    shape: list[int],
    axis: Optional[int],
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Build a torch.Tensor from a zero-copy mmap slice + apply TP slicing.

    np.frombuffer(mv, ...) returns a read-only view over the mmap.  We then
    apply TP slicing (axis-0 contiguous / axis≥1 non-contig), and finally
    .copy() once to detach from the mmap before it is unmapped/unlinked.
    """
    np_dtype = _ST_NP_DTYPE[dtype_str]
    arr = np.frombuffer(raw_mv, dtype=np_dtype).reshape(shape)
    if axis is not None and shape and shape[axis] % tp_size == 0:
        per = shape[axis] // tp_size
        idx = [slice(None)] * len(shape)
        idx[axis] = slice(tp_rank * per, (tp_rank + 1) * per)
        arr = arr[tuple(idx)]
    arr = np.array(arr, copy=True, order='C')  # detach from mmap + contig + writable
    t = torch.from_numpy(arr)
    if dtype_str == "BF16":
        t = t.view(torch.bfloat16)
    elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
        t = t.view(torch.float8_e4m3fn)
    elif dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
        t = t.view(torch.float8_e5m2)
    return t


def _stream_shard_via_mmap(
    blob: Any,
    st_file: str,
    header: dict,
    data_offset: int,
    tasks: list[tuple[str, dict, Optional[int]]],
    tp_size: int,
    tp_rank: int,
    shm_dir: str,
    chunk_size_mb: int,
    chunk_workers: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Plan A: single-stream GET → /dev/shm tmpfs → mmap zero-copy → TP slice.

    Empirically, a single `blob.download_to_filename` stream is FASTER than
    `transfer_manager.download_chunks_concurrently(max_workers=16)` on v4 →
    GCS. Likely per-object aggregate egress throttling at GCS (single-stream
    ~200 MB/s, 16 concurrent streams on same object ~135 MB/s). When 8 hosts
    independently download the same object, per-connection bandwidth shrinks
    as concurrency rises, so more threads per host is net negative.

    /dev/shm is tmpfs-on-RAM so download_to_filename is effectively a memory
    write, not disk IO.  mmap zero-copy parses tensors from that file without
    re-reading bytes.  File is unlinked immediately after processing.

    chunk_size_mb / chunk_workers env vars are retained for future tuning but
    currently unused; switch to transfer_manager requires benchmarking.
    """
    del chunk_size_mb, chunk_workers  # currently unused; see docstring
    os.makedirs(shm_dir, exist_ok=True)
    tmp = os.path.join(shm_dir,
                       f"shard_{os.getpid()}_{uuid.uuid4().hex}.st")
    try:
        blob.download_to_filename(tmp, timeout=600)
        with open(tmp, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            try:
                # Use `with memoryview(...)` so all child slice memoryviews
                # (raw_mv) are released before mm.close(); otherwise mm.close()
                # raises BufferError: cannot close exported pointers exist.
                with memoryview(mm) as mv:
                    for name, meta, axis in tasks:
                        dtype_str = meta["dtype"]
                        shape = meta["shape"]
                        t_start, t_end = meta["data_offsets"]
                        raw_mv = mv[data_offset + t_start:data_offset + t_end]
                        tensor = _tensor_from_mmap_slice(
                            raw_mv, dtype_str, shape, axis, tp_size, tp_rank)
                        # Release child slice before next iteration so it
                        # doesn't accumulate as an exported pointer on mv.
                        raw_mv.release()
                        yield name, tensor
            finally:
                mm.close()
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _gcs_sliced_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    plan: dict[str, tuple[int, int, int]],
    tp_size: int,
    tp_rank: int,
    local_expert_ids: Optional[set],
    pp_skip_fn: Optional[Callable],
    bucket_name: str,
    mount_point: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Direct GCS iterator with per-shard routing:

    - **Full-shard mmap** (read_frac ≥ TPU_FULL_SHARD_THR, default 0.5):
      one HTTP GET → /dev/shm → mmap → zero-copy parse.  Best for MoE-heavy
      shards where axis≥1 fallback would have read the whole shard anyway.
    - **Per-tensor Range GET** (read_frac < THR): 64-way parallel Range GETs,
      each fetching only the needed bytes.  Best for sparse shards like
      embedding + norms where TP-sliced axis-0 only needs 1/tp_size.
    """
    from tqdm import tqdm
    from vllm.model_executor.model_loader.ep_weight_filter import \
        should_skip_weight
    from vllm.model_executor.model_loader.weight_utils import (
        _BAR_FORMAT, _natural_sort_key, enable_tqdm)

    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)

    full_shard_thr = float(os.environ.get(_FULL_SHARD_THR_ENV,
                                          _FULL_SHARD_THR_DEFAULT))
    shm_dir = os.environ.get("TPU_LOADER_SHM_DIR", _SHM_DIR_DEFAULT)
    chunk_size_mb = int(os.environ.get(_CHUNK_SIZE_MB_ENV,
                                       _CHUNK_SIZE_MB_DEFAULT))
    chunk_workers = int(os.environ.get(_CHUNK_WORKERS_ENV,
                                       _CHUNK_WORKERS_DEFAULT))
    logger.info(
        "[TP-selective GCS] full_shard_thr=%.2f shm_dir=%s "
        "chunk_size_mb=%d chunk_workers=%d",
        full_shard_thr, shm_dir, chunk_size_mb, chunk_workers)

    loaded = sliced_count = full_count = skipped = 0
    total_full_bytes = 0      # on-disk bytes for all tensors in all shards
    total_read_bytes = 0      # bytes this rank actually reads (sliced-axis0 uses 1/tp; axis≥1 or no-axis = full tensor)
    total_elapsed_s = 0.0
    plan_a_shards = plan_b_shards = 0  # count of shards routed each way

    with ThreadPoolExecutor(max_workers=64) as pool:
        for shard_idx, st_file in enumerate(
                tqdm(sorted_files,
                     desc="Loading safetensors (GCS direct)",
                     disable=not enable_tqdm(use_tqdm_on_load),
                     bar_format=_BAR_FORMAT)):
            shard_start = time.perf_counter()
            rel_path = os.path.relpath(st_file, mount_point)
            blob = bucket.blob(rel_path)

            header, data_offset = _parse_st_header(blob)

            # Collect ordered task list for this shard + per-shard byte accounting.
            tasks: list[tuple[str, dict, Optional[int]]] = []
            shard_full_bytes = 0
            shard_read_bytes = 0
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                if pp_skip_fn is not None and pp_skip_fn(name):
                    skipped += 1
                    continue
                if should_skip_weight(name, local_expert_ids):
                    skipped += 1
                    continue
                entry = plan.get(name)
                if entry is not None:
                    axis: Optional[int] = entry[0]
                else:
                    axis = _expert_axis_for(name)
                tasks.append((name, meta, axis))
                # Byte accounting for this rank.
                t_start, t_end = meta["data_offsets"]
                t_bytes = t_end - t_start
                shard_full_bytes += t_bytes
                shape = meta["shape"]
                if axis == 0 and shape and shape[0] % tp_size == 0:
                    # Contiguous axis-0 slice → 1/tp_size bytes.
                    shard_read_bytes += t_bytes // tp_size
                else:
                    # axis≥1 fallback OR no axis (replicated) → full tensor read.
                    shard_read_bytes += t_bytes

            read_frac = shard_read_bytes / max(shard_full_bytes, 1)
            use_plan_a = read_frac >= full_shard_thr

            if use_plan_a:
                # Plan A: integer-shard download via mmap, single HTTP stream.
                plan_a_shards += 1
                for name, meta, axis in tasks:
                    shape = meta["shape"]
                    if axis is not None and shape and shape[axis] % tp_size == 0:
                        sliced_count += 1
                    else:
                        full_count += 1
                    loaded += 1
                yield from _stream_shard_via_mmap(
                    blob, st_file, header, data_offset, tasks,
                    tp_size, tp_rank, shm_dir,
                    chunk_size_mb, chunk_workers)
            else:
                # Plan B: 64-way parallel per-tensor Range GETs.
                plan_b_shards += 1
                future_map = {
                    pool.submit(
                        _fetch_tensor, blob, name, meta, data_offset,
                        axis, tp_size, tp_rank,
                    ): name
                    for name, meta, axis in tasks
                }
                results: dict[str, torch.Tensor] = {}
                for fut in as_completed(future_map):
                    n, t = fut.result()
                    results[n] = t

                # Yield in stable (header) order so downstream weight_loader
                # sees tensors in a predictable sequence.
                for name, meta, axis in tasks:
                    tensor = results[name]
                    shape = meta["shape"]
                    if axis is not None and shape and shape[axis] % tp_size == 0:
                        sliced_count += 1
                    else:
                        full_count += 1
                    loaded += 1
                    yield name, tensor

            shard_elapsed = time.perf_counter() - shard_start
            total_full_bytes += shard_full_bytes
            total_read_bytes += shard_read_bytes
            total_elapsed_s += shard_elapsed
            rate_mb_s = (shard_read_bytes / 1e6) / max(shard_elapsed, 1e-6)
            route = "A" if use_plan_a else "B"
            # Plan A wire bytes = full shard (one GET); Plan B wire bytes ≈ read_bytes.
            wire_mb = (shard_full_bytes if use_plan_a else shard_read_bytes) // (1 << 20)
            logger.info(
                "[SHARD-TIME GCS route=%s] rank=%d/%d idx=%d file=%s elapsed=%.2fs "
                "full=%dMB read=%dMB wire=%dMB rate=%.1fMB/s",
                route, tp_rank, tp_size, shard_idx, os.path.basename(st_file),
                shard_elapsed,
                shard_full_bytes // (1 << 20),
                shard_read_bytes // (1 << 20),
                wire_mb,
                rate_mb_s)

    logger.info(
        "[TP-selective GCS] loaded=%d (sliced=%d full=%d) skipped=%d tp=%d/%d "
        "| routed A=%d B=%d | TOTAL elapsed=%.1fs full=%dMB read=%dMB "
        "avg_rate=%.1fMB/s | read_frac=%.3f",
        loaded, sliced_count, full_count, skipped, tp_rank, tp_size,
        plan_a_shards, plan_b_shards,
        total_elapsed_s,
        total_full_bytes // (1 << 20),
        total_read_bytes // (1 << 20),
        (total_read_bytes / 1e6) / max(total_elapsed_s, 1e-6),
        total_read_bytes / max(total_full_bytes, 1))


def tp_sliced_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    plan: dict[str, tuple[int, int, int]],
    tp_size: int,
    tp_rank: int,
    local_expert_ids: set[int] | None = None,
    pp_skip_fn: Callable[[str], bool] | None = None,
    safetensors_load_strategy: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate safetensors files yielding per-host TP-sliced tensors.

    Automatically uses direct GCS Range GETs when the weight files live on a
    gcsfuse mount (detected via /proc/mounts).  Set TPU_GCS_WEIGHT_LOAD=0 to
    force the safetensors/gcsfuse fallback path.

    Falls back to full-load for tensors absent from ``plan`` (replicated
    params) and for tensors whose TP axis is not divisible by ``tp_size``.
    """
    # --- GCS fast path ---
    if hf_weights_files and os.environ.get(_GCS_WEIGHT_LOAD_ENV, "1") != "0":
        gcs_info = _find_gcs_mount(hf_weights_files[0])
        if gcs_info is not None:
            bucket_name, mount_point = gcs_info
            logger.info(
                "[TP-selective] GCS direct mode: bucket=%s mount=%s",
                bucket_name, mount_point)
            yield from _gcs_sliced_iterator(
                hf_weights_files, use_tqdm_on_load, plan, tp_size, tp_rank,
                local_expert_ids, pp_skip_fn, bucket_name, mount_point)
            return

    # --- Fallback: safetensors via gcsfuse ---
    logger.info(
        "[TP-selective] gcsfuse path (TPU_GCS_WEIGHT_LOAD defaults to 1; "
        "set =0 to force this fallback)")
    from safetensors import safe_open
    from tqdm import tqdm
    from vllm.model_executor.model_loader.ep_weight_filter import \
        should_skip_weight
    from vllm.model_executor.model_loader.weight_utils import (
        _BAR_FORMAT, _natural_sort_key, enable_tqdm)

    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)

    def _slice_tensor(f, name: str, axis: int) -> torch.Tensor:
        sl = f.get_slice(name)
        shape = sl.get_shape()
        if shape[axis] % tp_size != 0:
            return f.get_tensor(name)
        per_rank = shape[axis] // tp_size
        start = tp_rank * per_rank
        idx = [slice(None)] * len(shape)
        idx[axis] = slice(start, start + per_rank)
        return sl[tuple(idx)]

    def _tensor_bytes(sl) -> int:
        shape = sl.get_shape()
        dtype_str = sl.get_dtype()
        elem = _ST_DTYPE_ELEM_SIZE.get(dtype_str, 0)
        n = 1
        for d in shape:
            n *= d
        return n * elem

    loaded = sliced_count = full_count = skipped = 0
    total_full_bytes = 0     # on-disk size of all loaded tensors
    total_read_bytes = 0     # bytes this rank actually reads (upper bound: axis≥1 assumed full)
    total_elapsed_s = 0.0

    for shard_idx, st_file in enumerate(
            tqdm(sorted_files,
                 desc="Loading safetensors checkpoint shards",
                 disable=not enable_tqdm(use_tqdm_on_load),
                 bar_format=_BAR_FORMAT)):
        shard_start = time.perf_counter()
        shard_full_bytes = 0
        shard_read_bytes = 0
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if pp_skip_fn is not None and pp_skip_fn(name):
                    skipped += 1
                    continue
                if should_skip_weight(name, local_expert_ids):
                    skipped += 1
                    continue

                entry = plan.get(name)
                if entry is None:
                    axis = _expert_axis_for(name)
                else:
                    axis, _ts, _tr = entry
                # Byte accounting: peek tensor shape/dtype without reading.
                sl_meta = f.get_slice(name)
                t_bytes = _tensor_bytes(sl_meta)
                shard_full_bytes += t_bytes
                shape = sl_meta.get_shape()
                if axis is None:
                    shard_read_bytes += t_bytes     # replicated → full
                elif axis == 0 and shape and shape[0] % tp_size == 0:
                    shard_read_bytes += t_bytes // tp_size
                elif shape and shape[axis] % tp_size == 0:
                    # axis≥1: PySafeSlice may do partial read, but upper bound = full.
                    shard_read_bytes += t_bytes
                else:
                    shard_read_bytes += t_bytes     # non-divisible fallback

                if entry is None:
                    if axis is not None:
                        tensor = _slice_tensor(f, name, axis)
                        sliced_count += 1
                    else:
                        tensor = f.get_tensor(name)
                        full_count += 1
                else:
                    tensor = _slice_tensor(f, name, axis)
                    sliced_count += 1

                loaded += 1
                yield name, tensor

        shard_elapsed = time.perf_counter() - shard_start
        total_full_bytes += shard_full_bytes
        total_read_bytes += shard_read_bytes
        total_elapsed_s += shard_elapsed
        rate_mb_s = (shard_read_bytes / 1e6) / max(shard_elapsed, 1e-6)
        logger.info(
            "[SHARD-TIME gcsfuse] rank=%d/%d idx=%d file=%s elapsed=%.2fs "
            "full=%dMB read_ub=%dMB rate_ub=%.1fMB/s",
            tp_rank, tp_size, shard_idx, os.path.basename(st_file),
            shard_elapsed,
            shard_full_bytes // (1 << 20),
            shard_read_bytes // (1 << 20),
            rate_mb_s)

    logger.info(
        "[TP-selective] loaded=%d (sliced=%d, full=%d), skipped=%d, tp=%d/%d "
        "| TOTAL elapsed=%.1fs full=%dMB read_ub=%dMB avg_rate_ub=%.1fMB/s "
        "| read_frac_ub=%.3f",
        loaded, sliced_count, full_count, skipped, tp_rank, tp_size,
        total_elapsed_s,
        total_full_bytes // (1 << 20),
        total_read_bytes // (1 << 20),
        (total_read_bytes / 1e6) / max(total_elapsed_s, 1e-6),
        total_read_bytes / max(total_full_bytes, 1))
