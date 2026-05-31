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
"""Colocated-python data parallelism for tpu-inference on Pathways (v2).

A single Pathways controller process holds `dp_size` engine pairs:

    SchedulerShard[i]   (sidecar / colocated CPU host i)
        • stock vLLM Scheduler  (request queue, KV-cache *manager*, prefix index)
        • CPU-only; never touches TPU devices

    RankExecutor[i]     (controller)
        • full vLLM EngineCore scoped to rank i's TPU chips
          (via device_config.slice, as DisaggExecutor does)
        • we drive only its `model_executor.execute_model`; its own
          internal Scheduler is dead weight

    DPEngineCore        (controller, vLLM EngineCore subclass)
        • routes add_request → least-outstanding shard
        • N driver threads (one per rank ⇒ concurrent colocated calls);
          each runs schedule → execute → update per step
        • funnel queue drained by step()

Per-step protocol (one driver thread):

    sched_out  = sched_shard.schedule(pin)                   # sidecar
    model_out  = rank_exec.execute(sched_out)                # controller TPU
    engine_outs = sched_shard.update_from_output(            # sidecar
        pin, sched_out, model_out)

Two colocated boundary crossings per step. The objects crossing are vLLM
msgspec / dataclass structs (`Request`, `SchedulerOutput`, `ModelRunnerOutput`,
`EngineCoreOutputs`) — colocated_python serializes them via cloudpickle
(see `jax.experimental.colocated_python.serialization`).

Why v2 (Scheduler-only on sidecar, runner stays on controller) before v3
(runner CPU prep also on sidecar): v3 needs a refactor of `TPUModelRunner` to
expose a `prep` path that returns CPU JAX arrays without device_put. v2 proves
the colocated boundary works end-to-end with the existing runner first. See
§§11.3, 11.6 of `colocated_dp_design.md`.

Empirical constraint that motivated v2 (replacing v1, where the sidecar held
a full EngineCore including the TPU model): inside a colocated_python program
on Pathways, `jax.devices()` returns only the colocated CPU devices — TPU
devices are not addressable from inside.  Validated by the user's
`examples/colocated_dp/spike_devices_inside_sidecar.py`.  Hence: model_fn
and KV cache must live on the controller.
"""

import atexit
import copy
import queue
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


# ============================================================================
# Colocated boundary: cloudpickle ↔ uint8 jax.Array blob.
#
# jax.experimental.colocated_python rejects any non-jax.Array leaf in method
# args/returns (see `_get_spec` in `colocated_python/func.py:114`, which
# raises `ValueError("colocated_python only supports jax.Array as input and
# output, but got <type>")`). So every per-step object we want to ship across
# the boundary (`Request`, `SchedulerOutput`, `ModelRunnerOutput`,
# `EngineCoreOutputs`, ...) must be cloudpickled into bytes, wrapped as a
# uint8 jax.Array on the rank's colocated-CPU sharding, and unwrapped on the
# other side.
#
# The blob also doubles as the "pin" that selects the dispatch host —
# colocated_python sees the input array's sharding and routes the call to
# whichever colocated CPU device owns it.
# ============================================================================

import cloudpickle  # noqa: E402  (kept local so import order stays explicit)
import os as _os  # noqa: E402


# Fixed blob sizes — every packed blob pads to exactly this many bytes for
# the method it's used by. colocated_python's executor discovers the output
# shape on the *first* method call and reuses that spec for every subsequent
# call on the same method (see jaxlib `colocated_python_sidecar.cc` RET_CHECK
# `output_spec[i].shape() == outputs[i][j].shape()`). With variable-sized
# cloudpickle outputs the second call always trips that check.
#
# Each colocated method has its OWN input and output spec, so we can use
# different sizes per method: `_TINY_BYTES` for state-query acks/booleans
# (~ns to ship across the boundary), `_BLOB_BYTES` for vLLM data-structure
# payloads (Request / SchedulerOutput / ModelRunnerOutput / EngineCoreOutputs).
#
# (Previous design used a single 16 MB blob for *every* call multiplexed
# through a `call(method_name, args)` dispatcher — has_requests was paying the
# same ~200 ms per-call as schedule. Eliminated.)
#
# Bump `_BLOB_BYTES` via the env var if you hit "blob too large" — e.g. for
# very long prompts (Request size scales with prompt tokens) or huge batches
# (SchedulerOutput scales with batch).
_BLOB_HEADER_BYTES = 8
_TINY_BYTES = 64
_BLOB_BYTES = int(
    _os.environ.get("TPU_COLOCATED_DP_BLOB_BYTES",
                    str(2 * 1024 * 1024)))


# Env vars to copy from controller → sidecar. The sidecar process does NOT
# inherit the controller's environment (colocated_python ships code, not a
# process). Without HF auth tokens, StructuredOutputManager and any other
# code that loads tokenizers from HuggingFace will 401. Without
# JAX_PLATFORMS / VLLM_TPU_USING_PATHWAYS / tpu-inference toggles, the
# sidecar may take wrong code paths.
_SIDECAR_ENV_ALLOWLIST: Tuple[str, ...] = (
    # HF auth + caching
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_HUB_OFFLINE",
    # JAX / Pathways
    "JAX_PLATFORMS",
    "JAX_COORDINATOR_ADDRESS",
    "JAX_NUM_PROCESSES",
    "JAX_PROCESS_ID",
    # vLLM + tpu-inference toggles
    "VLLM_USE_V1",
    "VLLM_TPU_USING_PATHWAYS",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_LOGGING_LEVEL",
    "TPU_COLOCATED_DP",
    "TPU_MULTIPROCESS_DP",
    "TPU_MULTIHOST_BACKEND",
    "TPU_NAME",
    "TPU_ACCELERATOR_TYPE",
    "MODEL_IMPL_TYPE",
    "NEW_MODEL_DESIGN",
    "SKIP_JAX_PRECOMPILE",
    # Misc
    "TOKENIZERS_PARALLELISM",
)


def _capture_sidecar_env() -> Dict[str, str]:
    """Snapshot the allow-listed env vars to ship to the sidecar."""
    return {k: _os.environ[k] for k in _SIDECAR_ENV_ALLOWLIST
            if k in _os.environ}


def _pack(obj: Any, sharding: NamedSharding, size: int) -> jax.Array:
    """Cloudpickle `obj` into a fixed-`size`-byte uint8 jax.Array on `sharding`.

    Layout: 8-byte little-endian length header + payload + zero padding to
    `size`. The fixed shape is required because colocated_python locks each
    method's spec from the first call (see header comment). `size` is chosen
    per-method by the caller — `_TINY_BYTES` for booleans/acks,
    `_BLOB_BYTES` for vLLM payloads.
    """
    try:
        data = cloudpickle.dumps(obj)
    except TypeError as e:
        # The most common failure mode at the colocated boundary is an
        # un-`device_get`'d JAX array or a Sharding/Device object slipping
        # into a payload. Give a clear breadcrumb naming the dataclass
        # field that contained the bad value so the caller can extend
        # `_sanitize` in execute_with_cpu_prep.
        from dataclasses import is_dataclass, fields as _fields
        diag = type(obj).__name__
        if is_dataclass(obj):
            bad = []
            for f in _fields(obj):
                v = getattr(obj, f.name, None)
                try:
                    cloudpickle.dumps(v)
                except Exception:
                    bad.append(f.name)
            diag += f" (unpicklable fields: {bad or '?'})"
        raise TypeError(
            f"_pack: cloudpickle failed on {diag}: {e}") from e
    n = len(data)
    payload_max = size - _BLOB_HEADER_BYTES
    if n > payload_max:
        raise ValueError(
            f"Packed object too large for colocated boundary: {n} bytes > "
            f"{payload_max} max (slot size {size}). For payload methods, "
            f"bump TPU_COLOCATED_DP_BLOB_BYTES (currently {_BLOB_BYTES}). "
            f"For tiny methods, this indicates a logic bug — only small "
            f"primitives should use the tiny slot.")
    buf = np.zeros(size, dtype=np.uint8)
    buf[:_BLOB_HEADER_BYTES] = np.frombuffer(
        np.uint64(n).tobytes(), dtype=np.uint8)
    buf[_BLOB_HEADER_BYTES:_BLOB_HEADER_BYTES + n] = np.frombuffer(
        bytearray(data), dtype=np.uint8)
    return jax.device_put(buf, sharding)


def _pack_tiny(obj: Any, sharding: NamedSharding) -> jax.Array:
    """Pack into `_TINY_BYTES` — used by booleans / ints / acks."""
    return _pack(obj, sharding, _TINY_BYTES)


def _pack_blob(obj: Any, sharding: NamedSharding) -> jax.Array:
    """Pack into `_BLOB_BYTES` — used by Request / SchedulerOutput /
    ModelRunnerOutput / EngineCoreOutputs payloads."""
    return _pack(obj, sharding, _BLOB_BYTES)


# ============================================================================
# Phase 3 — drained EngineCoreOutputs `new_token_ids` ride as a JAX array.
#
# `new_token_ids` is the only bulky field in a drained EngineCoreOutputs that
# we still need to cross every step (the controller's tokenizer uses it).
# Strip it out into a flat, fixed-shape JAX-array buffer and ship that as a
# native pytree leaf. The rest of EngineCoreOutputs (req_id strings,
# finish_reasons, stats — small and irregular) stays in the cloudpickle
# blob. `_extract_drained_tokens` walks drained in a deterministic order
# producing (flat_tokens, offsets); `_reattach_drained_tokens` walks the
# same order to put them back. Both sides assume:
#   - drained is iterated in list order
#   - within each EngineCoreOutputs dict, client_idx keys iterated sorted
#   - within each EngineCoreOutputs, .outputs list iterated in order
# Sized to MAX_DRAIN_TOKENS = max_num_reqs * MAX_DRAIN_TOKENS_PER_REQ;
# if a drain exceeds it (rare — controller drains every step), we fall
# back to leaving new_token_ids in cloudpickle for that drain (signalled
# via `tokens_stripped=False` in the metadata blob).
# ============================================================================

# Generous per-step bound: 4 tokens/req allows spec-decode to grow without
# tripping the fallback. Multiply by max_num_reqs at SchedulerShard init.
_MAX_DRAIN_TOKENS_PER_REQ = 4
_MAX_DRAIN_OUTPUTS = 256  # number of EngineCoreOutput entries across drain


def _extract_drained_tokens(
        drained: List[Dict[int, EngineCoreOutputs]],
        max_tokens: int,
        max_outputs: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Strip `new_token_ids` from every EngineCoreOutput in `drained`.

    Returns ``(flat_tokens, offsets)`` where ``offsets[k]`` is the start
    index in `flat_tokens` for the k-th visited EngineCoreOutput (and
    ``offsets[-1]`` is the total length). Returns ``None`` if either
    cap (`max_tokens` / `max_outputs`) would be exceeded — caller
    should fall back to cloudpickling new_token_ids in place.
    """
    flat: List[int] = []
    offsets: List[int] = [0]
    n_outputs = 0
    for eo_dict in drained:
        for client_idx in sorted(eo_dict.keys()):
            eo = eo_dict[client_idx]
            for output in eo.outputs:
                n_outputs += 1
                if n_outputs > max_outputs:
                    return None
                tokens = output.new_token_ids or []
                if len(flat) + len(tokens) > max_tokens:
                    return None
                flat.extend(tokens)
                offsets.append(len(flat))
                output.new_token_ids = []   # stripped; reattached on controller
    return flat, offsets


def _reattach_drained_tokens(
        drained: List[Dict[int, EngineCoreOutputs]],
        flat_tokens: "np.ndarray",
        offsets: List[int],
) -> None:
    """Inverse of `_extract_drained_tokens` — walk the same order, slice
    `flat_tokens[offsets[i]:offsets[i+1]]` back into each EngineCoreOutput's
    `new_token_ids`."""
    i = 0
    for eo_dict in drained:
        for client_idx in sorted(eo_dict.keys()):
            eo = eo_dict[client_idx]
            for output in eo.outputs:
                start, end = int(offsets[i]), int(offsets[i + 1])
                if start < end:
                    output.new_token_ids = flat_tokens[start:end].tolist()
                # else: leave as the empty list put there by the stripper
                i += 1


def _unpack(blob: jax.Array) -> Any:
    """Inverse of `_pack`. Works for any size — only reads the length header."""
    arr = np.asarray(blob)
    n = int(
        np.frombuffer(arr[:_BLOB_HEADER_BYTES].tobytes(),
                      dtype=np.uint64)[0])
    return cloudpickle.loads(arr[_BLOB_HEADER_BYTES:_BLOB_HEADER_BYTES +
                                 n].tobytes())


# ============================================================================
# v3 boundary payload: sidecar-prepared CPU arrays for controller TPU dispatch
# ============================================================================


from dataclasses import dataclass as _dc, field as _field  # noqa: E402


def _to_cpu_prep(prep_tuple: Tuple, sched_out: Any,
                 input_batch: Any,
                 requests: Optional[Dict[str, Any]] = None) -> "CpuPrep":
    """Convert TPUModelRunner._prepare_inputs's 10-tuple return into a
    `CpuPrep` dataclass for boundary shipping.

    Also pulls a tiny set of fields off `sched_out` that the controller's
    `_sample_from_logits` reads — `SchedulerOutput` itself stays on the
    sidecar, but `total_num_scheduled_tokens` and `num_scheduled_tokens`
    must cross because they're needed by the controller's sampling path.
    """
    (input_ids, input_positions, attn_metadata, sampling_metadata,
     logits_indices, spec_decode_metadata, logits_indices_selector,
     padded_num_reqs, req_ids_dp,
     padded_num_scheduled_tokens_per_dp_rank) = prep_tuple
    return CpuPrep(
        input_ids=input_ids,
        input_positions=input_positions,
        attn_metadata=attn_metadata,
        sampling_metadata=sampling_metadata,
        logits_indices=logits_indices,
        spec_decode_metadata=spec_decode_metadata,
        logits_indices_selector=logits_indices_selector,
        padded_num_reqs=padded_num_reqs,
        req_ids_dp=req_ids_dp,
        padded_num_scheduled_tokens_per_dp_rank=(
            padded_num_scheduled_tokens_per_dp_rank),
        total_num_scheduled_tokens=int(sched_out.total_num_scheduled_tokens),
        num_scheduled_tokens=dict(sched_out.num_scheduled_tokens),
        req_id_to_index=dict(input_batch.req_id_to_index),
        request_states={
            rid: (int(r.num_computed_tokens), int(r.num_tokens))
            for rid, r in (requests or {}).items()
            if rid in input_batch.req_id_to_index
        },
    )


@_dc
class CpuPrep:
    """Sidecar → controller per-step payload (v3 prep-on-sidecar).

    Replaces the v3-collapse `SchedulerOutput` payload. Contents are the
    OUTPUT of `TPUModelRunner._prepare_inputs` running in `cpu_only` mode on
    the sidecar (CPU JAX arrays on a degenerate 1-device CPU mesh; everything
    is a full replica of the global-shape array).

    The controller's `execute_with_cpu_prep` re-shards each field onto its
    TPU mesh via `jax.device_put(np.asarray(field), tpu_named_sharding)`
    before calling `model_fn`.

    Fields mirror `_prepare_inputs`'s return tuple exactly so the
    controller-side reconstruction is mechanical.
    """
    input_ids: Any                          # cpu jax.Array, shape (padded_total_tokens,)
    input_positions: Any                    # cpu jax.Array, same shape
    attn_metadata: Any                      # AttentionMetadata or dict thereof
    sampling_metadata: Any                  # TPUSupportedSamplingMetadata
    logits_indices: Any                     # cpu jax.Array
    spec_decode_metadata: Optional[Any]     # None for v3 MVP (no spec decode)
    logits_indices_selector: Optional[list] # numpy or None
    padded_num_reqs: int
    req_ids_dp: Dict[int, List[str]]
    padded_num_scheduled_tokens_per_dp_rank: int
    # Pulled off SchedulerOutput so the controller's `_sample_from_logits`
    # can build a stub scheduler_output without us shipping the full one.
    total_num_scheduled_tokens: int = 0
    num_scheduled_tokens: Dict[str, int] = _field(default_factory=dict)
    # Snapshot of the sidecar's input_batch req-mapping so the controller's
    # `_sample_from_logits` produces a ModelRunnerOutput with the correct
    # req_ids / req_id_to_index. Without this the controller-side
    # input_batch is empty (we removed `persistent_batch_manager.update_states`
    # from there in v3), num_reqs reads as 0, sampled_token_ids comes back
    # empty, and the sidecar's scheduler.update_from_output then KeyErrors.
    req_id_to_index: Dict[str, int] = _field(default_factory=dict)
    # Snapshot of (num_computed_tokens, num_tokens) per active request —
    # the controller's `_sample_from_logits` reads `self.requests[req_id]`
    # (tpu_runner.py:1345) to compute `request_seq_lens`. The actual
    # CachedRequestState lives on the sidecar; we ship just the two ints
    # and rebuild a stub on the controller before `_execute_with_prepared_inputs`.
    request_states: Dict[str, Tuple[int, int]] = _field(default_factory=dict)


# ============================================================================
# Sidecar half: SchedulerShard. Runs inside a colocated_python_class instance.
# ============================================================================


class SchedulerShard:
    """Holds one vLLM Scheduler on a colocated CPU host.

    `__init__` receives plain Python args — colocated_python's wrapper captures
    them and cloudpickles them inside its `initializer` closure (see
    `colocated_python/obj.py:_make_method._first_call`), so they don't need to
    be jax.Arrays.

    Method-level I/O *must* be jax.Arrays only. We pack/unpack via fixed-size
    uint8 blobs (see header comment). Each method has its own per-method input
    and output shape spec — state queries use `_TINY_BYTES`, payload methods
    use `_BLOB_BYTES`. There is NO single dispatcher method; the colocated
    wrapper forwards each method directly.

    `update_from_output` is fire-and-forget — it pushes the resulting
    `EngineCoreOutputs` onto an internal queue (`_outbox`) and returns only a
    tiny ack. The controller drains the outbox in batches via `drain_outputs`,
    so the slow `EngineCoreOutputs` blob crossing happens at most once per
    drain instead of every step.
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: Any,
                 scheduler_block_size: int, hash_block_size: int,
                 env_vars: Optional[Dict[str, str]] = None,
                 tpu_device_ids: Optional[List[int]] = None):
        # Env vars on the sidecar process do NOT inherit from the controller
        # (colocated_python ships code, not the process). Propagate the
        # critical ones (HF auth, vLLM/tpu-inference toggles, cache locations)
        # before anything that may hit HuggingFace or read envs. Use
        # `setdefault` so a deployment-side value (if any) wins.
        import os as _os
        for k, v in (env_vars or {}).items():
            _os.environ.setdefault(k, v)

        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import init_none_hash
        from vllm.v1.structured_output import StructuredOutputManager

        # StructuredOutputManager loads the tokenizer (via HF) inside its
        # __init__. We don't support structured output in v2 (see design doc
        # §11), so this is dead weight here. We still construct one because
        # vLLM's Scheduler requires it as a constructor arg.
        self._structured_output_manager = StructuredOutputManager(vllm_config)
        SchedCls = vllm_config.scheduler_config.get_scheduler_cls()
        self._scheduler = SchedCls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self._structured_output_manager,
            include_finished_set=False,
            log_stats=False,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )

        # v3: instantiate a CPU-only TPUModelRunner on the sidecar to run
        # CPU prep (`_prepare_inputs`, persistent_batch_manager.update_states,
        # _modify_prev_results, _update_placeholder) right here.  The runner
        # owns its OWN input_batch and persistent_batch_manager — the
        # controller's full runner no longer owns this state.
        #
        # The CPU runner uses a degenerate 1-device CPU mesh; sharded
        # arrays it produces are full-global-shape replicas on the one CPU.
        # The controller re-shards them onto its TPU mesh via `device_put`
        # in `execute_with_cpu_prep`.
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        # `_prepare_inputs` needs self.kv_cache_config — populate from the
        # one the controller computed via profiling and shipped to us.
        # v3: build a CPU mesh whose SHAPE matches this rank's TPU mesh.
        # `tpu_device_ids` is the controller-side TPU device.id list for
        # this rank. We look them up in `jax.devices()` (Pathways exposes
        # the global view from inside the sidecar) and ask colocated_python
        # for the colocated CPU devices — one CPU device per TPU device.
        # The CPU mesh ends up isomorphic to the controller's TPU mesh, so
        # `jax.device_put(cpu_array, tpu_sharding)` later is a direct
        # per-device transfer with no reshape.
        from jax.experimental import colocated_python as _cp
        if tpu_device_ids:
            tpu_id_set = set(int(i) for i in tpu_device_ids)
            tpu_devs_resolved = [d for d in jax.devices()
                                 if int(d.id) in tpu_id_set]
            if len(tpu_devs_resolved) != len(tpu_id_set):
                logger.warning(
                    "SchedulerShard: resolved only %d of %d tpu_device_ids "
                    "from jax.devices(). Falling back to local CPU device.",
                    len(tpu_devs_resolved), len(tpu_id_set))
                cpu_devs_for_runner = jax.local_devices()
            else:
                cpu_devs_for_runner = list(
                    _cp.colocated_cpu_devices(tpu_devs_resolved))
        else:
            # Backward-compat: caller didn't pass ids → degenerate single
            # CPU device. Works only for tp_size=1 per-rank configs.
            cpu_devs_for_runner = jax.local_devices()

        self._cpu_runner = TPUModelRunner(
            vllm_config=vllm_config,
            devices=cpu_devs_for_runner,
            rank=0,
            is_first_rank=True,
            is_last_rank=True,
            cpu_only=True,
        )
        self._cpu_runner.kv_cache_config = kv_cache_config

        # Block hashing happens on controller (in `preprocess_add_request`)
        # so we don't need a hasher here, but the prefix cache may need
        # NONE_HASH initialized in this process for correctness.
        if vllm_config.cache_config.enable_prefix_caching:
            init_none_hash(
                get_hash_fn_by_name(
                    vllm_config.cache_config.prefix_caching_hash_algo))

        # `update_from_output` writes into this queue; `drain_outputs` reads
        # from it. Lets us collapse the EngineCoreOutputs blob crossing from
        # 1-per-step to 1-per-drain (often <1-per-step under batching).
        self._outbox: "queue.Queue[Dict[int, EngineCoreOutputs]]" = (
            queue.Queue())
        # Held between `step` calls — when the controller comes back with
        # the TPU-sample dict, we pair it with the `sched_out` /
        # `padded_num_reqs` / `logits_indices_selector` that produced it
        # for `_postprocess_tpu_sample` and `scheduler.update_from_output`.
        # None of these cross the boundary.
        self._pending_sched_out: Optional[Any] = None
        self._pending_padded_num_reqs: int = 0
        self._pending_logits_indices_selector: Optional[Any] = None

        logger.info("SchedulerShard ready on colocated CPU host (cpu_only "
                    "runner instantiated for v3 prep-on-sidecar)")

    # ---- direct methods: each one has its own input/output spec ----------
    # Naming convention: take `pin` (tiny) for pure queries, `blob` (full
    # _BLOB_BYTES) for payload args. Return `_pack_tiny` for booleans/acks,
    # `_pack_blob` for payloads. All `__error__` propagation is left as
    # ordinary Python exceptions — colocated_python propagates them across.

    # state queries (tiny → tiny)

    def has_requests(self, pin: jax.Array) -> jax.Array:
        return _pack_tiny(bool(self._scheduler.has_requests()), pin.sharding)

    def has_unfinished_requests(self, pin: jax.Array) -> jax.Array:
        return _pack_tiny(bool(self._scheduler.has_unfinished_requests()),
                          pin.sharding)

    def num_unfinished_requests(self, pin: jax.Array) -> jax.Array:
        return _pack_tiny(int(self._scheduler.get_num_unfinished_requests()),
                          pin.sharding)

    def request_counts(self, pin: jax.Array) -> jax.Array:
        return _pack_tiny(
            (len(self._scheduler.running), len(self._scheduler.waiting)),
            pin.sharding)

    def make_stats(self, pin: jax.Array) -> jax.Array:
        # Stats is small but not bounded; use blob to be safe.
        return _pack_blob(self._scheduler.make_stats(), pin.sharding)

    # request lifecycle (blob in → tiny out: fire-and-forget acks)

    def add_request(self, blob: jax.Array) -> jax.Array:
        request, _wave = _unpack(blob)
        self._scheduler.add_request(request)
        return _pack_tiny(True, blob.sharding)

    def finish_requests(self, blob: jax.Array) -> jax.Array:
        request_ids, status = _unpack(blob)
        self._scheduler.finish_requests(request_ids, status)
        return _pack_tiny(True, blob.sharding)

    def update_draft_token_ids(self, blob: jax.Array) -> jax.Array:
        draft_token_ids = _unpack(blob)
        self._scheduler.update_draft_token_ids(draft_token_ids)
        return _pack_tiny(True, blob.sharding)

    # step protocol

    def schedule(self, pin: jax.Array) -> jax.Array:
        # tiny in → blob out (SchedulerOutput).
        return _pack_blob(self._scheduler.schedule(), pin.sharding)

    def get_grammar_bitmask(self, blob: jax.Array) -> jax.Array:
        scheduler_output = _unpack(blob)
        return _pack_blob(
            self._scheduler.get_grammar_bitmask(scheduler_output),
            blob.sharding)

    def update_from_output(self, blob: jax.Array) -> jax.Array:
        """Fire-and-forget: produce EngineCoreOutputs and stash in outbox.
        The controller drains via `drain_outputs` separately, so the
        EngineCoreOutputs blob does NOT cross the boundary on every step."""
        scheduler_output, model_runner_output = _unpack(blob)
        engine_outs = self._scheduler.update_from_output(
            scheduler_output, model_runner_output)
        if engine_outs:
            self._outbox.put(engine_outs)
        return _pack_tiny(True, blob.sharding)

    def drain_outputs(self, pin: jax.Array) -> jax.Array:
        """Return everything queued in `_outbox` (possibly empty). One blob
        crossing per drain, instead of one per step inside update_from_output."""
        batch: List[Dict[int, EngineCoreOutputs]] = []
        while True:
            try:
                batch.append(self._outbox.get_nowait())
            except queue.Empty:
                break
        return _pack_blob(batch, pin.sharding)

    # Note: the v2 helper `_apply_sampled_tokens` was removed —
    # `_cpu_runner._postprocess_tpu_sample` (called from `step()` below)
    # now does both the input_batch token-append writes AND the
    # ModelRunnerOutput construction in one pass.

    # Stable meta values used on every boundary cross — `meta_fields` of
    # registered pytrees become part of the PyTreeDef, so they MUST be
    # the same across all calls for colocated_python's fixed-spec
    # contract. The actual per-call values travel in the metadata blob
    # and are restored on the controller before model_fn.
    _BOUNDARY_PADDED_NUM_REQS = 0     # any constant works; controller overrides
    _BOUNDARY_DO_SAMPLING = True
    _BOUNDARY_LOGPROBS = False

    def _stabilize_attn_md(self, attn_md):
        """Force `padded_num_reqs` to the constant boundary value (the
        actual value rides in the metadata blob)."""
        import dataclasses
        return dataclasses.replace(
            attn_md, padded_num_reqs=self._BOUNDARY_PADDED_NUM_REQS)

    def _stabilize_sampling_md(self, sampling_md, target_size: int):
        """Make TPUSupportedSamplingMetadata's pytree spec invariant across
        calls: pad/populate all data leaves to fixed shapes and force
        meta_fields to constant values. Actual `do_sampling` / `logprobs`
        travel in the metadata blob and are restored on the controller."""
        import dataclasses
        # Always-populated data leaves; zero placeholder if the original
        # was None (greedy batches).
        def _take_or_zero(v, dtype):
            if isinstance(v, jax.Array):
                return self._pad_to(v, target_size)
            return jnp.zeros((target_size, ), dtype=dtype)

        return dataclasses.replace(
            sampling_md,
            temperature=_take_or_zero(getattr(sampling_md, "temperature", None),
                                       jnp.float32),
            top_k=_take_or_zero(getattr(sampling_md, "top_k", None),
                                 jnp.int32),
            top_p=_take_or_zero(getattr(sampling_md, "top_p", None),
                                 jnp.float32),
            # cache_collision_dummy: fixed (2,) shape across all calls
            # (MVP doesn't use logprobs so the "needs_logprobs ⇒ shape (1,)"
            # branch never triggers).
            _cache_collision_dummy=jnp.zeros((2, ), jnp.int32),
            do_sampling=self._BOUNDARY_DO_SAMPLING,
            logprobs=self._BOUNDARY_LOGPROBS,
        )

    def _idle_pytree_cache(self, sharding: NamedSharding) -> Tuple[
            jax.Array, Any, Any, jax.Array]:
        """Pre-allocated zero pytree returned when this step has no work.

        Same pytree structure (treedef + leaf shapes/dtypes) as the
        active-step pytree, so colocated_python's fixed-spec contract
        holds across calls.  Cached so we don't device_put on every
        idle call.
        """
        if not hasattr(self, "_idle_cache"):
            import dataclasses
            from tpu_inference.layers.common.attention_metadata import \
                AttentionMetadata
            from tpu_inference.layers.jax.sample.sampling_metadata import \
                TPUSupportedSamplingMetadata
            r = self._cpu_runner
            max_n, max_t = r.max_num_reqs, r.max_num_tokens
            dp = r.dp_size
            max_blocks = r.max_num_blocks_per_req

            def _z(shape, dtype=jnp.int32):
                return jax.device_put(np.zeros(shape, dtype=dtype), sharding)

            # AttentionMetadata: padded_num_reqs forced to boundary const,
            # mamba_state_indices=None (active also uses None for MVP).
            # block_tables is FLAT 1D `(max_num_reqs * max_num_blocks_per_req,)`
            # in the active case — `_prepare_inputs` slices it out of a flat
            # `device_buffer` via `jnp.split` (utils.py:497) and never
            # reshapes back to 2D. Matching the flat shape avoids
            # `RET_CHECK ... output_spec [256,16] vs tensor.shape [4096]`.
            zero_attn = AttentionMetadata(
                input_positions=_z((max_t, )),
                block_tables=_z((max_n * max_blocks, )),
                seq_lens=_z((max_n, )),
                query_start_loc=_z((max_n + dp, )),
                request_distribution=_z((3 * dp, )),
                mamba_state_indices=None,
                padded_num_reqs=self._BOUNDARY_PADDED_NUM_REQS,
            )
            # Sampling metadata: all leaves populated with zeros, meta
            # fields at boundary constants — matches _stabilize_sampling_md.
            zero_sampling = TPUSupportedSamplingMetadata(
                temperature=_z((max_n, ), jnp.float32),
                top_k=_z((max_n, ), jnp.int32),
                top_p=_z((max_n, ), jnp.float32),
                _cache_collision_dummy=_z((2, ), jnp.int32),
                do_sampling=self._BOUNDARY_DO_SAMPLING,
                logprobs=self._BOUNDARY_LOGPROBS,
            )
            self._idle_cache = (_z((max_t, )),  # input_ids
                                zero_attn,
                                zero_sampling,
                                _z((max_n, )))  # logits_indices
        return self._idle_cache

    def _log_pytree_shapes(self, tag: str, tree: Any) -> None:
        """One-shot shape dump of every leaf in the boundary pytree.
        Logs once per `tag` value (IDLE / ACTIVE) so the active-vs-idle
        spec mismatch becomes trivially diff-able in the deploy log."""
        if not hasattr(self, "_logged_shapes"):
            self._logged_shapes = set()
        if tag in self._logged_shapes:
            return
        self._logged_shapes.add(tag)
        leaves, treedef = jax.tree.flatten(tree)
        logger.warning("BOUNDARY_DUMP %s treedef=%s", tag, treedef)
        for idx, leaf in enumerate(leaves):
            if isinstance(leaf, jax.Array):
                logger.warning("BOUNDARY_DUMP %s leaf[%d] shape=%s dtype=%s",
                               tag, idx, leaf.shape, leaf.dtype)
            else:
                logger.warning("BOUNDARY_DUMP %s leaf[%d] type=%s value=%r",
                               tag, idx, type(leaf).__name__, leaf)

    @staticmethod
    def _canonicalize_outputs(target_sharding: NamedSharding,
                              tree: Any) -> Any:
        """Force every `jax.Array` leaf of `tree` onto `target_sharding`.

        colocated_python compiles a single executable for the return value
        and rejects outputs whose device sets disagree
        (``INVALID_ARGUMENT: Output devices for output N have a different
        set of devices from executable devices``). Our outputs come from
        TWO different `Mesh` objects:

          - cpu_runner's mesh (built by `_init_cpu_mesh` from
            `jax.local_devices()` with axis names matching the TPU mesh)
            — produces input_ids, attn_metadata.*, sampling_metadata.*,
            logits_indices.
          - controller-passed `cpu_sharding` (built from
            `colocated_python.colocated_cpu_devices(tpu_devs)` with
            axis `("x",)`) — used for the blob, drained_tokens,
            drained_offsets, and the idle-cache zero arrays.

        Same underlying devices, different `Mesh` objects → JAX treats
        the shardings as distinct device sets. We `device_put` every
        leaf onto `target_sharding` (the controller's `cpu_sharding`,
        replicated across all the rank's CPU devices) so the executable
        sees one unified device set.
        """
        return jax.tree.map(
            lambda x: jax.device_put(x, target_sharding)
            if isinstance(x, jax.Array) else x,
            tree)

    def _pad_to(self, x: jax.Array, target_size: int) -> jax.Array:
        """Zero-pad 1-D JAX array to `target_size` along axis 0. No-op if
        already at target. Caller guarantees `x.shape[0] <= target_size`."""
        if x.shape[0] == target_size:
            return x
        return jnp.pad(x, [(0, target_size - x.shape[0])])

    def _pad_sampling_metadata(self, sampling_md: Any,
                                target_size: int) -> Any:
        """Pad TPUSupportedSamplingMetadata's per-request fields to
        `target_size` so the pytree shape is invariant across calls."""
        import dataclasses
        updates = {}
        for name in ("temperature", "top_k", "top_p"):
            v = getattr(sampling_md, name, None)
            if v is not None and isinstance(v, jax.Array):
                updates[name] = self._pad_to(v, target_size)
        return dataclasses.replace(sampling_md, **updates) if updates else sampling_md

    def step(self, prev_next_tokens: jax.Array,
             has_prev: jax.Array) -> jax.Array:
        """v3-pure combined step: update + schedule + CPU PREP + drain.

        `prev_next_tokens` arrives as a native `jax.Array` pytree leaf on
        this sidecar's CPU sharding — no cloudpickle of its buffer. The
        controller's `RankExecutor.execute_with_cpu_prep` did a direct
        TPU→sidecar-CPU `jax.device_put` on the model's sample output.

        `has_prev` is a `uint8` scalar JAX array (0 / 1) telling us
        whether `prev_next_tokens` carries real data this call. On the
        very first call there's no prior step to update_from_output for,
        so the controller passes `has_prev=0` + a zero placeholder array.

        Returns the `(cpu_prep, drained)` cloudpickle blob the same way
        as before — those are Python containers + variable-length data,
        still need pack/unpack. Phase 2 of this refactor lifts more of
        them out of cloudpickle.
        """
        if int(np.asarray(has_prev).item()):
            # Build ModelRunnerOutput right here on the sidecar from the
            # JAX-array prev_next_tokens + our own input_batch/requests.
            # `_postprocess_tpu_sample` materializes the JAX array to
            # numpy internally only where it actually needs to index.
            assert self._pending_sched_out is not None, (
                "has_prev=1 but no _pending_sched_out — step() ordering "
                "invariant violated.")
            mr_out = self._cpu_runner._postprocess_tpu_sample(
                next_tokens=prev_next_tokens,
                logprobs=None,  # MVP: no logprobs across boundary yet
                scheduler_output=self._pending_sched_out,
                padded_num_reqs=self._pending_padded_num_reqs,
                logits_indices_selector=self._pending_logits_indices_selector,
            )
            engine_outs = self._scheduler.update_from_output(
                self._pending_sched_out, mr_out)
            if engine_outs:
                self._outbox.put(engine_outs)
            self._pending_sched_out = None
        # Drain outbox.
        drained: List[Dict[int, EngineCoreOutputs]] = []
        while True:
            try:
                drained.append(self._outbox.get_nowait())
            except queue.Empty:
                break
        # All cpu_prep/drained returns ship via `_pack_blob` on this rank's
        # CPU sharding (same as before). We use `prev_next_tokens.sharding`
        # — same sharding, but referenced via the JAX-array arg now that
        # there is no blob_in.
        out_sharding = prev_next_tokens.sharding

        # Phase 3 token buffer sizing — fixed-shape per the boundary
        # contract.  Sized once at SchedulerShard init based on
        # self._cpu_runner.max_num_reqs * _MAX_DRAIN_TOKENS_PER_REQ.
        max_drain_tokens = self._cpu_runner.max_num_reqs * _MAX_DRAIN_TOKENS_PER_REQ
        max_drain_offsets = _MAX_DRAIN_OUTPUTS + 1   # +1 for the leading 0
        zero_tokens_buf = np.zeros(max_drain_tokens, dtype=np.int32)
        zero_offsets_buf = np.zeros(max_drain_offsets, dtype=np.int32)

        def _build_drained_jax(drained_list):
            """Try to strip new_token_ids from drained into JAX arrays.

            Returns ``(tokens_jax, offsets_jax, tokens_stripped: bool)``.
            On overflow falls back to (zero buffers, False) and the caller
            keeps the new_token_ids inside the cloudpickled drained list.
            """
            if not drained_list:
                return (jax.device_put(zero_tokens_buf, out_sharding),
                        jax.device_put(zero_offsets_buf, out_sharding),
                        True)  # no-op strip = succeed
            extracted = _extract_drained_tokens(
                drained_list, max_drain_tokens, _MAX_DRAIN_OUTPUTS)
            if extracted is None:
                logger.warning(
                    "drain exceeded MAX_DRAIN_TOKENS (%d) or "
                    "_MAX_DRAIN_OUTPUTS (%d); falling back to cloudpickle "
                    "of new_token_ids.", max_drain_tokens, _MAX_DRAIN_OUTPUTS)
                return (jax.device_put(zero_tokens_buf, out_sharding),
                        jax.device_put(zero_offsets_buf, out_sharding),
                        False)
            flat, offsets = extracted
            tokens_buf = zero_tokens_buf.copy()
            tokens_buf[:len(flat)] = flat
            offsets_buf = zero_offsets_buf.copy()
            offsets_buf[:len(offsets)] = offsets
            return (jax.device_put(tokens_buf, out_sharding),
                    jax.device_put(offsets_buf, out_sharding),
                    True)

        # Helper: idle return — fixed-shape zero pytree + small metadata blob.
        def _idle_return(drained_list):
            ids_z, attn_z, samp_z, li_z = self._idle_pytree_cache(out_sharding)
            tokens_jax, offsets_jax, stripped = _build_drained_jax(
                drained_list)
            blob = _pack_blob({
                "has_work": False,
                "drained": drained_list,
                "tokens_stripped": stripped,
                "n_drained_outputs": (
                    sum(len(eo.outputs)
                        for eo_dict in drained_list
                        for eo in eo_dict.values()) if stripped else 0),
                # Stub fields so the blob's keys are stable across calls.
                "padded_num_reqs": 0,
                "padded_total_num_scheduled_tokens": 0,
                "logits_indices_selector": None,
                "spec_decode_metadata": None,
            }, out_sharding)
            out = self._canonicalize_outputs(
                out_sharding,
                (ids_z, attn_z, samp_z, li_z, tokens_jax, offsets_jax, blob))
            self._log_pytree_shapes("IDLE", out)
            return out

        # Schedule next step + run CPU prep on this sidecar.
        if not self._scheduler.has_requests():
            return _idle_return(drained)
        sched_out = self._scheduler.schedule()
        # input_batch updates from scheduler output (add/remove/cache state).
        # Must run even when total_num_scheduled_tokens==0 to register
        # finished requests for cleanup — mirrors `_execute_model`.
        self._cpu_runner.persistent_batch_manager.update_states(
            sched_out, self._cpu_runner.get_mrope_input_positions_fn
            if hasattr(self._cpu_runner, "get_mrope_input_positions_fn")
            else None)

        # Empty-schedule short-circuit: the scheduler may return a
        # SchedulerOutput with 0 scheduled tokens (e.g. all running
        # requests are blocked on grammar compilation, or every request
        # finished this step). `_prepare_inputs` asserts > 0, and the
        # controller's `_execute_model` returns EMPTY_MODEL_RUNNER_OUTPUT
        # in this case (tpu_runner.py:959). We do the same on the sidecar:
        # call scheduler.update_from_output(sched_out, EMPTY_) so any
        # finished_req_ids land in EngineCoreOutputs (drained next call),
        # and signal idle to the controller by returning cpu_prep=None.
        if sched_out.total_num_scheduled_tokens == 0:
            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
            engine_outs = self._scheduler.update_from_output(
                sched_out, EMPTY_MODEL_RUNNER_OUTPUT)
            if engine_outs:
                drained.append(engine_outs)
            # No pending TPU step — controller won't send back a model_out.
            self._pending_sched_out = None
            self._pending_padded_num_reqs = 0
            self._pending_logits_indices_selector = None
            return _idle_return(drained)

        # CPU prep — produces CPU JAX arrays for controller's TPU dispatch.
        prep_tuple = self._cpu_runner._prepare_inputs(sched_out)
        (input_ids, _input_positions_unused, attn_metadata, sampling_metadata,
         logits_indices, spec_decode_metadata, logits_indices_selector,
         padded_num_reqs, req_ids_dp,
         padded_num_scheduled_tokens_per_dp_rank) = prep_tuple
        # `_input_positions_unused` is the same as `attn_metadata.input_positions`
        # (build_attn at tpu_runner.py:~1947 sets them identically). We
        # ship only the one inside attn_metadata to avoid double-shipping.

        # Pad all variable-length JAX arrays to MAX sizes so the boundary
        # pytree spec is invariant across bucket choices.
        import dataclasses
        max_n = self._cpu_runner.max_num_reqs
        max_t = self._cpu_runner.max_num_tokens
        # Capture actual meta values BEFORE we stabilize them for the boundary.
        actual_padded_num_reqs = padded_num_reqs
        actual_do_sampling = bool(getattr(sampling_metadata, "do_sampling",
                                           False))
        actual_logprobs = bool(getattr(sampling_metadata, "logprobs", False))

        input_ids = self._pad_to(input_ids, max_t)
        attn_metadata = dataclasses.replace(
            attn_metadata,
            input_positions=self._pad_to(attn_metadata.input_positions, max_t))
        logits_indices = self._pad_to(logits_indices, max_n)
        # Stabilize meta_fields / fully-populate data leaves so the
        # AttentionMetadata + TPUSupportedSamplingMetadata pytree specs
        # are identical across all calls (active and idle).
        attn_metadata = self._stabilize_attn_md(attn_metadata)
        sampling_metadata = self._stabilize_sampling_md(sampling_metadata,
                                                         max_n)

        # Phase 3: pull `new_token_ids` out of drained EngineCoreOutputs
        # into a flat fixed-shape JAX array. Falls back to keeping them
        # in cloudpickle if the per-step buffer caps are exceeded.
        tokens_jax, offsets_jax, stripped = _build_drained_jax(drained)
        n_drained_outputs = (
            sum(len(eo.outputs) for eo_dict in drained for eo in eo_dict.values())
            if stripped else 0)
        # Small metadata blob — only the genuinely-Python bits + drained
        # outputs (with new_token_ids stripped if `tokens_stripped`).
        blob = _pack_blob({
            "has_work": True,
            "drained": drained,
            "tokens_stripped": stripped,
            "n_drained_outputs": n_drained_outputs,
            "padded_num_reqs": actual_padded_num_reqs,
            "padded_total_num_scheduled_tokens":
                _input_positions_unused.shape[0],  # used by controller to slice
            "logits_indices_selector": logits_indices_selector,
            "spec_decode_metadata": spec_decode_metadata,
            # Restore-on-controller meta for the stabilized dataclasses
            # (their meta_fields were forced to constants for the boundary).
            "actual_do_sampling": actual_do_sampling,
            "actual_logprobs": actual_logprobs,
        }, out_sharding)

        self._pending_sched_out = sched_out
        self._pending_padded_num_reqs = padded_num_reqs
        self._pending_logits_indices_selector = logits_indices_selector
        out = self._canonicalize_outputs(
            out_sharding,
            (input_ids, attn_metadata, sampling_metadata, logits_indices,
             tokens_jax, offsets_jax, blob))
        self._log_pytree_shapes("ACTIVE", out)
        return out

    # cache resets / lifecycle

    def reset_prefix_cache(self, blob: jax.Array) -> jax.Array:
        reset_running_requests, reset_connector = _unpack(blob)
        ok = self._scheduler.reset_prefix_cache(
            reset_running_requests=reset_running_requests,
            reset_connector=reset_connector)
        return _pack_tiny(bool(ok), blob.sharding)

    def reset_encoder_cache(self, pin: jax.Array) -> jax.Array:
        self._scheduler.reset_encoder_cache()
        return _pack_tiny(True, pin.sharding)

    def shutdown(self, pin: jax.Array) -> jax.Array:
        try:
            self._scheduler.shutdown()
        except Exception as e:
            logger.warning("scheduler shutdown error: %s", e)
        try:
            self._structured_output_manager.clear_backend()
        except Exception:
            pass
        return _pack_tiny(True, pin.sharding)


# ============================================================================
# Controller-side shard client: packs/unpacks around the single `call(blob)`
# method, exposing a friendly per-operation surface used by DPEngineCore.
# ============================================================================


class _ShardClient:
    """Controller-side wrapper around one `colocated_python_class` shard.

    Calls go DIRECTLY to the per-method sidecar entry points — no method-name
    multiplexer, no oversized blob. Each method picks the right blob size
    (`_pack_tiny` / `_pack_blob`) for its payload. Tiny-output calls drop a
    `pin` array onto the shard's CPU sharding and unpack a small ack;
    payload-input calls pack the args themselves (which both pins the host
    and ships the data).

    `update_from_output` is fire-and-forget (returns only a tiny ack); the
    resulting `EngineCoreOutputs` is accumulated in the sidecar's outbox and
    pulled in batches via `drain_outputs`.
    """

    def __init__(self, shard: Any, cpu_sharding: NamedSharding,
                 max_num_reqs: int):
        self._shard = shard
        self._sharding = cpu_sharding
        # `max_num_reqs` is the rank's runner.max_num_reqs — used to size
        # the per-step `next_tokens` JAX array that crosses the boundary.
        # Fixed-shape is required because colocated_python locks each
        # method's spec from the first call.
        self._max_num_reqs = max_num_reqs
        # Pre-place a tiny pin array on this rank's CPU sharding. Reused for
        # every pure-query call (has_requests, schedule, drain_outputs, …) so
        # we only pay device_put cost once per shard, not once per call.
        self._pin = jax.device_put(np.zeros(_TINY_BYTES, dtype=np.uint8),
                                   cpu_sharding)

    # ---- request-lifecycle ----------------------------------------------

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        # Send the Request payload; ack is tiny and we don't need to wait.
        # block_until_ready is intentionally NOT called — colocated_python
        # serializes per-thread calls on the sidecar, so order is preserved
        # for the next call on this thread.
        self._shard.add_request(_pack_blob((request, request_wave),
                                           self._sharding))

    def finish_requests(self, request_ids: List[str],
                        status: RequestStatus) -> None:
        self._shard.finish_requests(_pack_blob((request_ids, status),
                                               self._sharding))

    def update_draft_token_ids(self, draft_token_ids: Any) -> None:
        self._shard.update_draft_token_ids(_pack_blob(draft_token_ids,
                                                      self._sharding))

    # ---- step protocol --------------------------------------------------

    def schedule(self) -> Any:
        return _unpack(self._shard.schedule(self._pin))

    def get_grammar_bitmask(self, scheduler_output: Any) -> Any:
        return _unpack(
            self._shard.get_grammar_bitmask(
                _pack_blob(scheduler_output, self._sharding)))

    def update_from_output_fire(
            self, scheduler_output: Any,
            model_runner_output: ModelRunnerOutput) -> None:
        """Fire-and-forget: ship (sched_out, model_out) to the sidecar; the
        resulting EngineCoreOutputs is enqueued on the sidecar's outbox and
        pulled by `drain_outputs`. Returns immediately without unpacking the
        ack — overlaps with the next loop iteration."""
        self._shard.update_from_output(
            _pack_blob((scheduler_output, model_runner_output),
                       self._sharding))

    def drain_outputs(self) -> List[Dict[int, EngineCoreOutputs]]:
        """Pull whatever's accumulated in the sidecar's outbox. May be []."""
        return _unpack(self._shard.drain_outputs(self._pin))

    def step(
        self,
        prev_next_tokens: jax.Array,
        has_prev: jax.Array,
    ) -> Tuple[jax.Array, Any, Any, jax.Array, Dict[str, Any]]:
        """v3-pure per-step boundary call.

        Returns a pytree of:
          - input_ids: jax.Array (max_num_tokens,) — direct leaf
          - attn_metadata: AttentionMetadata pytree — JAX-array leaves
            travel natively
          - sampling_metadata: TPUSupportedSamplingMetadata pytree
          - logits_indices: jax.Array (max_num_reqs,)
          - metadata: dict from unpacked tiny blob — contains:
              has_work, drained (with new_token_ids reattached from
              the JAX leaves below), padded_num_reqs,
              padded_total_num_scheduled_tokens, logits_indices_selector,
              spec_decode_metadata

        Phase 3: each drained EngineCoreOutput's `new_token_ids` ships
        as a flat JAX-array leaf (`drained_tokens` / `drained_offsets`)
        and is reattached here before the metadata dict is returned. If
        the per-step caps overflowed the sidecar leaves them in the
        cloudpickle (signalled by `meta["tokens_stripped"] == False`).
        """
        ret = self._shard.step(prev_next_tokens, has_prev)
        (input_ids, attn_md, sampling_md, logits_indices,
         drained_tokens_jax, drained_offsets_jax, blob) = ret
        meta = _unpack(blob)
        if meta.get("tokens_stripped", False):
            n_outputs = int(meta.get("n_drained_outputs", 0))
            if n_outputs > 0:
                # device_get the two JAX arrays — small and already on
                # this rank's CPU sharding (so it's a local materialize,
                # not a cross-host fetch).
                tokens_np = np.asarray(drained_tokens_jax)
                offsets_np = np.asarray(drained_offsets_jax)
                _reattach_drained_tokens(
                    meta["drained"], tokens_np,
                    offsets_np[:n_outputs + 1].tolist())
        return input_ids, attn_md, sampling_md, logits_indices, meta

    def first_call_state(self) -> Tuple[jax.Array, jax.Array]:
        """Initial `(prev_next_tokens, has_prev)` for the driver loop's
        first iteration. Zero-valued, with has_prev=0 so the sidecar
        skips the update_from_output branch."""
        if not hasattr(self, "_zero_state_cache"):
            zero_tokens = jax.device_put(
                np.zeros(self._max_num_reqs, dtype=np.int32), self._sharding)
            has_prev_false = jax.device_put(np.uint8(0), self._sharding)
            self._zero_state_cache = (zero_tokens, has_prev_false)
        return self._zero_state_cache

    def _has_prev_true_cache(self) -> jax.Array:
        """Cached `has_prev=1` JAX scalar on this rank's CPU sharding —
        sent on every iteration after the first."""
        if not hasattr(self, "_has_prev_true"):
            self._has_prev_true = jax.device_put(np.uint8(1), self._sharding)
        return self._has_prev_true

    # ---- state queries (tiny in, tiny out — fast) -----------------------

    def has_requests(self) -> bool:
        return bool(_unpack(self._shard.has_requests(self._pin)))

    def has_unfinished_requests(self) -> bool:
        return bool(_unpack(self._shard.has_unfinished_requests(self._pin)))

    def get_num_unfinished_requests(self) -> int:
        return int(_unpack(self._shard.num_unfinished_requests(self._pin)))

    def request_counts(self) -> Tuple[int, int]:
        return tuple(_unpack(self._shard.request_counts(self._pin)))  # type: ignore[return-value]

    def make_stats(self) -> Any:
        return _unpack(self._shard.make_stats(self._pin))

    # ---- cache resets ---------------------------------------------------

    def reset_prefix_cache(self, reset_running_requests: bool,
                           reset_connector: bool) -> bool:
        return bool(
            _unpack(
                self._shard.reset_prefix_cache(
                    _pack_blob((reset_running_requests, reset_connector),
                               self._sharding))))

    def reset_encoder_cache(self) -> None:
        self._shard.reset_encoder_cache(self._pin)

    def shutdown(self) -> None:
        try:
            jax.block_until_ready(self._shard.shutdown(self._pin))
        except Exception as e:
            logger.warning("ShardClient.shutdown error: %s", e)


# ============================================================================
# Controller half: RankExecutor + DPEngineCore.
# ============================================================================


def _make_per_rank_config(base: VllmConfig) -> VllmConfig:
    """Clone `base` and collapse it to a single-rank (dp=1) config.

    No device handles are attached — this config is safe to cloudpickle and
    ship to the sidecar (`jaxlib._jax.Device` objects are not picklable).
    The controller-side `RankExecutor` re-wraps a *copy* with
    `device_config.slice` set for `DisaggExecutor`; see `_attach_slice`.
    """
    cfg = copy.deepcopy(base)
    pc = cfg.parallel_config
    pc.data_parallel_size = 1
    pc.data_parallel_rank = 0
    pc.data_parallel_size_local = 1
    from tpu_inference.layers.common.sharding import ShardingConfigManager
    cfg.sharding_config = ShardingConfigManager.from_vllm_config(cfg)
    return cfg


def _attach_slice(cfg: VllmConfig, rank_idx: int,
                  tpu_devices: List[jax.Device],
                  all_tpu_devices: List[jax.Device]) -> VllmConfig:
    """Return a copy of `cfg` with `device_config.slice` set for
    `DisaggExecutor._init_executor` (controller-side use only).

    DisaggExecutor's slice format: `(rank_idx, sizes_list, all_devices)`. It
    slices `all_devices[sum(sizes[:idx]):sum(sizes[:idx])+sizes[idx]]`. Build
    sizes so the i-th slice corresponds to rank i's contiguous chunk of
    `all_tpu_devices`.

    NOTE: the returned config contains live `jaxlib._jax.Device` handles in
    `device_config.slice` and is therefore NOT picklable. Keep it on the
    controller only.
    """
    out = copy.deepcopy(cfg)
    per_rank_count = len(tpu_devices)
    num_ranks_total = len(all_tpu_devices) // per_rank_count
    sizes = [per_rank_count] * num_ranks_total
    setattr(out.device_config, "slice", (rank_idx, sizes, all_tpu_devices))
    return out


class RankExecutor:
    """One per-rank TPU executor on the controller.

    Wraps a full vLLM `EngineCore` scoped to rank-i's TPU chips. Its
    `engine.scheduler` is unused (the live scheduler lives on the sidecar) but
    is constructed by `EngineCore.__init__` and we tolerate the small overhead.

    We use this engine for:
      - model load + KV-cache memory profiling (gives us `kv_cache_config` /
        `scheduler_block_size` / `hash_block_size` to ship to the sidecar)
      - per-step `model_executor.execute_model` / `sample_tokens`
    """

    def __init__(self, base_vllm_config: VllmConfig, rank_idx: int,
                 tpu_devices: List[jax.Device],
                 all_tpu_devices: List[jax.Device], log_stats: bool,
                 sidecar_cpu_sharding: Optional[NamedSharding] = None):
        # `sidecar_cpu_sharding` is the rank's colocated-CPU NamedSharding
        # that the sidecar uses for its CPU mesh. Used by
        # `execute_with_cpu_prep` to `jax.device_put` next_tokens / logprobs
        # from TPU directly to the sidecar's local CPU mesh, so they ride
        # back to the sidecar as native JAX-array pytree leaves (no
        # cloudpickle).
        self.sidecar_cpu_sharding = sidecar_cpu_sharding
        # Build the controller-side config: dp=1, slice attached so
        # DisaggExecutor can scope its TPUWorker.
        ctrl_cfg = _attach_slice(_make_per_rank_config(base_vllm_config),
                                 rank_idx, tpu_devices, all_tpu_devices)
        self.rank_idx = rank_idx
        self.vllm_config = ctrl_cfg
        # vLLMEngineCore.__init__ calls model_executor's init_device /
        # load_model / KV profile and *mutates* the cfg in place — sets
        # cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        # (core.py:273), updates cache_config.block_size, may reduce
        # model_config.max_model_len. The sidecar Scheduler needs these
        # post-init values (it asserts num_gpu_blocks > 0).
        from tpu_inference.core.disagg_executor import DisaggExecutor
        self.engine = vLLMEngineCore(
            vllm_config=ctrl_cfg,
            executor_class=DisaggExecutor,
            log_stats=log_stats,
        )

        # The slice attr held live jaxlib._jax.Device handles; DisaggExecutor
        # consumed it during _init_executor and is done with it. Strip it
        # now so anything else that walks the cfg post-init doesn't trip on
        # non-picklable device handles. (DisaggExecutor advanced it to
        # `(rank_idx+1, sizes, jax_devices)` — still device-bearing, still
        # unused after init.)
        try:
            delattr(ctrl_cfg.device_config, "slice")
        except AttributeError:
            pass
        # ---- build the sidecar's vllm_config WITHOUT deepcopying ctrl_cfg.
        # vLLM's model_executor.init/load_model attaches assorted torchax
        # /meta tensors onto sub-configs (e.g. an empty-storage scalar from
        # a registered constant); torch.Tensor.__deepcopy__ then fails with
        #   "setStorage: ... storage size of 4 are out of bounds for
        #    storage of size 0"
        # Rebuild from the clean `base_vllm_config` instead, then patch in
        # the small set of fields EngineCore.__init__ mutates (see
        # vllm/v1/engine/core.py:_initialize_kv_caches: num_gpu_blocks,
        # block_size, possibly max_model_len).  The sidecar Scheduler reads
        # exactly those — see Scheduler.__init__ in vllm.
        sidecar_cfg = _make_per_rank_config(base_vllm_config)
        # Mirror the post-init mutations on the clean sidecar copy.
        sidecar_cfg.cache_config.num_gpu_blocks = (
            ctrl_cfg.cache_config.num_gpu_blocks)
        sidecar_cfg.cache_config.block_size = (
            ctrl_cfg.cache_config.block_size)
        sidecar_cfg.model_config.max_model_len = (
            ctrl_cfg.model_config.max_model_len)
        self.sidecar_vllm_config = sidecar_cfg
        # Extract the bits the sidecar Scheduler needs.
        sched = self.engine.scheduler
        # The Scheduler stores its construction args as attributes; vLLM names
        # vary by version. We grab them by attribute and fall back to recomputing.
        self.kv_cache_config = getattr(sched, "kv_cache_config", None)
        self.scheduler_block_size = getattr(sched, "block_size", None)
        self.hash_block_size = getattr(sched, "hash_block_size",
                                       self.scheduler_block_size)
        if self.kv_cache_config is None:
            raise RuntimeError(
                f"RankExecutor[{rank_idx}]: could not extract kv_cache_config "
                f"from the engine's scheduler ({type(sched).__name__}). The "
                f"vLLM Scheduler attribute layout may have changed.")
        logger.info(
            "RankExecutor[%d] ready (num_blocks=%d, block_size=%d, "
            "tpu_devices=%d)", rank_idx, self.kv_cache_config.num_blocks,
            self.scheduler_block_size, len(tpu_devices))

    # ---- per-step TPU compute (called from controller driver thread) ------
    #
    # A SINGLE process-wide lock around the Python-side dispatch path
    # (execute_model / sample_tokens). Required because vLLM's
    # `set_forward_context` (used by every attention layer to find the KV
    # cache) is a non-thread-local global. When two driver threads trace
    # `model_fn` concurrently — which happens on the first call per shape, or
    # on every call when SKIP_JAX_PRECOMPILE=1 — rank 1's trace captures rank
    # 0's KV-cache tracer through that global, producing the
    # `UnexpectedTracerError: ... bfloat16[180132,64,8,2,128] ... escape the
    # scope of the transformation` reported from
    # `unified_attention_with_output` (attention.py:751).
    #
    # We hold the lock only across the dispatch calls themselves. The
    # `future.result()` wait is *outside* the lock so the actual TPU compute
    # of multiple ranks can still overlap on the accelerators; only the
    # Python tracing/dispatch is serialized.
    _DISPATCH_LOCK: "threading.Lock" = threading.Lock()

    def execute(self, scheduler_output: Any) -> ModelRunnerOutput:
        """Run the model on TPU for one scheduler output. Mirrors the
        non-batch-queue path of `vLLMEngineCore.step` after the schedule call.

        v3 NOTE: this path is the v3-collapse fallback — controller still does
        prep. Used only as a safety net if cpu_prep cannot be reconstructed
        (e.g. unsupported sched_out shape). `execute_with_cpu_prep` is the
        preferred v3 entry point.
        """
        with RankExecutor._DISPATCH_LOCK:
            future = self.engine.model_executor.execute_model(
                scheduler_output, non_block=True)
        model_output = future.result()              # outside lock → TPU overlap
        if model_output is None:
            # Sampling deferred to a second call (e.g. async scheduling /
            # structured output). v2 doesn't wire grammar yet — pass None.
            with RankExecutor._DISPATCH_LOCK:
                model_output = self.engine.model_executor.sample_tokens(None)
        # Resolve AsyncTPUModelRunnerOutput → ModelRunnerOutput.
        from tpu_inference.runner.tpu_runner import AsyncTPUModelRunnerOutput
        if isinstance(model_output, AsyncTPUModelRunnerOutput):
            model_output = model_output.get_output()
        return model_output

    @staticmethod
    def _slice_sampling_md(sampling_md: Any, target_size: int) -> Any:
        """Counterpart to SchedulerShard._pad_sampling_metadata.

        The sidecar padded `temperature` / `top_k` / `top_p` to
        `max_num_reqs` for the boundary spec. Here we slice back down
        to the actual bucket size before model_fn dispatch."""
        import dataclasses
        updates = {}
        for name in ("temperature", "top_k", "top_p"):
            v = getattr(sampling_md, name, None)
            if isinstance(v, jax.Array) and v.shape[0] > target_size:
                updates[name] = v[:target_size]
        return dataclasses.replace(sampling_md,
                                    **updates) if updates else sampling_md

    def execute_with_cpu_prep(
        self,
        input_ids: jax.Array,
        attn_metadata: Any,
        sampling_metadata: Any,
        logits_indices: jax.Array,
        padded_num_reqs: int,
        padded_total_num_scheduled_tokens: int,
        actual_do_sampling: bool = True,
        actual_logprobs: bool = False,
    ) -> Tuple[jax.Array, Any]:
        """v3-pure controller-side: STATELESS TPU dispatch.

        Takes JAX-array pytree leaves directly (no CpuPrep dataclass,
        no cloudpickle for the array data). Each array is CPU-sharded
        on the rank's colocated CPU mesh — `jax.device_put` onto the
        TPU mesh is a direct per-device transfer (the CPU mesh is
        isomorphic to the TPU mesh per `TPUModelRunner._init_cpu_mesh`).

        Slices each variable-bucket array down to the actual bucket size
        before model_fn dispatch (sidecar pads to MAX for the fixed
        boundary spec, so model_fn would otherwise get max-shape inputs
        and the pre-compiled bucketed variant wouldn't match).

        Returns ``(next_tokens, logprobs)`` — JAX arrays on the
        sidecar's CPU mesh after a direct TPU→sidecar-CPU per-device
        copy. They flow back via colocated_python as native pytree leaves.
        """
        import dataclasses

        from tpu_inference.layers.common.sharding import ShardingAxisName

        runner = self.engine.model_executor.driver_worker.model_runner  # type: ignore[attr-defined]
        mesh = runner.mesh
        dp_attn = NamedSharding(mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))
        replicated = NamedSharding(mesh, PartitionSpec())

        # Slice the variable-bucket arrays back to their actual size.
        # block_tables / seq_lens / query_start_loc / request_distribution
        # are already at max shape — they need no slicing.
        # TODO: this slicing op is costly. 
        input_ids_b = input_ids[:padded_total_num_scheduled_tokens]
        input_positions_b = (attn_metadata.input_positions
                             [:padded_total_num_scheduled_tokens])
        logits_indices_b = logits_indices[:padded_num_reqs]
        # Restore the actual meta_field values that the sidecar forced
        # to constants for the boundary's fixed-spec contract.
        attn_metadata = dataclasses.replace(
            attn_metadata,
            input_positions=input_positions_b,
            padded_num_reqs=padded_num_reqs)
        sampling_metadata = dataclasses.replace(
            sampling_metadata,
            do_sampling=actual_do_sampling,
            logprobs=actual_logprobs)
        sampling_metadata = self._slice_sampling_md(sampling_metadata,
                                                     padded_num_reqs)

        # Direct per-device CPU→TPU transfers for every JAX leaf.
        attn_md = jax.tree.map(
            lambda x: jax.device_put(x, dp_attn) if isinstance(x, jax.Array)
            else x,
            attn_metadata)
        sampling_md = jax.tree.map(
            lambda x: jax.device_put(x, replicated) if isinstance(x, jax.Array)
            else x,
            sampling_metadata)
        input_ids_tpu = jax.device_put(input_ids_b, dp_attn)
        input_positions_tpu = attn_md.input_positions
        logits_indices_tpu = jax.device_put(logits_indices_b, replicated)

        # `_DISPATCH_LOCK` serializes the Python-side tracing of model_fn
        # across rank driver threads — vLLM's set_forward_context is a
        # non-thread-local global, so concurrent tracing would let a peer
        # rank's KV-cache tracer escape into ours (UnexpectedTracerError).
        with RankExecutor._DISPATCH_LOCK:
            next_tokens_tpu, logprobs_tpu = runner._dispatch_tpu_sample(
                input_ids=input_ids_tpu,
                input_positions=input_positions_tpu,
                attn_metadata=attn_md,
                sampling_metadata=sampling_md,
                logits_indices=logits_indices_tpu,
            )
        # Pad next_tokens up to runner.max_num_reqs *on TPU* before the
        # cross-host transfer. This keeps the per-step boundary shape
        # invariant (sidecar's prev_next_tokens is always (max_num_reqs,))
        # AND keeps the jnp.pad compile pinned to the TPU layout — if we
        # padded on the sidecar side after device_put, the JAX jit cache
        # would see the array first with its TPU tile layout and later
        # with no layout (CPU), raising
        # `INVALID_ARGUMENT: Input array layout is different for
        # executable jit__pad`.
        max_n = runner.max_num_reqs
        if next_tokens_tpu.shape[0] < max_n:
            next_tokens_tpu = jnp.pad(
                next_tokens_tpu, (0, max_n - next_tokens_tpu.shape[0]))

        # Direct per-device TPU → sidecar-CPU transfer. JAX handles the
        # cross-host copy natively (per the user's spike in
        # examples/colocated_dp/spike_devices_inside_sidecar.py). Returned
        # `jax.Array`s flow back to the sidecar through colocated_python
        # as native pytree leaves — no cloudpickle of the bytes.
        target = self.sidecar_cpu_sharding
        assert target is not None, (
            "RankExecutor.sidecar_cpu_sharding not set — DPEngineCore.__init__ "
            "should have passed it when constructing this RankExecutor.")
        with jax.transfer_guard_device_to_host("allow"), \
             jax.transfer_guard_host_to_device("allow"):
            next_tokens_cpu = jax.device_put(next_tokens_tpu, target)
            if logprobs_tpu is not None:
                logprobs_cpu = jax.tree.map(
                    lambda x: jax.device_put(x, target)
                    if isinstance(x, jax.Array) else x,
                    logprobs_tpu)
            else:
                logprobs_cpu = None
        return next_tokens_cpu, logprobs_cpu

    def take_draft_token_ids(self):
        return self.engine.model_executor.take_draft_token_ids()

    def collective_rpc(self, method, args=(), kwargs=None):
        return self.engine.collective_rpc(method, None, args, kwargs or {})

    def shutdown(self) -> None:
        try:
            self.engine.shutdown()
        except Exception as e:
            # vLLM EngineCore.shutdown() walks PyTorch's CachingDeviceAllocator
            # which doesn't recognise the JAX backend ("Allocator for jax is
            # not a DeviceAllocator"). This happens at process exit, after
            # generation has completed — nothing is leaked. Demote to debug
            # so the user doesn't see N copies of it on every clean run.
            logger.debug("RankExecutor[%d] shutdown error (benign): %s",
                         self.rank_idx, e)


def _partition_devices_by_host(
        devices: List[jax.Device]) -> Dict[int, List[jax.Device]]:
    groups: Dict[int, List[jax.Device]] = defaultdict(list)
    # for d in devices:
    #     groups[d.process_index].append(d)
    # split devices into contiguous groups of 8, assuming each host has 8 chips with
    # consecutive IDs. This is more robust to changes in JAX's device ordering
    groups = {i // 8: devices[i:i + 8] for i in range(0, len(devices), 8)}
    print(groups)
    return dict(sorted(groups.items()))


class DPEngineCore(vLLMEngineCore):
    """Controller-side orchestrator. vLLM EngineCore subclass.

    Holds `dp_size` (SchedulerShard, RankExecutor) pairs and N driver threads
    running schedule → execute → update per step. The outer surface matches
    what vLLM's `InprocClient` and `AsyncLLM` call.
    """

    @staticmethod
    def is_supported() -> bool:
        import vllm.envs as vllm_envs

        from tpu_inference import envs
        return bool(envs.TPU_COLOCATED_DP
                    and vllm_envs.VLLM_TPU_USING_PATHWAYS)

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
    ):
        # We do NOT call super().__init__() — we don't want a local model
        # executor. The minimal attribute set below makes
        # `preprocess_add_request`, `is_sleeping`, etc. work for InprocClient.
        from vllm.v1.structured_output import StructuredOutputManager

        self.vllm_config = vllm_config
        self.log_stats = log_stats
        self.async_scheduling = (
            vllm_config.scheduler_config.async_scheduling)
        self.use_spec_decode = vllm_config.speculative_config is not None
        self.is_pooling_model = (
            vllm_config.model_config.runner_type == "pooling")
        self.is_ec_consumer = True
        self.batch_queue = None
        self.batch_queue_size = 1
        self.aborts_queue = queue.Queue()
        self._idle_state_callbacks: list = []
        self.mm_receiver_cache = None
        self.available_gpu_memory_for_kv_cache = -1
        self.structured_output_manager = StructuredOutputManager(vllm_config)
        self.request_block_hasher = None
        if vllm_config.cache_config.enable_prefix_caching:
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                                     init_none_hash)
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)
            self.request_block_hasher = get_request_block_hasher(
                vllm_config.cache_config.block_size, caching_hash_fn)

        # ---- partition devices into one TPU group per Pathways host ------
        from jax.experimental import colocated_python

        all_tpu = jax.devices()
        host_groups = _partition_devices_by_host(all_tpu)
        num_hosts = len(host_groups)
        requested = getattr(vllm_config.sharding_config, "colocated_dp_size",
                            num_hosts)
        if requested != num_hosts:
            raise ValueError(
                f"TPU_COLOCATED_DP expects data_parallel_size ({requested}) "
                f"== number of Pathways hosts ({num_hosts}). "
                f"Multi-rank-per-host / multi-host-per-rank not yet supported.")
        self.dp_size = num_hosts

        # ---- create per-rank executor (controller) + per-rank sidecar ----
        ShardWrapper = colocated_python.colocated_python_class(SchedulerShard)
        self._rank_execs: List[RankExecutor] = []
        self._sched_clients: List[_ShardClient] = []
        for rank_idx, (host_idx,
                       tpu_devs) in enumerate(host_groups.items()):
            # 1) The rank's CPU sharding — every blob the controller sends to
            #    this sidecar is placed on it (which is also what dispatches
            #    the colocated_python call to the right host). The
            #    RankExecutor uses this same sharding to `device_put`
            #    next_tokens / logprobs back from TPU to the sidecar's CPU
            #    mesh — direct per-device transfer, no cloudpickle.
            cpu_devs = colocated_python.colocated_cpu_devices(tpu_devs)
            cpu_sharding = NamedSharding(Mesh(np.asarray(cpu_devs), ("x", )),
                                         PartitionSpec())
            # 2) Controller-side: spin up the TPU engine for this rank.
            #    Loads model on rank_idx's chips, profiles KV memory.
            rexec = RankExecutor(base_vllm_config=vllm_config,
                                 rank_idx=rank_idx,
                                 tpu_devices=tpu_devs,
                                 all_tpu_devices=all_tpu,
                                 log_stats=log_stats,
                                 sidecar_cpu_sharding=cpu_sharding)
            self._rank_execs.append(rexec)
            # 3) Sidecar Scheduler: takes the KV cache config the controller
            #    just computed via profiling, so block-allocation indices
            #    match the controller-side KV cache slots.
            #
            # IMPORTANT: reseed Python's `random` before each ShardWrapper
            # construction. vLLM's worker init runs
            # `vllm.utils.torch_utils.set_random_seed(model_config.seed)`
            # which calls `random.seed(seed)` *and*
            # `TpuPlatform.manual_seed_all(seed)` (which calls
            # `random.seed(seed)` again). Both ranks see the same seed, so
            # Python's `random` ends up in identical states after rank 0 and
            # rank 1 inits. `colocated_python._InstanceRegistry.new_instance`
            # (obj.py:42) draws its uid via `random.getrandbits(63)` — so
            # the second `ShardWrapper(...)` produces a colliding uid and
            # trips `assert uid not in self._storage`. Reseed
            # non-deterministically from `os.urandom` to defeat this.
            import random as _py_random
            _py_random.seed()
            # Fail fast with a clear error if anything in the sidecar args is
            # not cloudpickle-able (most likely culprits: live device handles
            # snuck into vllm_config; tensors snuck into kv_cache_config).
            # Pass the rank's TPU device ids (ints — safely picklable)
            # so the sidecar can resolve them inside its jax.devices()
            # view and ask colocated_python for the corresponding CPU
            # devices → mesh whose shape matches the controller's TPU mesh.
            sidecar_init_args = (rexec.sidecar_vllm_config,
                                 rexec.kv_cache_config,
                                 rexec.scheduler_block_size,
                                 rexec.hash_block_size,
                                 _capture_sidecar_env(),
                                 [int(d.id) for d in tpu_devs])
            try:
                cloudpickle.dumps(sidecar_init_args)
            except Exception as e:
                raise RuntimeError(
                    f"SchedulerShard[rank={rank_idx}] init args are not "
                    f"cloudpickle-able (colocated_python will fail when it "
                    f"ships the initializer closure to the colocated host). "
                    f"Root cause: {e!r}. The likely culprit is a live "
                    f"jaxlib Device or jax.Array hiding in vllm_config or "
                    f"kv_cache_config; strip it before constructing the "
                    f"shard.") from e
            shard = ShardWrapper(*sidecar_init_args)
            # max_num_reqs from the rank's runner — sets the fixed shape of
            # `prev_next_tokens` (the JAX-array leaf that crosses each
            # step in the controller → sidecar direction).
            rank_max_num_reqs = (
                rexec.engine.model_executor.driver_worker  # type: ignore[attr-defined]
                .model_runner.max_num_reqs)
            self._sched_clients.append(
                _ShardClient(shard, cpu_sharding, rank_max_num_reqs))

        # vLLM compatibility: LLMEngine.__init__ does
        # `self.model_executor = self.engine_core.engine_core.model_executor`
        # (llm_engine.py:124). We don't have a controller-local executor, so
        # expose rank 0's (mirrors DisaggEngineCore.__init__:526). It's only
        # read for surface-level attributes (e.g., supported_tasks); the real
        # per-step dispatch goes through the per-rank RankExecutors via the
        # driver threads.
        self.model_executor = self._rank_execs[0].engine.model_executor

        logger.info(
            "DPEngineCore initialized: dp_size=%d, per-host TPU device counts=%s",
            self.dp_size,
            {h: len(ds)
             for h, ds in host_groups.items()})

        # ---- routing / threading ------------------------------------------
        self._owner: Dict[str, int] = {}                       # req_id → rank
        self._load: List[List[int]] = [[0, 0] for _ in range(self.dp_size)]
        self._load_lock = threading.Lock()
        self._funnel: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._driver_threads = [
            threading.Thread(target=self._drive_rank,
                             args=(i, ),
                             daemon=True,
                             name=f"DPEngineCore-drive-{i}")
            for i in range(self.dp_size)
        ]
        for t in self._driver_threads:
            t.start()

        self.step_fn = self.step
        atexit.register(self._atexit_cleanup)

        if executor_fail_callback is not None:
            self._executor_fail_callback = executor_fail_callback
        logger.info("DPEngineCore ready")

    # ---- per-rank driver thread (the heart of the design) ----------------

    def _drive_rank(self, i: int) -> None:
        """v3 driver loop: ONE combined boundary call per step.

        Per step:
          1. ``sched_out, drained = client.step(prev_pair)`` — single boundary
             round-trip carrying:
              IN:  prior step's ``(sched_out, model_out)`` for update_from_output
              OUT: next step's ``SchedulerOutput`` (or None if idle) + any
                   ``EngineCoreOutputs`` produced since the last call.
          2. If ``sched_out is None`` (sidecar idle) → sleep this rank's
             driver thread briefly.
          3. Otherwise: ``model_out = rexec.execute(sched_out)`` (TPU compute,
             controller-local; no boundary crossing).
          4. Funnel any drained outputs into the central queue for
             ``DPEngineCore.step``.

        Compared to v2.1 (4 crossings/step: has_requests, schedule,
        update_from_output_fire, drain_outputs) this is 1 crossing/step,
        carrying the same per-step data in fewer round-trips. The only OTHER
        crossings in the entire engine are:
          - ``add_request`` (when LLMEngine submits a new request)
          - ``shutdown``
        Everything else (has_requests, get_num_unfinished_requests, etc.) is
        no longer on the per-step hot path.

        TPU dispatch from the controller is the irreducible constraint
        forcing this single round-trip: model_fn requires controller-side
        execution because the sidecar has no TPU device access. The remaining
        v3 follow-up (task #13) shrinks the per-step blob payload by moving
        TPUModelRunner's CPU prep onto the sidecar — the boundary architecture
        here will not change.
        """
        client = self._sched_clients[i]
        rexec = self._rank_execs[i]
        # v3-pure: `prev_next_tokens` is a JAX array (shape (max_num_reqs,),
        # int32) on this rank's sidecar CPU sharding. It rides across the
        # boundary as a native pytree leaf — no cloudpickle of its buffer.
        # `has_prev` is a uint8 JAX scalar (0=first call, 1=subsequent).
        prev_next_tokens, has_prev = client.first_call_state()
        while not self._stop.is_set():
            try:
                (input_ids, attn_md, sampling_md, logits_indices,
                 meta) = client.step(prev_next_tokens, has_prev)
            except Exception as e:
                if self._stop.is_set():
                    break
                logger.exception("rank %d step() failed: %s", i, e)
                time.sleep(0.05)
                prev_next_tokens, has_prev = client.first_call_state()
                continue
            self._funnel_drained(i, meta.get("drained", []))
            if not meta.get("has_work", False):
                # Sidecar idle. Reset so next step() doesn't try to
                # update_from_output stale data; yield briefly.
                prev_next_tokens, has_prev = client.first_call_state()
                time.sleep(0.001)
                continue
            try:
                # TPU dispatch — pytree leaves (input_ids, attn_md,
                # sampling_md, logits_indices) cross natively, no
                # cloudpickle of their buffers. Returns next_tokens /
                # logprobs as JAX arrays already on the sidecar's CPU
                # sharding, ready to flow back via the next step() call.
                next_tokens, _logprobs = rexec.execute_with_cpu_prep(
                    input_ids=input_ids,
                    attn_metadata=attn_md,
                    sampling_metadata=sampling_md,
                    logits_indices=logits_indices,
                    padded_num_reqs=meta["padded_num_reqs"],
                    padded_total_num_scheduled_tokens=meta[
                        "padded_total_num_scheduled_tokens"],
                    actual_do_sampling=meta.get("actual_do_sampling", True),
                    actual_logprobs=meta.get("actual_logprobs", False),
                )
            except Exception as e:
                logger.exception("rank %d TPU execute_with_cpu_prep() "
                                 "failed: %s", i, e)
                time.sleep(0.05)
                prev_next_tokens, has_prev = client.first_call_state()
                continue
            # `execute_with_cpu_prep` already padded next_tokens to
            # max_num_reqs on the TPU side (before the cross-host
            # transfer), so the shape contract for the sidecar's
            # prev_next_tokens is satisfied as-is — no further pad here.
            assert next_tokens.shape[0] == client._max_num_reqs, (
                f"next_tokens shape {next_tokens.shape} != "
                f"(max_num_reqs={client._max_num_reqs},) — "
                f"RankExecutor.execute_with_cpu_prep should have padded "
                f"on the TPU side before device_put.")
            prev_next_tokens = next_tokens
            has_prev = client._has_prev_true_cache()

    def _funnel_drained(
            self, i: int, drained: List[Dict[int, EngineCoreOutputs]]) -> None:
        """Push a batch of EngineCoreOutputs from rank ``i`` into the funnel
        and reconcile load/ownership tracking. No boundary crossing — this
        runs on whatever the controller already got from ``client.step``."""
        for engine_outs in drained:
            for o in engine_outs.values():
                if o.finished_requests:
                    with self._load_lock:
                        self._load[i][0] = max(
                            0,
                            self._load[i][0] - len(o.finished_requests))
                    for rid in o.finished_requests:
                        self._owner.pop(rid, None)
            self._funnel.put((i, engine_outs))

    # ---- routing ---------------------------------------------------------

    def _pick_rank(self) -> int:
        # P2C-style score: waiting weighed 4× heavier than running.
        with self._load_lock:
            scores = [w * 4 + r for r, w in self._load]
        return min(range(self.dp_size), key=lambda i: (scores[i], i))

    # ---- vLLM EngineCore surface ----------------------------------------

    def add_request(self, request: Request, request_wave: int = 0) -> None:
        idx = self._pick_rank()
        self._owner[request.request_id] = idx
        with self._load_lock:
            self._load[idx][1] += 1
        # Fire-and-forget; the blob's CPU sharding pins the call to the host.
        self._sched_clients[idx].add_request(request, request_wave)

    def abort_requests(self, request_ids: List[str]) -> None:
        by_rank: Dict[int, List[str]] = defaultdict(list)
        for rid in request_ids:
            idx = self._owner.pop(rid, None)
            if idx is not None:
                by_rank[idx].append(rid)
        for idx, ids in by_rank.items():
            self._sched_clients[idx].finish_requests(
                ids, RequestStatus.FINISHED_ABORTED)

    def step(self) -> Tuple[Dict[int, EngineCoreOutputs], bool]:
        try:
            _, engine_outs = self._funnel.get(timeout=1.0)
        except queue.Empty:
            return {}, False
        return engine_outs, True

    def post_step(self, model_executed: bool) -> None:
        # Driver threads handle per-step post-processing inside their loop;
        # nothing for the outer step() to do.
        pass

    # ---- aggregations: fan out to all ranks ------------------------------

    def get_supported_tasks(self) -> Tuple[SupportedTask, ...]:
        # All ranks are identical engines; query rank 0.
        return self._rank_execs[0].engine.get_supported_tasks()

    def reset_prefix_cache(self,
                           reset_running_requests: bool = False,
                           reset_connector: bool = False) -> bool:
        results = [
            client.reset_prefix_cache(reset_running_requests, reset_connector)
            for client in self._sched_clients
        ]
        for rexec in self._rank_execs:
            try:
                rexec.engine.model_executor.reset_mm_cache()
            except Exception:
                pass
        return all(results)

    def reset_encoder_cache(self) -> None:
        for client in self._sched_clients:
            client.reset_encoder_cache()
        for rexec in self._rank_execs:
            try:
                rexec.engine.model_executor.reset_encoder_cache()
            except Exception:
                pass

    def reset_mm_cache(self) -> None:
        for rexec in self._rank_execs:
            try:
                rexec.engine.reset_mm_cache()
            except Exception:
                pass

    def execute_dummy_batch(self) -> None:
        for rexec in self._rank_execs:
            rexec.engine.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return all(
            rexec.engine.add_lora(lora_request)
            for rexec in self._rank_execs)

    def remove_lora(self, lora_id: int) -> bool:
        return all(
            rexec.engine.remove_lora(lora_id) for rexec in self._rank_execs)

    def list_loras(self) -> set:
        out: set = set()
        for rexec in self._rank_execs:
            out.update(rexec.engine.list_loras())
        return out

    def pin_lora(self, lora_id: int) -> bool:
        return all(
            rexec.engine.pin_lora(lora_id) for rexec in self._rank_execs)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if not isinstance(method, str):
            raise NotImplementedError(
                "DPEngineCore.collective_rpc requires a method name (str).")
        outs: List[Any] = []
        for rexec in self._rank_execs:
            outs.extend(rexec.collective_rpc(method, args, kwargs))
        return outs

    def profile(self, is_start: bool = True,
                profile_prefix: Optional[str] = None) -> None:
        # On Pathways the controller is a SINGLE JAX process driving every
        # host's TPU chips, so a single `jax.profiler.start_trace` captures
        # work across all hosts (each host writes its own host-named
        # xplane.pb into profile_dir). Calling profile() on every rank's
        # engine would invoke start_trace N times in the same process —
        # JAX rejects the 2nd+ calls. Use rank 0 only.
        #
        # Unlike MPMD (TPU_MULTIPROCESS_DP), per-rank xplane filenames do
        # not clobber here because each Pathways host has a unique
        # hostname; no `dp<N>_` injection à la
        # tpu_inference/runner/utils.py:_inject_dp_rank_into_filename
        # is required.
        # if is_start:
            # Workaround for the pathwaysutils ↔ Pathways runtime mismatch
            # that produces "Mismatch between out_handlers and num_results:
            # 0 vs 1" when start_trace fires the profile-request executable.
            # See `_install_pathways_profile_handler_patch` below.
            # _install_pathways_profile_handler_patch()
        if self._rank_execs:
            self._rank_execs[0].engine.profile(is_start, profile_prefix)

    # ---- pause / sleep ---------------------------------------------------

    def pause_scheduler(self, mode="abort", clear_cache: bool = True):
        if clear_cache:
            self.reset_prefix_cache(reset_running_requests=True)
            self.reset_mm_cache()
            self.reset_encoder_cache()
        return None

    def resume_scheduler(self) -> None:
        return None

    def is_scheduler_paused(self) -> bool:
        return False

    def sleep(self, level: int = 1, mode="abort"):
        for rexec in self._rank_execs:
            rexec.engine.sleep(level, mode)
        return None

    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        for rexec in self._rank_execs:
            rexec.engine.wake_up(tags)

    def is_sleeping(self) -> bool:
        return False

    def save_sharded_state(self, path, pattern=None, max_size=None) -> None:
        for rexec in self._rank_execs:
            rexec.engine.save_sharded_state(path, pattern, max_size)

    # ---- lifecycle -------------------------------------------------------

    def shutdown(self) -> None:
        atexit.unregister(self._atexit_cleanup)
        self._stop.set()
        for t in self._driver_threads:
            t.join(timeout=2.0)
        for client in self._sched_clients:
            try:
                client.shutdown()
            except Exception as e:
                # Sidecar teardown over the colocated_python boundary races
                # with Pathways runtime shutdown at process exit; the call
                # may fail to deliver but the sidecar exits cleanly anyway.
                logger.debug("sidecar shutdown error (benign): %s", e)
        for rexec in self._rank_execs:
            rexec.shutdown()
        try:
            self.structured_output_manager.clear_backend()
        except Exception:
            pass

    def _atexit_cleanup(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


# ============================================================================
# Backwards-compatible aliases for v1 import paths.
# ============================================================================

# Old name in v1; kept so existing imports / docs don't break while we migrate.
ColocatedDPEngineCore = DPEngineCore


# ============================================================================
# Workaround: pathwaysutils profile-handler mismatch
# ============================================================================

_PATHWAYS_PROFILE_PATCH_INSTALLED = False


def _install_pathways_profile_handler_patch() -> None:
    """Tolerate the pathwaysutils profile-request handler-count mismatch.

    `pathwaysutils.profiling._start_pathways_trace_from_profile_request` calls
    `PluginExecutable.call()` with empty `out_avals` / `out_shardings`, but the
    Pathways runtime returns 1 result token for the profile request — yielding
    `ValueError: Mismatch between out_handlers and num_results: 0 vs 1` and
    aborting start_trace.

    This patch swallows that ValueError when it comes from a profile-request
    call (`prog_str` contains `"profileRequest"`), then forces a sync via the
    token future so the trace is actually started before we return. Subsequent
    `stop_trace()` is unaffected.

    Idempotent and a no-op if pathwaysutils isn't installed.
    """
    global _PATHWAYS_PROFILE_PATCH_INSTALLED
    if _PATHWAYS_PROFILE_PATCH_INSTALLED:
        return
    try:
        from pathwaysutils import plugin_executable as pe
    except Exception:
        return  # not running under pathwaysutils — nothing to patch

    orig_call = pe.PluginExecutable.call
    if getattr(orig_call, "_tpu_inference_patched", False):
        _PATHWAYS_PROFILE_PATCH_INSTALLED = True
        return

    def patched_call(self, in_arr=(), out_shardings=(), out_avals=(),
                     out_committed=True):
        try:
            return orig_call(self, in_arr, out_shardings, out_avals,
                             out_committed)
        except ValueError as e:
            # Only swallow the specific handler/result mismatch on a
            # profile-request executable; re-raise everything else.
            if "Mismatch between out_handlers and num_results" not in str(e):
                raise
            # Confirm this is the profile-request executable by inspecting
            # the program string we compiled in.  pathwaysutils stores it
            # only via the compiled module, so we fall back to "if no
            # out_avals were declared at all, swallow it" — that matches the
            # narrow shape of the bug.
            if out_avals or out_shardings:
                raise
            logger.debug(
                "Swallowed pathwaysutils handler/result mismatch on "
                "no-out-aval call (likely a profile-request token).")
            import concurrent.futures
            fut = concurrent.futures.Future()
            fut.set_result(None)
            return ((), fut)

    patched_call._tpu_inference_patched = True  # type: ignore[attr-defined]
    pe.PluginExecutable.call = patched_call
    _PATHWAYS_PROFILE_PATCH_INSTALLED = True
    logger.info(
        "Installed pathwaysutils.PluginExecutable.call patch "
        "(profile-request handler mismatch workaround).")


# ============================================================================
# Registration helpers (unchanged from v1).
# ============================================================================

_PATCHED = False
_ORIGINAL_ENGINE_CORE = None


def enable_colocated_dp_engine_core() -> None:
    """Monkeypatch `vllm.v1.engine.core.EngineCore` → `DPEngineCore`.

    Centralised so the platform can call it automatically when
    `TPU_COLOCATED_DP` is set. Idempotent.
    """
    global _PATCHED, _ORIGINAL_ENGINE_CORE
    if _PATCHED:
        return
    import vllm.v1.engine.core as core_mod
    _ORIGINAL_ENGINE_CORE = core_mod.EngineCore
    core_mod.EngineCore = DPEngineCore  # type: ignore[assignment]
    try:
        import vllm.v1.engine.core_client as cc
        if hasattr(cc, "EngineCore"):
            cc.EngineCore = DPEngineCore  # type: ignore[assignment]
    except Exception:
        pass
    _PATCHED = True
    logger.info("vllm EngineCore patched -> DPEngineCore (colocated DP)")


def disable_colocated_dp_engine_core() -> None:
    global _PATCHED, _ORIGINAL_ENGINE_CORE
    if not _PATCHED:
        return
    import vllm.v1.engine.core as core_mod
    core_mod.EngineCore = _ORIGINAL_ENGINE_CORE  # type: ignore[assignment]
    try:
        import vllm.v1.engine.core_client as cc
        if hasattr(cc, "EngineCore"):
            cc.EngineCore = _ORIGINAL_ENGINE_CORE  # type: ignore[assignment]
    except Exception:
        pass
    _PATCHED = False


def update_vllm_config_for_colocated_dp(vllm_config: VllmConfig) -> None:
    """Install the EngineCore patch when TPU_COLOCATED_DP is in effect."""
    from tpu_inference import envs
    if not envs.TPU_COLOCATED_DP:
        return
    colocated_dp_size = getattr(vllm_config.sharding_config,
                                "colocated_dp_size", 1)
    if colocated_dp_size <= 1:
        return
    enable_colocated_dp_engine_core()
