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
#
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
# sidecar may take wrong code paths. TODO(wenxindong): simply this.
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
        eco: EngineCoreOutputs,
        max_tokens: int,
        max_outputs: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Strip `new_token_ids` from every EngineCoreOutput in `eco`.

    Returns ``(flat_tokens, offsets)`` where ``offsets[k]`` is the start
    index in `flat_tokens` for the k-th visited EngineCoreOutput (and
    ``offsets[-1]`` is the total length). Returns ``None`` if either
    cap (`max_tokens` / `max_outputs`) would be exceeded — caller
    should fall back to cloudpickling new_token_ids in place.
    """
    flat: List[int] = []
    offsets: List[int] = [0]
    for n_outputs, output in enumerate(eco.outputs):
        if n_outputs >= max_outputs:
            return None
        tokens = output.new_token_ids or []
        if len(flat) + len(tokens) > max_tokens:
            return None
        flat.extend(tokens)
        offsets.append(len(flat))
        output.new_token_ids = []   # stripped; reattached on controller
    return flat, offsets


def _reattach_drained_tokens(
        eco: EngineCoreOutputs,
        flat_tokens: "np.ndarray",
        offsets: List[int],
) -> None:
    """Inverse of `_extract_drained_tokens` — walk the same order, slice
    `flat_tokens[offsets[i]:offsets[i+1]]` back into each EngineCoreOutput's
    `new_token_ids`."""
    for i, output in enumerate(eco.outputs):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if start < end:
            output.new_token_ids = flat_tokens[start:end].tolist()
        # else: leave as the empty list put there by the stripper


# ============================================================================
# req_id string ↔ int handle interning.
#
# vLLM identifies requests by string `request_id` (e.g. "chatcmpl-<uuid>" in
# online serving). Those strings appear in every drained `EngineCoreOutput`
# (`.request_id`) and in `EngineCoreOutputs.finished_requests`, and are the
# dominant variable-length cost left in the cloudpickle blob after Phase 3
# lifted `new_token_ids` out. The controller assigns each request a small int
# handle at `add_request` time and tells the sidecar; the sidecar swaps the
# strings for handles before packing (`_intern_drained_req_ids`), and the
# controller swaps them back (`_restore_drained_req_ids`) before handing the
# outputs to vLLM. Both sides tolerate a mix: a req_id missing from the map is
# left untouched (str stays str), so a lost/raced handle degrades gracefully
# to the old string-pickling behaviour instead of corrupting output.
# ============================================================================


def _intern_drained_req_ids(
        eco: EngineCoreOutputs,
        handle_by_id: Dict[str, int],
) -> None:
    """In-place: replace req_id strings with int handles to shrink the blob."""
    for output in eco.outputs:
        h = handle_by_id.get(output.request_id)
        if h is not None:
            output.request_id = h
    if eco.finished_requests:
        eco.finished_requests = {
            handle_by_id.get(rid, rid)
            for rid in eco.finished_requests
        }


def _restore_drained_req_ids(
        eco: EngineCoreOutputs,
        id_by_handle: Dict[int, str],
) -> None:
    """Inverse of `_intern_drained_req_ids`: int handles → req_id strings."""
    for output in eco.outputs:
        if isinstance(output.request_id, int):
            output.request_id = id_by_handle.get(
                output.request_id, output.request_id)
    if eco.finished_requests:
        eco.finished_requests = {
            id_by_handle.get(x, x) if isinstance(x, int) else x
            for x in eco.finished_requests
        }


def _unpack(blob: jax.Array) -> Any:
    """Inverse of `_pack`. Works for any size — only reads the length header."""
    arr = np.asarray(blob)
    n = int(
        np.frombuffer(arr[:_BLOB_HEADER_BYTES].tobytes(),
                      dtype=np.uint64)[0])
    return cloudpickle.loads(arr[_BLOB_HEADER_BYTES:_BLOB_HEADER_BYTES +
                                 n].tobytes())


def _single_client_outputs(
        engine_outs: Dict[int, EngineCoreOutputs]) -> Optional[EngineCoreOutputs]:
    """Collapse `scheduler.update_from_output`'s client-indexed dict to one
    `EngineCoreOutputs`.

    INITIAL-VERSION SIMPLIFICATION: `update_from_output` returns a dict keyed
    by `request.client_index` to support multiple frontend clients. This
    colocated-DP version assumes exactly one client, so the dict is always
    empty (scheduler produced nothing) or a single `{0: ...}` entry. We carry
    the bare `EngineCoreOutputs` through the outbox / boundary and re-wrap into
    `{0: ...}` only at the controller's vLLM-facing `step()`. Revisit if/when
    multi-client support is added.
    """
    if not engine_outs:
        return None
    assert len(engine_outs) == 1, (
        f"colocated DP assumes a single frontend client, but "
        f"update_from_output returned {len(engine_outs)} client buckets: "
        f"{list(engine_outs)}")
    return next(iter(engine_outs.values()))


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
        # __init__. We don't support structured output yet, so this is dead weight here. 
        # We still construct one because
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
        # Preferred path: map this rank's TPU device ids → their colocated CPU
        # devices, giving a CPU mesh isomorphic to the TPU mesh. This only
        # works if `jax.devices()` inside the sidecar exposes the TPU devices.
        # On some Pathways images it does NOT (it returns only the sidecar's
        # local CPU devices), so the id-resolution comes back empty and
        # `colocated_cpu_devices([])` raises IndexError. In that case fall
        # back to `jax.local_devices()` — inside the sidecar those ARE the
        # colocated CPU devices for this host's TPUs, which is what we want.
        cpu_devs_for_runner = None
        if tpu_device_ids:
            tpu_id_set = set(int(i) for i in tpu_device_ids)
            tpu_devs_resolved = [d for d in jax.devices()
                                 if int(d.id) in tpu_id_set]
            if tpu_devs_resolved and len(tpu_devs_resolved) == len(tpu_id_set):
                cpu_devs_for_runner = list(
                    _cp.colocated_cpu_devices(tpu_devs_resolved))
            else:
                logger.error(
                    "SchedulerShard: resolved only %d of %d tpu_device_ids "
                    "from jax.devices() (sidecar may not expose TPU devices). "
                    "Falling back to jax.local_devices().",
                    len(tpu_devs_resolved), len(tpu_id_set))
        if cpu_devs_for_runner is None:
            # No tpu_device_ids, or resolution failed.
            # todo(wenxindong): make tpu_device_ids non-optional / resolvable.
            cpu_devs_for_runner = list(jax.local_devices())

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
        # TODO(wenxindong): v3 cleanup — unify the hashing logic and remove this
        if vllm_config.cache_config.enable_prefix_caching:
            init_none_hash(
                get_hash_fn_by_name(
                    vllm_config.cache_config.prefix_caching_hash_algo))

        # `update_from_output` writes into this queue; `drain_outputs` reads
        # from it. Lets us collapse the EngineCoreOutputs blob crossing from
        # 1-per-step to 1-per-drain (often <1-per-step under batching).
        # Single-client simplification: store bare EngineCoreOutputs (the
        # client dict is collapsed via `_single_client_outputs`).
        self._outbox: "queue.Queue[EngineCoreOutputs]" = queue.Queue()
        # req_id (str) → int handle, seeded by the controller at add_request.
        # Used to intern req_id strings out of the drained blob. Entries are
        # popped when their request finishes (see `step`).
        self._handle_by_id: Dict[str, int] = {}
        # Held between `step` calls — when the controller comes back with
        # the TPU-sample dict, we pair it with the `sched_out` /
        # `padded_num_reqs` / `logits_indices_selector` that produced it
        # for `_postprocess_tpu_sample` and `scheduler.update_from_output`.
        # None of these cross the boundary.
        self._pending_sched_out: Optional[Any] = None
        self._pending_padded_num_reqs: int = 0
        self._pending_logits_indices_selector: Optional[Any] = None
        # Prepared model-input pytree stashed by `poll` (active cycle) for the
        # immediately-following `prepare_step` to ship. Does not cross.
        self._pending_prepared: Optional[Tuple[Any, Any, Any, Any]] = None

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
        request, _wave, handle = _unpack(blob)
        if handle is not None:
            self._handle_by_id[request.request_id] = handle
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
        eco = _single_client_outputs(
            self._scheduler.update_from_output(scheduler_output,
                                               model_runner_output))
        if eco is not None:
            self._outbox.put(eco)
        return _pack_tiny(True, blob.sharding)

    def drain_outputs(self, pin: jax.Array) -> jax.Array:
        """Return everything queued in `_outbox` (possibly empty). One blob
        crossing per drain, instead of one per step inside update_from_output."""
        batch: List[EngineCoreOutputs] = []
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
        
        TODO(wenxindong): pass cpu_sharding to the CPU model runner. 
        """
        return jax.tree.map(
            lambda x: jax.device_put(x, target_sharding)
            if isinstance(x, jax.Array) else x,
            tree)


    def poll(self, prev_next_tokens: jax.Array,
             has_prev: jax.Array) -> jax.Array:
        """Per-cycle FIXED-SPEC half of the boundary protocol.

        Consumes the previous step's sampled tokens (`has_prev=1`), drains
        finished outputs, schedules the next step, and — when there's work —
        runs CPU input-prep, stashing the prepared arrays for a following
        `prepare_step` call.

        `prev_next_tokens` is a native `jax.Array` on this sidecar's CPU
        sharding (the controller `device_put` the model's sample output here);
        `has_prev` is a uint8 scalar (0 on the first/idle-reset call).

        Returns a FIXED-shape pytree `(tokens_jax, offsets_jax, blob)`:
          - tokens_jax / offsets_jax: drained new_token_ids (Phase 3), padded
            to fixed MAX sizes.
          - blob: tiny cloudpickle dict — has_work, drained (req_ids
            interned), tokens_stripped, n_drained_outputs, and, when
            has_work, the (num_reqs, num_tokens) bucket sizes the controller
            needs to build `prepare_step`'s input pins.

        The output spec is INVARIANT across idle and active cycles (no model-
        input arrays here), so colocated_python's frozen-output-spec contract
        holds without an idle-pytree filler. `prepare_step` carries the
        variable-shape model inputs under its own (bucket-keyed) spec.
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
            eco = _single_client_outputs(
                self._scheduler.update_from_output(self._pending_sched_out,
                                                   mr_out))
            if eco is not None:
                self._outbox.put(eco)
            self._pending_sched_out = None
        # Drain outbox. INVARIANT: at most one EngineCoreOutputs is queued at
        # drain time. The two producers (the has_prev branch above and the
        # empty-schedule branch below) never both feed a single drain: the
        # empty-schedule branch defers to the outbox and returns idle, which
        # makes the controller reset has_prev=0, so the next step adds no put
        # of its own before draining the deferred one. Per rank there's a
        # single driver thread, so no concurrent producers either.
        drained: Optional[EngineCoreOutputs] = None
        while True:
            try:
                eco = self._outbox.get_nowait()
            except queue.Empty:
                break
            assert drained is None, (
                "outbox held >1 EngineCoreOutputs — single-per-step invariant "
                "violated (see comment above)")
            drained = eco
        # Intern req_id strings → int handles to shrink the cloudpickle blob.
        # Collect finished req_ids (still strings) FIRST so we can drop their
        # handle-map entries after interning — bounds the map to live requests.
        if drained is not None:
            _finished_ids = list(drained.finished_requests or ())
            _intern_drained_req_ids(drained, self._handle_by_id)
            for rid in _finished_ids:
                self._handle_by_id.pop(rid, None)
        # Drained JAX arrays + metadata blob ship via `_pack_blob` on this
        # rank's CPU sharding, referenced via the JAX-array arg.
        out_sharding = prev_next_tokens.sharding

        # Phase 3 token buffer sizing — fixed-shape per the boundary contract.
        max_drain_tokens = self._cpu_runner.max_num_reqs * _MAX_DRAIN_TOKENS_PER_REQ
        max_drain_offsets = _MAX_DRAIN_OUTPUTS + 1   # +1 for the leading 0
        zero_tokens_buf = np.zeros(max_drain_tokens, dtype=np.int32)
        zero_offsets_buf = np.zeros(max_drain_offsets, dtype=np.int32)

        def _build_drained_jax(eco: Optional[EngineCoreOutputs]) -> Tuple[jax.Array, jax.Array, bool]:
            """Strip new_token_ids from `eco` into fixed-shape JAX arrays."""
            if eco is None:
                return (jax.device_put(zero_tokens_buf, out_sharding),
                        jax.device_put(zero_offsets_buf, out_sharding),
                        True)  # no-op strip = succeed
            extracted = _extract_drained_tokens(
                eco, max_drain_tokens, _MAX_DRAIN_OUTPUTS)
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

        tokens_jax, offsets_jax, stripped = _build_drained_jax(drained)
        n_drained_outputs = (
            len(drained.outputs) if (stripped and drained is not None) else 0)

        def _ret(has_work: bool, num_reqs: int, num_tokens: int) -> jax.Array:
            """Pack the FIXED-spec poll return. `num_reqs` / `num_tokens` are
            the shapes `prepare_step` will emit (0 when idle); the controller
            uses them to build prepare_step's bucket-encoding input pins."""
            blob = _pack_blob({
                "has_work": has_work,
                "drained": drained,
                "tokens_stripped": stripped,
                "n_drained_outputs": n_drained_outputs,
                "num_reqs": num_reqs,
                "num_tokens": num_tokens,
            }, out_sharding)
            return self._canonicalize_outputs(
                out_sharding, (tokens_jax, offsets_jax, blob))

        # Schedule next step + run CPU prep on this sidecar.
        if not self._scheduler.has_requests():
            return _ret(False, 0, 0)
        sched_out = self._scheduler.schedule()
        # input_batch updates from scheduler output (add/remove/cache state).
        # Must run even when total_num_scheduled_tokens==0 to register
        # finished requests for cleanup — mirrors `_execute_model`.
        self._cpu_runner.persistent_batch_manager.update_states(
            sched_out, self._cpu_runner.get_mrope_input_positions_fn
            if hasattr(self._cpu_runner, "get_mrope_input_positions_fn")
            else None)

        # Empty-schedule short-circuit: 0 scheduled tokens (e.g. all running
        # requests blocked on grammar compilation, or everything finished).
        # `_prepare_inputs` asserts > 0, so flush finished outputs via
        # update_from_output(EMPTY_) — deferred to the outbox so the next
        # poll ships them — and report idle (no prepare_step this cycle).
        if sched_out.total_num_scheduled_tokens == 0:
            from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
            eco = _single_client_outputs(
                self._scheduler.update_from_output(sched_out,
                                                   EMPTY_MODEL_RUNNER_OUTPUT))
            if eco is not None:
                self._outbox.put(eco)
            self._pending_sched_out = None
            self._pending_prepared = None
            return _ret(False, 0, 0)

        # Active: run CPU prep, stash the prepared arrays for prepare_step.
        (input_ids, _input_positions_unused, attn_metadata, sampling_metadata,
         logits_indices, _spec_decode_metadata, logits_indices_selector,
         padded_num_reqs, _req_ids_dp,
         _padded_num_sched_per_dp) = self._cpu_runner._prepare_inputs(sched_out)

        self._pending_sched_out = sched_out
        self._pending_padded_num_reqs = padded_num_reqs
        self._pending_logits_indices_selector = logits_indices_selector
        self._pending_prepared = (input_ids, attn_metadata, sampling_metadata,
                                  logits_indices)
        # `num_reqs` / `num_tokens` are the two bucket dimensions that fully
        # determine every prepared-input array's shape (all others are
        # functions of these + fixed constants). The controller passes input
        # pins of these shapes to prepare_step so colocated re-specializes per
        # bucket with a consistent output spec.
        return _ret(True, logits_indices.shape[0], input_ids.shape[0])

    def prepare_step(self, pin_reqs: jax.Array,
                     pin_tokens: jax.Array) -> jax.Array:
        """Active-only DATA half: return the model inputs stashed by the
        preceding `poll(has_work=True)`.

        `pin_reqs` / `pin_tokens` are dummy arrays whose SHAPES (num_reqs /
        num_tokens) encode the bucket. colocated_python keys its
        specialization (and frozen output spec) on the input spec, so feeding
        the bucket as the input shape means each bucket gets its own
        specialization with a self-consistent output spec — no idle/active
        collision (this is only called when poll reported work) and no
        cross-bucket output-spec mismatch.
        """
        assert self._pending_prepared is not None, (
            "prepare_step called without a preceding poll(has_work=True) — "
            "boundary ordering invariant violated.")
        prepared = self._pending_prepared
        self._pending_prepared = None
        return self._canonicalize_outputs(pin_reqs.sharding, prepared)

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
        self._max_num_reqs = max_num_reqs
        # Pre-place a tiny pin array on this rank's CPU sharding. 
        self._pin = jax.device_put(np.zeros(_TINY_BYTES, dtype=np.uint8),
                                   cpu_sharding)

    # ---- request-lifecycle ----------------------------------------------

    def add_request(self,
                    request: Request,
                    request_wave: int = 0,
                    handle: Optional[int] = None) -> None:
        # Send the Request payload + its int handle (used to intern req_id
        # strings out of the drained blob). ack is tiny and we don't need to
        # wait. block_until_ready is intentionally NOT called — colocated_python
        # serializes per-thread calls on the sidecar, so order is preserved
        # for the next call on this thread.
        self._shard.add_request(_pack_blob((request, request_wave, handle),
                                           self._sharding))

    def finish_requests(self, request_ids: List[str],
                        status: RequestStatus) -> None:
        self._shard.finish_requests(_pack_blob((request_ids, status),
                                               self._sharding))

    def update_draft_token_ids(self, draft_token_ids: Any) -> None:
        self._shard.update_draft_token_ids(_pack_blob(draft_token_ids,
                                                      self._sharding))

    # ---- step protocol --------------------------------------------------

    # def schedule(self) -> Any:
    #     return _unpack(self._shard.schedule(self._pin))

    def get_grammar_bitmask(self, scheduler_output: Any) -> Any:
        return _unpack(
            self._shard.get_grammar_bitmask(
                _pack_blob(scheduler_output, self._sharding)))

    def poll(
        self,
        prev_next_tokens: jax.Array,
        has_prev: jax.Array,
    ) -> Dict[str, Any]:
        """FIXED-spec half of the boundary protocol.

        Calls the sidecar's `poll` (consume prev tokens → drain → schedule →
        decide work) and returns the unpacked metadata dict:
          - has_work: bool
          - drained: Optional[EngineCoreOutputs] (new_token_ids reattached
            from the JAX leaves; req_id strings still interned as int handles
            — the controller restores them in `_funnel_drained`)
          - num_reqs / num_tokens: bucket sizes for `prepare_step`'s pins
            (0 when idle)

        The returned pytree shape is invariant across idle/active, so no
        idle-pytree filler is needed.
        """
        (drained_tokens_jax, drained_offsets_jax,
         blob) = self._shard.poll(prev_next_tokens, has_prev)
        meta = _unpack(blob)
        if meta.get("tokens_stripped", False):
            n_outputs = int(meta.get("n_drained_outputs", 0))
            if n_outputs > 0:
                # Local materialize (arrays already on this rank's CPU
                # sharding), not a cross-host fetch.
                tokens_np = np.asarray(drained_tokens_jax)
                offsets_np = np.asarray(drained_offsets_jax)
                _reattach_drained_tokens(
                    meta["drained"], tokens_np,
                    offsets_np[:n_outputs + 1].tolist())
        return meta

    def prepare_step(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> Tuple[jax.Array, Any, Any, jax.Array]:
        """DATA half — call only when the preceding `poll` reported work.

        Builds dummy input pins of shape `(num_reqs,)` / `(num_tokens,)` so
        colocated_python specializes `prepare_step` per bucket (the pins'
        SHAPES are the bucket key; their values are unused). Returns the
        prepared model inputs `(input_ids, attn_md, sampling_md,
        logits_indices)` as native pytree leaves.
        """
        pin_reqs = jax.device_put(np.zeros(num_reqs, dtype=np.int8),
                                  self._sharding)
        pin_tokens = jax.device_put(np.zeros(num_tokens, dtype=np.int8),
                                    self._sharding)
        (input_ids, attn_md, sampling_md,
         logits_indices) = self._shard.prepare_step(pin_reqs, pin_tokens)
        return input_ids, attn_md, sampling_md, logits_indices

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
        # that the sidecar uses for its CPU mesh.
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
            executor_class=DisaggExecutor, # todo(wenxindong): try use vllm executor
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


    def execute_with_cpu_prep(
        self,
        input_ids: jax.Array,
        attn_metadata: Any,
        sampling_metadata: Any,
        logits_indices: jax.Array,
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

        # Direct per-device CPU→TPU transfers for every JAX leaf.
        with (
            jax.transfer_guard_device_to_host("disallow_explicit"),
            jax.transfer_guard_host_to_device("disallow_explicit"),
        ):
            attn_md = jax.tree.map(
                lambda x: jax.device_put(x, dp_attn) if isinstance(x, jax.Array)
                else x,
                attn_metadata)
            sampling_md = jax.tree.map(
                lambda x: jax.device_put(x, replicated) if isinstance(x, jax.Array)
                else x,
                sampling_metadata)
            input_ids_tpu = jax.device_put(input_ids, dp_attn)
            input_positions_tpu = attn_md.input_positions
            logits_indices_tpu = jax.device_put(logits_indices, replicated)

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

        # Direct per-device TPU → sidecar-CPU transfer. JAX handles the
        # cross-host copy natively (per the user's spike in
        # examples/colocated_dp/spike_devices_inside_sidecar.py). Returned
        # `jax.Array`s flow back to the sidecar through colocated_python
        # as native pytree leaves — no cloudpickle of the bytes.
        target = self.sidecar_cpu_sharding
        assert target is not None, (
            "RankExecutor.sidecar_cpu_sharding not set — DPEngineCore.__init__ "
            "should have passed it when constructing this RankExecutor.")
        with jax.transfer_guard_device_to_host("disallow_explicit"), \
             jax.transfer_guard_host_to_device("disallow_explicit"):
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
    # split devices into contiguous groups of 8, assuming each host has 8 chips with
    # consecutive IDs. 
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
            shard = ShardWrapper(*sidecar_init_args)

            rank_max_num_reqs = (
                rexec.engine.model_executor.driver_worker  # type: ignore[attr-defined]
                .model_runner.max_num_reqs)
            # TODO("wenxindong"): remove rank_max_num_reqs
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
        # req_id ↔ int handle, used to intern req_id strings out of the
        # cross-boundary drained blob (see `_intern_drained_req_ids`).
        # Assigned at add_request, freed when the request finishes.
        self._handle_by_id: Dict[str, int] = {}
        self._id_by_handle: Dict[int, str] = {}
        self._next_handle: int = 0
        self._handle_lock = threading.Lock()
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

        client = self._sched_clients[i]
        rexec = self._rank_execs[i]
        # v3-pure: `prev_next_tokens` is a JAX array (shape (max_num_reqs,),
        # int32) on this rank's sidecar CPU sharding. It rides across the
        # boundary as a native pytree leaf — no cloudpickle of its buffer.
        # `has_prev` is a uint8 JAX scalar (0=first call, 1=subsequent).
        prev_next_tokens, has_prev = client.first_call_state()
        while not self._stop.is_set():
            # 1) poll: fixed-spec call — consume prev tokens, drain, schedule,
            #    decide work. Returns drained outputs + has_work + bucket sizes.
            try:
                meta = client.poll(prev_next_tokens, has_prev)
            except Exception as e:
                if self._stop.is_set():
                    break
                logger.exception("rank %d poll() failed: %s", i, e)
                time.sleep(0.05)
                prev_next_tokens, has_prev = client.first_call_state()
                continue
            self._funnel_drained(i, meta.get("drained"))
            if not meta.get("has_work", False):
                # Sidecar idle. Reset so the next poll doesn't postprocess
                # stale tokens; yield briefly.
                prev_next_tokens, has_prev = client.first_call_state()
                time.sleep(0.001)
                continue
            # 2) prepare_step: bucket-keyed data call — fetch the prepared
            #    model inputs, then dispatch on TPU. The pins encode the
            #    bucket so colocated re-specializes per bucket with a
            #    consistent output spec.
            try:
                (input_ids, attn_md, sampling_md,
                 logits_indices) = client.prepare_step(meta["num_reqs"],
                                                       meta["num_tokens"])
                # TPU dispatch — pytree leaves cross natively. Returns
                # next_tokens as a JAX array already on the sidecar's CPU
                # sharding, ready to flow back via the next poll() call.
                next_tokens, _logprobs = rexec.execute_with_cpu_prep(
                    input_ids=input_ids,
                    attn_metadata=attn_md,
                    sampling_metadata=sampling_md,
                    logits_indices=logits_indices,
                )
            except Exception as e:
                logger.exception("rank %d prepare_step/execute failed: %s",
                                 i, e)
                time.sleep(0.05)
                prev_next_tokens, has_prev = client.first_call_state()
                continue
            prev_next_tokens = next_tokens
            has_prev = client._has_prev_true_cache()

    def _funnel_drained(
            self, i: int,
            engine_outs: Optional[EngineCoreOutputs]) -> None:
        """Push rank ``i``'s `EngineCoreOutputs` into the funnel and reconcile
        load/ownership tracking. No boundary crossing — this runs on whatever
        the controller already got from ``client.poll``.

        `engine_outs` is a single bare `EngineCoreOutputs` or None (single-
        client + single-per-step simplification — the client dict and the
        per-step list were both collapsed on the sidecar)."""
        if engine_outs is None:
            return
        # Swap int handles back to req_id strings before vLLM sees them.
        _restore_drained_req_ids(engine_outs, self._id_by_handle)
        if engine_outs.finished_requests:
            with self._load_lock:
                self._load[i][0] = max(
                    0,
                    self._load[i][0] - len(engine_outs.finished_requests))
            for rid in engine_outs.finished_requests:
                self._owner.pop(rid, None)
                # Free the handle now that the request is done.
                with self._handle_lock:
                    h = self._handle_by_id.pop(rid, None)
                    if h is not None:
                        self._id_by_handle.pop(h, None)
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
        rid = request.request_id
        self._owner[rid] = idx
        with self._handle_lock:
            handle = self._next_handle
            self._next_handle += 1
            self._handle_by_id[rid] = handle
            self._id_by_handle[handle] = rid
        with self._load_lock:
            self._load[idx][1] += 1
        # Fire-and-forget; the blob's CPU sharding pins the call to the host.
        self._sched_clients[idx].add_request(request, request_wave, handle)

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
        # Re-wrap the bare EngineCoreOutputs into the client-indexed dict vLLM
        # expects. Single-client simplification → key is always 0.
        return {0: engine_outs}, True

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
