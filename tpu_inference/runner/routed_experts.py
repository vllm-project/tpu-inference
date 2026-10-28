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
"""Worker-side routed-experts (R3) auxiliary output for the TPU runner.

vLLM's scheduler delegates R3 assembly to a worker-side AuxOutput connector
and only consumes ``ModelRunnerOutput.aux_output_connector_output``. The
upstream worker connector is tied to the GPU model runner, so the TPU runner
keeps R3 in a CPU buffer indexed by physical KV-cache slot. Prefix-cache hits
re-expose the same slots, so cached-prefix routing survives across requests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.aux_output_connector.connector import (
    AuxOutputConnectorMetadata, AuxRequestOutput)
from vllm.v1.kv_cache_interface import KVCacheConfig, is_full_attention_spec


def get_routed_experts_attn_gid(kv_cache_config: KVCacheConfig) -> int:
    """Return the full-attention KV cache group used for routed experts."""
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if is_full_attention_spec(group.kv_cache_spec):
            return gid
    raise ValueError(
        "Routed-experts capture requires a full-attention KV cache group.")


def reconstruct_slots(block_ids: List[int], num_tokens: int, block_size: int,
                      start_pos: int) -> np.ndarray:
    """Physical KV-cache slots of positions [start_pos, start_pos+num_tokens)."""
    if num_tokens <= 0:
        return np.array([], dtype=np.int32)
    assert start_pos >= 0, (
        f"[routed-experts] start_pos must be non-negative, got {start_pos}")

    pos = np.arange(start_pos, start_pos + num_tokens, dtype=np.int32)
    block_idx = pos // block_size
    if not block_ids:
        return np.zeros_like(pos)
    # Pad with block 0 so positions past the known blocks don't IndexError.
    block_ids_arr = np.zeros(max(len(block_ids),
                                 int(block_idx[-1]) + 1),
                             dtype=np.int32)
    block_ids_arr[:len(block_ids)] = block_ids
    return block_ids_arr[block_idx] * block_size + pos % block_size


class RoutedExpertsSlotBuffer:
    """Routed-experts rows indexed by one DP rank's physical KV-cache slots."""

    def __init__(self, num_slots: int, block_size: int,
                 dtype: np.dtype) -> None:
        self.num_slots = num_slots
        self.block_size = block_size
        self.dtype = dtype
        # Allocated on first store: the captured (num_layers, top_k) profile
        # is only known from the model's expert-indices output.
        self._by_slot: Optional[np.ndarray] = None

    def store(self, rows: np.ndarray, slots: np.ndarray) -> None:
        if len(rows) == 0:
            return
        if self._by_slot is None:
            self._by_slot = np.zeros((self.num_slots, *rows.shape[1:]),
                                     dtype=self.dtype)
        self._by_slot[slots] = rows

    def reset(self) -> None:
        self._by_slot = None

    def get(self, block_ids: List[int], token_start: int,
            token_end: int) -> np.ndarray:
        assert self._by_slot is not None, (
            "[routed-experts] read before any routing data was stored")
        token_start = min(max(token_start, 0), token_end)
        slots = reconstruct_slots(block_ids, token_end - token_start,
                                  self.block_size, token_start)
        # Fancy indexing copies, so the result survives later stores.
        return self._by_slot[slots]


@dataclass
class RoutedExpertsStepEntry:
    """One request's routed experts from a single executed step."""

    req_id: str
    # (num_executed, num_layers, top_k); row i is position token_start + i.
    rows: np.ndarray
    token_start: int
    # Leading rows accepted into the sequence (spec decode may reject a
    # suffix); 0 when the request sampled nothing this step.
    num_accepted: int
    block_ids: List[int]
    dp_rank: int = 0


@dataclass
class _RequestState:
    emit_cursor: int
    pending_steps: int = 0
    finished: bool = False


class TPUAuxOutputWorker:
    """Builds ``aux_output_connector_output`` from per-step routed experts."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig,
                 block_size: int, dp_size: int) -> None:
        num_experts = vllm_config.model_config.get_num_experts()
        dtype = np.dtype(np.uint8 if num_experts <= 256 else np.uint16)
        num_blocks_per_rank = kv_cache_config.num_blocks // dp_size
        self.block_size = block_size
        self._buffers = [
            RoutedExpertsSlotBuffer(num_blocks_per_rank * block_size,
                                    block_size, dtype) for _ in range(dp_size)
        ]
        self._requests: Dict[str, _RequestState] = {}
        self._generation = 0

    def begin_step(self,
                   metadata: Optional[AuxOutputConnectorMetadata]) -> None:
        """Apply one scheduler step's request start and finish events."""
        if metadata is None:
            return
        if metadata.generation > self._generation:
            # The scheduler bumps the generation after a prefix-cache reset,
            # which invalidates every slot's contents.
            assert not any(s.pending_steps for s in self._requests.values()), (
                "[routed-experts] generation changed with output in flight")
            for buffer in self._buffers:
                buffer.reset()
            self._requests.clear()
            self._generation = metadata.generation

        for req_id, emit_start in metadata.requests.items():
            # Async scheduling makes the scheduler's cursor optimistic, so it
            # only seeds requests the worker has not seen yet.
            state = self._requests.setdefault(req_id,
                                              _RequestState(emit_start))
            state.pending_steps += 1
        for req_id in metadata.finished_requests:
            state = self._requests.get(req_id)
            if state is None:
                continue
            state.finished = True
            if state.pending_steps == 0:
                del self._requests[req_id]

    def process_step(
        self,
        step_req_ids: Sequence[str],
        entries: Sequence[RoutedExpertsStepEntry],
    ) -> Dict[str, AuxRequestOutput]:
        """Store one step's rows and return each sampling request's new R3.

        ``step_req_ids`` are all requests executed in the step, including
        those without entries, so their in-flight counts are released.
        """
        # Store the whole step first: a request may read a prefix block that
        # another request in the same step just computed.
        for entry in entries:
            slots = reconstruct_slots(entry.block_ids, len(entry.rows),
                                      self.block_size, entry.token_start)
            self._buffers[entry.dp_rank].store(entry.rows, slots)

        outputs: Dict[str, AuxRequestOutput] = {}
        for entry in entries:
            state = self._requests.get(entry.req_id)
            if state is None or entry.num_accepted <= 0:
                continue
            token_end = entry.token_start + entry.num_accepted
            if state.emit_cursor > token_end:
                continue
            rows = self._buffers[entry.dp_rank].get(entry.block_ids,
                                                    state.emit_cursor,
                                                    token_end)
            outputs[entry.req_id] = AuxRequestOutput(state.emit_cursor, rows)
            state.emit_cursor = token_end

        for req_id in step_req_ids:
            state = self._requests.get(req_id)
            if state is None or state.pending_steps == 0:
                continue
            state.pending_steps -= 1
            if state.finished and state.pending_steps == 0:
                del self._requests[req_id]
        return outputs
