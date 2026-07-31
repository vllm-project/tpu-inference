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

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from tpu_inference.logger import init_logger
from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

MambaLayerStates = tuple[jax.Array, ...]
MambaGroupStates = tuple[MambaLayerStates, ...]
MambaStates = tuple[MambaGroupStates, ...]


@jax.jit(donate_argnames=("states_by_group", ))
def _copy_state_blocks(
    states_by_group: MambaStates,
    src_block_ids_by_group: tuple[jax.Array, ...],
    dst_block_ids_by_group: tuple[jax.Array, ...],
) -> MambaStates:
    """Copies every Mamba state leaf in one compiled operation."""

    def copy_state(
        state: jax.Array,
        src_block_ids: jax.Array,
        dst_block_ids: jax.Array,
    ) -> jax.Array:
        return state.at[dst_block_ids].set(state[src_block_ids])

    return tuple(
        tuple(
            tuple(
                copy_state(state, src_block_ids, dst_block_ids)
                for state in layer_states) for layer_states in group_states)
        for group_states, src_block_ids, dst_block_ids in zip(
            states_by_group,
            src_block_ids_by_group,
            dst_block_ids_by_group,
            strict=True,
        ))


class MambaStateManager:
    """Maintains scheduler-addressable Mamba state for align-mode caching."""

    def __init__(self, runner: "TPUModelRunner") -> None:
        self.runner = runner
        self.kv_cache_config: KVCacheConfig | None = None
        self.mamba_groups: dict[int, MambaSpec] = {}
        self.mamba_state_idx: dict[str, int] = {}
        self.current_state_block_ids: dict[int, dict[str, int]] = {}
        self.enabled = False

    def initialize(self, kv_cache_config: KVCacheConfig) -> None:
        self.kv_cache_config = kv_cache_config
        self.mamba_groups = {
            gid: group.kv_cache_spec
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        }
        self.mamba_state_idx.clear()
        self.current_state_block_ids = {gid: {} for gid in self.mamba_groups}
        self.enabled = bool(
            self.mamba_groups
            and self.runner.cache_config.enable_prefix_caching
            and not self.runner.kv_cache_manager.uses_compact_mamba_state)
        if not self.enabled:
            return

        cache_mode = self.runner.cache_config.mamba_cache_mode
        if cache_mode != "align":
            raise NotImplementedError(
                "TPU prefix caching for Mamba layers requires "
                "mamba_cache_mode='align'.")
        if self.runner.speculative_config is not None:
            raise NotImplementedError(
                "TPU Mamba prefix caching does not yet support speculative "
                "decoding.")

        specs = list(self.mamba_groups.values())
        first_spec = specs[0]
        if any(spec.block_size != first_spec.block_size for spec in specs[1:]):
            raise ValueError(
                "All Mamba KV-cache groups must use one block size.")
        logger.info(
            "Mamba prefix-cache state restore enabled for %d groups, %d "
            "layers, block_size=%d", len(self.mamba_groups),
            sum(
                len(kv_cache_config.kv_cache_groups[gid].layer_names)
                for gid in self.mamba_groups), first_spec.block_size)

    def precompile_copy_state_blocks(self) -> MambaStates | None:
        """Compiles the fixed Mamba state-copy tree before serving."""
        if not self.enabled:
            return None

        # Copying block 1 onto itself exercises the serving shape without
        # reading or writing the reserved null block.
        copies_by_group: dict[int, list[tuple[int, int, int]]] = {
            gid: []
            for gid in self.mamba_groups
        }
        copies_by_group[next(iter(self.mamba_groups))] = [(1, 1, 0)]
        return self._apply_copies(copies_by_group)

    def reset(self) -> None:
        self.mamba_state_idx.clear()
        for state_block_ids in self.current_state_block_ids.values():
            state_block_ids.clear()

    def is_mamba_group(self, kv_cache_gid: int) -> bool:
        return kv_cache_gid in self.mamba_groups

    def get_current_state_block_id(self, kv_cache_gid: int,
                                   req_id: str) -> int:
        try:
            return self.current_state_block_ids[kv_cache_gid][req_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Mamba state for request {req_id!r}, group {kv_cache_gid} "
                "was not prepared before attention metadata construction."
            ) from exc

    def update_request_lifecycle(self,
                                 scheduler_output: "SchedulerOutput") -> None:
        """Drops host bookkeeping for requests whose block tables changed."""
        if not self.enabled:
            return

        resumed_req_ids = set(
            scheduler_output.scheduled_cached_reqs.resumed_req_ids or ())
        preempted_req_ids = set(scheduler_output.preempted_req_ids or ())
        finished_req_ids = set(scheduler_output.finished_req_ids or ())
        inactive_req_ids = finished_req_ids | preempted_req_ids | resumed_req_ids
        for req_id in inactive_req_ids:
            self.mamba_state_idx.pop(req_id, None)
            for state_block_ids in self.current_state_block_ids.values():
                state_block_ids.pop(req_id, None)

    def preprocess(self, scheduler_output: "SchedulerOutput") -> None:
        """Restores a cached boundary state into each current running block."""
        if not self.enabled:
            return

        self.update_request_lifecycle(scheduler_output)

        for state_block_ids in self.current_state_block_ids.values():
            state_block_ids.clear()

        first_spec = next(iter(self.mamba_groups.values()))
        block_size = first_spec.block_size
        num_speculative_blocks = first_spec.num_speculative_blocks
        copies_by_group: dict[int, list[tuple[int, int, int]]] = {
            gid: []
            for gid in self.mamba_groups
        }
        assigned_dp_rank = getattr(scheduler_output, "assigned_dp_rank", {})

        for req_id in self.runner.input_batch.req_ids[:self.runner.input_batch.
                                                      num_reqs]:
            req_state = self.runner.requests[req_id]
            prev_state_idx = self.mamba_state_idx.get(req_id)
            if prev_state_idx is None:
                prev_state_idx = (req_state.num_computed_tokens -
                                  1) // block_size

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            num_blocks = (cdiv(
                req_state.num_computed_tokens + num_scheduled_tokens,
                block_size,
            ) + num_speculative_blocks)
            curr_state_idx = num_blocks - 1 - num_speculative_blocks
            self.mamba_state_idx[req_id] = curr_state_idx
            dp_rank = assigned_dp_rank.get(req_id, 0)

            for gid in self.mamba_groups:
                block_ids = req_state.block_ids[gid]
                if curr_state_idx >= len(block_ids):
                    raise RuntimeError(
                        f"Request {req_id!r} needs Mamba block index "
                        f"{curr_state_idx}, but group {gid} has only "
                        f"{len(block_ids)} blocks.")
                curr_block_id = block_ids[curr_state_idx]
                self.current_state_block_ids[gid][req_id] = curr_block_id

                if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
                    if prev_state_idx >= len(block_ids):
                        raise RuntimeError(
                            f"Request {req_id!r} has stale Mamba state index "
                            f"{prev_state_idx} for group {gid}.")
                    copies_by_group[gid].append(
                        (block_ids[prev_state_idx], curr_block_id, dp_rank))

        self._apply_copies(copies_by_group)

    def postprocess(self, scheduler_output: "SchedulerOutput") -> None:
        """Snapshots a running state when this step reaches an aligned boundary."""
        if not self.enabled:
            return

        first_spec = next(iter(self.mamba_groups.values()))
        block_size = first_spec.block_size
        assigned_dp_rank = getattr(scheduler_output, "assigned_dp_rank", {})
        copies_by_group: dict[int, list[tuple[int, int, int]]] = {
            gid: []
            for gid in self.mamba_groups
        }

        # Without speculative decoding one accepted token represents the bonus
        # token, so new_num_computed_tokens equals the state produced by this
        # forward. This mirrors vLLM's postprocess_mamba arithmetic.
        for req_id in self.runner.input_batch.req_ids[:self.runner.input_batch.
                                                      num_reqs]:
            req_state = self.runner.requests[req_id]
            num_tokens_running_state = (
                req_state.num_computed_tokens +
                scheduler_output.num_scheduled_tokens[req_id])
            aligned_num_tokens = num_tokens_running_state // block_size * block_size
            if aligned_num_tokens < num_tokens_running_state:
                continue

            src_state_idx = self.mamba_state_idx[req_id]
            dst_state_idx = aligned_num_tokens // block_size - 1
            if src_state_idx == dst_state_idx:
                continue

            dp_rank = assigned_dp_rank.get(req_id, 0)
            for gid in self.mamba_groups:
                block_ids = req_state.block_ids[gid]
                copies_by_group[gid].append(
                    (block_ids[src_state_idx], block_ids[dst_state_idx],
                     dp_rank))

        self._apply_copies(copies_by_group)

    def _apply_copies(
        self,
        copies_by_group: dict[int, list[tuple[int, int, int]]],
    ) -> MambaStates | None:
        """Copies all Mamba layer states in one compiled invocation."""
        if not any(copies_by_group.values()):
            return

        max_num_reqs = self.runner.max_num_reqs
        dp_size = self.runner.dp_size
        replicated = NamedSharding(self.runner.mesh, PartitionSpec())
        cache_indices_by_group: list[tuple[int, ...]] = []
        states_by_group: list[MambaGroupStates] = []
        src_block_ids_by_group: list[jax.Array] = []
        dst_block_ids_by_group: list[jax.Array] = []
        seen_cache_indices: set[int] = set()

        # Include every configured group so the donated pytree stays fixed when
        # only a subset of groups has copies in a given step.
        for gid in self.mamba_groups:
            copies = copies_by_group.get(gid, ())
            group = self.kv_cache_config.kv_cache_groups[gid]
            missing_layers = [
                layer_name for layer_name in group.layer_names
                if layer_name not in self.runner.layer_name_to_kvcache_index
            ]
            if missing_layers:
                raise RuntimeError(
                    "Mamba state restore is missing KV-cache mappings for: " +
                    ", ".join(missing_layers))
            cache_indices = [
                self.runner.layer_name_to_kvcache_index[layer_name]
                for layer_name in group.layer_names
            ]
            if len(set(cache_indices)) != len(cache_indices):
                raise RuntimeError(
                    "Mamba state restore requires one KV cache per layer; "
                    f"group {gid} contains duplicate cache indices.")
            duplicate_cache_indices = seen_cache_indices.intersection(
                cache_indices)
            if duplicate_cache_indices:
                raise RuntimeError(
                    "Mamba state restore requires each KV cache to belong to "
                    "one group; duplicate cache indices: " + ", ".join(
                        str(index)
                        for index in sorted(duplicate_cache_indices)))
            seen_cache_indices.update(cache_indices)

            destination_keys = [(dp_rank, dst) for _, dst, dp_rank in copies]
            if any(src == 0 or dst == 0 for src, dst, _ in copies):
                raise RuntimeError(
                    "Mamba state restore cannot copy to or from the null block."
                )
            if len(set(destination_keys)) != len(destination_keys):
                raise RuntimeError(
                    "Mamba state restore received duplicate destination blocks "
                    f"for group {gid}.")

            first_states = self.runner.kv_caches[cache_indices[0]]
            if not isinstance(first_states, tuple):
                raise TypeError("Mamba layers must have tuple state caches.")
            local_num_blocks = first_states[0].shape[0] // dp_size
            # Real copies exclude block 0, so trailing 0->0 entries are no-ops.
            src_block_ids = np.zeros(max_num_reqs, dtype=np.int32)
            dst_block_ids = np.zeros(max_num_reqs, dtype=np.int32)
            for index, (src, dst, dp_rank) in enumerate(copies):
                if src >= local_num_blocks or dst >= local_num_blocks:
                    raise IndexError(
                        f"Mamba block copy {src}->{dst} is outside rank "
                        f"{dp_rank}'s {local_num_blocks}-block shard.")
                rank_offset = dp_rank * local_num_blocks
                src_block_ids[index] = rank_offset + src
                dst_block_ids[index] = rank_offset + dst

            src_device = device_array(self.runner.mesh,
                                      src_block_ids,
                                      sharding=replicated)
            dst_device = device_array(self.runner.mesh,
                                      dst_block_ids,
                                      sharding=replicated)
            group_states: list[MambaLayerStates] = []
            for cache_idx in cache_indices:
                states = self.runner.kv_caches[cache_idx]
                if not isinstance(states, tuple):
                    raise TypeError(
                        "Mamba layers must have tuple state caches.")
                if states[0].shape[0] // dp_size != local_num_blocks:
                    raise ValueError(
                        "Mamba layers in one cache group must have equal block "
                        "counts.")
                group_states.append(states)

            cache_indices_by_group.append(tuple(cache_indices))
            states_by_group.append(tuple(group_states))
            src_block_ids_by_group.append(src_device)
            dst_block_ids_by_group.append(dst_device)

        copied_states_by_group = _copy_state_blocks(
            tuple(states_by_group),
            tuple(src_block_ids_by_group),
            tuple(dst_block_ids_by_group),
        )
        for cache_indices, group_states in zip(cache_indices_by_group,
                                               copied_states_by_group,
                                               strict=True):
            for cache_idx, states in zip(cache_indices,
                                         group_states,
                                         strict=True):
                self.runner.kv_caches[cache_idx] = states
        return copied_states_by_group
