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

from typing import Dict

import jax
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from tpu_inference.logger import init_logger
from tpu_inference.runner.input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


class PersistentBatchManager:

    def __init__(self, requests: Dict[str, CachedRequestState],
                 input_batch: InputBatch, encoder_cache: Dict[str,
                                                              'jax.Array'],
                 uses_mrope: bool, model_config, is_last_rank: bool):
        self.requests = requests
        self.input_batch = input_batch
        self.encoder_cache = encoder_cache
        self.uses_mrope = uses_mrope
        self.model_config = model_config
        self.is_last_rank = is_last_rank

    def _reorder_batch(self, scheduler_output: "VllmSchedulerOutput") -> int:
        """ Reorder the sheduled requests to RPA kernel friendly distribution
        (decode_only, fixed_chunked_prefill_only, mixed) and set the request
        distribution accordingly.

        With speculative decoding the order is three segments:
        [1-token decodes][spec verify windows][prefill/mixed]. Ragged paged
        attention keeps its 1-token decode front segment, while the GDN
        kernel's windowed mode covers the first two segments contiguously
        (see `AttentionMetadata.mamba_request_distribution`).

        Returns:
            The number of swaps in requests.
        """
        # Note(jevinjiang): currently we only consider decode_only.
        num_reqs = self.input_batch.num_reqs
        swap_cnt = 0
        if num_reqs <= 0:
            return swap_cnt
        # If total_num_scheduled_tokens == num_reqs, every request
        # is scheduled for exactly 1 token (all decode). No reordering needed.
        if scheduler_output.total_num_scheduled_tokens == num_reqs:
            num_decode = num_reqs
            self.input_batch.request_distribution = [
                num_decode, num_decode, num_reqs
            ]
            return swap_cnt

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        def segment(req_id: str) -> int:
            # 0: 1-token decode, 1: speculative verify window,
            # 2: prefill/mixed.
            if scheduler_output.num_scheduled_tokens[req_id] == 1:
                return 0
            if req_id in spec_decode_tokens:
                return 1
            return 2

        def partition(start: int, end: int, bound: int) -> tuple[int, int]:
            """Two-pointer partition of [start, end]: requests with
            segment <= bound before the rest. Returns (first index of the
            second part, swaps)."""
            nonlocal swap_cnt
            i, j = start, end
            while i < j:
                if segment(self.input_batch.req_ids[i]) <= bound:
                    i += 1
                elif segment(self.input_batch.req_ids[j]) > bound:
                    j -= 1
                else:
                    self.input_batch.swap_states(i, j)
                    i += 1
                    j -= 1
                    swap_cnt += 1
            if i == j and segment(self.input_batch.req_ids[i]) <= bound:
                i += 1
            return i

        # Pass 1: 1-token decode requests to the front.
        num_decode = partition(0, num_reqs - 1, 0)
        # Pass 2: speculative verify windows before prefill/mixed requests.
        num_windowed = num_decode
        if num_decode < num_reqs:
            num_windowed = partition(num_decode, num_reqs - 1, 1)

        self.input_batch.request_distribution = [
            num_decode, num_windowed, num_reqs
        ]

        return swap_cnt

    def update_states(self, scheduler_output: "VllmSchedulerOutput",
                      get_mrope_input_positions_fn) -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input TPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the TPU.
        """
        # Remove finished requests from the cached states.
        finished_req_states = {}
        for req_id in scheduler_output.finished_req_ids:
            finished_req_states[req_id] = self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                req_state = finished_req_states.get(req_id)
                if req_state is not None:
                    req_state.mamba_state_slot = None
                removed_req_indices.append(req_index)
            else:
                req_state = finished_req_states.get(req_id)
                if req_state is not None:
                    self.input_batch.release_mamba_slot(
                        req_state.mamba_state_slot)
                    req_state.mamba_state_slot = None

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = set(
            getattr(scheduler_output.scheduled_cached_reqs, "resumed_req_ids",
                    ()) or ())
        preempted_req_ids = set(
            getattr(scheduler_output, "preempted_req_ids", ()) or ())
        reset_mamba_req_ids = preempted_req_ids | resumed_req_ids

        # A request can be temporarily removed from the persistent batch while
        # keeping its physical mamba slot. If it is then preempted or resumed
        # before being re-added, the active-batch removal loop below cannot see
        # it. Reset the preserved slot here so a recomputed request cannot read
        # stale recurrent state from its old slot.
        for req_id in reset_mamba_req_ids:
            if req_id in self.input_batch.req_id_to_index:
                continue
            req_state = self.requests.get(req_id)
            if req_state is None:
                continue
            self.input_batch.release_mamba_slot(req_state.mamba_state_slot)
            req_state.mamba_state_slot = None

        # Usually resumed requests are not present in the persistent batch.
        # Forced preemption can make a resumed request still appear there; clear
        # it first so it is re-added through the normal resumed-request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids -
                                                resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.req_id_to_index.get(req_id)
            reset_mamba_slot = (req_id in preempted_req_ids
                                or req_id in resumed_req_ids)
            if req_index is not None:
                req_state = self.requests.get(req_id)
                if reset_mamba_slot:
                    if req_state is not None:
                        req_state.mamba_state_slot = None
                else:
                    self.requests[req_id].mamba_state_slot = int(
                        self.input_batch.mamba_state_indices_cpu[req_index])
            req_index = self.input_batch.remove_request(
                req_id, free_mamba_slot=reset_mamba_slot)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=new_req_data.pooling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self.requests[req_id].mrope_positions, self.requests[
                    req_id].mrope_position_delta = get_mrope_input_positions_fn(
                        self.requests[req_id].prompt_token_ids,
                        self.requests[req_id].mm_features,
                    )

        # Update the states of the running/resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if not self.is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (self.input_batch.num_prompt_tokens[req_index] +
                               num_output_tokens)
                    self.input_batch.num_tokens[req_index] = end_idx
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[
                req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not self.is_last_rank:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        dp_rank_map = getattr(scheduler_output, 'assigned_dp_rank', None)
        if not isinstance(dp_rank_map, dict):
            dp_rank_map = None
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            dp_rank = dp_rank_map.get(req_id, 0) if dp_rank_map else 0
            self.input_batch.add_request(req_state, req_index, dp_rank=dp_rank)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        batch_changed = len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0
        # TODO(jevinjiang): I assume we do not need to set batch_changed to true if just swapping requests.
        swap_cnt = self._reorder_batch(scheduler_output)
        if (isinstance(self.input_batch, InputBatch)
                and self.input_batch.has_mamba_layers
                and (batch_changed or swap_cnt > 0)):
            self.input_batch.assert_mamba_state_invariants(
                self.requests, dp_rank_map)
        return batch_changed
