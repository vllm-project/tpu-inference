from typing import TYPE_CHECKING, Any

import tpu_commons.runner.jax.input_batch_jax

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    from tpu_commons.runner.jax.input_batch_jax import (CachedRequestState,
                                                        InputBatch)


class UpdateStatesMixin:
    requests: dict[str, "CachedRequestState"]
    encoder_cache: dict[str, dict[int, Any]]
    input_batch: "InputBatch"

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
            True if there is a new/resumed/paused/finished request.
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            self.requests[req_id] = self._create_cached_request(new_req_data)
            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                if isinstance(req_state.block_ids[0], list):
                    for block_ids, new_block_ids in zip(req_state.block_ids,
                                                        req_data.new_block_ids,
                                                        strict=True):
                        block_ids.extend(new_block_ids)
                else:
                    req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = list(req_data.new_block_ids)

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[
                req_index] = req_data.num_computed_tokens
            if isinstance(self.input_batch.block_table, list):
                self.input_batch.block_table[0].append_row(
                    req_data.new_block_ids[0], req_index)
            else:
                self.input_batch.block_table.append_row(
                    list(req_data.new_block_ids), req_index)

        # Add the new or resumed requests to the persistent batch.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            req_index = removed_req_indices.pop(
            ) if removed_req_indices else None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        return bool(unscheduled_req_ids) or bool(req_ids_to_add)

    def _get_cached_request_class(self):
        return tpu_commons.runner.jax.input_batch_jax.CachedRequestState

    def _create_cached_request(
            self, new_req_data: "NewRequestData") -> "CachedRequestState":
        assert new_req_data.sampling_params is not None, \
            "Pooling is not supported in TPU yet"
        return self._get_cached_request_class()(
            req_id=new_req_data.req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            mm_inputs=new_req_data.mm_inputs,
            mm_positions=new_req_data.mm_positions,
            sampling_params=new_req_data.sampling_params,
            pooling_params=None,
            generator=None,
            block_ids=list(new_req_data.block_ids),
            num_computed_tokens=new_req_data.num_computed_tokens,
            output_token_ids=[],
            lora_request=new_req_data.lora_request,
        )
