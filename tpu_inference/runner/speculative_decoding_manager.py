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

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.utils import SpecDecodeMetadata
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from tpu_inference.layers.common.attention_metadata import \
        AttentionMetadata
    from tpu_inference.runner.tpu_runner import TPUModelRunner


class SpeculativeDecodingManager:

    def __init__(self, runner: TPUModelRunner):
        self.runner = runner
        # Cached draft tokens.
        self._draft_token_ids: Optional[list[list[int]]] = None
        self._req_indices_dp: Optional[dict] = None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.runner.input_batch.req_ids
        draft_token_ids = self._draft_token_ids

        if self._req_indices_dp is None:
            self._draft_token_ids = None
            self._req_indices_dp = None
            return DraftTokenIds(req_ids, draft_token_ids)

        # reorders per-rank outputs back to the original batch ordering
        reorded_draft_token_ids = [[] for _ in range(len(draft_token_ids))]
        max_num_reqs_per_dp_rank = self.runner.max_num_reqs // self.runner.dp_size
        for rank in range(self.runner.dp_size):
            req_indices = self._req_indices_dp[rank]
            for j, req_idx in enumerate(req_indices):
                reorded_draft_token_ids[req_idx] = draft_token_ids[
                    j + rank * max_num_reqs_per_dp_rank]
        self._draft_token_ids = None
        self._req_indices_dp = None
        return DraftTokenIds(req_ids, reorded_draft_token_ids)

    def propose_draft_token_ids(
        self,
        sampled_output: jnp.ndarray,
        logits_indices_selector: np.ndarray,
        last_sampled_token_id: jnp.ndarray,
        num_rejected_tokens: jnp.ndarray,
        discard_sampled_tokens_req_indices: list,
        aux_hidden_states: Optional[tuple[jnp.ndarray, ...]],
        attn_metadata: AttentionMetadata,
        async_scheduling: bool,
        spec_decode_metadata: SpecDecodeMetadata,
        scheduler_output: Optional[VllmSchedulerOutput] = None,
        input_ids: Optional[jnp.ndarray] = None,
        hidden_states: Optional[jnp.ndarray] = None,
    ) -> None:
        if async_scheduling:
            assert self.runner.speculative_config.use_eagle(
            ), "async scheduling is only supported with eagle3 spec decoding"
        if self.runner.speculative_config.method == "ngram":
            assert isinstance(self.runner.drafter, NgramProposer)
            # For n-gram based proposer, the drafter run on host
            # cpu, therefore, we need to first copy the sampled
            # token ids from device to host, then run the ngram proposer.
            valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
                self.runner, spec_decode_metadata, sampled_output,
                logits_indices_selector, discard_sampled_tokens_req_indices,
                self.runner.input_batch.num_reqs)
            self._draft_token_ids = self.runner.drafter.propose(
                valid_sampled_token_ids[:self.runner.input_batch.num_reqs],
                self.runner.input_batch.num_tokens_no_spec,
                self.runner.input_batch.token_ids_cpu)
        elif self.runner.speculative_config.use_eagle():
            assert input_ids is not None
            self._draft_token_ids = self.propose_eagle3_draft_token_ids(
                spec_decode_metadata,
                last_sampled_token_id,
                num_rejected_tokens,
                discard_sampled_tokens_req_indices,
                aux_hidden_states,
                attn_metadata,
                scheduler_output,
                input_ids,
                async_scheduling,
                hidden_states,
            )
            self._req_indices_dp = spec_decode_metadata.req_indices_dp
        else:
            raise NotImplementedError(
                f"Speculative decoding method "
                f"'{self.runner.speculative_config.method}' is not supported.")

    def propose_eagle3_draft_token_ids(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        last_sampled_token_id: jnp.ndarray,
        num_rejected_tokens: jnp.ndarray,
        discard_sampled_tokens_req_indices: list[int],
        aux_hidden_states: Optional[tuple[jnp.ndarray, ...]],
        attn_metadata: AttentionMetadata | dict[str, AttentionMetadata],
        scheduler_output: VllmSchedulerOutput,
        input_ids: jnp.ndarray,
        async_scheduling: bool,
        hidden_states: jnp.ndarray,
    ) -> list[list[int]] | jnp.ndarray:
        assert isinstance(self.runner.drafter, Eagle3Proposer)
        if isinstance(attn_metadata, dict):
            # When multiple KV cache groups are used (e.g., in hybrid models),
            # attn_metadata becomes a dict mapping layer names to AttentionMetadata.
            # Since all groups share the same seq_lens and input_positions, any would work for those.
            # However, we specifically look for an attention layer key to get the correct
            # block_tables structure, just in case the draft model (which is all attention) needs it.
            attn_key = None
            for key in attn_metadata.keys():
                if ".self_attn." in key:
                    attn_key = key
                    break
            if attn_key is not None:
                attn_metadata = attn_metadata[attn_key]
            else:
                attn_metadata = next(iter(attn_metadata.values()))

        req_ids = self.runner.input_batch.req_ids
        max_num_seqs = attn_metadata.seq_lens.shape[0]
        next_prompt_token_id = np.zeros(max_num_seqs, dtype=np.int32)
        is_in_prefill = np.zeros(max_num_seqs, dtype=np.int32)

        discard_sampled_tokens_req_indices_set = set(
            discard_sampled_tokens_req_indices)
        dp_size = self.runner.dp_size
        max_num_reqs_per_dp_rank = self.runner.max_num_reqs // dp_size
        num_reqs_dp = np.zeros((dp_size, ), dtype=np.int32)
        for rank in range(dp_size):
            req_indices = spec_decode_metadata.req_indices_dp[rank]
            num_reqs_dp[rank] = len(req_indices)
            for j, req_idx in enumerate(req_indices):
                if req_idx in discard_sampled_tokens_req_indices_set:
                    # Partial prefill
                    # Get the next token id from the request state.
                    req_id = req_ids[req_idx]
                    req_state = self.runner.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                               scheduler_output.num_scheduled_tokens[req_id])
                    next_token_id = req_state.get_token_id(seq_len)
                    next_prompt_token_id[
                        j + rank * max_num_reqs_per_dp_rank] = next_token_id
                    is_in_prefill[j + rank * max_num_reqs_per_dp_rank] = 1

        next_prompt_token_id, is_in_prefill, num_reqs_dp = device_array(
            self.runner.mesh,
            (next_prompt_token_id, is_in_prefill, num_reqs_dp),
            sharding=(PartitionSpec(ShardingAxisName.ATTN_DATA)))

        if self.runner.speculative_config.method == "mtp":
            aux_hidden_states_for_drafter = (hidden_states, )
        else:
            aux_hidden_states_for_drafter = tuple(aux_hidden_states) + (
                hidden_states, )

        target_hidden_states, input_ids, last_token_indices, attn_metadata = self.runner.drafter.prepare_inputs(
            attn_metadata,
            input_ids,
            aux_hidden_states_for_drafter,
            last_sampled_token_id,
            next_prompt_token_id,
            is_in_prefill,
            num_rejected_tokens,
            num_reqs_dp,
        )

        self.runner.kv_caches, draft_token_ids = self.runner.drafter.propose(
            kv_caches=self.runner.kv_caches,
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            last_token_indices=last_token_indices,
            target_hidden_states=target_hidden_states,
        )

        if async_scheduling:
            if jnp.ndim(draft_token_ids) == 1:
                draft_token_ids = jnp.expand_dims(draft_token_ids, 1)
            return draft_token_ids
        else:
            draft_token_ids = np.array(draft_token_ids)
            if draft_token_ids.ndim == 1:
                draft_token_ids = np.expand_dims(draft_token_ids, axis=-1)
            return draft_token_ids.tolist()

    def get_spec_decode_metadata(
        self,
        num_draft_tokens_dp: np.ndarray,
        dp_size: int,
        req_indices_dp: dict,
        req_ids_dp: dict,
        query_start_loc: np.ndarray,
        padded_num_reqs_per_dp_rank: int,
        padded_logits_length_dp_rank: int,
        max_num_reqs_per_dp_rank: int,
    ) -> SpecDecodeMetadata:
        # (dp_size = 1) Example Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        out_logits_indices = []
        out_target_logits_indices = []
        out_bonus_logits_indices = []
        out_draft_lengths = []

        def _pad(arr: np.ndarray, target_len: int) -> np.ndarray:
            pad_len = target_len - arr.shape[0]
            if pad_len == 0:
                return arr
            return np.concatenate([arr, np.zeros(pad_len, dtype=np.int32)])

        for dp_rank in range(dp_size):
            per_rank_req_idxs = req_indices_dp[dp_rank]
            n = len(per_rank_req_idxs)

            if n > 0:
                cur_rank_req_idxs = req_indices_dp[dp_rank]
                num_draft_tokens = num_draft_tokens_dp[cur_rank_req_idxs]
                num_sampled_tokens = num_draft_tokens + 1

                req_offset = dp_rank * max_num_reqs_per_dp_rank
                local_query_start_loc = query_start_loc[
                    req_offset + dp_rank:req_offset +
                    max_num_reqs_per_dp_rank + dp_rank + 1]
                cu_num_scheduled_tokens = local_query_start_loc[1:n + 1]

                # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
                # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
                cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)
                arange = np.concatenate(
                    [self.runner.arange_cpu[:n] for n in num_sampled_tokens])
                # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
                logits_indices = np.repeat(
                    cu_num_scheduled_tokens - num_sampled_tokens,
                    num_sampled_tokens)
                # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
                logits_indices += arange
                # Compute the bonus logits indices.
                bonus_logits_indices = cu_num_sampled_tokens - 1

                # Compute the draft logits indices.
                # arange: [0, 1, 2, 0, 1, 0]
                arange = np.concatenate(
                    [self.runner.arange_cpu[:n] for n in num_draft_tokens])
                # [0, 0, 0, 5, 5, 9]
                target_logits_indices = np.repeat(
                    cu_num_sampled_tokens - num_sampled_tokens,
                    num_draft_tokens)
                # [0, 1, 2, 5, 6, 9]
                target_logits_indices += arange
            else:
                logits_indices = np.array([], dtype=np.int32)
                target_logits_indices = np.array([], dtype=np.int32)
                bonus_logits_indices = np.array([], dtype=np.int32)
                num_draft_tokens = np.array([], dtype=np.int32)

            out_logits_indices.append(
                _pad(logits_indices, padded_logits_length_dp_rank))
            out_target_logits_indices.append(
                _pad(target_logits_indices, padded_logits_length_dp_rank))
            out_bonus_logits_indices.append(
                _pad(bonus_logits_indices, padded_num_reqs_per_dp_rank))
            out_draft_lengths.append(
                _pad(num_draft_tokens, padded_num_reqs_per_dp_rank))

        padded_logits_indices = np.concatenate(out_logits_indices)
        padded_target_logits_indices = np.concatenate(
            out_target_logits_indices)
        padded_bonus_logits_indices = np.concatenate(out_bonus_logits_indices)
        padded_num_draft_tokens = np.concatenate(out_draft_lengths)
        padded_num_draft_tokens_cpu = padded_num_draft_tokens
        # CPU -> TPU copy.
        (padded_num_draft_tokens, padded_logits_indices,
         padded_target_logits_indices,
         padded_bonus_logits_indices) = device_array(
             self.runner.mesh,
             (padded_num_draft_tokens, padded_logits_indices,
              padded_target_logits_indices, padded_bonus_logits_indices),
             sharding=NamedSharding(self.runner.mesh,
                                    PartitionSpec(ShardingAxisName.ATTN_DATA)))

        metadata = SpecDecodeMetadata(
            draft_lengths=padded_num_draft_tokens,
            target_logits_indices=padded_target_logits_indices,
            bonus_logits_indices=padded_bonus_logits_indices,
            final_logits_indices=padded_logits_indices,
        )
        metadata.draft_lengths_cpu = padded_num_draft_tokens_cpu
        metadata.req_indices_dp = req_indices_dp
        metadata.req_ids_dp = req_ids_dp
        return metadata
