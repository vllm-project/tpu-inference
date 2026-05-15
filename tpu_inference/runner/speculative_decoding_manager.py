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
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from tpu_inference.runner import utils as runner_utils
from tpu_inference.runner.utils import SpecDecodeMetadata
from tpu_inference.spec_decode.jax.eagle3 import Eagle3Proposer
from tpu_inference.spec_decode.jax.utils import extract_last_sampled_tokens
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

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.runner.input_batch.req_ids
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def propose_draft_token_ids(
        self,
        sampled_output: jnp.ndarray,
        logits_indices_selector: np.ndarray,
        discard_sampled_tokens_req_indices: list,
        aux_hidden_states: Optional[tuple[jnp.ndarray, ...]],
        attn_metadata: AttentionMetadata,
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        scheduler_output: Optional[VllmSchedulerOutput] = None,
        input_ids: Optional[jnp.ndarray] = None,
    ) -> None:
        if self.runner.speculative_config.method == "ngram":
            assert isinstance(self.runner.drafter, NgramProposer)
            # For n-gram based proposer, the drafter run on host
            # cpu, therefore, we need to first copy the sampled
            # token ids from device to host, then run the ngram proposer.
            valid_sampled_token_ids = runner_utils.host_extract_sampled_tokens(
                self.runner, spec_decode_metadata, sampled_output,
                logits_indices_selector, discard_sampled_tokens_req_indices)
            self._draft_token_ids = self.runner.drafter.propose(
                valid_sampled_token_ids[:self.runner.input_batch.num_reqs],
                self.runner.input_batch.num_tokens_no_spec,
                self.runner.input_batch.token_ids_cpu)
        elif self.runner.speculative_config.method == "eagle3":
            last_sampled_token_id, num_rejected_tokens = extract_last_sampled_tokens(
                spec_decode_metadata, sampled_output,
                self.runner.speculative_config.num_speculative_tokens,
                self.runner.input_batch.vocab_size, self.runner.max_num_reqs)
            self._draft_token_ids = self.propose_eagle3_draft_token_ids(
                spec_decode_metadata,
                last_sampled_token_id,
                num_rejected_tokens,
                discard_sampled_tokens_req_indices,
                aux_hidden_states,
                attn_metadata,
                scheduler_output,
                input_ids,
            )
        else:
            raise NotImplementedError(
                f"Speculative decoding method "
                f"'{self.runner.speculative_config.method}' is not supported.")

    def propose_eagle3_draft_token_ids(
        self,
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        last_sampled_token_id: jnp.ndarray,
        num_rejected_tokens: jnp.ndarray,
        discard_sampled_tokens_req_indices: list[int],
        aux_hidden_states: Optional[tuple[jnp.ndarray, ...]],
        attn_metadata: AttentionMetadata,
        scheduler_output: VllmSchedulerOutput,
        input_ids: jnp.ndarray,
    ) -> list[list[int]]:
        assert isinstance(self.runner.drafter, Eagle3Proposer)
        req_ids = self.runner.input_batch.req_ids
        max_num_seqs = attn_metadata.seq_lens.shape[0]
        next_prompt_token_id = np.zeros(max_num_seqs, dtype=np.int32)
        is_in_prefill = np.zeros(max_num_seqs, dtype=np.int32)
        for i in discard_sampled_tokens_req_indices:
            # Partial prefill
            # Get the next token id from the request state.
            req_id = req_ids[i]
            req_state = self.runner.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            next_token_id = req_state.get_token_id(seq_len)
            next_prompt_token_id[i] = next_token_id
            is_in_prefill[i] = 1

        next_prompt_token_id, is_in_prefill = device_array(
            self.runner.mesh, (next_prompt_token_id, is_in_prefill))

        target_hidden_states, input_ids, last_token_indices, attn_metadata = self.runner.drafter.prepare_inputs(
            attn_metadata,
            input_ids,
            aux_hidden_states,
            last_sampled_token_id,
            next_prompt_token_id,
            is_in_prefill,
            num_rejected_tokens,
        )

        self.runner.kv_caches, draft_token_ids = self.runner.drafter.propose(
            kv_caches=self.runner.kv_caches,
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            last_token_indices=last_token_indices,
            target_hidden_states=target_hidden_states,
        )
        draft_token_ids = np.array(draft_token_ids)
        if draft_token_ids.ndim == 1:
            draft_token_ids = np.expand_dims(draft_token_ids, axis=-1)
        return draft_token_ids.tolist()

    def get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
        input_ids: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
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
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)
        arange = np.concatenate(
            [self.runner.arange_cpu[:n] for n in num_sampled_tokens])
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
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
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]
        padded_logits_length = runner_utils.get_padded_token_len(
            self.runner.num_logits_paddings, logits_indices.shape[0])
        padded_logits_indices = np.concatenate([
            logits_indices,
            np.zeros(padded_logits_length - logits_indices.shape[0],
                     dtype=np.int32)
        ])

        assert bonus_logits_indices.shape[0] <= padded_num_reqs, (
            f"bonus_logits_indices.shape[0]={bonus_logits_indices.shape[0]} "
            f"padded_num_reqs={padded_num_reqs}")

        padded_bonus_logits_indices = np.concatenate([
            bonus_logits_indices,
            np.zeros(padded_num_reqs - bonus_logits_indices.shape[0],
                     dtype=np.int32)
        ])
        padded_num_draft_tokens = np.concatenate([
            num_draft_tokens,
            np.zeros(padded_num_reqs - num_draft_tokens.shape[0],
                     dtype=np.int32)
        ])
        padded_draft_token_ids = np.concatenate([
            draft_token_ids,
            np.zeros(padded_logits_length - draft_token_ids.shape[0],
                     dtype=np.int32)
        ])
        padded_target_logits_indices = np.concatenate([
            target_logits_indices,
            np.zeros(padded_logits_length - target_logits_indices.shape[0],
                     dtype=np.int32)
        ])

        padded_num_draft_tokens_cpu = padded_num_draft_tokens
        # CPU -> TPU copy.
        (padded_num_draft_tokens, padded_draft_token_ids,
         padded_logits_indices, padded_target_logits_indices,
         padded_bonus_logits_indices) = device_array(
             self.runner.mesh,
             (padded_num_draft_tokens, padded_draft_token_ids,
              padded_logits_indices, padded_target_logits_indices,
              padded_bonus_logits_indices))

        metadata = SpecDecodeMetadata(
            draft_token_ids=padded_draft_token_ids,
            draft_lengths=padded_num_draft_tokens,
            target_logits_indices=padded_target_logits_indices,
            bonus_logits_indices=padded_bonus_logits_indices,
            final_logits_indices=padded_logits_indices,
        )
        metadata.draft_lengths_cpu = padded_num_draft_tokens_cpu
        return metadata
