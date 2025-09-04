from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
import numpy as np
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from tpu_commons.runner import utils as runner_utils
from tpu_commons.utils import device_array

if TYPE_CHECKING:
    from tpu_commons.runner.tpu_jax_runner import TPUModelRunner


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on JAX/TPU, containing all necessary indices."""
    draft_token_ids: jnp.ndarray
    draft_lengths: jnp.ndarray
    draft_lengths_cpu: np.ndarray
    target_logits_indices: jnp.ndarray
    bonus_logits_indices: jnp.ndarray
    final_logits_indices: jnp.ndarray


class SpeculativeDecodingManager:

    def __init__(self, runner: "TPUModelRunner"):
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
        sampled_token_ids: list[list[int]],
    ) -> None:
        self._draft_token_ids = self.propose_ngram_draft_token_ids(
            sampled_token_ids)

    def propose_ngram_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        assert isinstance(self.runner.drafter, NgramProposer)
        draft_token_ids: list[list[int]] = []
        num_reqs = self.runner.input_batch.num_reqs
        for i, sampled_ids in enumerate(sampled_token_ids[:num_reqs]):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require sampling parameters that are not
            # supported with speculative decoding.
            req_id = self.runner.input_batch.req_ids[i]
            if req_id in self.runner.input_batch.spec_decode_unsupported_reqs:
                draft_token_ids.append([])
                continue

            num_tokens = self.runner.input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.runner.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            drafter_output = self.runner.drafter.propose(
                self.runner.input_batch.token_ids_cpu[i, :num_tokens])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())

        return draft_token_ids

    def get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
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
        draft_token_ids = self.runner.input_ids_cpu[logits_indices]
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
            draft_lengths_cpu=padded_num_draft_tokens_cpu,
            target_logits_indices=padded_target_logits_indices,
            bonus_logits_indices=padded_bonus_logits_indices,
            final_logits_indices=padded_logits_indices,
        )
        return metadata
