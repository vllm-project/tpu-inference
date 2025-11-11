# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

from dataclasses import dataclass
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingType
from vllm.utils.collection_utils import swap_dict_values
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.spec_decode.utils import is_spec_decode_unsupported

from tpu_inference.runner.block_table import MultiGroupBlockTable

_SAMPLING_EPS = 1e-5

# TODO(xiang): fix cpu tensor init


@dataclass
class CachedRequestState(NewRequestData):

    output_token_ids: Optional[list[int]] = None
    generator: Optional[Any] = None
    mrope_positions: Optional[jax.Array] = None
    mrope_position_delta: Optional[int] = None

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            return self.prompt_token_ids[idx]
        else:
            return self.output_token_ids[idx - self.num_prompt_tokens]


class InputBatch:

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
        is_spec_decode: bool = False,
    ):
        self.is_spec_decode = is_spec_decode
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu = np.zeros(
            (max_num_reqs, max_model_len),
            dtype=np.int32,
        )
        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(
            (max_num_reqs, ),
            dtype=np.int32,
        )

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            block_sizes=block_sizes,
        )

        # Sampling-related.
        self.temperature_cpu = np.empty((max_num_reqs, ), dtype=np.float32)
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p_cpu = np.empty((max_num_reqs, ), dtype=np.float32)

        self.top_k_cpu = np.empty((max_num_reqs, ), dtype=np.int32)

        # IDs of requests which do not support spec decoding
        self.spec_decode_unsupported_reqs: set[str] = set()

        # req_index -> (min_tokens, stop_token_ids)
        self.min_tokens: dict[int, tuple[int, set[int]]] = {}

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs, ),
                                             dtype=np.int32)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, Any] = {}

        self.num_logprobs: dict[str, int] = {}

        self.logit_bias: list[Optional[dict[int,
                                            float]]] = [None] * max_num_reqs
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: Optional[jax.Array] = None
        self.allowed_token_ids_mask_cpu: Optional[jax.Array] = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.req_output_token_ids: list[Optional[list[int]]] = []

        self.request_distribution: list[int] = [0, 0, 0]

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs, f"{req_index} < {self.max_num_reqs} failed!"

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        # NOTE(woosuk): This may include spec decode tokens.
        self.num_tokens[req_index] = request.num_tokens
        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        sampling_params = request.sampling_params

        if (self.is_spec_decode
                and is_spec_decode_unsupported(sampling_params)):
            self.spec_decode_unsupported_reqs.add(req_id)

        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Avoid later division by zero.
            self.temperature_cpu[req_index] = -1.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k >= self.vocab_size:
            top_k = 1
        self.top_k_cpu[req_index] = top_k
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens,
                                          sampling_params.all_stop_token_ids)

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask_cpu is None:
                # Lazy allocation for this tensor, which can be large.
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask = jnp.zeros(self.max_num_reqs,
                                                        self.vocab_size,
                                                        dtype=jnp.bool)
                self.allowed_token_ids_mask_cpu = np.zeros(self.max_num_reqs,
                                                           self.vocab_size,
                                                           dtype=np.bool)
            self.allowed_token_ids_mask_cpu[req_index] = True
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu[req_index][
                sampling_params.allowed_token_ids] = False

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[
                req_index] = sampling_params.bad_words_token_ids

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

    def remove_request(self, req_id: str) -> Optional[int]:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.spec_decode_unsupported_reqs.discard(req_id)
        self.min_tokens.pop(req_index, None)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)

        # LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            self.lora_id_to_request_ids[lora_id].discard(req_id)
            if len(self.lora_id_to_request_ids[lora_id]) == 0:
                self.lora_id_to_request_ids.pop(lora_id)
                self.lora_id_to_lora_request.pop(lora_id)
            self.request_lora_mapping[req_index] = 0

        self.logit_bias[req_index] = None
        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu is not None:
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] =\
            self._req_ids[i2], self._req_ids[i1] # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] =\
            self.req_output_token_ids[i2], self.req_output_token_ids[i1]
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] =\
            self.req_id_to_index[old_id_i2], self.req_id_to_index[old_id_i1]
        self.num_tokens[i1], self.num_tokens[i2] =\
            self.num_tokens[i2], self.num_tokens[i1]
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] =\
            self.num_tokens_no_spec[i2], self.num_tokens_no_spec[i1]
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] =\
            self.num_prompt_tokens[i2], self.num_prompt_tokens[i1]
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] =\
            self.num_computed_tokens_cpu[i2], self.num_computed_tokens_cpu[i1]
        self.temperature_cpu[i1], self.temperature_cpu[i2] =\
            self.temperature_cpu[i2], self.temperature_cpu[i1]
        self.top_p_cpu[i1], self.top_p_cpu[i2] =\
            self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] =\
            self.top_k_cpu[i2], self.top_k_cpu[i1]

        # NOTE: the following is unsafe
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # instead, we need to temporiarily copy the data for one of the indices
        # TODO(lucas): optimize this by only copying valid indices
        tmp = self.token_ids_cpu[i1, ...].copy()
        self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
        self.token_ids_cpu[i2, ...] = tmp

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.min_tokens, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] =\
            self.request_lora_mapping[i2], self.request_lora_mapping[i1]
        self.logit_bias[i1], self.logit_bias[i2] =\
            self.logit_bias[i2], self.logit_bias[i1]

        if self.allowed_token_ids_mask_cpu is not None:
            self.allowed_token_ids_mask_cpu[i1], \
                self.allowed_token_ids_mask_cpu[i2] =\
                self.allowed_token_ids_mask_cpu[i2], \
                    self.allowed_token_ids_mask_cpu[i1]
        self.block_table.swap_row(i1, i2)

    def condense(self, empty_req_indices: list[int]) -> None:
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Swap the states.
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens]
            self.num_tokens[empty_index] = num_tokens
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[
                last_req_index]
            self.num_computed_tokens_cpu[
                empty_index] = self.num_computed_tokens_cpu[last_req_index]
            self.block_table.move_row(last_req_index, empty_index)
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            min_token = self.min_tokens.pop(last_req_index, None)
            if min_token is not None:
                self.min_tokens[empty_index] = min_token

            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index]

            self.logit_bias[empty_index] = self.logit_bias[last_req_index]

            if self.allowed_token_ids_mask_cpu is not None:
                self.allowed_token_ids_mask_cpu[
                    empty_index] = self.allowed_token_ids_mask_cpu[
                        last_req_index]

            bad_words_token_ids = self.bad_words_token_ids.pop(
                last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids
            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs:]
        del self.req_output_token_ids[self.num_reqs:]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def max_num_logprobs(self) -> Optional[int]:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size self.num_reqs where,
               prompt_lora_mapping[i] is the LoRA id to use for the ith prompt.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """

        req_lora_mapping = self.request_lora_mapping[:self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping)
        token_lora_mapping = tuple(
            req_lora_mapping.repeat(num_scheduled_tokens))
        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values())

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests
