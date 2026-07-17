# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

from dataclasses import dataclass, field
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils.collection_utils import swap_dict_values
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from tpu_inference.runner.block_table import MultiGroupBlockTable
from tpu_inference.utils import device_array

_SAMPLING_EPS = 1e-5

# TODO(xiang): fix cpu tensor init


@dataclass
class CachedRequestState(NewRequestData):

    output_token_ids: Optional[list[int]] = None
    generator: Optional[Any] = None
    mrope_positions: Optional[jax.Array] = None
    mrope_position_delta: Optional[int] = None
    pooling_states: PoolingStates = field(default_factory=PoolingStates)
    # Accumulates prompt logprob chunks across chunked-prefill steps.
    # Tuple of (token_ids, logprobs, ranks) numpy arrays, each of shape
    # [num_prompt_tokens-1, ...]. Set to None when prefill completes.
    in_progress_prompt_logprobs_cpu: Optional[tuple] = None
    mamba_state_slot: Optional[int] = None

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
        num_speculative_tokens: int = 0,
        dp_size: int = 1,
    ):
        self.is_spec_decode = is_spec_decode
        self.max_num_reqs = max_num_reqs
        self.dp_size = dp_size
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
            num_speculative_tokens=num_speculative_tokens,
        )

        # Sampling-related.
        self.temperature_cpu = np.empty((max_num_reqs, ), dtype=np.float32)
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p_cpu = np.empty((max_num_reqs, ), dtype=np.float32)

        self.top_k_cpu = np.empty((max_num_reqs, ), dtype=np.int32)

        self.min_p_cpu = np.empty((max_num_reqs, ), dtype=np.float32)

        self.repetition_penalties_cpu = np.empty((max_num_reqs, ),
                                                 dtype=np.float32)

        # Incremental "already seen (prompt+output) tokens" mask for
        # repetition_penalty, indexed by req slot: (max_num_reqs, vocab_size)
        # bool on device. Lazily allocated the first time any active request
        # sets repetition_penalty != 1.0 (stays None otherwise -> zero cost).
        # `seen_scattered_upto[slot]` is the high-water mark of how many of
        # `token_ids_cpu[slot]`'s *accepted* tokens (num_tokens_no_spec) have
        # already been OR-ed into the mask, so each step only scatters the new
        # tokens. Both follow the request through swap_states/condense/add.
        self.seen_token_ids_mask: Optional[jax.Array] = None
        self.seen_scattered_upto = np.zeros(max_num_reqs, dtype=np.int32)

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
        self.num_prompt_logprobs: dict[str, int] = {}

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

        # Per-request physical slot id in the mamba kv-cache. Each request
        # gets a unique slot at `add_request` and keeps it for its lifetime
        # (slot ids follow the request through `condense` and `swap_states`,
        # see those methods below). Within each DP rank k, slot 0 (or
        # k * local_slots for dp_size > 1) is the null block; usable slots
        # start at base + 1. Why we track slot ids separately
        # from the persistent-batch position `req_idx`: `condense`
        # (https://github.com/vllm-project/vllm/blob/de3da0b/vllm/v1/worker/gpu_input_batch.py#L662)
        # moves requests into lower `req_idx` slots when earlier requests
        # finish, but the mamba recurrent state stays at its allocated
        # physical slot. Indexing the cache by the moving `req_idx` would
        # read stale state; indexing by `mamba_state_indices_cpu[req_idx]`
        # reads the slot that actually holds this request's state.
        # Initial pool size is based on max_num_reqs rounded up to dp_size;
        # init_mamba_pools() resizes after KV cache allocation is known.
        self.mamba_state_indices_cpu = np.zeros(max_num_reqs, dtype=np.int32)
        # Mamba slot pool, partitioned by DP rank so each rank's requests
        # get slots within that rank's shard of the device-side state array.
        # Matches the sharding in kv_cache_manager: mamba_num_blocks is
        # rounded up to dp_size, then split evenly across ranks.
        mamba_num_blocks = ((max_num_reqs + dp_size) // dp_size) * dp_size
        self._mamba_local_slots = mamba_num_blocks // dp_size
        self._free_mamba_slots_per_rank: list[list[int]] = []
        for k in range(dp_size):
            base = k * self._mamba_local_slots
            self._free_mamba_slots_per_rank.append(
                list(range(base + self._mamba_local_slots - 1, base, -1)))

        # for pooling models
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}
        self.has_mamba_layers = False

    def init_mamba_pools(self, mamba_num_blocks: int) -> None:
        """Reinitialize mamba slot pools with the actual device block count.

        Called after KV cache init, when the true mamba_num_blocks is known
        (compact-mamba may have been skipped, giving fewer blocks than the
        default estimate based on max_num_reqs).
        """
        self._mamba_local_slots = mamba_num_blocks // self.dp_size
        self._free_mamba_slots_per_rank = []
        for k in range(self.dp_size):
            base = k * self._mamba_local_slots
            self._free_mamba_slots_per_rank.append(
                list(range(base + self._mamba_local_slots - 1, base, -1)))

    def release_mamba_slot(self, slot: Optional[int]) -> None:
        if slot is None:
            return
        slot = int(slot)
        if slot == 0 or slot % self._mamba_local_slots == 0:
            return
        rank = slot // self._mamba_local_slots
        pool = self._free_mamba_slots_per_rank[rank]
        if slot not in pool:
            pool.append(slot)

    def assert_mamba_state_invariants(
        self,
        requests: Optional[dict[str, CachedRequestState]] = None,
        assigned_dp_rank: Optional[dict[str, int]] = None,
    ) -> None:
        active_slots: list[int] = []
        active_req_ids = self._req_ids[:self.num_reqs]
        free_slots = {
            int(slot)
            for pool in self._free_mamba_slots_per_rank
            for slot in pool
        }

        if sum(len(pool)
               for pool in self._free_mamba_slots_per_rank) != len(free_slots):
            raise AssertionError("Duplicate mamba slots in free pools")

        for req_index, req_id in enumerate(active_req_ids):
            if req_id is None:
                raise AssertionError(
                    f"Active mamba batch has a hole at index {req_index}")
            slot = int(self.mamba_state_indices_cpu[req_index])
            active_slots.append(slot)
            if slot <= 0 or slot % self._mamba_local_slots == 0:
                raise AssertionError(
                    f"Request {req_id} has invalid mamba slot {slot}")
            if slot in free_slots:
                raise AssertionError(
                    f"Active request {req_id} uses free mamba slot {slot}")
            if assigned_dp_rank is not None:
                expected_rank = assigned_dp_rank.get(req_id, 0)
                slot_rank = slot // self._mamba_local_slots
                if slot_rank != expected_rank:
                    raise AssertionError(
                        f"Request {req_id} on DP rank {expected_rank} has "
                        f"mamba slot {slot} from rank {slot_rank}")
            if requests is not None:
                req_state = requests.get(req_id)
                if req_state is not None and req_state.mamba_state_slot != slot:
                    raise AssertionError(
                        f"Request {req_id} active slot {slot} does not match "
                        f"cached slot {req_state.mamba_state_slot}")

        if len(set(active_slots)) != len(active_slots):
            from collections import Counter
            duplicate_active = {
                slot
                for slot, count in Counter(active_slots).items() if count > 1
            }
            raise AssertionError(
                f"Duplicate active mamba slots: {sorted(duplicate_active)}")

        tail = self.mamba_state_indices_cpu[self.num_reqs:]
        if tail.any():
            nonzero_tail = sorted(set(tail[tail != 0].tolist()))
            raise AssertionError(
                f"Non-zero mamba slots in padded tail: {nonzero_tail}")

        if requests is not None:
            preserved_slot_list = [
                int(req.mamba_state_slot) for req in requests.values()
                if req.mamba_state_slot is not None
            ]
            preserved_slots = set(preserved_slot_list)
            if len(preserved_slot_list) != len(preserved_slots):
                raise AssertionError("Duplicate preserved mamba slots")
            overlap = preserved_slots & free_slots
            if overlap:
                raise AssertionError(
                    f"Preserved mamba slots also in free pool: "
                    f"{sorted(overlap)}")

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def get_pooling_params(self) -> list[PoolingParams]:
        # although being list[str], it actually can be list[str | None]
        return [self.pooling_params[r] for r in self.req_ids if r]

    def get_pooling_states(self) -> list[PoolingStates]:
        # although being list[str], it actually can be list[str | None]
        return [self.pooling_states[r] for r in self.req_ids if r]

    def get_pooling_metadata(self) -> PoolingMetadata:
        pooling_params = self.get_pooling_params()
        pooling_states = self.get_pooling_states()

        # Extract prompt token IDs from token_ids_cpu
        # Shape of token_ids_cpu is (max_num_reqs, max_model_len)
        max_prompt_len = int(self.num_prompt_tokens[:self.num_reqs].max()
                             ) if self.num_reqs > 0 else 0
        prompt_token_ids_tensor = torch.zeros((self.num_reqs, max_prompt_len),
                                              dtype=torch.int32)
        for i in range(self.num_reqs):
            num_prompt = self.num_prompt_tokens[i]
            prompt_token_ids_tensor[i, :num_prompt] = torch.from_numpy(
                self.token_ids_cpu[i, :num_prompt]).to(torch.int32)

        return PoolingMetadata(
            prompt_lens=torch.from_numpy(
                self.num_prompt_tokens[:self.num_reqs]),
            prompt_token_ids=prompt_token_ids_tensor,
            prompt_token_ids_cpu=prompt_token_ids_tensor,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
        dp_rank: int = 0,
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

        # Reset the repetition-penalty seen-mask bookkeeping for this (reused)
        # slot so the next update_seen_token_ids_mask() rescatters this
        # request's tokens from scratch instead of inheriting a prior tenant's.
        self.seen_scattered_upto[req_index] = 0
        if self.seen_token_ids_mask is not None:
            self.seen_token_ids_mask = self.seen_token_ids_mask.at[
                req_index].set(False)

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)
        # Allocate a fresh mamba state slot for this request. The slot stays
        # with the request through the persistent batch's lifetime, even when
        # condense moves the request to a different `req_index`.
        if request.mamba_state_slot is None:
            request.mamba_state_slot = self._free_mamba_slots_per_rank[
                dp_rank].pop()
        else:
            slot_rank = int(
                request.mamba_state_slot) // self._mamba_local_slots
            assert slot_rank == dp_rank, (
                f"Preserved mamba slot {request.mamba_state_slot} belongs to "
                f"DP rank {slot_rank}, not {dp_rank}")
            pool = self._free_mamba_slots_per_rank[dp_rank]
            if request.mamba_state_slot in pool:
                pool.remove(request.mamba_state_slot)
        self.mamba_state_indices_cpu[req_index] = request.mamba_state_slot

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        def collect_sampling(sampling_params: SamplingParams) -> None:

            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Avoid later division by zero.
                self.temperature_cpu[req_index] = -1.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            self.top_p_cpu[req_index] = sampling_params.top_p
            self.min_p_cpu[req_index] = sampling_params.min_p
            self.repetition_penalties_cpu[req_index] = (
                sampling_params.repetition_penalty)
            top_k = sampling_params.top_k
            # Default to -1 (considering all tokens)
            if top_k >= self.vocab_size:
                top_k = -1
            self.top_k_cpu[req_index] = top_k
            if sampling_params.min_tokens:
                self.min_tokens[req_index] = (
                    sampling_params.min_tokens,
                    sampling_params.all_stop_token_ids)

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = sampling_params.logprobs
            if sampling_params.prompt_logprobs is not None:
                num_k = (self.vocab_size if sampling_params.prompt_logprobs
                         == -1 else sampling_params.prompt_logprobs)
                self.num_prompt_logprobs[req_id] = num_k
                n = len(request.prompt_token_ids) - 1
                if n > 0:
                    request.in_progress_prompt_logprobs_cpu = (
                        np.empty((n, num_k + 1), dtype=np.int32),
                        np.empty((n, num_k + 1), dtype=np.float32),
                        np.empty(n, dtype=np.int32),
                    )
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
                    self.allowed_token_ids_mask_cpu = np.zeros(
                        self.max_num_reqs, self.vocab_size, dtype=np.bool)
                self.allowed_token_ids_mask_cpu[req_index] = True
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu[req_index][
                    sampling_params.allowed_token_ids] = False

            if sampling_params.bad_words_token_ids:
                self.bad_words_token_ids[
                    req_index] = sampling_params.bad_words_token_ids

        if sampling_params := request.sampling_params:
            collect_sampling(sampling_params)

        self.pooling_params[req_id] = request.pooling_params
        self.pooling_states[req_id] = request.pooling_states

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

    def remove_request(self,
                       req_id: str,
                       *,
                       free_mamba_slot: bool = True) -> Optional[int]:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None
        # Return the mamba state slot back to the free pool. The slot's
        # contents in the kv cache are stale and will be zeroed by the
        # has_initial_state guard when the next request takes this slot id.
        slot = int(self.mamba_state_indices_cpu[req_index])
        if free_mamba_slot:
            self.release_mamba_slot(slot)
        # Clear this position to slot 0 (the null block) so the trailing
        # tail of `mamba_state_indices_cpu` (which the GDN op reads over
        # its full length every step) cannot alias an active slot.
        # Concrete trace with max_num_reqs=4:
        #
        #   start (4 active):           [1, 2, 3, 4]  num_reqs=4
        #   remove pos 0,1 (stale):     [1, 2, 3, 4]  num_reqs=2
        #   condense w/o source clear:  [3, 4, 3, 4]  ← tail aliases active
        #   condense w/  source clear:  [3, 4, 0, 0]  ← tail is null
        #
        # In the aliased case `recurrent_state.at[slots].set(...)` writes
        # twice to slot 3 in the same scatter — undefined on XLA, silent
        # state corruption. See
        # `test_mamba_state_indices_no_duplicate_in_padded_tail`.
        self.mamba_state_indices_cpu[req_index] = 0

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.spec_decode_unsupported_reqs.discard(req_id)
        self.min_tokens.pop(req_index, None)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)

        # It's ok to pop nothing for non-pooling model.
        self.pooling_params.pop(req_id, None)
        self.pooling_states.pop(req_id, None)

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
        max_active_token_count = max(int(self.num_tokens[i1]),
                                     int(self.num_tokens[i2]))
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
        self.min_p_cpu[i1], self.min_p_cpu[i2] =\
            self.min_p_cpu[i2], self.min_p_cpu[i1]
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] =\
            self.repetition_penalties_cpu[i2], self.repetition_penalties_cpu[i1]
        self.seen_scattered_upto[i1], self.seen_scattered_upto[i2] =\
            self.seen_scattered_upto[i2], self.seen_scattered_upto[i1]
        if self.seen_token_ids_mask is not None:
            self.seen_token_ids_mask = self.seen_token_ids_mask.at[[i1, i2]].set(
                self.seen_token_ids_mask[[i2, i1]])

        self.token_ids_cpu[[i1, i2], :max_active_token_count] = \
            self.token_ids_cpu[[i2, i1], :max_active_token_count]

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
        # The mamba state slot id is per-request and must follow the swap.
        self.mamba_state_indices_cpu[i1], self.mamba_state_indices_cpu[i2] = (
            self.mamba_state_indices_cpu[i2],
            self.mamba_state_indices_cpu[i1],
        )

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
            # The mamba state slot id is per-request: when the persistent
            # batch moves the request from `last_req_index` to `empty_index`,
            # the slot id must follow it so subsequent steps still index
            # the right physical slot in the mamba kv cache.
            self.mamba_state_indices_cpu[
                empty_index] = self.mamba_state_indices_cpu[last_req_index]
            # Clear the source: the slot id now lives at `empty_index`, so
            # leaving it here too would put a duplicate in the padded tail
            # (see the trace in `remove_request`).
            self.mamba_state_indices_cpu[last_req_index] = 0
            self.temperature_cpu[empty_index] = self.temperature_cpu[
                last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.min_p_cpu[empty_index] = self.min_p_cpu[last_req_index]
            self.repetition_penalties_cpu[empty_index] = (
                self.repetition_penalties_cpu[last_req_index])
            self.seen_scattered_upto[empty_index] = self.seen_scattered_upto[
                last_req_index]
            if self.seen_token_ids_mask is not None:
                self.seen_token_ids_mask = self.seen_token_ids_mask.at[
                    empty_index].set(self.seen_token_ids_mask[last_req_index])
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

    def update_seen_token_ids_mask(
        self,
        mesh,
        padded_num_reqs: int,
        sharding: Optional[Any] = None,
    ) -> Optional[jax.Array]:
        """Incrementally maintain the repetition-penalty seen-token mask.

        Returns the ``(padded_num_reqs, vocab_size)`` bool mask marking, per
        active request slot, every token id that has appeared in its prompt or
        accepted output -- or ``None`` when no active request has
        ``repetition_penalty != 1.0`` (the common case, so the sampler compiles
        with no penalty work at all).

        Driven by ``token_ids_cpu`` + ``num_tokens_no_spec`` (the accepted-token
        count, so rejected speculative tokens are never marked) and a per-slot
        high-water mark, so each call only scatters tokens added since the last
        call: O(num_reqs) per decode step, O(prompt_len) once at prefill. Slot
        moves (swap_states/condense/add_request) keep the mask row + high-water
        aligned, so this never rescans the full sequence.
        """
        num_reqs = self.num_reqs
        if num_reqs == 0 or not bool(
                np.any(self.repetition_penalties_cpu[:num_reqs] != 1.0)):
            return None

        if self.seen_token_ids_mask is None:
            self.seen_token_ids_mask = device_array(
                mesh,
                np.zeros((self.max_num_reqs, self.vocab_size), dtype=bool),
                sharding=sharding,
            )

        rows_list: list[np.ndarray] = []
        toks_list: list[np.ndarray] = []
        for slot in range(num_reqs):
            lo = int(self.seen_scattered_upto[slot])
            hi = int(self.num_tokens_no_spec[slot])
            if hi > lo:
                rows_list.append(np.full(hi - lo, slot, dtype=np.int32))
                toks_list.append(
                    self.token_ids_cpu[slot, lo:hi].astype(np.int32))
                self.seen_scattered_upto[slot] = hi
        if rows_list:
            rows = jnp.asarray(np.concatenate(rows_list))
            toks = jnp.asarray(np.concatenate(toks_list))
            self.seen_token_ids_mask = self.seen_token_ids_mask.at[
                rows, toks].set(True)

        return self.seen_token_ids_mask[:padded_num_reqs]

    @property
    def max_num_logprobs(self) -> Optional[int]:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def max_num_prompt_logprobs(self) -> Optional[int]:
        return (max(self.num_prompt_logprobs.values())
                if self.num_prompt_logprobs else None)

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
