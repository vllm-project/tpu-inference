# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBlockPlan:
    full_blocks: tuple[tuple[int, ...], ...]
    partial_canvas: tuple[int, ...] | None
    partial_mask: tuple[bool, ...] | None
    remainder_size: int


@dataclass
class PendingBlockOutput:
    tokens: list[int]
    next_anchor: int


def required_cache_end(
    prompt_length: int,
    max_new_tokens: int,
    block_size: int,
) -> int:
    if prompt_length < 0 or max_new_tokens < 0 or block_size < 1:
        raise ValueError(
            "Lengths must be non-negative and block_size positive")
    num_cached_tokens = prompt_length + max(0, max_new_tokens - 1)
    return ((num_cached_tokens + block_size - 1) // block_size * block_size)


def plan_seeded_prompt(
    prompt_token_ids: list[int],
    block_size: int,
    mask_token_id: int,
) -> PromptBlockPlan:
    if not prompt_token_ids:
        raise ValueError("Block diffusion requires a non-empty prompt")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    num_full_blocks, remainder_size = divmod(len(prompt_token_ids), block_size)
    full_blocks = tuple(
        tuple(prompt_token_ids[index * block_size:(index + 1) * block_size])
        for index in range(num_full_blocks))
    if remainder_size == 0:
        return PromptBlockPlan(
            full_blocks=full_blocks,
            partial_canvas=None,
            partial_mask=None,
            remainder_size=0,
        )

    remainder = prompt_token_ids[num_full_blocks * block_size:]
    partial_canvas = tuple(remainder) + (mask_token_id, ) * (block_size -
                                                             remainder_size)
    partial_mask = (False, ) * remainder_size + (True, ) * (block_size -
                                                            remainder_size)
    return PromptBlockPlan(
        full_blocks=full_blocks,
        partial_canvas=partial_canvas,
        partial_mask=partial_mask,
        remainder_size=remainder_size,
    )


def start_partial_block_output(
    committed_canvas: list[int],
    remainder_size: int,
    next_anchor: int,
) -> tuple[list[int], PendingBlockOutput]:
    generated = committed_canvas[remainder_size:]
    if not generated:
        raise ValueError(
            "A partial prompt block must generate at least one token")
    pending = PendingBlockOutput(tokens=generated[1:], next_anchor=next_anchor)
    return [generated[0]], pending


def flush_partial_block_output(state: PendingBlockOutput) -> list[int]:
    return [*state.tokens, state.next_anchor]


def complete_seeded_decode_block(
    committed_canvas: list[int],
    next_anchor: int,
) -> list[int]:
    if not committed_canvas:
        raise ValueError("committed_canvas must not be empty")
    return [*committed_canvas[1:], next_anchor]
