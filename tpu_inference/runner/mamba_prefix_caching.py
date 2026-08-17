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
"""Slot bookkeeping for mamba prefix caching (`mamba_cache_mode="align"`).

With prefix caching on, a hybrid model's recurrent state no longer lives in
the per-request slot pool `InputBatch` hands out — it lives in
prefix-cacheable blocks addressed by the mamba group's block table, so a
request sharing a prefix resumes from the state an earlier request wrote.

Kept free of JAX and runner imports so the addressing can be tested on its
own; the runner only supplies the block table and token counts.
"""
import numpy as np

__all__ = ["build_align_state_indices"]


def build_align_state_indices(
    block_table: np.ndarray,
    num_computed_tokens: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    block_size: int,
    max_num_reqs: int,
    req_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive the mamba state slots a step reads from and writes to.

    `align` mode keeps one recurrent-state checkpoint per mamba block. vLLM's
    invariant (see `Scheduler._mamba_block_aligned_split`) is that block row
    `p` holds the state after exactly `(p + 1) * block_size` tokens, so the
    state after `pos` tokens lives in row `(pos - 1) // block_size`.

    A step therefore reads the checkpoint covering `num_computed_tokens` and
    writes the one covering its last scheduled token. Those rows differ
    exactly when the step crosses a block boundary. GPU handles that by
    copying the state between the two blocks before the forward pass
    (`vllm/v1/worker/mamba_utils.py::preprocess_mamba`); the TPU kernel reads
    the first and writes the second in a single launch instead, which is
    equivalent and skips a round-trip of the state through HBM.

    On a prefix-cache hit the read row lands on a block another request
    populated, which is the state reuse that makes the shared prefix free.

    Args:
        block_table: `[max_num_reqs, max_num_blocks_per_req]` block ids of
            the mamba kv-cache group, indexed by persistent-batch position.
        num_computed_tokens: `[>= max_num_reqs]` tokens already computed per
            persistent-batch position.
        num_scheduled_tokens: `[>= num_reqs]` tokens scheduled this step, in
            `req_indices` order.
        block_size: mamba block size in tokens.
        max_num_reqs: length of the returned arrays.
        req_indices: `[num_reqs]` persistent-batch positions of the requests
            scheduled this step, in the order the kernel sees them.

    Returns:
        `(write_indices, read_indices)`, both `[max_num_reqs]` int32 and
        padded with block id 0 (vLLM's null block) so unused persistent-batch
        positions cannot alias a live request's state.
    """
    write_indices = np.zeros(max_num_reqs, dtype=np.int32)
    read_indices = np.zeros(max_num_reqs, dtype=np.int32)

    rows = np.asarray(req_indices, dtype=np.intp)
    num_reqs = rows.size
    if num_reqs == 0:
        return write_indices, read_indices

    num_computed = np.asarray(num_computed_tokens)[rows]
    seq_lens = num_computed + np.asarray(num_scheduled_tokens)[:num_reqs]

    # The read row goes unused when nothing has been computed yet (the
    # kernel's `has_initial_state` guard is false), so clamping the negative
    # row to 0 only has to keep it in range.
    write_rows = np.maximum(seq_lens - 1, 0) // block_size
    read_rows = np.maximum(num_computed - 1, 0) // block_size

    write_indices[:num_reqs] = block_table[rows, write_rows]
    read_indices[:num_reqs] = block_table[rows, read_rows]
    return write_indices, read_indices
