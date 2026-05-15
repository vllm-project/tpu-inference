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
"""DeepSeek-V4 MLA attention backend and metadata builder for TPU:

   PallasDeepseekV4MLABackend — provides get_page_size() for the compressed
   MLA KV cache. Utilized by tpu_platform.py for DeepseekV4ForCausalLM models.

   TpuDeepseekSparseSWAMetadataBuilder — builds per-forward-pass attention
   metadata to read and write to SWA & compressed MLA caches. Used by
   TpuDeepseekV4MLAAttention.forward() to pass to the Pallas kernel.
    -> GPU reference for (2): vllm/vllm/v1/attention/backends/mla/sparse_swa.py
    DeepseekSparseSWAMetadataBuilder
"""
import functools
from dataclasses import dataclass, field
from typing import Any

import jax
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",    # (num_tokens,) int32 — absolute token positions
        "swa_slot_mapping",   # (num_tokens,) int32 — flat physical slot per token into swa_cache
        "state_slot_mapping", # (num_tokens,) int32 — flat physical slot per token into state_cache
        "mla_block_table",    # (max_num_seqs, max_mla_blocks) int32 — block table for mla_cache flushes
        "swa_block_table",    # (max_num_seqs, max_swa_blocks) int32 — block table for swa_cache reads
        "decode_swa_indices", # (num_decode_tokens, window_size) int32 — sparse SWA gather indices
        "seq_lens",           # (max_num_seqs,) int32 — per-sequence lengths
        "query_start_loc",    # (max_num_seqs + 1,) int32 — prefix sums over query tokens
        "request_distribution", # (3,) int32 — [decode-only, prefill-only, mixed]
    ],
    meta_fields=[
        "num_decodes",         # number of decode requests in this batch
        "num_prefills",        # number of prefill requests in this batch
        "num_decode_tokens",   # total decode tokens (== num_decodes for non-speculative)
        "num_prefill_tokens",  # total prefill tokens
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class DeepseekV4AttentionMetadata:
    """Per-forward-pass attention metadata for DeepSeek-V4 on TPU.

    Includes separate slot mappings for three paged caches (swa, state, mla) and
    sparse decode SWA indices.

    GPU analogue: DeepseekSparseSWAMetadata + CompressorMetadata
    (vllm/v1/attention/backends/mla/sparse_swa.py and deepseek_compressor.py).

    """
    input_positions: jax.Array          # (num_tokens,)
    swa_slot_mapping: jax.Array         # (num_tokens,)
    state_slot_mapping: jax.Array       # (num_tokens,)
    mla_block_table: jax.Array          # (max_num_seqs, max_mla_blocks)
    swa_block_table: jax.Array          # (max_num_seqs, max_swa_blocks)
    decode_swa_indices: jax.Array       # (num_decode_tokens, window_size)
    seq_lens: jax.Array                 # (max_num_seqs,)
    query_start_loc: jax.Array          # (max_num_seqs + 1,)
    request_distribution: jax.Array     # (3,)

    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)


class PallasDeepseekV4MLABackend(AttentionBackend):
    """Attention backend stub for DeepSeek-V4 MLA on TPU.

    Used only to provide get_page_size() for the compressed MLA KV cache
    (the SWA cache defines its own block_size within the class init).
    get_impl_cls() is not used by V4: TpuDeepseekV4MLAAttention.forward()
    calls the Pallas kernel directly with no backend impl indirection.

    TODO (4a): Verify the correct MLA block size once the V4 Pallas kernel
    layout is finalized (depends on TODO 1b). V3 uses 1024; V4's compressed
    dual-cache layout (SWA block_size=64, MLA block_size=?) may differ.
    """

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_MLA"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError(
            "PallasDeepseekV4MLABackend has no impl class. "
            "V4 attention is driven directly by TpuDeepseekV4MLAAttention.")

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # TODO verify that this is the correct block size for the V4
        # compressed MLA cache. 
        # For reference, V3 uses 1024.
        return 1024


class TpuDeepseekSparseSWAMetadataBuilder:
    """Builds per-forward-pass attention metadata for DeepSeek-V4 on TPU.

    TODO Key fields to populate per forward pass
    (reflecting GPU DeepseekSparseSWAMetadata + CompressorMetadata):
      - swa_slot_mapping[num_tokens]: flat physical slot index per token into swa_cache.
          Used by start_update_kv_cache to write raw KV. Mirrors GPU DeepseekSparseSWAMetadata.slot_mapping.
      - state_slot_mapping[num_tokens]: flat physical slot index per token into state_cache.
          Used by start_update_kv_cache to write kv+score accumulators. Mirrors GPU CompressorMetadata.slot_mapping.
          When state_slot_mapping[t] % block_size == compress_ratio - 1, the kernel should 
          write to mla_cache and flush the state_cache.
      - mla_block_table[num_reqs, max_blocks]: block table for mla_cache. Used by
          the kernel to compute the physical address when flushing a compressed slot.
          Mirrors GPU CompressorMetadata.block_table.
      - num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens: batch split
      - decode_swa_indices [num_decode_tokens, window_size]: sparse SWA indices for decode
      - prefill_seq_lens, prefill_gather_lens: for chunked prefill
    Key methods to add:
      - __init__(vllm_config, compress_ratios, window_size): pre-allocate
        decode_swa_indices, token_to_req_indices scratch buffers
      - build(scheduler_output, kv_cache_manager) -> DeepseekV4AttentionMetadata:
        populate all fields above, compute decode/prefill split
    """

    def __init__(self):
        raise NotImplementedError(
            "TpuDeepseekSparseSWAMetadataBuilder is not yet implemented")

    def build(self) -> DeepseekV4AttentionMetadata:
        raise NotImplementedError(
            "TpuDeepseekSparseSWAMetadataBuilder is not yet implemented.")
