# pytype: skip-file
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-compatible DeepSeek-V4 KV/score compressor."""

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import (CompressorStateCache,
                                                DeepseekCompressor)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from tpu_inference.kernels.experimental.deepseek_v4.compressor import (
    compressor_forward, compressor_forward_indexer)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


class VllmCompressorStateCache(CompressorStateCache):
    """TPU-compatible compressor state cache.

  The base ``get_kv_cache_spec`` returns a ``SlidingWindowMLASpec`` describing
  the CUDA/FlashMLA paged layout, which does not apply on TPU. The TPU KV-cache
  spec is handled elsewhere, so here ``get_kv_cache_spec`` is a no-op.
  """

    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__(
            state_dim,
            dtype,
            compress_ratio,
            prefix,
        )
        coff = 1 + (compress_ratio == 4)
        self.head_dim = state_dim // 2 // coff
        # We let HCA's state cache overlay with HCA's compressed cache;
        # CSA's state cache overlay with CSA's compressed cache for now.
        # TODO: we may better let HCA's state cache overlay on CSA's compressed cache,
        # whose page size is bigger, for better performance.
        compressed_kv_cache_bz = get_current_vllm_config(
        ).cache_config.block_size // compress_ratio
        assert compressed_kv_cache_bz > 0
        # 4 due to 4 bytes per f32
        # 2 due to kv-dim + score-dim
        self.block_size = compressed_kv_cache_bz // 4 // 2 // coff

    # pylint: disable=unused-argument
    def get_kv_cache_spec(self, vllm_config):
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
        )


class VllmDeepseekCompressor(DeepseekCompressor):
    """TPU-compatible DeepSeek-V4 compressor.

  The base ``DeepseekCompressor`` forward dispatches to CUDA/triton kernels
  (``save_partial_states`` / ``compress_norm_rope_store_*``) that are not
  available on TPU. The TPU-native compress/store path is handled elsewhere,
  so here ``forward`` is a passthrough no-op.
  """

    def __init__(self, *args, **kwargs):
        # The base ctor builds ``self.state_cache`` from the vLLM
        # ``CompressorStateCache``. Temporarily rebind the symbol
        # the base ctor references so it constructs the TPU subclass directly.
        orig_state_cache = dsv4_compressor.CompressorStateCache
        dsv4_compressor.CompressorStateCache = VllmCompressorStateCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_compressor.CompressorStateCache = orig_state_cache

    # pylint: disable=unused-argument
    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:

        # head_dim == 512: sparse CSA/HCA main path -> compressor_forward.
        # head_dim == 128: lightning-indexer path  -> compressor_forward_indexer.
        # Both share the same call shape; only the inner store kernel differs.
        assert self.head_dim in (512, 128)
        forward_fn = (compressor_forward
                      if self.head_dim == 512 else compressor_forward_indexer)
        # TODO: For data in state cache, if bf16 instead of f32
        # don't hurt quality, we should use b16.
        kv_score = kv_score.to(torch.float32)

        attn_metadata = get_forward_context().attn_metadata
        # State-cache metadata carries the per-token scatter indices; the k_cache
        # layer owns the physical buffer the state and compressed-KV share.
        state_metadata = attn_metadata[self.state_cache.prefix]
        k_cache_metadata = attn_metadata[self.k_cache_prefix]

        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh
        cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache_prefix]
        # [num_pages, page_size//4, 4, width] uint8 — shared state + KV buffer.
        cache = wrapper_ctx.kv_caches[cache_index]
        state_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.state_cache.prefix]

        # code under `tpu_inference.kernels.experimental.deepseek_v4.compressor`
        # assume state cache and the compressed kv cache of the *same layer* overlay
        # on the same tensor. That is, layer-i's HCA state cache share the same
        # Tensor with layer-i's HCA compressed kv cache.
        # This assumption is too strict.
        # TODO: we may better let HCA's state cache overlay on CSA's main kv cache,
        # whose page size is bigger, for better performance.
        assert state_cache_index == cache_index

        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        in_specs = (
            data_spec,  # kv_score  [num_tokens, 2 * self.coff * self.head_dim]
            P(),  # ape                 [compress_ratio, coff*head_dim]
            P(),  # norm_weight         [head_dim]
            P(),  # cos_sin_cache       [max_pos, rope_head_dim]
            data_spec,  # positions             [num_tokens]
            data_spec,  # request_distribution
            data_spec,  # query_start_loc       [num_reqs + 1]
            data_spec,  # seq_lens              [num_reqs]
            data_spec,  # state_input_positions [num_tokens]
            data_spec,  # state_block_tables    [num_reqs * max_blocks]
            data_spec,  # k_input_positions     [num_tokens]
            data_spec,  # k_block_tables        [num_reqs * max_blocks]
            cache_spec,  # cache
        )
        out_specs = cache_spec

        def _compress(kv_score, ape, norm_weight, cos_sin_cache, positions,
                      request_distribution, query_start_loc, seq_lens,
                      state_input_positions, state_block_tables,
                      k_input_positions, k_block_tables, cache):
            num_valid_reqs = request_distribution[2]
            num_tokens = kv_score.shape[0]
            num_valid_tokens = query_start_loc[num_valid_reqs]

            q_per_req = query_start_loc[1:] - query_start_loc[:-1]
            max_num_seqs = seq_lens.shape[0]
            token_to_req_indices = jnp.repeat(jnp.arange(max_num_seqs),
                                              q_per_req,
                                              total_repeat_length=num_tokens)

            state_block_size = self.state_cache.block_size
            assert state_block_tables.shape[0] % max_num_seqs == 0
            max_num_blocks_per_req = state_block_tables.shape[0] // max_num_seqs
            state_cache_block_num = state_input_positions // state_block_size + token_to_req_indices * max_num_blocks_per_req
            state_cache_block_offset = state_input_positions % state_block_size
            slot_mapping = (
                state_block_tables[state_cache_block_num] * state_block_size +
                state_cache_block_offset)
            slot_mapping = jnp.where(
                jnp.arange(num_tokens) < num_valid_tokens, slot_mapping, -1)

            k_block_size = cache.shape[1] * cache.shape[2]
            assert k_block_tables.shape[0] % max_num_seqs == 0
            max_num_blocks_per_req = k_block_tables.shape[0] // max_num_seqs
            k_cache_block_num = (
                k_input_positions // self.compress_ratio
            ) // k_block_size + token_to_req_indices * max_num_blocks_per_req
            k_cache_block_offset = (k_input_positions //
                                    self.compress_ratio) % k_block_size
            kv_slot_mapping = (
                k_block_tables[k_cache_block_num] * k_block_size +
                k_cache_block_offset)
            kv_slot_mapping = jnp.where(
                jnp.arange(num_tokens) < num_valid_tokens, kv_slot_mapping, -1)

            block_table = state_block_tables.reshape(max_num_seqs, -1)

            return forward_fn(
                kv_score=kv_score,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_to_req_indices=token_to_req_indices,
                kv_slot_mapping=kv_slot_mapping,
                cache=cache,
                state_block_size=self.state_cache.block_size,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                rms_eps=self.rms_norm_eps,
                quant_block=self._quant_block,
            )

        new_cache = jax.shard_map(
            _compress,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            jax_view(kv_score),
            jax_view(self.ape),
            jax_view(self.norm.weight),
            jax_view(rotary_emb.cos_sin_cache),
            jax_view(positions),
            state_metadata.request_distribution,
            state_metadata.query_start_loc,
            state_metadata.seq_lens,
            state_metadata.input_positions,
            state_metadata.block_tables,
            k_cache_metadata.input_positions,
            k_cache_metadata.block_tables,
            cache,
        )
        wrapper_ctx.kv_caches[state_cache_index] = new_cache
