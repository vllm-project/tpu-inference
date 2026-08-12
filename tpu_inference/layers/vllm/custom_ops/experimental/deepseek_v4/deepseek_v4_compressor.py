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
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.compressor import (CompressorStateCache,
                                                DeepseekCompressor)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import \
    config as compressor_config
from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store.compressor_v1 import \
    compressor_forward
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
        # CSA's and the indexer's state caches overlay their own compressed-KV
        # array; HCA's overlays a *CSA NoPE* array.
        kv_cache_block_size = get_current_vllm_config().cache_config.block_size
        assert kv_cache_block_size // compress_ratio > 0
        mode = compressor_config.select_mode(self.head_dim,
                                             compress_ratio == 4)
        cfgs = compressor_config.Configs.make(
            mode,
            size_n=0,
            physical_page_size=compressor_config.physical_page_size(
                mode, kv_cache_block_size, compress_ratio),
            state_physical_page_size=compressor_config.state_host_page_size(
                mode, kv_cache_block_size, compress_ratio),
            head_dim=self.head_dim,
            compress_ratio=compress_ratio,
        )
        self.block_size = cfgs.state_block_size
        assert self.block_size > 0

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
        self._wkv_wgate_transposed = False

    def transpose_wkv_wgate(self) -> None:
        """Stores ``fused_wkv_wgate.weight`` transposed, once, at load time.

        vLLM lays linear weights out as [out_features, in_features], but the
        compress-and-store kernel consumes them as
        [hidden_size, 2 * coff * head_dim].
        """
        if self._wkv_wgate_transposed:
            return
        weight = self.fused_wkv_wgate.weight.data.t().contiguous()
        self.fused_wkv_wgate.weight = torch.nn.Parameter(weight,
                                                         requires_grad=False)
        self._wkv_wgate_transposed = True

    # pylint: disable=unused-argument
    def forward(
        self,
        hidden_states: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:

        # head_dim == 512: sparse CSA/HCA main path.
        # head_dim == 128: lightning-indexer path.
        # ``compressor_forward`` picks the store mode from (head_dim, overlap).
        assert self.head_dim in (512, 128)

        attn_metadata = get_forward_context().attn_metadata
        # State-cache metadata carries the per-token scatter indices; the k_cache
        # layer owns the physical buffer the state and compressed-KV share.
        state_metadata = attn_metadata[self.state_cache.prefix]
        k_cache_metadata = attn_metadata[self.k_cache_prefix]

        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh
        cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache_prefix]

        # Only CSA (head_dim=512, compress_ratio=4) has a separate jax.Array
        # to hold the RoPE channels separately.
        rope_cache = None
        rope_cache_index = None
        if self.head_dim == 512 and self.overlap:
            # CSA
            rope_cache_name = f"{self.k_cache_prefix}_rope"
            rope_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
                rope_cache_name]
            rope_cache = wrapper_ctx.kv_caches[rope_cache_index]

        cache = wrapper_ctx.kv_caches[cache_index]
        state_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.state_cache.prefix]

        separate_state = state_cache_index != cache_index
        if self.overlap:
            # CSA
            assert separate_state is False
        else:
            # HCA
            assert separate_state is True
        state_cache = (wrapper_ctx.kv_caches[state_cache_index]
                       if separate_state else None)

        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        in_specs = (
            data_spec,  # hidden_states     [num_tokens, hidden_size]
            P(),  # wkv_wgate         [hidden_size, 2 * coff * head_dim]
            P(),  # ape               [compress_ratio, coff * head_dim]
            P(),  # norm_weight       [head_dim]
            P(),  # cos_sin_cache     [max_pos, rope_head_dim]
            data_spec,  # positions            [num_tokens]
            data_spec,  # state_block_tables   [num_reqs * max_blocks]
            data_spec,  # query_start_loc      [num_reqs + 1]
            data_spec,  # k_block_tables       [num_reqs * max_kv_blocks]
            data_spec,  # request_distribution [3]
            cache_spec,  # cache
        )
        has_rope_cache = rope_cache is not None
        out_spec_list = [cache_spec]
        if has_rope_cache:
            in_specs += (cache_spec, )  # rope_cache
            out_spec_list.append(cache_spec)
        if separate_state:
            in_specs += (cache_spec, )  # state_cache
            out_spec_list.append(cache_spec)
        out_specs = (tuple(out_spec_list)
                     if len(out_spec_list) > 1 else out_spec_list[0])

        def _compress(hidden_states, wkv_wgate, ape, norm_weight,
                      cos_sin_cache, positions, state_block_tables,
                      query_start_loc, k_block_tables, request_distribution,
                      cache, *optional_caches):
            optional_caches = list(optional_caches)
            rope_c = optional_caches.pop(0) if has_rope_cache else None
            state_c = optional_caches.pop(0) if separate_state else None

            new_cache, new_rope_cache, new_state_cache = compressor_forward(
                hidden_states=hidden_states,
                wkv_wgate=wkv_wgate,
                ape=ape,
                norm_weight=norm_weight,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                block_table=state_block_tables,
                query_start_loc=query_start_loc,
                kv_block_table=k_block_tables,
                cache=cache,
                rope_cache=rope_c,
                state_cache=state_c,
                distribution=request_distribution,
                state_block_size=self.state_cache.block_size,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                overlap=self.overlap,
                rms_eps=self.rms_norm_eps,
                quant_block=self._quant_block,
            )
            outs = [new_cache]
            if has_rope_cache:
                outs.append(new_rope_cache)
            if separate_state:
                outs.append(new_state_cache)
            return tuple(outs) if len(outs) > 1 else outs[0]

        assert self._wkv_wgate_transposed
        operands = (
            jax_view(hidden_states),
            jax_view(self.fused_wkv_wgate.weight),
            jax_view(self.ape),
            jax_view(self.norm.weight),
            jax_view(rotary_emb.cos_sin_cache),
            jax_view(positions),
            state_metadata.block_tables,
            state_metadata.query_start_loc,
            k_cache_metadata.block_tables,
            state_metadata.request_distribution,
            cache,
        )
        if has_rope_cache:
            operands += (rope_cache, )
        if separate_state:
            operands += (state_cache, )

        outs = jax.shard_map(
            _compress,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(*operands)

        outs = list(outs) if isinstance(outs, tuple) else [outs]
        wrapper_ctx.kv_caches[cache_index] = outs.pop(0)
        if has_rope_cache:
            wrapper_ctx.kv_caches[rope_cache_index] = outs.pop(0)
        if separate_state:
            wrapper_ctx.kv_caches[state_cache_index] = outs.pop(0)
