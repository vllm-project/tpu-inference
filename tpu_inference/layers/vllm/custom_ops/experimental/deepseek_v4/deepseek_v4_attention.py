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
"""TPU interception for DeepSeek-V4 MLA attention (torchax path).

``DeepseekV4Attention`` is a plain ``nn.Module`` that vLLM instantiates directly
(the AMD decoder does ``self.attn = DeepseekV4ROCMAiterMLAAttention(...)``, a
``DeepseekV4Attention`` subclass, in ``deepseek_v4/amd/model.py``). Unlike the
MHC ops or the attention-impl bases, it is NOT a vLLM ``CustomOp`` and has no
``register_oot`` hook, so there is no registry-based way to swap it. Its
constructor is also CUDA-bound (allocates ``torch.cuda.Event``), so it cannot
run on TPU as-is.

Instead we substitute the class symbol before the model is built. Because
``amd/model.py`` does ``from ...amd.rocm import DeepseekV4ROCMAiterMLAAttention``,
the name is bound into the ``amd.model`` module namespace at import time; patching
it on ``amd.rocm`` alone would not take effect. ``patch_deepseek_v4_mla_cls``
rebinds it on ``amd.model`` directly. It is invoked from
``_maybe_patch_for_deepseek_v4`` in ``vllm_model_wrapper`` while ``is_rocm`` is
forced True and the package has been reloaded onto the AMD implementation.
"""
import functools
from unittest.mock import patch

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import (KVCacheSpec, MLAAttentionSpec,
                                        SlidingWindowMLASpec)

from tpu_inference.kernels.experimental.deepseek_v4 import rope as rope_kernel
from tpu_inference.kernels.experimental.deepseek_v4.core_attention.mla import \
    mla_ragged_paged_attention
from tpu_inference.kernels.experimental.deepseek_v4.core_attention.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from tpu_inference.kernels.experimental.deepseek_v4.core_attention.sparse_mla import \
    sparse_ragged_paged_attention
from tpu_inference.kernels.experimental.deepseek_v4.o_projection import \
    fused_reverse_rope_wo_a_projection
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.custom_ops.experimental.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from tpu_inference.layers.vllm.custom_ops.experimental.deepseek_v4.deepseek_v4_indexer import \
    VllmDeepseekV4Indexer
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``.
    """
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


def sharded_rope(rope_fn, mesh, x, positions, cos_sin_cache, **kwargs):
    data_spec = P(ShardingAxisName.ATTN_DATA)

    def _rope(x, positions, cos_sin_cache):
        return rope_fn(x, positions, cos_sin_cache, **kwargs)

    return jax.shard_map(
        _rope,
        mesh=mesh,
        in_specs=(data_spec, data_spec, P()),
        out_specs=data_spec,
        check_vma=False,
    )(x, positions, cos_sin_cache)


class VllmDeepseekV4SWACache(DeepseekV4SWACache):

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
        backend_cls=None,
    ):
        # vLLM #47808 added ``backend_cls`` to DeepseekV4SWACache and the base
        # attention ctor always passes it; None keeps the default SWA backend.
        super().__init__(head_dim,
                         window_size,
                         dtype,
                         prefix,
                         cache_config,
                         backend_cls=backend_cls)
        compressed_kv_cache_bz = cache_config.block_size
        # We would like to overlay the SWA cache with CSA's main NOPE cache
        # on the same KV-Tensor, whose shape is [num_pages, page_size, 4, 128]
        # u8.
        # Thus set swa cache's block size accordingly.
        csa_compression_ratio = 4
        # In SWA, we store kv cache in bf16 to avoid expensive quantization
        # and dequantization. The extra storage overhead is small since kvs
        # outside the sliding window can be reclaimed as needed.
        # 2 because bf16 occupies 2 bytes per element.
        self.block_size = min(
            compressed_kv_cache_bz // csa_compression_ratio // 2, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # ``mla_swa`` keeps the SWA entries as raw bf16, packed as uint8.
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=512 * 2,
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )


class VllmDeepseekV4MLAAttention(DeepseekV4Attention):

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        # The base ctor builds the indexer, compressor and SWA cache from the
        # vLLM upstream classes. Temporarily rebind the module-level symbols the
        # base ctor references so it constructs the TPU subclasses directly.
        orig_indexer = dsv4_attention.DeepseekV4Indexer
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_swa_cache = dsv4_attention.DeepseekV4SWACache
        dsv4_attention.DeepseekV4Indexer = VllmDeepseekV4Indexer
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache

        # The base ctor also allocates CUDA-backed stream-sync events (the
        # ``ln_events``), used only for GPU stream overlap. Mock them to no-ops.
        # vLLM #47668 reverted these from ``torch.Event`` back to
        # ``torch.cuda.Event``, so both symbols must be neutralized -- the
        # ``torch.Event`` mock alone no longer matches the reverted code, and a
        # real ``torch.cuda.Event`` is a dummy stub on TPU (no CUDA).
        orig_event = torch.Event
        orig_cuda_event = torch.cuda.Event
        torch.Event = lambda *args, **kwargs: None
        torch.cuda.Event = lambda *args, **kwargs: None
        try:
            # DeepSeek-V4's implementation use sth like:
            # torch.zeros(.. device=device). Pass `cpu``
            # instead of tpu to avoid error. Those buffer won't
            # be used in the forward anyway.
            with patch.object(current_platform, "device_type", "cpu"):
                super().__init__(
                    vllm_config,
                    prefix=prefix,
                    topk_indices_buffer=topk_indices_buffer,
                    aux_stream_list=aux_stream_list,
                )
        finally:
            dsv4_attention.DeepseekV4Indexer = orig_indexer
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4SWACache = orig_swa_cache
            torch.Event = orig_event
            torch.cuda.Event = orig_cuda_event

    # Abstract platform hooks required to instantiate the DeepseekV4Attention
    # ABC; unused on the TPU pass-through path.
    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (self.compress_ratio
                <= 1):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None

        # CSA (`sparse_mla`) reads the NoPE record from this array and the RoPE
        # channels from the companion `{prefix}_rope` array; the budget here
        # covers both. HCA (`mla`) stores raw bf16 in this array. Both are
        # packed as uint8.
        is_csa = self.compress_ratio == 4
        if is_csa:
            # In DSV4 FP8 format
            # 448 fp8, 64 bf16, 7 fp8 scales, 7 e8m0 scale for 448 fp8 (block size 64)
            # packed as uint8
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=align_to(448 + 64 * 2 + 7, 128),
                dtype=torch.uint8,
                compress_ratio=self.compress_ratio,
                alignment=None,
            )
        else:
            # For HCA, we store raw bf16 values in the KV cache to avoid
            # expernsive DSV4 FP8 quantization and dequantization. The
            # size of HCA cache only take very small memory overall, so it's ok.
            return MLAAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=1,
                head_size=512 * 2,
                dtype=torch.uint8,
                compress_ratio=self.compress_ratio,
                alignment=None,
            )

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """Inverse-RoPE + wo_a (per-group bmm) + wo_b output projection.
        """
        mesh = get_vllm_model_wrapper_context().mesh
        # fused_reverse_rope_wo_a_projection assume data parallelism
        # for the attention layer.
        z = jax.shard_map(
            functools.partial(
                fused_reverse_rope_wo_a_projection,
                head_dim=self.head_dim,
                inverse=True,
                quantize_activations=True,
            ),
            mesh=mesh,
            in_specs=(
                P(ShardingAxisName.ATTN_DATA),  # o
                P(ShardingAxisName.ATTN_DATA),  # positions
                P(),  # cos_sin_cache (replicated)
                P(),  # wo_a
                P(),  # wo_a_scale
            ),
            out_specs=P(ShardingAxisName.ATTN_DATA),
            check_vma=False,
        )(
            jax_view(o),
            jax_view(positions),
            jax_view(self.rotary_emb.cos_sin_cache),
            jax_view(self.wo_a.weight),
            jax_view(self.wo_a.weight_scale),
        )

        # --- wo_b: RowParallelLinear back to hidden_size (returns (out, bias)).
        out = self.wo_b(torch_view(z))
        if isinstance(out, tuple):
            out = out[0]
        return out

    def attn_gemm(self, hidden_states):
        # MergedColumnParallelLinear returns (output, bias); bias is None.
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        # The compressors' ``fused_wkv_wgate`` projections are fused into the
        # TPU compress-and-store kernel, which takes ``hidden_states``
        # directly.
        if self.indexer is not None:
            indexer = self.indexer
            # ReplicatedLinear returns (output, bias); bias is None.
            indexer_weights, _ = indexer.weights_proj(hidden_states)
        else:
            indexer_weights = None

        return qr_kv, indexer_weights

    def qnorm_rope(
            self,
            q: torch.Tensor,  # [num_tokens, n_local_heads, head_dim]
            positions: torch.Tensor,  # [num_tokens], int64
    ) -> torch.Tensor:
        """Per-head RMSNorm (no weight) + GPT-J interleaved RoPE on q.
        """
        mesh = get_vllm_model_wrapper_context().mesh
        return torch_view(
            sharded_rope(
                rope_kernel.qnorm_rope,
                mesh,
                jax_view(q),
                jax_view(positions),
                jax_view(self.rotary_emb.cos_sin_cache),
                eps=self.eps,
            ))

    def kv_rope(
            self,
            kv: torch.Tensor,  # [num_tokens, head_dim]
            positions: torch.Tensor,  # [num_tokens], int64
    ) -> torch.Tensor:
        mesh = get_vllm_model_wrapper_context().mesh
        return torch_view(
            sharded_rope(
                rope_kernel.rope,
                mesh,
                jax_view(kv),
                jax_view(positions),
                jax_view(self.rotary_emb.cos_sin_cache),
            ))

    def attention_impl(
            self,
            hidden_states: torch.Tensor,
            qr: torch.Tensor,
            kv: torch.Tensor,
            indexer_weights: torch.Tensor,
            positions: torch.Tensor,
            out: torch.Tensor,  # Not used
    ) -> torch.Tensor:
        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (forward_mqa
        # downstream reads q on default). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.

        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = self.qnorm_rope(q, positions)
        kv = self.kv_rope(kv, positions)

        # The TPU compressor fuses its own ``fused_wkv_wgate`` projection, so it
        # consumes ``hidden_states``.
        topk_indices = None
        if self.indexer is not None:
            assert self.compressor is not None
            topk_indices = self.indexer(hidden_states, qr, indexer_weights,
                                        positions, self.indexer_rotary_emb)
            self.compressor(hidden_states, positions, self.rotary_emb)
        elif self.compressor is not None:
            self.compressor(hidden_states, positions, self.rotary_emb)

        return self.forward_mqa(q,
                                kv,
                                positions,
                                None,
                                topk_indices=topk_indices)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,  # Not used
        *,
        topk_indices: torch.Tensor,
    ) -> None:
        # The SWA cache update +
        # core attention to SWA cache and compressed KV cache.
        swa_only = self.compress_ratio <= 1
        is_csa = (self.compress_ratio == 4)

        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh
        attn_metadata = get_forward_context().attn_metadata
        assert isinstance(attn_metadata, dict)

        # --- SWA layer cache + metadata (always present)
        swa_layer_name = self.swa_cache_layer.prefix
        swa_attn_metadata = attn_metadata[swa_layer_name]
        swa_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            swa_layer_name]
        sw_cache = wrapper_ctx.kv_caches[swa_cache_index]
        attention_sinks = jax_view(self.attn_sink)

        if is_csa:
            assert topk_indices is not None
            extra = jax_view(topk_indices)  # i32[max_num_tokens, csa_topk]
        else:
            # `positions` is the rope position of each q token: its
            # 0-indexed absolute position within its own sequence. HCA is
            # causal, so a query token at position p attends to kv indices
            # [0, p], i.e. p + 1 tokens.
            q_positions = jax_view(positions)  # i32[num_tokens]
            extra = (q_positions + 1).astype(jnp.int32) // self.compress_ratio

        two_caches_same_buffer = False
        main_cache_rope = None
        if not swa_only:
            main_layer_name = self.prefix
            main_attn_metadata = attn_metadata[main_layer_name]
            main_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
                main_layer_name]
            main_cache_kv = wrapper_ctx.kv_caches[main_cache_index]

            if is_csa:
                # CSA splits the compressed KV across two arrays: this layer's
                # main array holds the NoPE record and a companion array holds
                # the RoPE channels. Both are written by the compressor.
                main_cache_rope = wrapper_ctx.kv_caches[
                    wrapper_ctx.
                    layer_name_to_kvcache_index[f"{main_layer_name}_rope"]]

            main_kv_lens = main_attn_metadata.seq_lens // self.compress_ratio
            main_page_indices = main_attn_metadata.block_tables
            main_cu_q_lens = main_attn_metadata.query_start_loc
            main_distribution = main_attn_metadata.request_distribution
            two_caches_same_buffer = (main_cache_index == swa_cache_index)
        else:
            # If swa-only, main cache metadata does not exist,
            # we just use swa_attn_metadata as placeholder,
            # these fields won't be used anyway.
            main_cache_kv = sw_cache
            main_kv_lens = swa_attn_metadata.seq_lens
            main_page_indices = swa_attn_metadata.block_tables
            main_cu_q_lens = swa_attn_metadata.query_start_loc
            main_distribution = swa_attn_metadata.request_distribution

        # All array inputs and outputs are sharded along the leading axis on
        # ShardingAxisName.ATTN_DATA (caches on ShardingAxisName.BATCH);
        # `attention_sinks` is replicated. The kernels run per-shard.
        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        in_specs = (
            data_spec,  # q
            data_spec,  # new_kv
            cache_spec,  # sw_cache
            data_spec,  # swa_kv_lens
            data_spec,  # swa_page_indices
            data_spec,  # swa_cu_q_lens
            data_spec,  # swa_distribution
            cache_spec,  # main_cache_kv
            data_spec,  # main_kv_lens
            data_spec,  # extra (topk_indices for CSA / kv_lens_to_attend for HCA)
            data_spec,  # main_page_indices
            data_spec,  # main_cu_q_lens
            data_spec,  # main_distribution
            P(),  # attention_sinks (replicated)
        )
        if main_cache_rope is not None:
            in_specs += (cache_spec, )  # main_cache_rope
        out_specs = (
            data_spec,  # attention output
            cache_spec,  # updated swa cache
        )

        def _attention(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                       swa_cu_q_lens, swa_distribution, main_cache_kv,
                       main_kv_lens, extra, main_page_indices, main_cu_q_lens,
                       main_distribution, attention_sinks,
                       *main_cache_rope_operand):
            swa_output, updated_sw_cache, swa_l, swa_m = (
                mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=new_kv,
                    cache_kv=sw_cache,
                    kv_lens=swa_kv_lens,
                    page_indices=swa_page_indices,
                    cu_q_lens=swa_cu_q_lens,
                    distribution=swa_distribution,
                    attention_sinks=attention_sinks,
                    sm_scale=self.scale,
                    sliding_window=self.window_size,
                    logical_page_size=self.swa_cache_layer.block_size,
                    # TODO: tune num_kv_pages_per_block & num_queries_per_block
                    num_kv_pages_per_block=(2, 2, 2),
                    num_queries_per_block=(1, 32, 32),
                    q_compute_block_size=4,
                    unnormalized_output=False if swa_only else True,
                ))

            if swa_only:
                return swa_output, updated_sw_cache

            if two_caches_same_buffer:
                # main cache and swa cache overlay on the same buffer
                main_cache_kv = updated_sw_cache

            if is_csa:
                # CSA gathers only the top-k KV tokens per query, so it takes
                # `topk_indices` (as `extra`).
                gather_and_attention_chunk_size = 64 if (q.shape[0] %
                                                         64 == 0) else None
                if gather_and_attention_chunk_size is None:
                    # The kernel falls back to a single chunk of q.shape[0];
                    # the batch size must divide it.
                    attention_kernel_batch_size = _largest_divisor(
                        q.shape[0], 16)
                else:
                    attention_kernel_batch_size = 16
                output = sparse_ragged_paged_attention(
                    q=q,
                    cache_kv_nope=main_cache_kv,
                    cache_kv_rope=main_cache_rope_operand[0],
                    topk_indices=extra,
                    page_indices=main_page_indices,
                    cu_q_lens=main_cu_q_lens,
                    distribution=main_distribution,
                    attention_sinks=attention_sinks,
                    swa_accumution=swa_output,
                    swa_l=swa_l,
                    swa_m=swa_m,
                    sm_scale=self.scale,
                    # TODO: tune attention_kernel_batch_size &
                    # gather_and_attention_chunk_size.
                    gather_and_attention_chunk_size=
                    gather_and_attention_chunk_size,
                    attention_kernel_batch_size=attention_kernel_batch_size,
                )
            else:
                output = mla_ragged_paged_attention(
                    q=q,
                    cache_kv=main_cache_kv,
                    kv_lens=main_kv_lens,
                    kv_lens_to_attend=extra,
                    page_indices=main_page_indices,
                    cu_q_lens=main_cu_q_lens,
                    distribution=main_distribution,
                    attention_sinks=attention_sinks,
                    swa_accumution=swa_output,
                    swa_l=swa_l,
                    swa_m=swa_m,
                    sm_scale=self.scale,
                    # TODO: tune num_kv_pages_per_block & num_queries_per_block
                    num_kv_pages_per_block=(16, 16, 16),
                    num_queries_per_block=(1, 32, 32),
                )
            return output, updated_sw_cache

        operands = (
            jax_view(q),
            jax_view(kv),
            sw_cache,
            swa_attn_metadata.seq_lens,
            swa_attn_metadata.block_tables,
            swa_attn_metadata.query_start_loc,
            swa_attn_metadata.request_distribution,
            main_cache_kv,
            main_kv_lens,
            extra,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            attention_sinks,
        )
        if main_cache_rope is not None:
            operands += (main_cache_rope, )

        output, updated_sw_cache = jax.shard_map(
            _attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(*operands)

        wrapper_ctx.kv_caches[swa_cache_index] = updated_sw_cache
        return torch_view(output)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qr_kv, indexer_weights = (self.attn_gemm(hidden_states))
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self.q_norm(qr)
        kv = self.kv_norm(kv)

        attn_output = self.attention_impl(
            hidden_states,
            qr,
            kv,
            indexer_weights,
            positions,
            None,
        )

        return self._o_proj(attn_output, positions)
