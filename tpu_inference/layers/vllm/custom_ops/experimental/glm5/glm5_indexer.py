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
"""TPU-compatible GLM-5.2 Lightning Indexer (Dynamic Sparse Attention).

Mirrors ``tpu_inference/layers/vllm/custom_ops/experimental/deepseek_v4/deepseek_v4_indexer.py``'s
structure, but subclasses upstream vLLM's *generic* DSA ``Indexer``/
``DeepseekV32IndexerCache`` (``vllm/model_executor/models/deepseek_v2.py``,
shared by vanilla DeepSeek-V3.2 and GLM-5.2's ``GlmMoeDsaForCausalLM``)
instead of DeepSeek-V4's bespoke compressed-KV indexer.

Why this subclass is needed at all: upstream ``Indexer.forward()``'s
non-fused-path Q handling calls ``per_token_group_quant_fp8`` (CUDA op with a
Triton fallback -- vllm/model_executor/layers/quantization/utils/fp8_utils.py)
and the fused-path is gated behind ``current_platform.is_cuda()`` -- neither
runs on TPU. ``forward()`` here bypasses both, replacing the entire
Q/K RoPE+quant+top-k pipeline with the Phase 2 GLM-5.2 Pallas kernels
(``kernels/experimental/glm5/indexer/{kv_writer,topk}.py``) and DeepSeek-V4's
reused ``streamindex_topk`` kernel, following this repo's OOT/subclassing
convention rather than editing vLLM's shared ``deepseek_v2.py``.
"""

import os

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (DeepseekV32IndexerCache,
                                                     Indexer)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from tpu_inference.kernels.experimental.deepseek_v4.indexer.streamindex_topk import \
    streamindex_topk
from tpu_inference.kernels.experimental.glm5.indexer.kv_writer import \
    index_qk_rope_quant
from tpu_inference.kernels.experimental.glm5.indexer.topk import (
    indexer_kv_cache_width, )
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

# Physical row packing of the GLM-5.2 indexer's uint8 K-cache: 4 per-token
# records per physical cache row (``pack_indexer_kv_cache``'s convention in
# ``kernels/experimental/glm5/indexer/topk.py`` -- see that module's
# docstring). Must match ``_KV_PACKING`` there; duplicated here (rather than
# importing the private name) since it is also needed for the persistent
# cache's *allocation* shape (`VllmGlm5IndexerCache.get_kv_cache_spec`),
# independent of any single write/read call.
_KV_PACKING = 4

# TEMP DIAGNOSTIC (Phase 4 bring-up debugging): guarded by an env var so it's
# a no-op by default. Remove once bring-up correctness is confirmed.
_GLM5_DSA_DEBUG = bool(int(os.environ.get("GLM5_DSA_DEBUG", "0")))


class VllmGlm5IndexerCache(DeepseekV32IndexerCache):
    """TPU-compatible GLM-5.2 indexer KV-cache spec.

    Upstream ``DeepseekV32IndexerCache.get_kv_cache_spec`` reports
    ``head_size = head_dim + head_dim // quant_block_size * 4`` (one *fp32*
    scale per token, no physical packing -- CUDA/Triton's per-token-row
    layout). The Phase 2 GLM-5.2 Pallas kernels use a different physical
    layout instead (``kernels/experimental/glm5/indexer/topk.py::
    pack_indexer_kv_cache``): one *ue8m0* (1-byte) scale per token, 4 token
    records packed per physical TPU cache row, each record padded to a
    128-byte-aligned width (``indexer_kv_cache_width()``, 256B for GLM-5.2's
    ``head_dim=128``). Override the spec to describe *that* physical layout,
    not vLLM's generic default -- mirrors
    ``deepseek_v4_indexer.py::VllmDeepseekV4IndexerCache`` overriding the
    same method for the same reason (a TPU-kernel-specific packed layout).
    """

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=indexer_kv_cache_width(),
            dtype=torch.uint8,
            alignment=None,
        )


class VllmGlm5Indexer(Indexer):
    """TPU-compatible GLM-5.2 Lightning Indexer with StreamIndex top-k.

    Computes, per forward: fused RMSNorm(K, real affine LayerNorm applied
    here)+interleaved-RoPE+FP8-quant for the current step's new Q/K
    (``kv_writer.index_qk_rope_quant``), scatter-writes the new K into this
    layer's persistent packed uint8 K-cache (own JAX-native write -- no
    Phase 2 kernel covers incremental/scatter cache writes, see the
    docstring of ``_write_new_tokens`` for detail and the caveat this
    leaves), then computes top-k indices over the *full* historical cache
    (``streamindex_topk``, DeepSeek-V4's kernel reused at
    ``compression_ratio=1``) and writes them into the shared
    ``topk_indices_buffer``.
    """

    def __init__(self, *args, **kwargs):
        # The base ctor builds ``self.k_cache`` from vLLM's upstream
        # ``DeepseekV32IndexerCache``. Temporarily rebind the module-level
        # symbol the base ctor references so it constructs the TPU subclass
        # directly (same idiom as
        # ``deepseek_v4_indexer.py::VllmDeepseekV4Indexer.__init__``).
        orig_cache = deepseek_v2.DeepseekV32IndexerCache
        deepseek_v2.DeepseekV32IndexerCache = VllmGlm5IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            deepseek_v2.DeepseekV32IndexerCache = orig_cache

        # `get_current_vllm_config()` must NOT be called from `forward()`:
        # vLLM's decoder-layer forward runs inside a torch.compile/dynamo-
        # traced region (confirmed via a live crash on the TPU VM --
        # `AssertionError: Current vLLM config is not set` -- the global
        # `set_current_vllm_config()` context isn't guaranteed active during
        # traced/compiled replay). `__init__` runs at model-construction
        # time instead, safely inside that context (the established,
        # already-working convention elsewhere in this repo, e.g.
        # `deepseek_v4_compressor.py::VllmCompressorStateCache.__init__`) --
        # cache every vllm_config-derived value here, once, and read the
        # cached attribute in `forward()`.
        self.block_size = self.vllm_config.cache_config.block_size

        # `kv_writer.index_qk_rope_quant`'s `rope_base` defaults to 10000.0
        # -- GLM-5.2's real checkpoint config uses `rope_theta=8000000`
        # (`config.rope_parameters["rope_theta"]`; confirmed by reading the
        # live checkpoint's config.json). The main-attention path gets this
        # correctly for free (vLLM's own `get_rope(..., rope_parameters=
        # config.rope_parameters, ...)`, unmodified), but the indexer's own
        # separate RoPE (this Pallas kernel, not vLLM's `rotary_emb`) must
        # be threaded explicitly -- silently defaulting to 10000.0 was a
        # real bug (confirmed: not just theoretical) that produced
        # incoherent generations end-to-end on a live TPU VM run with real
        # GLM-5.2-NVFP4 weights.
        self.rope_base = float(self.config.rope_parameters["rope_theta"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,  # unused: RoPE is baked into index_qk_rope_quant below.
    ) -> None:
        del rotary_emb
        wrapper_ctx = get_vllm_model_wrapper_context()
        mesh = wrapper_ctx.mesh

        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        # Fused wk + weights_proj: one GEMM producing [head_dim + n_head],
        # matching upstream ``Indexer.forward``'s non-fused-path split.
        kw, _ = self.wk_weights_proj(hidden_states)
        # `kw[:, :self.head_dim]` is a plain torch slice, which under
        # torchax returns a lazy `torchax.view.View` (not a materialized
        # `torchax.tensor.Tensor`) -- fine for anything that goes through
        # `jax_view`/`torch_view` (both explicitly handle `View`), but
        # upstream vLLM's `LayerNorm.forward` (`self.k_norm` below) ends
        # with a plain `.type_as(x)` call, and torchax's `Tensor.type_as`
        # assumes `other` is a `Tensor` (reads `other._elem` directly) --
        # crashes with `AttributeError: 'View' object has no attribute
        # '_elem'` on a bare View. Round-trip through jax_view/torch_view to
        # materialize a real Tensor before it reaches any plain (non-
        # View-aware) torch op. Confirmed via a live run on the TPU VM
        # (real GLM-5.2-NVFP4 weights) that this crash is real and this
        # fix resolves it.
        k = torch_view(jax_view(kw[:, :self.head_dim]))
        raw_weights = kw[:, self.head_dim:]

        # Real vLLM ``Indexer.k_norm`` is an *affine* nn.LayerNorm (learned
        # weight/bias) -- apply it here (a plain torch op, not a CustomOp,
        # so it traces fine under torchax) and disable the Pallas kernel's
        # own (unweighted RMSNorm) k-norm. Phase 2 carry-forward note (a).
        k = self.k_norm(k)

        block_size = self.block_size

        kv_cache_index = wrapper_ctx.layer_name_to_kvcache_index[
            self.k_cache.prefix]
        kv_cache = wrapper_ctx.kv_caches[kv_cache_index]
        # `forward_context.attn_metadata` is not always a per-layer dict --
        # confirmed via a live crash on the TPU VM
        # (`TypeError: 'AttentionMetadata' object is not subscriptable`):
        # in this runtime it's a single shared `AttentionMetadata` object
        # covering every layer, not a `{layer_name: AttentionMetadata}`
        # dict (unlike what DeepSeek-V4's analogous indexer code assumes --
        # DSv4 has multiple genuinely-distinct KV-cache groups, which may be
        # why its runtime differs). Mirror upstream vLLM's own
        # `get_attention_context()` helper's robust handling
        # (`vllm/model_executor/layers/attention/attention.py`), which
        # already treats both shapes as valid, rather than assuming one.
        attn_metadata_raw = get_forward_context().attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw[self.k_cache.prefix]
        elif isinstance(attn_metadata_raw, list):
            # list[dict[str, AttentionMetadata]]: speculative decoding,
            # where [0] is the base-model (non-speculative) metadata dict.
            attn_metadata = attn_metadata_raw[0][self.k_cache.prefix]
        else:
            attn_metadata = attn_metadata_raw

        rope_dim = self.rope_dim
        rope_base = self.rope_base
        topk_tokens = self.topk_tokens
        softmax_scale = self.softmax_scale
        n_head_scale = self.n_head_scale

        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        in_specs = (
            data_spec,  # q
            data_spec,  # k
            data_spec,  # raw_weights
            data_spec,  # positions
            cache_spec,  # kv_cache
            data_spec,  # seq_lens
            data_spec,  # block_tables
            data_spec,  # cu_q_lens (query_start_loc)
            data_spec,  # distribution
        )
        out_specs = (cache_spec, data_spec)  # (new_kv_cache, topk_indices)

        def _indexer_step(q, k, raw_weights, positions, kv_cache, seq_lens,
                          block_tables, cu_q_lens, distribution):
            q_fp8, q_scale, k_fp8, k_scale = index_qk_rope_quant(
                q,
                k,
                positions,
                rope_dim=rope_dim,
                rope_base=rope_base,
                do_k_norm=False,
            )

            new_kv_cache = _write_new_tokens(
                kv_cache,
                k_fp8,
                k_scale,
                positions,
                block_tables,
                cu_q_lens,
                block_size=block_size,
            )

            q_dequant = q_fp8.astype(jnp.float32) * q_scale[..., None]
            folded_weights = (raw_weights.astype(jnp.float32) *
                              softmax_scale * n_head_scale)

            topk_indices = streamindex_topk(
                q_dequant,
                folded_weights,
                new_kv_cache,
                seq_lens,
                block_tables,
                cu_q_lens,
                distribution,
                k=topk_tokens,
                compression_ratio=1,
                # TODO(tpu-inference): tune num_kv_pages_per_block,
                # num_queries_per_block (mirrors
                # deepseek_v4_indexer.py's un-tuned defaults).
                num_kv_pages_per_block=(3, 2, 2),
                num_queries_per_block=(1, 128, 128),
            )
            if _GLM5_DSA_DEBUG:
                valid = topk_indices >= 0
                num_valid = jnp.sum(valid, axis=-1)
                safe_idx = jnp.where(valid, topk_indices, 0)
                max_idx = jnp.max(jnp.where(valid, safe_idx, -1), axis=-1)
                min_idx = jnp.min(jnp.where(valid, safe_idx, 2**30), axis=-1)
                jax.debug.print(
                    "[glm5-dsa-debug indexer] num_tokens={nt} seq_lens={sl} "
                    "positions[:8]={p} num_valid[:8]={nv} "
                    "max_idx[:8]={mx} min_idx[:8]={mn} "
                    "q_scale_mean={qs} k_scale_mean={ks}",
                    nt=q.shape[0],
                    sl=seq_lens,
                    p=positions[:8],
                    nv=num_valid[:8],
                    mx=max_idx[:8],
                    mn=min_idx[:8],
                    qs=jnp.mean(q_scale),
                    ks=jnp.mean(k_scale))
            return new_kv_cache, topk_indices

        new_kv_cache, topk_indices = jax.shard_map(
            _indexer_step,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            jax_view(q),
            jax_view(k),
            jax_view(raw_weights),
            jax_view(positions),
            kv_cache,
            attn_metadata.seq_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.request_distribution,
        )

        wrapper_ctx.kv_caches[kv_cache_index] = new_kv_cache

        # Shared by reference across every decoder layer within this
        # forward pass (including "skip" layers, which never call this
        # forward and just read it back) via `wrapper_ctx` -- NOT a
        # persistent torch buffer mutated in place. See
        # `VllmModelWrapperContext.topk_indices_buffer`'s docstring for why
        # (a live `jax.errors.UnexpectedTracerError` crash on the TPU VM: a
        # persistent, cross-jit-call Python object's underlying JAX value
        # cannot be reassigned from inside a `jax.jit` trace and read back
        # correctly once the trace exits).
        wrapper_ctx.topk_indices_buffer = topk_indices


def _write_new_tokens(
    kv_cache: jax.Array,  # [num_pages, block_size // 4, 4, width] uint8
    k_fp8: jax.Array,  # [T, head_dim] float8_e4m3fn
    k_scale: jax.Array,  # [T] float32 (power-of-two)
    positions: jax.Array,  # [T] int, absolute logical position per token
    block_tables: jax.Array,  # i32[num_reqs * max_blocks_per_req]
    cu_q_lens: jax.Array,  # i32[num_reqs + 1]
    *,
    block_size: int,
) -> jax.Array:
    """Scatter-writes the current step's new per-token K records into the
    persistent packed indexer K-cache at their physical (page, row, sub-slot)
    locations.

    No Phase 2 kernel covers this (``kv_writer.index_qk_rope_quant`` only
    *computes* new-token Q/K; ``topk.pack_indexer_kv_cache`` only *builds* a
    cache from token index 0, it does not scatter-update an existing one) --
    every other persistent-cache write in this repo (e.g.
    ``compress_and_store/compressor_v1.py::derive_metadata``,
    ``mla/v2``'s fused ragged-paged-attention kernels) is a dedicated Pallas
    kernel, not a plain-JAX scatter. This is therefore new, JAX-native (not
    Pallas) wiring code -- functionally reasoned from
    ``compressor_v1.py::derive_metadata``'s positions-driven block-table
    slot-mapping convention (reused here, not reinvented), but with no
    kernel-level test coverage of its own. Flag this prominently as the
    least-verified piece of the GLM-5.2 DSA wiring; a follow-up Pallas
    scatter-write kernel (or at least a dedicated JAX-level unit test) is
    recommended before relying on it for accuracy verification.

    Args:
      kv_cache: the persistent packed cache, ``[num_pages, block_size // 4,
        4, width]`` uint8 (``width == topk.indexer_kv_cache_width()``).
      k_fp8, k_scale: this step's new tokens' K, from
        ``kv_writer.index_qk_rope_quant``.
      positions: this step's new tokens' absolute logical position within
        their own request (the same tensor ``Indexer.forward`` receives).
      block_tables: the indexer cache's own per-request page table (this
        layer's ``attn_metadata.block_tables``, *not* the main attention
        layer's).
      cu_q_lens: cumulative new-token counts per request, used only to
        derive each token's owning request index (for the block-table row).
      block_size: logical tokens per page (``cache_config.block_size``,
        matching ``VllmGlm5IndexerCache.get_kv_cache_spec``'s ``block_size``
        and this cache's page-table geometry); must be a multiple of 4.

    Returns:
      The updated ``kv_cache`` (same shape/dtype).
    """
    _, _, packing, width = kv_cache.shape
    if packing != _KV_PACKING:
        raise ValueError(f"kv_cache packing {packing} != {_KV_PACKING}")

    num_tokens = positions.shape[0]
    query_lens = jnp.diff(cu_q_lens)
    batch_size = cu_q_lens.shape[0] - 1
    req = jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32),
                     query_lens,
                     total_repeat_length=num_tokens)
    stride = block_tables.shape[0] // batch_size

    page_idx_in_seq = positions // block_size
    offset_in_page = positions % block_size
    page_number = block_tables[req * stride + page_idx_in_seq]
    row_in_page = offset_in_page // _KV_PACKING
    sub_slot = offset_in_page % _KV_PACKING

    k_scale_e8m0 = k_scale.astype(jnp.float8_e8m0fnu)
    k_bytes = jax.lax.bitcast_convert_type(k_fp8, jnp.uint8)  # [T, head_dim]
    scale_bytes = jax.lax.bitcast_convert_type(k_scale_e8m0,
                                               jnp.uint8)[:, None]  # [T, 1]
    record = jnp.concatenate([k_bytes, scale_bytes], axis=-1)
    pad = width - record.shape[-1]
    if pad > 0:
        record = jnp.pad(record, ((0, 0), (0, pad)))
    elif pad < 0:
        raise ValueError(
            f"indexer record width {record.shape[-1]} exceeds cache width "
            f"{width}")

    return kv_cache.at[page_number, row_in_page, sub_slot].set(record)
