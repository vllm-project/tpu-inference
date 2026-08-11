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
"""Sparse MLA attention backend (GLM-5.2 DSA / DeepSeek-V3.2-style top-k).

Registered under ``AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE`` and selected
by ``TpuPlatform.get_attn_backend_cls`` whenever ``MLAAttention`` is built
with ``use_sparse=True`` (see ``tpu_platform.py``) -- i.e. for any model
whose ``MLAModules.is_sparse`` is set (GLM-5.2's ``DeepseekV2MLAAttention``,
via ``config.index_topk`` being present). Reads ``topk_indices`` from the
per-layer ``topk_indices_buffer`` (populated by
``layers/vllm/custom_ops/experimental/glm5/glm5_indexer.py::VllmGlm5Indexer``,
shared/reused across "skip" layers per ``index_topk_freq``) and attends only
over the selected KV positions via
``kernels/experimental/glm5/core_attention/sparse_attention.py::
sparse_causal_attention_kernel``, instead of ``PallasMLAttentionBackendImpl``
(``flash_attn_mla.py``)'s full dense ragged-paged attend.

Known simplifications / open risk areas in this pass (layer-wiring scope --
the underlying kernels are Phase 2's, not modified here):

1. **Cache write reuses the dense kernel, wastefully.** No "write-only" MLA
   cache-update kernel exists in this repo (``mla_ragged_paged_attention``
   fuses the cache write with a full dense attend); rather than write a new
   kernel, this backend calls the existing dense
   ``attention_interface.mla_attention`` for its ``new_kv_cache`` side
   effect and discards its (dense) attention output, then computes the real
   (sparse) output separately via ``sparse_causal_attention_kernel``. This
   is correctness-first, not compute-efficient (it pays for a full dense
   attend it throws away) -- flagged as a follow-up optimization, not a
   correctness concern.
2. **Quantized (FP8) main-attention cache -- RESOLVED.** Originally this
   backend raised ``NotImplementedError`` for a quantized
   (``--kv-cache-dtype=fp8``) main-attention cache, since
   ``sparse_causal_attention_kernel``'s per-row DMA gather required a
   float32 ``cache_kv`` (a Mosaic VMEM/HBM tiling constraint on narrow
   dtypes, not a numerics gap). This surfaced as a hard blocker once
   ``nvidia/GLM-5.2-NVFP4`` (the actual bring-up checkpoint) turned out to
   auto-select an FP8 KV cache regardless of CLI flags. The Phase 2 kernel
   now supports any floating ``cache_kv`` dtype natively (a leading-axis
   VMEM/HBM layout -- see its module docstring), and this backend mirrors
   the dense backend's (``flash_attn_mla.py``) static per-tensor FP8
   quantize-before-cache-write + dequant-scale-forwarding flow: when
   ``layer.kv_cache_quantized_dtype`` is set, Q is quantize/dequantize
   round-tripped (matching the dense kernel's internal FP8 rounding) unless
   ``DISABLE_MLA_Q_ACTIVATION_QUANTIZATION`` is set, K/V are quantized
   before the cache write via ``quantize_kv`` (same as the dense path), and
   the resulting FP8 ``new_kv_cache`` is passed to
   ``sparse_causal_attention_kernel`` directly (no upcast) with
   ``k_scale``/``v_scale`` forwarded for the kernel's dequant-on-gather step.
3. **Row layout assumption.** ``cache_kv``'s per-row layout is assumed to be
   ``concat(kv_c_normed, k_pe)`` (latent then rope), with V equal to the
   leading ``kv_lora_rank`` columns of the same row -- reasoned from
   ``mla/v2/kernel.py::get_kv_cache_shape``'s generic packed-page layout and
   ``sparse_causal_attention_kernel``'s module docstring (which cites
   ``xpu_mla_sparse.py``'s ``v = kv[..., :d_v]`` convention), not verified
   against a live numerical run. Flag for accuracy-verification scrutiny.
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.attention.backend import AttentionBackend, MLAAttentionImpl
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

import tpu_inference.envs as tpu_inference_envs
from tpu_inference.kernels.experimental.glm5.core_attention.sparse_attention import (
    flatten_paged_kv_cache, resolve_topk_to_physical_slots,
    sparse_causal_attention_kernel)
from tpu_inference.layers.common.attention_interface import mla_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import (
    quantize_kv, static_per_tensor_quantize_tensor)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.backends.flash_attn_mla import \
    PallasMLAttentionBackend
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)


def _quantize_dequantize_round_trip(dtype, x: jnp.ndarray, scale: float,
                                    out_dtype: jnp.dtype) -> jnp.ndarray:
    """Quantizes ``x`` to ``dtype`` with ``static_per_tensor_quantize_tensor``
    then immediately dequantizes back to ``out_dtype`` -- simulates the FP8
    rounding the dense backend's fused kernel applies internally to Q,
    without needing to actually store/thread a quantized Q dtype through
    ``sparse_causal_attention_kernel`` (whose Q input is a plain
    auto-pipelined BlockSpec block, not part of the manual per-row KV
    gather, so it isn't subject to that path's tiling constraints -- only
    the *numerics* of quantized Q need reproducing here, not its storage)."""
    q = static_per_tensor_quantize_tensor(dtype, x, scale)
    return (q.astype(jnp.float32) * scale).astype(out_dtype)


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE)
class PallasMLASparseAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_MLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["PallasMLASparseAttentionBackendImpl"]:
        return PallasMLASparseAttentionBackendImpl

    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # Must match the main-attention block size the GLM-5.2 indexer's
        # `resolve_topk_to_physical_slots` call assumes when it converts
        # logical top-k token positions into physical cache rows via this
        # layer's own block table -- keep in lockstep with the dense
        # backend's page size (both are plain MLA latent caches).
        return PallasMLAttentionBackend.get_page_size(vllm_config)


class PallasMLASparseAttentionBackendImpl(MLAAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        qk_head_dim: int | None = None,
        v_head_dim: int | None = None,
        # Sparse-specific: threaded in by `MLAAttention.__init__` only when
        # `use_sparse=True` (see vllm's mla_attention.py); shared/mutated by
        # `VllmGlm5Indexer.forward`, one persistent buffer per model.
        topk_indices_buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.topk_indices_buffer = topk_indices_buffer
        # See the calibration block in `forward()`: freezes after the first
        # call, mirroring vLLM's own `calculate_kv_scales`/`calc_kv_scales`
        # "calibrate once, reuse forever" semantics for a static per-tensor
        # quantization scale.
        self._kv_scale_calibrated = False
        # Main-attention cache block size (tokens/page) -- needed to convert
        # the indexer's logical top-k token positions into physical cache
        # rows via this layer's own block table (`resolve_topk_to_physical_
        # slots`). Read from the live vLLM config rather than hardcoding, so
        # this stays correct if `cache_config.block_size` is ever overridden.
        from vllm.config import get_current_vllm_config
        self._block_size = get_current_vllm_config().cache_config.block_size

    def forward_mha(self, *args, **kwargs) -> None:
        pass

    def forward_mqa(self, *args, **kwargs) -> tuple:
        pass

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                kv_cache: jnp.ndarray,
                attn_metadata: AttentionMetadata,
                mesh: Mesh,
                layer: MLAAttention,
                output: torch.Tensor | None = None,
                **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.topk_indices_buffer is None:
            raise RuntimeError(
                "PallasMLASparseAttentionBackendImpl requires a "
                "topk_indices_buffer (use_sparse=True); got None. This "
                "indicates the model's MLAModules.is_sparse wiring is "
                "inconsistent with the selected attention backend.")
        wrapper_ctx = get_vllm_model_wrapper_context()

        q_nope, q_pe = q
        q_nope_j = jax_view(q_nope)
        q_pe_j = jax_view(q_pe)
        kv_c_normed_j = jax_view(kv_c_normed)
        k_pe_j = jax_view(k_pe)
        input_dtype = q_nope_j.dtype
        num_tokens = q_nope_j.shape[0]

        if tpu_inference_envs.GLM5_DSA_DEBUG and not getattr(
                self, "_dbg_shapes_logged", False):
            # Shapes are static (known at trace time even under jax.jit) --
            # a plain Python-level log (not jax.debug.print) is fine and
            # only needs to fire once. Verifying GLM-5.2's real (non-128)
            # MLA head dims (qk_nope_head_dim=192, v_head_dim=256,
            # q_lora_rank=2048, per the real HF config) are actually
            # reaching this Impl at runtime, not silently defaulting/
            # mismatching -- see coordinator's hardcoded-128-audit request.
            logger.info(
                "[glm5-dsa-debug shapes] self.num_heads=%s "
                "self.qk_nope_head_dim=%s self.qk_rope_head_dim=%s "
                "self.v_head_dim=%s self.kv_lora_rank=%s "
                "self.q_lora_rank=%s q_nope_j.shape=%s q_pe_j.shape=%s "
                "kv_c_normed_j.shape=%s k_pe_j.shape=%s W_UK_T.shape=%s "
                "W_UV.shape=%s", self.num_heads, self.qk_nope_head_dim,
                self.qk_rope_head_dim, self.v_head_dim, self.kv_lora_rank,
                self.q_lora_rank, q_nope_j.shape, q_pe_j.shape,
                kv_c_normed_j.shape, k_pe_j.shape,
                jax_view(layer.W_UK_T).shape, jax_view(layer.W_UV).shape)
            self._dbg_shapes_logged = True

        if tpu_inference_envs.GLM5_DSA_DEBUG:
            jax.debug.print(
                "[glm5-dsa-debug pre-quant] kv_c_normed_mean_abs={km} "
                "kv_c_normed_max_abs={kx} k_pe_mean_abs={pm} "
                "k_pe_max_abs={px} kv_c_normed_dtype={kd}",
                km=jnp.mean(jnp.abs(kv_c_normed_j.astype(jnp.float32))),
                kx=jnp.max(jnp.abs(kv_c_normed_j.astype(jnp.float32))),
                pm=jnp.mean(jnp.abs(k_pe_j.astype(jnp.float32))),
                px=jnp.max(jnp.abs(k_pe_j.astype(jnp.float32))),
                kd=str(kv_c_normed_j.dtype))

        # Absorb q_nope into the shared latent space via W_UK_T, exactly as
        # the dense backend does -- the gathered cache rows are the *latent*
        # (kv_lora_rank + rope) representation, not per-head K/V, so q must
        # be absorbed the same way to attend against them directly.
        q_nope_absorbed = jnp.einsum("bnp,npl->nbl",
                                     q_nope_j,
                                     jax_view(layer.W_UK_T),
                                     preferred_element_type=jnp.float32)
        scale = jax_view(layer.W_UK_T_scale)
        q_nope_absorbed = (q_nope_absorbed * scale).astype(input_dtype)

        if tpu_inference_envs.GLM5_DSA_DEBUG:
            # Unquantized copies, saved before any fp8 round-trip below, for
            # a "ground truth" debug reference computed directly from
            # full-precision values (see the second debug block after
            # `physical_slots` is computed).
            _dbg_q_nope_absorbed_full = q_nope_absorbed
            _dbg_q_pe_full = q_pe_j
            _dbg_kv_c_normed_full = kv_c_normed_j
            _dbg_k_pe_full = k_pe_j

        if (layer.kv_cache_quantized_dtype is not None
                and not self._kv_scale_calibrated):
            # vLLM's own generic `calculate_kv_scales`/`calc_kv_scales`
            # calibration mechanism is broken for MLA models in this repo's
            # shared `VllmMultiHeadLatentAttentionWrapper.forward()` (passes
            # `q` -- an MLA `(q_nope, q_pe)` *tuple* -- straight into
            # `torch.ops.vllm.maybe_calc_kv_scales`, whose registered op
            # signature expects a plain `Tensor`; confirmed via a live crash
            # on the TPU VM with `--calculate-kv-scales`:
            # "RuntimeError: vllm::maybe_calc_kv_scales() Expected a value of
            # type 'Tensor' for argument 'query' but instead found type
            # 'tuple'." -- a pre-existing, generic bug, not GLM-5.2-specific,
            # out of scope to fix in the shared wrapper here). GLM-5.2-NVFP4's
            # attention layers are also excluded from the checkpoint's own
            # NVFP4 calibration (`quantization_config.ignore` covers
            # `self_attn*`), so there is no pretrained k_scale/v_scale to
            # load either -- `layer._k_scale_float`/`_v_scale_float` silently
            # default to 1.0 (vLLM's documented fallback when neither
            # calibration nor a checkpoint value is available).
            #
            # That default is fatally wrong for *this* backend's design:
            # dense MLA attention (`flash_attn_mla.py`, reused here only for
            # its cache-write side effect) never actually round-trips
            # freshly-written tokens through the quantized cache for its own
            # attention output (a FlashAttention-style same-batch fast path),
            # so an uncalibrated scale never surfaces there -- but this
            # sparse backend's gather *always* reads via the physical cache,
            # even for same-batch tokens, fully exposing fp8 e4m3's ~0.002
            # underflow floor when real activation magnitudes are well below
            # 1.0. Confirmed live: with `k_scale=v_scale=1.0`, the physical
            # cache's mean |value| was ~3e-4 (mostly-zero, underflowed) and
            # this backend's attention output was ~1000x smaller in
            # magnitude than the (discarded) dense output computed from the
            # exact same inputs/cache in the same call -- not a NaN/crash,
            # just silently near-zero, propagating into incoherent
            # generations.
            #
            # A *data-dependent* calibration (mirroring `MLAAttention.
            # calc_kv_scales`'s own `abs_max / range` formula, computed
            # directly on the still-torch `kv_c_normed` input here instead of
            # through the broken custom op) was attempted first, but this
            # entire `forward()` runs inside an outer `jax.jit` trace (the
            # runner's compiled per-step function, confirmed via a live
            # crash: `jax.errors.ConcretizationTypeError: Abstract tracer
            # value encountered where concrete value is expected` from
            # `torch.abs(kv_c_normed).max().item()` -- `.item()` requires a
            # concrete value, which isn't available mid-trace). Properly
            # fixing this needs either restructuring
            # `sparse_causal_attention_kernel` to accept `k_scale`/`v_scale`
            # as a dynamic (scalar-prefetch) kernel input instead of a
            # compile-time-static Python float, or computing the scale during
            # an eager (non-jitted) warmup/dummy-run phase if this runner has
            # one -- both out of scope for this bring-up pass. Interim fix:
            # a fixed, environment-overridable static scale, chosen from a
            # live measurement of this checkpoint's actual `kv_c_normed`
            # magnitude (~3e-4 mean-abs observed via `GLM5_DSA_DEBUG=1` with
            # the previous `k_scale=v_scale=1.0` default) rather than an
            # arbitrary guess -- but still a heuristic constant, not true
            # per-layer calibration; flagged prominently as a follow-up item.
            default_scale = tpu_inference_envs.GLM5_DSA_KV_SCALE_DEFAULT
            layer._k_scale_float = default_scale
            layer._v_scale_float = default_scale
            self._kv_scale_calibrated = True

        # `k_pe` (the rope half of K) needs a *separate*, much larger scale
        # than `kv_c_normed`/`k_pe`'s shared `k_scale` above -- see the
        # extensive comment in `sparse_causal_attention_kernel`'s kernel
        # body (`_sparse_attention_kernel_body`) for the full
        # measured-magnitude justification (~100-1000x difference,
        # confirmed live). Same "fixed, env-overridable static heuristic,
        # not true calibration" caveat as `GLM5_DSA_KV_SCALE_DEFAULT` above
        # applies here too.
        k_scale_rope = tpu_inference_envs.GLM5_DSA_KV_SCALE_ROPE_DEFAULT

        q_scale = k_scale = v_scale = None
        if layer.kv_cache_quantized_dtype:
            q_scale = layer._q_scale_float
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float

            if not tpu_inference_envs.DISABLE_MLA_Q_ACTIVATION_QUANTIZATION:
                # Round-trip Q through the same FP8 grid the dense backend's
                # fused kernel applies internally -- see
                # `_quantize_dequantize_round_trip`'s docstring.
                q_nope_absorbed = _quantize_dequantize_round_trip(
                    layer.kv_cache_quantized_dtype, q_nope_absorbed, q_scale,
                    input_dtype)
                q_pe_j = _quantize_dequantize_round_trip(
                    layer.kv_cache_quantized_dtype, q_pe_j, q_scale,
                    input_dtype)
            else:
                # Needed because q_pe comes in as FP32 (mirrors the dense
                # backend's else-branch).
                q_pe_j = q_pe_j.astype(input_dtype)

            # Quantize K/V *before* the cache write, exactly as the dense
            # backend does -- `mla_attention` writes `kv_c_normed`/`k_pe` to
            # `new_kv_cache` verbatim, so this is what determines
            # `new_kv_cache`'s actual on-disk dtype (must match what
            # `kv_cache_manager.py` allocated for `--kv-cache-dtype=fp8`).
            kv_c_normed_j, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                           kv_c_normed_j,
                                           value=None,
                                           k_scale=k_scale)
            # Rope portion quantized with its own, much larger scale --
            # see the `k_scale_rope` note above.
            k_pe_j, _ = quantize_kv(layer.kv_cache_quantized_dtype,
                                    k_pe_j,
                                    value=None,
                                    k_scale=k_scale_rope)

        k_pe_dense = k_pe_j.squeeze(1)

        # Reuse the dense cache-write+attend kernel purely for its
        # `new_kv_cache` side effect (see module docstring risk (1)); the
        # dense attention output is discarded.
        new_kv_cache, _dense_out = mla_attention(
            q_nope_absorbed,
            q_pe_j,
            kv_c_normed_j,
            k_pe_dense,
            kv_cache,
            attn_metadata,
            mesh,
            self.num_heads,
            self.qk_nope_head_dim,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sm_scale=self.scale,
        )

        # `sparse_causal_attention_kernel`'s per-row gather supports any
        # floating `cache_kv` dtype natively (including the FP8
        # `new_kv_cache` written above when `layer.kv_cache_quantized_dtype`
        # is set -- see its module docstring's "Quantized (FP8) cache
        # support" section); no upcast needed. `k_scale`/`v_scale` default to
        # 1.0 (a no-op) below for an unquantized cache.
        flat_cache = flatten_paged_kv_cache(new_kv_cache)

        # q_nope_absorbed is head-major [H, T, L] (matching
        # `mla_attention`'s "head-major" convention); q_pe is token-major
        # [T, H, R]. `sparse_causal_attention_kernel` wants one token-major
        # [T, H, L+R] array (K-cache row layout == concat(latent, rope), see
        # module docstring risk (3)).
        q_nope_token_major = jnp.transpose(q_nope_absorbed, (1, 0, 2))
        q_combined = jnp.concatenate([q_nope_token_major, q_pe_j], axis=-1)

        # `mla/v2/kernel.py::get_kv_cache_shape` pads each cache row's width
        # up to a 128-element alignment boundary
        # (`unsigned_align_to(kv_dim, 128)`), so `flat_cache`'s row is wider
        # than `q_combined`'s true `kv_lora_rank + qk_rope_head_dim` content
        # (confirmed via a live shape-mismatch crash on the TPU VM:
        # cache_kv head_dim 640 != q's 576, i.e. align_to(576, 128) == 640).
        # The cache is zero-initialized (`kv_cache.py::_get_kv_cache_
        # allocator`'s `jnp.zeros`) and only the real
        # `kv_lora_rank + qk_rope_head_dim` columns are ever written by the
        # dense cache-write kernel above, so the trailing columns are
        # alignment padding, not real data -- truncate them off rather than
        # padding `q_combined` up (avoids relying on an unverified
        # assumption that the two sides' zero-padding would coincide
        # exactly under quantization).
        qk_head_dim = q_combined.shape[-1]
        if flat_cache.shape[-1] != qk_head_dim:
            flat_cache = flat_cache[:, :qk_head_dim]

        # Read from `wrapper_ctx` (per-forward-pass, trace-scoped), not
        # `self.topk_indices_buffer` (a persistent torch buffer whose
        # in-place mutation across the jit boundary crashes -- see
        # `VllmModelWrapperContext.topk_indices_buffer`'s docstring).
        topk_indices = wrapper_ctx.topk_indices_buffer
        if topk_indices is None:
            raise RuntimeError(
                "PallasMLASparseAttentionBackendImpl: "
                "wrapper_ctx.topk_indices_buffer is None -- no indexer "
                "layer has run yet this forward pass. This indicates a "
                "layer-ordering bug (a sparse attention layer ran before "
                "any indexer, including 'skip' layers before the first "
                "real indexer layer) or a wiring gap.")
        block_table = attn_metadata.block_tables
        num_reqs = attn_metadata.seq_lens.shape[0]
        block_table_2d = block_table.reshape(num_reqs, -1)
        req_ids = _token_to_req_ids(attn_metadata.query_start_loc, num_tokens)
        physical_slots = resolve_topk_to_physical_slots(
            topk_indices,
            block_table_2d,
            req_ids,
            block_size=self._block_size,
        )

        if tpu_inference_envs.GLM5_DSA_DEBUG:
            total_slots = flat_cache.shape[0]
            valid = physical_slots >= 0
            safe = jnp.where(valid, physical_slots, 0)
            max_slot = jnp.max(jnp.where(valid, safe, -1))
            min_slot = jnp.min(jnp.where(valid, safe, 2**30))
            oob = jnp.sum((safe >= total_slots) & valid)
            jax.debug.print(
                "[glm5-dsa-debug sparse-attn] num_tokens={nt} "
                "total_slots={ts} max_slot={mx} min_slot={mn} "
                "num_out_of_bounds={oob} num_valid[:8]={nv}",
                nt=num_tokens,
                ts=total_slots,
                mx=max_slot,
                mn=min_slot,
                oob=oob,
                nv=jnp.sum(valid, axis=-1)[:8])

            # "Ground truth" reference: gather K/V for the *same* topk
            # selection directly from the full-precision (never quantized)
            # per-call kv_c_normed/k_pe -- valid because this bring-up is
            # all-prefill (no persisted-from-a-previous-call tokens), so
            # every selected logical position is also present in *this
            # call's* own kv_c_normed_j/k_pe_j at row
            # `cu_q_lens[req] + logical_pos` (the same concatenation order
            # cu_q_lens describes). Bypasses both the quantized physical
            # cache *and* the Pallas kernel entirely -- a fully independent
            # cross-check of dense_out vs. sparse_out, computed in plain JAX.
            cu_q_lens = attn_metadata.query_start_loc
            local_row = cu_q_lens[req_ids][:, None] + jnp.clip(
                topk_indices, 0, None)
            local_row = jnp.clip(local_row, 0, num_tokens - 1)
            gathered_kv_c = _dbg_kv_c_normed_full[local_row]  # [T, topk, L]
            gathered_k_pe = _dbg_k_pe_full[local_row, 0]  # [T, topk, R]
            q_full = jnp.concatenate([
                jnp.transpose(_dbg_q_nope_absorbed_full, (1, 0, 2)),
                _dbg_q_pe_full
            ],
                                     axis=-1)  # [T, H, L+R]
            k_full = jnp.concatenate([gathered_kv_c, gathered_k_pe],
                                     axis=-1)  # [T, topk, L+R]
            logits_full = jnp.einsum(
                "thd,tmd->thm", q_full.astype(jnp.float32),
                k_full.astype(jnp.float32)) * self.scale
            logits_full = jnp.where(valid[:, None, :], logits_full, -1e30)
            p_full = jax.nn.softmax(logits_full, axis=-1)
            pv_full = jnp.einsum("thm,tmd->thd", p_full,
                                 gathered_kv_c.astype(jnp.float32))
            out_full = jnp.transpose(pv_full, (1, 0, 2))  # [H, T, L]
            out_full = (jnp.einsum("nbl,nlv->bnv",
                                   out_full,
                                   jax_view(layer.W_UV),
                                   preferred_element_type=jnp.float32) *
                       jax_view(layer.W_UV_scale)).astype(input_dtype)
            out_full = out_full.reshape(-1, self.num_heads * self.v_head_dim)
            jax.debug.print(
                "[glm5-dsa-debug unquantized-ref] "
                "unquant_ref_mean_abs={rm}", rm=jnp.mean(
                    jnp.abs(out_full.astype(jnp.float32))))

        # `sparse_causal_attention_kernel` is a raw `pl.pallas_call` (Mosaic
        # kernel) -- confirmed via a live crash on the TPU VM
        # (`NotImplementedError: Mosaic kernels cannot be automatically
        # partitioned. Please wrap the call in a shard_map`) that it must be
        # invoked inside an explicit `jax.shard_map`, matching every other
        # raw Pallas kernel call in this codebase (e.g. `mla_attention`'s
        # own internal `jax.shard_map` around `mla_ragged_paged_attention`,
        # `glm5_indexer.py`'s `_indexer_step`) -- unlike those, this call
        # site was not yet wrapped, an oversight in the initial pass.
        v_head_dim = self.kv_lora_rank
        topk = topk_indices.shape[-1]
        softmax_scale = self.scale
        k_scale_static = k_scale if k_scale is not None else 1.0
        v_scale_static = v_scale if v_scale is not None else 1.0
        k_scale_rope_static = k_scale_rope if k_scale is not None else 1.0

        if tpu_inference_envs.GLM5_DSA_DEBUG:
            jax.debug.print(
                "[glm5-dsa-debug sparse-attn-inputs] "
                "q_combined_mean_abs={qm} flat_cache_mean_abs={fm} "
                "flat_cache_dtype={fd} q_combined_dtype={qd} "
                "k_scale={ks} v_scale={vs} k_scale_rope={ksr} topk={tk} "
                "v_head_dim={vhd} softmax_scale={ss} qk_head_dim={qhd}",
                qm=jnp.mean(jnp.abs(q_combined.astype(jnp.float32))),
                fm=jnp.mean(jnp.abs(flat_cache.astype(jnp.float32))),
                fd=str(flat_cache.dtype),
                qd=str(q_combined.dtype),
                ks=k_scale_static,
                vs=v_scale_static,
                ksr=k_scale_rope_static,
                tk=topk,
                vhd=v_head_dim,
                ss=softmax_scale,
                qhd=qk_head_dim)

        def _sparse_attn(q_combined, flat_cache, physical_slots):
            return sparse_causal_attention_kernel(
                q_combined,
                flat_cache,
                physical_slots,
                v_head_dim=v_head_dim,
                topk=topk,
                softmax_scale=softmax_scale,
                k_scale=k_scale_static,
                v_scale=v_scale_static,
                k_scale_rope=k_scale_rope_static,
            )

        data_spec = P(ShardingAxisName.ATTN_DATA)
        cache_spec = P(ShardingAxisName.BATCH)
        sparse_out = jax.shard_map(
            _sparse_attn,
            mesh=mesh,
            in_specs=(data_spec, cache_spec, data_spec),
            out_specs=data_spec,
            check_vma=False,
        )(q_combined, flat_cache, physical_slots)  # [T, H, kv_lora_rank]

        # Un-absorb via W_UV, mirroring the dense backend's tail.
        # `sparse_out` is already `[T, H, kv_lora_rank]` per
        # `sparse_causal_attention_kernel`'s contract.
        outputs = jnp.transpose(sparse_out, (1, 0, 2))  # [H, T, L]
        outputs = (jnp.einsum("nbl,nlv->bnv",
                              outputs,
                              jax_view(layer.W_UV),
                              preferred_element_type=jnp.float32) *
                   jax_view(layer.W_UV_scale)).astype(input_dtype)
        outputs = outputs.reshape(-1, self.num_heads * self.v_head_dim)

        if tpu_inference_envs.GLM5_DSA_DEBUG:
            # Since topk (2048) >> these short prompts' seq lens, the sparse
            # output should be numerically very close to what the (already-
            # computed, otherwise-discarded) dense attention output would
            # give -- un-absorb `_dense_out` the same way and diff them.
            dense_outputs = (jnp.einsum("nbl,nlv->bnv",
                                        _dense_out,
                                        jax_view(layer.W_UV),
                                        preferred_element_type=jnp.float32) *
                             jax_view(layer.W_UV_scale)).astype(input_dtype)
            dense_outputs_flat = dense_outputs.reshape(
                -1, self.num_heads * self.v_head_dim)
            diff = jnp.abs(
                outputs.astype(jnp.float32) -
                dense_outputs_flat.astype(jnp.float32))
            denom = jnp.abs(dense_outputs_flat.astype(jnp.float32)) + 1e-6
            jax.debug.print(
                "[glm5-dsa-debug sparse-vs-dense] mean_abs_diff={md} "
                "max_abs_diff={xd} mean_rel_diff={mr} "
                "sparse_has_nan={sn} sparse_mean_abs={sm} "
                "dense_mean_abs={dm}",
                md=jnp.mean(diff),
                xd=jnp.max(diff),
                mr=jnp.mean(diff / denom),
                sn=jnp.any(jnp.isnan(outputs.astype(jnp.float32))),
                sm=jnp.mean(jnp.abs(outputs.astype(jnp.float32))),
                dm=jnp.mean(jnp.abs(dense_outputs_flat.astype(jnp.float32))))

        out_torch = torch_view(outputs)
        if output is not None:
            output.copy_(out_torch)
        return out_torch, new_kv_cache


def _token_to_req_ids(query_start_loc: jnp.ndarray,
                      num_tokens: int) -> jnp.ndarray:
    """Per-token request id from cumulative query lengths, the same
    convention used by ``compress_and_store/compressor_v1.py::
    derive_metadata`` (``jnp.repeat`` over per-request query lengths)."""
    query_lens = jnp.diff(query_start_loc)
    batch_size = query_start_loc.shape[0] - 1
    return jnp.repeat(jnp.arange(batch_size, dtype=jnp.int32),
                      query_lens,
                      total_repeat_length=num_tokens)
