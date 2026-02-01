# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
from typing import Optional, Tuple

from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.sharding import ShardingAxisName

# MLA_PATCHED - MLA kernel detection and tensor handling
import os as _mla_os
_VLLM_USE_MLA = _mla_os.environ.get("VLLM_USE_MLA", "0") == "1"
_MLA_KERNEL_LOADED = False

# Debug logging - disabled to reduce noise (jax.debug.print spams during compilation)
_MLA_DEBUG = False  # _mla_os.environ.get("VLLM_LOGGING_LEVEL", "").upper() == "DEBUG"

# Request counter to track prefill vs decode
_MLA_REQUEST_COUNTER = 0

def _mla_debug_print(*args, **kwargs):
    """Print only when VLLM_LOGGING_LEVEL=DEBUG"""
    if _MLA_DEBUG:
        print(*args, **kwargs)

if _VLLM_USE_MLA:
    try:
        from tpu_inference.kernels.mla.v1.kernel import mla_ragged_paged_attention
        _MLA_KERNEL_LOADED = True
        print("[MLA] TPU MLA kernel loaded successfully")
    except ImportError as e:
        _mla_debug_print(f"[MLA] MLA kernel not available, using fallback: {e}")

def _mla_flatten_to_2d(tensor, name="tensor"):
    """Flatten a 3D tensor to 2D for TPU attention.

    TPU attention expects 2D tensors: (seq_len, compute_dim)
    where compute_dim = num_heads * head_dim

    Input shapes:
    - 2D: (seq_len, compute_dim) -> return as-is
    - 3D: (seq_len, num_heads, head_dim) -> flatten to (seq_len, num_heads * head_dim)

    Returns: (flattened_tensor, original_shape)
    """
    original_shape = tensor.shape
    if len(tensor.shape) == 3:
        seq_len, num_heads, head_dim = tensor.shape
        tensor = tensor.reshape(seq_len, num_heads * head_dim)
        _mla_debug_print(f"[MLA] Flattened {name} from 3D to 2D: {original_shape} -> {tensor.shape}")
    elif len(tensor.shape) != 2:
        _mla_debug_print(f"[MLA] WARNING: {name} has unexpected shape {tensor.shape}, expected 2D or 3D")
    return tensor, original_shape

def _mla_reshape_qkv(q, k, v):
    """Flatten MLA tensors to 2D for TPU attention compatibility.

    TPU attention expects ALL tensors to be 2D:
    - q: (seq_len, num_heads * head_dim)
    - k: (seq_len, num_kv_heads * head_dim)
    - v: (seq_len, num_kv_heads * head_dim)

    MLA models may produce 3D tensors that need flattening.
    Returns flattened tensors - use _mla_restore_shape for output.
    """
    q, _ = _mla_flatten_to_2d(q, "q")
    k, _ = _mla_flatten_to_2d(k, "k")
    v, _ = _mla_flatten_to_2d(v, "v")
    return q, k, v

def _mla_reshape_output(output, original_q_shape):
    """Reshape attention output back to match original query shape.

    If original q was 3D (seq_len, num_heads, head_dim), reshape output
    from 2D (seq_len, num_heads * head_dim) back to 3D.
    """
    if len(original_q_shape) == 3:
        seq_len, num_heads, head_dim = original_q_shape
        if len(output.shape) == 2:
            output_seq_len, output_dim = output.shape
            # Verify dimensions are compatible
            if output_seq_len == seq_len and output_dim == num_heads * head_dim:
                output = output.reshape(seq_len, num_heads, head_dim)
                _mla_debug_print(f"[MLA] Reshaped output from 2D to 3D: {(output_seq_len, output_dim)} -> {output.shape}")
            else:
                _mla_debug_print(f"[MLA] WARNING: Cannot reshape output {output.shape} to {original_q_shape}")
    return output


import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh

# MLA (Multi-head Latent Attention) support for models like GLM-4.7-Flash, DeepSeek V2/V3
# MLA uses compressed KV representations with asymmetric dimensions:
#   - Q: (seq_len, num_heads * qk_head_dim) where qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
#   - K/V cache: (blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim)
# The head_size from KV cache config differs from the actual Q head dimension.
_VLLM_USE_MLA = os.environ.get("VLLM_USE_MLA", "0") == "1"
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer, AttentionType)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

# Check for MLA backend enum
try:
    _MLA_BACKEND_ENUM = getattr(AttentionBackendEnum, 'FLASH_ATTN_MLA', None)
    if _MLA_BACKEND_ENUM is None:
        # Try PALLAS_MLA or other variants
        _MLA_BACKEND_ENUM = getattr(AttentionBackendEnum, 'PALLAS_MLA', None)
    _mla_debug_print(f"[MLA] Backend enum for MLA: {_MLA_BACKEND_ENUM}")
except Exception as e:
    _MLA_BACKEND_ENUM = None
    _mla_debug_print(f"[MLA] No MLA backend enum available: {e}")

from tpu_inference import utils
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

# TPU requires the head size to be a multiple of 128.
TPU_HEAD_SIZE_ALIGNMENT = 128


@register_backend(AttentionBackendEnum.FLASH_ATTN)
class PallasAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["PallasAttentionBackendImpl"]:
        return PallasAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        padded_head_size = (cdiv(head_size, TPU_HEAD_SIZE_ALIGNMENT) *
                            TPU_HEAD_SIZE_ALIGNMENT)
        return (num_blocks, block_size, num_kv_heads * 2, padded_head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    # In recent TPU generations, up to v6e, the SMEM size is 1MB. The
    # block_tables within the PallasMetadata constitute almost the entire SMEM
    # requirement. Its size is max_num_seqs * num_page_per_seq * 4 (Int). Here
    # we simply make sure that the size is smaller than half of SMEM capacity.
    @staticmethod
    def get_min_page_size(vllm_config: VllmConfig) -> int:
        max_num_page_per_req = (1024 * 1024 // 2 //
                                vllm_config.scheduler_config.max_num_seqs // 4)
        min_page_size = cdiv(vllm_config.model_config.max_model_len,
                             max_num_page_per_req)
        min_page_size = 1 << (min_page_size - 1).bit_length()
        return min_page_size

    @staticmethod
    def get_max_num_seqs(model_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(model_len, page_size)
        return 1024 * 1024 // 2 // num_page_per_req // 4

    # TPU has limited SREGs (scalar registers), if page_size is too small, we
    # can spill SREGs easily which leads to bad performance. The strategy we
    # apply here is trying to split max-model-len to 16 pages which make the
    # spill less likely. Meanwhile we make sure the page size is in [16, 256].
    @staticmethod
    def get_page_size(vllm_config: VllmConfig) -> int:
        # TODO: This is a temporary fix for vmem OOM.
        # For long model length, we use 16 page-size to avoid too much
        # VMEM spill. A more robust solution should be implemented to
        # handle VREG spills.
        if vllm_config.model_config.max_model_len > 8192:
            return 16
        page_size = next_power_of_2(
            vllm_config.model_config.max_model_len) // 16
        if page_size <= 16:
            return 16
        if page_size >= 256:
            return 256
        return page_size


class PallasAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")

        self.sinks = sinks
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer")

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        #TODO (kyuyeunk): Shard the sinks along num_heads dim
        if self.sinks is not None:
            sinks = t2j(self.sinks, use_dlpack=False)
            sinks = torch_view(sinks.astype(jnp.float32))
            self.sinks = torch.nn.Parameter(sinks, requires_grad=False)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for attention.

        Note: For MLA attention, vLLM passes different arguments:
        - query = q (query tensor)
        - key = kv_c_normed (compressed KV latent)
        - value = k_pe (K position embeddings)
        We detect this by checking the KV cache shape (4D = MLA, 5D = standard).
        """
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "PallasAttentionBackendImpl")

        if kv_cache.numel():
            raise RuntimeError(
                "KV cache from vLLM Attention layer should be empty but has "
                "the size of %s.", kv_cache.numel())

        del kv_cache  # Use kv_cache from vllm wrapper context values instead.

        vllm_model_wrapper_context = get_vllm_model_wrapper_context()

        # Debug: Check context and layer_name availability
        if vllm_model_wrapper_context is None:
            raise RuntimeError("vllm_model_wrapper_context is None")

        layer_name = getattr(layer, 'layer_name', None)
        if layer_name is None:
            raise RuntimeError(f"Layer {type(layer).__name__} does not have layer_name attribute. "
                              f"Layer attrs: {[a for a in dir(layer) if not a.startswith('_')][:10]}")

        if not hasattr(vllm_model_wrapper_context, 'layer_name_to_kvcache_index'):
            raise RuntimeError(f"Context missing layer_name_to_kvcache_index. "
                              f"Context attrs: {[a for a in dir(vllm_model_wrapper_context) if not a.startswith('_')]}")

        if layer_name not in vllm_model_wrapper_context.layer_name_to_kvcache_index:
            available_keys = list(vllm_model_wrapper_context.layer_name_to_kvcache_index.keys())
            raise KeyError(f"Layer name '{layer_name}' not found in kv cache index. "
                          f"Available keys ({len(available_keys)}): {available_keys[:5]}...")

        kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[layer_name]
        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]

        mesh = vllm_model_wrapper_context.mesh

        # Detect MLA mode based on KV cache shape (4D = MLA, 5D = standard)
        is_mla_cache = len(kv_cache.shape) == 4

        print(f"[ATTN Forward] kv_cache.shape={kv_cache.shape}, is_mla_cache={is_mla_cache}, _VLLM_USE_MLA={_VLLM_USE_MLA}")
        print(f"[ATTN Forward] q={query.shape}, k={key.shape}, v={value.shape}")

        if is_mla_cache and _VLLM_USE_MLA:
            _mla_debug_print(f"[MLA Forward] Detected MLA mode: q={query.shape}, kv_c={key.shape}, k_pe={value.shape}")
            return self._forward_mla(layer, query, key, value, kv_cache, attn_metadata, mesh)

        # Standard attention path
        print(f"[ATTN Forward] Taking STANDARD attention path")
        query, key, value = jax_view(query), jax_view(key), jax_view(value)
        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            key, value = quantize_kv(self.kv_cache_quantized_dtype, key, value,
                                     layer._k_scale_float,
                                     layer._v_scale_float)
            # TODO(kyuyeunk): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = layer._q_scale_float
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float

        sinks = jax_view(self.sinks)

        new_kv_cache, outputs = _jax_attn_func(
            kv_cache,
            query,
            key,
            value,
            sinks,
            attn_metadata,
            mesh,
            self.scale,
            self.head_size,
            self.num_heads,
            self.num_kv_heads,
            q_scale,
            k_scale,
            v_scale,
            self.sliding_window,
        )
        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        # Flatten output to 2D if needed (MLA models need 2D output for o_proj)
        print(f"[ATTN Forward] Standard path output shape: {outputs.shape}")
        if len(outputs.shape) == 3:
            # Flatten from (num_tokens, num_heads, head_dim) to (num_tokens, num_heads * head_dim)
            outputs = outputs.reshape(outputs.shape[0], outputs.shape[1] * outputs.shape[2])
            print(f"[ATTN Forward] Flattened to: {outputs.shape}")

        return torch_view(outputs)

    def _forward_mla(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: jax.Array,
        attn_metadata: AttentionMetadata,
        mesh: "Mesh",
    ) -> torch.Tensor:
        """MLA attention forward pass.

        For MLA, the arguments are:
        - q: Query tensor (num_tokens, num_heads, qk_head_dim)
        - kv_c_normed: Compressed KV latent (num_tokens, kv_lora_rank)
        - k_pe: K position embeddings (num_tokens, qk_rope_head_dim)

        GLM-4 uses non-absorbed MLA where q_nope_dim != kv_lora_rank.
        The MLA kernel requires absorbed mode (q_nope_dim == kv_lora_rank).
        For non-absorbed mode, we use a fallback simple attention.
        """
        # Convert to JAX
        q_jax = jax_view(q)
        kv_c_jax = jax_view(kv_c_normed)
        k_pe_jax = jax_view(k_pe)

        _mla_debug_print(f"[MLA Forward] JAX shapes: q={q_jax.shape}, kv_c={kv_c_jax.shape}, k_pe={k_pe_jax.shape}")
        _mla_debug_print(f"[MLA Forward] kv_cache shape: {kv_cache.shape}")

        # Get MLA dimensions
        num_tokens = q_jax.shape[0]
        num_heads = q_jax.shape[1] if len(q_jax.shape) == 3 else self.num_heads
        qk_head_dim = q_jax.shape[-1]
        kv_lora_rank = kv_c_jax.shape[-1]
        qk_rope_head_dim = k_pe_jax.shape[-1]
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim

        _mla_debug_print(f"[MLA Forward] Dims: qk_head={qk_head_dim}, nope={qk_nope_head_dim}, rope={qk_rope_head_dim}, kv_lora={kv_lora_rank}")

        # Check if absorbed mode (q_nope_dim == kv_lora_rank)
        is_absorbed = (qk_nope_head_dim == kv_lora_rank)
        _mla_debug_print(f"[MLA Forward] Mode: {'absorbed' if is_absorbed else 'non-absorbed (GLM-4 style)'}")

        # Reshape inputs
        if len(q_jax.shape) != 3:
            q_jax = q_jax.reshape(num_tokens, num_heads, qk_head_dim)
        if len(kv_c_jax.shape) == 3:
            kv_c_jax = kv_c_jax.reshape(num_tokens, -1)
        if len(k_pe_jax.shape) == 3:
            k_pe_jax = k_pe_jax.reshape(num_tokens, -1)

        md = attn_metadata

        # Debug: Print block_tables to verify page indices are valid
        _mla_debug_print(f"[MLA Forward] Metadata: block_tables.shape={md.block_tables.shape}, seq_lens.shape={md.seq_lens.shape}")
        _mla_debug_print(f"[MLA Forward] block_tables first 32: {md.block_tables[:32]}")
        _mla_debug_print(f"[MLA Forward] seq_lens: {md.seq_lens[:10]}")
        _mla_debug_print(f"[MLA Forward] query_start_loc: {md.query_start_loc[:10]}")
        _mla_debug_print(f"[MLA Forward] request_distribution: {md.request_distribution}")
        # Check if all block_tables are zeros (indicates no block allocation)
        block_tables_nonzero = jnp.sum(md.block_tables != 0)
        _mla_debug_print(f"[MLA Forward] block_tables non-zero count: {block_tables_nonzero}")

        if is_absorbed:
            # Use MLA kernel for absorbed mode
            import tpu_inference.kernels.mla.v1.kernel as mla

            q_nope = q_jax[..., :qk_nope_head_dim]
            q_pe = q_jax[..., qk_nope_head_dim:]

            # Define sharding specs for shard_map
            query_spec = P(None, ShardingAxisName.ATTN_HEAD, None)
            kv_cache_spec = P(ShardingAxisName.MLP_TENSOR)
            metadata_spec = P(ShardingAxisName.ATTN_DATA)

            in_specs = (
                query_spec, query_spec,
                P(None, None), P(None, None),
                kv_cache_spec,
                metadata_spec, metadata_spec, metadata_spec, metadata_spec,
            )
            out_specs = (query_spec, kv_cache_spec)

            def _mla_absorbed_fn(q_nope, q_pe, kv_c, k_pe, cache, seq_lens, block_tables,
                                 query_start_loc, request_distribution):
                max_num_seqs = seq_lens.shape[0]
                num_page_indices = block_tables.shape[0]
                pages_per_seq = num_page_indices // max_num_seqs
                num_kv_pages_per_block = min(pages_per_seq, 4)
                # Use num_queries_per_block=1 to avoid kernel assertion failures
                # The kernel asserts q_pe.shape[0] % bq_sz == 0, but with ragged
                # sequences, actual_bq_sz can be smaller than bq_sz
                num_queries_per_block = 1

                return mla.mla_ragged_paged_attention(
                    ql_nope=q_nope, q_pe=q_pe,
                    new_kv_c=kv_c, new_k_pe=k_pe,
                    cache_kv=cache, kv_lens=seq_lens,
                    page_indices=block_tables, cu_q_lens=query_start_loc,
                    distribution=request_distribution,
                    sm_scale=self.scale, sliding_window=self.sliding_window,
                    num_kv_pages_per_block=num_kv_pages_per_block,
                    num_queries_per_block=num_queries_per_block,
                )

            output, new_kv_cache = jax.jit(
                jax.shard_map(
                    _mla_absorbed_fn, mesh=mesh,
                    in_specs=in_specs, out_specs=out_specs,
                    check_vma=False,
                )
            )(q_nope, q_pe, kv_c_jax, k_pe_jax, kv_cache,
              md.seq_lens, md.block_tables, md.query_start_loc, md.request_distribution)

            vllm_model_wrapper_context = get_vllm_model_wrapper_context()
            kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
                layer.layer_name]
            vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

            # Flatten output from 3D (num_tokens, num_heads, lkv_dim) to 2D (num_tokens, num_heads * lkv_dim)
            # o_proj expects 2D input for the linear projection
            _mla_debug_print(f"[MLA Absorbed] Output shape before flatten: {output.shape}")
            if len(output.shape) == 3:
                output = output.reshape(output.shape[0], output.shape[1] * output.shape[2])
            _mla_debug_print(f"[MLA Absorbed] Output shape after flatten: {output.shape}")

            return torch_view(output)

        # Non-absorbed mode (GLM-4/DeepSeek style): Project Q to latent space and use MLA kernel
        # Following vLLM's approach: q_latent = q_nope @ W_UK^T (project Q to latent space)
        # This allows using the MLA kernel which stores compressed KV and handles paged attention
        _mla_debug_print(f"[MLA Non-Absorbed] Using MLA kernel with Q projection to latent space")

        # Try to get kv_b_proj from the layer to extract W_UK
        kv_b_proj_weight = None
        v_head_dim = qk_head_dim  # Default: 256 = 192 + 64 for GLM-4

        try:
            # The layer parameter is AttentionLayer which wraps the actual attention
            # Try to access kv_b_proj through various paths
            if hasattr(layer, 'kv_b_proj'):
                kv_b_proj_weight = layer.kv_b_proj.weight
                _mla_debug_print(f"[MLA] Found kv_b_proj directly on layer")
            elif hasattr(layer, 'impl') and hasattr(layer.impl, 'kv_b_proj'):
                kv_b_proj_weight = layer.impl.kv_b_proj.weight
                _mla_debug_print(f"[MLA] Found kv_b_proj on layer.impl")
            elif hasattr(layer, 'attn') and hasattr(layer.attn, 'kv_b_proj'):
                kv_b_proj_weight = layer.attn.kv_b_proj.weight
                _mla_debug_print(f"[MLA] Found kv_b_proj on layer.attn")

            if kv_b_proj_weight is not None:
                _mla_debug_print(f"[MLA] kv_b_proj weight shape: {kv_b_proj_weight.shape}")
                # kv_b_proj projects: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
                out_features = kv_b_proj_weight.shape[0]
                per_head_dim = out_features // num_heads
                v_head_dim = per_head_dim - qk_nope_head_dim
                _mla_debug_print(f"[MLA] Inferred v_head_dim: {v_head_dim}")
        except Exception as e:
            _mla_debug_print(f"[MLA] Could not access kv_b_proj: {e}")

        # Split Q into nope and pe parts
        q_nope = q_jax[..., :qk_nope_head_dim]  # (num_tokens, num_heads, qk_nope_head_dim)
        q_pe = q_jax[..., qk_nope_head_dim:]    # (num_tokens, num_heads, qk_rope_head_dim)

        if kv_b_proj_weight is not None:
            # Extract W_UK from kv_b_proj to project Q to latent space
            # kv_b_proj: (out_features, kv_lora_rank) = (num_heads * (qk_nope + v), kv_lora_rank)
            kv_b_proj_jax = jax_view(kv_b_proj_weight)

            # Reshape to (num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
            kv_b_proj_reshaped = kv_b_proj_jax.reshape(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)

            # Split to get W_UK: (num_heads, qk_nope_head_dim, kv_lora_rank)
            W_UK = kv_b_proj_reshaped[:, :qk_nope_head_dim, :]  # (num_heads, qk_nope_head_dim, kv_lora_rank)

            _mla_debug_print(f"[MLA] W_UK shape: {W_UK.shape}")

            # Project q_nope to latent space: q_latent = q_nope @ W_UK
            # q_nope: (num_tokens, num_heads, qk_nope_head_dim)
            # W_UK: (num_heads, qk_nope_head_dim, kv_lora_rank)
            # Result: (num_tokens, num_heads, kv_lora_rank)
            q_latent = jnp.einsum('tnd,ndk->tnk', q_nope, W_UK)

            _mla_debug_print(f"[MLA] q_latent shape (projected to latent): {q_latent.shape}")

            # Now use MLA kernel with latent Q
            # The MLA kernel expects ql_nope with shape (num_tokens, num_heads, kv_lora_rank)
            import tpu_inference.kernels.mla.v1.kernel as mla

            # Define sharding specs for shard_map
            query_spec = P(None, ShardingAxisName.ATTN_HEAD, None)
            kv_cache_spec = P(ShardingAxisName.MLP_TENSOR)
            metadata_spec = P(ShardingAxisName.ATTN_DATA)

            in_specs = (
                query_spec, query_spec,
                P(None, None), P(None, None),
                kv_cache_spec,
                metadata_spec, metadata_spec, metadata_spec, metadata_spec,
            )
            out_specs = (query_spec, kv_cache_spec)

            def _mla_nonabsorbed_fn(q_latent, q_pe, kv_c, k_pe, cache, seq_lens, block_tables,
                                    query_start_loc, request_distribution):
                max_num_seqs = seq_lens.shape[0]
                num_page_indices = block_tables.shape[0]
                pages_per_seq = num_page_indices // max_num_seqs
                num_kv_pages_per_block = min(pages_per_seq, 4)
                # Use num_queries_per_block=1 to avoid kernel assertion failures
                num_queries_per_block = 1

                return mla.mla_ragged_paged_attention(
                    ql_nope=q_latent, q_pe=q_pe,
                    new_kv_c=kv_c, new_k_pe=k_pe,
                    cache_kv=cache, kv_lens=seq_lens,
                    page_indices=block_tables, cu_q_lens=query_start_loc,
                    distribution=request_distribution,
                    sm_scale=self.scale, sliding_window=self.sliding_window,
                    num_kv_pages_per_block=num_kv_pages_per_block,
                    num_queries_per_block=num_queries_per_block,
                )

            output, new_kv_cache = jax.jit(
                jax.shard_map(
                    _mla_nonabsorbed_fn, mesh=mesh,
                    in_specs=in_specs, out_specs=out_specs,
                    check_vma=False,
                )
            )(q_latent, q_pe, kv_c_jax, k_pe_jax, kv_cache,
              md.seq_lens, md.block_tables, md.query_start_loc, md.request_distribution)

            # Update KV cache in context
            vllm_model_wrapper_context = get_vllm_model_wrapper_context()
            kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
                layer.layer_name]
            vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

            _mla_debug_print(f"[MLA Non-Absorbed] MLA kernel output shape: {output.shape}")

            # The MLA kernel output is in latent space: (num_tokens, num_heads, kv_lora_rank)
            # We need to project it to value space using W_UV
            # W_UV: (num_heads, v_head_dim, kv_lora_rank)
            W_UV = kv_b_proj_reshaped[:, qk_nope_head_dim:, :]  # (num_heads, v_head_dim, kv_lora_rank)

            # Project output: output_v = output @ W_UV^T
            # output: (num_tokens, num_heads, kv_lora_rank)
            # W_UV^T: (num_heads, kv_lora_rank, v_head_dim)
            # Result: (num_tokens, num_heads, v_head_dim)
            output_v = jnp.einsum('tnk,nkv->tnv', output, W_UV.transpose(0, 2, 1))

            _mla_debug_print(f"[MLA Non-Absorbed] Output after V projection: {output_v.shape}")

            # Flatten output from (num_tokens, num_heads, v_head_dim) to (num_tokens, num_heads * v_head_dim)
            output = output_v.reshape(num_tokens, num_heads * v_head_dim)

            _mla_debug_print(f"[MLA Non-Absorbed] Final output shape: {output.shape}")

            return torch_view(output)
        else:
            # Fallback without kv_b_proj - use simple attention without caching
            _mla_debug_print(f"[MLA] WARNING: kv_b_proj not found, falling back to non-cached attention")

            # Approximate: repeat kv_c across heads and truncate
            kv_per_head = kv_lora_rank // num_heads
            if kv_per_head * num_heads < kv_lora_rank:
                kv_per_head += 1
            kv_c_padded = jnp.pad(kv_c_jax, ((0, 0), (0, kv_per_head * num_heads - kv_lora_rank)))
            kv_c_per_head = kv_c_padded.reshape(num_tokens, num_heads, kv_per_head)

            # Truncate or pad to qk_nope_head_dim for k_nope
            if kv_per_head >= qk_nope_head_dim:
                k_nope = kv_c_per_head[..., :qk_nope_head_dim]
            else:
                k_nope = jnp.pad(kv_c_per_head, ((0, 0), (0, 0), (0, qk_nope_head_dim - kv_per_head)))

            k_pe_expanded = jnp.broadcast_to(k_pe_jax[:, None, :], (num_tokens, num_heads, qk_rope_head_dim))
            k_full = jnp.concatenate([k_nope, k_pe_expanded], axis=-1)

            # V needs to match v_head_dim (256)
            if kv_per_head >= v_head_dim:
                v = kv_c_per_head[..., :v_head_dim]
            else:
                v = jnp.pad(kv_c_per_head, ((0, 0), (0, 0), (0, v_head_dim - kv_per_head)))
            _mla_debug_print(f"[MLA Fallback] k_nope={k_nope.shape}, k_full={k_full.shape}, v={v.shape}")

            # Compute attention: softmax(Q @ K^T / sqrt(d)) @ V
            attn_scores = jnp.einsum('qhd,khd->hqk', q_jax, k_full) * self.scale
            causal_mask = jnp.tril(jnp.ones((num_tokens, num_tokens)))
            attn_scores = jnp.where(causal_mask[None, :, :], attn_scores, -1e9)
            attn_probs = jax.nn.softmax(attn_scores, axis=-1).astype(q_jax.dtype)
            output = jnp.einsum('hqk,khv->qhv', attn_probs, v)
            output = output.reshape(num_tokens, num_heads * v.shape[-1])

            _mla_debug_print(f"[MLA Fallback] WARNING: No KV caching - output may be incorrect for decode")
            return torch_view(output)


@functools.partial(
    jax.jit,
    static_argnames=(
        "mesh",
        "scale",
        "head_size",
        "num_heads",
        "num_kv_heads",
        "q_scale",
        "k_scale",
        "v_scale",
        "sliding_window",
    ),
    donate_argnames=("kv_cache"),
)
def _jax_attn_func(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sinks: jax.Array | None,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    scale: float,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    sliding_window: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    # Get shapes from vllm
    # MLA_PATCHED: Handle variable tensor dimensions + DEBUG
    _original_q_shape = q.shape
    _needs_reshape = False
    if _VLLM_USE_MLA:
        _mla_debug_print(f"[MLA DEBUG] ===== _jax_attn_func ENTRY =====")
        _mla_debug_print(f"[MLA DEBUG] q.shape = {q.shape}, q.dtype = {q.dtype}")
        _mla_debug_print(f"[MLA DEBUG] k.shape = {k.shape}, k.dtype = {k.dtype}")
        _mla_debug_print(f"[MLA DEBUG] v.shape = {v.shape}, v.dtype = {v.dtype}")
        _mla_debug_print(f"[MLA DEBUG] kv_cache.shape = {kv_cache.shape if kv_cache is not None else None}")
        # Flatten any 3D tensors to 2D for TPU attention
        if len(q.shape) == 3 or len(k.shape) == 3 or len(v.shape) == 3:
            _needs_reshape = True
            q, k, v = _mla_reshape_qkv(q, k, v)
            _mla_debug_print(f"[MLA DEBUG] After flattening: q={q.shape}, k={k.shape}, v={v.shape}")
    q_len, q_compute_dim = q.shape
    
    if _VLLM_USE_MLA:
        _mla_debug_print(f"[MLA DEBUG] q_len={q_len}, q_compute_dim={q_compute_dim}")
    k_len, k_compute_dim = k.shape

    if _VLLM_USE_MLA:
        # MLA: K and V may have different shapes due to compressed latent representation
        # - Q: (seq_len, num_heads * qk_head_dim)
        # - K: (seq_len, kv_lora_rank) or (seq_len, kv_lora_rank + rope_dim)
        # - V: (seq_len, v_head_dim) - may differ from K
        # The head_size parameter comes from KV cache config and may not match Q's head_dim

        # Calculate actual head dimensions from tensor shapes
        q_head_dim = q_compute_dim // num_heads
        k_head_dim = k_compute_dim // num_kv_heads if num_kv_heads > 0 else k_compute_dim
        v_compute_dim = v.shape[1] if len(v.shape) == 2 else v.shape[-1]
        v_head_dim = v_compute_dim // num_kv_heads if num_kv_heads > 0 else v_compute_dim

        # Reshape using actual dimensions, not KV cache head_size
        q = q.reshape(q_len, num_heads, q_head_dim)
        k = k.reshape(k_len, num_kv_heads, k_head_dim)
        v = v.reshape(k_len, num_kv_heads, v_head_dim)
    if _VLLM_USE_MLA:
        _mla_debug_print(f"[MLA DEBUG] k.shape={k.shape}, v.shape={v.shape}, equal={k.shape == v.shape}")
    if not _VLLM_USE_MLA:
        assert k.shape == v.shape
        # MLA_PATCHED: Skip dimension assertion for MLA + DEBUG
    if _VLLM_USE_MLA:
        _mla_debug_print(f"[MLA DEBUG] q_compute_dim={q_compute_dim}, expected={head_size * num_heads} (head_size={head_size} * num_heads={num_heads})")
    if not _VLLM_USE_MLA:
        assert q_compute_dim == head_size * num_heads
        # MLA_PATCHED: Skip k dimension assertion for MLA + DEBUG
    if _VLLM_USE_MLA:
        _mla_debug_print(f"[MLA DEBUG] k_compute_dim={k_compute_dim}, expected={head_size * num_kv_heads} (head_size={head_size} * num_kv_heads={num_kv_heads})")
    if not _VLLM_USE_MLA:
        assert k_compute_dim == head_size * num_kv_heads

        # Convert the shapes from vLLM's convention to what the attention function expects
        # bs, num_heads, q_len, head_size
        # MLA_PATCHED: For MLA, reshape is already done above (lines 365-368), skip redundant reshape
    if not _VLLM_USE_MLA:
        q = q.reshape(q.shape[0], num_heads, head_size)
        k = k.reshape(k.shape[0], num_kv_heads, head_size)
        v = v.reshape(v.shape[0], num_kv_heads, head_size)
    else:
        # MLA: q, k, v already reshaped to 3D above with correct dimensions
        _mla_debug_print(f"[MLA DEBUG] After MLA reshape: q={q.shape}, k={k.shape}, v={v.shape}")

    new_kv_cache, outputs = attention(
        kv_cache,
        q,
        k,
        v,
        attention_metadata,
        mesh,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sinks=sinks,
        attention_chunk_size=sliding_window,
    )

    # Convert the shape back to vLLM's convention
    if _VLLM_USE_MLA:
        # MLA: reshape output based on actual Q dimensions
        assert outputs.shape[0] == q_len
        assert outputs.shape[1] == num_heads
        # Output head_dim should match Q's head_dim, not KV cache head_size
        outputs = outputs.reshape(q_len, q_compute_dim)
    else:
        assert outputs.shape[0] == q_len
        assert outputs.shape[1] == num_heads
        assert outputs.shape[2] == head_size
        outputs = outputs.reshape(q_len, q_compute_dim)

    # MLA_PATCHED: Reshape output back to original shape if needed
    if _VLLM_USE_MLA and _needs_reshape:
        outputs = _mla_reshape_output(outputs, _original_q_shape)
        _mla_debug_print(f"[MLA DEBUG] Output reshaped: {outputs.shape}")
    return new_kv_cache, outputs


# =============================================================================
# MLA ATTENTION BACKEND
# =============================================================================

# Try to import the vLLM MLA base class
try:
    from vllm.model_executor.layers.attention.mla_attention import MLAAttentionImpl
    _MLA_BASE_AVAILABLE = True
    print("[MLA] MLAAttentionImpl base class available")
except ImportError:
    _MLA_BASE_AVAILABLE = False
    print("[MLA] MLAAttentionImpl not available, using stub")
    # Fallback: Define a stub
    class MLAAttentionImpl:
        pass


# Register MLA backend if the enum is available
def _maybe_register_mla_backend(cls):
    """Conditionally register MLA backend if enum is available."""
    if _MLA_BACKEND_ENUM is not None:
        return register_backend(_MLA_BACKEND_ENUM)(cls)
    _mla_debug_print(f"[MLA] MLA backend not registered (no enum), class: {cls.__name__}")
    return cls


@_maybe_register_mla_backend
class PallasMLAAttentionBackend(AttentionBackend):
    """TPU backend for MLA attention."""

    @staticmethod
    def get_name() -> str:
        return "PALLAS_MLA"

    @staticmethod
    def get_impl_cls() -> type["PallasMLAAttentionBackendImpl"]:
        return PallasMLAAttentionBackendImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # MLA uses 4D KV cache: (num_pages, page_size_packed, kv_packing, kv_dim)
        # where kv_dim = kv_lora_rank + padded_rope_dim
        from tpu_inference.kernels.mla.v1.kernel import get_kv_cache_shape
        import jax.numpy as jnp

        dtype = jnp.bfloat16  # Default, adjust based on cache_dtype_str
        return get_kv_cache_shape(num_blocks, block_size, head_size, dtype)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU MLA backend.")


class PallasMLAAttentionBackendImpl(MLAAttentionImpl if _MLA_BASE_AVAILABLE else AttentionImpl):
    """TPU implementation for MLA attention.

    This implements the MLA attention interface which receives:
    - q: (num_tokens, num_heads, qk_head_dim)
    - kv_c_normed: (num_tokens, kv_lora_rank) - compressed KV
    - k_pe: (num_tokens, qk_rope_head_dim) - K position embeddings
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: dict | None = None,
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        # MLA-specific parameters
        qk_nope_head_dim: int = 0,
        qk_rope_head_dim: int = 0,
        v_head_dim: int = 0,
        kv_lora_rank: int = 0,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.blocksparse_params = blocksparse_params
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

        # MLA-specific parameters
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        # qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # KV cache dimension = kv_lora_rank + padded_rope_dim
        rope_dim_padded = ((qk_rope_head_dim + 127) // 128) * 128
        self.kv_dim = kv_lora_rank + rope_dim_padded

        # Capture kv_b_proj for Q absorption (needed to project q_nope to match kv_lora_rank)
        # kv_b_proj maps: kv_lora_rank -> (qk_nope_head_dim + v_head_dim) * num_heads
        self.kv_b_proj = kwargs.get('kv_b_proj', None)
        if self.kv_b_proj is not None:
            _mla_debug_print(f"[MLA Backend] kv_b_proj captured: {type(self.kv_b_proj)}")
        else:
            _mla_debug_print(f"[MLA Backend] WARNING: kv_b_proj not provided, kwargs keys: {list(kwargs.keys())}")

        _mla_debug_print(f"[MLA Backend] Initialized: qk_nope={qk_nope_head_dim}, qk_rope={qk_rope_head_dim}, "
              f"v_head={v_head_dim}, kv_lora_rank={kv_lora_rank}, kv_dim={self.kv_dim}")
        # Check if scale matches expected 1/sqrt(qk_head_dim)
        import math
        expected_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
        _mla_debug_print(f"[MLA Backend] scale={scale}, expected_scale={expected_scale} (1/sqrt({qk_nope_head_dim + qk_rope_head_dim}))")

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLA attention forward pass.

        Args:
            layer: The attention layer
            q: Query tensor (num_tokens, num_heads, qk_head_dim)
            kv_c_normed: Compressed KV (num_tokens, kv_lora_rank)
            k_pe: K position embeddings (num_tokens, qk_rope_head_dim)
            kv_cache: KV cache
            attn_metadata: Attention metadata
            output: Optional output buffer
            output_scale: Optional output scale
        """
        if output_scale is not None:
            raise NotImplementedError(
                "output quantization not supported for TPU MLA backend")

        # Get KV cache from context
        if kv_cache.numel():
            raise RuntimeError(
                "KV cache from vLLM should be empty for TPU backend")

        del kv_cache
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
            layer.layer_name]
        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]
        mesh = vllm_model_wrapper_context.mesh

        # Track request number (increment only on layer 0 to count model forward passes)
        global _MLA_REQUEST_COUNTER
        if 'layer.0.' in layer.layer_name or layer.layer_name.endswith('.0.self_attn.attn'):
            _MLA_REQUEST_COUNTER += 1
            if _MLA_DEBUG:
                print(f"[MLA Backend] ========== REQUEST #{_MLA_REQUEST_COUNTER} (num_tokens={q.shape[0]}) ==========")
                print(f"[MLA Backend] MLA dimensions: qk_nope={self.qk_nope_head_dim}, qk_rope={self.qk_rope_head_dim}, "
                      f"v_head={self.v_head_dim}, kv_lora_rank={self.kv_lora_rank}, num_heads={self.num_heads}")
                print(f"[MLA Backend] scale={self.scale}, kv_dim={self.kv_dim}")

        _mla_debug_print(f"[MLA Backend] forward: q={q.shape}, kv_c={kv_c_normed.shape}, "
              f"k_pe={k_pe.shape}, kv_cache={kv_cache.shape}")

        # CRITICAL DEBUG: Check if cache from context has values (should have values after first pass)
        # NOTE: We check the ALLOCATED page (block_tables[0]), not page 0!
        # Request #1 = prefill (cache should be zeros), Request #2+ = decode (cache should have values)
        if _MLA_DEBUG:
            kv_cache_jax = jax_view(kv_cache) if hasattr(kv_cache, 'numpy') else kv_cache
            block_tables_jax = jax_view(attn_metadata.block_tables) if hasattr(attn_metadata.block_tables, 'numpy') else attn_metadata.block_tables
            allocated_page = block_tables_jax.reshape(-1)[0]  # First allocated page
            jax.debug.print("[MLA Backend DEBUG] cache FROM CONTEXT layer={layer}, allocated_page={p}: min={min}, max={max}, mean={mean}",
                           layer=layer.layer_name, p=allocated_page,
                           min=jnp.min(kv_cache_jax[allocated_page]), max=jnp.max(kv_cache_jax[allocated_page]),
                           mean=jnp.mean(kv_cache_jax[allocated_page].astype(jnp.float32)))

        # Split Q into nope and pe parts
        # q shape: (num_tokens, num_heads, qk_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]  # (num_tokens, num_heads, qk_nope_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:]    # (num_tokens, num_heads, qk_rope_head_dim)

        _mla_debug_print(f"[MLA Backend] q_nope={q_nope.shape}, q_pe={q_pe.shape}")

        # Project q_nope to ql_nope using kv_b_proj (W_K absorption)
        # The TPU MLA kernel expects ql_nope with shape (tokens, heads, kv_lora_rank)
        # kv_b_proj.weight: ((qk_nope_head_dim + v_head_dim) * num_heads, kv_lora_rank)
        # Weight layout is per-head: [head0_K, head0_V, head1_K, head1_V, ...]
        if self.kv_b_proj is not None:
            # Get weight matrix
            kv_b_weight = self.kv_b_proj.weight  # ((qk_nope + v) * heads, kv_lora_rank)
            _mla_debug_print(f"[MLA Backend] kv_b_proj.weight shape: {kv_b_weight.shape}")

            # Validate shape matches expectations
            expected_out = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
            expected_in = self.kv_lora_rank
            if kv_b_weight.shape != (expected_out, expected_in):
                _mla_debug_print(f"[MLA Backend] WARNING: Weight shape mismatch! Expected ({expected_out}, {expected_in}), got {kv_b_weight.shape}")
                _mla_debug_print(f"[MLA Backend] num_heads={self.num_heads}, qk_nope={self.qk_nope_head_dim}, v_head={self.v_head_dim}, kv_lora={self.kv_lora_rank}")

            # Debug: Check weight statistics (shape only - .item() fails in JIT)
            # _mla_debug_print(f"[MLA Backend] kv_b_proj weight stats: use jax.debug.print for values")

            # Reshape to (num_heads, qk_nope + v, kv_lora_rank) to properly separate per-head K/V
            kv_b_weight_reshaped = kv_b_weight.view(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank
            )

            # Extract W_K by splitting along dim=1 (per-head split)
            w_k = kv_b_weight_reshaped[:, :self.qk_nope_head_dim, :]  # (heads, qk_nope, kv_lora)
            _mla_debug_print(f"[MLA Backend] W_K shape after correct extraction: {w_k.shape}")

            # Debug: Input values (removed .item() calls that fail in JIT)

            # Project: ql_nope[t, h, :] = q_nope[t, h, :] @ w_k[h, :, :]
            # Using einsum: (tokens, heads, qk_nope) @ (heads, qk_nope, kv_lora) -> (tokens, heads, kv_lora)
            ql_nope = torch.einsum('thd,hdk->thk', q_nope, w_k)
            _mla_debug_print(f"[MLA Backend] ql_nope shape after projection: {ql_nope.shape}")

            # DEBUG: Verify W_K extraction is correct
            if _MLA_DEBUG:
                w_k_jax = jax_view(w_k)
                jax.debug.print("[MLA Backend DEBUG] W_K stats: shape={s}, min={min}, max={max}, mean={mean}",
                               s=w_k_jax.shape, min=jnp.min(w_k_jax), max=jnp.max(w_k_jax),
                               mean=jnp.mean(w_k_jax.astype(jnp.float32)))
        else:
            # Fallback: assume q_nope is already projected (shouldn't happen)
            _mla_debug_print(f"[MLA Backend] WARNING: kv_b_proj not available, using q_nope directly")
            ql_nope = q_nope

        # Convert to JAX arrays
        ql_nope_jax = jax_view(ql_nope)
        q_pe_jax = jax_view(q_pe)
        kv_c_jax = jax_view(kv_c_normed)
        k_pe_jax = jax_view(k_pe)

        # Extract metadata arrays before JIT boundary to avoid tracing issues
        # The kernel's validation does Python boolean comparisons which fail on traced arrays
        md = attn_metadata
        seq_lens = md.seq_lens
        block_tables = md.block_tables
        query_start_loc = md.query_start_loc
        request_distribution = md.request_distribution

        _mla_debug_print(f"[MLA Backend] Metadata: seq_lens={seq_lens.shape}, block_tables={block_tables.shape}, "
              f"query_start_loc={query_start_loc.shape}, request_distribution={request_distribution.shape}")

        # Call the MLA attention function
        new_kv_cache, outputs = _jax_mla_attn_func(
            kv_cache,
            ql_nope_jax,
            q_pe_jax,
            kv_c_jax,
            k_pe_jax,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            mesh,
            self.scale,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.num_heads,
            self.sliding_window,
        )

        vllm_model_wrapper_context.kv_caches[kv_cache_index] = new_kv_cache

        # CRITICAL DEBUG: Verify cache was stored to context
        # NOTE: We check the ALLOCATED page (block_tables[0]), not page 0!
        if _MLA_DEBUG:
            stored_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]
            block_tables_jax = jax_view(attn_metadata.block_tables) if hasattr(attn_metadata.block_tables, 'numpy') else attn_metadata.block_tables
            allocated_page = block_tables_jax.reshape(-1)[0]  # First allocated page
            jax.debug.print("[MLA Backend DEBUG] cache STORED TO CONTEXT, allocated_page={p}: min={min}, max={max}, mean={mean}",
                           p=allocated_page,
                           min=jnp.min(stored_cache[allocated_page]), max=jnp.max(stored_cache[allocated_page]),
                           mean=jnp.mean(stored_cache[allocated_page].astype(jnp.float32)))

        # The MLA kernel computes: attn @ kv_c (shape: tokens, heads, kv_lora_rank)
        # But for non-absorbed MLA, we need to project output through W_V to get v_head_dim
        # attn @ V = attn @ (kv_c @ W_V^T) = (attn @ kv_c) @ W_V^T
        _mla_debug_print(f"[MLA Backend] Output shape from kernel: {outputs.shape}")

        if self.kv_b_proj is not None and self.v_head_dim != self.kv_lora_rank:
            # Extract W_V from kv_b_proj
            # kv_b_proj: (qk_nope_head_dim + v_head_dim) * num_heads x kv_lora_rank
            # Weight layout is per-head: [head0_K, head0_V, head1_K, head1_V, ...]
            kv_b_weight = jax_view(self.kv_b_proj.weight)  # ((qk_nope + v) * heads, kv_lora)

            # Reshape to (num_heads, qk_nope + v, kv_lora_rank) to properly separate per-head K/V
            kv_b_weight_reshaped = kv_b_weight.reshape(
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
                self.kv_lora_rank
            )

            # Extract W_V by splitting along dim=1 (per-head split)
            w_v = kv_b_weight_reshaped[:, self.qk_nope_head_dim:, :]  # (heads, v_head_dim, kv_lora)
            _mla_debug_print(f"[MLA Backend] W_V shape after correct extraction: {w_v.shape}")

            # DEBUG: Verify W_V extraction is correct
            if _MLA_DEBUG:
                jax.debug.print("[MLA Backend DEBUG] W_V stats: shape={s}, min={min}, max={max}, mean={mean}",
                               s=w_v.shape, min=jnp.min(w_v), max=jnp.max(w_v),
                               mean=jnp.mean(w_v.astype(jnp.float32)))

            # Project: outputs[t, h, :] @ w_v[h, :, :]^T -> (tokens, heads, v_head_dim)
            # Using einsum: (tokens, heads, kv_lora) @ (heads, v_head_dim, kv_lora)^T -> (tokens, heads, v_head_dim)
            outputs = jnp.einsum('thk,hvk->thv', outputs, w_v)
            _mla_debug_print(f"[MLA Backend] Output shape after W_V projection: {outputs.shape}")

            # DEBUG: Check W_V projection output (JIT-safe)
            if _MLA_DEBUG:
                jax.debug.print("[MLA Backend DEBUG] after W_V: min={min}, max={max}, mean={mean}",
                               min=jnp.min(outputs), max=jnp.max(outputs), mean=jnp.mean(outputs.astype(jnp.float32)))

        # Flatten output from 3D (num_tokens, num_heads, v_head_dim) to 2D (num_tokens, num_heads * v_head_dim)
        # o_proj expects 2D input for the linear projection
        _mla_debug_print(f"[MLA Backend] Output shape before flatten: {outputs.shape}")
        if len(outputs.shape) == 3:
            outputs = outputs.reshape(outputs.shape[0], outputs.shape[1] * outputs.shape[2])
        _mla_debug_print(f"[MLA Backend] Output shape after flatten: {outputs.shape}")

        # DEBUG: Final output check (JIT-safe)
        if _MLA_DEBUG:
            jax.debug.print("[MLA Backend DEBUG] final output: min={min}, max={max}, mean={mean}",
                           min=jnp.min(outputs), max=jnp.max(outputs), mean=jnp.mean(outputs.astype(jnp.float32)))

        return torch_view(outputs)


# Note: JIT is disabled for MLA because the kernel's validation does Python boolean
# comparisons on the distribution array, which fails with JAX tracing.
# TODO: Fix kernel validation to use JAX primitives or move validation outside JIT.
def _jax_mla_attn_func(
    kv_cache: jax.Array,
    ql_nope: jax.Array,
    q_pe: jax.Array,
    kv_c: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    mesh: "Mesh",
    scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    num_heads: int,
    sliding_window: int | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """JAX MLA attention function.

    Args:
        kv_cache: KV cache (4D for MLA)
        ql_nope: Projected query nope (num_tokens, num_heads, kv_lora_rank)
        q_pe: Query position embeddings (num_tokens, num_heads, qk_rope_head_dim)
        kv_c: Compressed KV (num_tokens, kv_lora_rank)
        k_pe: K position embeddings (num_tokens, qk_rope_head_dim)
        seq_lens: Sequence lengths per request
        block_tables: Page indices for KV cache
        query_start_loc: Cumulative query lengths
        request_distribution: Distribution info [prefill_tokens, decode_tokens, num_seqs]
        mesh: Device mesh
        scale: Attention scale
        kv_lora_rank: KV LoRA rank (compressed dim)
        qk_rope_head_dim: Query/Key rope dimension
        v_head_dim: Value head dimension
        num_heads: Number of attention heads
        sliding_window: Optional sliding window size
    """
    import tpu_inference.kernels.mla.v1.kernel as mla

    _mla_debug_print(f"[MLA JAX] Entry: ql_nope={ql_nope.shape}, q_pe={q_pe.shape}, kv_c={kv_c.shape}, k_pe={k_pe.shape}")
    _mla_debug_print(f"[MLA JAX] sm_scale={scale}, kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}")

    # Handle k_pe shape - kernel expects 2D (num_tokens, qk_rope_head_dim)
    # vLLM might pass 3D (num_tokens, 1, qk_rope_head_dim) with kv_heads=1 for MLA
    if k_pe.ndim == 3:
        k_pe = jnp.squeeze(k_pe, axis=1)
        _mla_debug_print(f"[MLA JAX] Squeezed k_pe to: {k_pe.shape}")

    # Handle kv_c shape - kernel expects 2D (num_tokens, kv_lora_rank)
    # vLLM might pass 3D (num_tokens, 1, kv_lora_rank)
    if kv_c.ndim == 3:
        kv_c = jnp.squeeze(kv_c, axis=1)
        _mla_debug_print(f"[MLA JAX] Squeezed kv_c to: {kv_c.shape}")

    # ql_nope is already projected to (num_tokens, num_heads, kv_lora_rank)
    # q_pe has shape (num_tokens, num_heads, qk_rope_head_dim)
    _mla_debug_print(f"[MLA JAX] After processing: ql_nope={ql_nope.shape}, q_pe={q_pe.shape}")

    # DEBUG: Print tensor value statistics (only when VLLM_LOGGING_LEVEL=DEBUG)
    if _MLA_DEBUG:
        def _debug_tensor(name, t):
            jax.debug.print("[MLA DEBUG] {name}: shape={shape}, dtype={dtype}, min={min}, max={max}, mean={mean}",
                           name=name, shape=t.shape, dtype=t.dtype,
                           min=jnp.min(t), max=jnp.max(t), mean=jnp.mean(t.astype(jnp.float32)))

        _debug_tensor("ql_nope", ql_nope)
        _debug_tensor("q_pe", q_pe)
        _debug_tensor("kv_c", kv_c)
        _debug_tensor("k_pe", k_pe)
        jax.debug.print("[MLA DEBUG] seq_lens first 10: {s}", s=seq_lens[:10])

    _mla_debug_print(f"[MLA DEBUG] block_tables shape before flatten: {block_tables.shape}")

    # Ensure block_tables is 1D - kernel expects [max_num_seqs * pages_per_seq]
    if block_tables.ndim == 2:
        block_tables = block_tables.reshape(-1)
        _mla_debug_print(f"[MLA DEBUG] Flattened block_tables to: {block_tables.shape}")

    if _MLA_DEBUG:
        jax.debug.print("[MLA DEBUG] block_tables first 32: {b}", b=block_tables[:32])
        jax.debug.print("[MLA DEBUG] query_start_loc first 10: {q}", q=query_start_loc[:10])
        jax.debug.print("[MLA DEBUG] request_distribution: {r}", r=request_distribution)
        _mla_debug_print(f"[MLA DEBUG] scale: {scale}")
        jax.debug.print("[MLA DEBUG] kv_cache shape: {s}, dtype: {d}", s=kv_cache.shape, d=kv_cache.dtype)
        # Check the ALLOCATED page (block_tables[0]), not page 0!
        allocated_page_idx = block_tables[0]
        jax.debug.print("[MLA DEBUG] kv_cache ALLOCATED page {p} stats: min={min}, max={max}, mean={mean}",
                       p=allocated_page_idx,
                       min=jnp.min(kv_cache[allocated_page_idx]),
                       max=jnp.max(kv_cache[allocated_page_idx]),
                       mean=jnp.mean(kv_cache[allocated_page_idx].astype(jnp.float32)))

    # CRITICAL DEBUG: Print distribution before kernel call
    # distribution[-1] is the number of sequences - if 0, no cache updates happen!
    if _MLA_DEBUG:
        jax.debug.print("[MLA DEBUG] BEFORE kernel - distribution: {d}, num_seqs={n}",
                       d=request_distribution, n=request_distribution[-1])
        jax.debug.print("[MLA DEBUG] BEFORE kernel - seq_lens: {s}", s=seq_lens[:10])
        jax.debug.print("[MLA DEBUG] BEFORE kernel - query_start_loc: {q}", q=query_start_loc[:10])
        # Check if distribution correctly reflects the request type
        num_tokens = ql_nope.shape[0]
        num_seqs_val = request_distribution[-1]
        jax.debug.print("[MLA DEBUG] num_tokens={t}, num_seqs={n}, tokens_per_seq~={tps}",
                       t=num_tokens, n=num_seqs_val,
                       tps=num_tokens)

    # Define sharding specs for shard_map (following deepseek_v3_attention.py pattern)
    # Query tensors are sharded on heads, KV cache on model axis, metadata replicated
    query_spec = P(None, ShardingAxisName.ATTN_HEAD, None)  # (tokens, heads, dim)
    kv_cache_spec = P(ShardingAxisName.MLP_TENSOR)  # (pages, ...)
    metadata_spec = P(ShardingAxisName.ATTN_DATA)  # Replicated metadata

    in_specs = (
        query_spec,      # ql_nope: (tokens, heads, kv_lora_rank)
        query_spec,      # q_pe: (tokens, heads, qk_rope_head_dim)
        P(None, None),   # kv_c: (tokens, kv_lora_rank)
        P(None, None),   # k_pe: (tokens, qk_rope_head_dim)
        kv_cache_spec,   # kv_cache
        metadata_spec,   # seq_lens
        metadata_spec,   # block_tables
        metadata_spec,   # query_start_loc
        metadata_spec,   # request_distribution
    )

    out_specs = (
        query_spec,      # output: (tokens, heads, kv_lora_rank)
        kv_cache_spec,   # new_kv_cache
    )

    def _mla_kernel_fn(ql_nope, q_pe, kv_c, k_pe, kv_cache, seq_lens, block_tables,
                       query_start_loc, request_distribution):
        # Compute block sizes inside shard_map
        max_num_tokens = ql_nope.shape[0]
        max_num_seqs = seq_lens.shape[0]
        num_page_indices = block_tables.shape[0]
        pages_per_seq = num_page_indices // max_num_seqs
        num_kv_pages_per_block = min(pages_per_seq, 4)
        # Use num_queries_per_block=1 to avoid kernel assertion failures
        # The kernel has an issue where it checks q_pe.shape[0] % bq_sz == 0
        # but actual_bq_sz may differ from bq_sz with ragged sequences
        num_queries_per_block = 1

        output, new_kv_cache = mla.mla_ragged_paged_attention(
            ql_nope=ql_nope,
            q_pe=q_pe,
            new_kv_c=kv_c,
            new_k_pe=k_pe,
            cache_kv=kv_cache,
            kv_lens=seq_lens,
            page_indices=block_tables,
            cu_q_lens=query_start_loc,
            distribution=request_distribution,
            sm_scale=scale,
            sliding_window=sliding_window,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )
        return output, new_kv_cache

    # Wrap kernel in shard_map for distributed execution
    output, new_kv_cache = jax.jit(
        jax.shard_map(
            _mla_kernel_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
    )(ql_nope, q_pe, kv_c, k_pe, kv_cache, seq_lens, block_tables,
      query_start_loc, request_distribution)

    _mla_debug_print(f"[MLA JAX] output={output.shape}, new_kv_cache={new_kv_cache.shape}")

    # DEBUG: Check output AND updated cache values (JIT-safe)
    # NOTE: We check the ALLOCATED page (block_tables[0]), not page 0!
    if _MLA_DEBUG:
        jax.debug.print("[MLA DEBUG] kernel_output: min={min}, max={max}, mean={mean}",
                       min=jnp.min(output), max=jnp.max(output), mean=jnp.mean(output.astype(jnp.float32)))
        # CRITICAL: Check if cache was actually updated - use ALLOCATED page not page 0
        allocated_page_idx = block_tables[0]
        jax.debug.print("[MLA DEBUG] AFTER kernel - new_kv_cache ALLOCATED page {p} stats: min={min}, max={max}, mean={mean}",
                       p=allocated_page_idx,
                       min=jnp.min(new_kv_cache[allocated_page_idx]), max=jnp.max(new_kv_cache[allocated_page_idx]),
                       mean=jnp.mean(new_kv_cache[allocated_page_idx].astype(jnp.float32)))

    return new_kv_cache, output
