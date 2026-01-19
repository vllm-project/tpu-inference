# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from typing import Optional, Tuple, final, override

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (
    BlockSizes, SegmentIds, make_splash_mha_single_device)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    FullMask, MultiHeadMask)
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer, AttentionType)
from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                 register_backend)

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
class AttentionBackend(AttentionBackend):

    @override
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @override
    @staticmethod
    def get_impl_cls() -> type["AttentionDispatcher"]:
        return AttentionDispatcher

    @override
    @staticmethod
    def get_builder_cls():
        # Just a dummy class to make EncoderOnlyAttention happy.
        # Our attention metadata building flow is entirely different from
        # the original design from vLLM code base.
        return object

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


@final
class AttentionDispatcher(AttentionImpl[AttentionMetadata]):
    """AttentionDispatcher dispatch attention to the implementation."""

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
        **extra_kwargs,
    ) -> None:
        attn_class: type[AttentionImpl[AttentionMetadata]]
        match attn_type:
            case AttentionType.DECODER:
                attn_class = PallasAttentionBackendImpl
            case AttentionType.ENCODER_ONLY:
                attn_class = TokamaxAttentionBackendImpl
            case _:
                raise NotImplementedError(
                    f"Attention {attn_type} is not supported.")

        self._attn = attn_class(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_kwargs,
        )

    @override
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._attn.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # process_weights_after_loading is not part of AttentionImpl[T]
        # but vLLM does detect it's existence then invoke it.
        if hasattr(self._attn, "process_weights_after_loading"):
            self._attn.process_weights_after_loading(act_dtype)


@final
class TokamaxAttentionBackendImpl(AttentionImpl[AttentionMetadata]):
    """TokamaxAttentionBackendImpl is a bridge to tokamax."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **extra_kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")

        if attn_type != AttentionType.ENCODER_ONLY:
            raise NotImplementedError(
                f"Attention type {attn_type} is not supported.")

    @override
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:

        query, key, value = jax_view(query), jax_view(key), jax_view(value)
        out = _jax_encoder_attn_func(
            query,
            key,
            value,
            attn_metadata,
            head_size=self.head_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
        )
        return torch_view(out)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass


def _ceiling_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_to(x: int, a: int) -> int:
    return _ceiling_div(x, a) * a


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_size",
        "num_heads",
        "num_kv_heads",
    ),
)
def _jax_encoder_attn_func(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attn_metadata: AttentionMetadata,
    *,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    scale: int,
) -> jax.Array:
    # Get shapes from input tensors
    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_size * num_heads
    assert k_compute_dim == head_size * num_kv_heads

    assert q_len == k_len, "For encoder_only, token lengths should be the same"

    # Convert the shapes from vLLM's convention to what the attention function expects
    q_thd = q.reshape(q_len, num_heads, head_size)
    k_thd = k.reshape(k_len, num_kv_heads, head_size)
    v_thd = v.reshape(k_len, num_kv_heads, head_size)

    # vLLM does pre-compute the scale (=1/sqrt(head_size)) for our convenience.
    # (q * scale) @ k == (q @ k) * scale in the torch
    q_thd = q_thd * (scale)

    # TODO: Give some heuristic block size. Need to tune for best performance.
    block_q = min(q_len, 1024)
    # Align BKV to 128 per kernel request
    block_kv = min(k_len, 1024)
    block_kv = _align_to(block_kv, 128)
    block_kv_compute = min(block_kv, 256)
    # Align block_kv to block_kv_compute
    block_kv = _align_to(block_kv, block_kv_compute)
    assert block_kv % block_kv_compute == 0

    # Swap axes to head-first per kernel limit
    q_htd = q_thd.swapaxes(0, 1)
    k_htd = k_thd.swapaxes(0, 1)
    v_htd = v_thd.swapaxes(0, 1)

    def pad_token(t: jax.Array, size) -> jax.Array:
        # tensor is [num_head, token, head_dim]
        result = jnp.pad(t, ((0, 0), (0, size), (0, 0)), constant_values=0)
        return result

    # Pad q, k, v sequence, align to block sizes
    q_pad_htd = pad_token(q_htd, _align_to(q_len, block_q) - q_len)
    k_pad_htd = pad_token(k_htd, _align_to(k_len, block_kv) - k_len)
    v_pad_htd = pad_token(v_htd, _align_to(k_len, block_kv) - k_len)
    assert k_pad_htd.shape == v_pad_htd.shape

    def build_segment_ids() -> SegmentIds:
        # Create segment IDs since the sequence may contain many requests
        max_num_seqs = attn_metadata.seq_lens.shape[0]
        # Add max_num_seqs (fake ID) as the invalid padding value of segment ID
        zero_2_max_num_seqs = jnp.arange(max_num_seqs + 1, dtype=jnp.int32)
        seq_lens_concat_zero = jnp.concatenate([
            attn_metadata.seq_lens,
            jnp.array([0], dtype=attn_metadata.seq_lens.dtype),
        ])
        # When longer than total_repeat_length, remaining values will be discarded.
        # When shorter than total_repeat_length, the final value will be repeated.
        # With additional invalid segment ID at the end, we could make sure
        # the repeated value is not the same as the valid segment IDs.
        qkv_segment_ids = jnp.repeat(
            zero_2_max_num_seqs,
            seq_lens_concat_zero,
            total_repeat_length=q_len,
        )

        def build_padded_segment(size: int) -> jax.Array:
            padding_segment_id = max_num_seqs
            result = jnp.pad(
                qkv_segment_ids,
                (0, size),
                constant_values=padding_segment_id,
            )
            return result

        # Create segment IDs for the padded sequences
        segment_ids = SegmentIds(
            q=build_padded_segment(q_pad_seq_len - q_len),
            kv=build_padded_segment(k_pad_seq_len - k_len),
        )
        return segment_ids

    # Create attention mask
    # The mask should be applied to the padded sequence length
    q_pad_seq_len = q_pad_htd.shape[1]
    k_pad_seq_len = k_pad_htd.shape[1]
    assert q_pad_seq_len % block_q == 0
    assert k_pad_seq_len % block_kv == 0
    layer_mask = FullMask((q_pad_seq_len, k_pad_seq_len))
    mask = MultiHeadMask([layer_mask for _ in range(num_heads)])

    # Create BlockSizes
    block_sizes = BlockSizes(
        block_q=block_q,
        block_kv=block_kv_compute,
        block_kv_compute=block_kv_compute,
    )
    # Make and run the kernel
    kernel = make_splash_mha_single_device(
        mask,
        block_sizes=block_sizes,
    )
    output_htd = kernel(
        q_pad_htd,
        k_pad_htd,
        v_pad_htd,
        build_segment_ids(),
    )
    assert isinstance(output_htd, jax.Array), (
        f"With save_residual=False, expect jax.Array, but got {type(output_htd)}"
    )

    # Unpad and transpose back to vLLM's shape convention
    output = output_htd[:, :q_len, :].swapaxes(0, 1)
    return output.reshape(q_len, q_compute_dim).astype(q.dtype)


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
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        kv_cache_index = vllm_model_wrapper_context.layer_name_to_kvcache_index[
            layer.layer_name]
        kv_cache = vllm_model_wrapper_context.kv_caches[kv_cache_index]

        mesh = vllm_model_wrapper_context.mesh

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

        return torch_view(outputs)


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
    del scale  # Unused for now, as the attention function applies a default scale.

    # Get shapes from vllm
    q_len, q_compute_dim = q.shape
    k_len, k_compute_dim = k.shape
    assert k.shape == v.shape
    assert q_compute_dim == head_size * num_heads
    assert k_compute_dim == head_size * num_kv_heads

    # Convert the shapes from vLLM's convetion to what the attention function expects
    # bs, num_heads, q_len, head_size
    q = q.reshape(q_len, num_heads, head_size)
    # bs, num_kv_heads, k_len, head_size
    k = k.reshape(k_len, num_kv_heads, head_size)
    v = v.reshape(k_len, num_kv_heads, head_size)

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
    assert outputs.shape[0] == q_len
    assert outputs.shape[1] == num_heads
    assert outputs.shape[2] == head_size
    outputs = outputs.reshape(q_len, q_compute_dim)

    return new_kv_cache, outputs
