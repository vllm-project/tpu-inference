# ruff: noqa: E731, E741, F722
import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import \
    splash_attention_mask as mask_lib
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float
from transformers import modeling_flax_utils

from tpu_commons.models.jax.config import (MAX_ALLOWED_PAGE_INDICES_N,
                                           LoRAConfig)
from tpu_commons.models.jax.kernels.flash_attention import \
    flash_attention as flash_attention_with_scores_caching
from tpu_commons.models.jax.kernels.paged_attention_v2 import \
    paged_attention as paged_attention_with_scores_caching
from tpu_commons.models.jax.kernels.ragged_paged_attention import \
    ragged_paged_attention
from tpu_commons.models.jax.param_init import sharding_init
from tpu_commons.models.jax.quantization.awq import (Int4Einsum,
                                                     Int4EinsumBias, Int4MLP)
from tpu_commons.models.jax.quantization.bitsandbytes import (Int8Einsum,
                                                              Int8MLP)
from tpu_commons.models.jax.quantization_config import (QuantizationConfig,
                                                        QuantizationMethod)
from tpu_commons.models.jax.utils import get_megacore


def fetch_cache(cache: jax.Array, indices: jax.Array) -> jax.Array:
    """Fetches specific indices from the cache and reshapes.

    Args:
        cache (jax.Array): The cached tensor of shape (K, L, S, H).
        indices (jax.Array): Indices to fetch from the cache of shape (B, Tb).

    Returns:
        jax.Array: Resulting tensor of shape (B, K, T, H), where T = Tb * S.
    """
    K, _, S, H = cache.shape
    # Tb: seq_len in block, i.e. Tb = T // S
    B, Tb = indices.shape
    T = Tb * S
    # indices: (B, Tb) -> (I,)
    indices = indices.reshape(-1)
    # cache: (K, L, S, H)
    # result: (K, I, S, H)
    result = cache.at[:, indices, :, :].get()
    # (K, I, S, H) -> (K, B, T, H) -> (B, K, T, H)
    return result.reshape(K, B, T, H).transpose(1, 0, 2, 3)


@functools.partial(
    jax.jit,
    donate_argnames=["cache", "operand"],
)
def chunked_prefill_update_cache(cache, indices, operand, num_decode_seqs):
    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    assert B == 1
    operand = jnp.squeeze(operand, 0)
    # operand now: KTH

    # Handle Decode tokens kv cache update.
    decode_indices = jax.lax.slice(indices, (0, ), (T, ))

    # Number of valid indice update could be much smaller
    # than T. We improve performance by skipping the
    # update for those padded indices as much as possible.
    # TODO(b/396129273): tune the value of DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE
    # base on benchmarking
    DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE = 16
    assert T % DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE == 0
    decode_indices = decode_indices.reshape((
        T // DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
    ))
    decode_update_operand = operand.reshape(
        K,
        T // DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE,
        H,
    )
    cache = cache.reshape(K, L * S, H)
    cache = jax.lax.fori_loop(
        0,
        jnp.ceil(num_decode_seqs[0] /
                 DECODE_TOKEN_CACHE_UPDATE_BLOCK_SIZE).astype(jnp.int32),
        body_fun=lambda i, c: c.at[jnp.arange(K)[..., None], decode_indices[
            i], :].set(decode_update_operand[:, i, :, :]),
        init_val=cache,
    )
    cache = cache.reshape(K, L, S, H)

    # Handle Prefill tokens kv cache update.
    I = T // S
    prefill_indices = jax.lax.slice(indices, (T, ), (T + I, ))
    # cache: (K, L, S, H)
    # prefill_operand: (K, T, H) -> (K, I, S, H)
    # prefill_indices: (I,)
    prefill_operand = operand.reshape(K, I, S, H)
    cache = cache.at[:, prefill_indices, :, :].set(prefill_operand)
    return cache


def sharded_chunked_prefill_update_cache(mesh: Mesh, ) -> Callable[..., Any]:
    """Shards along KV heads."""
    in_specs = (
        P("model", None, None),  # cache [K, L, S, H]
        P(),  # indice
        P(None, "model", None, None),  # operand [B, K, T, H]
        P(),  # num_decode_seqs
    )
    out_specs = P("model", None, None, None)

    return jax.jit(
        shard_map.shard_map(
            chunked_prefill_update_cache,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


def update_cache(
    is_prefill,
    cache,
    indices,
    operand,
    prefill_seq_len=None,
    sliding_window=None,
    sink_size=None,
) -> jax.Array:

    # (8, 55640, 32, 128) (1, 8, 256, 128) -> K (8, 8, 32, 128)
    # I = B * T // S
    # k cache, operand

    B, K, T, H = operand.shape
    K_c, L, S, H = cache.shape
    assert K == K_c
    # NOTE: The cache updating is pretty tricky:
    # 1. The random access updating cache is not as performant as the slice updating.
    #    If the random access is necessary, make sure the indexing count is as small as possible.
    # 2. The random access updating may trigger extra tranpose (memory copy) of cache,
    #    which is a disaster because the cache is huge. This is a data formatting op inserted by
    #    the XLA compiler and not well documented.
    # To mitigate the issues above:
    # For prefill:
    # We reshape the operand so that we can update the cache in block wise, which only requires the block indices.
    # For decode:
    # We reshape the cache so that we can update the cache in token wise, which only requires the token indices (block_id + offset).
    if is_prefill:
        # In the case of sliding window, we should select sliding_window tokens from actual prompt, not from the padded tokens.
        if sliding_window:
            if sink_size is not None and T > sliding_window + sink_size:
                assert B == 1
                sink_slice = jax.lax.dynamic_slice_in_dim(operand,
                                                          0,
                                                          sink_size,
                                                          axis=2)
                recent_slice = jax.lax.dynamic_slice_in_dim(operand,
                                                            prefill_seq_len -
                                                            sliding_window,
                                                            sliding_window,
                                                            axis=2)
                operand = jnp.concatenate([sink_slice, recent_slice], axis=2)
                T = sliding_window + sink_size
            elif sink_size is None and T > sliding_window:
                assert B == 1
                start_index = jax.lax.max(0, prefill_seq_len - sliding_window)
                operand = jax.lax.dynamic_slice_in_dim(
                    operand, start_index, sliding_window,
                    axis=2)  # TODO: @pooyam Perf check this.
                T = sliding_window

        I = B * T // S
        # cache: (K, L, S, H)
        # operand: (B, K, T, H) -> (K, I, S, H)
        # indices: (B, T // S) -> (I,)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, I, S, H)
        indices = indices.reshape(I)
        cache = cache.at[:, indices, :, :].set(operand)
    else:
        # cache: (K, L, S, H) -> (K, L * S, H)
        # operand: (B, K, 1, H) -> (K, B, H)
        # indices: (B,)
        cache = cache.reshape(K, L * S, H)
        operand = jnp.swapaxes(operand, 0, 1).reshape(K, B, H)
        # NOTE: `cache.[:, indices, :].set()` will trigger the extra tranpose of the cache.
        # The `jnp.arange(K)[..., None]` trick is to avoid it. WTF?
        cache = cache.at[jnp.arange(K)[..., None], indices, :].set(operand)
        cache = cache.reshape(K, L, S, H)
    return cache


class Einsum(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh
    hidden_dim: Optional[int] = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh),
            self.shape,
            self.dtype,
        )
        return jnp.einsum(eqn, x, w)


class EinsumBias(nn.Module):
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes: Tuple[str, ...]
    mesh: Mesh
    hidden_dim: Optional[int] = None

    # We need this because not every EinsumBias usage is for qkv_proj.
    bias_shape: Optional[Tuple[int, ...]] = None
    bias_named_axes: Optional[Tuple[str, ...]] = None

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        bias_shape = self.bias_shape or (
            self.shape[0],
            self.shape[2],
        )  # (num_heads or num_kv_heads, head_dim)
        bias_named_axes = self.bias_named_axes or ("model", None)

        b = self.param(
            "bias",
            sharding_init(bias_named_axes, self.mesh),
            bias_shape,
            self.dtype,
        )

        w = self.param(
            "weight",
            sharding_init(self.named_axes, self.mesh),
            self.shape,
            self.dtype,
        )

        return jnp.einsum(eqn, x, w), b


class LoRALayer(nn.Module):
    num_lora: int
    shape_a: Tuple[int, ...]
    shape_b: Tuple[int, ...]
    dtype: jnp.dtype
    named_axes_a: Tuple[str, ...]
    named_axes_b: Tuple[str, ...]
    mesh: Mesh

    def setup(self):
        a = [
            self.param(
                f"a_{i}",
                sharding_init(self.named_axes_a, self.mesh),
                self.shape_a,
                self.dtype,
            ) for i in range(self.num_lora)
        ]
        b = [
            self.param(
                f"b_{i}",
                sharding_init(self.named_axes_b, self.mesh),
                self.shape_b,
                self.dtype,
            ) for i in range(self.num_lora)
        ]
        self.a_stack = jnp.stack(a, axis=0)
        self.b_stack = jnp.stack(b, axis=0)

    def __call__(self, eqn_a: str, eqn_b: str, x: jax.Array) -> jax.Array:
        B = x.shape[0]
        assert B <= self.num_lora
        x = jnp.pad(
            x,
            pad_width=[(0, self.num_lora - B)] + [(0, 0)] * len(self.shape_a),
            mode="constant",
            constant_values=0,
        )
        x = jnp.einsum(eqn_a, x, self.a_stack)
        x = jnp.einsum(eqn_b, x, self.b_stack)
        x = x[:B]
        return x


class RMSNorm(nn.Module):
    rms_norm_eps: float
    dtype: jnp.dtype
    mesh: Mesh

    @nn.compact
    def __call__(self, x) -> jax.Array:
        scale = self.param(
            "weight",
            sharding_init((None, ), self.mesh),
            (x.shape[-1], ),
            self.dtype,
        )
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(
            x * jnp.reciprocal(jnp.sqrt(var + self.rms_norm_eps)))
        normed_inputs = normed_inputs * scale
        return normed_inputs


@functools.partial(jax.jit, static_argnames=["act"])
def mlp_fwd(
    x: Float[Array, "T H"],
    up_proj: Float[Array, "N H F"],
    gate_proj: Float[Array, "N H F"] | None,
    down_proj: Float[Array, "N F H"],
    act: str,
):
    up = jnp.dot(x, up_proj)
    fuse = up
    if gate_proj is not None:
        gate = jnp.dot(x, gate_proj)
        gate = modeling_flax_utils.ACT2FN[act](gate)
        fuse = gate * up
    return jnp.dot(fuse, down_proj)


class MLP(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype
    mesh: Mesh
    act: str

    @nn.compact
    def __call__(self, x, with_gate_proj: bool = True) -> jax.Array:
        # Llama4 vision encoder FeedForward does not have gate projection
        gate_proj = None
        if with_gate_proj:
            gate_proj = self.param(
                "gate_proj",
                sharding_init((None, "model"), self.mesh),
                (self.hidden_size, self.intermediate_size),
                self.dtype,
            )

        up_proj = self.param(
            "up_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )
        down_proj = self.param(
            "down_proj",
            sharding_init(("model", None), self.mesh),
            (self.intermediate_size, self.hidden_size),
            self.dtype,
        )

        return mlp_fwd(
            x=x,
            up_proj=up_proj,
            gate_proj=gate_proj,
            down_proj=down_proj,
            act=self.act,
        )


class MLPLoRA(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: jnp.dtype
    mesh: Mesh
    act: str
    lora_config: LoRAConfig

    def setup(self):
        # Shard the intermediate_size dimension along the model axis.
        self.gate_proj = self.param(
            "gate_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )
        self.up_proj = self.param(
            "up_proj",
            sharding_init((None, "model"), self.mesh),
            (self.hidden_size, self.intermediate_size),
            self.dtype,
        )
        self.down_proj = self.param(
            "down_proj",
            sharding_init(("model", None), self.mesh),
            (self.intermediate_size, self.hidden_size),
            self.dtype,
        )
        self.gate_proj_lora = LoRALayer(
            num_lora=self.lora_config.max_num_lora,
            shape_a=(
                self.hidden_size,
                self.lora_config.max_lora_rank,
            ),
            shape_b=(
                self.lora_config.max_lora_rank,
                self.intermediate_size,
            ),
            dtype=self.dtype,
            named_axes_a=(None, None),
            named_axes_b=(None, "model"),
            mesh=self.mesh,
        )
        self.up_proj_lora = LoRALayer(
            num_lora=self.lora_config.max_num_lora,
            shape_a=(
                self.hidden_size,
                self.lora_config.max_lora_rank,
            ),
            shape_b=(
                self.lora_config.max_lora_rank,
                self.intermediate_size,
            ),
            dtype=self.dtype,
            named_axes_a=(None, None),
            named_axes_b=(None, "model"),
            mesh=self.mesh,
        )
        self.down_proj_lora = LoRALayer(
            num_lora=self.lora_config.max_num_lora,
            shape_a=(
                self.intermediate_size,
                self.lora_config.max_lora_rank,
            ),
            shape_b=(
                self.lora_config.max_lora_rank,
                self.hidden_size,
            ),
            dtype=self.dtype,
            named_axes_a=("model", None),
            named_axes_b=(None, None),
            mesh=self.mesh,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # M: max_num_lora
        # T: seq_len
        # D: hidden_size
        # R: max_lora_rank
        # F: intermediate_size

        gate = jnp.dot(x, self.gate_proj)
        gate_lora_outputs = self.gate_proj_lora("MTD,MDR->MTR", "MTR,MRF->MTF",
                                                x)
        gate = gate + gate_lora_outputs
        gate = modeling_flax_utils.ACT2FN[self.act](gate)

        up = jnp.dot(x, self.up_proj)
        up_lora_outputs = self.up_proj_lora("MTD,MDR->MTR", "MTR,MRF->MTF", x)
        up = up + up_lora_outputs

        fuse = gate * up

        down = jnp.dot(fuse, self.down_proj)
        down_lora_outputs = self.down_proj_lora("MTF,MFR->MTR", "MTR,MRD->MTD",
                                                fuse)
        down = down + down_lora_outputs
        return down


@functools.partial(
    jax.jit,
    static_argnames=[
        "window_size",
        "attn_logits_soft_cap",
        "is_mqa",
        "is_causal",
    ],
)
def apply_splash(
    q,
    k,
    v,
    head_mask: jax.Array,
    window_size,
    attn_logits_soft_cap,
    is_mqa,
    is_causal: bool = True,
) -> jax.Array:
    # q: (batch_size, num_heads, seq_len, head_dim)
    num_heads = q.shape[1]
    q_seq_len = q.shape[2]
    kv_seq_len = k.shape[2]
    assert kv_seq_len >= q_seq_len

    if is_causal:
        # Static masking.
        head_mask = mask_lib.LocalMask((q_seq_len, kv_seq_len),
                                       (window_size, 0),
                                       kv_seq_len - q_seq_len)[:, :]
    mask = jnp.stack([head_mask] * num_heads)

    block_sizes = splash.BlockSizes.get_default()

    if is_mqa:
        attn = splash.make_splash_mqa_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
        k = jnp.squeeze(k, 1)
        v = jnp.squeeze(v, 1)
    else:
        attn = splash.make_splash_mha_single_device(
            mask,
            block_sizes=block_sizes,
            attn_logits_soft_cap=attn_logits_soft_cap)
    attn = jax.vmap(attn)
    outputs = attn(q, k, v, None)

    return outputs


def sharded_splash_attention(
    mesh: Mesh,
    window_size: Optional[int] = None,
    attn_logits_soft_cap: Optional[float] = None,
    is_mqa: bool = False,
    is_causal: bool = True,
) -> Callable[..., Any]:
    in_specs = (
        P("data", "model", None, None),  # q
        P("data", "model", None, None),  # k
        P("data", "model", None, None),  # vx
    )
    if is_causal:
        head_mask_args = {"head_mask": jnp.int32(0)}
    else:
        in_specs = (
            *in_specs,
            P(),  # head_mask
        )
        head_mask_args = {}
    out_specs = P("data", "model", None, None)
    return jax.jit(
        shard_map.shard_map(
            functools.partial(
                apply_splash,
                window_size=window_size,
                attn_logits_soft_cap=attn_logits_soft_cap,
                is_mqa=is_mqa,
                is_causal=is_causal,
                **head_mask_args,
            ),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


def sharded_flash_attention(mesh: Mesh,
                            cache_attention_scores: bool = False,
                            causal: bool = True) -> Callable[..., Any]:
    in_specs = (
        P("data", "model", None, None),  # q
        P("data", "model", None, None),  # k
        P("data", "model", None, None),  # vx
    )
    if cache_attention_scores:
        flash_attention_fn = functools.partial(
            flash_attention_with_scores_caching, cache_attention_scores=True)
        out_specs = (
            P("data", "model", None, None),  # outputs
            P("data", "model", None, None),  # attention scores
        )
    else:
        flash_attention_fn = flash_attention
        out_specs = P("data", "model", None, None)
    return jax.jit(
        shard_map.shard_map(
            functools.partial(flash_attention_fn, causal=causal),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


def sharded_paged_attention(
    mesh: Mesh,
    attn_logits_soft_cap: Optional[float] = None,
    cache_attention_scores: bool = False,
) -> Callable[..., Any]:
    """Shards GQA PagedAttention along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P("model", None, None, None),  # k
        P("model", None, None, None),  # v
        P(),  # lengths
        P(),  # page_indices
    )
    if cache_attention_scores:
        paged_attention_fn = functools.partial(
            paged_attention_with_scores_caching, cache_attention_scores=True)
        out_specs = (
            P(None, "model", None),  # outputs
            P(None, "model", None),  # attention scores
        )
    else:
        paged_attention_fn = paged_attention
        out_specs = P(None, "model", None)

    def _paged_attention_fn(q, k, v, lengths, page_indices):
        if page_indices.size > MAX_ALLOWED_PAGE_INDICES_N:
            raise ValueError(
                "This will result in smem OOM. Use `paged_attention_with_guarded_smem` to run with minibatches."
            )
        return paged_attention_fn(
            q,
            k,
            v,
            lengths,
            page_indices,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=min(
                16, page_indices.shape[1]),  # 512 / page_size:32,
            megacore_mode="kv_head" if get_megacore() else None,
        )

    return jax.jit(
        shard_map.shard_map(
            _paged_attention_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


@functools.partial(jax.jit, static_argnums=[0])
def paged_attention_with_guarded_smem(
    paged_attention_kernel: Callable,
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    lengths: jax.Array,
    page_indices: jax.Array,
):
    # Addresses b/336316706. Summary:
    # Paged attention kernel stores `lengths` (batch_size * 4 bytes) and `page_indices` (batch_size * num_blocks_per_seq * 4 bytes) in SMEM.
    # Capacity of SMEM is quite limited which is also TPU version dependent. Models with higher context length or higher batch size, can cause OOM in SMEM.
    # There are two solutions:
    # 1. Reduce blocks per seq by increasing page size.
    # 2. Splitting the batch into several minibatches (Higher perf based on my benchmark).

    batch_size, blocks_per_seq = page_indices.shape

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_kernel(q, k_pages, v_pages, lengths,
                                      page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(batch_size, mini_batch_size)

    num_kernel_launches = batch_size // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_kernel_launches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_kernel_launches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_kernel_launches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_kernel_launches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_kernel_launches):
        outputs = outputs.at[i].set(
            paged_attention_kernel(q[i], k_pages, v_pages, seq_lens[i],
                                   block_indices[i]))

    outputs = outputs.reshape((batch_size, *outputs.shape[2:]))

    return outputs


def get_einsum(quantization_config: Optional[QuantizationConfig] = None,
               bias: bool = False) -> Type[nn.Module]:
    if quantization_config is not None:
        if quantization_config.get_quant_method(
        ) == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                return Int8Einsum
            else:
                raise ValueError(
                    "Only 8-bit einsum is supported for bitsandbytes.")
        elif quantization_config.get_quant_method() == QuantizationMethod.AWQ:
            if quantization_config.bits == 4:
                if quantization_config.zero_point is True and bias:
                    return Int4EinsumBias
                elif quantization_config.zero_point is True:
                    return Int4Einsum
                else:
                    raise ValueError(
                        "Only zero point quantization is supported for AWQ for now."
                    )
            else:
                raise ValueError("Only 4-bit einsum is supported for awq.")
        else:
            raise ValueError(
                f"Einsum not supported for {quantization_config.get_quant_method()}."
            )
    elif bias:
        return EinsumBias
    return Einsum


def get_mlp(
    quantization_config: Optional[QuantizationConfig] = None,
    lora_config: Optional[LoRAConfig] = None,
) -> Type[nn.Module]:
    if quantization_config is not None:
        if quantization_config.get_quant_method(
        ) == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                return Int8MLP
            else:
                raise ValueError(
                    "Only 8-bit MLP is supported for bitsandbytes.")
        elif quantization_config.get_quant_method() == QuantizationMethod.AWQ:
            if quantization_config.bits == 4:
                if quantization_config.zero_point is True:
                    return Int4MLP
                else:
                    raise ValueError(
                        "Only zero point quantization is supported for AWQ for now."
                    )
            else:
                raise ValueError("Only 4-bit MLP is supported for awq.")
        else:
            raise ValueError(
                f"MLP not supported for {quantization_config.get_quant_method()}."
            )
    if lora_config is not None and lora_config.enable_lora:
        return MLPLoRA
    return MLP


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "seq_lens",
        "block_indices",
        "kv_cache_write_indices",
        "decode_lengths",
        "decode_page_indices",
        "num_decode_seqs",
        "prefill_lengths",
        "prefill_page_indices",
        "prefill_query_start_offsets",
        "num_prefill_seqs",
    ],
    meta_fields=["chunked_prefill_enabled"],
)
@dataclass
class AttentionMetadata(object):
    input_positions: jax.Array
    # If mix attention, this is a list of len 2
    seq_lens: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    block_indices: Union[jax.Array, List[jax.Array]]
    # If mix attention, this is a list of len 2
    kv_cache_write_indices: Union[jax.Array, List[jax.Array]]
    # The following fields are set only when chunked prefill is enabled
    chunked_prefill_enabled: bool = False
    decode_lengths: jax.Array = None  # [max_num_decode_seqs]
    decode_page_indices: jax.Array = None  # [max_num_decode_seqs, pages_per_sequence]
    num_decode_seqs: jax.Array = None  # [1]
    prefill_lengths: jax.Array = None  # [max_num_prefill_seqs]
    prefill_page_indices: jax.Array = None  # [max_num_prefill_seqs, pages_per_sequence]
    prefill_query_start_offsets: jax.Array = None  # [max_num_prefill_seqs + 1]
    num_prefill_seqs: jax.Array = None  # [1]


def sharded_chunked_prefill_attention(mesh: Mesh, ) -> Callable[..., Any]:
    """Shards along KV heads."""
    # q: BNTH
    in_specs = (
        P(None, "model", None, None),  # q
        P("model", None, None, None),  # k_pages
        P("model", None, None, None),  # v_pages
        P(),  # decode_lengths
        P(),  # decode_page_indices
        P(),  # num_decode_seqs
        P(),  # prefill_lengths
        P(),  # prefill_page_indices
        P(),  # prefill_query_start_offsets
        P(),  # num_prefill_seqs
    )
    # output: BNTH
    out_specs = P(None, "model", None, None)

    return jax.jit(
        shard_map.shard_map(
            chunked_prefill_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


# Attention that can handle a token batch with a mix of decode and prefill tokens.
#
# Layout of q: [1, num_heads, num_tokens_in_batch, head_dim]
# For the 3rd dimension, decode tokens come first, followed by one or more
# prefill segments. There may be padding between the last decode token
# and the start of first prefill token, as well as between consecutive prefill
# segments, so that each prefill sequence segments are page-aligned.
@functools.partial(jax.jit, donate_argnames=["q"])
def chunked_prefill_attention(
        q: jax.Array,  # [1, num_heads, num_tokens_in_batch, head_dim]
        k_pages: jax.
    Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
        v_pages: jax.
    Array,  # [num_kv_heads, total_num_pages, page_size, head_dim]
        decode_lengths: jax.Array,  # [max_num_decode_seqs]
        decode_page_indices: jax.
    Array,  # [max_num_decode_seqs, pages_per_sequence]
        num_decode_seqs: jax.Array,  # [1]
        prefill_lengths: jax.Array,  # [max_num_prefill_seqs]
        prefill_page_indices: jax.
    Array,  # [max_num_prefill_seqs, pages_per_sequence]
        prefill_query_start_offsets: jax.Array,  # [max_num_prefill_seqs + 1]
        num_prefill_seqs: jax.Array,  # [1]
):
    batch_size, _, num_tokens_in_batch, _ = q.shape
    assert batch_size == 1
    q = jnp.squeeze(jnp.swapaxes(q, 1, 2), 0)
    # Now q: [num_tokens_in_batch, num_heads, head_dim]

    max_num_decode_seqs = decode_lengths.shape[0]
    assert max_num_decode_seqs <= num_tokens_in_batch
    decode_outputs = jnp.empty_like(q)
    decode_outputs = jax.lax.cond(
        num_decode_seqs[0] > 0,
        lambda: decode_outputs.at[0:max_num_decode_seqs].set(
            _attention_decode(
                q[0:max_num_decode_seqs],
                k_pages,
                v_pages,
                decode_lengths,
                decode_page_indices,
                num_decode_seqs,
            )),
        lambda: decode_outputs,
    )

    # TODO(b/396129273): Tune num_kv_pages_per_compute_block, num_queries_per_compute_block
    # in a generic way.
    prefill_outputs = ragged_paged_attention(
        q=q,
        k_pages=k_pages,
        v_pages=v_pages,
        kv_lens=prefill_lengths,
        page_indices=prefill_page_indices,
        cu_q_lens=prefill_query_start_offsets,
        num_seqs=num_prefill_seqs,
        num_kv_pages_per_block=16,
        num_queries_per_block=128,
    )

    ret = jnp.where(
        jnp.expand_dims(
            jnp.arange(num_tokens_in_batch) < num_decode_seqs[0], (1, 2)),
        decode_outputs,
        prefill_outputs,
    )

    return jnp.expand_dims(jnp.swapaxes(ret, 0, 1), 0)


@jax.jit
def _attention_decode(
    q,  # [max_num_decode_tokens, num_heads, head_dim]
    k_pages,
    v_pages,
    lengths,
    page_indices,
    num_decode_seqs,
):
    num_tokens_in_batch, blocks_per_seq = page_indices.shape
    paged_attention_fn = functools.partial(
        paged_attention,
        pages_per_compute_block=16,
        megacore_mode="kv_head" if get_megacore() else None,
    )

    if page_indices.size <= MAX_ALLOWED_PAGE_INDICES_N:
        return paged_attention_fn(q, k_pages, v_pages, lengths, page_indices)

    mini_batch_size = MAX_ALLOWED_PAGE_INDICES_N // blocks_per_seq

    # If batch_size is not disible by mini_batch_size,
    # we set mini_batch_size to a smaller value, i.e GCD,
    # which will trigger more kernel launches but it's fine.
    # TODO: Fix --decode_seqs_padding with this limitation.
    mini_batch_size = math.gcd(num_tokens_in_batch, mini_batch_size)
    num_mini_batches = num_tokens_in_batch // mini_batch_size

    outputs = jnp.zeros_like(q).reshape(
        (num_mini_batches, mini_batch_size, *q.shape[1:]))
    q = q.reshape((num_mini_batches, mini_batch_size, *q.shape[1:]))
    seq_lens = lengths.reshape((num_mini_batches, mini_batch_size))
    block_indices = page_indices.reshape(
        (num_mini_batches, mini_batch_size, page_indices.shape[1]))

    for i in range(num_mini_batches):
        outputs = jax.lax.cond(
            i * mini_batch_size < num_decode_seqs[0],
            lambda q, k_pages, v_pages, seq_lens, block_indices: outputs.at[i].
            set(
                paged_attention_fn(
                    q,
                    k_pages,
                    v_pages,
                    seq_lens,
                    block_indices,
                )),
            lambda q, k_pages, v_pages, seq_lens, block_indices: outputs,
            q[i],
            k_pages,
            v_pages,
            seq_lens[i],
            block_indices[i],
        )
    return outputs.reshape((num_tokens_in_batch, *outputs.shape[2:]))


def rms_norm(x: jax.Array, eps: float = 1e-6):
    return x * jax.lax.rsqrt(
        jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)


def l2_norm(arr: jax.Array):
    return arr * jax.lax.rsqrt(jnp.sum(jnp.square(arr), axis=-1,
                                       keepdims=True))
