# SPDX-License-Identifier: Apache-2.0
import functools

import jax
import numpy as np
import torch
import torch.nn.functional as F
import torchax
from jax import Array
from jax.experimental.pallas.ops.tpu.flash_attention import \
    flash_attention as flash_attention_kernel
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from torch.nn.utils import stateless as torch_stateless
from torchax.interop import call_jax
from vllm import envs
from vllm.forward_context import set_forward_context


def with_torchax_global(func):
    """Decorator that enables torchax globally before function call and
    disables after. Does nothing if torchax is not installed."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torchax.enable_globally()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            torchax.disable_globally()

    return wrapper


def get_cpu_tensor_from_torchax_tensor(tensor) -> torch.Tensor:
    assert isinstance(tensor, torchax.tensor.Tensor), \
        f"Expected torchax.Tensor, got {type(tensor)}"

    np_array = np.asarray(tensor.jax())
    if tensor.dtype == torch.bfloat16:
        np_array = np_array.astype(np.float32)

    cpu_torch_t = torch.from_numpy(np_array)
    if tensor.dtype == torch.bfloat16:
        cpu_torch_t = cpu_torch_t.to(torch.bfloat16)
    return cpu_torch_t


def wrap_model(m, vllm_config, static_forward_context):

    def func(weights, inputs, kv_caches, attn_metadata, num_tokens):
        with set_forward_context(attn_metadata,
                                 vllm_config,
                                 num_tokens=num_tokens):
            for layer_name, attn in static_forward_context.items():
                attn.kv_cache = [kv_caches[layer_name]]
            kwargs = {
                "input_ids": inputs[0],
                "positions": inputs[1],
            }
            if vllm_config.model_config.is_multimodal_model and len(
                    inputs) > 1:
                kwargs["inputs_embeds"] = inputs[2]
            # TODO: some buffers are tied, investigate how it works.
            res = torch.func.functional_call(m,
                                             weights,
                                             kwargs=kwargs,
                                             tie_weights=False)
            new_kv_cache = dict()
            for layer_name, attn in static_forward_context.items():
                new_kv_cache[layer_name] = attn.kv_cache
            return res, new_kv_cache

    func_wrapped = jax.jit(torchax.interop.jax_view(func),
                           static_argnums=(4, ),
                           donate_argnums=(2, ))

    return func_wrapped


def wrap_model_func(model, method_name):

    def func(params_and_buffers, *args, **kwargs):
        with torch_stateless._reparametrize_module(model, params_and_buffers):
            res = getattr(model, method_name)(*args, **kwargs)
        return res

    func_wrapped = jax.jit(torchax.interop.jax_view(func))

    return func_wrapped


def get_mesh():
    """Get a jax device mesh.

    TODO: We should get the mesh from a common function.
    """
    return Mesh(jax.devices(), axis_names=('x', ))


@functools.partial(
    jax.jit,
    static_argnames=[
        "use_kernel",
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
    ],
)
def _ragged_paged_attention(
    q: Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: Array,  # [total_num_pages, page_size,
    #  num_combined_kv_heads, head_dim]
    kv_lens: Array,  # i32[max_num_seqs]
    page_indices: Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: Array,  # i32[max_num_seqs + 1]
    num_seqs: Array,  # i32[1]
    use_kernel: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> Array:

    assert use_kernel, "use_kernel must be True for torchax path."

    from tpu_inference.kernels.ragged_paged_attention.kernel import \
        ragged_paged_attention as ragged_paged_attention_kernel

    def call_kernel(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs):
        """Calls the ragged paged attention kernel."""
        return ragged_paged_attention_kernel(
            q=q,
            kv_pages=kv_pages,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            num_seqs=num_seqs,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes)

    # Define sharding specifications for better readability
    attention_in_specs = (
        P(None, 'x', None),  # q: shard on head dimension
        P(None, None, 'x'),  # kv_pages: shard on kv_head dimension
        P(None),  # kv_lens: replicated
        P(None, None),  # page_indices: replicated
        P(None),  # cu_q_lens: replicated
        P(None),  # num_seqs: replicated
    )
    attention_out_specs = P(None, 'x', None)  # output: shard on head dimension

    @functools.partial(jax.shard_map,
                       mesh=get_mesh(),
                       in_specs=attention_in_specs,
                       out_specs=attention_out_specs,
                       check_vma=False)
    def wrap_shard_map(q, kv_pages, kv_lens, page_indices, cu_q_lens,
                       num_seqs):
        """Wraps the ragged paged attention kernel for sharding."""
        return call_kernel(q, kv_pages, kv_lens, page_indices, cu_q_lens,
                           num_seqs)

    args = (
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
    )

    if envs.VLLM_XLA_USE_SPMD:
        return wrap_shard_map(*args)
    else:
        return call_kernel(*args)


ragged_paged_attention = functools.partial(call_jax, _ragged_paged_attention)


@functools.partial(
    jax.jit,
    static_argnames=["page_size", "num_slices_per_block"],
    donate_argnames="kv_cache",
)
def _kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_combined_kv_heads, head_dim]
    slices: jax.
    Array,  # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    kv_cache: jax.
    Array,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    num_slices: jax.Array,  # [1]
    *,
    page_size: int = 32,
    num_slices_per_block: int = None,
) -> Array:
    # TODO: Get rid of this wrapper and call from pallas.py directly. Need to
    #       find a better way to get mesh in pallas.py.

    from tpu_inference.kernels.ragged_kv_cache_update import kv_cache_update

    mesh = None
    kv_cache_pspec = None
    if envs.VLLM_XLA_USE_SPMD:
        mesh = get_mesh()
        kv_cache_pspec = P(None, 'x', None)

    return kv_cache_update(new_kv,
                           slices,
                           kv_cache,
                           num_slices,
                           page_size=page_size,
                           num_slices_per_block=num_slices_per_block,
                           mesh=mesh,
                           kv_cache_pspec=kv_cache_pspec)


kv_cache_update = functools.partial(call_jax, _kv_cache_update)


@functools.partial(
    jax.jit,
    static_argnames=["causal", "sm_scale"],
)
def _flash_attention(
    q: Array,  # [batch, num_heads, seq_len, head_dim]
    k: Array,  # [batch, num_heads, seq_len, head_dim]
    v: Array,  # [batch, num_heads, seq_len, head_dim]
    causal: bool = False,
    sm_scale: float = 1.0,
) -> Array:

    def call_kernel(q, k, v):
        """Calls the flash attention kernel."""
        return flash_attention_kernel(q=q,
                                      k=k,
                                      v=v,
                                      causal=causal,
                                      sm_scale=sm_scale)

    # if SPMD enabled, shard by heads
    attention_in_specs = (
        P(None, 'x', None, None),  # q
        P(None, 'x', None, None),  # k
        P(None, 'x', None, None),  # v
    )
    attention_out_specs = P(None, 'x', None, None)

    @functools.partial(jax.shard_map,
                       mesh=get_mesh(),
                       in_specs=attention_in_specs,
                       out_specs=attention_out_specs,
                       check_vma=False)
    def wrap_shard_map(q, k, v):
        return call_kernel(q, k, v)

    args = (
        q,
        k,
        v,
    )

    if envs.VLLM_XLA_USE_SPMD:
        return wrap_shard_map(*args)
    else:
        return call_kernel(*args)


flash_attention_jax = functools.partial(call_jax, _flash_attention)


def flash_attention(
        q: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        k: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        v: torch.Tensor,  # [batch, num_heads, seq_len, head_dim]
        q_len: int,
        kv_len: int,
        causal: bool = False,
        sm_scale: float = 1.0,
        block_size=128):

    pad_q_len = (block_size - q_len % block_size) % block_size
    if pad_q_len > 0:
        q = F.pad(q, (0, 0, 0, pad_q_len))

    pad_kv_len = (block_size - kv_len % block_size) % block_size
    if pad_kv_len > 0:
        k = F.pad(k, (0, 0, 0, pad_kv_len))
        v = F.pad(v, (0, 0, 0, pad_kv_len))

    out = flash_attention_jax(q, k, v, causal, sm_scale)
    out = out.transpose(1, 2)
    if pad_q_len > 0:
        out = out[:, :q_len]
    return out
