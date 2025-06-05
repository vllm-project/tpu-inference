# SPDX-License-Identifier: Apache-2.0
import functools

import jax
import torch
from torch.nn.utils import stateless as torch_stateless

try:
    import torchax
    from torchax.interop import call_jax, jax_jit
    TORCHAX_AVAILABLE = True
except ImportError:
    TORCHAX_AVAILABLE = False

from vllm.forward_context import set_forward_context


def with_torchax_global(func):
    """Decorator that enables torchax globally before function call and
    disables after. Does nothing if torchax is not installed."""
    if not TORCHAX_AVAILABLE:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torchax.enable_globally()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            torchax.disable_globally()

    return wrapper


def wrap_model(m, vllm_config, static_forward_context):

    @functools.partial(
        jax_jit,
        kwargs_for_jax_jit={
            "static_argnums": (4, ),
            "donate_argnums": (2, )  # KV cache buffer donation.
        },
    )
    def func(weights, inputs, kv_caches, attn_metadata, num_tokens):
        with set_forward_context(attn_metadata,
                                 vllm_config,
                                 num_tokens=num_tokens):
            for layer_name, attn in static_forward_context.items():
                attn.kv_cache = [kv_caches[layer_name]]
            # TODO: some buffers are tied, investigate how it works.
            res = torch.func.functional_call(m,
                                             weights,
                                             kwargs={
                                                 "input_ids": inputs[0],
                                                 "positions": inputs[1],
                                             },
                                             tie_weights=False)
            new_kv_cache = dict()
            for layer_name, attn in static_forward_context.items():
                new_kv_cache[layer_name] = attn.kv_cache
            return res, new_kv_cache

    return func


def wrap_model_func(model, method_name):

    @jax_jit
    def func(params_and_buffers, *args, **kwargs):
        with torch_stateless._reparametrize_module(model, params_and_buffers):
            res = getattr(model, method_name)(*args, **kwargs)
        return res

    return func


def _ragged_paged_attention(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages: jax.
    Array,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    use_kernel: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):

    from torch_xla.experimental.pallas_kernels.ragged_paged_attention_v2 import \
        ragged_paged_attention as ragged_paged_attention_kernel
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
        vmem_limit_bytes=vmem_limit_bytes,
    )


ragged_paged_attention = functools.partial(call_jax, _ragged_paged_attention)
