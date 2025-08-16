# SPDX-License-Identifier: Apache-2.0
import functools

import jax
import numpy as np
import torch
import torchax
from jax import Array
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
        "sm_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "soft_cap",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "debug_mode",
    ],
    donate_argnames=("kv_cache",),
)
def _ragged_paged_attention(
    queries: Array,   # [max_num_batched_tokens, num_q_heads, head_dim]
    keys: Array,      # [max_num_batched_tokens, num_kv_heads, head_dim]
    values: Array,    # [max_num_batched_tokens, num_kv_heads, head_dim]
    kv_cache: Array,  # [total_num_pages, page_size,
    #  num_combined_kv_heads, head_dim]
    kv_lens: Array,  # i32[max_num_seqs]
    page_indices: Array,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens: Array,  # i32[max_num_seqs + 1]
    distribution: Array,  # i32[3]
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> Array:

    from tpu_commons.kernels.ragged_paged_attention.v3.kernel import (
        ragged_paged_attention as ragged_paged_attention_kernel,
    )

    def call_kernel(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        """Calls the ragged paged attention kernel."""
        # TODO(cuiq): We should flatten page_indices in the caller.
        page_indices = page_indices.flatten()
        return ragged_paged_attention_kernel(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            mask_value=mask_value,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            chunk_prefill_size=chunk_prefill_size,
            # Kernel tuning params.
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
            # Debug params.
            debug_mode=debug_mode,
        )

    # Define sharding specifications for better readability
    attention_in_specs = (
        P(None, "x", None),  # queries: shard on head dimension
        P(None, "x", None),  # keys: shard on head dimension
        P(None, "x", None),  # values: shard on head dimension
        P(None, None, "x", None),  # kv_cache: shard on kv_head dimension
        P(None),  # kv_lens: replicated
        P(None),  # page_indices: replicated
        P(None),  # cu_q_lens: replicated
        P(None),  # distribution: replicated
    )
    attention_out_specs = (
        P(None, "x", None),  # output: shard on head dimension
        P(None, None, "x", None),  # kv_cache: shard on kv_head dimension
    )

    @functools.partial(
        jax.shard_map,
        mesh=get_mesh(),
        in_specs=attention_in_specs,
        out_specs=attention_out_specs,
        check_vma=False,
    )
    def wrap_shard_map(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        """Wraps the ragged paged attention kernel for sharding."""
        return call_kernel(
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )

    args = (
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
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

    from tpu_commons.kernels.ragged_kv_cache_update import kv_cache_update

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
