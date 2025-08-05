from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.ragged_kv_cache_update import kv_cache_update
from tpu_commons.kernels.ragged_paged_attention.kernel import \
    ragged_paged_attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata


def sharded_ragged_paged_attention(sm_scale: float,
                                   mesh: Mesh,
                                   attention_chunk_size: int | None = None):
    """Shards along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P(None, None, "model", None),  # kv cache
        P(),  # kv_lens
        P(),  # page_indices
        P(),  # cu_q_lens
        P(),  # num_seqs
    )
    out_specs = P(None, "model", None)

    def _ragged_paged_attention(*args):
        return ragged_paged_attention(
            *args,
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
            soft_cap=None,
            mask_value=None,
            # NOTE(xiang): v6e chip has 128M VMEM capacity,
            # set this to 64M to avoid VMEM OOM,
            # otherwise the default value is 16M.
            vmem_limit_bytes=64 * 1024 * 1024,
        )

    return jax.jit(
        shard_map.shard_map(
            _ragged_paged_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        ))


def attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,  # before padding,
    attention_chunk_size: int | None = None
) -> Tuple[jax.Array, jax.Array]:
    # T: seq_len
    # N: num_heads
    # K: num_kv_heads
    # D: hidden_size
    # H: head_dim
    # L: num_blocks
    # S: block_size

    # q: (T, N, H)
    # k,v: (T, K, H)
    # kv_cache: (L, S, 2 * K, H)

    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    md = attention_metadata
    kv_cache = update_kv_cache(k, v, kv_cache, md.slot_mapping, md.num_slices,
                               mesh)

    # (T, N, H)
    output = sharded_ragged_paged_attention(head_dim_original**-0.5, mesh,
                                            attention_chunk_size)(
                                                q,
                                                kv_cache,
                                                md.seq_lens,
                                                md.block_tables,
                                                md.query_start_loc,
                                                md.num_seqs,
                                            )

    return kv_cache, output


def update_kv_cache(k: jax.Array, v: jax.Array, kv_cache: jax.Array,
                    slices: jax.Array, num_slices: jax.Array,
                    mesh: Mesh) -> jax.Array:
    """ Write K and V into KV cache.

    Args:
        k: (T, K, H)
        v: (T, K, H)
        kv_cache: (L, S, K*2, H)
    """
    L, S, K_2, H = kv_cache.shape
    T, K, H = k.shape

    # (T, K*2, H)
    # NOTE(xiang): KV needs to be interleaved as required by kernel
    kv = jnp.concat([k, v], axis=-1).reshape(T, K_2, H)

    kv_cache = kv_cache.reshape(-1, K_2, H)
    kv_cache = kv_cache_update(kv,
                               slices,
                               kv_cache,
                               num_slices,
                               page_size=S,
                               mesh=mesh,
                               kv_cache_pspec=P("data", "model", None), 
                               slices_spec=P(None, "data"),
                               num_slices_pspec=P("data"))
    jax.block_until_ready(kv_cache)
    kv_cache = kv_cache.reshape(L, S, K_2, H)
    return kv_cache
