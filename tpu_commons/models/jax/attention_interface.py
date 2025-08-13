from typing import Tuple

import jax
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata


def sharded_ragged_paged_attention(sm_scale: float,
                                   mesh: Mesh,
                                   attention_chunk_size: int | None = None):
    """Shards along KV heads."""
    qkv_spec = P(None, "model", None)
    kv_cache_spec = P("model", None, None, None, None, None)
    in_specs = (
        qkv_spec,  # q
        qkv_spec,  # k
        qkv_spec,  # v
        kv_cache_spec,  # kv cache
        P(),  # kv_lens
        P(),  # page_indices
        P(),  # cu_q_lens
        P(),  # distribution
    )
    out_specs = (qkv_spec, kv_cache_spec)

    def _ragged_paged_attention(*args):
        return ragged_paged_attention(
            *args,
            sm_scale=sm_scale,
            sliding_window=attention_chunk_size,
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
    # P: kv_packing = 32 // kv_bit_width
    # Q: q_packing = 32 // q_bit_width

    # TODO(jevinjiang, cuiq): transpose q weight offline.
    # q: (T, N, H)
    # k,v: (T, K, H)
    # kv_cache: (L, S, cidv(2 * K, P), P, H)

    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    md = attention_metadata

    # (T, N, H)
    output, kv_cache = sharded_ragged_paged_attention(
        head_dim_original**-0.5, mesh, attention_chunk_size)(
            q,
            k,
            v,
            kv_cache,
            md.seq_lens,
            md.block_tables.reshape(-1),
            md.query_start_loc,
            md.request_distribution,
        )

    return kv_cache, output
