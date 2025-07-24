from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.ragged_kv_cache_update import kv_cache_update
from tpu_commons.kernels.ragged_paged_attention.kernel import \
    ragged_paged_attention, ragged_mla_paged_attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata

# TODO(xiang): put this in attention metadata
# Block size used for kv cache updating kernel
NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK = 8


def sharded_ragged_paged_attention(sm_scale: float, mesh: Mesh):
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
            sliding_window=None,
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

def sharded_mla_ragged_paged_attention(sm_scale: float, mesh: Mesh):
    """Shards along KV heads."""
    in_specs = (
        P(None, "model", None),  # q
        P(None, None, None, "model"),  # kv cache
        P(),  # kv_lens
        P(),  # page_indices
        P(),  # cu_q_lens
        P(),  # num_seqs
    )
    out_specs = P(None, "model", None)

    def _ragged_mla_paged_attention(*args):
        return ragged_mla_paged_attention(
            *args,
            sm_scale=sm_scale,
            sliding_window=None,
            soft_cap=None,
            mask_value=None,
            # NOTE(xiang): v6e chip has 128M VMEM capacity,
            # set this to 64M to avoid VMEM OOM,
            # otherwise the default value is 16M.
            vmem_limit_bytes=64 * 1024 * 1024,
        )

    return jax.jit(
        shard_map.shard_map(
            _ragged_mla_paged_attention,
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
        head_dim_original: int | None = None,  # before padding
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
    output = sharded_ragged_paged_attention(head_dim_original**-0.5, mesh)(
        q,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.num_seqs,
    )

    return kv_cache, output


def mla_attention(
        kv_cache: jax.Array,
        q: jax.Array,
        latent_kv: jax.Array,
        k_rope: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
        head_dim_original: int | None = None,  # before padding
) -> Tuple[jax.Array, jax.Array]:
    # T: seq_len
    # N: num_heads
    # D: hidden_size
    # H: latent head_dim (v head is different from qk head dim)
    # Q: q head dim
    # K: K rope head dim
    # L: num_blocks
    # S: block_size

    # q: (T, N, Q)
    # latent_kv: (T, 1, H)
    # k_rope: (T, N, K)
    # kv_cache: (L, S, 1, H)

    if head_dim_original is None:
        head_dim_original = q.shape[-1]

    md = attention_metadata
    kv_cache = update_mla_kv_cache(latent_kv, k_rope, kv_cache, md.slot_mapping, md.num_slices,
                               mesh)

    # (T, N, H)
    output = sharded_mla_ragged_paged_attention(head_dim_original**-0.5, mesh)(
        q,
        k_rope,
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
    kv_cache = kv_cache_update(
        kv,
        slices,
        kv_cache,
        num_slices,
        page_size=S,
        num_slices_per_block=NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK,
        mesh=mesh,
        kv_cache_pspec=P(None, "model", None))
    kv_cache = kv_cache.reshape(L, S, K_2, H)
    return kv_cache


def update_mla_kv_cache(latent_kv: jax.Array, k_rope: jax.Array, kv_cache: jax.Array,
                    slices: jax.Array, num_slices: jax.Array,
                    mesh: Mesh) -> jax.Array:
    """Write the latent KV and k_rope into KV cache for MLA.

    Args:
        latent_kv: (T, 1, H) - Latent KV vectors
        k_rope: (T, 1, K) - K_rope vector 
        kv_cache: (L, S, 2, H) - KV cache to update
        slices: (3, num_slices) - Slot mapping metadata
        num_slices: (1,) - Number of slices
        mesh: JAX device mesh

    Returns:
        Updated KV cache with shape (L, S, 2, H)
    """
    L, S, K, H = kv_cache.shape
    assert K == 2
    # Add head dimension to latent_kv if needed
    if len(latent_kv.shape) == 2:
        latent_kv = latent_kv[:, None, :]
    
    # Concatenate latent_kv and k_rope along the feature dimension
    latent_kv_k_rope = jnp.concat([latent_kv, k_rope], axis=-1)

    # Pad to multiple of 128 for TPU tiling constraints
    multiplier = 128 * 4
    padding_needed = multiplier - latent_kv_k_rope.shape[-1] % multiplier
    latent_kv_k_rope = jnp.pad(latent_kv_k_rope, ((0, 0), (0, 0), (0, padding_needed)))
    
    # Reshape to match expected format for kv_cache_update
    latent_kv_k_rope = latent_kv_k_rope.reshape(latent_kv_k_rope.shape[0], 2, -1)
    kv_cache = kv_cache.reshape(-1, 2, H)
    
    # Update KV cache using the existing kernel
    kv_cache = kv_cache_update(
        latent_kv_k_rope,
        slices,
        kv_cache,
        num_slices,
        page_size=S,
        num_slices_per_block=NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK,
        mesh=mesh,
        kv_cache_pspec=P(None, None, "model"))  # Should we shard the head_dim ?
    
    # Reshape back to original format
    kv_cache = kv_cache.reshape(L, S, K, H)

    return kv_cache
