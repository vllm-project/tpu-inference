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
    kv_cache = update_mla_kv_cache(latent_kv, kv_cache, md.slot_mapping, md.num_slices,
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
    """ Write the latent KV into KV cache.

    Args:
        latent_kv: (T, 1, H)
        kv_cache: (L, S, 1, H)
    """
    L, S, K, H = kv_cache.shape

    print("latent_kv.shape", latent_kv.shape) #(32, 512)
    print("k_rope.shape", k_rope.shape) # (32, 32, 32)
    
    # add head dimension to latent_kv
    latent_kv = latent_kv[:, None, :]
    
    # concat latent_kv and k_rope
    latent_kv_k_rope = jnp.concat([latent_kv, k_rope], axis=-1)

    
    # pad to multiple of 128
    multiplier = 128*4
    latent_kv_k_rope = jnp.pad(latent_kv_k_rope, ((0, 0), (0, 0), (0, multiplier - latent_kv_k_rope.shape[-1] % multiplier)))
    print("- latent_kv_k_rope.shape", latent_kv_k_rope.shape)    
    kv_cache = kv_cache.reshape(-1, K, H)
    print("- kv_cache.shape", kv_cache.shape)
    print("slices.shape", slices.shape)
    
    if latent_kv_k_rope.shape[1] == 1:
        latent_kv_k_rope = jnp.tile(latent_kv_k_rope, (1, 2, 1))
        print("- padded latent_kv_k_rope.shape", latent_kv_k_rope.shape)
        kv_cache = jnp.tile(kv_cache, (1, 2, 1))
        print("- padded kv_cache.shape", kv_cache.shape)

    
    kv_cache = kv_cache_update(
        latent_kv_k_rope,
        slices,
        kv_cache,
        num_slices,
        page_size=S,
        num_slices_per_block=NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK,
        mesh=mesh,
        kv_cache_pspec=P(None, None, "model"))
    kv_cache = kv_cache[:, 0, :]
    kv_cache = kv_cache.reshape(L, S, K, H)
    #untile kv_cache
    

    return kv_cache


# def flash_mla_with_kvcache(
#     q: torch.Tensor,
#     k_cache: torch.Tensor,
#     block_table: torch.Tensor,
#     cache_seqlens: torch.Tensor,
#     head_dim_v: int,
#     tile_scheduler_metadata: torch.Tensor,
#     num_splits: torch.Tensor,
#     softmax_scale: Optional[float] = None,
#     causal: bool = False,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Arguments:
#         q: (batch_size, seq_len_q, num_heads_q, head_dim).
#         k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
#         block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
#         cache_seqlens: (batch_size), torch.int32.
#         head_dim_v: Head_dim of v.
#         tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), 
#                                  torch.int32, return by get_mla_metadata.
#         num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
#         softmax_scale: float. The scaling of QK^T before applying softmax. 
#                        Default to 1 / sqrt(head_dim).
#         causal: bool. Whether to apply causal attention mask.

#     Return:
#         out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
#         softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
#     """
#     if softmax_scale is None:
#         softmax_scale = q.shape[-1]**(-0.5)
#     out, softmax_lse = torch.ops._flashmla_C.fwd_kvcache_mla(
#         q,
#         k_cache,
#         None,
#         head_dim_v,
#         cache_seqlens,
#         block_table,
#         softmax_scale,
#         causal,
#         tile_scheduler_metadata,
#         num_splits,
#     )
#     return out, softmax_lse
