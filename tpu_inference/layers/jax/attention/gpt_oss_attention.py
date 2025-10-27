from dataclasses import InitVar, dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.rope import GptOssRotaryEmbedding

KVCache = Tuple[jax.Array, jax.Array]


@dataclass(kw_only=True)
class GptOssAttention(nnx.Module):
    """
    JAX implementation of the GPT-OSS Attention block
    """
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    dtype: jnp.dtype
    rngs: InitVar[nnx.Rngs]

    rope_theta: float
    initial_context_length: int = 4096
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    query_tnh: P = P()
    keyvalue_skh: P = P()
    attn_o_tnh: P = P()
    dnh_sharding: Sharding = ()
    dkh_sharding: Sharding = ()
    nhd_sharding: Sharding = ()
    n_sharding: Sharding = ()
    nh_sharding: Sharding = ()
    kh_sharding: Sharding = ()
    d_sharding: Sharding = ()

    random_init: bool = False
    mesh: Mesh

    def __post_init__(self, rngs: nnx.Rngs):
        """Initializes weights, biases, and RoPE module."""

        self.sm_scale = 1.0 / (self.head_dim**0.5)

        self.sinks_N = create_param(
            rngs,
            shape=(self.num_attention_heads, ),
            dtype=jnp.float32,
            sharding=self.n_sharding,
            random_init=self.random_init,
        )

        # Q, K, V projection kernels
        self.kernel_q_DNH = create_param(
            rngs,
            shape=(self.hidden_size, self.num_attention_heads, self.head_dim),
            dtype=self.dtype,
            sharding=self.dnh_sharding,
            random_init=self.random_init,
        )
        self.bias_q_NH = create_param(
            rngs,
            shape=(self.num_attention_heads, self.head_dim),
            dtype=self.dtype,
            sharding=self.nh_sharding,
            random_init=self.random_init,
        )
        self.kernel_k_DKH = create_param(
            rngs,
            shape=(self.hidden_size, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            sharding=self.dkh_sharding,
            random_init=self.random_init,
        )
        self.bias_k_KH = create_param(
            rngs,
            shape=(self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            sharding=self.kh_sharding,
            random_init=self.random_init,
        )
        self.kernel_v_DKH = create_param(
            rngs,
            shape=(self.hidden_size, self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            sharding=self.dkh_sharding,
            random_init=self.random_init,
        )
        self.bias_v_KH = create_param(
            rngs,
            shape=(self.num_key_value_heads, self.head_dim),
            dtype=self.dtype,
            sharding=self.kh_sharding,
            random_init=self.random_init,
        )
        # Output projection kernel
        self.kernel_o_proj_NHD = create_param(
            rngs,
            shape=(self.num_attention_heads, self.head_dim, self.hidden_size),
            dtype=self.dtype,
            sharding=self.nhd_sharding,
            random_init=self.random_init,
        )
        self.bias_o_D = create_param(
            rngs,
            shape=(self.hidden_size, ),
            dtype=self.dtype,
            sharding=self.d_sharding,
            random_init=self.random_init,
        )

        # RoPE Module
        self.rope = GptOssRotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
            initial_context_length=self.initial_context_length,
            rope_scaling_factor=self.rope_scaling_factor,
            rope_ntk_alpha=self.rope_ntk_alpha,
            rope_ntk_beta=self.rope_ntk_beta)

    def attention(
        self,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_SKH: jax.Array,
        v_SKH: jax.Array,
        sinks: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention by calling the ragged_paged_attention kernel."""
        md = attention_metadata
        kv_cache_spec = P(None, None, "model")

        in_specs = (
            self.query_tnh,  # q
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # v
            kv_cache_spec,  # kv_cache
            P(),  # md.seq_lens: Replicated
            P(),  # page_indices_flat: Replicated
            P(),  # query_start_loc: Replicated
            P(),  # distribution: Replicated
            #P(('model')),  # sinks
        )
        out_specs = (self.attn_o_tnh, kv_cache_spec)

        def _ragged_paged_attention_wrapper(*args):
            # Pass the GPT-OSS specific parameters to the kernel
            return ragged_paged_attention(
                *args,
                sm_scale=self.sm_scale,
                sliding_window=md.sliding_window,
            )

        output_TNH, kv_cache = jax.jit(
            shard_map.shard_map(
                _ragged_paged_attention_wrapper,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            ))(
                q_TNH,
                k_SKH,
                v_SKH,
                kv_cache,
                md.seq_lens,
                md.block_tables,
                md.query_start_loc,
                md.request_distribution,
                #sinks,
            )
        return kv_cache, output_TNH

    def __call__(self,
                 x_TD,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Forward pass for the Attention module using 3D kernels."""
        md = attention_metadata
        x_TD = jnp.asarray(x_TD, self.dtype)

        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum("TD,DNH->TNH", x_TD, self.kernel_q_DNH.value)
            q_TNH += self.bias_q_NH.value

        with jax.named_scope("k_proj"):
            k_TKH = jnp.einsum("TD,DKH->TKH", x_TD, self.kernel_k_DKH.value)
            k_TKH += self.bias_k_KH.value

        with jax.named_scope("v_proj"):
            v_TKH = jnp.einsum("TD,DKH->TKH", x_TD, self.kernel_v_DKH.value)
            v_TKH += self.bias_v_KH.value

        if use_attention_rope:
            q_TNH, k_TKH = self.rope(q_TNH, k_TKH, md.input_positions)

        with jax.named_scope("attn_op"):
            # Padding H dim of q,k,v to be the multiple of 128
            multiple_of_128 = ((self.head_dim - 1) // 128 + 1) * 128
            q_TNH = jnp.pad(q_TNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.head_dim)))
            k_TKH = jnp.pad(k_TKH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.head_dim)))
            v_TKH = jnp.pad(v_TKH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.head_dim)))
            new_kv_cache, attn_out_TNH = self.attention(
                kv_cache, q_TNH, k_TKH, v_TKH, self.sinks_N.value, md,
                self.mesh)
            attn_out_TNH = attn_out_TNH[..., :self.head_dim]

        with jax.named_scope("o_proj"):
            output_TD = jnp.einsum("TNH,NHD->TD", attn_out_TNH,
                                   self.kernel_o_proj_NHD.value)
            output_TD += self.bias_o_D.value

        return new_kv_cache, output_TD
