import math
from dataclasses import InitVar, dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.base import create_param
from tpu_commons.models.jax.common.layers import RMSNorm
from tpu_commons.models.jax.common.rope import DeepseekScalingRotaryEmbedding

KVCache = Tuple[jax.Array, jax.Array]


# TODO (wenxindongwork): Add MLA KV cache implementation. For now, cache complete KV vectors.
@dataclass(kw_only=True)
class MLA(nnx.Module):
    """An implementation of Multi-Head Latent Attention as
    described in the DeepSeek V3 paper.

    Attributes:
        mesh: The JAX device mesh for distributed computation.
    """
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: dict[str, Any]
    dtype: jnp.dtype
    mesh: Mesh

    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float

    # Sharding attributes
    nhd_sharding: Sharding = ()
    q_da_sharding: Sharding = ()
    anh_sharding: Sharding = ()
    kv_da_sharding: Sharding = ()

    activation_attention_td: Sharding = ()
    activation_q_td: Sharding = ()
    query_tnh: P = P()
    keyvalue_skh: P = P()

    attn_o_tnh: P = P()
    activation_attention_out_td: Sharding = ()

    random_init: bool = False
    attention_chunk_size: int | None = None
    rope_input_ordering: str = "split"
    quant: Any | None = None
    rope_mscale_all_dim: float = 1.0

    rngs: InitVar[nnx.Rngs]

    def __post_init__(self, rngs: nnx.Rngs):
        self.N = self.num_attention_heads
        self.K = self.num_key_value_heads
        self.D = self.hidden_size

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        assert self.N == self.K, "N and K must be equal for MLA"

        if self.rope_scaling["factor"] <= 1.0:
            yarn_mscale = 1.0
        else:
            yarn_mscale = 0.1 * self.rope_mscale_all_dim * math.log(
                self.rope_scaling["factor"]) + 1.0
        self.scale = self.qk_head_dim**-0.5 * yarn_mscale**2

        self.rope = DeepseekScalingRotaryEmbedding(
            rotary_dim=self.qk_rope_head_dim,
            rope_theta=self.rope_theta,
            original_max_position_embeddings=self.
            rope_scaling["original_max_position_embeddings"],
            scaling_factor=self.rope_scaling["factor"],
            dtype=self.dtype,
            beta_fast=self.rope_scaling["beta_fast"],
            beta_slow=self.rope_scaling["beta_slow"],
            mscale=self.rope_scaling["mscale"],
            mscale_all_dim=self.rope_scaling["mscale_all_dim"],
        )

        # Initializes the weight kernels
        self.kernel_q_down_proj_DA = create_param(rngs,
                                                  (self.D, self.q_lora_rank),
                                                  self.q_da_sharding,
                                                  self.dtype,
                                                  random_init=self.random_init)
        self.kernel_q_up_proj_ANH = create_param(
            rngs,
            (self.q_lora_rank, self.N, self.qk_head_dim),
            self.anh_sharding,
            self.dtype,
            random_init=self.random_init,
        )
        self.kernel_kv_down_proj_DA = create_param(
            rngs,
            (self.D, self.kv_lora_rank + self.qk_rope_head_dim),
            self.kv_da_sharding,
            self.dtype,
            random_init=self.random_init,
        )
        self.kernel_kv_up_proj_ANH = create_param(
            rngs,
            (self.kv_lora_rank, self.N,
             self.qk_nope_head_dim + self.v_head_dim),
            self.anh_sharding,
            self.dtype,
            random_init=self.random_init,
        )
        self.kernel_o_proj_NHD = create_param(
            rngs, (self.N, self.v_head_dim, self.D),
            self.nhd_sharding,
            self.dtype,
            random_init=self.random_init)
        self.q_rms_norm = RMSNorm(
            dims=self.q_lora_rank,
            epsilon=self.rms_norm_eps,
            with_scale=True,
            dtype=self.dtype,
            random_init=self.random_init,
            rngs=rngs,
        )

        self.kv_rms_norm = RMSNorm(
            dims=self.kv_lora_rank,
            random_init=self.random_init,
            epsilon=self.rms_norm_eps,
            with_scale=True,
            dtype=self.dtype,
            rngs=rngs,
        )

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Performs the forward pass of the attention module.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        md = attention_metadata
        x = jnp.asarray(x, self.dtype)
        x_SD = nnx.with_sharding_constraint(x, self.activation_attention_td)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td)

        with jax.named_scope("q_proj"):
            # Query down projection.
            q_TA = jnp.einsum("TD,DA -> TA", x_q_TD,
                              self.kernel_q_down_proj_DA.value)
            q_TA = self.q_rms_norm(q_TA)
            # Query up projection.
            q_TNH = jnp.einsum("TA,ANH -> TNH", q_TA,
                               self.kernel_q_up_proj_ANH.value)
            # Split the query into nope and rope.
            q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
            q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
            q_rope_TNH = self.rope.apply_rope(md.input_positions, q_rope_TNH)
            # Concatenate the nope and rope queries.
            q_TNH = jnp.concatenate([q_nope_TNH, q_rope_TNH], axis=-1)
            # Multiple the query by scaling factor
            q_TNH = nnx.with_sharding_constraint(q_TNH, self.query_tnh)

        with jax.named_scope("kv_proj"):
            # KV down projection.
            kv_SA = jnp.einsum("SD,DA -> SA", x_SD,
                               self.kernel_kv_down_proj_DA.value)
            # Split the key and value into latent kv vector and k rope vector.
            k_rope_SH = kv_SA[..., self.kv_lora_rank:]
            # Reshape k_rope_BSH to include head dimension for RoPE application
            k_rope_SNH = k_rope_SH[..., None, :]
            k_rope_SNH = self.rope.apply_rope(md.input_positions, k_rope_SNH)
            k_rope_SNH = jnp.broadcast_to(
                k_rope_SNH,
                (k_rope_SNH.shape[0], self.N, self.qk_rope_head_dim))
            kv_SA = kv_SA[..., :self.kv_lora_rank]
            kv_SA = self.kv_rms_norm(kv_SA)
            # KV up projection.
            kv_nope_SNH = jnp.einsum("SA,ANH -> SNH", kv_SA,
                                     self.kernel_kv_up_proj_ANH.value)
            # Split the latent kv vector into k nope vector and v vector.
            k_nope_SNH = kv_nope_SNH[..., :self.qk_nope_head_dim]
            v_SNH = kv_nope_SNH[..., self.qk_nope_head_dim:]
            # Concatenate the key vector.
            k_SNH = jnp.concatenate([k_nope_SNH, k_rope_SNH], axis=-1)
            k_SNH = nnx.with_sharding_constraint(k_SNH, self.keyvalue_skh)
            v_SNH = nnx.with_sharding_constraint(v_SNH, self.keyvalue_skh)

        with jax.named_scope("attn_op"):
            # TODO(wenxindongwork): K and V have different head dimension,
            # which is not supported by the current kv cache implementation.
            # For now we are padding the v dimension to match the k dimension.
            # Furthermore, deepseekv3 k head dimension is 192, which is
            # not supported by the current attention kernel, which expects
            # q, k, v head dimension to be multiple of 128. For now, we will
            # pad the q, k, v dimension to multiple of 128.
            # We should update the MLA kv cache implementation in the future.
            multiple_of_128 = ((self.qk_head_dim - 1) // 128 + 1) * 128
            q_TNH = jnp.pad(q_TNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.qk_head_dim)))
            k_SNH = jnp.pad(k_SNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.qk_head_dim)))
            v_SNH = jnp.pad(v_SNH, ((0, 0), (0, 0),
                                    (0, multiple_of_128 - self.v_head_dim)))
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SNH,
                v_SNH,
                attention_metadata,
                self.mesh,
            )
            # TODO(wenxindongwork): For now, unpad the outputs_TNH to match the v_head_dim.
            # We shall add the MLA kv cache implementation in the future.
            outputs_TNH = outputs_TNH[..., :self.v_head_dim]

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum("TNH,NHD -> TD", outputs_TNH,
                              self.kernel_o_proj_NHD.value)
            o_TD = nnx.with_sharding_constraint(
                o_TD, self.activation_attention_out_td)
        return new_kv_cache, o_TD

    def attention(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_SKH: jax.Array,
        v_SKH: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention and updates the KV cache.

        This function handles the core attention logic, which varies between
        prefill and generation modes. In prefill, it computes self-attention
        over the input sequence with a causal mask. In generation, it attends
        to the full history of keys and values stored in the cache.

        Args:
            is_prefill: A boolean indicating if the mode is 'prefill'.
            kv_cache: The key-value cache to be updated and used.
            q_TNH: Query tensor of shape `(query_seq, num_attention_heads, head_dim)`.
            k_SKH: Key tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            v_SKH: Value tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            attention_metadata: Metadata containing sequence lengths.
            mesh: The JAX device mesh (unused in this specific function but
                kept for potential future use or API consistency).

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(seq, num_q_heads, head_dim)`.
        """
        md = attention_metadata
        in_specs = (
            self.query_tnh,  # q
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # v
            P(None, None, "model"),  # kv_cache
            P(),  # md.seq_lens: Replicated
            P(),  # page_indices_flat: Replicated
            P(),  # query_start_loc: Replicated
            P(),  # distribution: Replicated
        )
        out_specs = (self.attn_o_tnh, P(None, None, "model"))

        def _ragged_paged_attention(*args):
            return ragged_paged_attention(
                *args,
                sm_scale=self.scale,
            )

        output_TNH, kv_cache = jax.jit(
            shard_map.shard_map(
                _ragged_paged_attention,
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
            )
        return kv_cache, output_TNH
