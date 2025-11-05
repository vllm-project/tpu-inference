from dataclasses import InitVar, dataclass
from typing import NamedTuple, Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Sharding, Mesh, PartitionSpec as P

from tpu_inference import utils
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention_interface import sharded_flash_attention
from tpu_inference.layers.jax.attention.attention import Attention, KVCache
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.jax.base import create_param
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class L2Norm(nnx.Module):
    """
  Implementation of L2 Norm in JAX (taken from MaxText repo - maxtext/MaxText/layers/attentions.py).

  Attributes:
    eps: float, epsilon used for numerical stability (default value should be ok for most cases).
  """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x):
        return x * jax.lax.rsqrt(
            jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)


@dataclass(kw_only=True)
class Llama4Attention(Attention):
    use_qk_norm: bool
    temperature_tuning: bool
    temperature_tuning_floor_scale: float
    temperature_tuning_scale: float
    activation_attention_td: Sharding
    activation_attention_out_td: Sharding
    is_causal: bool = True

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True,
                 **kwargs):
        """Performs the forward pass of the attention module.

        This method computes the attention output by projecting the input `x`
        to queries, keys, and values, applying RoPE and L2Norm if specified,
        performing scaled dot-product attention, and projecting the results
        back to the model dimension.
        If no RoPE (NoPE) is specified, one can also perform temperature tuning
        which is useful to combat dilution of attention scores in long-context attention.

        Args:
            x: The input tensor of shape `(seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.
            use_attention_rope: Whether to use RoPE.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        md = attention_metadata
        x = jnp.asarray(x, self.dtype)

        # 1. Input to the attention block (same as layer input hidden states)
        # jax.debug.print("JAX Attention input hidden states slice: {}",
        #                 x[0, :5])

        x_SD = nnx.with_sharding_constraint(x, self.activation_attention_td)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td)
        rope_scaling = self.rope_scaling
        rope_theta = self.rope_theta
        H = self.head_dim
        l2_norm = L2Norm()

        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD,
                               self.kernel_q_proj_DNH.value)

            # # 2. Output of q_proj before RoPE
            # jax.debug.print("JAX q_proj output slice before RoPE: {}",
            #                 q_TNH[0, 0, :5])

            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H, rope_theta,
                                   rope_scaling, self.rope_input_ordering)

                # # 3. Output of q_proj after RoPE
                # jax.debug.print("JAX q_proj output after RoPE slice: {}",
                #                 q_TNH[0, 0, :5])

                # Apply normaliation after RoPE
                if self.use_qk_norm:
                    q_TNH = l2_norm(q_TNH)
                    # jax.debug.print("JAX q_proj output after L2Norm slice: {}",
                    #                 q_TNH[0, 0, :5])
            else:
                if self.temperature_tuning:
                    q_TNH = self.apply_temperature_tuning(md, q_TNH)

            q_TNH = nnx.with_sharding_constraint(q_TNH, self.query_tnh)
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_k_proj_DKH.value)
            # # 4. Output of k_proj before RoPE
            # jax.debug.print("JAX k_proj output slice before RoPE: {}",
            #                 k_SKH[0, 0, :5])
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H, rope_theta,
                                   rope_scaling, self.rope_input_ordering)
                # # 5. Output of k_proj after RoPE
                # jax.debug.print("JAX k_proj output after RoPE slice: {}",
                #                 k_SKH[0, 0, :5])

                # Apply normaliation after RoPE
                if self.use_qk_norm:
                    k_SKH = l2_norm(k_SKH)
                    # jax.debug.print("JAX k_proj output after L2Norm slice: {}",
                    #                 k_SKH[0, 0, :5])
            k_SKH = nnx.with_sharding_constraint(k_SKH, self.keyvalue_skh)

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_v_proj_DKH.value)
            # # 6. Output of v_proj
            # jax.debug.print("JAX v_proj output slice: {}", v_SKH[0, 0, :5])
            v_SKH = nnx.with_sharding_constraint(v_SKH, self.keyvalue_skh)
            # jax.debug.print(
            #     "JAX v_proj output slice after sharding constraint: {}",
            #     v_SKH[0, 0, :5])

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # TODO(kyuyeunk/jacobplatin): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k_SKH, v_SKH = utils.quantize_kv(k_SKH, v_SKH,
                                             self.kv_cache_quantized_dtype,
                                             k_scale, v_scale)

        if not self.is_causal and self.head_dim == 88:
            ACTUAL_HEAD_DIM = 88
            TARGET_HEAD_DIM = 128
            PAD_WIDTH = TARGET_HEAD_DIM - ACTUAL_HEAD_DIM
            #DUMMY_INT = jnp.zeros((1, ), dtype=jnp.int32)

            # 1. Pad Q, K, V from 88 to 128 to satisfy the kernel's static shape alignment
            q_TNH = jnp.pad(q_TNH, [(0, 0), (0, 0), (0,0), (0, PAD_WIDTH)],
                            mode='constant',
                            constant_values=0)
            k_SKH = jnp.pad(k_SKH, [(0, 0), (0, 0), (0, 0), (0, PAD_WIDTH)],
                            mode='constant',
                            constant_values=0)
            v_SKH = jnp.pad(v_SKH, [(0, 0), (0, 0), (0, PAD_WIDTH)],
                            mode='constant',
                            constant_values=0)

            # 2. Store original dim and temporarily set module's head dim for the kernel call
            original_head_dim = self.head_dim
            self.head_dim = TARGET_HEAD_DIM

            # 3. Inject DUMMY KV CACHE (Replaces kv_cache=None from VisionEncoderLayer)
            dummy_num_blocks = 1
            dummy_block_size = 2  # <-- INCREASED from 1 to 2 to be divisible by kv_packing=2

            # Create a zero-filled placeholder array
            kv_cache = jnp.zeros((dummy_num_blocks, dummy_block_size,
                                  self.num_key_value_heads, 2, self.head_dim),
                                 dtype=x.dtype)

            # 4. Inject DUMMY METADATA (Replaces None values for kernel's dtype check)
            # Assuming the AttentionMetadata object is mutable and md is a reference to it.
            # We check the fields expected by the kernel's static_validate_inputs (line 1153)
            # and replace them with minimal arrays of the expected dtype (jnp.int32).

            DUMMY_INT_1 = jnp.zeros((1, ), dtype=jnp.int32)

            # 1. Replace block_tables (corresponds to page_indices/page_indices_ref in kernel)
            if attention_metadata.block_tables is None:
                attention_metadata.block_tables = DUMMY_INT_1

            # 2. Replace seq_lens (corresponds to kv_lens/kv_lens_ref in kernel)
            if attention_metadata.seq_lens is None:
                attention_metadata.seq_lens = DUMMY_INT_1

            # 3. Replace query_start_loc (corresponds to cu_q_lens/cu_q_lens_ref in kernel)
            if attention_metadata.query_start_loc is None:
                # cu_q_lens typically needs shape (num_seqs + 1), minimal is (2,) for 1 sequence
                attention_metadata.query_start_loc = jnp.zeros((2, ),
                                                               dtype=jnp.int32)

            # 4. Replace distribution (the immediate cause of the current error)
            if attention_metadata.request_distribution is None:
                # distribution requires shape (3,). Since it's bidirectional, we set it to (0, 0, 1)
                # which indicates 1 sequence is active, but none are decode or prefill.
                attention_metadata.request_distribution = jnp.array(
                    [0, 0, 1], dtype=jnp.int32)

            # 5. Handle kv_lens that kernel checks but is not explicitly in AttentionMetadata
            # Since seq_lens is handled, we'll create a new attribute if the kernel expects it under 'kv_lens'.
            # NOTE: If your AttentionMetadata wrapper eventually unpacks 'seq_lens' into 'kv_lens',
            # this step might be unnecessary, but we include it as a fallback if 'kv_lens' is a required attribute of 'attention_metadata'.
            if not hasattr(attention_metadata,
                           'kv_lens') or attention_metadata.kv_lens is None:
                attention_metadata.kv_lens = DUMMY_INT_1  # Use DUMMY_INT_1 (shape (1,))

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SKH,
                v_SKH,
                attention_metadata,
                self.mesh,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                **kwargs
            )
        # The outputs_TNH variable is the core attention output, but before the final projection.
        # This is the "Attention Output (before projection)" from your last log.
        # jax.debug.print("JAX Attention output (before projection): {}",
        #                 outputs_TNH[0, -1, :5])

        if not self.is_causal and self.head_dim == TARGET_HEAD_DIM:
            # Crop the output back to the original size (88)
            outputs_TNH = outputs_TNH[..., :original_head_dim]

            # Restore the original head dimension
            self.head_dim = original_head_dim

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum('TNH,NHD -> TD', outputs_TNH,
                              self.kernel_o_proj_NHD.value)
            o_TD = nnx.with_sharding_constraint(
                o_TD, self.activation_attention_out_td)

        # # This is the "Attention Output (after projection)" which is what we compared previously.
        # jax.debug.print("JAX Attention output (after projection): {}",
        #                 o_TD[0, :5])

        return new_kv_cache, o_TD

    def apply_temperature_tuning(self, md: AttentionMetadata,
                                 input_arr_TNH: jax.Array) -> jax.Array:
        """Applies temperature tuning to the input array of shape (T, N, H).
        Args:
            md: AttentionMetadata object containing the input positions.
            input_arr_TNH: Input array of shape (T, N, H) which will have scaled temperatures applied.
        """
        attn_scales = (jnp.log(
            jnp.floor((md.input_positions.astype(self.dtype) + 1.0) /
                      self.temperature_tuning_floor_scale) + 1.0) *
                       self.temperature_tuning_scale + 1.0)
        return input_arr_TNH * attn_scales[:, None, None]


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

  SegmentIds are used to generate segment mask, which prevents attention between
  different segments in the input sequence. Each array is a list of ids
  (integers).
  Only the token with the same id can attend to each other.

  Attributes:
    q: segment ids along the Q sequence.
    kv: segment ids along the KV sequence.
  """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]

@dataclass(kw_only=True)
class Llama4VisionAttention(nnx.Module): # <--- Inherits from nnx.Module
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: Optional[dict[str, Any]]
    dtype: jnp.dtype
    mesh: Mesh
    use_qk_norm: bool
    temperature_tuning: bool
    temperature_tuning_floor_scale: float
    temperature_tuning_scale: float
    activation_attention_td: Sharding
    activation_attention_out_td: Sharding
    is_causal: bool = True
    
    # 1. ADD: The required InitVar for nnx initialization (used in the constructor)
    rngs: InitVar[nnx.Rngs] 
    
    # 2. ADD: The structural attributes needed for sharding and RoPE
    dnh_sharding: Sharding = ()
    dkh_sharding: Sharding = ()
    nhd_sharding: Sharding = ()
    activation_q_td: Sharding = ()
    query_tnh: P = P()
    keyvalue_skh: P = P()
    rope_input_ordering: str = "interleaved" # Vision config default
    
    # 3. ADD: Placeholder for scales (used in inject_weights, even if unused internally)
    _q_scale: float = 1.0
    _k_scale: float = 1.0
    _v_scale: float = 1.0
    
    # 4. ADD: The required __post_init__ to process rngs and create params
    def __post_init__(self, rngs: nnx.Rngs):
        """Initializes the weight kernels for Q, K, V, and O projections."""
        N = self.num_attention_heads
        K = self.num_key_value_heads
        D = self.hidden_size
        H = self.head_dim
        random_init = False # Weights are loaded from PyTorch

        self.kernel_q_proj_DNH = create_param(rngs, (D, N, H),
                                              self.dnh_sharding,
                                              self.dtype,
                                              random_init=random_init)
        self.kernel_k_proj_DKH = create_param(rngs, (D, K, H),
                                              self.dkh_sharding,
                                              self.dtype,
                                              random_init=random_init)
        self.kernel_v_proj_DKH = create_param(rngs, (D, K, H),
                                              self.dkh_sharding,
                                              self.dtype,
                                              random_init=random_init)
        self.kernel_o_proj_NHD = create_param(rngs, (N, H, D),
                                              self.nhd_sharding,
                                              self.dtype,
                                              random_init=random_init)
    def __call__(self,
                 x, # This is the hidden state (S, D) or (B*T, D)
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True,
                 **kwargs):
        
        md = attention_metadata
        x = jnp.asarray(x, self.dtype)
        x_SD = nnx.with_sharding_constraint(x, self.activation_attention_td)
        x_q_TD = nnx.with_sharding_constraint(x, self.activation_q_td)
        rope_scaling = self.rope_scaling
        rope_theta = self.rope_theta
        H = self.head_dim
        l2_norm = L2Norm()
        
        # 1. Input Projection and RoPE Application (Output is Rank 3: [S, N/K, H])
        # [ ... q_proj, k_proj, v_proj blocks omitted for brevity, they produce q_TNH, k_SKH, v_SKH ... ]
        # NOTE: Your existing code for Q, K, V projection and RoPE/L2Norm application must be kept here.
        # It results in: q_TNH, k_SKH, v_SKH (all padded to H=128)
        
        # --- START: RETAINED QKV PROJECTION LOGIC ---

        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD, self.kernel_q_proj_DNH.value)
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H, rope_theta,
                                   rope_scaling, self.rope_input_ordering)
                if self.use_qk_norm:
                    q_TNH = l2_norm(q_TNH)
            q_TNH = nnx.with_sharding_constraint(q_TNH, self.query_tnh)

        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD, self.kernel_k_proj_DKH.value)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H, rope_theta,
                                   rope_scaling, self.rope_input_ordering)
                if self.use_qk_norm:
                    k_SKH = l2_norm(k_SKH)
            k_SKH = nnx.with_sharding_constraint(k_SKH, self.keyvalue_skh)

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,DKH -> SKH', x_SD, self.kernel_v_proj_DKH.value)
            v_SKH = nnx.with_sharding_constraint(v_SKH, self.keyvalue_skh)
        
        # --- Flash Attention Migration Logic (Starts here, no change needed) ---
        
        #needed to slice these. Had an extra dimension replicated
        q_TNH = q_TNH[:, 0, :, :] 
        k_SKH = k_SKH[:, 0, :, :] 

        # *** DEBUG ADDITION (keep for sanity check) ***
        jax.debug.print("DEBUG: q_TNH shape AFTER slice: {}", q_TNH.shape)
        jax.debug.print("DEBUG: q_TNH rank AFTER slice: {}", q_TNH.ndim)
        # **********************************************

        # Check Q_TNH shape is now (T, N, H)
        T_attn, N, H = q_TNH.shape
        B = 1 # Batch size is 1 for the Vision Encoder (fixed)
        
        # Target block size for sequence dimension padding
        BLOCK_SIZE = 128
        pad_len = (BLOCK_SIZE - (T_attn % BLOCK_SIZE)) % BLOCK_SIZE

        # 2. Reshape to Flash Attention Input Format: [B, N, T, H]
        # a. Add Batch Axis and apply padding for the Sequence (T) dimension

        # Q Tensor (q_TNH is currently [T, N, H])
        q_TNH = jnp.pad(q_TNH, [(0, pad_len), (0, 0), (0, 0)], mode='constant', constant_values=0)
        q_TNH = jnp.expand_dims(q_TNH, axis=0) # [1, T_padded, N, H]

        # K Tensor (k_SKH is currently [T, K, H])
        k_SKH = jnp.pad(k_SKH, [(0, pad_len), (0, 0), (0, 0)], mode='constant', constant_values=0)
        k_SKH = jnp.expand_dims(k_SKH, axis=0) # [1, T_padded, K, H]

        # V Tensor (v_SKH is currently [T, K, H])
        v_SKH = jnp.pad(v_SKH, [(0, pad_len), (0, 0), (0, 0)], mode='constant', constant_values=0)
        v_SKH = jnp.expand_dims(v_SKH, axis=0) # [1, T_padded, K, H]

        # Update T_attn to the padded length
        T_padded = T_attn + pad_len 

        # b. Transpose T and N axes: [1, N, T_padded, H]
        q_BNTH = jnp.transpose(q_TNH, (0, 2, 1, 3))
        k_BKTH = jnp.transpose(k_SKH, (0, 2, 1, 3))
        v_BKTH = jnp.transpose(v_SKH, (0, 2, 1, 3))
        
        # 3. Generate Segment Ids (Simplified for Vision Encoder)
        segment_ids_q = jnp.full((B, T_padded), 0, dtype=jnp.int32)
        segment_ids = SegmentIds(q=segment_ids_q, kv=segment_ids_q)

        with jax.named_scope("flash_attn_op"):
            # Execute Flash Attention kernel
            outputs_BNTH = sharded_flash_attention(
                mesh=self.mesh,
                causal=False, 
                sm_scale=1.0, #self.head_dim**-0.5,
            )(q_BNTH, k_BKTH, v_BKTH, segment_ids)
            
            new_kv_cache = kv_cache
            
        # 4. Reverse Transpose and Reshape for Output Projection (NEW FIX 2)
        
        # a. Reverse Transpose: [B, N, T_padded, H] -> [T_padded, B, N, H]
        outputs_TBH = jnp.transpose(outputs_BNTH, (2, 0, 1, 3)) 

        # *** UNPAD/CROP: Remove the padded elements ***
        outputs_TBH = outputs_TBH[:T_attn, ...]

        # b. Squeeze the Batch=1 dimension to get TNH
        outputs_TNH = jnp.squeeze(outputs_TBH, axis=1) # [T, N, H]

        # 5. Output Projection (o_proj)
        with jax.named_scope("o_proj"):
            # Standard Llama/Attention einsum: [T, N, H] * [N, H, D] -> [T, D]
            o_TD = jnp.einsum('TNH,NHD -> TD', outputs_TNH,
                              self.kernel_o_proj_NHD.value)
            o_TD = nnx.with_sharding_constraint(
                o_TD, self.activation_attention_out_td)

        # We return the attention output in the 2D format expected by the outer layer loop
        return new_kv_cache, o_TD