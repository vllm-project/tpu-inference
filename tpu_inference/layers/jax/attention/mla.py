# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared Multi-head Latent Attention (MLA) layers.

Extracted from the DeepSeek-V3 model file so other MLA-family models can
reuse it. Beyond DeepSeek-V3 this supports the Kimi-Linear / Kimi-K3 family
(HF ``moonshotai/Kimi-Linear-48B-A3B-Instruct``, ``moonshotai/Kimi-K3``),
which differ in three config-driven ways:

* ``use_nope``: no rotary embedding is applied anywhere. The
  ``qk_rope_head_dim`` slice still exists and is still MQA-shared through the
  cache, it simply carries un-rotated values.
* ``use_output_gate``: ``attn_out *= sigmoid(g_proj(x))`` before ``o_proj``.
* ``q_lora_rank=None``: the query is produced by a single ``q_proj`` instead
  of the ``q_a_proj`` / ``q_a_layernorm`` / ``q_b_proj`` low-rank stack.

All three default to the DeepSeek-V3 behaviour, so existing callers are
unaffected.
"""

import math
from abc import abstractmethod
from dataclasses import InitVar, dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference import utils
from tpu_inference.kernels.quantized_matmul.util import quantize_tensor
from tpu_inference.layers.common.attention_interface import mla_attention
from tpu_inference.layers.common.quantization import (
    dequantize_tensor, quantize_kv, static_per_tensor_quantize_tensor)
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.attention.attention import AttentionMetadata
from tpu_inference.layers.jax.base import _init_fn as init_fn
from tpu_inference.layers.jax.base import sharded_initializer
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.layers.jax.rope import DeepseekScalingRotaryEmbedding
from tpu_inference.models.jax.utils.weight_utils import shard_put

KVCache = Tuple[jax.Array, jax.Array]


def _weight_init(random_init: bool):
    return sharded_initializer if random_init else nnx.initializers.uniform()


@dataclass(kw_only=True)
class MLABaseAttention(JaxModule):
    """
    Base class containing shared logic for MLA-family attention mechanisms.
    Handles initialization of common layers and defines skeleton forward pass.
    """
    # Core configuration
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    dtype: jnp.dtype
    kv_cache_dtype: str
    mesh: Mesh

    # Attention-specific configuration
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float

    # ``None`` selects a single fused q_proj instead of the low-rank
    # q_a_proj / q_a_layernorm / q_b_proj stack (Kimi-Linear-48B).
    q_lora_rank: Optional[int] = None

    # Rotary embedding. Required unless ``use_nope`` is set.
    rope: Optional[DeepseekScalingRotaryEmbedding] = None
    # Skip rotary entirely; the qk_rope slice carries un-rotated values and
    # stays MQA-shared through the cache (Kimi-Linear / Kimi-K3).
    use_nope: bool = False
    # attn_out *= sigmoid(g_proj(x)) before o_proj (Kimi-K3).
    use_output_gate: bool = False

    # Sharding
    rd_sharding: P = P()
    q_da_sharding: P = P()
    ap_sharding: P = P()
    kv_da_sharding: P = P()
    activation_attention_td: P = P()
    activation_q_td: P = P()
    query_nth: P = P()
    query_tnh: P = P()
    keyvalue_skh: P = P()
    attn_o_nth: P = P()
    activation_attention_out_td: P = P()
    # Weight initialization
    random_init: bool = False
    rope_mscale_all_dim: float = 1.0

    # RNG for weight initialization
    rngs: InitVar[nnx.Rngs]

    quant_config: Optional[QuantizationConfig] = None

    # Scales for Q/KV quantization (per-tensor)
    _q_scale: float = 1
    _k_scale: float = 1
    _v_scale: float = 1

    prefix: str = ""

    def __post_init__(self, rngs: nnx.Rngs):
        self.N = self.num_attention_heads
        self.K = self.num_key_value_heads
        self.D = self.hidden_size
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.use_nope:
            # No rotary => no YaRN mscale correction; plain 1/sqrt(d) scale.
            assert self.rope is None or self.use_nope
            self.scale = self.qk_head_dim**-0.5
        else:
            assert self.rope is not None, (
                "[mla] rope must be provided unless use_nope=True")
            if self.rope.scaling_factor <= 1.0:
                yarn_mscale = 1.0
            else:
                yarn_mscale = 0.1 * self.rope_mscale_all_dim * math.log(
                    self.rope.scaling_factor) + 1.0

            self.scale = self.qk_head_dim**-0.5 * yarn_mscale**2

        weight_init = _weight_init(self.random_init)

        if self.q_lora_rank is not None:
            self.q_a_proj = JaxEinsum(
                einsum_str="TD,DA->TA",
                kernel_shape=(self.D, self.q_lora_rank),
                rngs=rngs,
                quant_config=self.quant_config,
                param_dtype=self.dtype,
                kernel_init=nnx.with_partitioning(weight_init,
                                                  self.q_da_sharding),
                prefix=self.prefix + ".q_a_proj",
            )

            self.q_b_proj = JaxEinsum(
                einsum_str="TA,AP->TP",
                kernel_shape=(self.q_lora_rank, self.N * self.qk_head_dim),
                rngs=rngs,
                quant_config=self.quant_config,
                param_dtype=self.dtype,
                kernel_init=nnx.with_partitioning(weight_init,
                                                  self.ap_sharding),
                prefix=self.prefix + ".q_b_proj")
        else:
            self.q_proj = JaxEinsum(
                einsum_str="TD,DP->TP",
                kernel_shape=(self.D, self.N * self.qk_head_dim),
                rngs=rngs,
                quant_config=self.quant_config,
                param_dtype=self.dtype,
                kernel_init=nnx.with_partitioning(weight_init,
                                                  self.ap_sharding),
                prefix=self.prefix + ".q_proj")

        self.kv_a_proj_with_mqa = JaxEinsum(
            einsum_str="SD,DA->SA",
            kernel_shape=(self.D, self.kv_lora_rank + self.qk_rope_head_dim),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init,
                                              self.kv_da_sharding),
            prefix=self.prefix + ".kv_a_proj_with_mqa")

        self.o_proj = JaxEinsum(
            einsum_str="TR,RD->TD",
            kernel_shape=(self.N * self.v_head_dim, self.D),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.rd_sharding),
            prefix=self.prefix + ".o_proj")

        # NOTE: parameter creation order below is load-bearing. `nnx.Rngs` is
        # stateful, so reordering these constructors changes the random weights
        # a `random_init=True` module gets. Keep q_a_layernorm here (after
        # o_proj) to stay identical to the pre-extraction DeepSeek-V3 layer.
        if self.q_lora_rank is not None:
            self.q_a_layernorm = JaxRmsNorm(
                self.q_lora_rank,
                epsilon=self.rms_norm_eps,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                param_dtype=self.dtype,
                dtype=self.dtype,
                quant_config=self.quant_config,
                prefix=self.prefix + ".q_a_layernorm",
                rngs=rngs)

        if self.use_output_gate:
            self.g_proj = JaxEinsum(
                einsum_str="TD,DR->TR",
                kernel_shape=(self.D, self.N * self.v_head_dim),
                rngs=rngs,
                quant_config=self.quant_config,
                param_dtype=self.dtype,
                kernel_init=nnx.with_partitioning(weight_init,
                                                  self.rd_sharding),
                prefix=self.prefix + ".g_proj")

        self.kv_a_layernorm = JaxRmsNorm(
            self.kv_lora_rank,
            epsilon=self.rms_norm_eps,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            param_dtype=self.dtype,
            dtype=self.dtype,
            quant_config=self.quant_config,
            prefix=self.prefix + ".kv_a_layernorm",
            rngs=rngs)

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                self.kv_cache_dtype)

        self.kv_b_proj = JaxEinsum(
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.N * (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(init_fn, self.ap_sharding),
            prefix=self.prefix + ".kv_b_proj",
        )

    def _project_q(self, x_q_TD: jax.Array) -> jax.Array:
        """Query projection up to (but excluding) the head reshape."""
        if self.q_lora_rank is not None:
            return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x_q_TD)))
        return self.q_proj(x_q_TD)

    @abstractmethod
    def compute_q_projection(self, *args) -> jax.Array:
        raise NotImplementedError

    @abstractmethod
    def compute_kv_projection(self, *args) -> Tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    @abstractmethod
    def compute_attention(self, *args) -> Tuple[KVCache, jax.Array]:
        raise NotImplementedError

    def process_output(self, outputs_TNH) -> jax.Array:
        return outputs_TNH

    def __call__(
            self, x: jax.Array, kv_cache: KVCache,
            attention_metadata: AttentionMetadata
    ) -> Tuple[KVCache, jax.Array]:
        """Performs the forward pass of the attention module.  Expects that the
        child class has implemented the `compute_q_projection`, `compute_kv_projection`,
        and `compute_attention` methods.

        Args:
            x: The input tensor of shape `(batch_size, seq_len, d_model)`.
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
        x_SD = lax.with_sharding_constraint(x, self.activation_attention_td)
        x_q_TD = lax.with_sharding_constraint(x, self.activation_q_td)

        with jax.named_scope("q_proj"):
            q_data = self.compute_q_projection(x_q_TD, md.input_positions)

        with jax.named_scope("kv_proj"):
            kv_data = self.compute_kv_projection(x_SD, md.input_positions)

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.compute_attention(
                q_data, kv_data, kv_cache, md)

            outputs_TNH = self.process_output(outputs_TNH)

            if outputs_TNH.shape[-1] != self.v_head_dim:
                outputs_TNH = outputs_TNH[..., :self.v_head_dim]

            with jax.named_scope("o_proj"):
                outputs_TR = outputs_TNH.reshape(outputs_TNH.shape[0],
                                                 self.N * self.v_head_dim)
                if self.use_output_gate:
                    # Sigmoid computed in fp32 to match the reference
                    # implementation's numerics.
                    gate_TR = jax.nn.sigmoid(
                        self.g_proj(x_q_TD).astype(jnp.float32))
                    outputs_TR = (outputs_TR.astype(jnp.float32) *
                                  gate_TR).astype(self.dtype)
                o_TD = self.o_proj(outputs_TR)

            return new_kv_cache, o_TD


# The absorbed-MLA forward runs on `k_up_proj`/`v_up_proj`, which `MLAEinsum`
# splits out of the checkpoint's fused `kv_b_proj` at load time. No checkpoint
# tensor carries their names, so they never show up as "loaded".
_ABSORBED_MLA_DERIVED_MODULES = (".k_up_proj.", ".v_up_proj.")


def derived_absorbed_mla_param_names(model) -> set:
    """Parameter names an MLA model materializes instead of loading.

    vLLM's model loader raises if any `named_parameters()` entry is missing
    from the set a model's `load_weights` returns. Quantized MLA checkpoints
    get a pass because their `quant_method` declares
    `process_weights_after_loading`, which the loader whitelists wholesale; an
    unquantized MLA checkpoint (Kimi-Linear-48B is bf16 throughout, Kimi-K3
    keeps `self_attn.*` in bf16) has no such method, so the model has to
    report these names itself.
    """
    return {
        name
        for name, _ in model.named_parameters()
        if any(part in name for part in _ABSORBED_MLA_DERIVED_MODULES)
    }


class MLAEinsum(JaxEinsum):
    """Extending JaxEinsum to handle MLA.

    This class is used for MLA, where:
    1) the weight is split into k/v parts after loading, and
    2) modify the MLA layer to set k/v weights
    """

    def __init__(self,
                 mla_layer,
                 einsum_str: str,
                 kernel_shape: tuple[int, ...],
                 rngs,
                 bias_shape: Optional[tuple[int, ...]] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        super().__init__(einsum_str,
                         kernel_shape,
                         rngs,
                         bias_shape=bias_shape,
                         quant_config=quant_config,
                         prefix=prefix,
                         **kwargs)
        self.loaded = set()
        self.mla_layer = mla_layer
        self.quant_config = quant_config

    def named_children(self):
        # Override, otherwise "mla_layer" will be visited, causing infinite recursion.
        yield from []

    def _split_unquantized(self):
        """bf16 path: split the loaded kv_b_proj weight into k_up / v_up.

        The absorbed-MLA forward needs ``k_up_proj``/``v_up_proj`` rather than
        the fused ``kv_b_proj``; for quantized checkpoints that split happens
        below in the fp8 branch. Models whose MLA weights are not quantized
        (Kimi-Linear-48B is bf16 throughout; Kimi-K3 quantizes only the routed
        experts and leaves ``self_attn.*`` in bf16) need the same split without
        any (de)quantization.
        """
        mla_layer = self.mla_layer
        A = mla_layer.kv_lora_rank
        N = mla_layer.N
        qk_nope_head_dim = mla_layer.qk_nope_head_dim
        v_head_dim = mla_layer.v_head_dim
        weight = self.weight.value
        if weight.shape != (A, N * (qk_nope_head_dim + v_head_dim)):
            raise ValueError(
                f"[mla] unexpected kv_b_proj shape {weight.shape}, expected "
                f"{(A, N * (qk_nope_head_dim + v_head_dim))} "
                f"(kv_lora_rank={A}, num_heads={N}, "
                f"qk_nope_head_dim={qk_nope_head_dim}, "
                f"v_head_dim={v_head_dim})")
        weight = weight.reshape(A, N, qk_nope_head_dim + v_head_dim)
        k_ANH, v_ANH = jnp.split(weight, [qk_nope_head_dim], axis=-1)

        setattr(
            mla_layer, "k_up_proj",
            JaxEinsum(
                einsum_str="TNH,ANH->NTA",
                kernel_shape=(A, N, qk_nope_head_dim),
                rngs=nnx.Rngs(0),
                param_dtype=self.weight.value.dtype,
                prefix=mla_layer.prefix + ".k_up_proj",
                quant_config=None,
            ))
        setattr(
            mla_layer, "v_up_proj",
            JaxEinsum(
                einsum_str="NTA,ANH->TNH",
                kernel_shape=(A, N, v_head_dim),
                rngs=nnx.Rngs(0),
                param_dtype=self.weight.value.dtype,
                prefix=mla_layer.prefix + ".v_up_proj",
                quant_config=None,
            ))
        mla_layer.k_up_proj.weight.value = shard_put(k_ANH,
                                                     mla_layer.anh_sharding)
        mla_layer.v_up_proj.weight.value = shard_put(v_ANH,
                                                     mla_layer.anh_sharding)
        delattr(self, 'weight')

    def load_weights(self, weights) -> set:
        """Load `kv_b_proj`, then split it, and report what was loaded.

        Returns the names of the parameters consumed **in this call**, module-
        local (``"weight"``, ``"weight_scale_inv"``); the caller qualifies them
        with this module's prefix. The return value is bookkeeping only -- it
        does not change what is loaded -- but it is not optional: vLLM's
        `AutoWeightsLoader._load_module` calls a child module's `load_weights`
        and, when it returns None, logs "Unable to collect loaded parameters
        for module %s" at WARNING with the module's full repr interpolated,
        once per MLA layer on every load that goes through the auto-loader.

        The derived `k_up_proj`/`v_up_proj` params are deliberately not
        reported here: no checkpoint tensor carries their names and they hang
        off the parent `MLAAttention`, not off this module, so the model
        reports them via `derived_absorbed_mla_param_names`.
        """
        named_params = dict(self.named_parameters())
        if len(self.loaded) >= 2:
            raise ValueError(
                f"Expect at most 2 params to load for kv_b_proj, already got {self.loaded}, still have {[name for name, _ in weights]} coming."
            )
        loaded_now: set = set()
        for name, weight in weights:
            param = named_params[name]
            weight_loader = getattr(param, "weight_loader")
            weight_loader(param, weight)
            self.loaded.add(name)
            loaded_now.add(name)
        if len(self.loaded) != len(named_params):
            return loaded_now
        # Presence of the scale param -- not of a quant_config -- is what
        # decides the path: an unquantized model still carries a
        # QuantizationConfig (UnquantizedConfig), it just never creates
        # `weight_scale_inv`. Kimi-Linear-48B is bf16 throughout and Kimi-K3
        # keeps `self_attn.*` in bf16, so both land here.
        if not hasattr(self, "weight_scale_inv"):
            self._split_unquantized()
            return loaded_now
        # After loading, split the weights into k/v
        with cpu_mesh_context():
            dequantized_weight = dequantize_tensor(
                self.weight,
                self.weight_scale_inv,
                (0, 1),
                block_size=None,
            )
            A, N, qk_nope_head_dim, v_head_dim = self.mla_layer.kv_lora_rank, self.mla_layer.N, self.mla_layer.qk_nope_head_dim, self.mla_layer.v_head_dim
            if dequantized_weight.shape != (A, N *
                                            (qk_nope_head_dim + v_head_dim)):
                raise ValueError(
                    f"Unexpected weight shape after dequantization: {dequantized_weight.shape}, expected {(A, N * (qk_nope_head_dim + v_head_dim))=}"
                )
            dequantized_weight = dequantized_weight.reshape(
                A, N, qk_nope_head_dim + v_head_dim)
            k_ANH, v_ANH = jnp.split(dequantized_weight, [qk_nope_head_dim],
                                     axis=-1)
            k_ANH_weight, k_ANH_scale = quantize_tensor(k_ANH,
                                                        self.weight.dtype,
                                                        dim=-1)
            v_ANH_weight, v_1NH_scale = quantize_tensor(v_ANH,
                                                        self.weight.dtype,
                                                        dim=0)
            # As of writing, sharded_quantized_batched_matmul expects scale to be
            # a different shape order than weight
            k_N1A_scale = k_ANH_scale.transpose(1, 2, 0)
            v_N1H_scale = v_1NH_scale.transpose(1, 0, 2)
        mla_layer = self.mla_layer
        setattr(
            mla_layer, "k_up_proj",
            JaxEinsum(
                einsum_str="TNH,ANH->NTA",
                kernel_shape=(A, N, qk_nope_head_dim),
                rngs=nnx.Rngs(0),
                prefix=mla_layer.prefix + ".k_up_proj",
                quant_config=self.quant_config,
            ))
        setattr(
            mla_layer, "v_up_proj",
            JaxEinsum(
                einsum_str="NTA,ANH->TNH",
                kernel_shape=(A, N, v_head_dim),
                rngs=nnx.Rngs(0),
                prefix=mla_layer.prefix + ".v_up_proj",
                quant_config=self.quant_config,
            ))
        # Cannot apply anh_sharding to scales, otherwise it complains about shape mismatch.
        mla_layer.k_up_proj.weight.value = shard_put(
            k_ANH_weight, self.mla_layer.anh_sharding)
        mla_layer.k_up_proj.weight_scale_inv.value = shard_put(k_N1A_scale, ())
        mla_layer.v_up_proj.weight.value = shard_put(
            v_ANH_weight, self.mla_layer.anh_sharding)
        mla_layer.v_up_proj.weight_scale_inv.value = shard_put(v_N1H_scale, ())

        delattr(self, 'weight')
        delattr(self, 'weight_scale_inv')
        delattr(self, 'quant_method')
        return loaded_now


@dataclass(kw_only=True)
class MLAAttention(MLABaseAttention):
    """Absorbed Multi-head Latent Attention (MLA)."""
    anh_sharding: Sharding = ()

    def __post_init__(self, rngs: nnx.Rngs):
        super().__post_init__(rngs)

        weight_init = _weight_init(self.random_init)
        self.kv_b_proj = MLAEinsum(
            mla_layer=self,
            einsum_str="SA,AL->SL",
            kernel_shape=(self.kv_lora_rank,
                          self.N * (self.qk_nope_head_dim + self.v_head_dim)),
            rngs=rngs,
            quant_config=self.quant_config,
            param_dtype=self.dtype,
            kernel_init=nnx.with_partitioning(weight_init, self.ap_sharding),
            prefix=self.prefix + ".kv_b_proj",
        )

    def compute_q_projection(
            self, x_q_TD: jax.Array,
            input_positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the query projection for MLA.

        Args:
            x_q_TD: The input tensor of shape `(tokens_query, d_model)`.
            input_positions: The input positions tensor of shape `(padded_total_num_scheduled_tokens,)`.

        Returns:
            A tuple of query tensor of shape `(tokens_query, num_query_heads, q_lora_rank)` and
            rope tensor of shape `(tokens_query, num_query_heads, head_dim)`.
        """
        q_TP = self._project_q(x_q_TD)
        q_TNH = q_TP.reshape(q_TP.shape[0], self.N, self.qk_head_dim)

        q_nope_TNH = q_TNH[..., :self.qk_nope_head_dim]
        q_rope_TNH = q_TNH[..., self.qk_nope_head_dim:]
        if not self.use_nope:
            q_rope_TNH = self.rope.apply_rope(input_positions, q_rope_TNH)

        q_NTA = self.k_up_proj(q_nope_TNH)

        q_NTA = lax.with_sharding_constraint(q_NTA, self.query_nth)
        return (q_NTA, q_rope_TNH)

    def compute_kv_projection(
            self, x_SD: jax.Array,
            input_positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """
        Computes the key-value projection for MLA.

        Args:
            x_SD: The input tensor of shape `(tokens_kv, d_model)`.
            input_positions: The input positions tensor of shape `(padded_total_num_scheduled_tokens,)`.

        Returns:
            A tuple of key-value tensor of shape `(tokens_kv, q_lora_rank)` and
            rope tensor of shape `(tokens_kv, head_dim)`.
        """
        kv_SA = self.kv_a_proj_with_mqa(x_SD)

        k_rope_SH = kv_SA[..., self.kv_lora_rank:]
        if not self.use_nope:
            k_rope_SNH = k_rope_SH[..., None, :]
            k_rope_SNH = self.rope.apply_rope(input_positions, k_rope_SNH)
            assert k_rope_SNH.shape[1] == 1
            k_rope_SH = k_rope_SNH[:, 0, :]

        kv_SA = kv_SA[..., :self.kv_lora_rank]
        kv_SA = self.kv_a_layernorm(kv_SA)
        kv_SA = lax.with_sharding_constraint(kv_SA, self.keyvalue_skh)

        return (kv_SA, k_rope_SH)

    def compute_attention(self, q_data: Tuple[jax.Array, jax.Array],
                          kv_data: Tuple[jax.Array,
                                         jax.Array], kv_cache: KVCache,
                          md: AttentionMetadata) -> Tuple[KVCache, jax.Array]:
        """
        Computes the attention for MLA.

        Args:
            q_data: A tuple of query tensor of shape `(tokens_query, num_query_heads, q_lora_rank)` and
                rope tensor of shape `(tokens_query, num_query_heads, head_dim)`.
            kv_data: A tuple of key-value tensor of shape `(tokens_kv, q_lora_rank)` and
                rope tensor of shape `(tokens_kv, head_dim)`.
            kv_cache: The key-value cache.
            md: The attention metadata.

        Returns:
            A tuple of key-value cache and output tensor of shape `(tokens_query, num_query_heads, q_lora_rank)`.
        """

        q_NTA, q_rope_TNH = q_data
        k_SA, k_rope_SH = kv_data

        q_scale = k_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            # TODO: May need to apply quantization separately for k_c & k_pe
            k_SA, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                  k_SA,
                                  value=None,
                                  k_scale=k_scale)
            k_rope_SH, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                       k_rope_SH,
                                       value=None,
                                       k_scale=k_scale)
            q_NTA = static_per_tensor_quantize_tensor(
                self.kv_cache_quantized_dtype, q_NTA, self._q_scale)
            q_rope_TNH = static_per_tensor_quantize_tensor(
                self.kv_cache_quantized_dtype, q_rope_TNH, self._q_scale)

        return mla_attention(q_NTA,
                             q_rope_TNH,
                             k_SA,
                             k_rope_SH,
                             kv_cache,
                             md,
                             self.mesh,
                             self.num_attention_heads,
                             self.qk_nope_head_dim,
                             query_nth_sharding=self.query_nth,
                             query_tnh_sharding=self.query_tnh,
                             keyvalue_skh_sharding=self.keyvalue_skh,
                             attn_o_nth_sharding=self.attn_o_nth,
                             q_scale=q_scale,
                             k_scale=k_scale,
                             v_scale=k_scale,
                             sm_scale=self.scale)

    def process_output(self, outputs_NTA: jax.Array) -> jax.Array:
        """
        Processes output for MLA specifically.

        Args:
            outputs_NTA: The output tensor of shape `(num_heads, tokens_query, q_lora_rank)`.

        Returns:
            The processed output tensor of shape `(tokens_query, num_query_heads, head_dim)`.
        """

        # MLA Specific: Apply V-Up Projection after attention
        # Outputs from MLA kernel are N-major [N, T, A]; project to T-major [T, N, H]
        outputs_TNH = self.v_up_proj(outputs_NTA)
        return outputs_TNH.astype(self.dtype)
