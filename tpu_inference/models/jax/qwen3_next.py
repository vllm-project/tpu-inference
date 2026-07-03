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
"""JAX native implementation of Qwen3-Next (hybrid gated delta net linear
attention + gated full attention, sparse MoE with a shared expert)."""

import functools
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.gdn_attention import run_jax_gdn_attention
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear, JaxLmHead
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.moe.utils import (get_expert_parallelism,
                                                select_moe_backend)
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import (apply_rope,
                                                     get_rope_scaling,
                                                     get_rope_theta)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    JaxAutoWeightsLoader, LoadableWithIterator,
    load_nnx_param_from_reshaped_torch)
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

# Indirection over the fused Pallas conv1d + GDN kernel so CPU tests can
# substitute a pure JAX reference implementation.
_gdn_core = run_jax_gdn_attention


class Conv1dWeight(JaxModule):
    """Holds the depthwise causal conv weight under a `weight` param so its
    parameter name lines up with the HF checkpoint (`...conv1d.weight`)."""

    def __init__(self, conv_dim: int, kernel_size: int, dtype: jnp.dtype,
                 rng: nnx.Rngs):
        self.weight = create_param(rng, (conv_dim, 1, kernel_size),
                                   ("model", None, None), dtype)


class Qwen3NextGatedDeltaNet(JaxModule):
    """Gated delta net linear attention block.

    The projections, unpacking, output norm and gate run as plain JAX ops;
    the depthwise causal conv1d and the gated delta rule scan run in the
    fused GDN kernel, which also reads and writes the per request
    (conv_state, recurrent_state) cache slots."""

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 quant_config,
                 prefix: str = ""):
        self.mesh = mesh
        self.n_kq = config.linear_num_key_heads
        self.n_v = config.linear_num_value_heads
        self.d_k = config.linear_key_head_dim
        self.d_v = config.linear_value_head_dim
        self.kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.n_kq * self.d_k
        self.value_dim = self.n_v * self.d_v
        self.conv_dim = 2 * self.key_dim + self.value_dim
        hidden_size = config.hidden_size

        self.in_proj_qkvz = JaxEinsum(
            "TD,DP->TP",
            (hidden_size, 2 * self.key_dim + 2 * self.value_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".in_proj_qkvz",
        )
        self.in_proj_ba = JaxEinsum(
            "TD,DP->TP",
            (hidden_size, 2 * self.n_v),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".in_proj_ba",
        )
        self.conv1d = Conv1dWeight(self.conv_dim, self.kernel_size, dtype, rng)
        self.A_log = create_param(rng, (self.n_v, ), ("model", ), jnp.float32)
        self.dt_bias = create_param(rng, (self.n_v, ), ("model", ),
                                    jnp.float32)
        # Gated RMSNorm over d_v; standard scale (not the zero centered
        # variant the rest of the model uses). The silu(z) gate is applied
        # outside, matching HF's norm-before-gate ordering.
        self.norm = JaxRmsNorm(
            self.d_v,
            epsilon=config.rms_norm_eps,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".norm",
        )
        self.out_proj = JaxEinsum(
            "TP,PD->TD",
            (self.value_dim, hidden_size),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".out_proj",
        )

    def _split_qkvz_ba(self, qkvz: jax.Array, ba: jax.Array):
        """Unpacks the interleaved per key head projection layout, matching
        HF's fix_query_key_value_ordering."""
        t = qkvz.shape[0]
        r = self.n_v // self.n_kq
        qkvz = qkvz.reshape(t, self.n_kq, 2 * self.d_k + 2 * r * self.d_v)
        q, k, v, z = jnp.split(
            qkvz, [self.d_k, 2 * self.d_k, 2 * self.d_k + r * self.d_v],
            axis=-1)
        v = v.reshape(t, self.n_v, self.d_v)
        z = z.reshape(t, self.n_v, self.d_v)
        ba = ba.reshape(t, self.n_kq, 2 * r)
        b, a = jnp.split(ba, [r], axis=-1)
        b = b.reshape(t, self.n_v)
        a = a.reshape(t, self.n_v)
        return q, k, v, z, b, a

    def __call__(
        self,
        kv_cache: Tuple[jax.Array, jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        md = attention_metadata
        t = x.shape[0]
        q, k, v, z, b, a = self._split_qkvz_ba(self.in_proj_qkvz(x),
                                               self.in_proj_ba(x))
        mixed_qkv = jnp.concatenate([
            q.reshape(t, self.key_dim),
            k.reshape(t, self.key_dim),
            v.reshape(t, self.value_dim),
        ],
                                    axis=-1)

        # The kernel runs per TP shard; interleave the blocked [Q|K|V]
        # layout so each shard receives its own contiguous [Q|K|V] slice.
        tp_size = get_mesh_shape_product(self.mesh, ShardingAxisName.ATTN_HEAD)
        split_sizes = [self.key_dim, self.key_dim, self.value_dim]
        if tp_size > 1:
            mixed_qkv = reorder_concatenated_tensor_for_sharding(
                mixed_qkv, split_sizes, tp_size, 1)
            conv_weight = reorder_concatenated_tensor_for_sharding(
                self.conv1d.weight.value, split_sizes, tp_size, 0)
        else:
            conv_weight = self.conv1d.weight.value

        conv_state, recurrent_state = kv_cache
        # With speculative decoding the conv cache carries extra rows; the
        # kernel consumes exactly kernel_size - 1.
        conv_state_tail = None
        if conv_state.shape[1] > self.kernel_size - 1:
            conv_state_tail = conv_state[:, self.kernel_size - 1:, :]
            conv_state = conv_state[:, :self.kernel_size - 1, :]

        (new_conv_state, new_recurrent_state), core_out = _gdn_core(
            mixed_qkv,
            b,
            a,
            conv_state,
            recurrent_state,
            conv_weight,
            None,
            self.A_log.value,
            self.dt_bias.value,
            md.mamba_state_indices,
            md.query_start_loc,
            md.request_distribution,
            md.seq_lens,
            n_kq=self.n_kq,
            n_v=self.n_v,
            d_k=self.d_k,
            d_v=self.d_v,
            kernel_size=self.kernel_size,
            mesh=self.mesh,
        )

        if conv_state_tail is not None:
            new_conv_state = jnp.concatenate([new_conv_state, conv_state_tail],
                                             axis=1)

        core_out = core_out.reshape(t, self.n_v, self.d_v)
        gated = self.norm(core_out.astype(jnp.float32)) * jax.nn.silu(
            z.astype(jnp.float32))
        out = self.out_proj(gated.reshape(t, self.value_dim).astype(x.dtype))
        return (new_conv_state, new_recurrent_state), out


class Qwen3NextAttention(JaxModule):
    """Gated full attention block. The q projection produces the query and
    a per head sigmoid output gate in one fused kernel; rotary embedding is
    partial (first partial_rotary_factor * head_dim dims)."""

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = get_rope_theta(config, default=10000.0)
        self.rope_scaling = get_rope_scaling(config)
        self.rms_norm_eps = config.rms_norm_eps

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)
        self.rotary_dim = int(self.head_dim_original *
                              getattr(config, "partial_rotary_factor", 1.0))

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.mesh = mesh

        # The gate halves live fused inside q_proj, so the kernel layout is
        # fixed to DNH with a doubled head dim.
        self.q_proj = JaxEinsum(
            "TD,DNH->TNH",
            (self.hidden_size, self.num_heads, 2 * self.head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.q_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_norm",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.k_norm = JaxRmsNorm(
            self.head_dim,
            epsilon=self.rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_norm",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def _rope(self, x: jax.Array, positions: jax.Array) -> jax.Array:
        # apply_rope zero pads its output past the head_dim argument, so it
        # must only see the rotary slice; the remaining dims pass through.
        rotated = apply_rope(x[..., :self.rotary_dim], positions,
                             self.rotary_dim, self.rope_theta,
                             self.rope_scaling)
        return jnp.concatenate([rotated, x[..., self.rotary_dim:]], axis=-1)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        # (T, N, 2H): query half and output gate half per head.
        q, gate = jnp.split(self.q_proj(x), 2, axis=-1)
        q = self._rope(self.q_norm(q), md.input_positions)
        k = self._rope(self.k_norm(self.k_proj(x)), md.input_positions)
        v = self.v_proj(x)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        new_kv_cache, o = attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        o = o * jax.nn.sigmoid(gate)
        return new_kv_cache, self.o_proj(o)


class Qwen3NextRouter(JaxLinear):
    """MoE router that adapts its output to the backend: raw logits for the
    fused kernels (which route internally), (weights, indices) for the
    unfused ones. Top-k followed by softmax over the selected logits equals
    HF's softmax then top-k then renormalize (norm_topk_prob=True)."""

    def __init__(self, *args, num_experts_per_tok: int,
                 moe_backend: MoEBackend, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_backend = moe_backend

    def __call__(self, x: jax.Array):
        logits = super().__call__(x)
        if self.moe_backend in MoEBackend.fused_moe_backends():
            return logits
        weights, selected = jax.lax.top_k(logits, self.num_experts_per_tok)
        return jax.nn.softmax(weights, axis=-1), selected


class Qwen3NextMoeExperts(JaxMoE):
    """JaxMoE that also accepts the fused 3D expert tensors some Qwen3-Next
    checkpoints store (`experts.gate_up_proj` (E, 2F, D) with the gate rows
    first, `experts.down_proj` (E, D, F)) next to the per expert layout the
    base class handles. Fused tensors are expanded into per expert shards
    and delegated so the fused kernel post processing sees the layout it
    expects; for the unfused backends the kernels are transposed into the
    (E, D, F) / (E, F, D) orientation the dense matmul consumes directly."""

    def _load_weights(self, weights, *, mesh: Mesh | None = None):
        expanded = []
        for name, torch_weight in weights:
            if name.endswith("experts.gate_up_proj"):
                intermediate = torch_weight.shape[1] // 2
                for expert_id in range(torch_weight.shape[0]):
                    expanded.append(
                        (f"{self.prefix}.{expert_id}.gate_proj.weight",
                         torch_weight[expert_id, :intermediate, :]))
                    expanded.append(
                        (f"{self.prefix}.{expert_id}.up_proj.weight",
                         torch_weight[expert_id, intermediate:, :]))
            elif name.endswith("experts.down_proj"):
                for expert_id in range(torch_weight.shape[0]):
                    expanded.append(
                        (f"{self.prefix}.{expert_id}.down_proj.weight",
                         torch_weight[expert_id]))
            else:
                expanded.append((name, torch_weight))

        loaded = super()._load_weights(expanded, mesh=mesh)

        if self.moe_backend not in MoEBackend.fused_moe_backends():
            # The base loader keeps the HF (out, in) orientation per expert,
            # which the fused backends fix up in their weight post
            # processing. The dense fallback consumes the kernels as is, so
            # orient them here once all experts have arrived.
            for param_name in loaded:
                param = getattr(self, param_name)
                param.set_value(jnp.transpose(param.get_value(), (0, 2, 1)))
        return loaded


class Qwen3NextMLP(JaxModule):
    """Plain silu MLP used for the shared expert."""

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 quant_config,
                 prefix: str = ""):
        self.gate_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".gate_proj",
        )
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".up_proj",
        )
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3NextSparseMoeBlock(JaxModule):
    """Sparse MoE block with a sigmoid gated shared expert on top of the
    routed experts."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config
        hidden_size = config.hidden_size

        edf_sharding = (None, None, None)
        expert_axis_name = edf_sharding[0]
        num_expert_parallelism = get_expert_parallelism(expert_axis_name, mesh)
        use_ep = num_expert_parallelism > 1
        moe_backend = select_moe_backend(use_ep)

        self.gate = Qwen3NextRouter(
            hidden_size,
            config.num_experts,
            dtype=dtype,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".gate",
            num_experts_per_tok=config.num_experts_per_tok,
            moe_backend=moe_backend,
        )

        self.enable_return_routed_experts = True
        self.experts = Qwen3NextMoeExperts(
            dtype=dtype,
            num_local_experts=config.num_experts,
            hidden_size=hidden_size,
            intermediate_size_moe=config.moe_intermediate_size,
            hidden_act=config.hidden_act,
            rngs=rng,
            router=self.gate,
            num_experts_per_tok=config.num_experts_per_tok,
            mesh=mesh,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            activation_ffw_ted=P(ShardingAxisName.MLP_DATA, None, None),
            edf_sharding=P(None, ),
            efd_sharding=P(None, ),
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            moe_backend=moe_backend,
            quant_config=quant_config,
            enable_return_routed_experts=self.enable_return_routed_experts,
            prefix=prefix + ".experts")

        self.shared_expert = Qwen3NextMLP(
            hidden_size,
            config.shared_expert_intermediate_size,
            dtype=dtype,
            rng=rng,
            quant_config=quant_config,
            prefix=prefix + ".shared_expert",
        )
        self.shared_expert_gate = JaxLinear(
            hidden_size,
            1,
            dtype=dtype,
            rngs=rng,
            use_bias=False,
            quant_config=quant_config,
            prefix=prefix + ".shared_expert_gate",
        )

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, Optional[jax.Array]]:
        out, expert_ids = self.experts(x)
        shared = self.shared_expert(x)
        out = out + jax.nn.sigmoid(self.shared_expert_gate(x)) * shared
        return out, expert_ids


class Qwen3NextDecoderLayer(JaxModule):

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config,
                 layer_idx: int,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(
                config=config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                quant_config=quant_config,
                prefix=prefix + ".linear_attn",
            )
        else:
            self.self_attn = Qwen3NextAttention(
                config=config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=kv_cache_dtype,
                quant_config=quant_config,
                prefix=prefix + ".self_attn",
            )
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config,
                                               rng=rng,
                                               mesh=mesh,
                                               prefix=prefix + ".mlp")
        else:
            raise NotImplementedError(
                "Dense MLP layers are not implemented; all Qwen3-Next "
                f"checkpoints use MoE on every layer. Found {mlp_only_layers=}"
                f", {config.num_experts=}, {config.decoder_sparse_step=}.")

    def __call__(
        self,
        kv_cache,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ):
        hidden_states = self.input_layernorm(x)
        if self.layer_type == "linear_attention":
            kv_cache, attn_output = self.linear_attn(kv_cache, hidden_states,
                                                     attention_metadata)
        else:
            kv_cache, attn_output = self.self_attn(kv_cache, hidden_states,
                                                   attention_metadata)
        attn_output += x

        residual = attn_output
        attn_output = self.post_attention_layernorm(attn_output)
        mlp_output, expert_ids = self.mlp(attn_output)
        outputs = residual + mlp_output

        return kv_cache, outputs, expert_ids


class Qwen3NextModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_text_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        tp_size = (vllm_config.parallel_config.tensor_parallel_size
                   if vllm_config.parallel_config is not None else 1)
        padded_vocab_size = utils.align_to(vocab_size, tp_size)

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=padded_vocab_size,
                features=hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: Qwen3NextDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                layer_idx=layer_index,
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches,
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ):
        if self.is_first_rank:
            assert inputs_embeds is None
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None

        x = inputs_embeds
        new_kv_caches = []
        all_expert_ids = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue
            kv_cache, x, expert_ids = layer(kv_caches[i], x,
                                            attention_metadata)
            if expert_ids is not None:
                all_expert_ids.append(expert_ids)
            new_kv_caches.append(kv_cache)

        if self.is_last_rank:
            x = self.norm(x)

        stacked_expert_ids = jnp.stack(all_expert_ids,
                                       axis=0) if all_expert_ids else None
        return new_kv_caches, x, stacked_expert_ids


def _load_zero_centered_norm(jax_param: nnx.Param,
                             torch_weight,
                             *,
                             param_name: str = "Unknown"):
    """Qwen3NextRMSNorm is zero centered (`out = norm(x) * (1 + w)`); fold
    the +1 into the scale at load time so the standard RMSNorm applies."""
    folded = (torch_weight.float() + 1.0).to(torch_weight.dtype)
    load_nnx_param_from_reshaped_torch(jax_param,
                                       folded,
                                       param_name=param_name)


def _load_float32_param(jax_param: nnx.Param,
                        torch_weight,
                        *,
                        param_name: str = "Unknown"):
    """A_log and dt_bias participate in exp/softplus and are kept in
    float32 regardless of the checkpoint dtype."""
    load_nnx_param_from_reshaped_torch(jax_param,
                                       torch_weight.float(),
                                       param_name=param_name)


# Norms folded with +1 at load time. The gated norm inside linear_attn uses
# a standard scale and is excluded.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


class Qwen3NextForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Qwen3NextModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                tp_size = (vllm_config.parallel_config.tensor_parallel_size
                           if vllm_config.parallel_config is not None else 1)
                padded_vocab_size = utils.align_to(vocab_size, tp_size)
                hidden_size = model_config.hf_config.hidden_size
                self.lm_head = JaxLmHead(
                    hidden_size=hidden_size,
                    vocab_size=padded_vocab_size,
                    dtype=model_config.dtype,
                    param_dtype=model_config.dtype,
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()
        else:
            self.lm_head = PPMissingLayer()

        for name, param in self.named_parameters():
            if (name.endswith(_ZERO_CENTERED_NORM_SUFFIXES)
                    or name == "model.norm.weight"):
                param.set_metadata(
                    "weight_loader",
                    functools.partial(_load_zero_centered_norm,
                                      param_name=name))
            elif name.endswith((".A_log", ".dt_bias")):
                param.set_metadata(
                    "weight_loader",
                    functools.partial(_load_float32_param, param_name=name))

    def load_weights(self, weights) -> set:
        # The checkpoint ships an MTP tower (`mtp.*`) that the base model
        # does not use. Names get a `model.` prefix inside the loader when
        # they do not start with one, so skip both spellings.
        loader = JaxAutoWeightsLoader(self,
                                      skip_prefixes=["mtp.", "model.mtp."])
        return loader.load_weights(weights)

    def __call__(
        self,
        kv_caches,
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, x, expert_indices = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )
        return kv_caches, x, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self,
                   'lm_head') and not isinstance(self.lm_head, PPMissingLayer):
            return self.lm_head(hidden_states)

        assert isinstance(self.model.embed_tokens, JaxEmbed)
        return self.model.embed_tokens.decode(hidden_states)
