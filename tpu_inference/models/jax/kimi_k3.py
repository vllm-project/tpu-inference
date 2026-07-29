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
"""Kimi-Linear / Kimi-K3 JAX text model.

One model file serves the whole family (HF ``moonshotai/Kimi-Linear-48B-A3B-
Instruct`` and ``moonshotai/Kimi-K3``); every difference is read off the
config rather than branched on the checkpoint name:

===========================  ==================  ==========================
config field                 Kimi-Linear-48B     Kimi-K3
===========================  ==================  ==========================
hidden_act                   silu                situ (+ the two betas)
attn_res_block_size          absent              12   (attention residuals)
routed_expert_hidden_size    absent              3584 (latent MoE)
linear_attn_config
  .gate_lower_bound          absent (softplus)   -5.0 (sigmoid lower bound)
  .use_full_rank_gate        absent (g_a/g_b)    true (g_proj)
mla_use_output_gate          absent              true
q_lora_rank                  null                1536
===========================  ==================  ==========================

Layer stack: ``linear_attn_config.kda_layers`` (1-indexed) selects the KDA
linear-attention layers, the rest are gated NoPE MLA. The first
``first_k_dense_replace`` layers have a dense MLP, the rest are MoE.

Reference: the HF repos' ``modeling_kimi_linear.py`` (``KimiDecoderLayer``,
``KimiSparseMoeBlock``, ``KimiMoEGate``, ``_apply_attn_res``), the Kimi Linear
paper (arXiv:2510.26692) for KDA, and the attention-residuals report
(arXiv:2603.15031).

Scope: text-only. Vision-tower and multimodal-projector weights are skipped by
the loader and image requests are rejected.
"""

import functools
from dataclasses import InitVar, dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.kda_attention import (KDAParams, KDAState,
                                                       kda_attention)
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import \
    ShardingAxisNameBase as ShardingAxisName
from tpu_inference.layers.jax import JaxModule, JaxModuleList
from tpu_inference.layers.jax.activation import SITU_BETA, SITU_LINEAR_BETA
from tpu_inference.layers.jax.attention.mla import MLAAttention
from tpu_inference.layers.jax.base import _init_fn as init_fn
from tpu_inference.layers.jax.base import create_param, sharded_initializer
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLmHead
from tpu_inference.layers.jax.mlp import GatedMLP
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3Router
from tpu_inference.models.jax.utils.weight_utils import (
    JaxAutoWeightsLoader, LoadableWithIterator,
    load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

# Checkpoint prefixes the text-only path does not implement. Kept as an
# explicit skip list (rather than ignoring unknown weights wholesale) so a
# genuinely unexpected tensor still fails loudly.
VISION_SKIP_PREFIXES = (
    "vision_tower",
    "model.vision_tower",
    "multi_modal_projector",
    "model.multi_modal_projector",
    "mm_projector",
    "model.mm_projector",
)

# Routed-expert weight names in the Kimi checkpoints vs the canonical names
# `JaxMoE._load_weights` expects.
_EXPERT_PARAM_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

# Params whose checkpoint layout is NOT "2-D [out, in], transpose it", as
# (name suffix, reshape_dims, permute_dims).
_LOADER_LAYOUT_OVERRIDES = (
    # The HF embedding is [vocab, hidden] and so is ours.
    ("embed_tokens.weight", None, (0, 1)),
    # Depthwise conv weight is [channels, 1, kernel_size] on both sides.
    ("conv1d.weight", None, (0, 1, 2)),
    # Per-head KDA decay. Kimi-K3 stores it as [H], Kimi-Linear-48B as
    # [1, 1, H, 1]; flattening accepts both and is a no-op for the former.
    ("self_attn.A_log", (-1, ), None),
)


def _weight_init(random_init: bool):
    return sharded_initializer if random_init else nnx.initializers.uniform()


def attach_default_weight_loaders(model: JaxModule) -> None:
    """Give every parameter an explicit checkpoint loader.

    ``JaxAutoWeightsLoader`` otherwise infers a reshape/permute from the
    parameter name, and its rules are written for MHA models: a name ending in
    ``q_proj.weight`` / ``o_proj.weight`` is assumed to be a 3-D
    ``[heads, head_dim, hidden]`` parameter and unpacking its shape raises on
    anything else. Every Kimi projection is a plain 2-D matrix, so state the
    layout instead of letting it be inferred.
    """
    for name, param in model.named_parameters():
        if hasattr(param, "weight_loader"):
            # Set by a quant method, which owns the layout for that param.
            continue
        reshape_dims = permute_dims = None
        for suffix, reshape, permute in _LOADER_LAYOUT_OVERRIDES:
            if name.endswith(suffix):
                reshape_dims, permute_dims = reshape, permute
                break
        param.set_metadata(
            "weight_loader",
            functools.partial(load_nnx_param_from_reshaped_torch,
                              reshape_dims=reshape_dims,
                              permute_dims=permute_dims,
                              param_name=name))


class KimiConfig:
    """Flattened view of a Kimi text config, with the family defaults applied.

    Reading the config through one object keeps the "Kimi-Linear-48B leaves
    this out" knowledge in a single place instead of scattering
    ``getattr(cfg, ..., default)`` calls through the modules.
    """

    def __init__(self, hf_config: Any):
        # KimiK3ForConditionalGeneration nests the text stack.
        self.hf_config = getattr(hf_config, "text_config", hf_config)
        c = self.hf_config

        self.hidden_size: int = c.hidden_size
        self.intermediate_size: int = c.intermediate_size
        self.num_hidden_layers: int = c.num_hidden_layers
        self.vocab_size: int = c.vocab_size
        self.rms_norm_eps: float = c.rms_norm_eps
        self.hidden_act: str = c.hidden_act
        self.situ_beta: float = getattr(c, "activation_situ_beta", SITU_BETA)
        self.situ_linear_beta: float = getattr(c,
                                               "activation_situ_linear_beta",
                                               SITU_LINEAR_BETA)

        # --- MLA ---
        self.num_attention_heads: int = c.num_attention_heads
        self.num_key_value_heads: int = c.num_key_value_heads
        self.q_lora_rank: Optional[int] = getattr(c, "q_lora_rank", None)
        self.kv_lora_rank: int = c.kv_lora_rank
        self.qk_nope_head_dim: int = c.qk_nope_head_dim
        self.qk_rope_head_dim: int = c.qk_rope_head_dim
        self.v_head_dim: int = c.v_head_dim
        self.mla_use_nope: bool = getattr(c, "mla_use_nope", False)
        self.mla_use_output_gate: bool = getattr(c, "mla_use_output_gate",
                                                 False)

        # --- KDA ---
        linear_cfg = getattr(c, "linear_attn_config", None) or {}
        if not isinstance(linear_cfg, dict):
            linear_cfg = vars(linear_cfg)
        # `kda_layers` is 1-indexed in the checkpoint config.
        self.kda_layers: set = {
            i - 1
            for i in linear_cfg.get("kda_layers", ())
        }
        self.kda_num_heads: int = linear_cfg.get("num_heads",
                                                 self.num_attention_heads)
        self.kda_head_dim: int = linear_cfg.get("head_dim", self.v_head_dim)
        self.kda_conv_kernel_size: int = linear_cfg.get(
            "short_conv_kernel_size", 4)
        # Absent => the -exp(A_log)*softplus decay form (Kimi-Linear-48B).
        self.kda_gate_lower_bound: Optional[float] = linear_cfg.get(
            "gate_lower_bound", None)
        self.kda_use_full_rank_gate: bool = linear_cfg.get(
            "use_full_rank_gate", False)

        # --- MoE ---
        self.num_experts: int = getattr(c, "num_experts", 0)
        self.num_experts_per_token: int = getattr(c, "num_experts_per_token",
                                                  1)
        self.moe_intermediate_size: int = getattr(c, "moe_intermediate_size",
                                                  0)
        self.num_shared_experts: Optional[int] = getattr(
            c, "num_shared_experts", None)
        self.first_k_dense_replace: int = getattr(c, "first_k_dense_replace",
                                                  0)
        self.moe_layer_freq: int = getattr(c, "moe_layer_freq", 1)
        self.moe_renormalize: bool = getattr(c, "moe_renormalize", True)
        self.moe_router_activation_func: str = getattr(
            c, "moe_router_activation_func", "sigmoid")
        self.num_expert_group: int = getattr(c, "num_expert_group", 1)
        self.topk_group: int = getattr(c, "topk_group", 1)
        self.routed_scaling_factor: float = getattr(c, "routed_scaling_factor",
                                                    1.0)
        # Absent => routed experts run at full hidden width (Kimi-Linear-48B).
        self.routed_expert_hidden_size: Optional[int] = getattr(
            c, "routed_expert_hidden_size", None)
        self.latent_moe_use_norm: bool = getattr(c, "latent_moe_use_norm",
                                                 False)

        # --- Attention residuals ---
        # Absent => the plain residual stream (Kimi-Linear-48B).
        self.attn_res_block_size: Optional[int] = getattr(
            c, "attn_res_block_size", None)

    @property
    def use_attn_residuals(self) -> bool:
        return self.attn_res_block_size is not None

    @property
    def moe_hidden_size(self) -> int:
        """Width the routed experts operate at (the latent dim for Kimi-K3)."""
        return self.routed_expert_hidden_size or self.hidden_size

    def is_kda_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.kda_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.num_experts == 0 or layer_idx < self.first_k_dense_replace:
            return False
        return (layer_idx + 1) % self.moe_layer_freq == 0

    def is_attn_res_checkpoint(self, layer_idx: int) -> bool:
        return (self.use_attn_residuals
                and layer_idx % self.attn_res_block_size == 0)


def apply_attn_res(prefix_sum_TD: jax.Array,
                   block_residual: Sequence[jax.Array],
                   proj_weight_D1: jax.Array, norm_weight_D: jax.Array,
                   eps: float) -> jax.Array:
    """Mix the depth checkpoints with the running prefix sum.

    A 1-dimensional learned pseudo-query attends over the candidate vectors
    ``{block_residual..., prefix_sum}``: each candidate is RMS-normalized
    (the norm's scale is *not* applied inside the norm), scored against
    ``norm.weight * proj.weight``, and the softmax over those scores mixes the
    un-normalized candidates. Computed in fp32, matching the reference
    (``modeling_kimi_linear.py::_apply_attn_res``).
    """
    v_TBD = jnp.stack(tuple(block_residual) + (prefix_sum_TD, ), axis=1)
    v_f32 = v_TBD.astype(jnp.float32)
    variance = jnp.mean(v_f32 * v_f32, axis=-1, keepdims=True)
    k_TBD = v_f32 * lax.rsqrt(variance + eps)
    score_weight_D = (norm_weight_D.astype(jnp.float32) *
                      proj_weight_D1.reshape(-1).astype(jnp.float32))
    scores_TB = jnp.sum(k_TBD * score_weight_D, axis=-1)
    probs_TB = jax.nn.softmax(scores_TB, axis=-1)
    # The mix replaces the residual stream, so it is computed at full fp32
    # precision rather than the TPU default (which costs ~4 digits of
    # agreement per layer here); the contraction is over <=8 depth
    # checkpoints, so this is free.
    out_TD = jnp.einsum('TB,TBD->TD',
                        probs_TB,
                        v_f32,
                        precision=lax.Precision.HIGHEST)
    return out_TD.astype(prefix_sum_TD.dtype)


def build_attn_res_site(*, hidden_size: int, dtype: jnp.dtype,
                        rms_norm_eps: float, rngs: nnx.Rngs, random_init: bool,
                        prefix: str) -> Tuple[JaxEinsum, JaxRmsNorm]:
    """Build one attention-residual mix site: its ``(proj, norm)`` pair.

    The two tensors are returned rather than wrapped in a module because the
    checkpoint names them as siblings (``<site>_res_proj.weight`` /
    ``<site>_res_norm.weight``), and the recursive weight loader resolves
    names by attribute path.
    """
    proj = JaxEinsum(
        einsum_str="TD,DO->TO",
        kernel_shape=(hidden_size, 1),
        rngs=rngs,
        quant_config=None,
        param_dtype=dtype,
        kernel_init=nnx.with_partitioning(_weight_init(random_init), P()),
        prefix=prefix + "_proj",
    )
    norm = JaxRmsNorm(hidden_size,
                      epsilon=rms_norm_eps,
                      scale_init=nnx.with_partitioning(init_fn, (None, )),
                      param_dtype=dtype,
                      dtype=dtype,
                      quant_config=None,
                      prefix=prefix + "_norm",
                      rngs=rngs)
    return proj, norm


class _RawWeight(JaxModule):
    """A module owning a single ``weight`` parameter.

    Used for checkpoint entries like ``self_attn.q_conv1d.weight`` that are
    plain tensors rather than a linear layer.
    """

    def __init__(self, shape: Tuple[int, ...], dtype: jnp.dtype,
                 rngs: nnx.Rngs, sharding: P, random_init: bool):
        self.weight = create_param(rngs,
                                   shape=shape,
                                   dtype=dtype,
                                   sharding=sharding,
                                   random_init=random_init)


@dataclass(kw_only=True)
class KimiDeltaAttention(JaxModule):
    """KDA (Kimi Delta Attention) linear-attention layer.

    Owns the checkpoint parameters and delegates the math to
    ``layers/common/kda_attention.py``. Its state (three conv windows plus the
    recurrent state) is threaded through the ``kv_cache`` slot exactly like an
    attention layer's KV cache, so the decoder loop stays uniform.

    NOTE(sharding): the KDA parameters are replicated. The linear-attention
    math is head-parallel and will be sharded on ``ATTN_HEAD`` together with
    the hybrid-KV-cache work; correctness comes first here.
    """
    hidden_size: int
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    rms_norm_eps: float
    dtype: jnp.dtype
    use_full_rank_gate: bool
    gate_lower_bound: Optional[float] = None
    random_init: bool = False
    quant_config: Optional[QuantizationConfig] = None
    prefix: str = ""

    rngs: InitVar[nnx.Rngs]

    def __post_init__(self, rngs: nnx.Rngs):
        D = self.hidden_size
        HP = self.num_heads * self.head_dim
        K = self.head_dim
        weight_init = _weight_init(self.random_init)

        def _einsum(name: str, shape: Tuple[int, int]) -> JaxEinsum:
            return JaxEinsum(
                einsum_str="TD,DP->TP",
                kernel_shape=shape,
                rngs=rngs,
                quant_config=self.quant_config,
                param_dtype=self.dtype,
                kernel_init=nnx.with_partitioning(weight_init, P()),
                prefix=f"{self.prefix}.{name}",
            )

        def _conv() -> _RawWeight:
            return _RawWeight(shape=(HP, 1, self.conv_kernel_size),
                              dtype=self.dtype,
                              rngs=rngs,
                              sharding=P(),
                              random_init=self.random_init)

        self.q_proj = _einsum("q_proj", (D, HP))
        self.k_proj = _einsum("k_proj", (D, HP))
        self.v_proj = _einsum("v_proj", (D, HP))
        self.q_conv1d = _conv()
        self.k_conv1d = _conv()
        self.v_conv1d = _conv()
        self.f_a_proj = _einsum("f_a_proj", (D, K))
        self.f_b_proj = _einsum("f_b_proj", (K, HP))
        self.b_proj = _einsum("b_proj", (D, self.num_heads))
        if self.use_full_rank_gate:
            self.g_proj = _einsum("g_proj", (D, HP))
        else:
            self.g_a_proj = _einsum("g_a_proj", (D, K))
            self.g_b_proj = _einsum("g_b_proj", (K, HP))
        self.o_proj = _einsum("o_proj", (HP, D))
        self.A_log = create_param(rngs,
                                  shape=(self.num_heads, ),
                                  dtype=jnp.float32,
                                  sharding=P(),
                                  random_init=self.random_init)
        self.dt_bias = create_param(rngs,
                                    shape=(HP, ),
                                    dtype=jnp.float32,
                                    sharding=P(),
                                    random_init=self.random_init)
        self.o_norm = JaxRmsNorm(K,
                                 epsilon=self.rms_norm_eps,
                                 scale_init=nnx.with_partitioning(
                                     init_fn, (None, )),
                                 param_dtype=self.dtype,
                                 dtype=self.dtype,
                                 quant_config=None,
                                 prefix=self.prefix + ".o_norm",
                                 rngs=rngs)

    def _params(self) -> KDAParams:
        """Assemble the functional parameter bundle.

        ``kda_attention`` takes torch-style ``[out, in]`` matrices (it computes
        ``x @ w.T``) while these params are stored transposed, hence ``.T``.
        """

        def _t(layer: JaxEinsum) -> jax.Array:
            # Weights keep their checkpoint dtype; the layer decides the
            # compute dtype (they differ when a bf16 checkpoint is run in fp32
            # for parity checks, and `lax.conv_general_dilated` rejects mixed
            # dtypes).
            return layer.weight.value.astype(self.dtype).T

        gate_kwargs = {}
        if self.use_full_rank_gate:
            gate_kwargs["g_proj"] = _t(self.g_proj)
        else:
            gate_kwargs["g_a_proj"] = _t(self.g_a_proj)
            gate_kwargs["g_b_proj"] = _t(self.g_b_proj)
        return KDAParams(
            q_proj=_t(self.q_proj),
            k_proj=_t(self.k_proj),
            v_proj=_t(self.v_proj),
            q_conv_weight=self.q_conv1d.weight.value.astype(self.dtype),
            k_conv_weight=self.k_conv1d.weight.value.astype(self.dtype),
            v_conv_weight=self.v_conv1d.weight.value.astype(self.dtype),
            # The decay is computed in fp32 in the reference regardless of the
            # checkpoint dtype.
            A_log=self.A_log.value.astype(jnp.float32),
            dt_bias=self.dt_bias.value.astype(jnp.float32),
            f_a_proj=_t(self.f_a_proj),
            f_b_proj=_t(self.f_b_proj),
            b_proj=_t(self.b_proj),
            o_norm_weight=self.o_norm.weight.value.astype(self.dtype),
            o_proj=_t(self.o_proj),
            **gate_kwargs)

    def __call__(self, x_TD: jax.Array, state: KDAState,
                 md: AttentionMetadata) -> Tuple[KDAState, jax.Array]:
        query_lens = md.query_start_loc[1:] - md.query_start_loc[:-1]
        has_initial_state = (md.seq_lens - query_lens) > 0
        return kda_attention(x_TD,
                             self._params(),
                             state,
                             md.mamba_state_indices,
                             md.query_start_loc,
                             md.request_distribution,
                             has_initial_state,
                             num_heads=self.num_heads,
                             head_dim=self.head_dim,
                             kernel_size=self.conv_kernel_size,
                             gate_lower_bound=self.gate_lower_bound,
                             rms_norm_eps=self.rms_norm_eps)


class KimiRoutedExperts(JaxMoE):
    """Routed experts, with the Kimi checkpoint's ``w1``/``w2``/``w3`` naming.

    NOTE(backend): Kimi-K3 runs this on an unfused MoE backend. The fused GMM
    kernels take the raw router logits and select experts internally, which
    drops the ``e_score_correction_bias`` that this model's noaux_tc routing
    depends on, and they have no ``situ`` activation branch. When the perf
    phase moves K3 onto GMM, the bias must be passed explicitly through
    ``extra_backend_kwargs["e_score_correction_bias"]``.
    """

    def _load_weights(self, weights: Iterable, **kwargs):

        def adapted():
            for name, weight in weights:
                parts = name.split(".")
                if len(parts) >= 2 and parts[-2] in _EXPERT_PARAM_MAP:
                    parts[-2] = _EXPERT_PARAM_MAP[parts[-2]]
                    name = ".".join(parts)
                # `JaxMoE._load_weights` only prepends the expert axis, it does
                # not transpose, while the expert kernels are declared
                # `(E, in, out)` and the checkpoint stores `[out, in]`.
                if weight.ndim == 2:
                    weight = weight.transpose(0, 1)
                yield name, weight

        return super()._load_weights(adapted(), **kwargs)


class KimiSparseMoeBlock(JaxModule):
    """Kimi MoE block: routed experts (optionally latent) + shared experts.

    For Kimi-K3 the routed path is a *latent* MoE -- the token is projected
    7168 -> 3584 once, every expert runs at that width, the weighted expert
    sum is RMS-normalized and projected back up. The shared experts always run
    at full hidden width on the original input, so they cannot be folded into
    the routed path the way DeepSeek-V3's ``SharedFusedMoe`` does.
    """

    def __init__(self,
                 *,
                 config: "KimiConfig",
                 mesh: Mesh,
                 dtype: jnp.dtype,
                 quant_config: Optional[QuantizationConfig],
                 rngs: nnx.Rngs,
                 moe_backend: MoEBackend,
                 num_expert_parallelism: int = 1,
                 random_init: bool = False,
                 prefix: str = ""):
        self.config = config
        self.use_latent_moe = config.routed_expert_hidden_size is not None
        moe_hidden = config.moe_hidden_size
        weight_init = _weight_init(random_init)

        self.gate = DeepSeekV3Router(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_token,
            n_groups=config.num_expert_group,
            topk_groups=config.topk_group,
            norm_topk_prob=config.moe_renormalize,
            rngs=rngs,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=dtype,
            moe_backend=moe_backend,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            ed_sharding=P(None, None),
            e_sharding=P(None, ),
            scoring_func=config.moe_router_activation_func,
            random_init=random_init,
            quant_config=None)

        if self.use_latent_moe:
            self.routed_expert_down_proj = JaxEinsum(
                einsum_str="TD,DL->TL",
                kernel_shape=(config.hidden_size, moe_hidden),
                rngs=rngs,
                quant_config=None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(weight_init, P(None, None)),
                prefix=f"{prefix}.routed_expert_down_proj")
            self.routed_expert_up_proj = JaxEinsum(
                einsum_str="TL,LD->TD",
                kernel_shape=(moe_hidden, config.hidden_size),
                rngs=rngs,
                quant_config=None,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(weight_init, P(None, None)),
                prefix=f"{prefix}.routed_expert_up_proj")
            if config.latent_moe_use_norm:
                self.routed_expert_norm = JaxRmsNorm(
                    moe_hidden,
                    epsilon=config.rms_norm_eps,
                    scale_init=nnx.with_partitioning(init_fn, (None, )),
                    param_dtype=dtype,
                    dtype=dtype,
                    quant_config=None,
                    prefix=f"{prefix}.routed_expert_norm",
                    rngs=rngs)

        self.experts = KimiRoutedExperts(
            dtype=dtype,
            num_local_experts=config.num_experts,
            apply_expert_weight_before_computation=False,
            expert_axis_name=ShardingAxisName.ATTN_DATA_EXPERT,
            num_expert_parallelism=num_expert_parallelism,
            hidden_size=moe_hidden,
            intermediate_size_moe=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_token,
            mesh=mesh,
            hidden_act=config.hidden_act,
            rngs=rngs,
            random_init=random_init,
            quant_config=quant_config,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            activation_ffw_ted=P(ShardingAxisName.MLP_DATA, None, None),
            # The routed experts are the bulk of the weights (~94 GB of
            # Kimi-Linear-48B, ~1.4 TB of Kimi-K3), so they are sharded over
            # the expert axis rather than replicated -- same layout DeepSeek-V3
            # uses on its unfused backends.
            edf_sharding=P(ShardingAxisName.ATTN_DATA_EXPERT, None, None),
            efd_sharding=P(ShardingAxisName.ATTN_DATA_EXPERT, None, None),
            moe_backend=moe_backend,
            qwix_quantized_weight_dtype=None,
            prefix=f"{prefix}.experts",
            router=self.gate,
            scoring_func=config.moe_router_activation_func,
            renormalize=config.moe_renormalize)
        # Read back by `situ_params` inside the MoE backends: SiTU transforms
        # the up branch too, so the betas cannot ride on the activation name.
        self.experts.situ_beta = config.situ_beta
        self.experts.situ_linear_beta = config.situ_linear_beta

        if config.num_shared_experts:
            self.shared_experts = GatedMLP(
                dtype=dtype,
                hidden_act=config.hidden_act,
                hidden_size=config.hidden_size,
                intermediate_size=(config.moe_intermediate_size *
                                   config.num_shared_experts),
                situ_beta=config.situ_beta,
                situ_linear_beta=config.situ_linear_beta,
                rngs=rngs,
                random_init=random_init,
                activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
                df_sharding=P(None, ShardingAxisName.ATTN_HEAD),
                fd_sharding=P(ShardingAxisName.ATTN_HEAD, None),
                quant_config=None,
                prefix=f"{prefix}.shared_experts")

    def named_parameters(self, *args, **kwargs):
        for name, param in super().named_parameters(*args, **kwargs):
            # `self.gate` is also reachable as `experts.router`; yield it once,
            # under its checkpoint name.
            if ".router." in name or name.startswith("router."):
                continue
            yield name, param

    def __call__(self, x_TD: jax.Array) -> jax.Array:
        identity_TD = x_TD
        routed_in = x_TD
        if self.use_latent_moe:
            routed_in = self.routed_expert_down_proj(x_TD)

        # The router scores the FULL-width hidden state, not the latent one.
        routing = self.gate(x_TD)
        if self.config.routed_scaling_factor != 1.0:
            # The reference applies the scale to the top-k weights inside the
            # gate. With a latent MoE that is NOT the same as scaling the
            # block output: the latent RMSNorm sits in between and is
            # scale-invariant. (Kimi-K3 uses 1.0; Kimi-Linear-48B uses 2.446.)
            assert isinstance(routing, tuple), (
                "[kimi-k3] routed_scaling_factor needs an unfused MoE backend "
                "whose router returns (weights, indices); fused backends "
                "select experts internally from raw logits")
            weights_TX, indices_TX = routing
            routing = (weights_TX * self.config.routed_scaling_factor,
                       indices_TX)
        y, _ = self.experts(routed_in, router_logits=routing)

        if self.use_latent_moe:
            if self.config.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        if hasattr(self, "shared_experts"):
            y = y + self.shared_experts(identity_TD)
        return y


class KimiDecoderLayer(JaxModule):
    """One decoder layer, with or without attention residuals.

    With attention residuals the plain residual stream is replaced by a
    ``prefix_sum`` (running sum of the attention and MLP outputs) plus a list
    of depth checkpoints (``block_residual``); each of the two sites mixes
    them with a learned 1-dim pseudo-query. See :func:`apply_attn_res`.
    """

    def __init__(self, *, config: "KimiConfig", layer_idx: int,
                 input_layernorm: JaxRmsNorm,
                 post_attention_layernorm: JaxRmsNorm, self_attn: JaxModule,
                 mlp: JaxModule, rngs: nnx.Rngs, dtype: jnp.dtype,
                 random_init: bool, prefix: str):
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.self_attn = self_attn
        # The checkpoint names the feed-forward `block_sparse_moe` on MoE
        # layers and `mlp` on dense ones; the weight loader resolves by
        # attribute path, so register it under the matching name.
        self.mlp_attr = ("block_sparse_moe" if isinstance(
            mlp, KimiSparseMoeBlock) else "mlp")
        setattr(self, self.mlp_attr, mlp)
        if config.use_attn_residuals:
            site_kwargs = dict(hidden_size=config.hidden_size,
                               dtype=dtype,
                               rms_norm_eps=config.rms_norm_eps,
                               rngs=rngs,
                               random_init=random_init)
            (self.self_attention_res_proj,
             self.self_attention_res_norm) = build_attn_res_site(
                 prefix=f"{prefix}.self_attention_res", **site_kwargs)
            self.mlp_res_proj, self.mlp_res_norm = build_attn_res_site(
                prefix=f"{prefix}.mlp_res", **site_kwargs)

    def _mlp(self) -> JaxModule:
        return getattr(self, self.mlp_attr)

    def _mix(self, proj: JaxEinsum, norm: JaxRmsNorm, prefix_sum_TD,
             block_residual):
        return apply_attn_res(prefix_sum_TD, block_residual, proj.weight.value,
                              norm.weight.value, self.config.rms_norm_eps)

    def __call__(
        self,
        x_TD: jax.Array,
        *,
        kv_cache,
        attention_metadata: AttentionMetadata,
        block_residual: Optional[List[jax.Array]] = None,
    ):
        if not self.config.use_attn_residuals:
            residual = x_TD
            hidden = self.input_layernorm(x_TD)
            new_cache, attn_out = self.self_attn(hidden, kv_cache,
                                                 attention_metadata)
            hidden = residual + attn_out
            residual = hidden
            hidden = self.post_attention_layernorm(hidden)
            return (new_cache, residual + self._mlp()(hidden), block_residual)

        prefix_sum = x_TD
        hidden = x_TD
        if block_residual:
            hidden = self._mix(self.self_attention_res_proj,
                               self.self_attention_res_norm, prefix_sum,
                               block_residual)

        if self.config.is_attn_res_checkpoint(self.layer_idx):
            # Start a new depth block: this layer's input becomes a checkpoint
            # and the prefix sum restarts from the attention output.
            block_residual = list(block_residual) + [prefix_sum]
            prefix_sum = None

        hidden = self.input_layernorm(hidden)
        new_cache, attn_out = self.self_attn(hidden, kv_cache,
                                             attention_metadata)
        prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

        hidden = self._mix(self.mlp_res_proj, self.mlp_res_norm, prefix_sum,
                           block_residual)
        hidden = self.post_attention_layernorm(hidden)
        prefix_sum = prefix_sum + self._mlp()(hidden)

        return new_cache, prefix_sum, block_residual


class KimiLinearModel(JaxModule):
    """The Kimi text stack: embeddings, decoder layers, final norm."""

    def __init__(self,
                 *,
                 config: "KimiConfig",
                 rngs: nnx.Rngs,
                 mesh: Mesh,
                 dtype: jnp.dtype,
                 kv_cache_dtype: str = "auto",
                 quant_config: Optional[QuantizationConfig] = None,
                 moe_backend: MoEBackend = MoEBackend.DENSE_MAT,
                 random_init: bool = False,
                 prefix: str = "model"):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.moe_backend = moe_backend

        self.embed_tokens = JaxEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            param_dtype=dtype,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(
                init_fn, (ShardingAxisName.MLP_TENSOR, )),
            rngs=rngs,
            quant_config=quant_config,
            prefix=prefix + ".embed_tokens",
        )

        self.layers = JaxModuleList([
            self._build_layer(i,
                              rngs=rngs,
                              dtype=dtype,
                              quant_config=quant_config,
                              kv_cache_dtype=kv_cache_dtype,
                              random_init=random_init,
                              prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])

        if config.use_attn_residuals:
            (self.output_attn_res_proj,
             self.output_attn_res_norm) = build_attn_res_site(
                 hidden_size=config.hidden_size,
                 dtype=dtype,
                 rms_norm_eps=config.rms_norm_eps,
                 rngs=rngs,
                 random_init=random_init,
                 prefix=prefix + ".output_attn_res")

        self.norm = JaxRmsNorm(config.hidden_size,
                               epsilon=config.rms_norm_eps,
                               scale_init=nnx.with_partitioning(
                                   init_fn, (None, )),
                               param_dtype=dtype,
                               dtype=dtype,
                               quant_config=quant_config,
                               prefix=prefix + ".norm",
                               rngs=rngs)

    def _build_layer(self, layer_idx: int, *, rngs, dtype, quant_config,
                     kv_cache_dtype, random_init, prefix) -> KimiDecoderLayer:
        config = self.config

        def _norm(name):
            return JaxRmsNorm(config.hidden_size,
                              epsilon=config.rms_norm_eps,
                              scale_init=nnx.with_partitioning(
                                  init_fn, (None, )),
                              param_dtype=dtype,
                              dtype=dtype,
                              quant_config=quant_config,
                              prefix=f"{prefix}.{name}",
                              rngs=rngs)

        input_layernorm = _norm("input_layernorm")
        post_attention_layernorm = _norm("post_attention_layernorm")

        if config.is_kda_layer(layer_idx):
            self_attn = KimiDeltaAttention(
                hidden_size=config.hidden_size,
                num_heads=config.kda_num_heads,
                head_dim=config.kda_head_dim,
                conv_kernel_size=config.kda_conv_kernel_size,
                rms_norm_eps=config.rms_norm_eps,
                dtype=dtype,
                use_full_rank_gate=config.kda_use_full_rank_gate,
                gate_lower_bound=config.kda_gate_lower_bound,
                random_init=random_init,
                quant_config=None,
                rngs=rngs,
                prefix=f"{prefix}.self_attn")
        else:
            self_attn = MLAAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=1,
                head_dim=config.v_head_dim,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                mesh=self.mesh,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                rms_norm_eps=config.rms_norm_eps,
                rope=None,
                use_nope=config.mla_use_nope,
                use_output_gate=config.mla_use_output_gate,
                random_init=random_init,
                rngs=rngs,
                quant_config=quant_config,
                activation_attention_td=P(ShardingAxisName.ATTN_DATA, None),
                activation_q_td=P(ShardingAxisName.ATTN_DATA, None),
                query_nth=P(None, ShardingAxisName.ATTN_DATA, None),
                query_tnh=P(ShardingAxisName.ATTN_DATA, None, None),
                keyvalue_skh=P(ShardingAxisName.ATTN_DATA, None),
                activation_attention_out_td=P(ShardingAxisName.ATTN_DATA,
                                              None),
                attn_o_nth=P(None, ShardingAxisName.ATTN_DATA, None),
                anh_sharding=(None, ShardingAxisName.ATTN_HEAD, None),
                q_da_sharding=P(None, ShardingAxisName.ATTN_HEAD),
                ap_sharding=P(None, ShardingAxisName.ATTN_HEAD),
                kv_da_sharding=P(None, ShardingAxisName.ATTN_HEAD),
                rd_sharding=P(ShardingAxisName.ATTN_HEAD, None),
                prefix=f"{prefix}.self_attn")

        if config.is_moe_layer(layer_idx):
            mlp = KimiSparseMoeBlock(config=config,
                                     mesh=self.mesh,
                                     dtype=dtype,
                                     quant_config=quant_config,
                                     rngs=rngs,
                                     moe_backend=self.moe_backend,
                                     random_init=random_init,
                                     prefix=f"{prefix}.block_sparse_moe")
        else:
            mlp = GatedMLP(dtype=dtype,
                           hidden_act=config.hidden_act,
                           hidden_size=config.hidden_size,
                           intermediate_size=config.intermediate_size,
                           situ_beta=config.situ_beta,
                           situ_linear_beta=config.situ_linear_beta,
                           rngs=rngs,
                           random_init=random_init,
                           activation_ffw_td=P(ShardingAxisName.MLP_DATA,
                                               None),
                           df_sharding=P(None, ShardingAxisName.ATTN_HEAD),
                           fd_sharding=P(ShardingAxisName.ATTN_HEAD, None),
                           quant_config=quant_config,
                           prefix=f"{prefix}.mlp")

        return KimiDecoderLayer(
            config=config,
            layer_idx=layer_idx,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            self_attn=self_attn,
            mlp=mlp,
            rngs=rngs,
            dtype=dtype,
            random_init=random_init,
            prefix=prefix)

    def __call__(
        self,
        kv_caches: List[Any],
        input_ids: Optional[jax.Array],
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[Any], jax.Array]:
        x = (inputs_embeds
             if inputs_embeds is not None else self.embed_tokens(input_ids))

        block_residual: List[jax.Array] = []
        for i, layer in enumerate(self.layers):
            kv_caches[i], x, block_residual = layer(
                x,
                kv_cache=kv_caches[i],
                attention_metadata=attention_metadata,
                block_residual=block_residual)

        if self.config.use_attn_residuals:
            x = apply_attn_res(x, block_residual,
                               self.output_attn_res_proj.weight.value,
                               self.output_attn_res_norm.weight.value,
                               self.config.rms_norm_eps)
        return kv_caches, self.norm(x)


class KimiLinearForCausalLM(JaxModule, LoadableWithIterator):
    """Kimi-Linear-48B / Kimi-K3 text stack (KDA + gated NoPE MLA hybrid)."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng_key: jax.Array,
                 mesh: Mesh,
                 *,
                 moe_backend: MoEBackend = MoEBackend.DENSE_MAT,
                 random_init: bool = False) -> None:
        self.vllm_config = vllm_config
        self.mesh = mesh
        rngs = nnx.Rngs(rng_key)

        model_config = vllm_config.model_config
        config = KimiConfig(model_config.hf_config)
        self.config = config
        dtype = model_config.dtype

        self.model = KimiLinearModel(
            config=config,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            quant_config=vllm_config.quant_config,
            moe_backend=moe_backend,
            random_init=random_init,
            prefix="model")
        self.lm_head = JaxLmHead(
            hidden_size=config.hidden_size,
            vocab_size=model_config.get_vocab_size(),
            param_dtype=dtype,
            dtype=dtype,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(
                init_fn, (None, ShardingAxisName.MLP_TENSOR)),
            prefix="lm_head",
        )
        attach_default_weight_loaders(self)

    def __call__(
        self,
        kv_caches: List[Any],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        *args,
        **kwargs,
    ) -> Tuple[List[Any], jax.Array, List[jax.Array], Optional[jax.Array]]:
        kv_caches, x = self.model(kv_caches, input_ids, attention_metadata,
                                  inputs_embeds)
        return kv_caches, x, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable) -> set:
        if not isinstance(weights, Iterable):
            return super().load_weights(weights)
        loader = JaxAutoWeightsLoader(self,
                                      skip_prefixes=list(VISION_SKIP_PREFIXES))
        return loader.load_weights(weights)


class KimiK3ForConditionalGeneration(KimiLinearForCausalLM):
    """Kimi-K3 (text-only serving; vision-tower weights are skipped).

    The multimodal checkpoint wraps the same text stack in a ``text_config``,
    which :class:`KimiConfig` unwraps. Image inputs are rejected -- MoonViT-V2
    is a later phase.
    """

    def get_multimodal_embeddings(self, **kwargs):
        raise NotImplementedError(
            "[kimi-k3] vision inputs are not supported: this build serves the "
            "text stack only (MoonViT-V2 is not implemented). Send text-only "
            "requests.")
