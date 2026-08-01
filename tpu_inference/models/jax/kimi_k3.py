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
import math
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
from tpu_inference.layers.common.kda_attention import (
    KDAParams, KDAState, kda_a_log_from_checkpoint, kda_attention)
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import \
    ShardingAxisNameBase as ShardingAxisName
from tpu_inference.layers.jax import JaxModule, JaxModuleList
from tpu_inference.layers.jax.activation import SITU_BETA, SITU_LINEAR_BETA
from tpu_inference.layers.jax.attention.mla import (
    MLAAttention, derived_absorbed_mla_param_names)
from tpu_inference.layers.jax.base import _init_fn as init_fn
from tpu_inference.layers.jax.base import create_param, sharded_initializer
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLmHead
from tpu_inference.layers.jax.mlp import GatedMLP
from tpu_inference.layers.jax.moe.moe import JaxMoE
from tpu_inference.layers.jax.moe.utils import get_expert_parallelism
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.common.linear_attention import (
    compute_linear_attention_layers, linear_attn_config)
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3Router
from tpu_inference.models.jax.utils.weight_utils import (
    JaxAutoWeightsLoader, LoadableWithIterator,
    load_nnx_param_from_reshaped_torch)
from tpu_inference.utils import get_mesh_shape_product

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

# The released Kimi-K3 checkpoint is the multimodal wrapper: every text tensor
# is stored under ``language_model.`` (``language_model.model.*`` and
# ``language_model.lm_head.weight``) alongside the vision tensors above, and
# `architectures` is ``KimiK3ForConditionalGeneration``. Kimi-Linear-48B stores
# the same text stack unprefixed, so strip the prefix when it is present rather
# than requiring one layout -- the same way :class:`KimiConfig` accepts the
# text config either nested in ``text_config`` or at the top level, and the
# same convention the other conditional-generation models here follow.
WRAPPER_TEXT_PREFIX = "language_model."

# Routed-expert weight names in the Kimi checkpoints vs the canonical names
# `JaxMoE._load_weights` expects.
_EXPERT_PARAM_MAP = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def routed_expert_shardings() -> Tuple[P, P]:
    """``(edf, efd)`` specs for the routed-expert kernels.

    The routed experts are almost the whole model -- ~94 GiB of
    Kimi-Linear-48B, ~1.4 TB of Kimi-K3 -- so they are split along two axes at
    once:

      * the expert axis ``E`` over :data:`ShardingAxisName.ATTN_DATA_EXPERT`,
        which is what lets the MEGABLX_GMM backend dispatch each token to the
        experts resident on this shard; and
      * the intermediate axis ``F`` over the tensor axis
        :data:`ShardingAxisName.MODEL`, the axis attention and the dense
        weights are already sharded on.

    Sharding only ``E`` -- the earlier layout -- forces a choice no single
    mesh can satisfy. Kimi-K3's non-expert stack is ~113 GB and needs
    ``model`` > 1 to fit, but on a fixed device budget every device given to
    ``model`` is one taken from the expert axis, so the experts end up
    replicated instead. Composing the two removes the conflict: at 32 devices,
    ``E`` over 4 and ``F`` over 8 splits the expert bytes 32 ways while
    ``model=8`` still shards attention.

    The collectives follow from where each contraction lands, and are already
    implemented -- this introduces no new communication:

      * ``gate``/``up`` contract ``D``, replicated in ``edf``, so each shard's
        product is already complete and nothing is reduced;
      * ``down`` contracts ``F``, which is sharded, so every shard holds a
        partial sum over its slice and they must be added.
        ``sparse_moe_distributed_fwd`` reduces over ``<spec>[1]`` of each
        kernel spec -- exactly ``F`` for ``efd`` and ``None`` for ``edf`` --
        and ``dense_moe_func`` runs under automatic sharding, where XLA
        inserts the same reduction.

    ``ATTN_DATA_EXPERT`` and ``MODEL`` are disjoint mesh axes, so the two
    splits multiply rather than collide. ``MODEL`` is used for ``F`` rather
    than a compound axis containing ``attn_dp``: under attention data
    parallelism distinct ``attn_dp`` ranks hold *different tokens*, so
    reducing a partial sum across that axis would mix them.
    """
    return (P(ShardingAxisName.ATTN_DATA_EXPERT, None, ShardingAxisName.MODEL),
            P(ShardingAxisName.ATTN_DATA_EXPERT, ShardingAxisName.MODEL, None))


def experts_are_block_quantized(quant_config) -> bool:
    """Whether the routed-expert weights arrive with per-block scales.

    Not ``quant_config is not None``: the unquantized path supplies an
    ``UnquantizedConfig`` rather than ``None``, and its weights carry no block
    scales, so the tiling constraint that goes with them does not apply.
    """
    if quant_config is None:
        return False
    from tpu_inference.layers.jax.quantization.unquantized import \
        UnquantizedConfig
    return not isinstance(quant_config, UnquantizedConfig)


def routed_expert_sharding_ways(mesh: Mesh) -> Tuple[int, int]:
    """``(expert_ways, tensor_ways)`` the routed-expert kernels are split."""
    return (get_mesh_shape_product(mesh,
                                   list(ShardingAxisName.ATTN_DATA_EXPERT)),
            get_mesh_shape_product(mesh, ShardingAxisName.MODEL))


def validate_routed_expert_sharding(config: "KimiConfig", mesh: Mesh, *,
                                    block_quantized: bool) -> Tuple[int, int]:
    """Check the routed experts really are split, and that the dims divide.

    Raises rather than letting the load proceed: both failure modes surface
    far from their cause. A mesh that replicates the experts dies as an
    out-of-memory part-way through loading the layers, and a dimension that
    does not divide dies inside the sharding machinery without naming the mesh
    that produced it.
    """
    expert_ways, tensor_ways = routed_expert_sharding_ways(mesh)
    mesh_desc = ", ".join(f"{n}={s}" for n, s in mesh.shape.items())

    if config.num_experts % expert_ways:
        raise ValueError(
            f"[kimi] num_experts={config.num_experts} is not divisible by the "
            f"{expert_ways} ways the expert axis "
            f"{ShardingAxisName.ATTN_DATA_EXPERT} is split (mesh: "
            f"{mesh_desc}).")
    if config.moe_intermediate_size % tensor_ways:
        raise ValueError(
            f"[kimi] moe_intermediate_size={config.moe_intermediate_size} is "
            f"not divisible by the {tensor_ways} ways the tensor axis "
            f"'{ShardingAxisName.MODEL}' is split (mesh: {mesh_desc}).")
    if block_quantized and tensor_ways > 1:
        # Measured on a 2x4 mesh with the tiny fixtures: a per-shard F that is
        # not a multiple of 128 makes the grouped matmul emit a handful of
        # NaNs (9 of 8192 output entries) while every other entry stays
        # correct to ~1e-8, so it corrupts silently rather than failing. The
        # grouped matmul tiles the intermediate dimension at a multiple of
        # 128; sharding F is what can leave a remainder there, and only the
        # block-quantized path is affected (the bf16 twin is clean at the same
        # per-shard width). Kimi-K3's F=3072 over 8 gives 384, which is fine.
        # A multiple of 128 is also a multiple of `MXFP4_BLOCK_SIZE`, so this
        # subsumes "the per-shard block-scale count must divide".
        f_local = config.moe_intermediate_size // tensor_ways
        if f_local % 128:
            raise ValueError(
                f"[kimi] splitting moe_intermediate_size="
                f"{config.moe_intermediate_size} {tensor_ways} ways over "
                f"'{ShardingAxisName.MODEL}' leaves {f_local} per shard, "
                f"which is not a multiple of 128. The block-quantized grouped "
                f"matmul returns NaNs for a few entries at that width instead "
                f"of failing. Choose a tensor-parallel width that divides "
                f"{config.moe_intermediate_size} into multiples of 128 "
                f"(mesh: {mesh_desc}).")

    num_devices = math.prod(mesh.axis_sizes)
    if num_devices > 1 and expert_ways * tensor_ways == 1:
        raise ValueError(
            f"[kimi] the routed experts would be replicated on all "
            f"{num_devices} devices: the expert axis "
            f"{ShardingAxisName.ATTN_DATA_EXPERT} and the tensor axis "
            f"'{ShardingAxisName.MODEL}' are both 1 in this mesh "
            f"({mesh_desc}). They are the bulk of the model, so this runs out "
            f"of memory part-way through loading. Give the mesh expert "
            f"parallelism, tensor parallelism, or both.")

    params = (config.num_experts * 3 * config.moe_hidden_size *
              config.moe_intermediate_size)
    logger.info(
        "[kimi] routed experts split %d ways: E over %s (%d) x F over '%s' "
        "(%d); %.3fG of %.3fG expert parameters per device (mesh: %s)",
        expert_ways * tensor_ways, ShardingAxisName.ATTN_DATA_EXPERT,
        expert_ways, ShardingAxisName.MODEL, tensor_ways,
        params / expert_ways / tensor_ways / 1e9, params / 1e9, mesh_desc)
    return expert_ways, tensor_ways


# Params whose checkpoint layout is NOT "2-D [out, in], transpose it", as
# (name suffix, reshape_dims, permute_dims).
_LOADER_LAYOUT_OVERRIDES = (
    # The HF embedding is [vocab, hidden] and so is ours.
    ("embed_tokens.weight", None, (0, 1)),
    # Depthwise conv weight is [channels, 1, kernel_size] on both sides.
    ("conv1d.weight", None, (0, 1, 2)),
)


def load_kda_a_log(jax_param,
                   torch_weight,
                   _shard_id: int = -1,
                   *,
                   param_name: str = "self_attn.A_log") -> None:
    """Load the KDA log-decay parameter.

    The parameter is one scalar per head, but the released checkpoints store
    that vector in two different layouts; which layouts are accepted, and the
    evidence that the parameter is per head rather than per channel, are in
    :func:`~tpu_inference.layers.common.kda_attention.kda_a_log_from_checkpoint`.
    """
    flat = kda_a_log_from_checkpoint(torch_weight,
                                     jax_param.get_value().shape[0],
                                     param_name)
    load_nnx_param_from_reshaped_torch(jax_param,
                                       flat,
                                       _shard_id,
                                       reshape_dims=None,
                                       permute_dims=None,
                                       param_name=param_name)


# Params needing a dedicated loader rather than a reshape/permute of the
# generic rule, as (name suffix, loader).
_LOADER_FN_OVERRIDES = (("self_attn.A_log", load_kda_a_log), )


def summarize_kda_decay_layout(
        weights: Iterable[Tuple[str, Any]]) -> Iterable[Tuple[str, Any]]:
    """Log, once per load, how this checkpoint stores the KDA decay.

    ``A_log`` means the same thing in every Kimi checkpoint -- one log-decay
    scalar per head -- but the released ones store that vector in two
    different layouts and nothing in the config says which
    (see :func:`load_kda_a_log`); only the tensor's shape does. Recording it on
    the first KDA layer rather than after the load keeps the line in the log
    even when a later tensor is what fails.
    """
    logged = False
    for name, weight in weights:
        if not logged and name.endswith("self_attn.A_log"):
            logged = True
            logger.info(
                "[kimi] KDA decay: checkpoint stores A_log as %s; reading it "
                "as one log-decay scalar per head (any trailing entries must "
                "be zero padding).", list(getattr(weight, "shape", ())))
        yield name, weight


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
        loader = None
        for suffix, loader_fn in _LOADER_FN_OVERRIDES:
            if name.endswith(suffix):
                loader = functools.partial(loader_fn, param_name=name)
                break
        if loader is None:
            reshape_dims = permute_dims = None
            for suffix, reshape, permute in _LOADER_LAYOUT_OVERRIDES:
                if name.endswith(suffix):
                    reshape_dims, permute_dims = reshape, permute
                    break
            loader = functools.partial(load_nnx_param_from_reshaped_torch,
                                       reshape_dims=reshape_dims,
                                       permute_dims=permute_dims,
                                       param_name=name)
        param.set_metadata("weight_loader", loader)


def adapt_wrapper_checkpoint(
        weights: Iterable[Tuple[str, Any]]) -> Iterable[Tuple[str, Any]]:
    """Present a multimodal-wrapper checkpoint as the text stack.

    Strips :data:`WRAPPER_TEXT_PREFIX` from the text tensors and drops the
    vision ones, which this build does not implement. A checkpoint that is
    already text-only passes through unchanged.

    The drop is counted and reported once rather than per tensor: the released
    Kimi-K3 checkpoint carries 168 vision tensors, and a line each would bury
    the rest of the load log. Only the listed prefixes are dropped, so a
    genuinely unexpected tensor still reaches the loader and fails loudly.
    """
    skipped: list[str] = []
    for name, weight in weights:
        if any(name.startswith(p + ".") for p in VISION_SKIP_PREFIXES):
            skipped.append(name)
            continue
        if name.startswith(WRAPPER_TEXT_PREFIX):
            name = name[len(WRAPPER_TEXT_PREFIX):]
        yield name, weight
    if skipped:
        logger.info(
            "[kimi] skipped %d vision/projector checkpoint tensors (e.g. %s); "
            "this build serves the text stack only.", len(skipped),
            ", ".join(sorted(skipped)[:3]))


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
        # Parsed through the shared helper so that the KV-cache manager (which
        # must know which layers are recurrent before any module exists) and
        # the model agree on the layer list by construction.
        linear_cfg = linear_attn_config(c)
        self.kda_layers: set = set(compute_linear_attention_layers(c))
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
    the hybrid-KV-cache work; correctness comes first here. The state-carrying
    math still runs under a ``shard_map`` over ``ATTN_DATA`` (see
    ``kda_attention``): that is not an optimization but a correctness
    requirement, because the ragged metadata is laid out per DP rank.
    """
    hidden_size: int
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    rms_norm_eps: float
    dtype: jnp.dtype
    use_full_rank_gate: bool
    mesh: Mesh
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
            # fp32, not the model dtype: Kimi-K3 checkpoints the short-conv
            # weights in fp32 (Kimi-Linear-48B checkpoints them in bf16), and
            # a bf16 parameter would round them away at load. vLLM's KDA layer
            # holds them in fp32 for the same reason
            # (`params_dtype=torch.float32` on its conv1d). See `_params` for
            # where the compute dtype is applied.
            return _RawWeight(shape=(HP, 1, self.conv_kernel_size),
                              dtype=jnp.float32,
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
        # fp32 for the same reason as the conv weights: Kimi-K3 checkpoints
        # `o_norm.weight` in fp32, and the gated RMSNorm that consumes it runs
        # in fp32 anyway, so a bf16 parameter would round the checkpoint value
        # away for nothing. (Kimi-Linear-48B checkpoints it in bf16; widening
        # that is exact.) Only the weight is used -- the norm math itself
        # lives in `kda_attention._gated_rmsnorm`.
        self.o_norm = JaxRmsNorm(K,
                                 epsilon=self.rms_norm_eps,
                                 scale_init=nnx.with_partitioning(
                                     init_fn, (None, )),
                                 param_dtype=jnp.float32,
                                 dtype=jnp.float32,
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
            # for parity checks).
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
            # The conv weights are held in fp32 (see `_conv`) but the ragged
            # conv's prefill branch is a `lax.conv_general_dilated`, which
            # rejects mixed dtypes, so the weight meets the activations at the
            # model dtype. This is the one place the checkpoint's extra conv
            # precision is dropped; the other two branches (the decode einsum
            # and the accumulate-in-fp32 loop) would take fp32 directly, so
            # running the whole conv in fp32 is a compute-dtype decision for
            # `ragged_conv1d`, not a loading one.
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
            # Not cast: the gated RMSNorm runs in fp32 and the parameter is
            # already fp32, so the checkpoint value reaches the math intact.
            o_norm_weight=self.o_norm.weight.value,
            o_proj=_t(self.o_proj),
            **gate_kwargs)

    def __call__(
            self, x_TD: jax.Array, state: Sequence[jax.Array],
            md: AttentionMetadata) -> Tuple[Tuple[jax.Array, ...], jax.Array]:
        """Runs the sublayer over one `kv_cache` slot.

        The slot arrives as a plain tuple of the four state arrays -- that is
        how the KV-cache manager allocates a `MambaSpec` layer, and how the
        decoder loop threads it. Naming them via `KDAState` here (and handing
        a plain tuple back) keeps the jitted model's input and output pytrees
        the same shape, so `self.kv_caches = model_fn(...)` round-trips
        without a retrace.
        """
        if md.has_initial_state is None:
            raise ValueError(
                "[kda] AttentionMetadata.has_initial_state is None, so this "
                "layer cannot tell which slots must resume recurrent/conv "
                "state. The runner publishes the field for models that "
                "register mamba kv-cache layers; a None here means this "
                "model's KDA layers were not registered as such. Not "
                "re-deriving it from query_start_loc on purpose -- that "
                "derivation is wrong under attention DP (see the field docs "
                "in layers/common/attention_metadata.py).")
        has_initial_state = md.has_initial_state.astype(jnp.bool_)
        new_state, out = kda_attention(x_TD,
                                       self._params(),
                                       KDAState(*state),
                                       md.mamba_state_indices,
                                       md.query_start_loc,
                                       md.request_distribution,
                                       has_initial_state,
                                       num_heads=self.num_heads,
                                       head_dim=self.head_dim,
                                       kernel_size=self.conv_kernel_size,
                                       gate_lower_bound=self.gate_lower_bound,
                                       rms_norm_eps=self.rms_norm_eps,
                                       mesh=self.mesh)
        return tuple(new_state), out


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
        edf_sharding, efd_sharding = routed_expert_shardings()

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
            # The routed experts are the bulk of the weights, and they are
            # split over the expert axis *and* the tensor axis at once --
            # see `routed_expert_shardings` for why one axis is not enough
            # and where the down-projection's reduction comes from.
            edf_sharding=edf_sharding,
            efd_sharding=efd_sharding,
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
                 num_expert_parallelism: int = 1,
                 random_init: bool = False,
                 prefix: str = "model"):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.moe_backend = moe_backend
        self.num_expert_parallelism = num_expert_parallelism

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
                mesh=self.mesh,
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
            mlp = KimiSparseMoeBlock(
                config=config,
                mesh=self.mesh,
                dtype=dtype,
                quant_config=quant_config,
                rngs=rngs,
                moe_backend=self.moe_backend,
                num_expert_parallelism=self.num_expert_parallelism,
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
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
        layer_name_to_kv_cache: Optional[dict] = None,
    ) -> Tuple[List[Any], jax.Array]:
        """Runs the stack.

        Two things are per-layer rather than global on a hybrid model:

        * ``attention_metadata`` is a ``{layer_name: metadata}`` dict whenever
          the model spans more than one kv-cache group (the KDA layers and the
          MLA layers land in different groups, each with its own block table).
        * ``kv_caches`` is ordered by kv-cache tensor, not by layer, so the
          slot for layer ``i`` has to be looked up rather than indexed.
        """
        x = (inputs_embeds
             if inputs_embeds is not None else self.embed_tokens(input_ids))

        block_residual: List[jax.Array] = []
        for i, layer in enumerate(self.layers):
            layer_name = f"layer.{i}"
            layer_md = (attention_metadata[layer_name] if isinstance(
                attention_metadata, dict) else attention_metadata)
            cache_idx = (layer_name_to_kv_cache[layer_name]
                         if layer_name_to_kv_cache
                         and layer_name in layer_name_to_kv_cache else i)
            kv_caches[cache_idx], x, block_residual = layer(
                x,
                kv_cache=kv_caches[cache_idx],
                attention_metadata=layer_md,
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
                 moe_backend: MoEBackend | None = None,
                 random_init: bool = False) -> None:
        self.vllm_config = vllm_config
        self.mesh = mesh
        rngs = nnx.Rngs(rng_key)

        model_config = vllm_config.model_config
        config = KimiConfig(model_config.hf_config)
        self.config = config
        dtype = model_config.dtype

        # How the routed-expert bytes are split, checked and reported before
        # anything is allocated: a mesh that replicates them only shows up as
        # an out-of-memory some layers into the load.
        self.num_expert_parallelism, self.num_expert_tensor_parallelism = (
            validate_routed_expert_sharding(
                config,
                mesh,
                block_quantized=experts_are_block_quantized(
                    vllm_config.quant_config)))
        # `num_expert_parallelism` stays the *expert*-axis count on purpose:
        # it is what the GMM backend uses to decide whether tokens have to be
        # dispatched to another shard, which sharding F does not change.
        assert self.num_expert_parallelism == get_expert_parallelism(
            ShardingAxisName.ATTN_DATA_EXPERT, mesh)
        if moe_backend is None:
            moe_backend = self._select_moe_backend()
        logger.info(
            "[kimi] MoE backend=%s, expert_parallelism=%d, "
            "expert_tensor_parallelism=%d (experts sharded on %s x '%s')",
            moe_backend.value, self.num_expert_parallelism,
            self.num_expert_tensor_parallelism,
            ShardingAxisName.ATTN_DATA_EXPERT, ShardingAxisName.MODEL)

        self.model = KimiLinearModel(
            config=config,
            rngs=rngs,
            mesh=mesh,
            dtype=dtype,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            quant_config=vllm_config.quant_config,
            moe_backend=moe_backend,
            num_expert_parallelism=self.num_expert_parallelism,
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

    def _select_moe_backend(self) -> MoEBackend:
        """Pick a backend that can see the router's expert selection.

        Kimi routes with noaux_tc: the router adds ``e_score_correction_bias``
        to the scores it ranks by but not to the weights it returns, and K3
        additionally uses the SiTU activation. The fused backends take raw
        router logits and select experts themselves, so they would silently
        drop the bias (changing which experts run) and have no SiTU branch —
        hence one of the two unfused backends, which consume the
        ``(weights, indices)`` the router already produced.

        Between those two: MEGABLX_GMM dispatches each token to the experts
        resident on this shard, which is what makes a split expert axis work
        at all, so it is the choice whenever the expert axis is split.
        DENSE_MAT evaluates every expert for every token and therefore needs
        every expert's weights locally.

        Note this keys on the *expert* axis only. Sharding ``F`` over the
        tensor axis (see :func:`routed_expert_shardings`) divides the bytes
        each device holds but not which experts it holds, so it does not make
        DENSE_MAT viable for a large expert count -- with the expert axis at 1
        every device still owns a slice of all 896 K3 experts.
        """
        if self.num_expert_parallelism > 1:
            return MoEBackend.MEGABLX_GMM
        if self.num_expert_tensor_parallelism > 1:
            # Not fatal -- a small MoE is fine like this -- but it is how a
            # mesh silently ends up evaluating every expert on every device,
            # so say so with the numbers rather than let it show up as an OOM.
            logger.warning(
                "[kimi] the expert axis %s is 1, so every device holds a "
                "slice of all %d experts and DENSE_MAT evaluates all of them "
                "per token; only the tensor axis '%s' (%d) is dividing the "
                "expert bytes. Give the mesh expert parallelism to use "
                "MEGABLX_GMM instead.", ShardingAxisName.ATTN_DATA_EXPERT,
                self.config.num_experts, ShardingAxisName.MODEL,
                self.num_expert_tensor_parallelism)
        return MoEBackend.DENSE_MAT

    def __call__(
        self,
        kv_caches: List[Any],
        input_ids: jax.Array,
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        layer_name_to_kv_cache: Optional[Sequence[Tuple[str, int]]] = None,
        *args,
        **kwargs,
    ) -> Tuple[List[Any], jax.Array, List[jax.Array], Optional[jax.Array]]:
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=dict(layer_name_to_kv_cache)
            if layer_name_to_kv_cache else None)
        return kv_caches, x, [], None

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        return self.lm_head(hidden_states)

    def load_weights(self, weights: Iterable) -> set:
        if not isinstance(weights, Iterable):
            return super().load_weights(weights)
        loader = JaxAutoWeightsLoader(self)
        loaded = loader.load_weights(
            summarize_kda_decay_layout(adapt_wrapper_checkpoint(weights)))
        # The absorbed-MLA up-projections are split out of `kv_b_proj` during
        # that call rather than read from the checkpoint, so they carry no
        # checkpoint name of their own.
        return loaded | derived_absorbed_mla_param_names(self)


class KimiK3ForConditionalGeneration(KimiLinearForCausalLM):
    """Kimi-K3 (text-only serving; vision-tower weights are skipped).

    The multimodal checkpoint wraps the same text stack twice over: in a
    ``text_config``, which :class:`KimiConfig` unwraps, and under a
    ``language_model.`` weight prefix, which
    :func:`adapt_wrapper_checkpoint` strips. Image inputs are rejected -- MoonViT-V2
    is a later phase.
    """

    def get_multimodal_embeddings(self, **kwargs):
        raise NotImplementedError(
            "[kimi-k3] vision inputs are not supported: this build serves the "
            "text stack only (MoonViT-V2 is not implemented). Send text-only "
            "requests.")
