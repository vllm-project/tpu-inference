"""
Current Used abbreviation.
#TODO follow dragon book
B: batch
S: sequence length
T: token len
D: d_model
F: d_ff, hidden_dim of ffw/expert
V: vocab size
C: expert capacity
K: top K per token
G: number of groups
H: head dim in Attention
Q: number of query heads
N: number of KV heads
E: number of experts
"""
"""
Sharding_Strategy ->
Sharding:
self.Mesh i.e. Mesh((2,4,4,4), ('dp', 'sp', 'ep', 'tp')) 128 devices
self.ShardingConfig
    prefill_axes i.e. activation_attention_bsd = (None, 'dp', 'ep')
    decode_axes i.e. activation_attention_bsd = ('dp', None, 'ep')
"""
from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh

from tpu_commons.models.jax.common.constants import *


@dataclass
class Sharding_Strategy:
    tensor_parallelism: int = 1
    expert_parallelism: int = 1
    sequence_parallelism: int = 1
    data_parallelism: int = 1


@dataclass
class logical_mesh_axes:
    prefill_axes: tuple[str, ...] | None = None
    decode_axes: tuple[str, ...] | None = None  # (None, 'dp', 'ep')

    def update(self, axes: tuple[str, ...], op_mode: OPERATION_MODE):
        if op_mode == OPERATION_MODE.PREFILL:
            self.prefill_axes = axes
        elif op_mode == OPERATION_MODE.DECODE:
            self.decode_axes = axes

    def get_axes(self, op_mode: OPERATION_MODE = OPERATION_MODE.DECODE):
        if op_mode == OPERATION_MODE.PREFILL:
            return self.prefill_axes
        elif op_mode == OPERATION_MODE.DECODE:
            return self.decode_axes


@dataclass
class OpShardingConfig:

    # Activation for attn:
    # an example: logical_mesh_axes(prefill_axes=(None, 'sp', 'tp'), decode_axes=('sp', None, 'ep')
    activation_attention_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation for attn out:
    activation_attention_out_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation for q:
    activation_q_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention Out activation
    attn_o_bsqh: LOGICAL_MESH_AXIS_NAME | None = None
    # Q vector:
    query_bsqh: LOGICAL_MESH_AXIS_NAME | None = None
    # K/V vector:
    keyvalue_bsnh: LOGICAL_MESH_AXIS_NAME | None = None

    # Attention Q weight:
    attn_q_weight_qdh: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention K weight:
    attn_k_weight_ndh: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention V weight
    attn_v_weight_ndh: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention Out weight.
    attn_o_weight_qhd: LOGICAL_MESH_AXIS_NAME | None = None

    # K/V cache.
    keyvalue_generate_mode_cache_bsnh: LOGICAL_MESH_AXIS_NAME | None = None
    keyvalue_prefill_mode_cache_bsnh: LOGICAL_MESH_AXIS_NAME | None = None

    # Activation for ffw:
    activation_ffw_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # FFW hidden activation:
    ffw_hidden_bsf: LOGICAL_MESH_AXIS_NAME | None = None

    # FFW weight:
    ffw_weight_df: LOGICAL_MESH_AXIS_NAME | None = None
    # FFW weight:
    ffw_weight_fd: LOGICAL_MESH_AXIS_NAME | None = None
    # MoE weights
    moe_weights_edf: LOGICAL_MESH_AXIS_NAME | None = None
    moe_weights_efd: LOGICAL_MESH_AXIS_NAME | None = None
    moe_router_de: LOGICAL_MESH_AXIS_NAME | None = None

    # Embedding
    emb_weight_vd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation between layer:
    activation_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Final activation:
    prelogit_bsd: LOGICAL_MESH_AXIS_NAME | None = None
    # Logit activation:
    logits_bsv: LOGICAL_MESH_AXIS_NAME | None = None
    # RMS norm scale weight
    norm_scale: LOGICAL_MESH_AXIS_NAME | None = None
    # vocab sharding
    vocab_dv: LOGICAL_MESH_AXIS_NAME | None = None


class ShardingConfig:
    """Container for operation-specific sharding configurations."""

    def __init__(self, prefill_sharding_cfg=None, generate_sharding_cfg=None):
        # Use a factory pattern to avoid mutable default arguments
        self.prefill_sharding_cfg = prefill_sharding_cfg if prefill_sharding_cfg is not None else OpShardingConfig(
        )
        self.generate_sharding_cfg = generate_sharding_cfg if generate_sharding_cfg is not None else OpShardingConfig(
        )


class Sharding:
    """
    Sharding block, which stores and generates the ShardingConfig
    for tensors based on the Sharding_Strategy.
    """
    sharding_strategy: Sharding_Strategy
    sharding_cfg: ShardingConfig
    LOGICAL_MESH_AXIS_NAME: LOGICAL_MESH_AXIS_NAME

    def __init__(self, strategy_dict: dict):
        self.sharding_strategy = Sharding_Strategy(**strategy_dict)
        self.mesh = self.build_mesh(self.sharding_strategy)
        self.sharding_cfg = self.make_sharding_config()

    def validate_sharding_strategy(self, ):
        """
        Validate if the sharding strategy is correct and be able to fit in devices
        """

        #TODO check num_devices % parallelism == 0

        #TODO check num_devices == multiply(parallelism(with inferred))
        return

    def get_sharding_cfg(self) -> ShardingConfig:
        return self.sharding_cfg

    def build_mesh(self, strategy: Sharding_Strategy) -> Mesh:
        # TODO to decide if we should name as x,y,z or 'data','tensor', 'expert' etc
        axis_order = {
            "dp": strategy.data_parallelism,
            "ep": strategy.expert_parallelism,
            "sp": strategy.sequence_parallelism,
            "tp": strategy.tensor_parallelism,
        }
        # TODO add logic to infer axis when the degree is -1
        mesh_axis_names = []
        mesh_shape = []
        for axis, dim in axis_order.items():
            if dim > 1:
                mesh_axis_names.append(axis)
                mesh_shape.append(dim)

        if not mesh_shape:
            mesh_shape = [1]
            mesh_axis_names = ['dp']  # default

        devices = np.asarray(jax.devices()).reshape(mesh_shape)
        return Mesh(devices, axis_names=tuple(mesh_axis_names))

    #TODO add method to read sharding config directly user specified config file

    def make_sharding_config(self) -> ShardingConfig:
        #TODO organize into update_prefill() and update_decode for each axis
        #TODO verify the sharding axes
        self.sharding_cfg = ShardingConfig()
        prefill_sharding_cfg = self.sharding_cfg.prefill_sharding_cfg
        generate_sharding_cfg = self.sharding_cfg.generate_sharding_cfg

        # Populate Prefill Config
        prefill_sharding_cfg.activation_attention_bsd = (
            None, LOGICAL_MESH_AXIS_NAME.SEQUENCE_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        prefill_sharding_cfg.activation_attention_out_bsd = (
            None, LOGICAL_MESH_AXIS_NAME.SEQUENCE_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)

        # Populate Generate (Decode) Config
        generate_sharding_cfg.activation_attention_bsd = (
            LOGICAL_MESH_AXIS_NAME.BATCH_AXIS_NAME, None,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.activation_attention_out_bsd = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.activation_q_bsd = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.attn_o_bsqh = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_HEAD_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.query_bshq = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_HEAD_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.activation_ffw_bsd = (
            None, None, LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME)
        generate_sharding_cfg.ffw_hidden_bsf = (
            None, None, LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME)
        generate_sharding_cfg.ffw_weight_df = (
            None, None, LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME)
        generate_sharding_cfg.ffw_weight_fd = (
            LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME, None)
        generate_sharding_cfg.moe_weights_edf = (
            LOGICAL_MESH_AXIS_NAME.EXPERT_AXIS_NAME, None,
            LOGICAL_MESH_AXIS_NAME.MOE_TENSOR_AXIS_NAME)
        generate_sharding_cfg.moe_weights_efd = (
            LOGICAL_MESH_AXIS_NAME.EXPERT_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.MOE_TENSOR_AXIS_NAME, None)

        return self.sharding_cfg
