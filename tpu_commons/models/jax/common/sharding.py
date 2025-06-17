from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import Mesh

from tpu_commons.models.jax.common.constants import LOGICAL_MESH_AXIS_NAME


@dataclass
class ShardingStrategy:
    """Defines the high-level parallelism strategy.

    This class specifies how many ways each type of parallelism (tensor, expert,
    sequence, data) should be distributed across the available devices.

    Attributes:
        tensor_parallelism: The degree of tensor parallelism (e.g., splitting
            weights of a single layer).
        expert_parallelism: The degree of expert parallelism for MoE models.
        sequence_parallelism: The degree of sequence parallelism (splitting
            activations along the sequence length dimension).
        data_parallelism: The degree of data parallelism (splitting the batch
            across devices).
    """
    tensor_parallelism: int = 1
    expert_parallelism: int = 1
    sequence_parallelism: int = 1
    data_parallelism: int = 1


#TODO split this into block unique sharding config, i.e. attentionShardingConfig, MoEShardingConfig
@dataclass
class OpShardingConfig:
    """Holds detailed sharding configurations for individual tensors, namely logical rules.

    Each attribute in this class corresponds to a specific weight or activation
    tensor within a transformer model. The value of each attribute is a
    tuple of logical mesh axis names (e.g., 'dp', 'sp', 'tp'), which defines
    how the corresponding tensor's dimensions are partitioned across the device mesh.
    The dimension order in the attribute name (e.g., `btd` for batch, sequence,
    d_model) maps directly to the sharding tuple.

    TODO: update the mesh axis names to be clear and reduce confusion between prefill & generate
    """

    # Activation for attn input: (Batch, Sequence, Dim)
    activation_attention_btd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation for attn out: (Batch, Sequence, Dim)
    activation_attention_out_btd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation for q projection input: (Batch, Sequence, Dim)
    activation_q_btd: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention Out activation after projection: (Batch, Sequence, NumHeads, HeadDim)
    attn_o_btnh: LOGICAL_MESH_AXIS_NAME | None = None
    # Q vector: (Batch, Sequence, NumHeads, HeadDim)
    query_btnh: LOGICAL_MESH_AXIS_NAME | None = None
    # K/V vector: (Batch, Sequence, NumKVHeads, HeadDim)
    keyvalue_bskh: LOGICAL_MESH_AXIS_NAME | None = None

    # Attention Q weight: (NumHeads, Dim, HeadDim)
    attn_q_weight_ndh: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention K weight: (NumKVHeads, Dim, HeadDim)
    attn_k_weight_kdh: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention V weight: (NumKVHeads, Dim, HeadDim)
    attn_v_weight_kdh: LOGICAL_MESH_AXIS_NAME | None = None
    # Attention Out weight: (NumHeads, HeadDim, Dim)
    attn_o_weight_nhd: LOGICAL_MESH_AXIS_NAME | None = None

    # K/V cache for generation: (Batch, Sequence, NumKVHeads, HeadDim)
    keyvalue_generate_mode_cache_bskh: LOGICAL_MESH_AXIS_NAME | None = None
    # K/V cache for prefill: (Batch, Sequence, NumKVHeads, HeadDim)
    keyvalue_prefill_mode_cache_bskh: LOGICAL_MESH_AXIS_NAME | None = None

    # Activation for ffw input: (Batch, Sequence, Dim)
    activation_ffw_btd: LOGICAL_MESH_AXIS_NAME | None = None
    # FFW hidden activation: (Batch, Sequence, FfwDim)
    ffw_hidden_btf: LOGICAL_MESH_AXIS_NAME | None = None

    # FFW up/gate weight: (Dim, FfwDim)
    ffw_weight_df: LOGICAL_MESH_AXIS_NAME | None = None
    # FFW down weight: (FfwDim, Dim)
    ffw_weight_fd: LOGICAL_MESH_AXIS_NAME | None = None
    # MoE gate/up weights: (NumExperts, Dim, FfwDim)
    moe_weights_edf: LOGICAL_MESH_AXIS_NAME | None = None
    # MoE down weights: (NumExperts, FfwDim, Dim)
    moe_weights_efd: LOGICAL_MESH_AXIS_NAME | None = None
    # MoE router weights: (Dim, NumExperts)
    moe_router_de: LOGICAL_MESH_AXIS_NAME | None = None

    # Embedding weight: (VocabSize, Dim)
    emb_weight_vd: LOGICAL_MESH_AXIS_NAME | None = None
    # Activation between layers: (Batch, Sequence, Dim)
    activation_btd: LOGICAL_MESH_AXIS_NAME | None = None
    # Final activation before logits: (Batch, Sequence, Dim)
    prelogit_btd: LOGICAL_MESH_AXIS_NAME | None = None
    # Logit activation: (Batch, Sequence, VocabSize)
    logits_btv: LOGICAL_MESH_AXIS_NAME | None = None
    # RMS norm scale weight: (Dim,)
    norm_scale: LOGICAL_MESH_AXIS_NAME | None = None
    # Vocab projection weight (tied embeddings): (Dim, VocabSize)
    vocab_dv: LOGICAL_MESH_AXIS_NAME | None = None


class ShardingConfig:
    """Container for operation-specific sharding configurations.

    This class holds two separate `OpShardingConfig` objects, one for the
    'prefill' phase and one for the 'generate' (or decode) phase of model
    execution. This allows tailoring sharding strategies to the different
    computational patterns of each phase.

    Example Sharding Strategy and Configuration:

    Sharding Strategy defines the high-level parallelism dimensions.
    For a device mesh like `Mesh((2, 4, 4, 4), ('dp', 'sp', 'ep', 'tp'))` on 128 devices:
    - dp: Data Parallelism (2-way)
    - sp: Sequence Parallelism (4-way)
    - ep: Expert Parallelism (4-way)
    - tp: Tensor Parallelism (4-way)

    ShardingConfig then maps tensor dimensions to these logical mesh axes.
    For example, a tensor with shape (Batch, Sequence, Dimension) could be sharded
    differently for prefill and decode/generate operations:

    - Prefill (long sequences, small batch):
    Sharding sequence dim on the 'sp' axis is often efficient.
    `prefill_sharding_cfg.activation_attention_btd = (None, 'sp', 'tp')`

    - Generate (short sequences, large batch):
    Sharding batch dim on the 'dp' axis is often efficient.
    `generate_sharding_cfg.activation_attention_btd = ('dp', None, 'tp')`
    """

    def __init__(self, prefill_sharding_cfg=None, generate_sharding_cfg=None):
        """Initializes the ShardingConfig.

        Args:
            prefill_sharding_cfg: An `OpShardingConfig` for the prefill phase.
                If None, a default config is created.
            generate_sharding_cfg: An `OpShardingConfig` for the generate phase.
                If None, a default config is created.
        """
        # Use a factory pattern to avoid mutable default arguments
        self.prefill_sharding_cfg = prefill_sharding_cfg if prefill_sharding_cfg is not None else OpShardingConfig(
        )
        self.generate_sharding_cfg = generate_sharding_cfg if generate_sharding_cfg is not None else OpShardingConfig(
        )


class Sharding:
    """Generates and manages sharding configurations based on a high-level strategy.

    This class takes a `ShardingStrategy`, builds the corresponding JAX `Mesh`
    of devices, and populates a `ShardingConfig` with detailed tensor sharding
    rules for both prefill and generation phases. It also allows for runtime
    overrides of these rules.

    Attributes:
        sharding_strategy: The high-level `ShardingStrategy` instance.
        sharding_cfg: The generated `ShardingConfig` with detailed rules.
        mesh: The JAX `Mesh` object representing the device grid.
    """
    sharding_strategy: ShardingStrategy
    sharding_cfg: ShardingConfig
    LOGICAL_MESH_AXIS_NAME: LOGICAL_MESH_AXIS_NAME

    def __init__(self,
                 strategy_dict: dict,
                 prefill_sharding_cfg: dict | None = None,
                 generate_sharding_cfg: dict | None = None):
        """Initializes the Sharding manager.

        Args:
            strategy_dict: A dictionary mapping parallelism types (e.g.,
                'tensor_parallelism') to their degrees.
            prefill_sharding_cfg: A dictionary of overrides for the prefill
                sharding config. Keys are attribute names in `OpShardingConfig`,
                and values are the new sharding tuples.
            generate_sharding_cfg: A dictionary of overrides for the generate
                sharding config.
        """
        self.sharding_strategy = ShardingStrategy(**strategy_dict)
        self.mesh = self.build_mesh(self.sharding_strategy)
        self.sharding_cfg = self.make_sharding_config(
            prefill_overrides=prefill_sharding_cfg,
            generate_overrides=generate_sharding_cfg)

    def validate_sharding_strategy(self, ):
        """Validates if the sharding strategy is compatible with the environment.

        This method is a placeholder now, and will check if the product of parallelism degrees
        matches the number of available devices.
        """
        #TODO: check num_devices % parallelism == 0
        #TODO: check num_devices == multiply(parallelism(with inferred))
        return

    def get_sharding_cfg(self) -> ShardingConfig:
        """Returns the generated sharding configuration."""
        return self.sharding_cfg

    def build_mesh(self, strategy: ShardingStrategy) -> Mesh:
        """Constructs a JAX device mesh from a sharding strategy.

        This method creates a logical grid of devices based on the parallelism
        degrees defined in the strategy. The logical axis names ('dp', 'ep',
        'sp', 'tp') are used to map tensor dimensions to the physical device grid.

        Args:
            strategy: The `ShardingStrategy` defining the mesh shape.

        Returns:
            A JAX `Mesh` object.
        """
        # TODO: to decide if we should name as x,y,z or 'data','tensor', 'expert' etc
        axis_order = {
            "dp": strategy.data_parallelism,
            "ep": strategy.expert_parallelism,
            "sp": strategy.sequence_parallelism,
            "tp": strategy.tensor_parallelism,
        }
        # TODO: add logic to infer axis when the degree is -1
        mesh_axis_names = []
        mesh_shape = []
        for axis, dim in axis_order.items():
            if dim > 1:
                mesh_axis_names.append(axis)
                mesh_shape.append(dim)

        if not mesh_shape:
            mesh_shape = [1]
            mesh_axis_names = [
                'dp'
            ]  # default to data parallelism if no other strategy is specified

        devices = np.asarray(jax.devices()).reshape(mesh_shape)
        return Mesh(devices, axis_names=tuple(mesh_axis_names))

    def _apply_overrides(self, config_obj: OpShardingConfig,
                         overrides: dict | None):
        """Applies runtime overrides to a sharding configuration object.

        Args:
            config_obj: The sharding configuration object (e.g., prefill_sharding_cfg)
                to be updated.
            overrides: A dictionary where keys are attribute names of the config
                object and values are the new sharding tuples.

        Raises:
            AttributeError: If a key in the overrides dictionary is not a valid
                attribute of the configuration object.
        """
        if overrides:
            for key, value in overrides.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                else:
                    # Raise an error for invalid keys to prevent silent failures
                    raise AttributeError(
                        f"'{key}' is not a valid attribute of {type(config_obj).__name__}"
                    )

    def make_sharding_config(
            self,
            prefill_overrides: dict | None = None,
            generate_overrides: dict | None = None) -> ShardingConfig:
        """Creates the detailed `ShardingConfig` with specific partitioning rules
        and applies any runtime overrides.

        This method populates the `prefill_sharding_cfg` and
        `generate_sharding_cfg` with hardcoded sharding rules that are generally
        effective for transformer models, and then updates them with any provided
        overrides.

        Args:
            prefill_overrides: A dictionary with attribute names and their new values
                for the prefill sharding configuration.
            generate_overrides: A dictionary with attribute names and their new values
                for the generate sharding configuration.

        Returns:
            The populated and overridden `ShardingConfig` object.
        """
        #TODO: organize into update_prefill() and update_decode for each axis
        #TODO: verify the sharding axes
        self.sharding_cfg = ShardingConfig()
        prefill_sharding_cfg = self.sharding_cfg.prefill_sharding_cfg
        generate_sharding_cfg = self.sharding_cfg.generate_sharding_cfg

        # Populate Prefill Config
        # During prefill, sequence length is long, so we shard along the sequence axis.
        prefill_sharding_cfg.activation_attention_btd = (
            None, LOGICAL_MESH_AXIS_NAME.SEQUENCE_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        prefill_sharding_cfg.activation_attention_out_btd = (
            None, LOGICAL_MESH_AXIS_NAME.SEQUENCE_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)

        # Populate Generate (Decode) Config
        # During decode, batch size is the large dimension, so we shard along the batch axis.
        generate_sharding_cfg.activation_attention_btd = (
            LOGICAL_MESH_AXIS_NAME.BATCH_AXIS_NAME, None,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.activation_attention_out_btd = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.activation_q_btd = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.attn_o_btnh = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_HEAD_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.query_btnh = (
            None, None, LOGICAL_MESH_AXIS_NAME.ATTN_HEAD_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.ATTN_TENSOR_AXIS_NAME)
        generate_sharding_cfg.activation_ffw_btd = (
            None, None, LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME)
        generate_sharding_cfg.ffw_hidden_btf = (
            None, None, LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME)
        # FFW weights are typically sharded along the hidden dimension (F).
        generate_sharding_cfg.ffw_weight_df = (
            None, LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME)
        generate_sharding_cfg.ffw_weight_fd = (
            LOGICAL_MESH_AXIS_NAME.MLP_TENSOR_AXIS_NAME, None)
        # MoE weights are sharded along the expert axis and the hidden dimension.
        generate_sharding_cfg.moe_weights_edf = (
            LOGICAL_MESH_AXIS_NAME.EXPERT_AXIS_NAME, None,
            LOGICAL_MESH_AXIS_NAME.MOE_TENSOR_AXIS_NAME)
        generate_sharding_cfg.moe_weights_efd = (
            LOGICAL_MESH_AXIS_NAME.EXPERT_AXIS_NAME,
            LOGICAL_MESH_AXIS_NAME.MOE_TENSOR_AXIS_NAME, None)

        # Apply overriding the runtime sharding rules
        self._apply_overrides(prefill_sharding_cfg, prefill_overrides)
        self._apply_overrides(generate_sharding_cfg, generate_overrides)

        return self.sharding_cfg


class ShardingInfo:
    #TODO a sharding info class for visualizing & debugging the sharding performance
    # Will implement it for the next version
    pass
