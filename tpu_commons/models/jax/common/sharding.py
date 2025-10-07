import json
from dataclasses import dataclass

import numpy as np
from jax.sharding import Mesh
from vllm.config import VllmConfig

BATCH_AXIS_NAME = 'data'
SEQUENCE_AXIS_NAME = 'data'
DATA_AXIS_NAME = 'data'
ATTN_HEAD_AXIS_NAME = 'model'
ATTN_TENSOR_AXIS_NAME = None
MLP_TENSOR_AXIS_NAME = ('model', 'expert')
MOE_TENSOR_AXIS_NAME = 'model'
EXPERT_AXIS_NAME = 'expert'
VOCAB_AXIS_NAME = ('data', 'expert', 'model')


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
class ShardingRulesConfig:
    """Holds detailed sharding configurations for individual tensors, namely logical rules.

    Each attribute in this class corresponds to a specific weight or activation
    tensor within a transformer model. The value of each attribute is a
    tuple of logical mesh axis names (e.g., 'dp', 'sp', 'tp'), which defines
    how the corresponding tensor's dimensions are partitioned across the device mesh.
    The dimension order in the attribute name (e.g., `btd` for batch, sequence,
    d_model) maps directly to the sharding tuple.

    TODO: update the mesh axis names to be clear and reduce confusion between prefill & generate
    """

    # Activation for attn input: (Batch * Sequence, Dim)
    activation_attention_td: tuple = (None, None)
    # Activation for attn out: (Batch * Sequence, Dim)
    activation_attention_out_td: tuple = (None, None)
    # Activation for q projection input: (Batch * Sequence, Dim)
    activation_q_td: tuple = (None, None)
    # Attention Out activation after projection: (Batch * Sequence, NumHeads, HeadDim)
    attn_o_tnh: tuple = (None, None, None)
    # Q vector: (Batch * Sequence, NumHeads, HeadDim)
    query_tnh: tuple = (None, None, None)
    # K/V vector: (Batch * Sequence, NumKVHeads, HeadDim)
    keyvalue_skh: tuple = (None, None, None)

    # Attention Q weight: (Dim, NumHeads, HeadDim)
    attn_q_weight_dnh: tuple = (None, None, None)
    # Attention K weight: (Dim, NumKVHeads, HeadDim)
    attn_k_weight_dkh: tuple = (None, None, None)
    # Attention V weight: (Dim, NumKVHeads, HeadDim)
    attn_v_weight_dkh: tuple = (None, None, None)
    # Attention Out weight: (NumHeads, HeadDim, Dim)
    attn_o_weight_nhd: tuple = (None, None, None)

    # Activation for ffw input: (Batch * Sequence, Dim)
    activation_ffw_td: tuple = (None, None)

    # Activation for ffw input: (Batch * Sequence, Expert, Dim)
    activation_ffw_ted: tuple = (None, None, None)

    # FFW hidden activation: (Batch * Sequence, FfwDim)
    ffw_hidden_tf: tuple = (None, None)

    # FFW up/gate weight: (Dim, FfwDim)
    ffw_weight_df: tuple = (None, None)
    # FFW down weight: (FfwDim, Dim)
    ffw_weight_fd: tuple = (None, None)
    # MoE gate/up weights: (NumExperts, Dim, FfwDim)
    moe_weights_edf: tuple = (None, None, None)
    # MoE down weights: (NumExperts, FfwDim, Dim)
    moe_weights_efd: tuple = (None, None, None)
    # MoE router weights: (Dim, NumExperts)
    moe_router_de: tuple = (None, None)
    # MoE router bias weights: (NumExperts,)
    moe_router_bias_e: tuple = (None, )

    # Embedding weight: (VocabSize, Dim)
    emb_weight_vd: tuple = (None, None)
    # Activation between layers: (Batch * Sequence, Dim)
    activation_td: tuple = (None, None)
    # Final activation before logits: (Batch * Sequence, Dim)
    prelogit_td: tuple = (None, None)
    # Logit activation: (Batch * Sequence, VocabSize)
    logits_tv: tuple = (None, None)
    # RMS norm scale weight: (Dim,)
    norm_scale: tuple = (None)
    # Vocab projection weight (tied embeddings): (Dim, VocabSize)
    vocab_vd: tuple = (None, None)
    vocab_dv: tuple = (None, None)


class ShardingConfig:
    """Container for operation-specific sharding configurations.

    This class holds two separate `ShardingRulesConfig` objects, one for the
    'prefill' phase and one for the 'generate' (or decode) phase of model
    execution. This allows tailoring sharding strategies to the different
    computational patterns of each phase.

    Example Sharding Strategy and Configuration:

    Sharding Strategy defines the high-level parallelism dimensions.
    For a device mesh like `Mesh((2, 4, 4, 4), ('data', 'seq', 'expert', 'tensor'))` on 128 devices:
    - data: Data Parallelism (2-way)
    - seq: Sequence Parallelism (4-way)
    - expert: Expert Parallelism (4-way)
    - tensor: Tensor Parallelism (4-way)

    ShardingConfig then maps tensor dimensions to these logical mesh axes.
    For example, a tensor with shape (Batch, Sequence, Dimension) could be sharded
    differently for prefill and decode/generate operations:

    - Prefill (long sequences, small batch):
    Sharding sequence dim on the 'sp' axis is often efficient.
    `prefill_rules.activation_attention_btd = (None, 'seq', 'tensor')`

    - Generate (short sequences, large batch):
    Sharding batch dim on the 'dp' axis is often efficient.
    `generate_rules.activation_attention_btd = ('data', None, 'tensor')`
    """

    def __init__(self,
                 prefill_rules=None,
                 generate_rules=None,
                 default_rules_cls=ShardingRulesConfig):
        """Initializes the ShardingConfig.

        Args:
            prefill_rules: An `ShardingRulesConfig` for the prefill phase.
                If None, a default config is created.
            generate_rules: An `ShardingRulesConfig` for the generate phase.
                If None, a default config is created.
            default_rules_cls: The default sharding rules (class) to use.
        """
        # Use a factory pattern to avoid mutable default arguments
        self.default_rules_cls = default_rules_cls
        self.prefill_rules = prefill_rules if prefill_rules is not None else default_rules_cls(
        )
        self.generate_rules = generate_rules if generate_rules is not None else default_rules_cls(
        )


def build_mesh(devices, strategy: dict[str, int]) -> Mesh:
    """Constructs a JAX device mesh from a sharding strategy.

    This method creates a logical grid of devices based on the parallelism
    degrees defined in the strategy. The logical axis names ('dp', 'ep',
    'sp', 'tp') are used to map tensor dimensions to the physical device grid.

    Args:
        strategy: A dictionary from upper level config.

    Returns:
        A JAX `Mesh` object.
    """

    axis_order = {
        "data": strategy.get("data_parallelism", 1),
        "expert": strategy.get("expert_parallelism", 1),
        "seq": strategy.get("sequence_parallelism", 1),
        "model": strategy.get("tensor_parallelism", 1),
    }
    # TODO: add logic to infer axis when the degree is -1
    mesh_axis_names = []
    mesh_shape = []
    for axis, dim in axis_order.items():
        mesh_axis_names.append(axis)
        mesh_shape.append(dim)

    if not mesh_shape:
        mesh_shape = [1]
        mesh_axis_names = [
            'data'
        ]  # default to data parallelism if no other strategy is specified

    devices = np.asarray(devices).reshape(mesh_shape)
    return Mesh(devices, axis_names=tuple(mesh_axis_names))


class Sharding:
    """Generates and manages sharding configurations based on a high-level strategy.

    This class populates a `ShardingConfig` with detailed tensor sharding
    rules for both prefill and generation phases. It also allows for runtime
    overrides of these rules.

    Attributes:
        sharding_cfg: The generated `ShardingConfig` with detailed rules.
    """

    def __init__(self,
                 prefill_rules: dict | None = None,
                 generate_rules: dict | None = None,
                 default_rules_cls=ShardingRulesConfig,
                 vllm_config: VllmConfig = None):
        """Initializes the Sharding manager.

        Args:
            prefill_rules: A dictionary of overrides for the prefill
                sharding config. Keys are attribute names in `ShardingRulesConfig`,
                and values are the new sharding tuples.
            generate_rules: A dictionary of overrides for the generate
                sharding config.
        """
        self.vllm_config = vllm_config
        self.default_rules_cls = default_rules_cls
        self.sharding_cfg = self.make_sharding_config(
            default_rules_cls=default_rules_cls,
            prefill_overrides=prefill_rules,
            generate_overrides=generate_rules)

    def _get_overrides(self, sharding_phase: str):
        """Return the overrides from the vLLM config for the given sharding phase."""
        overrides = {}
        try:
            overrides = self.vllm_config.additional_config["sharding"][
                "logical_rules"]["all"]
        except KeyError:
            pass

        try:
            additional_overrides = self.vllm_config.additional_config[
                "sharding"]["logical_rules"][f"{sharding_phase}"]
            overrides.update(additional_overrides)
        except KeyError:
            pass
        return overrides

    def __str__(self):
        """Succinct representation of relevant Sharding settings and overrides."""
        output_str = f"  Using {self.default_rules_cls.__name__} logical rules.\n"
        output_str += f"  {self.__class__.__name__:} overrides:\n"
        output_str += f"    prefill logical_rule overrides:\n    {json.dumps(self._get_overrides('prefill'), indent=4, default=str)}\n\n"
        output_str += f"    generate logical_rule overrides:\n    {json.dumps(self._get_overrides('generate'), indent=4, default=str)}\n\n"
        return output_str

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

    def _apply_overrides(self, config_obj: ShardingRulesConfig,
                         overrides: dict | None):
        """Applies runtime overrides to a sharding configuration object.

        Args:
            config_obj: The sharding configuration object (e.g., prefill_rules)
                to be updated.
            overrides: A dictionary where keys are attribute names of the config
                object and values are the new sharding tuples.

        Raises:
            AttributeError: If a key in the overrides dictionary is not a valid
                attribute of the configuration object.
        """
        for key, value in overrides.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                # Raise an error for invalid keys to prevent silent failures
                raise AttributeError(
                    f"'{key}' is not a valid attribute of {type(config_obj).__name__}"
                )

    def _make_default_sharding_config(self, prefill_rules, generate_rules):

        # Populate Prefill Config
        # During prefill, sequence length is long, so we shard along the sequence axis.
        prefill_rules.activation_attention_td = (DATA_AXIS_NAME,
                                                 ATTN_TENSOR_AXIS_NAME)
        prefill_rules.activation_attention_out_td = (DATA_AXIS_NAME,
                                                     ATTN_TENSOR_AXIS_NAME)
        prefill_rules.activation_q_td = (DATA_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)
        #TODO: the default qkv and kvcache is sharded on head dim
        # We may change it after we finalize the KVCache design
        prefill_rules.attn_o_tnh = (DATA_AXIS_NAME, ATTN_HEAD_AXIS_NAME, None)
        prefill_rules.query_tnh = (DATA_AXIS_NAME, ATTN_HEAD_AXIS_NAME, None)
        prefill_rules.keyvalue_skh = (DATA_AXIS_NAME, ATTN_HEAD_AXIS_NAME,
                                      None)

        # Populate Generate (Decode) Config
        # During decode, batch size is the large dimension, so we shard along the batch axis.
        generate_rules.activation_attention_td = (DATA_AXIS_NAME,
                                                  ATTN_TENSOR_AXIS_NAME)
        generate_rules.activation_attention_out_td = (DATA_AXIS_NAME,
                                                      ATTN_TENSOR_AXIS_NAME)
        generate_rules.activation_q_td = (DATA_AXIS_NAME,
                                          ATTN_TENSOR_AXIS_NAME)
        #TODO: the default qkv and kvcache is sharded on head dim
        # We may change it after we finalize the KVCache design
        generate_rules.attn_o_tnh = (DATA_AXIS_NAME, ATTN_HEAD_AXIS_NAME, None)
        generate_rules.query_tnh = (DATA_AXIS_NAME, ATTN_HEAD_AXIS_NAME, None)
        generate_rules.keyvalue_skh = (DATA_AXIS_NAME, ATTN_HEAD_AXIS_NAME,
                                       None)
        generate_rules.attn_q_weight_dnh = (None, ATTN_HEAD_AXIS_NAME,
                                            ATTN_TENSOR_AXIS_NAME)
        generate_rules.attn_k_weight_dkh = (None, ATTN_HEAD_AXIS_NAME,
                                            ATTN_TENSOR_AXIS_NAME)
        generate_rules.attn_v_weight_dkh = (None, ATTN_HEAD_AXIS_NAME,
                                            ATTN_TENSOR_AXIS_NAME)
        generate_rules.attn_o_weight_nhd = (ATTN_HEAD_AXIS_NAME, None,
                                            ATTN_TENSOR_AXIS_NAME)
        generate_rules.activation_ffw_td = (DATA_AXIS_NAME, None)
        generate_rules.activation_ffw_ted = (DATA_AXIS_NAME, EXPERT_AXIS_NAME,
                                             None)
        generate_rules.ffw_hidden_tf = (DATA_AXIS_NAME, MLP_TENSOR_AXIS_NAME)
        # FFW weights are typically sharded along the hidden dimension (F).
        generate_rules.ffw_weight_df = (None, MLP_TENSOR_AXIS_NAME)
        generate_rules.ffw_weight_fd = (MLP_TENSOR_AXIS_NAME, None)
        # MoE weights are sharded along the expert axis and the hidden dimension.
        generate_rules.moe_weights_edf = (EXPERT_AXIS_NAME, None,
                                          MOE_TENSOR_AXIS_NAME)
        generate_rules.moe_weights_efd = (EXPERT_AXIS_NAME,
                                          MOE_TENSOR_AXIS_NAME, None)
        generate_rules.moe_router_de = (None, EXPERT_AXIS_NAME)

        # Embedding weight: (VocabSize, Dim)
        generate_rules.emb_weight_vd = (MLP_TENSOR_AXIS_NAME, None)
        generate_rules.activation_td = (DATA_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)
        generate_rules.prelogit_td = (DATA_AXIS_NAME, ATTN_TENSOR_AXIS_NAME)
        generate_rules.logits_tv = (DATA_AXIS_NAME, MLP_TENSOR_AXIS_NAME)
        generate_rules.vocab_vd = (VOCAB_AXIS_NAME, None)
        generate_rules.vocab_dv = (None, VOCAB_AXIS_NAME)

    def make_sharding_config(
            self,
            default_rules_cls: ShardingRulesConfig,
            prefill_overrides: dict | None = None,
            generate_overrides: dict | None = None) -> ShardingConfig:
        """Creates the detailed `ShardingConfig` with specific partitioning rules
        and applies any runtime overrides.

        This method populates the `prefill_rules` and
        `generate_rules` with hardcoded sharding rules that are generally
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
        sharding_cfg = ShardingConfig(default_rules_cls=default_rules_cls)
        prefill_rules = sharding_cfg.prefill_rules
        generate_rules = sharding_cfg.generate_rules

        # Extract the overrides from the vllm_config if they are not provided programatically.
        if prefill_overrides is None:
            prefill_overrides = self._get_overrides("prefill")
        if generate_overrides is None:
            generate_overrides = self._get_overrides("generate")

        # Apply default sharding configs
        self._make_default_sharding_config(prefill_rules, generate_rules)

        # Apply overriding the runtime sharding rules
        self._apply_overrides(prefill_rules, prefill_overrides)
        self._apply_overrides(generate_rules, generate_overrides)

        return sharding_cfg

    #TODO: Add __repr__


class ShardingInfo:
    #TODO a sharding info class for visualizing & debugging the sharding performance
    # Will implement it for the next version
    pass
