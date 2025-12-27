# Copyright 2025 Google LLC
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

import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List, Optional

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference import envs, utils

if TYPE_CHECKING:
    from vllm.v1.configs.vllm_config import VllmConfig

MESH_AXIS_NAMES = ("data", "attn_dp", "expert", "model")
MESH_AXIS_NAMES_2D = ('data', 'model')


class ShardingAxisNameBase:
    """Base class for sharding axis names."""
    SEQUENCE = ('data', 'attn_dp')
    ATTN_DATA = ('data', 'attn_dp')
    MLP_DATA = 'data'
    ATTN_HEAD = 'model'
    ATTN_TENSOR = None
    MLP_TENSOR = ('attn_dp', 'model', 'expert')
    MOE_TENSOR = ('attn_dp', 'model')
    EXPERT = ('attn_dp', 'expert', 'model')
    VOCAB = ('expert', 'attn_dp', 'model')


class ShardingAxisName2D:
    """Sharding axis names for 2D data parallelism scenarios.
    NOTE(wenxindongwork): The new MoE kernel expects a 2D mesh for now.
    We should use ShardingAxisNameBase once the new MoE kernel supports
    more general mesh shapes. For now, this is the default sharding axes.
    """
    SEQUENCE = 'data'
    ATTN_DATA = 'data'
    MLP_DATA = 'data'
    ATTN_HEAD = 'model'
    ATTN_TENSOR = None
    MLP_TENSOR = 'model'
    MOE_TENSOR = 'model'
    EXPERT = 'model'
    VOCAB = ('data', 'model')


try:
    _use_base_sharding = envs.NEW_MODEL_DESIGN
    if _use_base_sharding:
        ShardingAxisName = ShardingAxisNameBase
    else:
        ShardingAxisName = ShardingAxisName2D
except Exception:
    ShardingAxisName = ShardingAxisName2D


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
    attention_data_parallelism: int = 1


class ShardingConfigManager:
    """Manages sharding configuration parsing and access from vLLM config.

    Usage:
        sharding_config = ShardingConfigManager.from_vllm_config(vllm_config)
        tp_size = sharding_config.tp_size

    During initialization, we set `vllm_config.sharding_config` to
    `ShardingConfigManager.from_vllm_config(vllm_config)`, so you can access
    `vllm_config.sharding_config.tp_size` directly.
    """

    def __init__(self,
                 sharding_strategy: ShardingStrategy,
                 device_indexes: Optional[List] = None):

        self.sharding_strategy: ShardingStrategy = sharding_strategy
        self.device_indexes: Optional[List[int]] = device_indexes
        self._total_devices: int = int(
            math.prod(asdict(sharding_strategy).values()))
        if device_indexes:
            assert self._total_devices == len(device_indexes)

    @classmethod
    def from_vllm_config(cls,
                         vllm_config: 'VllmConfig') -> 'ShardingConfigManager':

        sharding_strategy = vllm_config.additional_config.get(
            "sharding", {}).get("sharding_strategy", {})
        parallel_config = vllm_config.parallel_config
        tensor_parallelism = parallel_config.tensor_parallel_size
        data_parallelism = parallel_config.data_parallel_size
        expert_parallelism = sharding_strategy.get("expert_parallelism", 1)
        sequence_parallelism = sharding_strategy.get("sequence_parallelism", 1)
        device_indexes = sharding_strategy.get("device_indexes", None)

        enable_dp_attention = sharding_strategy.get("enable_dp_attention",
                                                    False)
        if enable_dp_attention:
            # Replicate attention layer when num_kv_heads < TP
            num_kv_heads = 1 if vllm_config.model_config.use_mla else vllm_config.model_config.get_total_num_kv_heads(
            )
            cache_dtype = vllm_config.cache_config.cache_dtype
            if cache_dtype == 'auto':
                cache_dtype = vllm_config.model_config.dtype
            kv_dtype = utils.get_jax_dtype_from_str_dtype(
                cache_dtype) or jnp.bfloat16
            packing = 4 // jnp.dtype(kv_dtype).itemsize

            # The default head dim is 128 but 64 is also supported as a special case.
            if vllm_config.model_config.get_head_size() == 64:
                packing *= 2

            # When num_kv_heads * 2 / packing < TP, tensor parallelism would
            # duplicate KV heads across devices, wasting kv cache memory.
            # Use attention DP instead to reduce per-device num_kv_heads and
            # eliminate this waste.
            num_kv_heads_per_device_in_kv_cache = (num_kv_heads * 2) / packing
            attn_dp = max(
                int(tensor_parallelism // num_kv_heads_per_device_in_kv_cache),
                1)
            tensor_parallelism = tensor_parallelism // attn_dp
        else:
            attn_dp = 1

        sharding_strategy = ShardingStrategy(
            tensor_parallelism=tensor_parallelism,
            data_parallelism=data_parallelism,
            expert_parallelism=expert_parallelism,
            sequence_parallelism=sequence_parallelism,
            attention_data_parallelism=attn_dp)

        # Must override here to avoid vLLM spinning up multiple DP engines.
        if vllm_config.parallel_config.data_parallel_size > 1:
            vllm_config.parallel_config.data_parallel_size = 1
            vllm_config.parallel_config.data_parallel_rank = 0
            vllm_config.parallel_config.data_parallel_size_local = 1

        cls.validate(vllm_config, sharding_strategy)
        return cls(sharding_strategy, device_indexes)

    @classmethod
    def validate(cls, vllm_config, sharding_strategy):
        total_dp_size = sharding_strategy.data_parallelism * sharding_strategy.attention_data_parallelism
        if total_dp_size > 1:
            if vllm_config.speculative_config is not None:
                raise ValueError(
                    f"Speculative decoding is not supported with data parallelism "
                    f"(DP size: {total_dp_size}). Please disable speculative decoding or "
                    f"set data parallelism to 1.")
            if vllm_config.lora_config is not None:
                raise ValueError(
                    f"LoRA is not supported with data parallelism "
                    f"(DP size: {total_dp_size}). Please disable LoRA or "
                    f"set data parallelism to 1.")
        if sharding_strategy.attention_data_parallelism > 1:
            if not envs.NEW_MODEL_DESIGN:
                raise ValueError(
                    "Must run Attention DP with NEW_MODEL_DESIGN enabled. Please set "
                    "NEW_MODEL_DESIGN=True")

    @property
    def total_dp_size(self) -> int:
        return self.sharding_strategy.data_parallelism * self.sharding_strategy.attention_data_parallelism

    @property
    def model_dp_size(self) -> int:
        return self.sharding_strategy.data_parallelism

    @property
    def attn_dp_size(self) -> int:
        return self.sharding_strategy.attention_data_parallelism

    @property
    def tp_size(self) -> int:
        return self.sharding_strategy.tensor_parallelism

    @property
    def expert_size(self) -> int:
        return self.sharding_strategy.expert_parallelism

    @property
    def sequence_size(self) -> int:
        return self.sharding_strategy.sequence_parallelism

    @property
    def total_devices(self) -> int:
        return self._total_devices

    def __str__(self):
        return (f"ShardingConfigManager(total_devices={self.total_devices}, "
                f"sharding_strategy={self.sharding_strategy}, "
                f"device_indexes={self.device_indexes})")


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
                 vllm_config: 'VllmConfig' = None):
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
        prefill_rules.activation_attention_td = (ShardingAxisName.ATTN_DATA,
                                                 ShardingAxisName.ATTN_TENSOR)
        prefill_rules.activation_attention_out_td = (
            ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_TENSOR)
        prefill_rules.activation_q_td = (ShardingAxisName.ATTN_DATA,
                                         ShardingAxisName.ATTN_TENSOR)
        #TODO: the default qkv and kvcache is sharded on head dim
        # We may change it after we finalize the KVCache design
        prefill_rules.attn_o_tnh = (ShardingAxisName.ATTN_DATA,
                                    ShardingAxisName.ATTN_HEAD, None)
        prefill_rules.query_tnh = (ShardingAxisName.ATTN_DATA,
                                   ShardingAxisName.ATTN_HEAD, None)
        prefill_rules.keyvalue_skh = (ShardingAxisName.ATTN_DATA,
                                      ShardingAxisName.ATTN_HEAD, None)

        # Populate Generate (Decode) Config
        # During decode, batch size is the large dimension, so we shard along the batch axis.
        generate_rules.activation_attention_td = (ShardingAxisName.ATTN_DATA,
                                                  ShardingAxisName.ATTN_TENSOR)
        generate_rules.activation_attention_out_td = (
            ShardingAxisName.MLP_DATA, ShardingAxisName.ATTN_TENSOR)
        generate_rules.activation_q_td = (ShardingAxisName.ATTN_DATA,
                                          ShardingAxisName.ATTN_TENSOR)
        #TODO: the default qkv and kvcache is sharded on head dim
        # We may change it after we finalize the KVCache design
        generate_rules.attn_o_tnh = (ShardingAxisName.ATTN_DATA,
                                     ShardingAxisName.ATTN_HEAD, None)
        generate_rules.query_tnh = (ShardingAxisName.ATTN_DATA,
                                    ShardingAxisName.ATTN_HEAD, None)
        generate_rules.keyvalue_skh = (ShardingAxisName.ATTN_DATA,
                                       ShardingAxisName.ATTN_HEAD, None)
        generate_rules.attn_q_weight_dnh = (None, ShardingAxisName.ATTN_HEAD,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.attn_k_weight_dkh = (None, ShardingAxisName.ATTN_HEAD,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.attn_v_weight_dkh = (None, ShardingAxisName.ATTN_HEAD,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.attn_o_weight_nhd = (ShardingAxisName.ATTN_HEAD, None,
                                            ShardingAxisName.ATTN_TENSOR)
        generate_rules.activation_ffw_td = (ShardingAxisName.MLP_DATA, None)
        generate_rules.activation_ffw_ted = (ShardingAxisName.MLP_DATA,
                                             ShardingAxisName.EXPERT, None)
        generate_rules.ffw_hidden_tf = (ShardingAxisName.MLP_DATA,
                                        ShardingAxisName.MLP_TENSOR)
        # FFW weights are typically sharded along the hidden dimension (F).
        generate_rules.ffw_weight_df = (None, ShardingAxisName.MLP_TENSOR)
        generate_rules.ffw_weight_fd = (ShardingAxisName.MLP_TENSOR, None)
        # MoE weights are sharded along the expert axis and the hidden dimension.
        generate_rules.moe_weights_edf = (ShardingAxisName.EXPERT, None,
                                          ShardingAxisName.MOE_TENSOR)
        generate_rules.moe_weights_efd = (ShardingAxisName.EXPERT,
                                          ShardingAxisName.MOE_TENSOR, None)
        generate_rules.moe_router_de = (None, ShardingAxisName.EXPERT)

        # Embedding weight: (VocabSize, Dim)
        generate_rules.emb_weight_vd = (ShardingAxisName.MLP_TENSOR, None)
        generate_rules.activation_td = (ShardingAxisName.MLP_DATA,
                                        ShardingAxisName.ATTN_TENSOR)
        generate_rules.prelogit_td = (ShardingAxisName.MLP_DATA,
                                      ShardingAxisName.MLP_TENSOR)
        generate_rules.logits_tv = (ShardingAxisName.MLP_DATA,
                                    ShardingAxisName.MLP_TENSOR)
        generate_rules.vocab_vd = (ShardingAxisName.VOCAB, None)
        generate_rules.vocab_dv = (None, ShardingAxisName.VOCAB)

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
