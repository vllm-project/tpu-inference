import abc
from collections import namedtuple
from dataclasses import dataclass
import shardings
from typing import Any, Dict, List


################################
# Model Recipe classes and utils
################################
@dataclass
class ModelRecipe:
    base_num_decoder_layers: 48
    base_emb_dim = 5120
    d_model=4096,
    base_mlp_dim = 16384,
    vocab_size = 202048,
    base_moe_mlp_dim: 8192,
    num_kv_heads=8,
    num_query_heads=32,
    tokenizer_type = "huggingface",
    tokenizer_path = "meta-llama/Llama-4-Scout-17B-16E",
    normalization_layer_epsilon = 1e-05,
    # Llama4 models apply an L2Norm to the Query and Keys after RoPE
    use_qk_norm = True
    # Every fourth layer should NOT use RoPE
    nope_layer_interval = 4
    

@dataclass
class MoEModelRecipe(ModelRecipe):
    num_experts = 16,
    capacity_factor = -1.0 # TODO: this will be removed once we support dropless with megablox/ragged_dot
    shared_experts = 1
    num_experts_per_tok = 1
    # TODO: delete the following variables once we add support for dropless with megablox/ragged_dot
    sparse_matmul = False
    megablox = False


###################################
# Sharding Recipe classes and utils
###################################
ShardingRecipe = namedtuple("ShardingRecipe", ["attention", "feedforward", "vocab"])

class AbstractLayerShardingRecipe(abc.ABC):
    SUPPORTED_MODES = ["prefill", "decode"]
    def __init__(self, class_name: str):
        self.class_name = class_name
        self.prefill = None
        self.decode = None
        self.logical_axes_rules = None
    
    def validate_overrides(self, override_rules):
        if override_rules:
            for mode in  override_rules:
                if mode not in self.SUPPORTED_MODES:
                    raise ValueError(f"Provided {mode} in override_rules is not part of the supported list: {self.SUPPORTED_MODES}.")
            override_rules_dict = override_rules[mode]
            for (axis_name, rules) in override_rules_dict.items():
                if axis_name in getattr(self, f"{mode}.{axis_name}"):
                    for rule in rules:
                        if rule not in shardings.MESH_AXES:
                            raise ValueError(f"Mesh axis in {rule} is not one of the supported axes: {shardings.MESH_AXES}.")
                else:
                    raise ValueError(f"Logical axis name {axis_name} is not part of the supported list {self.mode}") # TODO: May need to create string representation of dataclass.

    def overwrite_rules(self, override_rules):
        if override_rules:
            self.validate_overrides()
            for mode in self.SUPPORTED_MODES:
                override_rules_dict = override_rules["mode"]
                for (axis_name, rule) in override_rules_dict.items():
                    setattr(self, f"{mode}.logical_axes_rules.{axis_name}" , rule)

    @abc.abstractmethod
    def base_recipe(self):
        pass

    # def __init__(self, 
    #              attention_type,
    #              feedforward_type,
    #              override_rules: Dict[str, Any]):
    #     self.attention = AttentionLayerShardingRecipe(attention_type)
    #     self.feedforward = FFWLayerShardingRecipe(feedforward_type)
    #     # self.vocab = VocabShardingRecipe()
    #     self.override_rules = override_rules
    
    # def validate_overrides(self):
    #     for sharding_layer in self.override_rules:
    #         if sharding_layer not in self.SUPPORTED_SHARDINGS:
    #             raise ValueError("Sharding type {sharding_layer} is not one of supported sharding layers: {self.SUPPORTED_SHARDINGS}.")
    #         sharding_layer_dict = self.override_rules[sharding_layer]
    #         if ""
    #         super().validate_overrides()

    # def build_attention(self,
    #           attention_type: str,
    #           feedforward_type: str,
    #           attention_prefill: shardings.AttentionLogicalAxesRules = None,
    #           attention_decode: shardings.AttentionLogicalAxesRules = None,
    #           feedforward_prefill: shardings.DenseLayerLogicalAxesRules = None,
    #           feedforward_decode: shardings.DenseLayerLogicalAxesRules = None,
    #           )


class AttentionLayerShardingRecipe(AbstractLayerShardingRecipe):
    SUPPORTED_CLASSES = ["attention", "deepseek_attention"]

    def __init__(self,
                 class_name: str,
                 prefill:  shardings.AttentionLogicalAxesRules = None,
                 decode: shardings.AttentionLogicalAxesRules = None):
        self.class_name = class_name
        self.prefill = self.base_recipe() if prefill is None else prefill
        self.decode = self.base_recipe() if decode is None else decode

    def base_recipe(self) -> shardings.AttentionLogicalAxesRules:
        if self.class_name not in self.SUPPORTED_CLASSES:
            raise ValueError("Sharding type {self.class_name} is not one of supported classes: {self.SUPPORTED_CLASSES}.")
        
        if self.class_name == "attention":
            return shardings.AttentionLogicalAxesRules() # assuming it has initializations for attention layers (like in base.yml)
    

class FFWLayerShardingRecipe(AbstractLayerShardingRecipe):
    SUPPORTED_CLASSES = ["dense", "moe"]

    def __init__(self,
                 class_name: str,
                 prefill:  shardings.DenseLayerLogicalAxesRules = None,
                 decode: shardings.DenseLayerLogicalAxesRules = None):
        self.class_name = class_name,
        self.prefill = self.base_recipe() if prefill is None else prefill
        self.decode = self.base_recipe() if decode is None else decode
        self.logical_rules = None

    def base_recipe(self) -> shardings.DenseLayerLogicalAxesRules:
        if self.class_name not in self.SUPPORTED_CLASSES:
            raise ValueError("Sharding type {self.class_name} is not one of supported classes: {self.SUPPORTED_CLASSES}.")
        if self.class_name == "dense":
            return shardings.DenseLayerLogicalAxesRules()
        # elif self.class_name == "moe":
        #     self.logical_axes_rules = shardings.AttentionLogicalAxesRules()

##################################
# Serving Recipe classes and utils
##################################