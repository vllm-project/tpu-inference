from dataclasses import dataclass
import recipe
from vllm.config import VllmConfig


@dataclass
# self.model = Llama4Scout(self.experiment_recipe) <-- can initialize Llama4Model using this reicpe.
class Llama4ScoutRecipe(ExperimentConfig):
    def __init__(self, overrides: VllmConfig):
        self.model = recipe.MoEModelRecipe(
            base_num_decoder_layers = 48,
            base_emb_dim = 5120,
            d_model = 4096,
            act='gelu',
            base_mlp_dim = 16384,
            vocab_size = 202048,
            base_moe_mlp_dim = 8192,
            num_kv_heads = 8,
            num_query_heads = 32,
            tokenizer_type = "huggingface",
            tokenizer_path = "meta-llama/Llama-4-Scout-17B-16E",
            normalization_layer_epsilon = 1e-05,
            num_experts = 16,
            capacity_factor = -1.0, # TODO: this will be removed once we support dropless with megablox/ragged_dot
            shared_experts = 1,
            num_experts_per_tok = 1,
            # Llama4 models apply an L2Norm to the Query and Keys after RoPE
            use_qk_norm = True,
            # Every fourth layer should NOT use RoPE
            nope_layer_interval = 4,

            # TODO: delete the following variables once we add support for dropless with megablox/ragged_dot
            sparse_matmul = False,
            megablox = False
        )
        # optimal sharding settings
        self.sharding = recipe.ShardingRecipe(
            attention=recipe.AttentionLayerShardingRecipe(),
            feedforward=recipe.FFWLayerShardingRecipe(),
            vocab=recipe.VocabShardingRecipe(),
        ),
        # user_overrides
        self.sharding.attention.prefill.activation_attention_batch = ['data', 'tensor'],
        # self.serving_config = ServingRecipe(

        # ) # minimal serving settings
        