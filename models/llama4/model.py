# Input flags are stored in below configs
# model_flag_config: Config 
#   d_model: 2048
#   n_layers: 61
#   n_moe_layer: 58
# parallelism_flag_config: Config
#   tp: 2
#   ep: 4
# quant_flag_config: Config
#
# Each Block(attn, mlp etc) has Config Class and Live Class
# There're 2 ways to initialize a Config:
# 1. manual specify as Config(d_model=, num_layer=, ..)
# 2. auto assignment from a config as Config.from_cfg(cfg)


# The foundation Model/ModelConfig class is to-be-implemented

class LlamaConfig(ModelConfig):
    parallelism_flag_config: Config # From User
    quant_flag_config: Config # From User
    model_flag_config: Config # From User
    ...

    def __init__(self):

        # For design demo purpose, need cleaner way for assignment
        self.global_KV_cache_config = kv_cache.GlobalKVCacheConfig(self.model_flag_config.n_layers)
        self.embedder_config = layers.EmbedderConfig.from_cfg(model_flag_config)
        self.transformer_moe_blocks_config = TransformerConfig(
            embedder_config=self.embedder_config,
            attention=LlamaAttentionConfig(model_flag_config),
            moe=LlamaMoEConfig(model_flag_config),
            num_groups=self.model_flag_config.num_groups,
            routing=True,
            ...
        )
        self.transformer_dense_blocks_config = TransformerConfig(
            attention=LlamaAttentionConfig(model_flag_config),
            ffw=layers.FFWConfig(model_flag_config),
            num_groups=self.model_flag_config.num_groups,
            routing=False,
            ...
        )
        ...
    def build_routing_flags(self):
        
        match self.model_flag_config.routing_type:
            case "topk":
                router_type = moe.RouterType.TopK
        ...
        return Config(
            num_experts = self.model_flag_config.num_experts,
            expert_capacity = self.model_flag_config.expert_capacity,
            k = self.model_flag_config.topk
            router_type = router_type
            routed_bias = self.model_flag_config.routed_bias
            routed_scaling_factor = self.model_flag_config.routed_scaling_factor
            ...
        )
    
# Class Model(nnx.module) will be added
class LlamaModel(Model):
    cfg: LlamaConfig
    dtype: jnp.dtype
    embedder: layers.Embedder
    dense_blocks: list[]
    moe_blocks: list[]
    final_norm: layers.RMSNorm
    global_KV_cache: kv_cache.GlobalKVCache
    sharding_cfg: sharding.ShardingConfig
    quantization: Quantization.Quantization = None
        
    def create_embedder(self):
        if self.cfg.embedder:
            return  = self.cfg.embedder.make(name='embedder')
        else:
            # default
            return Embdder(
                ...
            )
    self create_global_runtime_params(self):
        # demo only, need an organized way to process layer-level runtime params
        self.global_runtime_params = [
            layers.RuntimeParams(
                kv_cache=self.global_KV_cache.get_layer_kv_cache(i)
                sharding_cfg=self.sharding_cfg
                quantization:=self.quantization)
            for i in self.cfg.num_layer
        ]

    def create_sharding_cfg(self):
        ...
        sharding = sharding.Sharding(self.cfg.parallelism_flag_config)
        self.runtime_params.update_sharding_cfg(sharding.make_sharding_config())
        return self.runtime_params.sharding_cfg
    
    def create_quantization(self):
        ...
        self.runtime_params.update_quantization(Quantization.QuantizationConfig(self.cfg.quant_flag_config).make())
        return self.runtime_params.quantization

    def create_KV_cache(self):
        ...
        return self.global_KV_cache_config.make(self.sharding_cfg, self.quantization)

    def setup(self) -> None:

        self.embedder = self.create_embedder()
        # a better way to guarantee the order of initialization 
        # i.e. sharding_cfg, quantization should be ready before global_KV_cache etc
        self.sharding_cfg = self.create_sharding_cfg()
        self.quantization = self.create_quantization()
        self.global_KV_cache = self.create_KV_cache()
        self.global_runtime_params = create_global_runtime_params()

        self.dense_blocks = [
            self.cfg.transformer_moe_blocks_config.make(
                name=f'dense_layer_{i}',
                runtime_param=self.global_runtime_params[i])
            for i in self.cfg.dense_layers
        ]
        self.moe_blocks = [
            self.cfg.transformer_dense_blocks_config.make(
                name=f'moe_layer_{i}',
                # demo only, retrieval of runtime_param should not be upon index
                runtime_param=self.global_runtime_params[self.cfg.dense_layers + i])
            for i in self.cfg.moe_layers
        ]
        self.final_norm = base.RMSNorm(
            num_groups=self.cfg,
            epsilon=1e-6,
            ...
        )

    def __call__(
        self,
        inputs,
    ):
        x = self.embedder.encode(inputs)
        for i, block in self.dense_blocks + self.moe_blocks:
            x, new_cache = block(x)
            self.global_KV_cache.append(new_attn_cache)
        
        final_activation = self.final_norm(x)
        decoder_output = self.embedder.decode(final_activation)

        return decoder_output
