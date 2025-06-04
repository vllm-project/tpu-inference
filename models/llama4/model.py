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
# Each Block(attn, mlp etc) has Config Class and Body Class
# There're 2 ways to initialize a block:
# 1. Config(input_flags_config).make()
# 2. Block(d_model=, n_layers=, etc)



class LlamaConfig:
    parallelism_flag_config: Config # From User
    quant_flag_config: Config # From User
    model_flag_config: Config # From User

    embedder_config: EmbedderConfig
    global_KV_cache_config: kv_cache.GlobalKVCacheConfig
    transformer_moe_blocks_config: TransformerConfig
    transformer_dense_blocks_config: TransformerConfig

    ...

    def __init__(self):

        # For deisgn demo purpose, need cleaner way for assignment

        embedder_flags = self.build_embedder_flags()
        attention_flags = self.build_attention_flags()
        moe_flags = self.build_moe_flags()
        ffw_flags = self.build_ffw_flags()


        self.embedder_config = layers.EmbedderConfig(embedder_flags)
        self.global_KV_cache_config = kv_cache.GlobalKVCacheConfig(self.model_flag_config.n_layers)


        self.transformer_moe_blocks_config = TransformerConfig(
            attention=LlamaAttentionConfig(attention_flags),
            moe=LlamaMoEConfig(moe_flags),
            num_groups=self.model_flag_config.num_groups,
            routing=True,
            ...

        )
        self.transformer_dense_blocks_config = TransformerConfig(
            attention=LlamaAttentionConfig(attention_flags),
            ffw=layers.FFWConfig(ffw_flags),
            num_groups=self.model_flag_config.num_groups,
            routing=False,
            ...
        )
        ...
    
    
    def build_embedder_flags(self):
        return Config(
          vocab_size=self.model_flag_configlf.vocab_size
          d_model=self.model_flag_config.d_model
          normalize_embeddings=self.model_flag_config.normalize_embeddings
            ...
        )

    def build_attention_flags(self):
        return Config(
            d_model = self.model_flag_config.d_model,
            num_query_heads = self.model_flag_config.num_query_heads,
            num_kv_heads = self.model_flag_config.num_query_heads,
            attention_type = self.model_flag_config.attention_type,
            qk_nope_head_dim = self.model_flag_config.qk_nope_head_dim,
            qk_rope_head_dim = = self.model_flag_config.qk_rope_head_dim,
            v_head_dim = self.model_flag_config.v_head_dim
            enable_dropout = self.model_flag_config.enable_dropout
            ...
        )
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
    def build_moe_flags(self):
        return Config(
            d_model = self.model_flag_config.d_model,
            base_moe_mlp_dim = self.model_flag_config.base_moe_mlp_dim,
            mlp_activations = self.model_flag_config.mlp_activations,
            enable_dropout = self.model_flag_config.enable_dropout,
            num_experts = self.model_flag_config.num_experts,
            num_experts_per_tok = self.model_flag_config.num_experts_per_tok,
            router_config = self.build_routing_flags()
            ...
        )
    def build_ffw_flags(self):
        return Config(
            d_model = self.model_flag_config.d_model,
            base_mlp_dim = self.model_flag_config.base_mlp_dim,
            num_dense_layers = self.model_flag_config.num_dense_layers,
            mlp_activations = self.model_flag_config.mlp_activations,
            enable_dropout = self.model_flag_config.enable_dropout,
            ...
        )
    
    

class LlamaModel(nn.Module):
    cfg: LlamaConfig
    dtype: jnp.dtype
    embedder: 
    dense_blocks: list[]
    moe_blocks: list[]
    final_norm: 
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

    def create_sharding_cfg(self):
        ...
        sharding = sharding.Sharding(self.cfg.parallelism_flag_config)
        return sharding.make_sharding_config()
    
    def create_quantizaiton(self):
        ...
        return Quantization.QuantizationConfig(self.cfg.quant_flag_config)

    def create_KV_cache(self):
        ...
        return self.global_KV_cache_config.make(self.sharding_cfg, self.quantization)

    def setup(self) -> None:

        self.embedder = self.create_embedder()
        self.sharding_cfg = self.create_sharding_cfg()
        self.quantization = self.create_quantizaiton()
        self.global_KV_cache = self.create_KV_cache()


        self.dense_blocks = [
            self.cfg.transformer_moe_blocks_config.make(
                name=f'layer_{i}',
                sharding_cfg=self.sharding_cfg,
                quantization=self.quantization)
            for i in self.cfg.dense_layers
        ]
        self.moe_blocks = [
            self.cfg.transformer_dense_blocks_config.make(
                name=f'layer_{i}',
                sharding_cfg=self.sharding_cfg,
                quantization=self.quantization)
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
