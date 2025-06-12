@dataclass
class TransformerConfig(Config):
    """
    light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    """
    embeddings_cfg: EmbedderConfig 
    attention_cfg: AttentionConfig
    kv_cache_cfg: KVCacheConfig
    moe_cfg: MoEConfig = None
    ffw_cfg: FFWConfig = None
    router_cfg: RoutingConfig = None
    cfg: Any
    num_groups: int
    routing: bool = False

    def from_cfg(self, flags_cfg):
        self.embeddings_cfg = EmbedderConfig.from_cfg(flags_cfg)
        self.attention_cfg = AttentionConfig.from_cfg(flags_cfg)
        self.kv_cache_cfg = KVCacheConfig.from_cfg(flags_cfg)
        if self.routing:
            self.moe_cfg = MoEConfig.from_cfg(flags_cfg)
            self.router_cfg = RoutingConfig.from_cfg(flags_cfg)
        self.ffw_cfg = FFWConfig.from_cfg(flags_cfg)
        self.cfg = flags_cfg

@dataclass
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving
    """
    cfg: TransformerConfig
    kernel_init: Initializer # TODO create factories for initializer(?)
    mesh: Mesh
    sharding_cfg: ShardingConfig
    quant: Quantization | None = None

    def _create_module(self, module_cls: Type[nnx.Module], cfg: Any, **overrides) -> nn.Module:
        args = {
            "mesh": self.mesh,
            "kernel_init": self.kernel_init,# TODO create factories for initializer(?)
            "sharding_cfg": self.sharding_cfg,
            "quant": self.quant
        }
        args.update(overrides)
        return module_cls(cfg=cfg, **args)

    def setup(self) -> None:
        self.embedder = Embedder(
            cfg=self.embeddings_cfg,
            mesh=self.mesh,
            embedding_init=self.kernel_init,
            sharding_cfg=self.sharding_cfg,
            quant=self.quant
        )
        
        self.attn = self._create_module(Attention, cfg=self.attention_cfg)
        
        self.kv_cache = KVCache(
            cfg=self.kv_cache_cfg,
            mesh=self.mesh,
            sharding_cfg=self.sharding_cfg,
        )

        self.mlp = self._create_module(FFW, cfg=self.ffw_cfg)

        if cfg.routing:
            self.router = self._create_module(Router, cfg=self.router_cfg)
            self.moe = self._create_module(MoE, cfg=self.moe_cfg)
        
        self.post_attention_norm = self._create_module(
            RMSNorm,
            cfg={"dims": self.cfg.d_model},
            scale_init=self.kernel_init,
            kernel_init=None
        )
        self.post_mlp_norm = self._create_module(
            RMSNorm,
            cfg={"dims": self.cfg.d_model},
            scale_init=self.kernel_init,
            kernel_init=None
        )


    def __call__(
        self,
        x,
        positions,
        attn_cache,
        op_mode,
        kv_cache_updater,
    ):
        x = self.embedder(x)
        # use the current interface for kv_cache in attention()
        kv_cache = (self.kv_cache.key_cache[op_mode], self.kv_cache.value_cache[op_mode])
        new_cache, score = self.self_attn(x, op_mode, kv_cache, attention_metadata)        
        x = self.post_attention_norm(x + score)
        if self.cfg.routing:
            y = self.moe(x, op_mode)
        else:
            y = self.mlp(x, op_mode)
        logits = self.post_mlp_norm(x + y)

        return new_cache, logits

    # Other methods
    ...

    
