@dataclass
class TransformerConfig(Config):
    """
    light weighted transformer config, which includes config for all sub-modules
    it uses make() to create the live module from this config
    """
    embeddingsConfig: EmbedderConfig 
    attention: AttentionConfig
    moe: MoEConfig = None
    ffw: FFWConfig = None

    num_groups: int
    routing: bool = False
    ...

    @classmethod
    def from_cfg(cls, flags_cfg: dict):
        required_params = {f.name for f in fields(cls)}
        provided_params = set(flags_cfg.keys())
        missing_params  = required_params - provided_params

        if missing_params:
            ...

        transformer_flags = {k: flags_cfg[k] for k in required_params}
        return cls(**transformer_flags)

    def make(self, runtime_param: Optional[layer.RuntimeParams] = None) -> TransformerBlock:
        
        self_attn = self.attention.make(runtime_param)
        post_attention_norm = base.RMSNorm(
            num_groups=self.num_groups,
            epsilon=1e-6,
            #...
        )

        if self.routing:
            router_norm = base.RMSNorm(
                num_groups=self.num_experts,
                epsilon=1e-6,
                #...
            )
            mlp = self.moe.make(runtime_param)
        else:
            mlp = self.ffw.make(runtime_param)
            post_mlp_norm = base.RMSNorm(
                num_groups=self.num_groups,
                epsilon=1e-6,
                #...
            )

        return TransformerBlock(
            cfg=self,
            embeddings=self.embeddings,
            self_attn=self_attn,
            post_attention_norm=post_attention_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )

@dataclass
class TransformerBlock(nnx.Module):
    """
    A heavy weight module which serves as the stateful live blocks in serving
    """
    cfg: TransformerConfig
    embeddings: Embedder
    self_attn: Attention
    post_attention_norm: RMSNorm
    mlp: FFW
    post_mlp_norm: RMSNorm
    #...

    def setup(self) -> None:
        ...
    
    def __call__(
        self,
        x,
        positions,
        attn_cache
    ):
        x = self.embeddings
        score, new_cache = self.self_attn(x)
        x = self.post_attention_norm(x + score)
        y = self.mlp(x)
        logits = self.post_mlp_norm(x + y)
        ...
        return logits, new_cache

    # Other methods
    ...

    
