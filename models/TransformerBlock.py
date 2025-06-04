@dataclass
class TransformerConfig:
    embeddings: 
    attention: AttentionConfig
    moe: MoEConfig = None
    ffw: FFWConfig = None

    num_groups: int
    routing: bool = False
    ...

    def __init__(self, yaml_config):
        ...

    def make(self, kv_cache=None, sharding_cfg=None, quantization=None):
        
        self_attn = self.attention.make(
            sharding_cfg=sharding_cfg,
            kv_cache=kv_cache,
            quantization=quantization)
        post_attention_norm = base.RMSNorm(
            num_groups=self.num_groups,
            epsilon=1e-6,
            ...
        )

        if self.routing:
            router_norm = base.RMSNorm(
                num_groups=self.num_experts,
                epsilon=1e-6,
                ...
            )
            mlp = self.moe.make(
                sharding_cfg=sharding_cfg,
                quantization=quantization
            )
        else:
            mlp = self.ffw.make(
                sharding_cfg=sharding_cfg,
                quantization=quantization                
            )
        post_mlp_norm = base.RMSNorm(
            num_groups=self.num_groups,
            epsilon=1e-6,
            ...
        )


        return TransformerBlock(
            cfg=self,
            KV_cache=kv_cache,
            embeddings=self.embeddings,
            self_attn=self_attn,
            quant=quantization,
            post_attention_norm=post_attention_norm,
            mlp=mlp,
            post_mlp_norm=post_mlp_norm,
        )



@dataclass
class TransformerBlock(nn.Module):
    cfg: TransformerConfig
    KV_cache: kv_cache.KV_cache
    embeddings: 
    self_attn: 
    quant:
    post_attention_norm:
    mlp:
    post_mlp_norm:


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
        return logits

    # Other methods
    ...

    
