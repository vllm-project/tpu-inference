@dataclass
class AttentionConfig:
    d_model: int
    num_heads: int
    num_kv_heads: int   
    qk_dim: int
    v_dim int

    def __init__(self, yaml_config):
        ...
    def make(self, kv_cache=None, sharding_cfg=None, quantization=None) -> Attention:
        ...
        return Attention(
            d_model=self.d_model
            num_heads=self.num_heads
            num_kv_heads=self.num_kv_heads 
            qk_dim=self.qk_dim
            v_dim=self.v_dim
            cfg=self,
            kv_cache=kv_cache,
            sharding_cfg=sharding_cfg,
            quantization=quantization,
        )

@dataclass
class Attention(nn.Module):
    """Attention Block"""
    d_model: int
    num_heads: int
    num_kv_heads: int   
    qk_dim: int
    v_dim: int
    cfg: AttentionConfig = None
    kv_cache: KVCache = None
    sharding_cfg: ShardingConfig = default_sharding()
    quantization: Quantization | None = None

    def setup(self):
        ...
    
    def __call__(
        self, 
        x,
        attn_cache
    )

    def get_cfg(self) -> AttentionConfig:
        ...
    
