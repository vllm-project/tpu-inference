# A lightweight block serves as a blue-print.
# Could be created either from cfg or explicit parameters
# self.make() create the live module from the config
@dataclass(frozen=True)
class AttentionConfig:
    d_model: int
    num_heads: int
    num_kv_heads: int   
    qk_dim: int
    v_dim int
    ...

    @classmethod
    def from_cfg(cls, flags_cfg: dict):
        required_params = {f.name for f in fields(cls)}
        provided_params = set(flags_cfg.keys())
        missing_params  = required_params - provided_params

        if missing_params:
            ...

        attention_flags = {k: flags_cfg[k] for k in required_params}
        return cls(**attention_flags)

    def make(self, name, runtime_param: Optional[layer.RuntimeParams] = None) -> Attention:
        ...
        return Attention(
            cfg=self,
            kernel=kernel,
            attention_mask=attention_mask,
            query_pos=query_pos,
            kv_cache=runtime_param.kv_cache,
            sharding_cfg=runtime_param.sharding_cfg,
            quantization=runtime_param.quantization,
        )

# A heavy weight module which serves as the stateful live blocks in serving
@dataclass
class Attention(nnx.Module):
    """Attention Block"""
    kernel
    attention_mask
    query_pos
    cfg: AttentionConfig
    kv_cache: KVCache = create_default_KVCache()
    sharding_cfg: ShardingConfig = default_sharding()
    quantization: Quantization | None = None
    ...
    # other attributes

    def setup(self):
        ...
    def input_projection(self, x):
        #q, k, v = kernel * x
        return (q,k,v)
    def sharding(op_mode):
        ...
    def __call__(
        self, 
        x,
        attn_cache
    ):
        (self.q, self.k, self.v) = self.input_projection(x, self.kernel)
        self.sharding()
        attn_weight = jnp.einsum("btkgd,bskd->bkgts", self.q, self.k)
        attn_weight = self.sharding(attn_weight)
        logits = einsum("bkgts,bskd->btkgd", attn_weights, self.v)
        out = jnp.einsum("btkgd,btkgdh->bth", logits, self.kernel)
        self.kv_cache.update(self.k, self.v, ...)
        return out

    def get_cfg(self) -> AttentionConfig:
        ...
    
    # other methods