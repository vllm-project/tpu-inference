# Current implementation split the Cache and method.
# TODO: we could discuss want to encapsulate the KVCache and updater into the same class
# 
@dataclass
class KVCacheConfig(Config)::
    """Configuration for KV cache."""
    batch_size: int
    cache_len: int
    num_kv_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.float32


class KVCache:
    cfg: KVCacheConfig,
    key_cache: Dict[str, Float[ArrayT, '...']]
    value_cache: Dict[str, Float[ArrayT, '...']]
    length: Int[ArrayT, '...']
    mesh: Mesh,
    sharding_cfg: ShardingConfig,

    def __init__(self, cfg, sharding_cfg, mesh):
        self.cfg = cfg,
        self.mesh = mesh
        self.sharding_cfg = sharding_cfg
        self.create_sharding()
        self._initialize_caches()
    
    def _initialize_caches(self,) -> 'KVCache':
        cache_shape = (
            self.cfg.batch_size,
            self.cfg.cache_len,
            self.cfg.num_kv_heads,
            self.cfg.head_dim
        )
        self.key_cache['prefill'] = jnp.zeros(
            cache_shape,
            dtype=self.cfg.dtype,
            sharding=self.kv_sharding['prefill'])
        self.key_cache['decode']= jnp.zeros(
            cache_shape,
            dtype=self.cfg.dtype,
            sharding=self.kv_sharding['decode'])
        self.value_cache['prefill']  = jnp.zeros(
            cache_shape,
            dtype=self.cfg.dtype,
            sharding=self.kv_sharding['prefill'])
        self.value_cache['decode'] = jnp.zeros(
            cache_shape,
            dtype=self.cfg.dtype,
            sharding=self.kv_sharding['decode'])
        self.length_B = jnp.zeros((self.cfg.batch_size,), dtype=jnp.dtype),

    def create_sharding(self, ):
        self.kv_sharding = dict()
        self.kv_sharding['prefill'] =  NamedSharding(
            self.mesh, P(self.sharding_cfg.keyvalue_prefill_mode_cache_bsnh.get_axes()))
        self.kv_sharding['decode'] =  NamedSharding(
            self.mesh, P(self.sharding_cfg.keyvalue_generate_mode_cache_bsnh.get_axes()))
    
class KVCacheUpdater:
    # Split update_cache for prefill and generate
    def update_cache(self, operand, cache, sequence_len, dtype):
        operand_BSNH = operand.astype(cache.dtype)
        sequence_len = sequence_len_B
        batch_size, max_seq_len = operand_BSNH.shape[:2] # it's usually 1 for generating, unless speculative-decoding
        cache_size = cache.shape[-3] # How many tokens' KV would be stored

        # [3, 1, 2] -> [[3], [1], [2]], seq0 has 3 tokens, seq1 has 1 token, etc
        shift_B = sequence_len_B[:, None]
        batch_idx = jnp.arange(batch_size)[:, None]
        # [[3, 4, 5], [1, 2, 3], [2, 3, 4]]
        offset = (iota(dtype, (batch_size, max_seq_len),  1) + shift) % cache_size
        # update[i, j] will be stored in cache[batch_idx[i, j], offset[i, j]]
        # namely cache[i, sequence_len[i] + j]
        cache = cache.at[batch_idx, offset].set(operand, mode='drop')

        return cache

