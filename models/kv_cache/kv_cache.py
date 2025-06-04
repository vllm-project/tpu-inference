# the idea's from Gemax to create a hier as:
# PartialCache(sharded Cache) <- KVCache(single layer) <- GlobalKVCache(multi-layer)

class PartialCache:
    # to decide which proto to use for k,v
    keys: MaybeQuantizedKVCacheTensor
    values: MaybeQuantizedKVCacheTensor
    length: Int[ArrayT, ]
    ....

    def zeros() -> 'PartialCache':
        ...
    def shardings(
        ...
    ) -> 'PartialCache':
        ...
    

class KVCache:
    cfg: KVCacheConfig,
    sharding: ShardingConfig,
    prefill_kv_cache:  PartialCache,
    generation_kv_cache:  PartialCache,
    quant_kv_cache: bool,

    def __init__(self):
        ...
    def make(self) -> 'KVCache':
        ...
    def update(
        self,
        k,
        v,
        op_mode,
        ...
    ) -> 'KVCache':
        ...

    def sharding(self) -> 'KVCache':
        ...
    
class GlobalKVCacheConfig:
    n_layer: int

    def __init__(self):
        ...
    def make(self, sharding_cfg=None, quantization=None) -> 'GlobalKVCache':
        cache = []
        for i in cfg.layer:
            layer_cache = KVCache.make()
            cache.append(layer_cache)
        
        return GlobalKVCache(
            cfg=self,
            cache=cache,
            sharding_cfg=sharding_cfg,
            quantization=None)

class GlobalKVCache:
    cfg: KVCacheConfig
    cache: list['KVCache']
    sharding_cfg: ShardingConfig = default_sharding()
    quantization: Quantization | None = None


    def __init__(self):
        ...
    def make(self) -> 'GlobalKVCache':
        cache = []
        for i in cfg.layer:
            layer_cache = KVCache.make()
            cache.append(layer_cache)
        
        return GlobalKVCache(cache)
    def get_cfg(self) -> GlobalKVCacheConfig:
        ...
    def shardings(self) -> 'GlobalKVCache':
        ...
