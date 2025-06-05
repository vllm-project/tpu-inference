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
    def sharding(
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
    
class GlobalKVCacheConfig(Config):
    n_layer: int

    def __init__(self):
        ...
    def make(self, runtime_param: Optional[layer.RuntimeParams] = None) -> 'GlobalKVCache':
        cache = []
        for i in cfg.layer:
            layer_cache = KVCache.make()
            cache.append(layer_cache)
        
        return GlobalKVCache(
            cfg=self,
            cache=cache,
            sharding_cfg=layers.runtime_param.sharding_cfg,
            quantization=layers.runtime_param.quantization)

class GlobalKVCache:
    cfg: KVCacheConfig
    # or better data type(dict) for each layer
    cache: list['KVCache']
    sharding_cfg: ShardingConfig = default_sharding()
    quantization: Quantization | None = None

    def __init__(self):
        ...
    def get_cfg(self) -> 'GlobalKVCacheConfig':
        ...
    def sharding(self) -> 'GlobalKVCache':
        ...
    def update(self):
        ...
    def get_layer_kv_cache(layer:int):
        ...
        return self.cache[layer]
