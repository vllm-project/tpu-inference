# Current implementation split the Cache and method.
# TODO: we could discuss if we do want to encapsulate the KVCache and updater into the same class
#
from dataclasses import dataclass

from tpu_commons.models.jax.common.constants import *
from tpu_commons.models.jax.common.layers import *
from tpu_commons.models.jax.common.sharding import *

iota = jax.lax.broadcasted_iota


class KVCacheUpdaterBase:
    """Abstract base class for cache update strategies."""

    def update(self, key_cache, value_cache, new_keys, new_values,
               current_lengths, cfg):
        raise NotImplementedError


class StandardUpdater(KVCacheUpdaterBase):
    # Split update for prefill and generate
    def update_cache(
        self,
        operand: Float[Array, "B S N H"],
        cache: Float[Array, "B C N H"],
        sequence_len: Int[Array, "B"],
        dtype: Any,
    ) -> Float[Array, "B C N H"]:
        operand_BSNH = operand.astype(cache.dtype)
        sequence_len_B = sequence_len
        # it's usually 1 for generating, unless speculative-decoding
        batch_size, max_seq_len = operand_BSNH.shape[:2]
        cache_size = cache.shape[-3]  # How many tokens' KV would be stored

        # [3, 1, 2] -> [[3], [1], [2]], seq0 has 3 tokens, seq1 has 1 token, etc
        shift_B = sequence_len_B[:, None]
        batch_idx = jnp.arange(batch_size)[:, None].astype(jnp.int32)
        # [[3, 4, 5], [1, 2, 3], [2, 3, 4]]
        offset = ((iota(dtype, (batch_size, max_seq_len), 1) + shift_B) %
                  cache_size).astype(jnp.int32)
        # update[i, j] will be stored in cache[batch_idx[i, j], offset[i, j]]
        # namely cache[i, sequence_len[i] + j]
        cache = cache.at[batch_idx, offset].set(operand_BSNH, mode='drop')

        return cache


@dataclass
class KVCacheConfig(Config):
    """Configuration for KV cache."""
    batch_size: int
    cache_len: int
    num_kv_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.float32


@dataclass
class KVCache(nnx.Module):
    cfg: KVCacheConfig
    mesh: Mesh
    sharding_cfg: ShardingConfig
    updater: KVCacheUpdaterBase

    def __post_init__(self):
        """Initializes the cache tensors and sharding objects."""
        self.create_sharding()
        self.key_cache = {}
        self.value_cache = {}
        self._initialize_caches()

    def _initialize_caches(self, ) -> 'KVCache':
        cache_shape = (self.cfg.batch_size, self.cfg.cache_len,
                       self.cfg.num_kv_heads, self.cfg.head_dim)

        self.key_cache['prefill'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bsnh['prefill']))
        self.value_cache['prefill'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bsnh['prefill']))
        self.key_cache['generate'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bsnh['generate']))
        self.value_cache['generate'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bsnh['generate']))

    def create_sharding(self, ):

        mode_dependent_attrs = [
            "keyvalue_prefill_mode_cache_bsnh",
        ]
        for attr_name in mode_dependent_attrs:
            prefill_sharding_config = getattr(
                self.sharding_cfg.prefill_sharding_cfg, attr_name)
            generate_sharding_config = getattr(
                self.sharding_cfg.generate_sharding_cfg, attr_name)

            sharding_dict = {
                'prefill': NamedSharding(self.mesh,
                                         P(prefill_sharding_config)),
                'generate': NamedSharding(self.mesh,
                                          P(generate_sharding_config))
            }
            setattr(self, attr_name, sharding_dict)

    def update(self,
               new_keys: Float[Array, "B S N H"],
               new_values: Float[Array, "B S N H"],
               current_lengths: Int[Array, "B"],
               op_mode: str = 'prefill'):
        key_cache_variable = self.key_cache[op_mode]
        value_cache_variable = self.value_cache[op_mode]

        new_key_value = self.updater.update_cache(new_keys,
                                                  key_cache_variable.value,
                                                  current_lengths,
                                                  self.cfg.dtype)
        new_value_value = self.updater.update_cache(new_values,
                                                    value_cache_variable.value,
                                                    current_lengths,
                                                    self.cfg.dtype)

        key_cache_variable.value = new_key_value
        value_cache_variable.value = new_value_value

        return key_cache_variable, value_cache_variable
