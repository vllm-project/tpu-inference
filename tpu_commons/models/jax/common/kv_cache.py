# Current implementation split the Cache and method.
# TODO: we could discuss if we do want to encapsulate the KVCache and updater into the same class
#
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Float, Int

from tpu_commons.models.jax.common.layers import Config
from tpu_commons.models.jax.common.sharding import ShardingConfig

iota = jax.lax.broadcasted_iota


class KVCacheUpdaterBase:
    """Abstract base class for KV cache update strategies.

    This class defines the interface for different methods of updating the
    key-value cache in a transformer model. Subclasses should implement the

    `update` method.
    """

    def update(self, key_cache, value_cache, new_keys, new_values,
               current_lengths, cfg):
        """Updates the key and value caches with new entries."""
        raise NotImplementedError


class StandardUpdater(KVCacheUpdaterBase):
    """A standard KV cache updater that uses a circular buffer strategy."""

    def update_cache(
        self,
        operand: Float,
        cache: Float,
        sequence_len: Int,
        dtype: Any,
    ) -> Float:
        """Updates a cache tensor with new data using indexed assignments.

        This function implements a circular buffer update mechanism. It calculates
        the correct indices to write the new `operand` data into the `cache`
        based on the current sequence lengths, wrapping around the cache's
        sequence dimension if necessary.

        Args:
            operand: The new key or value data to be written to the cache.
                Shape: (batch_size, seq_len, num_kv_heads, head_dim).
            cache: The KV cache tensor to be updated.
                Shape: (batch_size, cache_len, num_kv_heads, head_dim).
            sequence_len: A 1D array containing the current length of each
                sequence in the batch *before* the update.
            dtype: The data type to use for index calculations.

        Returns:
            The updated cache tensor with the new data written at the
            appropriate indices.
        """
        operand_BSKH = operand.astype(cache.dtype)
        sequence_len_B = sequence_len
        # it's usually 1 for generating, unless speculative-decoding
        batch_size, max_seq_len = operand_BSKH.shape[:2]
        cache_size = cache.shape[-3]  # How many tokens' KV would be stored

        # [3, 1, 2] -> [[3], [1], [2]], seq0 has 3 tokens, seq1 has 1 token, etc
        shift_B = sequence_len_B[:, None]
        batch_idx = jnp.arange(batch_size)[:, None].astype(jnp.int32)

        # Create offsets for each position in the input sequence.
        # e.g., for a batch element with sequence_len=3 and input len=2,
        # the offsets will be [3, 4]. These are then wrapped by the cache size.
        # offset = [[3, 4], [1, 2], [2, 3]]
        offset = ((iota(jnp.int32, (batch_size, max_seq_len), 1) + shift_B) %
                  cache_size).astype(jnp.int32)
        # update[i, j] will be stored in cache[batch_idx[i, j], offset[i, j]]
        # which is equivalent to cache[i, (sequence_len[i] + j) % cache_size]
        cache = cache.at[batch_idx, offset].set(operand_BSKH, mode='drop')

        return cache


@dataclass
class KVCacheConfig(Config):
    """Configuration for the KV cache.

    Attributes:
        batch_size: The number of sequences in a batch.
        cache_len: The total length of the cache buffer for each sequence.
        num_kv_heads: The number of key/value heads.
        head_dim: The dimension of each attention head.
        dtype: The data type for the cache tensors (e.g., jnp.float32).
    """
    batch_size: int
    cache_len: int
    num_kv_heads: int
    head_dim: int
    dtype: jnp.dtype = jnp.float32


@dataclass
class KVCache(nnx.Module):
    """A stateful module to manage the Key-Value cache for attention.

    This class holds the key and value tensors used in autoregressive decoding.
    It separates caches for 'prefill' and 'generate' modes to allow for
    different sharding strategies. The actual update logic is delegated to an
    `updater` object.

    Attributes:
        cfg: The KVCacheConfig configuration object.
        mesh: The JAX device mesh for distributed computation.
        sharding_cfg: Configuration for tensor sharding strategies.
        updater: An instance of a KVCacheUpdaterBase subclass that defines
            the cache update logic.
        key_cache: A dictionary mapping op_mode ('prefill', 'generate') to
            the key cache nnx.Variable.
        value_cache: A dictionary mapping op_mode ('prefill', 'generate') to
            the key cache nnx.Variable.
    """
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

    def _initialize_caches(self) -> 'KVCache':
        """Creates and initializes the key and value cache tensors.

        Initializes the cache tensors as nnx.Variables with zeros, placing them
        on the correct devices according to the sharding configuration for both
        'prefill' and 'generate' modes.
        """
        cache_shape = (self.cfg.batch_size, self.cfg.cache_len,
                       self.cfg.num_kv_heads, self.cfg.head_dim)

        self.key_cache['prefill'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bskh['prefill']))
        self.value_cache['prefill'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bskh['prefill']))
        self.key_cache['generate'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bskh['generate']))
        self.value_cache['generate'] = nnx.Variable(
            jax.device_put(jnp.zeros(cache_shape, dtype=self.cfg.dtype),
                           self.keyvalue_prefill_mode_cache_bskh['generate']))

    def create_sharding(self, ):
        """Creates sharding rules for the cache tensors.

        This method sets up separate NamedSharding objects for the 'prefill'
        and 'generate' operational modes based on the provided sharding config.
        """
        mode_dependent_attrs = [
            "keyvalue_prefill_mode_cache_bskh",
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
               new_keys: Float,
               new_values: Float,
               current_lengths: Int,
               op_mode: str = 'prefill'):
        """Updates the key and value caches with new entries.

        This method selects the appropriate cache based on the `op_mode` and
        delegates the update operation to the `updater` instance.

        Args:
            new_keys: The new key tensors to add to the cache.
            new_values: The new value tensors to add to the cache.
            current_lengths: A 1D tensor of current sequence lengths for each
                item in the batch.
            op_mode: The operational mode, either 'prefill' or 'generate'.
                This determines which cache tensor and sharding to use.

        Returns:
            A tuple of the updated key and value cache `nnx.Variable` objects.
        """
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
