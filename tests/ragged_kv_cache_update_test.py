import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import test_util as jtu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_commons.kernels.ragged_kv_cache_update import kv_cache_update


def kv_cache_update_ref(new_kv, slot_mapping, kv_cache):
    """Reference implementation of KV cache update."""
    for i in range(slot_mapping.shape[1]):
        start_idx, new_kv_idx, slice_len = slot_mapping[:, i]
        kv_cache = kv_cache.at[start_idx:start_idx + slice_len].set(
            new_kv[new_kv_idx:new_kv_idx + slice_len])
    return kv_cache


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class KVCacheUpdateTest(jtu.JaxTestCase):

    def _generate_input_data(self, page_size, combined_kv_head_num, head_dim,
                             num_slices_per_block):
        page_num = 20
        padded_num_tokens = 128
        prng_key = jax.random.key(1234)
        kv_cache = jnp.zeros(
            (page_num * page_size, combined_kv_head_num, head_dim),
            dtype=jnp.bfloat16)
        new_kv = jax.random.normal(
            prng_key, (padded_num_tokens, combined_kv_head_num, head_dim),
            dtype=jnp.bfloat16)
        slice_lens = np.array([7, page_size, page_size, 1, 1, 1, 9],
                              dtype=np.int32)
        num_slices = jnp.array([len(slice_lens)], dtype=np.int32)
        kv_cache_start_indices = np.array([
            page_size * 2 - 7, page_size * 2, page_size * 3, page_size * 4 + 6,
            page_size * 5 + 7, page_size * 6 + 8, page_size * 15 + 3
        ],
                                          dtype=np.int32)
        new_kv_cache_indices = np.concatenate(
            [np.array([0], dtype=np.int32),
             np.cumsum(slice_lens[:-1])])
        slot_mapping_np = np.stack(
            [kv_cache_start_indices, new_kv_cache_indices, slice_lens], axis=1)
        padded_size = (slot_mapping_np.shape[0] + num_slices_per_block -
                       1) // num_slices_per_block * num_slices_per_block
        slot_mapping_np = np.pad(
            slot_mapping_np,
            [[0, padded_size - slot_mapping_np.shape[0]], [0, 0]],
            constant_values=0)
        slot_mapping_np = np.transpose(slot_mapping_np)
        slot_mapping = jnp.array(slot_mapping_np, dtype=jnp.int32)
        return new_kv, slot_mapping, kv_cache, num_slices

    @parameterized.product(
        page_size=[32, 33],
        combined_kv_head_num=[2, 16],
        head_dim=[128, 256],
        num_slices_per_block=[4, 8],
    )
    def test_basic(self, page_size: int, combined_kv_head_num: int,
                   head_dim: int, num_slices_per_block: int):
        new_kv, slot_mapping, kv_cache, num_slices = self._generate_input_data(
            page_size, combined_kv_head_num, head_dim, num_slices_per_block)
        old_kv_cache_copy = kv_cache.copy()

        updated_kv_cache = kv_cache_update(
            new_kv,
            slot_mapping,
            kv_cache,
            num_slices,
            page_size=page_size,
            num_slices_per_block=num_slices_per_block)
        updated_kv_cache_ref = kv_cache_update_ref(new_kv,
                                                   np.asarray(slot_mapping),
                                                   old_kv_cache_copy)
        self.assertAllClose(updated_kv_cache,
                            updated_kv_cache_ref,
                            atol=1e-4,
                            rtol=1e-4)

    @parameterized.product(
        page_size=[32, 33],
        combined_kv_head_num=[16, 32],
        head_dim=[128, 256],
        num_slices_per_block=[4, 8],
    )
    def test_torchax_shard_map(self, page_size: int, combined_kv_head_num: int,
                               head_dim: int, num_slices_per_block: int):
        new_kv, slot_mapping, kv_cache, num_slices = self._generate_input_data(
            page_size, combined_kv_head_num, head_dim, num_slices_per_block)
        old_kv_cache_copy = kv_cache.copy()

        mesh = jax.sharding.Mesh(jax.devices(), 'x')
        kv_cache_pspec = P(None, 'x', None)

        new_kv = jax.device_put(new_kv, NamedSharding(mesh, kv_cache_pspec))
        slot_mapping = jax.device_put(slot_mapping, NamedSharding(mesh, P()))
        kv_cache = jax.device_put(kv_cache,
                                  NamedSharding(mesh, kv_cache_pspec))
        num_slices = jax.device_put(num_slices, NamedSharding(mesh, P()))

        updated_kv_cache = kv_cache_update(new_kv, slot_mapping, kv_cache,
                                           num_slices,
                                           page_size=page_size,
                                           num_slices_per_block=\
                                               num_slices_per_block,
                                           mesh=mesh,
                                           kv_cache_pspec=kv_cache_pspec,)
        updated_kv_cache_ref = kv_cache_update_ref(new_kv,
                                                   np.asarray(slot_mapping),
                                                   old_kv_cache_copy)
        self.assertAllClose(updated_kv_cache,
                            updated_kv_cache_ref,
                            atol=1e-4,
                            rtol=1e-4)
