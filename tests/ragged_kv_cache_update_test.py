import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import test_util as jtu

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

    @parameterized.product(
        page_size=[32, 33],
        combined_kv_head_num=[2, 16],
        head_dim=[128, 256],
        num_slices_per_block=[4, 8],
    )
    def test_basic(self, page_size: int, combined_kv_head_num: int,
                   head_dim: int, num_slices_per_block: int):
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

        old_kv_cache_copy = kv_cache.copy()
        updated_kv_cache = kv_cache_update(
            new_kv,
            slot_mapping,
            kv_cache,
            page_size=page_size,
            num_slices_per_block=num_slices_per_block)
        updated_kv_cache_ref = kv_cache_update_ref(new_kv, slot_mapping_np,
                                                   old_kv_cache_copy)
        self.assertAllClose(updated_kv_cache,
                            updated_kv_cache_ref,
                            atol=1e-4,
                            rtol=1e-4)
