# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax._src import test_util as jtu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.ragged_paged_attention.v2.ragged_kv_cache_update import \
    kv_cache_update


def kv_cache_update_ref(new_kv, slot_mapping, kv_cache):
    """Reference implementation of KV cache update."""
    for i in range(slot_mapping.shape[1]):
        start_idx, new_kv_idx, slice_len = slot_mapping[:, i]
        kv_cache = kv_cache.at[start_idx:start_idx + slice_len].set(
            new_kv[new_kv_idx:new_kv_idx + slice_len])
    return kv_cache


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class KVCacheUpdateTest(jtu.JaxTestCase):

    def _generate_data(self, page_size, combined_kv_head_num, head_dim):
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
        slot_mapping_np = np.transpose(slot_mapping_np)
        slot_mapping = jnp.array(slot_mapping_np, dtype=jnp.int32)
        return new_kv, slot_mapping, kv_cache, num_slices

    @parameterized.product(
        page_size=[32, 33],
        combined_kv_head_num=[2, 16],
        head_dim=[128, 256],
        num_slices_per_block=[None, 8],
        dynamic_validate_inputs=[False, True],
    )
    def test_basic(self, page_size: int, combined_kv_head_num: int,
                   head_dim: int, num_slices_per_block: int,
                   dynamic_validate_inputs: bool):
        new_kv, slot_mapping, kv_cache, num_slices = self._generate_data(
            page_size, combined_kv_head_num, head_dim)
        old_kv_cache_copy = kv_cache.copy()

        with jax.disable_jit(disable=dynamic_validate_inputs):
            updated_kv_cache = kv_cache_update(
                new_kv,
                slot_mapping,
                kv_cache,
                num_slices,
                page_size=page_size,
                num_slices_per_block=num_slices_per_block,
                dynamic_validate_inputs=dynamic_validate_inputs)
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
        num_slices_per_block=[None, 8],
    )
    def test_torchax_shard_map(self, page_size: int, combined_kv_head_num: int,
                               head_dim: int, num_slices_per_block: int):
        new_kv, slot_mapping, kv_cache, num_slices = self._generate_data(
            page_size, combined_kv_head_num, head_dim)
        old_kv_cache_copy = kv_cache.copy()

        mesh = Mesh(jax.devices(), 'x')
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

    def test_invalid_inputs(self):
        # Test all the cases when the inputs are invalid in the `_dynamic_validate_inputs` method
        page_size = 32
        combined_kv_head_num = 2
        head_dim = 128

        new_kv, slot_mapping, kv_cache, num_slices = self._generate_data(
            page_size, combined_kv_head_num, head_dim)

        with jax.disable_jit():
            # Case 1: new_kv_start < 0
            invalid_slot_mapping = slot_mapping.at[1, 0].set(-1)
            with self.assertRaisesRegex(
                    ValueError, "new_kv_start=-1 must be greater than"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 2: kv_cache_start < 0
            invalid_slot_mapping = slot_mapping.at[0, 0].set(-1)
            with self.assertRaisesRegex(
                    ValueError, "kv_cache_start=-1 must be greater than"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 3: slice_len <= 0
            invalid_slot_mapping = slot_mapping.at[2, 0].set(0)
            with self.assertRaisesRegex(
                    ValueError, "slice_len=0 must be less or equal to"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 4: slice_len > page_size
            invalid_slot_mapping = slot_mapping.at[2, 0].set(page_size + 1)
            with self.assertRaisesRegex(
                    ValueError,
                    f"slice_len={page_size + 1} must be less or equal to"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 5: new_kv_start + slice_len > new_token_num
            invalid_slot_mapping = slot_mapping.at[1, 0].set(new_kv.shape[0])
            with self.assertRaisesRegex(
                    ValueError,
                    "new_kv_start=128 \+ slice_len=7 must be less or equal to new_token_num=128"
            ):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 6: kv_cache_start + slice_len > kv_cache_token_num
            invalid_slot_mapping = slot_mapping.at[0, 0].set(kv_cache.shape[0])
            with self.assertRaisesRegex(
                    ValueError,
                    "kv_cache_start=640 \+ slice_len=7 must be less or equal to kv_cache_token_num=640"
            ):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 7: Each slice must reside in the same page
            invalid_slot_mapping = slot_mapping.at[0, 0].set(page_size - 1)
            invalid_slot_mapping = invalid_slot_mapping.at[2, 0].set(page_size)
            with self.assertRaisesRegex(
                    ValueError, "Each slice must reside in the same page"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 8: new_kv slices are not continuous
            invalid_slot_mapping = slot_mapping.at[1,
                                                   1].set(slot_mapping[1, 1] +
                                                          1)
            with self.assertRaisesRegex(ValueError, "is expeced to equal to"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)

            # Case 9: Overlap among the kv cache slices
            invalid_slot_mapping = slot_mapping.at[0, 4].set(slot_mapping[0,
                                                                          3])
            with self.assertRaisesRegex(
                    ValueError, "Overlap detected in kv_cache intervals"):
                kv_cache_update(new_kv,
                                invalid_slot_mapping,
                                kv_cache,
                                num_slices,
                                page_size=page_size,
                                dynamic_validate_inputs=True)
