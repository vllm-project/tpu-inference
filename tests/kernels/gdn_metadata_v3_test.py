# Copyright 2026 Google LLC
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

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from tpu_inference.kernels.gdn.v3 import config, metadata


class GDNMetadataTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="single_sequence", max_seqs=1),
        dict(testcase_name="partial_second_tile", max_seqs=5),
    )
    def test_batched_metadata_pads_partial_tile(self, max_seqs):
        tile_size = 4
        padded_max_seqs = ((max_seqs + tile_size - 1) // tile_size) * tile_size
        padding = padded_max_seqs - max_seqs
        dtypes = config.Dtypes(*([jnp.float32] * 5))
        cfg = config.GDNConfig(
            mode=config.GDNMode.BATCHED,
            dtypes=dtypes,
            batch_size=16,
            dim_size=1,
            kernel_size=4,
            tile_size=tile_size,
            num_kq_heads=1,
            num_v_heads=1,
            kq_head_dim=1,
            v_head_dim=1,
        )

        metadata_ref = metadata.compute_batched_seq_metadata(
            cfg=cfg,
            seq_lens=jnp.full(max_seqs, 21, dtype=jnp.int32),
            query_start_loc=jnp.arange(max_seqs + 1, dtype=jnp.int32),
            state_indices=jnp.arange(1, max_seqs + 1, dtype=jnp.int32),
            read_offsets=jnp.arange(max_seqs, dtype=jnp.int32),
            end_seq=jnp.array(max_seqs, dtype=jnp.int32),
        )

        def pad(values):
            return np.pad(values, (0, padding))

        is_valid = pad(np.ones(max_seqs, dtype=np.bool_))
        expected = {
            "p_id_to_s_idx": np.arange(padded_max_seqs),
            "p_id_to_r_base": pad(np.arange(max_seqs)),
            "p_id_to_r_size": pad(np.ones(max_seqs, dtype=np.int32)),
            "p_id_is_first_tile": is_valid,
            "p_id_is_last_tile": is_valid,
            "s_idx_has_initial_state": is_valid,
            "s_idx_to_state_indices": pad(np.arange(1, max_seqs + 1)),
            "s_idx_to_read_offset": pad(np.arange(max_seqs)),
        }
        actual = {
            "p_id_to_s_idx": metadata_ref.p_id_to_s_idx.data,
            "p_id_to_r_base": metadata_ref.p_id_to_r_base.data,
            "p_id_to_r_size": metadata_ref.p_id_to_r_size.data,
            "p_id_is_first_tile": metadata_ref.p_id_is_first_tile.data,
            "p_id_is_last_tile": metadata_ref.p_id_is_last_tile.data,
            "s_idx_has_initial_state": metadata_ref.s_idx_has_initial_state,
            "s_idx_to_state_indices": metadata_ref.s_idx_to_state_indices,
            "s_idx_to_read_offset": metadata_ref.s_idx_to_read_offset,
        }

        self.assertEqual(int(metadata_ref.num_tiles),
                         padded_max_seqs // tile_size)
        for name, expected_value in expected.items():
            with self.subTest(name=name):
                np.testing.assert_array_equal(actual[name], expected_value)
