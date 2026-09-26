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

import os
import unittest
from unittest.mock import patch

# Configure simulated CPU devices for multi-device EP mesh testing before importing JAX
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.common import fused_moe_gmm
from tpu_inference.layers.common.fused_moe_gmm import (moe_gmm_local,
                                                       valid_rows_mask)


class TestFusedMoeGmmOnehotMasking(unittest.TestCase):

    def test_onehot_unpermute_masks_unvisited_nan_rows(self):
        """Verifies that `moe_gmm_local` on the `is_onehot` path zeroes out
        unvisited `gmm2_res` rows (left uninitialized by `gmm_v2` when
        `zero_initialize=False`) before `combine @ gmm2_res`, preventing
        IEEE-754 `0.0 * NaN = NaN` from corrupting valid rows.
        """
        num_tokens = 8
        topk = 2
        batch_size = num_tokens * topk  # 16
        hidden_size = 32
        inter_size = 16
        global_num_experts = 8
        local_group_size = 2  # EP=4 shard holding 2 of 8 experts
        group_offset = jnp.array([2], dtype=jnp.int32)  # experts [2, 3]

        # 2 tokens routed to each of the 8 experts -> total 16
        group_sizes = jnp.full((global_num_experts, ), 2, dtype=jnp.int32)
        # Identity permutation for simplicity
        topk_argsort_revert_indices = jnp.arange(batch_size, dtype=jnp.int32)
        topk_weights = jnp.full((num_tokens, topk), 0.5, dtype=jnp.float32)

        x = jnp.ones((batch_size, hidden_size), dtype=jnp.bfloat16)
        w1 = jnp.ones((local_group_size, hidden_size, inter_size * 2),
                      dtype=jnp.bfloat16)
        w2 = jnp.ones((local_group_size, inter_size, hidden_size),
                      dtype=jnp.bfloat16)

        row_mask = valid_rows_mask(
            batch_size,
            group_sizes,
            group_offset,
            group_offset + local_group_size,
        )

        call_idx = [0]

        def fake_gmm_wrapper(lhs,
                             rhs,
                             rhs_scale,
                             rhs_bias,
                             gs,
                             go,
                             fuse_act=None,
                             preferred_element_type=None):
            call_idx[0] += 1
            out_cols = inter_size if fuse_act is not None else hidden_size
            valid_vals = jnp.full(( lhs.shape[0], out_cols),
                                  2.0,
                                  dtype=jnp.bfloat16)
            nan_vals = jnp.full((lhs.shape[0], out_cols),
                                jnp.nan,
                                dtype=jnp.bfloat16)
            # Simulate gmm_v2(..., zero_initialize=False): rows outside
            # [token_start, token_end) contain stale NaN bit patterns in HBM.
            return jnp.where(row_mask[:, None], valid_vals, nan_vals)

        with patch.object(fused_moe_gmm, "gmm_wrapper",
                          side_effect=fake_gmm_wrapper), \
             patch.object(jax.lax, "psum", side_effect=lambda v, axis_name: v):
            out = moe_gmm_local(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                group_sizes,
                group_offset,
                topk_argsort_revert_indices,
                topk_weights,
                activation="silu",
                topk=topk,
                parallelism="ep",
                onehot_moe_permute_threshold=1024,
                defer_all_reduce=True,
            )

        out_np = np.asarray(out, dtype=np.float32)
        self.assertFalse(np.isnan(out_np).any(),
                         f"Expected no NaNs in output, got: {out_np}")
        # Rows 2 and 3 (tokens 1) have both topk slots in [4, 8) -> 0.5*2 + 0.5*2 = 2.0
        # All other tokens are outside [4, 8) on this shard -> 0.0
        expected = np.zeros((num_tokens, hidden_size), dtype=np.float32)
        expected[2, :] = 2.0
        expected[3, :] = 2.0
        np.testing.assert_allclose(out_np, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
