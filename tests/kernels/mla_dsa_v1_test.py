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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from tests.kernels.mla_v2_test import generate_mla_inputs
from tpu_inference.kernels.mla.v1 import kernel as kernel_v1


class MlaDsaV1Test(absltest.TestCase):

    def test_ref_mla_ragged_paged_attention_with_dsa(self):
        rng = np.random.default_rng(42)
        q_len = 8
        kv_len = 16
        num_heads = 4
        lkv_dim = 128
        r_dim = 64
        page_size = 4
        num_pages = 100

        inputs = generate_mla_inputs(
            seq_lens=[(q_len, kv_len)],
            num_heads=num_heads,
            lkv_dim=lkv_dim,
            r_dim=r_dim,
            page_size=page_size,
            q_dtype=jnp.float32,
            kv_dtype=jnp.float32,
            num_pages=num_pages,
            rng=rng,
        )
        ql_nope, q_pe, new_kv_c, new_k_pe, cache_kv, kv_lens, page_indices, cu_q_lens, distribution = inputs

        topk = 4
        topk_indices = jax.random.randint(jax.random.PRNGKey(0), (q_len, topk),
                                          0, kv_len)

        outputs = kernel_v1.ref_mla_ragged_paged_attention(
            ql_nope=ql_nope,
            q_pe=q_pe,
            new_kv_c=new_kv_c,
            new_k_pe=new_k_pe,
            cache_kv=cache_kv,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            topk_indices=topk_indices,
        )

        # Test shape corresponds to the concatenated outputs, wait actually the ref kernel
        # returns the output of the attention mechanism per sequence concatenated? Wait,
        # it returns the concatenated outputs! So it should be [q_len, num_heads, lkv_dim]
        # Wait, the ref kernel returns `jnp.concatenate(outputs, axis=0)`. Let's check:
        self.assertIsNotNone(outputs)

        # We know we only gave it 1 sequence. So the result should be:
        self.assertEqual(outputs[0].shape, (q_len, num_heads, lkv_dim))


if __name__ == "__main__":
    absltest.main()
