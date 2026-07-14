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
import pytest
from jax.sharding import Mesh

import tpu_inference.layers.common.attention_interface as attention_interface
from tpu_inference.layers.vllm.ops.scaled_dot_product_attention import (
    scaled_dot_product_attention, vllm_vit_sdpa)


@pytest.fixture
def dp_mesh():
    """A 2-way 'data' x 1-way 'model' mesh, mirroring `enable_dp_attention`."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip('Need >=2 devices to exercise the DP axis.')
    return Mesh(np.array(devices[:2]).reshape(2, 1), ('data', 'model'))


class TestAttnDpBatchAxisFix:
    """
    With `--additional_config='{"sharding": {"sharding_strategy":
    {"enable_dp_attention": true}}}'`, the JAX mesh's 'data' axis has size > 1.
    vLLM's ViT flattens every image in a request into a single sequence via
    cu_seqlens/segment_ids, so the batch dimension seen by
    `scaled_dot_product_attention` / `vllm_vit_sdpa` is always 1 -- sharding that
    size-1 axis by a DP>1 'data' axis fails shard_map's divisibility check.
    The fix pins `batch_axis=None` (replicated) instead of the default
    `batch_axis="data"` for both ops.
    """

    def test_default_batch_axis_fails_under_dp_mesh_with_batch_one(
            self, dp_mesh):
        """Documents the bug: batch=1 can't be sharded by a size-2 'data' axis."""
        q = k = v = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)
        ab = jnp.zeros((1, 2, 128, 128), dtype=jnp.float32)

        with jax.set_mesh(dp_mesh):
            attn_fn = attention_interface.sharded_flash_attention(
                dp_mesh, causal=False, sm_scale=1.0, use_attention_bias=True)
            with pytest.raises(ValueError, match='divisible'):
                attn_fn(q, k, v, ab, None)

    def test_scaled_dot_product_attention_survives_dp_mesh(self, dp_mesh):
        q = k = v = jnp.ones((1, 2, 128, 64), dtype=jnp.float32)

        with jax.set_mesh(dp_mesh):
            out = scaled_dot_product_attention(q, k, v)

        assert out.shape == (1, 2, 128, 64)

    def test_vllm_vit_sdpa_survives_dp_mesh(self, dp_mesh):
        # (batch, seq_len, num_heads, head_dim), as vllm_vit_sdpa expects.
        q = k = v = jnp.ones((1, 128, 2, 64), dtype=jnp.float32)

        with jax.set_mesh(dp_mesh):
            out = vllm_vit_sdpa(q, k, v, cu_seqlens=[0, 64, 128])

        assert out.shape == (1, 128, 2, 64)
