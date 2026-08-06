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

import contextlib
import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from tpu_inference.models.jax.deepseek_v3 import MLAEinsum


class TestMLAEinsumLoadWeights(unittest.TestCase):

    def test_partial_load_returns_loaded_names(self):
        # vLLM's AutoWeightsLoader logs a warning (dumping the full module
        # repr) for every module whose load_weights returns None. MLAEinsum
        # receives its two params (weight, weight_scale_inv) in separate
        # calls, so each call must report the names it loaded to avoid one
        # multi-line warning per decoder layer.
        fake_einsum = MagicMock()
        fake_einsum.loaded = set()
        weight_param = MagicMock()
        scale_param = MagicMock()
        fake_einsum.named_parameters.return_value = [
            ("weight", weight_param),
            ("weight_scale_inv", scale_param),
        ]

        loaded = MLAEinsum.load_weights(fake_einsum, [("weight", MagicMock())])

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded, {"weight"})
        self.assertEqual(fake_einsum.loaded, {"weight"})
        weight_param.weight_loader.assert_called_once()

    def test_load_more_than_two_params_raises(self):
        fake_einsum = MagicMock()
        fake_einsum.loaded = {"weight", "weight_scale_inv"}
        fake_einsum.named_parameters.return_value = []

        with self.assertRaises(ValueError):
            MLAEinsum.load_weights(fake_einsum, [("extra", MagicMock())])

    @patch("tpu_inference.models.jax.deepseek_v3.shard_put")
    @patch("tpu_inference.models.jax.deepseek_v3.JaxEinsum")
    @patch("tpu_inference.models.jax.deepseek_v3.quantize_tensor")
    @patch("tpu_inference.models.jax.deepseek_v3.dequantize_tensor")
    @patch("tpu_inference.models.jax.deepseek_v3.cpu_mesh_context")
    def test_full_load_splits_kv_and_returns_loaded_names(
            self, mock_mesh_ctx, mock_dequantize, mock_quantize,
            mock_jax_einsum, mock_shard_put):
        # Once every named param has been loaded, load_weights runs the k/v
        # split path to the end and must still return the names loaded in
        # this call (not None), so vLLM's AutoWeightsLoader stays quiet.
        kv_lora_rank = 2  # A
        num_heads = 1  # N
        qk_nope_head_dim = 2
        v_head_dim = 1

        mock_mesh_ctx.return_value = contextlib.nullcontext()
        mock_dequantize.return_value = jnp.zeros(
            (kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim)),
            dtype=jnp.float32)
        # quantize_tensor returns (weight, scale); the scale is transposed
        # with 3 axes, so hand back 3-D arrays.
        mock_quantize.return_value = (jnp.zeros((1, 1, 1), dtype=jnp.float32),
                                      jnp.zeros((1, 1, 1), dtype=jnp.float32))

        mla_layer = MagicMock()
        mla_layer.kv_lora_rank = kv_lora_rank
        mla_layer.N = num_heads
        mla_layer.qk_nope_head_dim = qk_nope_head_dim
        mla_layer.v_head_dim = v_head_dim
        mla_layer.prefix = "layers.0.self_attn.kv_b_proj"
        mla_layer.anh_sharding = ()

        fake_einsum = MagicMock()
        fake_einsum.loaded = set()
        fake_einsum.mla_layer = mla_layer
        fake_einsum.quant_config = MagicMock()
        fake_einsum.weight = MagicMock()
        fake_einsum.weight_scale_inv = MagicMock()
        weight_param = MagicMock()
        # A single named param means this one call completes the load and
        # reaches the final return.
        fake_einsum.named_parameters.return_value = [("weight", weight_param)]

        loaded = MLAEinsum.load_weights(fake_einsum, [("weight", MagicMock())])

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded, {"weight"})
        weight_param.weight_loader.assert_called_once()
        # The k/v split path ran: k_up_proj and v_up_proj were built and
        # their weights/scales placed on the mesh.
        self.assertEqual(mock_jax_einsum.call_count, 2)
        self.assertEqual(mock_shard_put.call_count, 4)
        self.assertIs(mla_layer.k_up_proj, mock_jax_einsum.return_value)
        self.assertIs(mla_layer.v_up_proj, mock_jax_einsum.return_value)


if __name__ == "__main__":
    unittest.main()
