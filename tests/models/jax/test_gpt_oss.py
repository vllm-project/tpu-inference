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

import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from tpu_inference.models.jax.gpt_oss import GptOss


class MockHFConfigDirectTheta:

    def __init__(self):
        self.num_hidden_layers = 1
        self.num_local_experts = 2
        self.vocab_size = 1000
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.head_dim = 64
        self.hidden_size = 512
        self.intermediate_size = 1024
        self.num_experts_per_tok = 1
        self.rms_norm_eps = 1e-5
        self.swiglu_limit = 7.0
        self.sliding_window = None
        self.rope_theta = 10000.0
        self.rope_scaling = {
            "factor": 1.0,
            "beta_slow": 1.0,
            "beta_fast": 1.0,
            "original_max_position_embeddings": 2048
        }


class MockHFConfigScalingTheta:

    def __init__(self):
        self.num_hidden_layers = 1
        self.num_local_experts = 2
        self.vocab_size = 1000
        self.num_attention_heads = 8
        self.num_key_value_heads = 2
        self.head_dim = 64
        self.hidden_size = 512
        self.intermediate_size = 1024
        self.num_experts_per_tok = 1
        self.rms_norm_eps = 1e-5
        self.swiglu_limit = 7.0
        self.sliding_window = None
        # No self.rope_theta attribute here
        self.rope_scaling = {
            "rope_theta": 20000.0,
            "factor": 1.0,
            "beta_slow": 1.0,
            "beta_fast": 1.0,
            "original_max_position_embeddings": 2048
        }


class MockVllmConfig:

    def __init__(self, hf_config):
        self.model_config = MagicMock()
        self.model_config.dtype = jnp.bfloat16
        self.model_config.enable_return_routed_experts = False
        self.model_config.hf_config = hf_config
        self.model_config.quantization = None
        self.cache_config = MagicMock(cache_dtype="auto")
        self.additional_config = {}


class TestGptOssModel(unittest.TestCase):
    """Unit tests for GptOss JAX model."""

    def setUp(self):
        self.mesh = Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1, 1, -1),
            axis_names=(
                "data",
                "attn_dp",
                "expert",
                "model",
            ),
        )
        # GptOssMoE derives its MoE backend from the current vLLM parallel
        # config, so model construction needs one set.
        self.default_vllm_config = VllmConfig(device_config=DeviceConfig(
            device="cpu"))

    def test_model_init_with_direct_rope_theta(self):
        """Tests that GptOss initializes correctly when rope_theta is directly in hf_config."""
        hf_config = MockHFConfigDirectTheta()
        vllm_config = MockVllmConfig(hf_config)

        with jax.set_mesh(self.mesh), set_current_vllm_config(
                self.default_vllm_config):
            model = GptOss(
                vllm_config=vllm_config,
                rng=jax.random.PRNGKey(42),
                mesh=self.mesh,
                force_random_weights=True,
            )

        # Retrieve the first layer's self-attention block
        first_layer_attn = model.layers[0].attn
        self.assertEqual(first_layer_attn.rope_theta, 10000.0)

    def test_model_init_with_scaling_rope_theta(self):
        """Tests that GptOss initializes correctly when rope_theta is in rope_scaling."""
        hf_config = MockHFConfigScalingTheta()
        vllm_config = MockVllmConfig(hf_config)

        with jax.set_mesh(self.mesh), set_current_vllm_config(
                self.default_vllm_config):
            model = GptOss(
                vllm_config=vllm_config,
                rng=jax.random.PRNGKey(42),
                mesh=self.mesh,
                force_random_weights=True,
            )

        first_layer_attn = model.layers[0].attn
        self.assertEqual(first_layer_attn.rope_theta, 20000.0)


if __name__ == "__main__":
    unittest.main()
