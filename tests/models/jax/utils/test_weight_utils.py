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
import tempfile
from unittest.mock import MagicMock

import jax
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh
from safetensors.torch import save_file
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator


class TorchMLP(nn.Module):
    """MLP implemented with PyTorch."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(2, 6)
        self.act = nn.ReLU()
        self.w2 = nn.Linear(6, 2, bias=True)

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class JaxMLP(JaxModule, LoadableWithIterator):
    """MLP implemented with JAX."""

    def __init__(self, rngs):
        super().__init__()
        self.w1 = JaxLinear(2, 6, rngs)
        self.act = nnx.relu
        self.w2 = JaxLinear(6, 2, rngs, use_bias=True)

    def __call__(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class TestJaxAutoWeightsLoader:

    def test_load_from_safetensors(self):
        """Load weights from a safetensors file saved from a PyTorch model.
        """
        torch_model = TorchMLP()
        with torch.no_grad():
            torch_model.w1.weight.fill_(1.1)
            torch_model.w2.weight.fill_(0.9)
            torch_model.w2.bias.fill_(0.1)

        mock_vllm_config = MagicMock()
        mock_vllm_config.parallel_config = MagicMock()
        mock_vllm_config.parallel_config.enable_expert_parallel = False

        # Save the PyTorch model weights to a safetensors file. Load them
        # into the JAX model.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, "model.safetensors")
            save_file(torch_model.state_dict(), tmp_file_path)

            devices = jax.local_devices()
            mesh = Mesh(devices, axis_names=('p', ))
            with jax.set_mesh(mesh), set_current_vllm_config(mock_vllm_config):
                jax_model = JaxMLP(rngs=nnx.Rngs(0))

                model_config = MagicMock()
                model_config.quantization = None
                model_config.model = tmpdir
                model_config.revision = None

                loader = get_model_loader(
                    LoadConfig(load_format="safetensors"))
                loader.load_weights(jax_model, model_config)

        np.testing.assert_allclose(torch_model.w1.weight.T.detach().numpy(),
                                   jax_model.w1.weight.value)
        np.testing.assert_allclose(torch_model.w2.weight.T.detach().numpy(),
                                   jax_model.w2.weight.value)

        # Forward pass to verify correctness.
        input_values = [[0.1, 0.2], [0.3, 0.4]]
        torch_input = torch.tensor(input_values)
        jax_input = np.array(input_values)
        torch_output = torch_model(torch_input).detach().numpy()
        jax_output = jax_model(jax_input)
        np.testing.assert_allclose(torch_output,
                                   jax_output,
                                   rtol=1e-3,
                                   atol=1e-2)

    def test_weight_prefix_mapping(self):
        """Test that 'model.' is prepended correctly based on module structure."""
        from unittest.mock import MagicMock, patch

        from tpu_inference.models.jax.utils.weight_utils import \
            JaxAutoWeightsLoader

        class MockModel(JaxModule):

            def __init__(self):
                super().__init__()
                self.model = MagicMock()  # Simulate having 'model' attribute

        model = MockModel()

        def mock_named_children():
            yield "model", model.model
            yield "lm_head", MagicMock()
            yield "visual", MagicMock()

        model.named_children = mock_named_children

        loader = JaxAutoWeightsLoader(model)

        weights = [
            ("embed_tokens.weight", torch.zeros((2, 2))),
            ("lm_head.weight", torch.zeros((2, 2))),
            ("visual.weight", torch.zeros((2, 2))),
            ("model.layers.0.weight", torch.zeros((2, 2))),
        ]

        with patch(
                'tpu_inference.models.jax.utils.weight_utils.AutoWeightsLoader._load_module'
        ) as mock_super_load:
            mock_super_load.return_value = iter([])  # Yield nothing
            list(loader._load_module("", model, weights))

            assert mock_super_load.called
            args, kwargs = mock_super_load.call_args
            modified_weights = list(args[2])

            expected_keys = [
                "model.embed_tokens.weight",
                "lm_head.weight",
                "visual.weight",
                "model.layers.0.weight",
            ]

            received_keys = [name for name, _ in modified_weights]
            assert received_keys == expected_keys

        # Test with model WITHOUT 'model' attribute
        class MockModelNoModel(JaxModule):

            def __init__(self):
                super().__init__()
                self.w1 = MagicMock()

        model_no = MockModelNoModel()
        loader_no = JaxAutoWeightsLoader(model_no)

        with patch(
                'tpu_inference.models.jax.utils.weight_utils.AutoWeightsLoader._load_module'
        ) as mock_super_load_no:
            mock_super_load_no.return_value = iter([])
            list(loader_no._load_module("", model_no, weights))

            args, kwargs = mock_super_load_no.call_args
            modified_weights_no = list(args[2])
            received_keys_no = [name for name, _ in modified_weights_no]

            expected_keys_no = [
                "embed_tokens.weight",
                "lm_head.weight",
                "visual.weight",
                "model.layers.0.weight",
            ]
            assert received_keys_no == expected_keys_no

    def test_weight_interception_for_pooler(self):
        """Test that pooler weights are intercepted and loaded into PyTorch pooler."""
        from unittest.mock import MagicMock

        from tpu_inference.models.jax.utils.weight_utils import \
            JaxAutoWeightsLoader

        class MockModel(JaxModule):

            def __init__(self):
                super().__init__()

        model = MockModel()
        mock_pytorch_pooler = MagicMock()

        loader = JaxAutoWeightsLoader(model,
                                      pytorch_pooler=mock_pytorch_pooler)

        weights = [
            ("embed_tokens.weight", torch.zeros((2, 2))),
            ("pooler.weight", torch.zeros((2, 2))),
            ("model.pooler.bias", torch.zeros((2, ))),
        ]

        from unittest.mock import patch
        with patch(
                'tpu_inference.models.jax.utils.weight_utils.AutoWeightsLoader._load_module'
        ) as mock_super_load:
            seen_weights = []

            def side_effect(bp, mod, w_iter):
                for item in w_iter:
                    seen_weights.append(item)
                return iter(seen_weights)

            mock_super_load.side_effect = side_effect

            list(loader._load_module("", model, weights))

            assert mock_pytorch_pooler.load_state_dict.called
            args, kwargs = mock_pytorch_pooler.load_state_dict.call_args
            loaded_state_dict = args[0]

            assert "weight" in loaded_state_dict
            assert "bias" in loaded_state_dict

            received_keys = [name for name, _ in seen_weights]
            assert "pooler.weight" not in received_keys
            assert "model.pooler.bias" not in received_keys
            assert "embed_tokens.weight" in received_keys

    def test_dynamic_weight_mapping(self):
        """Test that weights are mapped dynamically based on root children."""
        from unittest.mock import MagicMock, patch

        from tpu_inference.models.jax.utils.weight_utils import \
            JaxAutoWeightsLoader

        class MockModel(JaxModule):

            def __init__(self):
                super().__init__()
                self.model = MagicMock()
                self.custom_head = MagicMock()

        model = MockModel()
        loader = JaxAutoWeightsLoader(model)

        weights = [
            ("custom_head.weight", torch.zeros((2, 2))),
            ("layers.0.weight", torch.zeros((2, 2))),
            ("model.layers.1.weight", torch.zeros((2, 2))),
        ]

        with patch(
                'tpu_inference.models.jax.utils.weight_utils.AutoWeightsLoader._load_module'
        ) as mock_super_load:
            mock_super_load.return_value = iter([])

            # We need to mock named_children to return the expected names
            def mock_named_children():
                yield "model", model.model
                yield "custom_head", model.custom_head

            model.named_children = mock_named_children

            list(loader._load_module("", model, weights))

            assert mock_super_load.called
            args, kwargs = mock_super_load.call_args
            modified_weights = list(args[2])

            expected_keys = [
                "custom_head.weight",  # Path A: root child, as-is
                "model.layers.0.weight",  # Path B: not root child, root has 'model', prepended
                "model.layers.1.weight",  # Path A: root child ('model'), as-is
            ]

            received_keys = [name for name, _ in modified_weights]
            assert received_keys == expected_keys
