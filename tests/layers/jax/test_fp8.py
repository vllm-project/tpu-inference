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
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from safetensors.torch import save_file
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.configs import JaxQuantLinearConfig
# yapf: disable
from tpu_inference.layers.jax.quantization.fp8 import (
    Fp8Config, Fp8LinearMethod, load_blockwise_fp8_scale)
# yapf: enable
from tpu_inference.models.jax.utils.weight_utils import LoadableWithIterator


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh, axis_names=MESH_AXIS_NAMES) as m:
        yield m


@pytest.fixture
def rng():
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestFp8Linear:

    def test_fp8_linear_init(self, mesh):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))

        quant_config = get_tpu_quantization_config(vllm_config)
        assert isinstance(quant_config, Fp8Config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        assert isinstance(layer.quant_method, Fp8LinearMethod)
        assert hasattr(layer, "weight_scale")
        assert layer.weight_scale.value.shape == (output_dim, )
        assert layer.weight_scale.value.dtype == jnp.float32
        assert hasattr(layer.weight_scale, "weight_loader")

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            hidden_states = jnp.ones((batch_size, input_dim))
            out = layer(hidden_states)
            assert out.shape == (batch_size, output_dim)

    def test_fp8_linear_correctness(self, mesh, rng):
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))
        quant_config = get_tpu_quantization_config(vllm_config)

        input_dim = 16
        output_dim = 32
        batch_size = 1
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        k1, k2, k3 = jax.random.split(rng, 3)
        w_val = jax.random.normal(k1, (input_dim, output_dim),
                                  dtype=jnp.float32)
        s_val = jax.random.uniform(k2, (output_dim, ), dtype=jnp.float32)

        layer.weight.value = w_val
        layer.weight_scale.value = s_val

        hidden_states = jax.random.uniform(k3, (batch_size, input_dim),
                                           dtype=jnp.float32)

        effective_w = w_val * s_val
        expected = jnp.dot(hidden_states, effective_w)

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            out = layer(hidden_states)
            assert jnp.allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_fp8_linear_scale_loader_logic(self, mesh):
        """
        Unit test to verify the fp8 weight loader logic.

        This isolates the scale loading mechanism. The full loading process (including
        the main weight and file interaction) is covered by `test_fp8_linear_load_from_safetensors`.
        """
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B", quantization="fp8"))
        quant_config = get_tpu_quantization_config(vllm_config)

        input_dim = 16
        output_dim = 32
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Create a dummy torch tensor for scale
        torch_scale = torch.rand((output_dim, ), dtype=torch.float32)

        layer.weight_scale.mesh = mesh
        layer.weight_scale.weight_loader(layer.weight_scale, torch_scale)
        jax_scale = layer.weight_scale.value
        np_scale_from_torch = torch_scale.numpy()

        assert jnp.allclose(jax_scale, np_scale_from_torch)

    def test_fp8_linear_load_from_safetensors(self, mesh):
        """Load weights and scales from a safetensors file simulating a quantized checkpoint.

        This test focuses on verifying the safetensors loading mechanism.
        Post-processing (process_weights_after_loading) is tested separately
        in the other test methods in this class.
        """
        from unittest.mock import patch

        input_dim = 16
        output_dim = 32

        class TorchFP8Model(torch.nn.Module):
            """Simulated FP8 model in PyTorch (using Float32 for simplicity)."""

            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim,
                                              output_dim,
                                              bias=False)
                self.linear.weight_scale = torch.nn.Parameter(
                    torch.rand((output_dim, ), dtype=torch.float32))

        class JaxFP8Model(JaxModule, LoadableWithIterator):

            def __init__(self, rngs, quant_config):
                super().__init__()
                self.linear = JaxLinear(input_size=input_dim,
                                        output_size=output_dim,
                                        rngs=rngs,
                                        quant_config=quant_config)

        # Create dummy weights and scales in torch
        torch_model = TorchFP8Model(input_dim, output_dim)
        torch.nn.init.normal_(torch_model.linear.weight)

        # Save to safetensors
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_file_path = os.path.join(tmpdir, "model.safetensors")
            state_dict = torch_model.state_dict()
            save_file(state_dict, tmp_file_path)

            # Create a dummy config.json to satisfy ModelConfig validation
            import json
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"model_type": "qwen2", "hidden_size": 16}, f)

            vllm_config = VllmConfig(
                model_config=ModelConfig(model=tmpdir, quantization="fp8"))
            quant_config = get_tpu_quantization_config(vllm_config)

            with mesh:
                jax_model = JaxFP8Model(rngs=nnx.Rngs(0),
                                        quant_config=quant_config)
                jax_model.linear.weight.mesh = mesh
                jax_model.linear.weight_scale.mesh = mesh

                loader = get_model_loader(
                    LoadConfig(load_format="safetensors"))

                mock_model_config = MagicMock()
                mock_model_config.model = tmpdir
                mock_model_config.quantization = "fp8"
                mock_model_config.revision = None

                # Mock process_weights_after_loading to isolate this test
                # to the loading mechanism only. Post-processing is tested
                # in other test methods.
                with patch.object(Fp8LinearMethod,
                                  'process_weights_after_loading'):
                    loader.load_weights(jax_model, mock_model_config)

        # Verify Weight
        expected_weight = torch_model.linear.weight.detach().numpy().T
        np.testing.assert_allclose(jax_model.linear.weight.value,
                                   expected_weight,
                                   rtol=1e-5)

        # Verify Scale
        expected_scale = torch_model.linear.weight_scale.detach().numpy()
        np.testing.assert_allclose(jax_model.linear.weight_scale.value,
                                   expected_scale,
                                   rtol=1e-5)

    def _load_from_checkpoint(self,
                              mesh,
                              state_dict,
                              input_dim,
                              output_dim,
                              block_size=None,
                              monkeypatch=None):
        """Helper: save state_dict to safetensors, load into a JaxFP8Model."""
        import json

        from safetensors.numpy import save_file as np_save_file

        if block_size is not None and monkeypatch is not None:
            monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        class JaxFP8Model(JaxModule, LoadableWithIterator):

            def __init__(self, rngs, quant_config):
                super().__init__()
                self.linear = JaxLinear(input_size=input_dim,
                                        output_size=output_dim,
                                        rngs=rngs,
                                        quant_config=quant_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            np_save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))

            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump({"model_type": "qwen2", "hidden_size": 16}, f)

            vllm_config = VllmConfig(
                model_config=ModelConfig(model=tmpdir, quantization="fp8"))
            quant_config = get_tpu_quantization_config(vllm_config)

            with mesh:
                jax_model = JaxFP8Model(rngs=nnx.Rngs(0),
                                        quant_config=quant_config)
                jax_model.linear.weight.mesh = mesh
                jax_model.linear.weight_scale.mesh = mesh

                loader = get_model_loader(
                    LoadConfig(load_format="safetensors"))

                mock_model_config = MagicMock()
                mock_model_config.model = tmpdir
                mock_model_config.quantization = "fp8"
                mock_model_config.revision = None

                loader.load_weights(jax_model, mock_model_config)

        return jax_model

    def test_load_checkpoint_unquantized(self, mesh, monkeypatch):
        """Float32 checkpoint → blockwise FP8 via REQUANTIZE_BLOCK_SIZE."""

        input_dim = 16
        output_dim = 32
        block_size = 8
        n_blocks = input_dim // block_size

        rng = np.random.default_rng(42)
        # HF checkpoint format: (output_dim, input_dim), float32
        state_dict = {
            "linear.weight":
            rng.standard_normal((output_dim, input_dim)).astype(np.float32),
            "linear.weight_scale":
            rng.random((n_blocks * output_dim, )).astype(np.float32),
        }

        jax_model = self._load_from_checkpoint(mesh, state_dict, input_dim,
                                               output_dim, block_size,
                                               monkeypatch)

        assert jax_model.linear.weight.value.dtype == jnp.float8_e4m3fn
        assert jax_model.linear.weight_scale.value.shape == (n_blocks, 1,
                                                             output_dim)

    def test_load_checkpoint_fp8_perchannel(self, mesh):
        """Pre-quantized per-channel FP8 checkpoint → kept as-is."""
        import ml_dtypes

        input_dim = 16
        output_dim = 32

        rng = np.random.default_rng(42)
        weight = rng.standard_normal(
            (output_dim, input_dim)).astype(np.float32)
        scale = rng.random((output_dim, )).astype(np.float32)

        state_dict = {
            "linear.weight": weight.astype(ml_dtypes.float8_e4m3fn),
            "linear.weight_scale": scale,
        }

        jax_model = self._load_from_checkpoint(mesh, state_dict, input_dim,
                                               output_dim)

        assert jax_model.linear.weight.value.dtype == jnp.float8_e4m3fn
        # Weight values preserved (transposed: HF (out, in) → JAX (in, out))
        expected = jnp.array(weight.T).astype(jnp.float8_e4m3fn)
        assert jnp.array_equal(jax_model.linear.weight.value, expected)
        np.testing.assert_allclose(jax_model.linear.weight_scale.value,
                                   scale,
                                   rtol=1e-5)

    @pytest.mark.parametrize("scale_shape", ["flat", "2d"],
                             ids=["flat_scales", "shaped_scales"])
    def test_load_checkpoint_fp8_blockwise(self, mesh, monkeypatch,
                                           scale_shape):
        """Pre-quantized blockwise FP8 checkpoint → dequant and re-quantized.

        Scales may be stored flat (n_blocks * n_out,) or shaped (n_blocks, n_out)
        in the checkpoint; both are supported.
        """
        import ml_dtypes

        input_dim = 16
        output_dim = 32
        block_size = 8
        n_blocks = input_dim // block_size

        rng = np.random.default_rng(42)
        if scale_shape == "flat":
            scales = rng.random((n_blocks * output_dim, )).astype(np.float32)
        else:
            scales = rng.random((n_blocks, output_dim)).astype(np.float32)

        state_dict = {
            "linear.weight":
            rng.standard_normal(
                (output_dim, input_dim)).astype(ml_dtypes.float8_e4m3fn),
            "linear.weight_scale":
            scales,
        }

        jax_model = self._load_from_checkpoint(mesh, state_dict, input_dim,
                                               output_dim, block_size,
                                               monkeypatch)

        assert jax_model.linear.weight.value.dtype == jnp.float8_e4m3fn
        assert jax_model.linear.weight_scale.value.shape == (n_blocks, 1,
                                                             output_dim)

    def test_fp8_linear_blockwise_init(self, mesh, monkeypatch):
        """Test Fp8Config with blockwise quantization initialization.

        Uses dimensions from tuned configurations:
        - TPU v7: (7, 16, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn')
        """
        block_size = 128
        # Set REQUANTIZE_BLOCK_SIZE env var (consistent with vLLM approach)
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        quant_config = Fp8Config()

        # TPU v7 config: (7, 16, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn')
        input_dim = 4096  # 4096 / 128 = 32 blocks
        output_dim = 14336
        batch_size = 16

        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        assert isinstance(layer.quant_method, Fp8LinearMethod)
        assert hasattr(layer, "weight_scale")

        n_blocks = input_dim // block_size
        assert layer.weight_scale.value.shape == (n_blocks, 1, output_dim)
        assert layer.weight_scale.value.dtype == jnp.float32
        assert hasattr(layer.weight_scale, "weight_loader")

        assert layer.quant_method.linear_config.enable_quantized_matmul_kernel
        assert layer.quant_method.linear_config.block_size == block_size

        layer.quant_method.linear_config.mesh = mesh

        # Requantize weights (randomly initialized as float32) to valid FP8 for the kernel
        layer.quant_method.process_weights_after_loading(layer)

        # Verify requantization produced FP8 weights and correct scale shape
        assert layer.weight.value.dtype == jnp.float8_e4m3fn
        assert layer.weight_scale.value.shape == (n_blocks, 1, output_dim)

        with mesh:
            hidden_states = jnp.ones((batch_size, input_dim))
            out = layer(hidden_states)
            assert out.shape == (batch_size, output_dim)

    def test_fp8_linear_blockwise_correctness(self, mesh, rng, monkeypatch):
        """Test forward pass correctness with blockwise quantization.

        Uses dimensions from tuned configurations:
        - TPU v7: (7, 16, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn')
        """
        block_size = 128
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        quant_config = Fp8Config()

        input_dim = 4096  # 4096 / 128 = 32 blocks
        output_dim = 14336
        batch_size = 16
        n_blocks = input_dim // block_size

        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        k1, k2, k3 = jax.random.split(rng, 3)
        w_val = jax.random.normal(k1, (input_dim, output_dim),
                                  dtype=jnp.float32)
        w_f8 = w_val.astype(jnp.float8_e4m3fn)
        w_f32_from_f8 = w_f8.astype(jnp.float32)

        # Blockwise scales: (n_blocks, 1, n_out)
        s_val = jax.random.uniform(k2, (n_blocks, 1, output_dim),
                                   dtype=jnp.float32)

        layer.weight.value = w_f8
        layer.weight_scale.value = s_val

        hidden_states = jax.random.uniform(k3, (batch_size, input_dim),
                                           dtype=jnp.float32)

        w_blocked = w_f32_from_f8.reshape(n_blocks, block_size, output_dim)

        w_scaled = w_blocked * s_val

        # Reshape back to (input_dim, output_dim)
        effective_w = w_scaled.reshape(input_dim, output_dim)
        expected = jnp.dot(hidden_states, effective_w)

        layer.quant_method.linear_config.mesh = mesh

        with mesh:
            out = layer(hidden_states)
            assert jnp.allclose(out, expected, rtol=0.5, atol=5.0)

    def test_fp8_linear_blockwise_scale_loader(self, mesh, monkeypatch):
        """Test loading blockwise scales from checkpoint format.

        Uses dimensions from tuned configurations:
        - TPU v7: (7, 16, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn')
        """
        block_size = 128
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        quant_config = Fp8Config()

        input_dim = 4096  # 4096 / 128 = 32 blocks
        output_dim = 14336
        n_blocks = input_dim // block_size

        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Create checkpoint-format scale: (n_blocks, n_out)
        torch_scale = torch.rand((n_blocks, output_dim), dtype=torch.float32)

        layer.weight_scale.mesh = mesh
        layer.weight_scale.weight_loader(layer.weight_scale, torch_scale)
        jax_scale = layer.weight_scale.value

        # Verify shape transformation: (n_blocks, n_out) -> (n_blocks, 1, n_out)
        assert jax_scale.shape == (n_blocks, 1, output_dim)

        expected = torch_scale.numpy()[:, None, :]
        assert jnp.allclose(jax_scale, expected)

    def test_fp8_blockwise_input_dim_validation(self, monkeypatch):
        """Test that blockwise quantization validates input_dim is divisible by block_size."""
        block_size = 7
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        quant_config = Fp8Config()

        input_dim = 16
        output_dim = 32

        with pytest.raises(
                ValueError,
                match="Input dimension .* must be divisible by block size"):
            JaxLinear(input_size=input_dim,
                      output_size=output_dim,
                      rngs=nnx.Rngs(0),
                      quant_config=quant_config)

    def test_fp8_blockwise_scale_loader_validation(self, mesh, monkeypatch):
        """Test that blockwise scale loader validates checkpoint shape mismatches.

        Uses dimensions from tuned configurations:
        - TPU v7: (7, 16, 14336, 4096, 'float8_e4m3fn', 'float8_e4m3fn')
        """
        block_size = 128
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))
        input_dim = 4096  # 4096 / 128 = 32 blocks
        output_dim = 14336

        quant_config = Fp8Config()
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Test with wrong size (different block size in checkpoint)
        wrong_blocks = 16  # Different from n_blocks=32
        torch_scale_wrong = torch.rand((wrong_blocks, output_dim),
                                       dtype=torch.float32)

        layer.weight_scale.mesh = mesh
        with pytest.raises(ValueError,
                           match="Checkpoint scale shape mismatch"):
            layer.weight_scale.weight_loader(layer.weight_scale,
                                             torch_scale_wrong)

    def test_fp8_requantization_requires_block_size(self, mesh, rng):
        """Test that requantization without REQUANTIZE_BLOCK_SIZE raises an error.

        Consistent with vLLM, per-channel requantization is not supported.
        Per-channel FP8 is only supported when loading from a pre-quantized checkpoint.
        """
        # No monkeypatch.setenv - per-channel mode (no block size)
        quant_config = Fp8Config()

        input_dim = 16
        output_dim = 32

        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Verify per-channel config
        assert not layer.quant_method.linear_config.enable_quantized_matmul_kernel
        assert layer.weight_scale.value.shape == (output_dim, )

        # Set float32 weights (simulating a non-quantized checkpoint)
        k1 = jax.random.PRNGKey(0)
        original_weight = jax.random.normal(k1, (input_dim, output_dim),
                                            dtype=jnp.float32)
        layer.weight.value = original_weight

        layer.quant_method.linear_config.mesh = mesh

        # Attempting to requantize without REQUANTIZE_BLOCK_SIZE should raise
        with pytest.raises(ValueError, match="REQUANTIZE_BLOCK_SIZE"):
            layer.quant_method.process_weights_after_loading(layer)

    def test_fp8_perchannel_from_checkpoint(self, mesh, rng):
        """Test that per-channel FP8 works when loaded from pre-quantized checkpoint.

        When weights are already FP8, no requantization happens.
        """
        # No REQUANTIZE_BLOCK_SIZE - per-channel mode
        quant_config = Fp8Config()

        input_dim = 16
        output_dim = 32
        batch_size = 4

        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Verify per-channel config
        assert not layer.quant_method.linear_config.enable_quantized_matmul_kernel
        assert layer.weight_scale.value.shape == (output_dim, )

        # Simulate loading from pre-quantized checkpoint (weights already FP8)
        k1, k2 = jax.random.split(rng, 2)
        fp8_weight = jax.random.normal(k1, (input_dim, output_dim),
                                       dtype=jnp.float32).astype(
                                           jnp.float8_e4m3fn)
        scale = jax.random.uniform(k2, (output_dim, ), dtype=jnp.float32)

        layer.weight.value = fp8_weight
        layer.weight_scale.value = scale

        layer.quant_method.linear_config.mesh = mesh

        # process_weights_after_loading should NOT raise (weights already FP8)
        layer.quant_method.process_weights_after_loading(layer)

        # Weights should still be FP8
        assert layer.weight.value.dtype == jnp.float8_e4m3fn

        # Verify forward pass works
        with mesh:
            hidden_states = jax.random.uniform(k1, (batch_size, input_dim),
                                               dtype=jnp.float32)
            out = layer(hidden_states)
            assert out.shape == (batch_size, output_dim)
            assert not jnp.isnan(out).any()

    def test_fp8_requantization_numerical_correctness(self, mesh, rng,
                                                      monkeypatch):
        """Test that requantization preserves numerical accuracy.

        Verifies that: original_weight @ x ≈ dequantize(quantized_weight, scale) @ x
        within reasonable FP8 quantization error bounds.

        Uses dimensions from tuned configurations (TPU v7).
        """
        block_size = 128
        monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", str(block_size))

        quant_config = Fp8Config()

        # Use dimensions that have tuned kernel configs
        input_dim = 4096  # 4096 / 128 = 32 blocks
        output_dim = 14336
        batch_size = 16
        n_blocks = input_dim // block_size

        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config)

        # Create original float32 weights
        k1, k2 = jax.random.split(rng, 2)
        original_weight = jax.random.normal(k1, (input_dim, output_dim),
                                            dtype=jnp.float32)
        layer.weight.value = original_weight

        # Compute expected output with original weights
        hidden_states = jax.random.uniform(k2, (batch_size, input_dim),
                                           dtype=jnp.float32)
        expected_output = jnp.dot(hidden_states, original_weight)

        layer.quant_method.linear_config.mesh = mesh

        # Trigger requantization
        layer.quant_method.process_weights_after_loading(layer)

        # Get quantized weights and scales
        w_q = layer.weight.value
        scale = layer.weight_scale.value

        assert w_q.dtype == jnp.float8_e4m3fn
        assert scale.shape == (n_blocks, 1, output_dim)

        # Manually dequantize and compute output
        w_f32 = w_q.astype(jnp.float32)
        w_blocked = w_f32.reshape(n_blocks, block_size, output_dim)
        w_dequant = (w_blocked * scale).reshape(input_dim, output_dim)
        manual_output = jnp.dot(hidden_states, w_dequant)

        # Verify kernel output matches manual dequantization
        with mesh:
            kernel_output = layer(hidden_states)

        # Kernel uses different precision/rounding than manual dequantization
        # Use same tolerances as test_fp8_linear_blockwise_correctness
        assert jnp.allclose(kernel_output, manual_output, rtol=0.5, atol=5.0)

        # Both should be reasonably close to the original (within FP8 error)
        # FP8 has ~4 bits of mantissa. For large matmuls with random weights,
        # error accumulates and can reach 20-30%
        relative_error = jnp.abs(kernel_output - expected_output) / (
            jnp.abs(expected_output) + 1e-6)
        mean_rel_error = jnp.mean(relative_error)

        # FP8 quantization with large matrices can introduce up to 30% error
        assert mean_rel_error < 0.35, f"Mean relative error {mean_rel_error} too high"


class TestFp8Sharding:

    def test_sharding_propagation_linear(self):
        """Test that JaxLinear's weight sharding is propagated to Fp8LinearMethod's config."""
        init_fn = nnx.initializers.ones
        input_dim = 16
        output_dim = 32

        # 1. Define Fp8Config
        quant_config = Fp8Config()

        # 2. Create JaxLinear with explicit partitioning
        # Weight shape: (Input, Output) = (16, 32)
        # Partitioning: (None, MLP_TENSOR) -> Output dim is sharded
        layer = JaxLinear(input_size=input_dim,
                          output_size=output_dim,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config,
                          kernel_init=nnx.with_partitioning(
                              init_fn, (None, ShardingAxisName.MLP_TENSOR)))

        # 3. Verify layer weight has sharding info
        # nnx.Param.sharding should be the tuple (None, MLP_TENSOR)
        assert layer.weight.sharding == (None, ShardingAxisName.MLP_TENSOR)

        # 4. Verify QuantLinearConfig has correct TRANSPOSED sharding
        # Original: (In, Out) -> (None, MLP_TENSOR)
        # Kernel expects: (Out, In)
        # Expected config sharding: (MLP_TENSOR, None)
        linear_config = layer.quant_method.linear_config
        assert linear_config.weight_sharding == P(ShardingAxisName.MLP_TENSOR,
                                                  None)

    def test_sharding_propagation_einsum_3d(self):
        """Test sharding propagation for 3D weight (e.g. Attention q_proj)."""
        init_fn = nnx.initializers.ones
        # Shape: (Hidden, Heads, HeadDim)
        hidden_size = 16
        num_heads = 4
        head_dim = 8
        kernel_shape = (hidden_size, num_heads, head_dim)
        einsum_str = "TD,DNH->TNH"

        quant_config = Fp8Config()

        # Partitioning: (None, ATTN_HEAD, None) -> Heads dim is sharded
        layer = JaxEinsum(einsum_str=einsum_str,
                          kernel_shape=kernel_shape,
                          rngs=nnx.Rngs(0),
                          quant_config=quant_config,
                          kernel_init=nnx.with_partitioning(
                              init_fn,
                              (None, ShardingAxisName.ATTN_HEAD, None)))

        assert layer.weight.sharding == (None, ShardingAxisName.ATTN_HEAD,
                                         None)

        # 4. Verify QuantLinearConfig has correct FLATTENED and TRANSPOSED sharding
        # Original: (D, N, H) -> (None, ATTN_HEAD, None)
        # Flattened (D, N*H) -> (None, ATTN_HEAD) (Assuming N is blocked on ATTN_HEAD)
        # Transposed (N*H, D) -> (ATTN_HEAD, None)
        linear_config = layer.quant_method.linear_config
        assert linear_config.weight_sharding == P(ShardingAxisName.ATTN_HEAD,
                                                  None)


class TestEinsumParsing:

    def test_parse_simple_linear(self):
        # "mn,np->mp" or equivalent for Linear(In, Out)
        # Weight shape (In, Out)
        # Inputs: m, n (Batch, In)
        # Weight: n, p (In, Out)
        # Output: m, p (Batch, Out)
        einsum_str = "mn,np->mp"
        input_dim = 16
        output_dim = 32
        weight_shape = (input_dim, output_dim)

        input_size, output_size, c_dims, o_dims = JaxQuantLinearConfig._parse_einsum_dims(
            einsum_str, weight_shape)

        assert input_size == input_dim
        assert output_size == output_dim
        assert c_dims == [0]  # 'n' is at index 0 of weight
        assert o_dims == [1]  # 'p' is at index 1 of weight

    def test_parse_attention_q_proj(self):
        # Qwen2 Attention: "TD,DNH->TNH"
        # Inputs: T, D (Batch/Seq, Hidden)
        # Weight: D, N, H (Hidden, Heads, HeadDim)
        # Output: T, N, H (Batch/Seq, Heads, HeadDim)

        einsum_str = "TD,DNH->TNH"
        hidden_size = 16
        num_heads = 4
        head_dim = 8
        weight_shape = (hidden_size, num_heads, head_dim)

        input_size, output_size, c_dims, o_dims = JaxQuantLinearConfig._parse_einsum_dims(
            einsum_str, weight_shape)

        # Contracting: D (dim 0) -> size 16
        # Output: N, H (dims 1, 2) -> size 4*8 = 32
        assert input_size == hidden_size
        assert output_size == num_heads * head_dim
        assert c_dims == [0]
        assert o_dims == [1, 2]

    def test_parse_output_projection(self):
        einsum_str = "TNH,NHD->TD"
        num_heads = 4
        head_dim = 8
        hidden_size = 16
        weight_shape = (num_heads, head_dim, hidden_size)

        input_size, output_size, c_dims, o_dims = JaxQuantLinearConfig._parse_einsum_dims(
            einsum_str, weight_shape)

        assert input_size == num_heads * head_dim
        assert output_size == hidden_size
        assert c_dims == [0, 1]
        assert o_dims == [2]

    def test_fallback_invalid_string(self):
        # Fallback to Linear logic
        input_dim = 16
        output_dim = 32
        weight_shape = (input_dim, output_dim)
        input_size, output_size, c_dims, o_dims = JaxQuantLinearConfig._parse_einsum_dims(
            "", weight_shape)

        assert input_size == input_dim
        assert output_size == output_dim
        assert c_dims == [0]
        assert o_dims == [1]


class TestLoadBlockwiseFp8Scale:

    def test_load_blockwise_fp8_scale(self):
        output_dim = 4
        n_blocks = 2

        devices = jax.local_devices()
        mesh = Mesh(devices, axis_names=('p', ))

        with jax.set_mesh(mesh):
            # Case 1: 1D flattened input
            torch_scale_1d = torch.arange(n_blocks * output_dim,
                                          dtype=torch.float32)
            jax_param = nnx.Param(
                jnp.zeros((n_blocks, 1, output_dim), dtype=jnp.float32))

            load_blockwise_fp8_scale(jax_param, torch_scale_1d, output_dim,
                                     n_blocks, "test_param")

            expected_scale = jnp.arange(n_blocks * output_dim,
                                        dtype=jnp.float32).reshape(
                                            n_blocks, 1, output_dim)
            np.testing.assert_allclose(jax_param.value, expected_scale)

            # Case 2: 2D input
            torch_scale_2d = torch.arange(n_blocks * output_dim,
                                          dtype=torch.float32).reshape(
                                              n_blocks, output_dim)
            jax_param = nnx.Param(
                jnp.zeros((n_blocks, 1, output_dim), dtype=jnp.float32))

            load_blockwise_fp8_scale(jax_param, torch_scale_2d, output_dim,
                                     n_blocks, "test_param")

            np.testing.assert_allclose(jax_param.value, expected_scale)

            # Case 3: Size mismatch (1D)
            torch_scale_wrong_size = torch.arange(n_blocks * output_dim + 1,
                                                  dtype=torch.float32)
            with pytest.raises(ValueError,
                               match="Checkpoint scale size mismatch"):
                load_blockwise_fp8_scale(jax_param, torch_scale_wrong_size,
                                         output_dim, n_blocks, "test_param")

            # Case 4: Shape mismatch (2D)
            torch_scale_wrong_shape = torch.zeros((n_blocks + 1, output_dim))
            with pytest.raises(ValueError,
                               match="Checkpoint scale shape mismatch"):
                load_blockwise_fp8_scale(jax_param, torch_scale_wrong_shape,
                                         output_dim, n_blocks, "test_param")

            # Case 5: Wrong number of dimensions
            torch_scale_3d = torch.zeros((n_blocks, output_dim, 1))
            with pytest.raises(
                    ValueError,
                    match="Checkpoint scale has unexpected number of dimensions"
            ):
                load_blockwise_fp8_scale(jax_param, torch_scale_3d, output_dim,
                                         n_blocks, "test_param")
