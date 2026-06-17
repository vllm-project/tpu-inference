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
"""Unit tests for the Torchax DFlash draft model wrapper."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from tpu_inference.models.torchax.dflash import (DFlashTorchaxWrapper,
                                                 _DFlashRunner)


@pytest.fixture(scope="module")
def mesh():
    """Creates a mesh with 1 device for testing."""
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    m = Mesh(device_mesh, axis_names=('data', 'attn_dp', 'expert', 'model'))
    with jax.set_mesh(m):
        yield m


class DummyHFModel(torch.nn.Module):
    """A dummy PyTorch module acting as the HF DFlash model."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 8, bias=False)
        self.hidden_norm = torch.nn.LayerNorm(8)

    def forward(self,
                noise_embedding,
                target_hidden,
                position_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                is_causal=False):
        del target_hidden, position_ids, attention_mask, past_key_values, use_cache, is_causal
        return noise_embedding


def test_dflash_runner():
    """Tests the _DFlashRunner forward routing."""
    hf_model = DummyHFModel()
    runner = _DFlashRunner(hf_model)

    # Test draft_forward routing
    noise = torch.ones((1, 3, 8))
    target = torch.ones((1, 5, 16))
    pos = torch.ones((1, 8))
    out = runner(noise_embedding=noise, target_hidden=target, position_ids=pos)
    assert out.shape == (1, 3, 8)

    # Test combine_hidden routing
    raw = torch.ones((5, 16))
    out_combined = runner(raw_hidden=raw)
    assert out_combined.shape == (5, 8)

    # Test compute_logits routing
    hidden = torch.ones((3, 8))
    embed = torch.ones((10, 8))
    out_logits = runner(hidden_state=hidden, embed_weight=embed)
    assert out_logits.shape == (3, 10)


def mock_jax_view(t):
    """Mock for jax_view to convert PyTorch tensors to JAX arrays without C++ assertions."""
    if isinstance(t, dict):
        return {k: mock_jax_view(v) for k, v in t.items()}
    if isinstance(t, torch.Tensor):
        if t.dtype == torch.bfloat16:
            # Cast to float32 on CPU first, convert to numpy, then load as bfloat16 in JAX
            return jnp.array(t.detach().cpu().float().numpy(),
                             dtype=jnp.bfloat16)
        return jnp.array(t.detach().cpu().numpy())
    return t


def mock_torch_view(t):
    """Mock for torch_view to convert JAX arrays to PyTorch tensors without C++ assertions."""
    if isinstance(t, dict):
        return {k: mock_torch_view(v) for k, v in t.items()}
    if isinstance(t, jax.Array) or isinstance(t, np.ndarray):
        # Handle bfloat16 conversion since torch.from_numpy doesn't support ml_dtypes.bfloat16
        dtype_str = str(getattr(t, "dtype", ""))
        if "bfloat16" in dtype_str:
            float32_arr = np.array(t, dtype=np.float32)
            return torch.from_numpy(float32_arr).to(torch.bfloat16)
        return torch.from_numpy(np.array(t))
    return t


class DummyContextManager:
    """Dummy context manager to replace torchax.default_env."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@patch("tpu_inference.models.torchax.dflash.AutoModel.from_pretrained")
@patch("tpu_inference.models.torchax.dflash.shard_model_to_tpu")
@patch("tpu_inference.models.torchax.dflash.jax_view",
       side_effect=mock_jax_view)
def test_dflash_torchax_wrapper_load(mock_jax, mock_shard,
                                     mock_from_pretrained, mesh):
    """Verifies that DFlashTorchaxWrapper.load configures model parameters and shared embeddings correctly."""
    hf_model = DummyHFModel()
    mock_from_pretrained.return_value = hf_model

    # Mock shard_model_to_tpu to return a dict of torch tensors
    mock_shard.return_value = {"weight": torch.ones((2, 2))}

    wrapper = DFlashTorchaxWrapper(mesh)

    # Set up mock target_model_state with embed_tokens.embedding
    class MockEmbedTokens:

        def __init__(self):
            self.embedding = jnp.ones((100, 8), dtype=jnp.bfloat16)

    class MockTargetModel:

        def __init__(self):
            self.embed_tokens = MockEmbedTokens()

    class MockTargetModelState:

        def __init__(self):
            self.model = MockTargetModel()

    target_state = MockTargetModelState()

    with jax.set_mesh(mesh):
        wrapper.load(
            draft_model_path="mock-draft-path",
            target_model_state=target_state,
        )

    assert wrapper.model is not None
    assert wrapper.model.dflash == hf_model
    assert isinstance(wrapper.params, dict)
    assert "weight" in wrapper.params
    # jax_view should have converted the sharded torch tensor to jax array
    assert isinstance(wrapper.params["weight"], jax.Array)
    # verify shared embedding is correctly captured
    assert wrapper.embed_weight_jax.shape == (100, 8)
    mock_jax.assert_called_once()


@patch("tpu_inference.models.torchax.dflash.AutoModel.from_pretrained")
@patch("tpu_inference.models.torchax.dflash.shard_model_to_tpu")
@patch("tpu_inference.models.torchax.dflash.jax_view",
       side_effect=mock_jax_view)
@patch("tpu_inference.models.torchax.dflash.torch_view",
       side_effect=mock_torch_view)
@patch("tpu_inference.models.torchax.dflash.torchax.default_env",
       side_effect=DummyContextManager)
def test_dflash_torchax_wrapper_fns(mock_env, mock_torch, mock_jax, mock_shard,
                                    mock_from_pretrained, mesh):
    """Verifies that the JIT-compiled functions return correct shapes and delegate to torch.func.functional_call."""
    hf_model = DummyHFModel()
    mock_from_pretrained.return_value = hf_model
    mock_shard.return_value = {"weight": torch.ones((2, 2))}

    wrapper = DFlashTorchaxWrapper(mesh)

    class MockEmbedTokens:

        def __init__(self):
            self.embedding = jnp.ones((10, 8), dtype=jnp.bfloat16)

    class MockTargetModel:

        def __init__(self):
            self.embed_tokens = MockEmbedTokens()

    class MockTargetModelState:

        def __init__(self):
            self.model = MockTargetModel()

    target_state = MockTargetModelState()

    with jax.set_mesh(mesh):
        wrapper.load("mock-draft-path", target_state)

    # Patch jax.jit to be a no-op so we can run and mock the functions under CPU/test
    with patch("jax.jit", lambda f, *args, **kwargs: f):
        draft_fn = wrapper.get_draft_forward_fn()
        combine_fn = wrapper.get_combine_hidden_fn()
        logits_fn = wrapper.get_compute_logits_fn()

    params = wrapper.params
    embed_weight = wrapper.embed_weight_jax

    # 1. Test draft_forward function
    noise_input_ids = jnp.array([1, 2, 3], dtype=jnp.int32)
    target_hidden = jnp.ones((5, 16), dtype=jnp.bfloat16)
    position_ids = jnp.arange(8, dtype=jnp.int32)
    attention_mask = jnp.ones((8, ), dtype=jnp.int32)

    with patch("torch.func.functional_call") as mock_func_call:
        # Mock functional_call to return a PyTorch tensor of shape (1, 3, 8)
        mock_func_call.return_value = torch.ones((1, 3, 8),
                                                 dtype=torch.bfloat16)

        output = draft_fn(
            params,
            noise_input_ids,
            target_hidden,
            position_ids,
            embed_weight,
            attention_mask,
        )

        assert output.shape == (3, 8)
        assert isinstance(output, jax.Array)

        # Verify functional_call arguments
        mock_func_call.assert_called_once()
        called_args, called_kwargs = mock_func_call.call_args
        assert called_args[0] == wrapper.model
        assert "noise_embedding" in called_kwargs["kwargs"]
        assert "target_hidden" in called_kwargs["kwargs"]
        assert "position_ids" in called_kwargs["kwargs"]
        assert "attention_mask" in called_kwargs["kwargs"]

        # Check that kwargs contain PyTorch tensors of the correct shape
        assert called_kwargs["kwargs"]["noise_embedding"].shape == (1, 3, 8)
        assert called_kwargs["kwargs"]["target_hidden"].shape == (1, 5, 16)
        assert called_kwargs["kwargs"]["position_ids"].shape == (1, 8)
        assert called_kwargs["kwargs"]["attention_mask"].shape == (1, 8)

    # 2. Test combine_fn
    raw_hidden = jnp.ones((5, 16), dtype=jnp.bfloat16)
    with patch("torch.func.functional_call") as mock_func_call:
        mock_func_call.return_value = torch.ones((5, 8), dtype=torch.bfloat16)

        output_combined = combine_fn(params, raw_hidden)

        assert output_combined.shape == (5, 8)
        assert isinstance(output_combined, jax.Array)
        mock_func_call.assert_called_once()
        called_args, called_kwargs = mock_func_call.call_args
        assert called_kwargs["kwargs"]["raw_hidden"].shape == (5, 16)

    # 3. Test logits_fn
    hidden_states = jnp.ones((3, 8), dtype=jnp.bfloat16)
    with patch("torch.func.functional_call") as mock_func_call:
        mock_func_call.return_value = torch.ones((3, 10), dtype=torch.bfloat16)

        output_logits = logits_fn(params, hidden_states, embed_weight)

        assert output_logits.shape == (3, 10)
        assert isinstance(output_logits, jax.Array)
        mock_func_call.assert_called_once()
        called_args, called_kwargs = mock_func_call.call_args
        assert called_kwargs["kwargs"]["hidden_state"].shape == (3, 8)
        assert called_kwargs["kwargs"]["embed_weight"].shape == (10, 8)
