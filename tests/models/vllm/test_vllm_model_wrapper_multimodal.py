import inspect
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
import torch

from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper


def test_wrap_embed_multimodal_func_with_image_grid_thw():
    """
    Test that image_grid_thw is preserved when the underlying model signature supports it.
    """
    mock_wrapper = MagicMock()
    # Mock vllm_config.model_config.is_multimodal_model
    mock_wrapper.vllm_config.model_config.is_multimodal_model = True

    # Use a dummy class to represent the inner model with image_grid_thw in its signature
    class MockInnerModelWithGrid:
        def embed_multimodal(self, image_grid_thw=None, **kwargs):
            pass

    mock_inner = MockInnerModelWithGrid()

    # Mock the _VllmRunner wrapper
    mock_runner = MagicMock()
    mock_runner.vllm_model = mock_inner
    mock_wrapper.model = mock_runner

    # Get the function under test using the class method directly by passing mock_wrapper as self
    embed_multimodal_func = VllmModelWrapper.wrap_embed_multimodal_func(mock_wrapper)

    # Use patches to avoid executing real JAX/Torch conversions if they are in the path
    # and to spy on functional_call
    with patch("torchax.default_env"), \
         patch("torch.func.functional_call") as mock_functional_call:

        params_and_buffers = {}
        image_grid_thw_data = [1, 2, 3] # mock data
        kwargs = {
            "pixel_values": np.zeros((1, 3, 224, 224)),
        }

        embed_multimodal_func(
            params_and_buffers,
            image_grid_thw_data,
            **kwargs
        )

        mock_functional_call.assert_called_once()
        _, call_kwargs = mock_functional_call.call_args
        
        # Verify it was delegated as embed_multimodal
        assert call_kwargs["kwargs"]["call_method"] == "embed_multimodal"
        
        # Verify image_grid_thw was preserved in call_kwargs inside kwargs
        actual_call_kwargs = call_kwargs["kwargs"]["call_kwargs"]
        assert "image_grid_thw" in actual_call_kwargs


def test_wrap_embed_multimodal_func_without_image_grid_thw():
    """
    Test that image_grid_thw is removed when the underlying model signature does NOT support it.
    """
    mock_wrapper = MagicMock()
    mock_wrapper.vllm_config.model_config.is_multimodal_model = True

    # Use a dummy class to represent the legacy inner model without image_grid_thw
    class MockInnerModelLegacy:
        def embed_multimodal(self, pixel_values=None):
            pass

    mock_inner = MockInnerModelLegacy()

    mock_runner = MagicMock()
    mock_runner.vllm_model = mock_inner
    mock_wrapper.model = mock_runner

    embed_multimodal_func = VllmModelWrapper.wrap_embed_multimodal_func(mock_wrapper)

    with patch("torchax.default_env"), \
         patch("torch.func.functional_call") as mock_functional_call:

        params_and_buffers = {}
        image_grid_thw_data = [1, 2, 3] # mock data
        kwargs = {
            "pixel_values": np.zeros((1, 3, 224, 224)),
        }

        embed_multimodal_func(
            params_and_buffers,
            image_grid_thw_data,
            **kwargs
        )

        mock_functional_call.assert_called_once()
        _, call_kwargs = mock_functional_call.call_args
        
        assert call_kwargs["kwargs"]["call_method"] == "embed_multimodal"
        
        # Verify image_grid_thw was removed from the call_kwargs
        actual_call_kwargs = call_kwargs["kwargs"]["call_kwargs"]
        assert "image_grid_thw" not in actual_call_kwargs


def test_wrap_embed_multimodal_func_with_kwargs_fallback():
    """
    Test that image_grid_thw is preserved if the signature accepts **kwargs
    even if "image_grid_thw" is not explicitly named.
    """
    mock_wrapper = MagicMock()
    mock_wrapper.vllm_config.model_config.is_multimodal_model = True

    # Signature accepts **kwargs
    class MockInnerModelWithKwargs:
        def embed_multimodal(self, **kwargs):
            pass

    mock_inner = MockInnerModelWithKwargs()

    mock_runner = MagicMock()
    mock_runner.vllm_model = mock_inner
    mock_wrapper.model = mock_runner

    embed_multimodal_func = VllmModelWrapper.wrap_embed_multimodal_func(mock_wrapper)

    with patch("torchax.default_env"), \
         patch("torch.func.functional_call") as mock_functional_call:

        params_and_buffers = {}
        image_grid_thw_data = [1, 2, 3]
        kwargs = {
            "pixel_values": np.zeros((1, 3, 224, 224)),
        }

        embed_multimodal_func(
            params_and_buffers,
            image_grid_thw_data,
            **kwargs
        )

        mock_functional_call.assert_called_once()
        _, call_kwargs = mock_functional_call.call_args
        
        actual_call_kwargs = call_kwargs["kwargs"]["call_kwargs"]
        assert "image_grid_thw" in actual_call_kwargs
