import pytest
import torch
import torchax
from unittest.mock import MagicMock, patch

from tpu_inference.models.vllm.experimental.generic_patcher import apply_generic_tpu_patches
import vllm.model_executor.layers.utils as vllm_layer_utils
from vllm.model_executor.layers.layernorm import RMSNorm

@patch.object(torch.Tensor, 'to', return_value=torch.zeros(128))
def test_patch_rms_norm(mock_tensor_to):
    apply_generic_tpu_patches()
    
    # Avoid calling __init__ which triggers vLLM config lookups
    layer = object.__new__(RMSNorm)
    torch.nn.Module.__init__(layer)
    
    layer.weight = torch.nn.Parameter(torch.ones(128))
    layer.has_weight = True
    layer.variance_epsilon = 1e-6
    layer.hidden_size = 128
    layer.variance_size_override = None
    
    with patch.object(layer, 'forward_static', return_value=torch.zeros(128)) as mock_forward_static:
        x = torch.ones(128)
        layer.forward_native(x)
        
        mock_forward_static.assert_called_once()
        args, kwargs = mock_forward_static.call_args
        
        # Args: x, variance_epsilon, hidden_size, dtype, weight_param, residual, variance_size_override
        weight_param = args[4]
        assert not isinstance(weight_param, torch.nn.Parameter)
        assert isinstance(weight_param, torch.Tensor)
        mock_tensor_to.assert_called_with(device="jax")

@patch.object(torch.Tensor, 'to', return_value=torch.zeros(128))
def test_patch_default_unquantized_gemm(mock_tensor_to):
    apply_generic_tpu_patches()
    
    x = torch.ones(128)
    weight = torch.nn.Parameter(torch.ones(128, 128))
    bias = torch.nn.Parameter(torch.ones(128))
    
    with patch('torch.nn.functional.linear', return_value=torch.zeros(128)) as mock_linear:
        vllm_layer_utils.default_unquantized_gemm(None, x, weight, bias)
        
        mock_linear.assert_called_once()
        args, kwargs = mock_linear.call_args
        
        w_arg = args[1]
        b_arg = args[2]
        assert not isinstance(w_arg, torch.nn.Parameter)
        assert isinstance(w_arg, torch.Tensor)
        assert not isinstance(b_arg, torch.nn.Parameter)
        assert isinstance(b_arg, torch.Tensor)
        # Verify .to(device="jax") was called for both weight and bias
        assert mock_tensor_to.call_count >= 2
