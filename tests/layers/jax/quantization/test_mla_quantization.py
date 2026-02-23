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

from unittest.mock import MagicMock
import numpy as np
import torch
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from tpu_inference.layers.jax.quantization.fp8 import Fp8Config, Fp8BlockwiseLinearMethod
from tpu_inference.layers.jax.linear import JaxEinsum

@pytest.fixture
def rngs():
    return nnx.Rngs(42)

def test_mla_k_up_proj_step1_loading(rngs):
    # MLA k_up_proj parameters: TNH,ANH->TNA
    # x: (T, 128, 128)
    # weight: (512, 128, 128)
    # output: (T, 128, 512)
    kv_lora_rank = 512
    num_heads = 128
    qk_nope_head_dim = 128
    
    hf_quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    fp8_config = Fp8Config(hf_quant_config)
    
    layer = JaxEinsum(
        einsum_str="TNH,ANH->TNA",
        kernel_shape=(kv_lora_rank, num_heads, qk_nope_head_dim),
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.4.self_attn.k_up_proj"
    )
    
    method = layer.quant_method
    assert isinstance(method, Fp8BlockwiseLinearMethod)
    
    # Verify decoupled 2D placeholder shapes on CPU
    # Weight: OutTotal=512, InTotal=128*128=16384
    assert layer.weight.value.shape == (512, 16384)
    # Scale: OutBlocks = 512/128 = 4, InBlocks = 16384/128 = 128
    assert layer.weight_scale_inv.value.shape == (4, 128)
    
    # Mock loaded weights with actual checkpoint shapes
    torch_weight = torch.randn(16384, 512) # (InTotal, OutTotal)
    torch_scale = torch.randn(128, 4)      # (InBlocks, OutBlocks)
    
    # Loading should now succeed without AssertionError
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Verify loaded data is in the 2D buffer correctly (transposed)
    assert layer.weight.value.shape == (512, 16384)
    assert layer.weight_scale_inv.value.shape == (4, 128)
    assert layer.weight._is_loaded is True
    assert layer.weight_scale_inv._is_loaded is True

def test_mla_v_up_proj_step1_loading(rngs):
    # MLA v_up_proj parameters: TNA,ANH->TNH
    # x: (T, 128, 512)
    # weight: (512, 128, 128)
    # output: (T, 128, 128)
    kv_lora_rank = 512
    num_heads = 128
    v_head_dim = 128
    
    hf_quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    fp8_config = Fp8Config(hf_quant_config)
    
    layer = JaxEinsum(
        einsum_str="TNA,ANH->TNH",
        kernel_shape=(kv_lora_rank, num_heads, v_head_dim),
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.4.self_attn.v_up_proj"
    )
    
    method = layer.quant_method
    assert isinstance(method, Fp8BlockwiseLinearMethod)
    
    # Verify decoupled 2D placeholder shapes on CPU
    # weight: ANH. A and N are shared with activation TNA (Input side).
    # H is free (Output side).
    # Weight: OutTotal=128 (H), InTotal=512*128=65536 (A*N)
    assert layer.weight.value.shape == (128, 65536)
    # Scale: OutBlocks = 128/128 = 1, InBlocks = 65536/128 = 512
    assert layer.weight_scale_inv.value.shape == (1, 512)
    
    # Mock loaded weights with actual checkpoint shapes
    torch_weight = torch.randn(65536, 128) # (InTotal, OutTotal)
    torch_scale = torch.randn(1, 512)      # (OutBlocks, InBlocks)
    
    # Loading should now succeed
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    assert layer.weight.value.shape == (128, 65536)
    assert layer.weight_scale_inv.value.shape == (1, 512)

if __name__ == "__main__":
    pytest.main([__file__])
