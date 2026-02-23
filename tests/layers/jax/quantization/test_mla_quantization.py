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

def test_mla_k_up_proj_quantization(rngs):
    # MLA k_up_proj parameters: TNH,ANH->TNA
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
    
    # Verify create_weights_jax initialized the correct scale shape
    assert layer.weight_scale_inv.value.shape == (4, 128)
    
    # 2. Mock loaded weights and process them
    # Checkpoint weight: (InTotal, OutTotal) = (16384, 512) as reported by user
    torch_weight = torch.randn(16384, 512) 
    # Checkpoint scale: (OutBlocks, InBlocks) = (128, 4) as reported by user
    torch_scale = torch.randn(128, 4) 
    
    # This should handle the transpose/reshape correctly
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    assert layer.weight.value.shape == (512, 128, 128)
    assert layer.weight_scale_inv.value.shape == (4, 128)
    
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.weight_sharding = jax.sharding.PartitionSpec(None, None, None)
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    method.weight_sharding = jax.sharding.PartitionSpec(None, None, None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    assert hasattr(layer, 'weight')
    assert layer.weight.value.shape == (512, 128, 128)
    assert layer.weight_scale_inv.value.shape == (65536,)

def test_mla_v_up_proj_quantization(rngs):
    # MLA v_up_proj parameters: TNA,ANH->TNH
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
    
    # 2. Mock loaded weights and process them
    # Checkpoint weight: (InTotal, OutTotal) = (512, 16384)
    torch_weight = torch.randn(512, 16384) 
    # Checkpoint scale: (OutBlocks, InBlocks) = (128, 4)
    torch_scale = torch.randn(128, 4) 
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    assert layer.weight.value.shape == (512, 128, 128)
    assert layer.weight_scale_inv.value.shape == (4, 128)
    
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.weight_sharding = jax.sharding.PartitionSpec(None, None, None)
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    method.weight_sharding = jax.sharding.PartitionSpec(None, None, None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    assert hasattr(layer, 'weight')
    assert layer.weight.value.shape == (512, 128, 128)
    assert layer.weight_scale_inv.value.shape == (16384,)

if __name__ == "__main__":
    pytest.main([__file__])
