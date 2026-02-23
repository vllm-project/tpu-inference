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

def get_spec(sharding):
    if hasattr(sharding, 'spec'):
        return sharding.spec
    return jax.sharding.PartitionSpec()

def test_mla_k_up_proj_lifecycle(rngs):
    # MLA k_up_proj parameters: TNH,ANH->TNA
    # Rank A (512) is Output. N*H (128*128=16384) is Input.
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
    
    # Verify Decoupled Placeholder (Step 1)
    # Weight: OutTotal=512 (A), InTotal=16384 (N*H)
    assert layer.weight.value.shape == (512, 16384)
    
    # Mock Loaded Checkpoint (InTotal, OutTotal)
    torch_weight = torch.randn(16384, 512) 
    torch_scale = torch.randn(128, 4)      # (InBlocks, OutBlocks)
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Process Weights (Step 2)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.weight_sharding = jax.sharding.PartitionSpec(None, None, None)
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    method.weight_sharding = jax.sharding.PartitionSpec('x', None, None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layouts (Step 2.1.2)
    assert layer.weight.value.shape == (512, 128, 128)
    # Scale: Rank A was Output (512). Logical shape (512,)
    assert layer.weight_scale_inv.value.shape == (512,)
    
    # Verify Sharding (Step 2.1.3)
    # In CPU tests with SingleDeviceSharding, we might get an empty spec.
    # We just want to ensure it doesn't crash and ideally matches if NamedSharding is present.
    if hasattr(layer.weight_scale_inv.sharding, 'spec'):
        assert layer.weight_scale_inv.sharding.spec == jax.sharding.PartitionSpec('x')

def test_mla_v_up_proj_lifecycle(rngs):
    # MLA v_up_proj parameters: TNA,ANH->TNH
    # Rank A (512) and Heads N (128) are Input side. HeadDim H (128) is Output.
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
    
    # Verify Decoupled Placeholder (Step 1)
    # Weight: OutTotal=128 (H), InTotal=512*128=65536 (A*N)
    assert layer.weight.value.shape == (128, 65536)
    
    # Mock Loaded Checkpoint (InTotal, OutTotal)
    torch_weight = torch.randn(65536, 128) 
    torch_scale = torch.randn(1, 512)      # (InBlocks, OutBlocks)
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Process Weights (Step 2)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.weight_sharding = jax.sharding.PartitionSpec(None, None, None)
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    method.weight_sharding = jax.sharding.PartitionSpec(None, 'x', None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layouts (Step 2.1.2)
    assert layer.weight.value.shape == (512, 128, 128)
    # Scale: Output was HeadDim (128). Logical shape (128,)
    assert layer.weight_scale_inv.value.shape == (128,)
    
    # Verify Sharding (Step 2.1.3)
    if hasattr(layer.weight_scale_inv.sharding, 'spec'):
        assert layer.weight_scale_inv.sharding.spec == jax.sharding.PartitionSpec(None)

if __name__ == "__main__":
    pytest.main([__file__])
