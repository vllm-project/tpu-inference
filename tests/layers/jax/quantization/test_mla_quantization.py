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
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear

@pytest.fixture
def rngs():
    return nnx.Rngs(42)

def get_spec(sharding):
    if hasattr(sharding, 'spec'):
        return sharding.spec
    return jax.sharding.PartitionSpec()

def test_standard_linear_lifecycle(rngs):
    # Standard 2D linear TD,DA->TA (In=7168, Out=1536)
    # Optimized layout is (1536, 7168)
    in_features = 7168
    out_features = 1536
    
    hf_quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    fp8_config = Fp8Config(hf_quant_config)
    original_w_sharding = jax.sharding.PartitionSpec('x', None)
    
    layer = JaxLinear(
        input_size=in_features,
        output_size=out_features,
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.0.mlp.gate_proj",
        use_bias=False,
        kernel_init=nnx.with_partitioning(nnx.initializers.uniform(), original_w_sharding)
    )
    
    method = layer.quant_method
    assert isinstance(method, Fp8BlockwiseLinearMethod)
    
    # Verify Decoupled Placeholder (Step 1)
    # Weight: OutTotal=1536, InTotal=7168
    assert layer.weight.value.shape == (1536, 7168)
    assert layer.weight.sharding == ()
    assert layer.weight_scale_inv.sharding == ()
    
    # Mock Loaded Checkpoint (OutTotal, InTotal)
    torch_weight = torch.randn(1536, 7168) 
    torch_scale = torch.randn(12, 56)      # (OutBlocks, InBlocks)
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Process Weights (Step 2)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Optimized 2D Layouts (Step 2.1.2)
    assert layer.weight.value.shape == (1536, 7168)
    # Scale: Output was 1536. Logical shape (1536,)
    assert layer.weight_scale_inv.value.shape == (1536,)
    
    # Verify Sharding
    if hasattr(layer.weight.sharding, 'spec'):
        assert layer.weight.sharding.spec == original_w_sharding
        assert layer.weight_scale_inv.sharding.spec == jax.sharding.PartitionSpec('x')

def test_mla_k_up_proj_lifecycle(rngs):
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
    original_w_sharding = jax.sharding.PartitionSpec('x', None, None)
    
    layer = JaxEinsum(
        einsum_str="TNH,ANH->TNA",
        kernel_shape=(kv_lora_rank, num_heads, qk_nope_head_dim),
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.4.self_attn.k_up_proj",
        kernel_init=nnx.with_partitioning(nnx.initializers.uniform(), original_w_sharding)
    )
    
    method = layer.quant_method
    
    # Verify Decoupled Placeholder (Step 1)
    assert layer.weight.value.shape == (512, 16384)
    assert layer.weight.sharding == ()
    assert layer.weight_scale_inv.sharding == ()
    
    # Mock Loaded Checkpoint (OutTotal, InTotal)
    torch_weight = torch.randn(512, 16384) 
    torch_scale = torch.randn(4, 128)      # (OutBlocks, InBlocks)
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Process Weights (Step 2)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layouts (Step 2.1.2)
    assert layer.weight.value.shape == (512, 128, 128)
    assert layer.weight_scale_inv.value.shape == (512,)
    
    if hasattr(layer.weight.sharding, 'spec'):
        assert layer.weight.sharding.spec == original_w_sharding
        assert layer.weight_scale_inv.sharding.spec == jax.sharding.PartitionSpec('x')

def test_mla_v_up_proj_lifecycle(rngs):
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
    original_w_sharding = jax.sharding.PartitionSpec(None, 'x', None)
    
    layer = JaxEinsum(
        einsum_str="TNA,ANH->TNH",
        kernel_shape=(kv_lora_rank, num_heads, v_head_dim),
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.4.self_attn.v_up_proj",
        kernel_init=nnx.with_partitioning(nnx.initializers.uniform(), original_w_sharding)
    )
    
    method = layer.quant_method
    
    # Verify Decoupled Placeholder (Step 1)
    assert layer.weight.value.shape == (128, 65536)
    assert layer.weight.sharding == ()
    assert layer.weight_scale_inv.sharding == ()
    
    # Mock Loaded Checkpoint (OutTotal, InTotal)
    torch_weight = torch.randn(128, 65536) 
    torch_scale = torch.randn(1, 512)      # (OutBlocks, InBlocks)
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Process Weights (Step 2)
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layouts (Step 2.1.2)
    assert layer.weight.value.shape == (512, 128, 128)
    assert layer.weight_scale_inv.value.shape == (128,)
    
    if hasattr(layer.weight.sharding, 'spec'):
        assert layer.weight.sharding.spec == original_w_sharding
        assert layer.weight_scale_inv.sharding.spec == jax.sharding.PartitionSpec(None)

if __name__ == "__main__":
    pytest.main([__file__])
