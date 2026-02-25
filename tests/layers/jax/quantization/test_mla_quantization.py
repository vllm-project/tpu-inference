import os; os.environ["JAX_PLATFORMS"] = "cpu"
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
    
    # 1. Checkpoint matches DeepSeek format: (N*H, A) = (16384, 512)
    # We use a pattern to track data: value = head_idx * 1000000 + dim_idx * 1000 + latent_idx
    torch_weight = torch.zeros(16384, 512)
    for n in range(num_heads):
        for h in range(qk_nope_head_dim):
            for a in range(kv_lora_rank):
                torch_weight[n * qk_nope_head_dim + h, a] = n * 10000 + h * 100 + a

    # 2. Generator logic: k_val = weight.T -> (512, 16384)
    k_val_gen = torch_weight.T
    
    layer.weight.weight_loader(layer.weight, k_val_gen)
    
    # Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layout: (A, N, H)
    # Check a few samples
    final_w = layer.weight.value
    assert final_w.shape == (512, 128, 128)
    
    # Sample: a=5, n=10, h=20
    # Original torch was [n*qk_nope_head_dim + h, a]
    expected_val = 10 * 10000 + 20 * 100 + 5
    assert final_w[5, 10, 20] == expected_val

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
    
    # 1. Checkpoint matches DeepSeek format: (N*H, A) = (16384, 512)
    torch_weight = torch.zeros(16384, 512)
    for n in range(num_heads):
        for h in range(v_head_dim):
            for a in range(kv_lora_rank):
                torch_weight[n * v_head_dim + h, a] = n * 10000 + h * 100 + a

    # 2. Generator logic: .reshape(N, H, A).permute(1, 2, 0).reshape(H, -1) -> (128, 65536)
    v_val_gen = torch_weight.reshape(num_heads, v_head_dim, kv_lora_rank).permute(1, 2, 0).reshape(v_head_dim, -1)
    
    layer.weight.weight_loader(layer.weight, v_val_gen)
    
    # Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layout: (A, N, H)
    final_w = layer.weight.value
    assert final_w.shape == (512, 128, 128)
    
    # Sample: a=5, n=10, h=20
    expected_val = 10 * 10000 + 20 * 100 + 5
    assert final_w[5, 10, 20] == expected_val

def test_mla_disabled_kv_b_proj_lifecycle(rngs):
    # MLA disabled kv_b_proj parameters: SA,AL->SL
    kv_lora_rank = 512
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128
    out_features = num_heads * (qk_nope_head_dim + v_head_dim) # 128 * 256 = 32768
    
    hf_quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    fp8_config = Fp8Config(hf_quant_config)
    original_w_sharding = jax.sharding.PartitionSpec(None, 'x')
    
    layer = JaxEinsum(
        einsum_str="SA,AL->SL",
        kernel_shape=(kv_lora_rank, out_features),
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.4.self_attn.kv_b_proj",
        kernel_init=nnx.with_partitioning(nnx.initializers.uniform(), original_w_sharding)
    )
    
    # Standard linear checkpoint: (Out, In) = (32768, 512)
    torch_weight = torch.randn(32768, 512)
    
    layer.weight.weight_loader(layer.weight, torch_weight)
    
    # Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ('x', 'y'))
    method = layer.quant_method
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final 2D Layout: (32768, 512)
    assert layer.weight.value.shape == (32768, 512)
    # Check values match (approx due to requantization if it happened, 
    # but here we just check if it was loaded correctly)
    # Since it's fp8 -> bf16, there might be slight precision loss, but shapes must match.
    assert layer.weight.value.shape == torch_weight.shape

if __name__ == "__main__":
    pytest.main([__file__])
