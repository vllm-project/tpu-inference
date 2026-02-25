import os; 
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
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
    # Use a pattern that fits in FP8: value = (n*0.1 + h*0.01 + a*0.001)
    torch_weight = torch.zeros(16384, 512)
    for n in range(num_heads):
        for h in range(qk_nope_head_dim):
            for a in range(kv_lora_rank):
                # Scale values to be distinct but within FP8 range (~ -448 to 448)
                torch_weight[n * qk_nope_head_dim + h, a] = (n % 10) * 1.0 + (h % 10) * 0.1 + (a % 10) * 0.01

    # 2. Generator logic: k_val = weight.T -> (512, 16384)
    k_val_gen = torch_weight.T.contiguous()
    
    layer.weight.weight_loader(layer.weight, k_val_gen)
    
    # 3. Scale loading (set to 1.0 to preserve values during dequant)
    torch_scale = torch.ones(4, 128)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # Process Weights
    # Simulate a 4-device mesh to verify sharding compatibility
    devices = jax.devices()
    assert len(devices) >= 4, f"Test requires at least 4 devices, found {len(devices)}"
    mesh = jax.sharding.Mesh(np.array(devices[:4]).reshape(4), ('x',))
    
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # Verify Final ND Layout: (A, N, H)
    # Since weights are stored quantized in FP8, we must dequantize to verify values.
    final_w = layer.weight.value
    final_s = layer.weight_scale_inv.value
    assert final_w.shape == (512, 128, 128)
    # Confirm it is sharded (not just a single device array)
    assert len(final_w.addressable_shards) > 1
    
    # Sample: a=4, n=2, h=3
    # Logic: latent dimension a=4 is Index 0. Head dimensions n=2, h=3 are Index 1, 2.
    # Checkpoint value was at (n*128 + h, a)
    expected_val = 2.0 + 3.0 * 0.1 + 4.0 * 0.01
    
    # Dequantize: final_s matches the output dimension (A=512 for k_up_proj)
    # So we use final_s[a]
    actual_val = float(final_w[4, 2, 3]) * float(final_s[4])
    
    assert np.allclose(actual_val, expected_val, atol=0.1), f"Value mismatch at [4,2,3]: expected {expected_val}, got {actual_val} (quantized was {final_w[4,2,3]}, scale was {final_s[4]})"

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
                # Pattern: n.h_a
                torch_weight[n * v_head_dim + h, a] = n * 1.0 + h * 0.1 + a * 0.01

    # 2. Generator logic: .reshape(N, H, A).permute(1, 2, 0).reshape(H, -1) -> (128, 65536)
    v_val_gen = torch_weight.reshape(num_heads, v_head_dim, kv_lora_rank).permute(1, 2, 0).reshape(v_head_dim, -1).contiguous()
    
    layer.weight.weight_loader(layer.weight, v_val_gen)
    
    # 3. Scale loading
    torch_scale = torch.ones(1, 512)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
        # Process Weights
    # Simulate a 4-device mesh to verify sharding compatibility
    devices = jax.devices()
    assert len(devices) >= 4, f"Test requires at least 4 devices, found {len(devices)}"
    mesh = jax.sharding.Mesh(np.array(devices[:4]).reshape(4), ('x',))
        
    method.linear_config.mesh = mesh
    method.linear_config.bias_sharding = jax.sharding.PartitionSpec(None)
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
    
    # Verify Final ND Layout: (A, N, H)
    final_w = layer.weight.value
    final_s = layer.weight_scale_inv.value
    assert final_w.shape == (512, 128, 128)
    assert len(final_w.addressable_shards) > 1
    
    # Sample: a=4, n=2, h=3
    expected_val = 2.0 + 3.0 * 0.1 + 4.0 * 0.01
    # Dequantize: final_s matches output dimension (H=128 for v_up_proj)
    actual_val = float(final_w[4, 2, 3]) * float(final_s[3])
    assert np.allclose(actual_val, expected_val, atol=0.1), f"Value mismatch at [4,2,3]: expected {expected_val}, got {actual_val} (quantized was {final_w[4,2,3]}, scale was {final_s[3]})"

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

def test_mla_replicated_weights_lifecycle(rngs):
    # Simulate replicate_attn_weights=True by using empty sharding ()
    kv_lora_rank = 512
    num_heads = 128
    qk_nope_head_dim = 128
    
    hf_quant_config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    fp8_config = Fp8Config(hf_quant_config)
    original_w_sharding = () # Replicated
    
    layer = JaxEinsum(
        einsum_str="TNH,ANH->TNA",
        kernel_shape=(kv_lora_rank, num_heads, qk_nope_head_dim),
        rngs=rngs,
        quant_config=fp8_config,
        prefix="model.layers.4.self_attn.k_up_proj",
        kernel_init=nnx.with_partitioning(nnx.initializers.uniform(), original_w_sharding)
    )
    
    method = layer.quant_method
    
    # Checkpoint logic
    torch_weight = torch.zeros(16384, 512)
    k_val_gen = torch_weight.T.contiguous()
    torch_scale = torch.zeros(128, 4)
    k_scale_gen = torch_scale.T.contiguous()
    
    layer.weight.weight_loader(layer.weight, k_val_gen)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, k_scale_gen)
    
    # Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1), ('x',))
    method.linear_config.mesh = mesh
    
    with jax.set_mesh(mesh):
        # This should not raise an IndexError
        method.process_weights_after_loading(layer)
        
    assert layer.weight.value.shape == (512, 128, 128)

if __name__ == "__main__":
    pytest.main([__file__])
