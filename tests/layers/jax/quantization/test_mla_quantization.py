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

def test_standard_linear_lifecycle(rngs):
    # Standard 2D linear MatMul(X, W). Weight is (In, Out) = (512, 256)
    in_features = 512
    out_features = 256
    
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
    
    # 1. Mock Checkpoint: (Out, In) = (256, 512)
    torch_weight = torch.zeros(256, 512)
    for r in range(256):
        for c in range(512):
            torch_weight[r, c] = (r % 10) * 1.0 + (c % 10) * 0.01
            
    # Patterned Scales: (OutBlocks=2, InBlocks=4)
    torch_scale = torch.zeros(2, 4)
    for ob in range(2):
        for ib in range(4):
            torch_scale[ob, ib] = 1.0 + ob * 0.1 + ib * 0.01
            
    layer.weight.weight_loader(layer.weight, torch_weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, torch_scale)
    
    # 2. Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1), ('x',))
    method.linear_config.mesh = mesh
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # 3. Verify Reality: Standard JaxLinear MUST BE (In, Out) = (512, 256)
    final_w = layer.weight.value
    final_s = layer.weight_scale_inv.value
    assert final_w.shape == (512, 256)
    
    # Sample check at block boundary: row=130 (In), col=5 (Out)
    # Original checkpoint (Out, In) index was (5, 130)
    expected_val = (5 % 10) * 1.0 + (130 % 10) * 0.01
    # Dequantize: final_w[in, out] * final_s[out]
    actual_val = float(final_w[130, 5]) * float(final_s[5])
    assert np.allclose(actual_val, expected_val, atol=0.1)

    # NEW: Verify Forward Pass
    x = jnp.ones((1, 512), dtype=jnp.bfloat16)
    try:
        y = layer(x)
        assert y.shape == (1, 256)
        assert not jnp.isnan(y).any()
    except Exception as e:
        pytest.fail(f"Forward pass failed for JaxLinear: {e}")

def test_mla_k_up_proj_lifecycle(rngs):
    # MLA k_up_proj parameters: TNH,ANH->TNA. Weight is (A, N, H) = (512, 128, 128)
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
    torch_weight = torch.zeros(16384, 512)
    for nh in range(16384):
        for a in range(512):
            torch_weight[nh, a] = (nh % 10) * 1.0 + (a % 10) * 0.01
            
    # Generator logic: k_val = weight.T -> (512, 16384)
    k_val_gen = torch_weight.T.contiguous()
            
    # Scales: checkpoint is (N, A_blocks) = (128, 4)
    # Generator logic: k_val = weight[:split_idx, ...].T.contiguous() -> (4, 128)
    torch_scale = torch.zeros(128, 4)
    for n in range(128):
        for ab in range(4):
            torch_scale[n, ab] = 1.0 + n * 0.01 + ab * 0.1
    k_scale_gen = torch_scale.T.contiguous()
            
    layer.weight.weight_loader(layer.weight, k_val_gen)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, k_scale_gen)
    
    # 2. Process Weights on multi-device mesh
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:4]).reshape(4), ('x',))
    method.linear_config.mesh = mesh
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # 3. Verify Final ND Layout: (A, N, H) = (512, 128, 128)
    final_w = layer.weight.value
    final_s = layer.weight_scale_inv.value
    assert final_w.shape == (512, 128, 128)
    assert len(final_w.addressable_shards) > 1
    
    # Scale final_s should be (A, InBlocks) = (512, 128)
    # Checkpoint (NH, A) was (259, 130). 
    # nh=259 -> block_idx = 259 // 128 = 2.
    expected_val = (259 % 10) * 1.0 + (130 % 10) * 0.01
    actual_val = float(final_w[130, 2, 3]) * float(final_s[130, 2])
    assert np.allclose(actual_val, expected_val, atol=0.1)

    # NEW: Verify Forward Pass
    # x: (T, N, H) = (1, 128, 128)
    x = jnp.ones((1, 128, 128), dtype=jnp.bfloat16)
    try:
        y = layer(x)
        assert y.shape == (1, 128, 512)
        assert not jnp.isnan(y).any()
    except Exception as e:
        pytest.fail(f"Forward pass failed for k_up_proj: {e}")

def test_mla_v_up_proj_lifecycle(rngs):
    # MLA v_up_proj parameters: TNA,ANH->TNH. Weight is (A, N, H) = (512, 128, 128)
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
    for nh in range(16384):
        for a in range(512):
            torch_weight[nh, a] = (nh % 10) * 1.0 + (a % 10) * 0.01
            
    # Generator logic: .reshape(N, H, A).permute(1, 2, 0).reshape(H, -1) -> (128, 65536)
    v_val_gen = torch_weight.reshape(num_heads, v_head_dim, kv_lora_rank).permute(1, 2, 0).reshape(v_head_dim, -1).contiguous()
            
    # Scales: checkpoint is (N, A_blocks) = (128, 4)
    # Generator logic: v_val = weight[split_idx:, ...].T.contiguous().reshape(1, -1).contiguous() -> (1, 512)
    torch_scale = torch.zeros(128, 4)
    for n in range(128):
        for ab in range(4):
            torch_scale[n, ab] = 1.0 + n * 0.01 + ab * 0.1
    v_scale_gen = torch_scale.T.contiguous().reshape(1, -1).contiguous()
            
    layer.weight.weight_loader(layer.weight, v_val_gen)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, v_scale_gen)
    
    # 2. Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:4]).reshape(4), ('x',))
    method.linear_config.mesh = mesh
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # 3. Verify Final ND Layout: (A, N, H) = (512, 128, 128)
    final_w = layer.weight.value
    final_s = layer.weight_scale_inv.value
    assert final_w.shape == (512, 128, 128)
    
    # Scale final_s should be (H, InBlocks) = (128, 512)
    # Checkpoint (NH, A) was (259, 4).
    # Generator v_val_gen layout is (H, A*N).
    # nh=259, a=4 -> h=3, a=4, n=2.
    # index in A*N is a*num_heads + n = 4*128 + 2 = 514.
    # block_idx = 514 // 128 = 4.
    expected_val = (259 % 10) * 1.0 + (4 % 10) * 0.01
    actual_val = float(final_w[4, 2, 3]) * float(final_s[3, 4])
    assert np.allclose(actual_val, expected_val, atol=0.1)

    # NEW: Verify Forward Pass
    # x: (T, N, A) = (1, 128, 512)
    x = jnp.ones((1, 128, 512), dtype=jnp.bfloat16)
    try:
        y = layer(x)
        assert y.shape == (1, 128, 128)
        assert not jnp.isnan(y).any()
    except Exception as e:
        pytest.fail(f"Forward pass failed for v_up_proj: {e}")

def test_mla_disabled_kv_b_proj_lifecycle(rngs):
    # Standard linear Einsum: MatMul(X, W). Weight is (In, Out) = (512, 32768)
    kv_lora_rank = 512
    num_heads = 128
    qk_nope_head_dim = 128
    v_head_dim = 128
    out_features = num_heads * (qk_nope_head_dim + v_head_dim)
    
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
    
    # 1. Mock Checkpoint: (Out, In) = (32768, 512)
    torch_weight = torch.randn(32768, 512)
    layer.weight.weight_loader(layer.weight, torch_weight)
    
    # 2. Process Weights
    devices = jax.devices()
    mesh = jax.sharding.Mesh(np.array(devices[:1]).reshape(1), ('x',))
    method = layer.quant_method
    method.linear_config.mesh = mesh
    
    with jax.set_mesh(mesh):
        method.process_weights_after_loading(layer)
        
    # 3. Verify physical layout: MUST be (512, 32768) to match JaxEinsum(SA,AL->SL)
    assert layer.weight.value.shape == (512, 32768)

    # NEW: Verify Forward Pass
    x = jnp.ones((1, 512), dtype=jnp.bfloat16)
    try:
        y = layer(x)
        assert y.shape == (1, 32768)
        assert not jnp.isnan(y).any()
    except Exception as e:
        pytest.fail(f"Forward pass failed for disabled MLA kv_b_proj: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
