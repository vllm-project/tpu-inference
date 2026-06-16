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

import tempfile
import unittest

import jax
import pytest
import torch
from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh
from torch import nn
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import (JaxLinear,
                                             JaxMergedColumnParallelLinear,
                                             JaxQKVParallelLinear)
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedConfig, UnquantizedMergedLinearMethod)
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.utils.weight_utils import JaxAutoWeightsLoader


class VllmMLP(nn.Module):
    """An example MLP module using vLLM layer."""

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 quant_config=None,
                 prefix: str = ""):
        super().__init__()
        self.up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            disable_tp=True,
            quant_config=quant_config,
            prefix=prefix + ".gate_up_proj",
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            disable_tp=True,
            quant_config=quant_config,
            prefix=prefix + ".down_proj",
        )


class JaxMLP(JaxModule):
    """A MLP module using JaxLinear layer."""

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            quant_config=None,
            rng: nnx.Rngs = nnx.Rngs(0),
    ):
        super().__init__()
        self.up_proj = JaxLinear(
            hidden_size,
            intermediate_size * 2,
            use_bias=False,
            quant_config=quant_config,
            rngs=rng,
        )
        self.act_fn = nnx.silu
        self.down_proj = JaxLinear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            quant_config=quant_config,
            rngs=rng,
        )


class TestJaxLinear(unittest.TestCase):

    def test_parameter_names_match_vllm_unquantized(self):
        """Tests the parameter names of JaxLinear layer."""

        hidden_size = 16
        intermediate_size = 32
        vllm_config = VllmConfig(model_config=ModelConfig(
            model="Qwen/Qwen3-0.6B"))

        # As of vllm0.12.0, vllm.model_executor.parameter.BasevLLMParameter calls
        # get_tensor_model_parallel_rank() and
        # get_tensor_model_parallel_world_size() even though disable_tp=True, which
        # causes error during initialization. So we mock them here.
        with set_current_vllm_config(vllm_config):
            from vllm.distributed.parallel_state import (
                ensure_model_parallel_initialized,
                init_distributed_environment)
            temp_file = tempfile.mkstemp()[1]
            init_distributed_environment(
                1,
                0,
                local_rank=0,
                distributed_init_method=f"file://{temp_file}",
                backend="gloo")
            ensure_model_parallel_initialized(1, 1)
            # vllm linear layer
            vllm_mlp = VllmMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                quant_config=None,
                prefix="mlp",
            )

        mesh = Mesh(jax.devices('cpu')[:1], ("model", ))
        unquantize_config = get_tpu_quantization_config(vllm_config, mesh)
        with jax.set_mesh(mesh):
            jax_mlp = JaxMLP(
                hidden_size,
                intermediate_size,
                quant_config=unquantize_config,
                rng=nnx.Rngs(0),
            )

        self.assertDictEqual(
            {
                k: v.value.shape
                for k, v in jax_mlp.named_parameters()
            }, {
                k: tuple(v.shape)[::-1]
                for k, v in vllm_mlp.named_parameters()
            })

    def test_sharding_assignment(self):
        """Tests sharding assignment of JaxLinear layer."""

        mesh = Mesh(jax.devices('cpu')[:1], ("model", ))
        unquantize_config = get_tpu_quantization_config(
            VllmConfig(model_config=ModelConfig(model="Qwen/Qwen3-0.6B")),
            mesh)
        with jax.set_mesh(mesh):
            jax_linear = JaxLinear(
                16,
                32,
                kernel_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                  sharding=(None, "model")),
                use_bias=True,
                quant_config=unquantize_config,
                rngs=nnx.Rngs(0),
            )

        self.assertSequenceEqual(jax_linear.weight.sharding, (None, "model"))
        self.assertEqual(f"{jax.typeof(jax_linear.weight.value)}",
                         "float32[16,32]")

    @parameterized.expand([1, 2])
    def test_jax_qkv_parallel_linear_correctness(self, tp_size):
        """Tests JaxQKVParallelLinear correctness against separate JaxEinsum projections."""
        num_devices = len(jax.devices('cpu'))
        if tp_size > num_devices:
            self.skipTest(
                f"Not enough CPU devices (required: {tp_size}, available: {num_devices})"
            )

        hidden_size = 32
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8
        seq_len = 3
        use_bias = True

        mesh = Mesh(jax.devices('cpu')[:tp_size], ("model", ))
        with jax.set_mesh(mesh):
            rngs = nnx.Rngs(0)

            # 1. Create separate projections
            q_proj = JaxEinsum(
                "TD,DNH->TNH",
                (hidden_size, num_heads, head_dim),
                bias_shape=(num_heads, head_dim) if use_bias else None,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
            k_proj = JaxEinsum(
                "TD,DKH->TKH",
                (hidden_size, num_kv_heads, head_dim),
                bias_shape=(num_kv_heads, head_dim) if use_bias else None,
                param_dtype=jnp.float32,
                rngs=rngs,
            )
            v_proj = JaxEinsum(
                "TD,DKH->TKH",
                (hidden_size, num_kv_heads, head_dim),
                bias_shape=(num_kv_heads, head_dim) if use_bias else None,
                param_dtype=jnp.float32,
                rngs=rngs,
            )

            # Assign random weights to separate projections
            key = jax.random.PRNGKey(42)
            k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

            q_w = jax.random.normal(k1, (hidden_size, num_heads, head_dim))
            k_w = jax.random.normal(k2, (hidden_size, num_kv_heads, head_dim))
            v_w = jax.random.normal(k3, (hidden_size, num_kv_heads, head_dim))

            q_proj.weight.value = q_w
            k_proj.weight.value = k_w
            v_proj.weight.value = v_w

            if use_bias:
                q_b = jax.random.normal(k4, (num_heads, head_dim))
                k_b = jax.random.normal(k5, (num_kv_heads, head_dim))
                v_b = jax.random.normal(k6, (num_kv_heads, head_dim))
                q_proj.bias.value = q_b
                k_proj.bias.value = k_b
                v_proj.bias.value = v_b

            # 2. Prepare raw sequential weights/biases
            raw_w = jnp.concatenate([
                q_w.reshape(hidden_size, -1),
                k_w.reshape(hidden_size, -1),
                v_w.reshape(hidden_size, -1),
            ],
                                    axis=-1)

            if use_bias:
                raw_b = jnp.concatenate([
                    q_b.reshape(-1),
                    k_b.reshape(-1),
                    v_b.reshape(-1),
                ],
                                        axis=-1)

            # Rearrange using reorder_concatenated_tensor_for_sharding
            split_sizes = [
                num_heads * head_dim, num_kv_heads * head_dim,
                num_kv_heads * head_dim
            ]
            rearranged_w = reorder_concatenated_tensor_for_sharding(
                raw_w, split_sizes, tp_size, dim=-1)
            if use_bias:
                rearranged_b = reorder_concatenated_tensor_for_sharding(
                    raw_b, split_sizes, tp_size, dim=-1)

            # 3. Initialize JaxQKVParallelLinear and set the rearranged weights
            qkv_proj = JaxQKVParallelLinear(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                use_bias=use_bias,
                dtype=jnp.float32,
                rngs=rngs,
                tp_size=tp_size,
            )
            qkv_proj.weight.value = rearranged_w
            if use_bias:
                qkv_proj.bias.value = rearranged_b

            # 4. Run forward pass
            x = jax.random.normal(k7, (seq_len, hidden_size))

            q_expected = q_proj(x)
            k_expected = k_proj(x)
            v_expected = v_proj(x)

            q_actual, k_actual, v_actual = qkv_proj(x)

            # 5. Assert equality
            self.assertTrue(
                jnp.allclose(q_expected, q_actual, atol=1e-5),
                f"Q projection output mismatch for tp_size={tp_size}")
            self.assertTrue(
                jnp.allclose(k_expected, k_actual, atol=1e-5),
                f"K projection output mismatch for tp_size={tp_size}")
            self.assertTrue(
                jnp.allclose(v_expected, v_actual, atol=1e-5),
                f"V projection output mismatch for tp_size={tp_size}")

    def test_merged_column_parallel_fuses_projections(self):
        """JaxMergedColumnParallelLinear exposes a single fused kernel sized to
        the sum of its projection widths (no separate gate/up params) and
        dispatches to the merged unquantized method, which carries each
        projection's size so the per-shard interleave is well defined."""

        hidden_size = 16
        intermediate_size = 32
        mesh = Mesh(jax.devices('cpu')[:1], ("model", ))
        with jax.set_mesh(mesh):
            jax_merged = JaxMergedColumnParallelLinear(
                hidden_size,
                [intermediate_size, intermediate_size],
                use_bias=False,
                quant_config=UnquantizedConfig({}),
                rngs=nnx.Rngs(0),
                prefix="mlp.gate_up_proj",
            )

        # One fused kernel of shape (in, gate + up); no split gate/up params.
        self.assertEqual(list(dict(jax_merged.named_parameters()).keys()),
                         ["weight"])
        self.assertEqual(tuple(jax_merged.weight.value.shape),
                         (hidden_size, 2 * intermediate_size))

        method = jax_merged.quant_method
        self.assertIsInstance(method, UnquantizedMergedLinearMethod)
        self.assertEqual(method.linear_config.output_sizes,
                         [intermediate_size, intermediate_size])
        self.assertEqual(method.linear_config.n_shards, 1)

    def test_merged_column_parallel_sharding(self):
        """The fused kernel's output dim (gate + up) is sharded on `model`."""

        mesh = Mesh(jax.devices('cpu')[:1], ("model", ))
        with jax.set_mesh(mesh):
            jax_merged = JaxMergedColumnParallelLinear(
                16,
                [32, 32],
                kernel_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                  sharding=(None, "model")),
                use_bias=False,
                quant_config=UnquantizedConfig({}),
                rngs=nnx.Rngs(0),
                prefix="mlp.gate_up_proj",
            )

        self.assertSequenceEqual(jax_merged.weight.sharding, (None, "model"))
        self.assertEqual(f"{jax.typeof(jax_merged.weight.value)}",
                         "float32[16,64]")


class TestJaxQKVParallelLinear:
    """Tests for JaxQKVParallelLinear using pytest parametrize."""

    @pytest.mark.parametrize("tp_size", [1, jax.local_device_count()])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_consolidated_weight_correctness(self, tp_size, use_bias):
        """Tests JaxQKVParallelLinear correctness against separate torch projections."""
        hidden_size = 32
        num_heads = 4
        num_kv_heads = 2
        head_dim = 8
        seq_len = 3

        # Use torch to generate random weights and inputs in [-0.5, 0.5).
        # Small values keep the matmul output magnitude low so bfloat16
        # accumulation errors stay within a tight atol.
        torch.manual_seed(42)
        q_w = torch.rand(num_heads * head_dim, hidden_size) - 0.5
        k_w = torch.rand(num_kv_heads * head_dim, hidden_size) - 0.5
        v_w = torch.rand(num_kv_heads * head_dim, hidden_size) - 0.5
        x_t = torch.rand(seq_len, hidden_size) - 0.5
        q_b = torch.rand(num_heads * head_dim) - 0.5 if use_bias else None
        k_b = torch.rand(num_kv_heads * head_dim) - 0.5 if use_bias else None
        v_b = torch.rand(num_kv_heads * head_dim) - 0.5 if use_bias else None

        def _ref(x, w, b, *shape):
            out = x @ w.T
            if b is not None:
                out = out + b
            return out.reshape(*shape).numpy()

        q_exp = _ref(x_t, q_w, q_b, seq_len, num_heads, head_dim)
        k_exp = _ref(x_t, k_w, k_b, seq_len, num_kv_heads, head_dim)
        v_exp = _ref(x_t, v_w, v_b, seq_len, num_kv_heads, head_dim)

        # Concatenate weights in HF format (total_out, in) for weight loading.
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        qkv_b = torch.cat([q_b, k_b, v_b], dim=0) if use_bias else None

        mesh = Mesh(jax.devices()[:tp_size], ("model", ))
        with jax.set_mesh(mesh):
            # Build the layer and load weights via JaxAutoWeightsLoader.
            # The loader auto-transposes the (total_out, in) HF weights to
            # the (in, total_out) JAX kernel layout.
            qkv_proj = JaxQKVParallelLinear(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                use_bias=use_bias,
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )
            checkpoint = [("weight", qkv_w)]
            if use_bias:
                checkpoint.append(("bias", qkv_b))
            JaxAutoWeightsLoader(qkv_proj).load_weights(checkpoint)

            # Run forward pass and compare against torch float32 reference.
            # atol=1e-2 covers the ~0.003 gap between CPU float32 matmul and
            # TPU float32 matmul (which uses bfloat16 accumulators internally).
            x = jnp.array(x_t.numpy())
            q_actual, k_actual, v_actual = qkv_proj(x)

            assert jnp.allclose(jnp.array(q_exp), q_actual,
                                atol=1e-2), f"Q mismatch for tp_size={tp_size}"
            assert jnp.allclose(jnp.array(k_exp), k_actual,
                                atol=1e-2), f"K mismatch for tp_size={tp_size}"
            assert jnp.allclose(jnp.array(v_exp), v_actual,
                                atol=1e-2), f"V mismatch for tp_size={tp_size}"
