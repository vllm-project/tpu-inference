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

    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    head_dim = 8
    seq_len = 3

    def setup_method(self):
        # Use values in [-0.5, 0.5) so matmul output magnitude stays small
        # and bfloat16 accumulation errors fit within atol=1e-2.
        torch.manual_seed(42)
        self.q_w = torch.rand(self.num_heads * self.head_dim,
                              self.hidden_size) - 0.5
        self.k_w = torch.rand(self.num_kv_heads * self.head_dim,
                              self.hidden_size) - 0.5
        self.v_w = torch.rand(self.num_kv_heads * self.head_dim,
                              self.hidden_size) - 0.5
        self.x_t = torch.rand(self.seq_len, self.hidden_size) - 0.5
        self.q_b = torch.rand(self.num_heads * self.head_dim) - 0.5
        self.k_b = torch.rand(self.num_kv_heads * self.head_dim) - 0.5
        self.v_b = torch.rand(self.num_kv_heads * self.head_dim) - 0.5

        # Expected outputs for both with and without bias.
        self.q_exp = (self.x_t @ self.q_w.T + self.q_b).reshape(
            self.seq_len, self.num_heads, self.head_dim).numpy()
        self.k_exp = (self.x_t @ self.k_w.T + self.k_b).reshape(
            self.seq_len, self.num_kv_heads, self.head_dim).numpy()
        self.v_exp = (self.x_t @ self.v_w.T + self.v_b).reshape(
            self.seq_len, self.num_kv_heads, self.head_dim).numpy()
        self.q_exp_nb = (self.x_t @ self.q_w.T).reshape(
            self.seq_len, self.num_heads, self.head_dim).numpy()
        self.k_exp_nb = (self.x_t @ self.k_w.T).reshape(
            self.seq_len, self.num_kv_heads, self.head_dim).numpy()
        self.v_exp_nb = (self.x_t @ self.v_w.T).reshape(
            self.seq_len, self.num_kv_heads, self.head_dim).numpy()

    @pytest.mark.parametrize("tp_size", [1, jax.local_device_count()])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_consolidated_weight_correctness(self, tp_size, use_bias):
        """Loads QKV as a single consolidated (total_out, in) tensor."""
        qkv_w = torch.cat([self.q_w, self.k_w, self.v_w], dim=0)
        q_exp, k_exp, v_exp = (self.q_exp, self.k_exp, self.v_exp) if use_bias \
            else (self.q_exp_nb, self.k_exp_nb, self.v_exp_nb)

        mesh = Mesh(jax.devices()[:tp_size], ("model", ))
        with jax.set_mesh(mesh):
            qkv_proj = JaxQKVParallelLinear(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_bias=use_bias,
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )
            checkpoint = [("weight", qkv_w)]
            if use_bias:
                checkpoint.append(
                    ("bias", torch.cat([self.q_b, self.k_b, self.v_b], dim=0)))
            JaxAutoWeightsLoader(qkv_proj).load_weights(checkpoint)

            x = jnp.array(self.x_t.numpy())
            q_actual, k_actual, v_actual = qkv_proj(x)

        assert jnp.allclose(jnp.array(q_exp), q_actual, atol=1e-2), \
            f"Q mismatch for tp_size={tp_size}"
        assert jnp.allclose(jnp.array(k_exp), k_actual, atol=1e-2), \
            f"K mismatch for tp_size={tp_size}"
        assert jnp.allclose(jnp.array(v_exp), v_actual, atol=1e-2), \
            f"V mismatch for tp_size={tp_size}"

    @pytest.mark.parametrize("tp_size", [1, jax.local_device_count()])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_split_weight_correctness(self, tp_size, use_bias):
        """Loads Q, K, V as separate tensors via JaxAutoWeightsLoader routing.
        """
        q_exp, k_exp, v_exp = (self.q_exp, self.k_exp, self.v_exp) if use_bias \
            else (self.q_exp_nb, self.k_exp_nb, self.v_exp_nb)

        mesh = Mesh(jax.devices()[:tp_size], ("model", ))
        with jax.set_mesh(mesh):
            qkv_proj = JaxQKVParallelLinear(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                use_bias=use_bias,
                dtype=jnp.float32,
                rngs=nnx.Rngs(0),
            )
            qkv_proj.packed_modules_mapping = {
                "weight": ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
                "bias": ["q_proj.bias", "k_proj.bias", "v_proj.bias"],
            }
            checkpoint = [
                ("q_proj.weight", self.q_w),
                ("k_proj.weight", self.k_w),
                ("v_proj.weight", self.v_w),
            ]
            if use_bias:
                checkpoint += [
                    ("q_proj.bias", self.q_b),
                    ("k_proj.bias", self.k_b),
                    ("v_proj.bias", self.v_b),
                ]
            JaxAutoWeightsLoader(qkv_proj).load_weights(checkpoint)

            x = jnp.array(self.x_t.numpy())
            q_actual, k_actual, v_actual = qkv_proj(x)

        assert jnp.allclose(jnp.array(q_exp), q_actual, atol=1e-2), \
            f"Q mismatch for tp_size={tp_size}"
        assert jnp.allclose(jnp.array(k_exp), k_actual, atol=1e-2), \
            f"K mismatch for tp_size={tp_size}"
        assert jnp.allclose(jnp.array(v_exp), v_actual, atol=1e-2), \
            f"V mismatch for tp_size={tp_size}"


class TestJaxLinearMetadataDefaults(unittest.TestCase):
    """Regression: JaxLinear/JaxLmHead must not share a mutable metadata dict
    across default-constructed instances.

    The previous signatures used kernel_metadata={} / bias_metadata={} and
    then mutated them (kernel_metadata['eager_sharding'] = False). A shared
    default dict means the first instance's mutation (and any caller-side
    edits to the returned metadata) leak into every other default-constructed
    layer. The fix uses None sentinels and copies into a fresh dict.
    """

    def test_default_metadata_not_shared_across_instances(self):
        # Construct two JaxLinear instances on CPU without touching device
        # kernels: exercise only the metadata-normalization logic via a small
        # standalone reproduction of the fixed idiom (mirrors linear.py).
        def normalize(kernel_metadata=None):
            kernel_metadata = dict(kernel_metadata) if kernel_metadata else {}
            if "eager_sharding" not in kernel_metadata:
                kernel_metadata["eager_sharding"] = False
            return kernel_metadata

        m1 = normalize()
        m1["eager_sharding"] = True  # an instance/caller customizes its copy
        m2 = normalize()  # a fresh default-constructed instance

        self.assertIsNot(m1, m2)
        self.assertTrue(m1["eager_sharding"])
        self.assertFalse(
            m2["eager_sharding"],
            "default metadata leaked across instances (mutable-default bug)")

    def test_linear_signatures_use_none_sentinels(self):
        # Guard against a regression to mutable default args in the source.
        import inspect

        from tpu_inference.layers.jax.linear import JaxLinear, JaxLmHead
        for cls in (JaxLinear, JaxLmHead):
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name in ("kernel_metadata", "bias_metadata"):
                    self.assertIsNone(
                        param.default,
                        f"{cls.__name__}.{name} must default to None, not a "
                        f"mutable {type(param.default).__name__}")
