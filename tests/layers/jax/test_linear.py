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
from flax import nnx
from jax.sharding import Mesh
from torch import nn
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config


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

        self.assertSequenceEqual(jax_linear.weight.sharding,
                                 (None, "model"))
        self.assertEqual(f"{jax.typeof(jax_linear.weight.value)}",
                         "float32[16,32]")
