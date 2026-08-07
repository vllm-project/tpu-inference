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
"""End-to-end load + forward test for compressed-tensors int8 W8A8.

Exercises the JAX path added in this PR (``Int8ChannelwiseLinearMethod``)
against a real, public ``int-quantized`` checkpoint whose scheme is:
  - weights     : int8, symmetric, per-channel, static (``weight_scale``)
  - activations : int8, symmetric, per-token, dynamic

We use a Qwen3 (dense ``Qwen3ForCausalLM``) checkpoint because that
architecture runs on the native flax_nnx path where this method lives.
(The Qwen3.5 VLM suggested in review is routed to the vLLM/torchax path
via ``_VLLM_PREFERRED_ARCHITECTURES``, so it would not exercise this code.)
"""
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.layers.jax.quantization.int8 import \
    Int8ChannelwiseLinearMethod
from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches

# A real, public compressed-tensors ``int-quantized`` W8A8 checkpoint
# (per-channel static int8 weights + per-token dynamic int8 activations),
# i.e. the exact scheme this PR dispatches to Int8ChannelwiseLinearMethod.
INT8_W8A8_MODEL = "RedHatAI/Qwen3-4B-Instruct-2507-quantized.w8a8"
# Loading a couple of decoder layers is enough to prove the int8 path;
# keeps device memory / compile time bounded for the full 4B checkpoint.
NUM_LAYERS = 2


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.models.jax.qwen3.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


class TestQwen3Int8W8A8:

    def test_int8_w8a8_load_and_forward(self, rng, mesh, mock_vllm_config):
        """Load a real int8 W8A8 checkpoint and run a forward pass.

        Asserts that (1) the checkpoint is recognized and routed to
        ``Int8ChannelwiseLinearMethod``, (2) linear weights materialize as
        int8 with a channelwise (1-D) float ``weight_scale``, and (3) a
        forward pass through the int8 matmul path yields finite outputs.
        """
        init_pp_distributed_environment(
            ip="",
            rank=0,
            world_size=1,
            device=jax.devices()[0],
            need_pp=False,
        )

        cfg = mock_vllm_config(INT8_W8A8_MODEL, "auto")
        # Only build/load a few layers -- weight loading of the first layers
        # is representative, and it bounds memory for the 4B checkpoint.
        cfg.model_config.hf_config.num_hidden_layers = NUM_LAYERS
        cfg.load_config.load_format = "skip_layers_model_loader_for_test"
        cfg.load_config.num_layers_to_load_for_test = NUM_LAYERS
        cfg.parallel_config = None
        cfg.quant_config = get_tpu_quantization_config(cfg)

        # The checkpoint's compressed-tensors config must be picked up; if the
        # int8 scheme were unsupported, model construction below would raise
        # NotImplementedError from get_quant_method.
        assert cfg.quant_config is not None

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader import get_model_loader

        with jax.set_mesh(mesh):
            model = Qwen3ForCausalLM(cfg, rng, mesh)
            loader = get_model_loader(cfg.load_config)
            with set_current_vllm_config(cfg):
                loader.load_weights(model, cfg.model_config)

        # A quantized linear must own an Int8ChannelwiseLinearMethod and have
        # materialized int8 weights + a channelwise (float) dequant scale.
        # (The param is declared fp32 but the loader keeps the checkpoint's
        # own scale dtype -- bf16 here -- same as the fp8 path.)
        layer0 = model.model.layers[model.model.start_layer]
        quantized_linears = {
            "q_proj": layer0.self_attn.q_proj,
            "o_proj": layer0.self_attn.o_proj,
            "gate_proj": layer0.mlp.gate_proj,
            "down_proj": layer0.mlp.down_proj,
        }
        for name, lin in quantized_linears.items():
            assert isinstance(lin.quant_method, Int8ChannelwiseLinearMethod), (
                f"{name} did not route to Int8ChannelwiseLinearMethod")
            assert lin.weight[...].dtype == jnp.int8, (
                f"{name}.weight is {lin.weight[...].dtype}, expected int8")
            scale = lin.weight_scale[...]
            assert jnp.issubdtype(scale.dtype, jnp.floating), (
                f"{name}.weight_scale is {scale.dtype}, expected a float scale"
            )
            # channelwise dequant scale is a 1-D per-output-feature vector
            assert scale.ndim == 1, (
                f"{name}.weight_scale shape {scale.shape} is not channelwise")

        # Forward pass exercises the int8 (dynamic per-token) matmul kernel.
        hf_config = cfg.model_config.hf_config
        hidden_size = hf_config.hidden_size
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = 128

        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=32,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            mesh=mesh,
            layer_names=["layer"] * NUM_LAYERS,
            cache_dtype=jnp.bfloat16,
        )

        num_tokens = 8
        num_reqs = 1
        max_num_blocks_per_req = 4
        input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
        attention_metadata = AttentionMetadata(
            input_positions=jnp.arange(num_tokens, dtype=jnp.int32),
            block_tables=jnp.zeros((num_reqs, max_num_blocks_per_req),
                                   dtype=jnp.int32).reshape(-1),
            seq_lens=jnp.array([num_tokens], dtype=jnp.int32),
            query_start_loc=jnp.array([0, num_tokens], dtype=jnp.int32),
            request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        )

        # The int8 matmul lowers to jax.shard_map under an (abstract) mesh, so
        # the forward must run inside jit -- eager execution leaves the
        # embedded activations on a SingleDeviceSharding that the sharding
        # constraint rejects. This mirrors how the runner invokes the model.
        @nnx.jit
        def _forward(m, kv_caches, input_ids, attention_metadata):
            return m(kv_caches, input_ids, attention_metadata)

        with jax.set_mesh(mesh):
            _, hidden_states, aux_hidden_states, _ = _forward(
                model, kv_caches, input_ids, attention_metadata)

        assert hidden_states.shape == (num_tokens, hidden_size)
        assert jnp.all(jnp.isfinite(hidden_states)), \
            "int8 forward produced non-finite hidden states"
