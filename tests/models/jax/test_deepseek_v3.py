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

import jax
import numpy as np
import pytest
from jax.sharding import Mesh
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.deepseek_v3 import DeepseekV3ForCausalLM

# Tiny DeepSeek-V3-architecture checkpoint (4 layers, 16 routed experts) used in
# place of the full 671B model. Requires the expert count to be read from the HF
# config rather than hardcoded to 256.
MODEL_NAME = "hugg1ngfac3/deepseek-r1-tiny"

NUM_LAYERS_TO_LOAD = 4

GBYTES = 1024 * 1024 * 1024


@pytest.fixture
def mesh():
    """Single-device mesh with the full DeepSeek axis set.

    Overrides the package mesh, which lacks the ``attn_dp_expert`` and ``dcp``
    axes that DeepSeek's expert/tensor sharding references.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")
    devices = np.array(jax.local_devices()[:1]).reshape(
        (1, ) * len(MESH_AXIS_NAMES))
    with Mesh(devices, axis_names=MESH_AXIS_NAMES) as m:
        yield m


class TestDeepseekV3ForCausalLM:

    @pytest.mark.parametrize(
        "pp_rank,pp_world_size,load_format",
        [
            (rank, world, fmt)
            for rank, world in [(0, 1), (0, 4), (1, 4), (3, 4)]
            for fmt in ["skip_layers_model_loader_for_test", "jax_dummy"]
            # Single-rank dummy adds nothing over the pp-sharded dummy cases.
            if not (world == 1 and fmt == "jax_dummy")
        ])
    def test_model_loading(self, pp_rank, pp_world_size, load_format, rng,
                           mesh, mock_vllm_config,
                           assert_weight_loading_memory_bounded):
        """Builds the model and loads weights, with bounded device memory.

        ``jax_dummy`` checks the model graph builds; ``skip_layers_...`` loads
        the real FP8 weights, which only map correctly once the expert count is
        read from the HF config.
        """
        vllm_config = mock_vllm_config(MODEL_NAME, "auto")
        vllm_config.load_config.load_format = load_format
        vllm_config.load_config.num_layers_to_load_for_test = NUM_LAYERS_TO_LOAD
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.enable_expert_parallel = False

        init_pp_distributed_environment(
            ip="",
            rank=pp_rank,
            world_size=pp_world_size,
            device=jax.devices()[0],
            need_pp=False,
        )
        vllm_config.quant_config = get_tpu_quantization_config(vllm_config)

        model_config = vllm_config.model_config

        with jax.set_mesh(mesh):
            model = DeepseekV3ForCausalLM(vllm_config, rng, mesh)

        # Load weights with peak device memory bounded to catch regressions.
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            # 4GB floor matches test_qwen3_moe: the absolute floor absorbs
            # per-layer load transients that don't scale with the (few) loaded
            # layers, avoiding flakes from sub-GB run-to-run allocation jitter.
            with assert_weight_loading_memory_bounded(
                    model,
                    description=f"load_weights({MODEL_NAME}, {load_format})",
                    threshold_multiplier=0.3,
                    min_threshold_bytes=4 * GBYTES,
            ), set_current_vllm_config(vllm_config):
                loader.load_weights(model, model_config)

        assert model is not None
