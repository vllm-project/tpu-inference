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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import tpu_inference.runner.compilation_manager as compilation_manager_module
from tpu_inference.runner.compilation_manager import CompilationManager


class TestPrecompileBackbone:

    def test_hybrid_layers_share_one_block_table_per_group(self, monkeypatch):

        class FakeArray:

            def __init__(self, shape):
                self.shape = tuple(shape)

        groups = [
            SimpleNamespace(layer_names=["mamba.0", "mamba.1"]),
            SimpleNamespace(layer_names=["attention.0"]),
        ]
        runner = SimpleNamespace(
            vllm_config=SimpleNamespace(sharding_config=SimpleNamespace(
                total_dp_size=1)),
            mesh=object(),
            max_num_reqs=8,
            kv_cache_config=SimpleNamespace(
                has_mamba_layers=True,
                kv_cache_groups=groups,
            ),
            kv_cache_manager=SimpleNamespace(uses_compact_mamba_state=False),
            input_batch=SimpleNamespace(block_table=[
                SimpleNamespace(max_num_blocks_per_req=64),
                SimpleNamespace(max_num_blocks_per_req=1024),
            ]),
            speculative_config=None,
            enable_multitoken_decode=False,
            maybe_select_dummy_loras=lambda *args, **kwargs: nullcontext(),
            lora_config=None,
            lora_utils=SimpleNamespace(extract_lora_metadata=lambda: None),
            state_leaves=(),
            kv_caches=(),
            layer_name_to_kvcache_index={},
            is_first_rank=True,
            is_last_rank=True,
            rank=0,
        )
        manager = object.__new__(CompilationManager)
        manager.runner = runner
        manager._create_dummy_tensor = MagicMock(
            side_effect=lambda shape, *args, **kwargs: FakeArray(shape))
        manager._run_compilation = MagicMock()
        monkeypatch.setattr(
            "tpu_inference.runner.compilation_manager.NamedSharding",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            "tpu_inference.runner.compilation_manager.device_array",
            lambda mesh, value, sharding=None: FakeArray(np.shape(value)),
        )

        manager._precompile_backbone_helper(
            "backbone",
            input_ids=FakeArray((16, )),
            positions=FakeArray((16, )),
            inputs_embeds=None,
            num_reqs=8,
        )

        attention_metadata = manager._run_compilation.call_args.args[5]
        assert (attention_metadata["mamba.0"].block_tables
                is attention_metadata["mamba.1"].block_tables)
        assert (attention_metadata["mamba.0"].block_tables
                is not attention_metadata["attention.0"].block_tables)
        assert attention_metadata["mamba.0"].mamba_state_indices is None
        assert attention_metadata["attention.0"].mamba_state_indices is None


class TestCaptureModel:

    def test_mamba_copy_precompile_precedes_backbone_warmups(
            self, monkeypatch):
        events = []
        runner = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            maybe_setup_dummy_loras=lambda _config: nullcontext(),
            lora_config=None,
            mesh=object(),
            mamba_state_manager=SimpleNamespace(
                enabled=True,
                precompile_copy_state_blocks=lambda: events.append("copy"),
            ),
            is_multimodal_model=True,
            precompile_vision_encoder_fn=None,
            scheduler_config=SimpleNamespace(async_scheduling=False),
            speculative_config=None,
            is_last_rank=False,
        )
        manager = object.__new__(CompilationManager)
        manager.runner = runner
        manager._run_compilation = lambda _name, fn: fn()
        manager._precompile_backbone_text_only = lambda: events.append("text")
        manager._precompile_input_embeddings_merger = lambda: None
        manager._precompile_backbone_with_inputs_embeds = lambda: events.append(
            "embeds")
        monkeypatch.setattr(compilation_manager_module.envs,
                            "SKIP_JAX_PRECOMPILE", False)
        monkeypatch.setattr(compilation_manager_module.jax, "set_mesh",
                            lambda _mesh: nullcontext())

        manager.capture_model()

        assert events == ["copy", "text", "embeds"]
