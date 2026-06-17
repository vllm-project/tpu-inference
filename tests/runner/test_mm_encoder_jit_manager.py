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
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torchax.interop import jax_view
from vllm.config import (CompilationConfig, ModelConfig, MultiModalConfig,
                         VllmConfig, set_current_vllm_config)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs, EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers)

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    shard_model_to_tpu
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.models.vllm.vllm_model_wrapper import _VllmRunner
from tpu_inference.runner.mm_encoder_jit_manager import (
    MMEncoderJITManager, _TorchaxEncoderModelAdapter)


def _make_vllm_config(budgets=None, max_vision_items=0):
    return VllmConfig(
        model_config=ModelConfig(
            dtype=torch.bfloat16,
            multimodal_config=MultiModalConfig(limit_per_prompt={"video": 0}),
        ),
        compilation_config=CompilationConfig(
            encoder_cudagraph_token_budgets=budgets or [],
            encoder_cudagraph_max_vision_items_per_batch=max_vision_items,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
    )


class _DummyModel:
    """Minimal SupportsEncoderCudaGraph implementation for unit tests.

    Budget range (128, 512) produces token_budgets=[128, 256, 512] and
    max_batch_size=4 (= 512 // 128) via the parent's auto-infer path.
    """

    supports_encoder_cudagraph = True

    def get_encoder_cudagraph_config(self):
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixel_values"],
            out_hidden_size=64,
        )

    def get_encoder_cudagraph_budget_range(self, vllm_config):
        return (128, 512)

    def get_max_frames_per_video(self):
        return 1

    def prepare_encoder_cudagraph_capture_inputs(self, token_budget,
                                                 max_batch_size,
                                                 max_frames_per_batch, device,
                                                 dtype):
        return EncoderCudaGraphCaptureInputs(
            values={"pixel_values": torch.zeros(token_budget, 4, dtype=dtype)})

    def get_encoder_cudagraph_item_specs(self, mm_kwargs):
        return []

    def select_encoder_cudagraph_items(self, mm_kwargs, indices):
        return mm_kwargs

    def prepare_encoder_cudagraph_replay_buffers(self, mm_kwargs,
                                                 max_batch_size,
                                                 max_frames_per_batch):
        return EncoderCudaGraphReplayBuffers(values={})

    def postprocess_encoder_output(self,
                                   output,
                                   indices,
                                   per_item_out_tokens,
                                   dest,
                                   clone=False,
                                   batch_mm_kwargs=None):
        pass

    def encoder_eager_forward(self, mm_kwargs):
        pass

    def encoder_cudagraph_forward(self, inputs):
        pass

    def get_input_modality(self, mm_kwargs):
        return "image"


def _make_manager(budgets=None, max_vision_items=0):
    return MMEncoderJITManager(
        vllm_config=_make_vllm_config(budgets, max_vision_items),
        vllm_runner=MagicMock(spec=torch.nn.Module),
        vllm_model=_DummyModel(),
        params_and_buffers={},
    )


class TestAdapterPostprocessEncoderOutput:
    """postprocess_encoder_output slices a flat jax.Array into per-item outputs.

    Unlike the GPU base class (which calls scatter_output_slices on a torch
    tensor), the adapter receives a jax.Array from the JIT forward and must
    scatter it correctly — no clone/copy needed since jax arrays are immutable.
    """

    def _adapter(self):
        return _TorchaxEncoderModelAdapter(MagicMock(), MagicMock(), {})

    def test_single_item_full_slice(self):
        output = jnp.arange(10 * 4).reshape(10, 4).astype(jnp.bfloat16)
        dest = {}
        self._adapter().postprocess_encoder_output(output, [0], [10], dest)
        assert 0 in dest
        np.testing.assert_array_equal(np.asarray(dest[0]), np.asarray(output))

    def test_two_items_different_token_counts(self):
        # 3 tokens for item 0, 5 tokens for item 1 — packed contiguously
        output = jnp.arange(8 * 4).reshape(8, 4).astype(jnp.bfloat16)
        dest = {}
        self._adapter().postprocess_encoder_output(output, [0, 1], [3, 5],
                                                   dest)
        np.testing.assert_array_equal(np.asarray(dest[0]),
                                      np.asarray(output[:3]))
        np.testing.assert_array_equal(np.asarray(dest[1]),
                                      np.asarray(output[3:8]))

    def test_non_contiguous_original_indices(self):
        """Greedy packing may reorder items; scatter must use original indices.

        per_item_out_tokens is the FULL array across all items (indexed by
        original item index), matching how _execute_local calls postprocess.
        """
        output = jnp.arange(6 * 2).reshape(6, 2).astype(jnp.bfloat16)
        dest = {}
        # Items at original indices 2 and 5 in a batch of 6; all others dummy.
        per_item_out_tokens = [0, 0, 3, 0, 0, 3]
        self._adapter().postprocess_encoder_output(output, [2, 5],
                                                   per_item_out_tokens, dest)
        assert set(dest.keys()) == {2, 5}
        np.testing.assert_array_equal(np.asarray(dest[2]),
                                      np.asarray(output[:3]))
        np.testing.assert_array_equal(np.asarray(dest[5]),
                                      np.asarray(output[3:]))

    def test_delegates_unknown_attr_to_model(self):
        mock_model = MagicMock()
        mock_model.get_encoder_cudagraph_config.return_value = "sentinel"
        adapter = _TorchaxEncoderModelAdapter(mock_model, MagicMock(), {})
        assert adapter.get_encoder_cudagraph_config() == "sentinel"
        mock_model.get_encoder_cudagraph_config.assert_called_once()


class TestPadToTemplate:
    """Test MMEncoderJITManager._pad_to_template — zero-padding, slicing, and dtype-casting.

    Pure torch logic — tested without a full manager init by constructing a
    bare instance via object.__new__ and setting only budget_templates.
    """

    def _manager(self, template: dict) -> MMEncoderJITManager:
        m = object.__new__(MMEncoderJITManager)
        m.budget_templates = {256: template}
        return m

    def test_general_case_zeros_then_copies(self):
        """src smaller than template: zero the buffer and slice-copy src."""
        tmpl = torch.zeros(10, 4)
        src = torch.ones(4, 4)
        result = self._manager({"pv": tmpl})._pad_to_template({"pv": src}, 256)
        assert result["pv"].shape == (10, 4)
        assert torch.all(result["pv"][:4] == 1.0)
        assert torch.all(result["pv"][4:] == 0.0)

    def test_already_template_shaped_passthrough(self):
        """src already has the template shape: returned as-is (no allocation)."""
        tmpl = torch.zeros(8, 4)
        src = torch.ones(8, 4)
        result = self._manager({"pv": tmpl})._pad_to_template({"pv": src}, 256)
        assert result["pv"] is src

    def test_scalar_uses_template_value(self):
        """0-dim scalars (e.g. max_seqlen): always use the budget-fixed template."""
        tmpl = torch.tensor(256)
        src = torch.tensor(64)
        result = self._manager({
            "max_seqlen": tmpl
        })._pad_to_template({"max_seqlen": src}, 256)
        assert result["max_seqlen"] is tmpl

    def test_none_src_uses_template(self):
        """Key absent from replay_values (or explicitly None): use template."""
        tmpl = torch.zeros(8, 4)
        result = self._manager({"optional": tmpl})._pad_to_template({}, 256)
        assert result["optional"] is tmpl

    def test_dtype_cast_on_copy(self):
        """src dtype differs from template: cast to template dtype during copy."""
        tmpl = torch.zeros(10, 4, dtype=torch.bfloat16)
        src = torch.ones(4, 4, dtype=torch.float32)
        result = self._manager({"pv": tmpl})._pad_to_template({"pv": src}, 256)
        assert result["pv"].dtype == torch.bfloat16
        assert torch.all(result["pv"][:4] == 1.0)


class TestMMEncoderJITManagerInit:
    """Test MMEncoderJITManager.__init__ — budget derivation + template construction."""

    def test_auto_inferred_budgets(self):
        """Fully auto-inferred path: (128, 512) → [128, 256, 512], max_batch=4."""
        manager = _make_manager()
        assert manager.token_budgets == [128, 256, 512]
        # max_batch_size = min(max_budget // min_budget, min(budgets))
        #                = min(512 // 128, 128) = min(4, 128) = 4
        assert manager.max_batch_size == 4
        assert manager.max_frames_per_batch == 0  # video limit is 0

    def test_explicit_budgets_respected(self):
        """User-specified budgets and max_vision_items bypass auto-infer."""
        manager = _make_manager(budgets=[256, 512], max_vision_items=2)
        assert manager.token_budgets == [256, 512]
        assert manager.max_batch_size == 2

    def test_budget_templates_keyed_by_budget(self):
        """One template entry per budget, shape matches the budget size."""
        manager = _make_manager()
        assert set(manager.budget_templates.keys()) == set(
            manager.token_budgets)
        for budget, tmpl in manager.budget_templates.items():
            assert "pixel_values" in tmpl
            assert tmpl["pixel_values"].shape[0] == budget

    def test_jit_forward_is_callable(self):
        """_jit_forward is a callable built once at init (not per-call)."""
        manager = _make_manager()
        assert callable(manager._jit_forward)


def _image_mm_kwargs(t=1, h=4, w=4):
    """Build minimal image mm_kwargs for a t×h×w grid.

    With the defaults (1×4×4) output_tokens = 1*(4/2)*(4/2) = 4, which fits
    comfortably inside the smallest budget (64) produced by the model's
    auto-infer range.
    """
    # Derived from Qwen3.5-0.8B vision_config:
    #   patch_size=16, temporal_patch_size=2, in_channels=3,
    _PATCH_FEAT = 2 * 16 * 16 * 3  # temporal_patch_size * patch_size^2 * in_channels
    return {
        "pixel_values": torch.randn(t * h * w,
                                    _PATCH_FEAT,
                                    dtype=torch.bfloat16),
        "image_grid_thw": [(t, h, w)],
    }


@pytest.fixture(scope="module")
def qwen35_mm_encoder():
    engine_args = EngineArgs(
        model="Qwen/Qwen3.5-0.8B",
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        dtype="bfloat16",
        load_format="dummy",
        limit_mm_per_prompt={"image": 1},
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.device_config.device = "cpu"
    vllm_config.compilation_config.cudagraph_mm_encoder = True

    # sharded_flash_attention uses batch_axis="data" and head_axis="model";
    # both axes must be present. With one device both are size-1 (replicated).
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape((1, 1)), ("data", "model"))
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config, mesh)

    init_pp_distributed_environment(ip="",
                                    rank=0,
                                    world_size=1,
                                    device=jax.devices()[0],
                                    need_pp=False)

    with (cpu_mesh_context(), mock_patch("torch._sync", return_value=None),
          set_current_vllm_config(vllm_config)):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        vllm_model = vllm_get_model(vllm_config=vllm_config)

    vllm_runner = _VllmRunner(vllm_model)
    params_jax = jax_view(shard_model_to_tpu(vllm_runner, mesh))
    manager = MMEncoderJITManager(
        vllm_config=vllm_config,
        vllm_runner=vllm_runner,
        vllm_model=vllm_model,
        params_and_buffers=params_jax,
    )
    yield manager, mesh


class TestMMEncoderJITManagerIntegration:
    """Integration tests using a real Qwen3.5-0.8B model on TPU.

    The qwen35_mm_encoder fixture loads the model once per module with random
    weights (load_format=dummy), shards it onto the first TPU device, and
    builds a live MMEncoderJITManager. Tests cover the three device-bound paths
    that unit tests cannot reach with mocks alone.
    """

    def test_prepare_padded_torch_real_model(self, qwen35_mm_encoder):
        """_prepare_padded_torch produces correctly-shaped outputs for every
        template key when driven by the real model's replay-buffer method."""
        manager, _ = qwen35_mm_encoder
        mm_kwargs = _image_mm_kwargs()
        smallest_budget = manager.token_budgets[0]
        padded = manager._prepare_padded_torch(mm_kwargs, smallest_budget)

        template = manager.budget_templates[smallest_budget]
        assert set(padded.keys()) == set(template.keys())
        for key, tmpl in template.items():
            if hasattr(tmpl, "shape"):
                assert padded[key].shape == tmpl.shape, (
                    f"key={key}: padded {padded[key].shape} != tmpl {tmpl.shape}"
                )
        n_patches = _image_mm_kwargs()["pixel_values"].shape[0]
        assert padded["pixel_values"].shape[0] >= n_patches

    def test_capture_budget_graph(self, qwen35_mm_encoder):
        """_capture_budget_graph primes the XLA cache for the smallest budget
        and records it in budget_graphs so capture()/get_cumulative_stats work."""
        manager, mesh = qwen35_mm_encoder
        smallest_budget = manager.token_budgets[0]
        manager.budget_graphs.pop(smallest_budget, None)

        # shard_map inside the vision attention kernel reads get_abstract_mesh(),
        # which requires an active jax.set_mesh context.
        with jax.set_mesh(mesh):
            manager._capture_budget_graph(smallest_budget)

        assert smallest_budget in manager.budget_graphs

    def test_execute_within_budget(self, qwen35_mm_encoder):
        """execute() runs the full encoder path for a single within-budget image
        and returns a per-item list of jax.Array with the expected shape."""
        # Derived from Qwen3.5-0.8B vision_config:
        #   spatial_merge_size=2, out_hidden_size=1024
        _SPATIAL_MERGE = 2
        _OUT_HIDDEN = 1024
        manager, mesh = qwen35_mm_encoder
        mm_kwargs = _image_mm_kwargs(t=1, h=4, w=4)
        # 1×4×4 grid → output_tokens = 1*(4/2)*(4/2) = 4
        expected_tokens = 1 * (4 // _SPATIAL_MERGE) * (4 // _SPATIAL_MERGE)

        hits_before = manager.graph_hits
        with jax.set_mesh(mesh):
            result = manager.execute(mm_kwargs)
        jax.block_until_ready(result)

        assert len(result) == 1, "one image → one output entry"
        assert isinstance(result[0], jax.Array)
        assert result[0].shape == (expected_tokens, _OUT_HIDDEN), (
            f"expected ({expected_tokens}, {_OUT_HIDDEN}), got {result[0].shape}"
        )
        assert manager.graph_hits > hits_before, (
            "graph_hits must increment for a within-budget image")
