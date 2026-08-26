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
                                             init_distributed_environment,
                                             model_parallel_is_initialized)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.model_executor.models.qwen2_5_vl import \
    _pad_cumulative_seqlens_buffer
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
    model_config = ModelConfig(dtype=torch.bfloat16)
    model_config.multimodal_config = MultiModalConfig(
        limit_per_prompt={"video": 0})
    return VllmConfig(
        model_config=model_config,
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
    """Test MMEncoderJITManager._pad_to_template — per-key padding logic,
    slicing, and dtype-casting.

    Pure torch logic — tested without a full manager init by constructing a
    bare instance via object.__new__ and setting the few attributes the method
    reads (budget_templates, config, max_batch_size).
    """

    def _manager(self,
                 template: dict,
                 padding_logics: dict | None = None) -> MMEncoderJITManager:
        m = object.__new__(MMEncoderJITManager)
        m.budget_templates = {256: template}
        m.max_batch_size = 4
        m.config = EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=list(template),
            out_hidden_size=64,
            padding_logics=padding_logics or {},
        )
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

    def test_padding_logic_from_config_is_used(self):
        """A key with a registered padding logic must use it instead of the
        zero-fill default.

        Qwen2.5-VL registers `_pad_cumulative_seqlens_buffer` for cu_seqlens /
        cu_window_seqlens and leaves cu_window_seqlens UNPADDED at replay time.
        Zero-filling the tail makes `lens = cu[1:] - cu[:-1]` go negative and
        the ViT attention builds garbage segment ids, so the tail must repeat
        the last cumulative offset (= empty trailing sequences) instead.
        """
        tmpl = torch.zeros(6, dtype=torch.int32)
        src = torch.tensor([0, 4, 10], dtype=torch.int32)
        manager = self._manager({"cu_window_seqlens": tmpl},
                                padding_logics={
                                    "cu_window_seqlens":
                                    _pad_cumulative_seqlens_buffer
                                })
        result = manager._pad_to_template({"cu_window_seqlens": src}, 256)
        torch.testing.assert_close(
            result["cu_window_seqlens"],
            torch.tensor([0, 4, 10, 10, 10, 10], dtype=torch.int32))

    def test_no_padding_logic_still_zero_pads(self):
        """Keys without a registered logic keep the upstream default: zero the
        buffer, slice-copy src onto its head."""
        tmpl = torch.zeros(6, dtype=torch.int32)
        src = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = self._manager({
            "window_index": tmpl
        })._pad_to_template({"window_index": src}, 256)
        torch.testing.assert_close(
            result["window_index"],
            torch.tensor([1, 2, 3, 0, 0, 0], dtype=torch.int32))

    def test_template_shaped_src_is_cast_to_template_dtype(self):
        """src already template-shaped but a different dtype: cast, otherwise
        the jit signature changes and every request recompiles."""
        tmpl = torch.zeros(4, 2, dtype=torch.bfloat16)
        src = torch.ones(4, 2, dtype=torch.float32)
        result = self._manager({"pv": tmpl})._pad_to_template({"pv": src}, 256)
        assert result["pv"].dtype == torch.bfloat16
        assert result["pv"].shape == (4, 2)
        assert torch.all(result["pv"] == 1.0)

    def test_oversized_src_raises_with_shapes(self):
        """A replay buffer longer than its template is a packing bug — fail
        loudly with the key and both shapes, not a bare copy_ error."""
        tmpl = torch.zeros(4, 2)
        src = torch.ones(9, 2)
        with pytest.raises(ValueError, match="leading-dim overflow"):
            self._manager({"pv": tmpl})._pad_to_template({"pv": src}, 256)

    def test_trailing_dim_mismatch_raises_with_shapes(self):
        """Only the leading dim is padded, so a differing trailing dim is a
        layout bug and must fail loudly rather than slice-copy silently."""
        tmpl = torch.zeros(8, 2)
        src = torch.ones(4, 3)
        with pytest.raises(ValueError, match="trailing-dim mismatch"):
            self._manager({"pv": tmpl})._pad_to_template({"pv": src}, 256)


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
        assert callable(manager.model._jit_forward)


def _image_mm_kwargs(t=1, h=4, w=4):
    """Build minimal image mm_kwargs for a t×h×w grid.

    With the defaults (1×4×4) output_tokens = 1*(4/2)*(4/2) = 4, which fits
    comfortably inside the smallest budget (64) produced by the model's
    auto-infer range.
    """
    # Derived from Qwen3.5-0.8B vision_config:
    #   patch_size=16, temporal_patch_size=2, in_channels=3,
    _PATCH_FEAT = 2 * 16 * 16 * 3  # temporal_patch_size * patch_size^2 * in_channels
    # Seeded on the grid so a failure reproduces run to run.
    gen = torch.Generator().manual_seed(hash((t, h, w)) & 0xFFFFFFFF)
    return {
        "pixel_values":
        torch.randn(t * h * w,
                    _PATCH_FEAT,
                    dtype=torch.bfloat16,
                    generator=gen),
        "image_grid_thw": [(t, h, w)],
    }


def _reinit_params_for_sensitivity(params: dict, seed: int = 0) -> dict:
    """Replace vLLM's dummy weights with a non-degenerate random init.

    ``load_format="dummy"`` fills EVERY parameter — norm scales included — with
    uniform(-1e-3, 1e-3). In that regime each ViT block's residual contribution
    is ~1e-9 of the residual stream, so the encoder output is essentially
    ``patch_embed`` + ``merger`` and is numerically INSENSITIVE to which tokens
    the attention actually sees: a padded-vs-eager comparison then passes even
    when budget-padding rows leak into a real image's attention segment
    (measured on Qwen2.5-VL-3B: max|out| 1.0e-3, and the same 7.5e-6 max
    difference with and without that bug).

    Re-init norm scales to 1, biases to 0 and every other float parameter to
    normal(0, 0.02) so the blocks actually shape the output and the comparison
    has teeth. The same weights feed both the padded JIT path and the eager
    reference, so equivalence still has to hold exactly.
    """
    key = jax.random.key(seed)
    out: dict = {}
    for i, (name, value) in enumerate(sorted(params.items())):
        if not (hasattr(value, "dtype")
                and jnp.issubdtype(value.dtype, jnp.floating)):
            out[name] = value
        elif name.endswith(".bias"):
            out[name] = jnp.zeros_like(value)
        elif "norm" in name or "ln_" in name:
            out[name] = jnp.ones_like(value)
        else:
            out[name] = (jax.random.normal(
                jax.random.fold_in(key, i), value.shape, dtype=jnp.float32) *
                         0.02).astype(value.dtype)
    return out


def _build_mm_encoder_manager(engine_args: EngineArgs,
                              params_transform=None,
                              **compilation_overrides):
    """Load a real model with dummy weights and build a live manager.

    Shared by the per-model integration fixtures. The torch distributed init is
    process-global, so it runs at most once even with several fixtures.

    Args:
      engine_args: Model/scheduler args; ``load_format="dummy"`` keeps this to
        a config download plus random weights.
      params_transform: Optional callable applied to the sharded jax parameter
        dict before the manager is built (see
        ``_reinit_params_for_sensitivity``).
      compilation_overrides: Extra ``compilation_config`` attributes to set
        (e.g. ``encoder_cudagraph_max_vision_items_per_batch``).

    Returns:
      ``(manager, mesh)`` — the mesh must be active (``jax.set_mesh``) around
      any call that reaches the vision attention kernel.
    """
    vllm_config = engine_args.create_engine_config()
    vllm_config.device_config.device = "cpu"
    vllm_config.compilation_config.cudagraph_mm_encoder = True
    for key, value in compilation_overrides.items():
        setattr(vllm_config.compilation_config, key, value)

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

    # TPUPlatform.check_and_update_config attaches `sharding_config` to the
    # VllmConfig as a plain attribute — it is not a declared dataclass field.
    # Models with a nested text config (Qwen2.5-VL) call
    # VllmConfig.with_hf_config -> vllm.config.utils.replace during init, which
    # rebuilds the config from __dict__ and raises "Field 'sharding_config' not
    # found in VllmConfig" on any extra key. Same hasattr/delattr dance as
    # VllmModelWrapper.load_weights in
    # tpu_inference/models/vllm/vllm_model_wrapper.py (which strips it from its
    # load-time config copy); keep the two in sync. Re-attached below because
    # the manager and the sharding helpers run against this same config.
    sharding_config = None
    if hasattr(vllm_config, "sharding_config"):
        sharding_config = vllm_config.sharding_config
        delattr(vllm_config, "sharding_config")

    with (cpu_mesh_context(), mock_patch("torch._sync", return_value=None),
          set_current_vllm_config(vllm_config)):
        if not model_parallel_is_initialized():
            temp_file = tempfile.mkstemp()[1]
            init_distributed_environment(
                1,
                0,
                local_rank=0,
                distributed_init_method=f"file://{temp_file}",
                backend="gloo")
            ensure_model_parallel_initialized(1, 1)
        vllm_model = vllm_get_model(vllm_config=vllm_config)

    if sharding_config is not None:
        vllm_config.sharding_config = sharding_config

    vllm_runner = _VllmRunner(vllm_model)
    params_jax = jax_view(shard_model_to_tpu(vllm_runner, mesh))
    if params_transform is not None:
        params_jax = params_transform(params_jax)
    manager = MMEncoderJITManager(
        vllm_config=vllm_config,
        vllm_runner=vllm_runner,
        vllm_model=vllm_model,
        params_and_buffers=params_jax,
    )
    return manager, mesh


@pytest.fixture(scope="module")
def qwen35_mm_encoder():
    yield _build_mm_encoder_manager(
        EngineArgs(
            model="Qwen/Qwen3.5-0.8B",
            max_model_len=256,
            max_num_batched_tokens=256,
            max_num_seqs=4,
            dtype="bfloat16",
            load_format="dummy",
            limit_mm_per_prompt={"image": 1},
        ))


@pytest.fixture(scope="module")
def qwen25vl_mm_encoder():
    """Qwen2.5-VL on the torchax path — the model the nightly benchmark runs.

    Shrunk to the smallest shape that still covers both padding hazards, so
    the class stays cheap enough for the "JAX unit tests part2" job:

    * ``vision_config.depth=2`` with ``fullatt_block_indexes=[1]`` keeps one
      window-attention block (layer 0, driven by ``cu_window_seqlens`` — the
      buffer whose padding logic matters) and one full-attention block
      (layer 1, driven by ``cu_seqlens`` — the buffer whose trailing budget
      padding must land in its own segment). The stock 32 blocks only repeat
      those two cases. Every other vision hyper-parameter — patch_size 14,
      temporal_patch_size 2, in_channels 3, spatial_merge_size 2, window_size
      112, out_hidden_size 2048 — is untouched, so all the geometry the test
      computes stays the real model's.
    * ``text_config.num_hidden_layers=1`` because the language model is never
      run here; it only inflates dummy-weight init and sharding time.

    max_vision_items_per_batch is pinned to 2 so a batch of exactly
    max_batch_size real items (the case where cu_seqlens has NO trailing empty
    segment, yet pixel_values is still padded up to the budget) is reachable
    without building large images.
    """
    yield _build_mm_encoder_manager(
        EngineArgs(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=4,
            dtype="bfloat16",
            load_format="dummy",
            # video must be 0, otherwise the manager sizes cu_seqlens for
            # frames and captures a video-shaped grid.
            limit_mm_per_prompt={
                "image": 4,
                "video": 0
            },
            # Nested dicts land on the sub-configs via
            # ModelConfig._apply_dict_overrides -> _update_nested, which runs
            # before hf_text_config / model_arch_config are derived.
            hf_overrides={
                "vision_config": {
                    "depth": 2,
                    "fullatt_block_indexes": [1],
                },
                "text_config": {
                    "num_hidden_layers": 1
                },
            },
        ),
        params_transform=_reinit_params_for_sensitivity,
        encoder_cudagraph_max_vision_items_per_batch=2)


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


# Qwen2.5-VL vision_config: in_channels=3, temporal_patch_size=2,
# patch_size=14, spatial_merge_size=2, out_hidden_size=2048. The
# qwen25vl_mm_encoder fixture only overrides depth / fullatt_block_indexes, so
# every value below is still the stock model's.
_QWEN25VL_PATCH_FEAT = 3 * 2 * 14 * 14  # 1176
_QWEN25VL_SPATIAL_MERGE = 2
_QWEN25VL_OUT_HIDDEN = 2048


def _qwen25vl_mm_kwargs(grids):
    """Build image mm_kwargs for a list of (t, h, w) Qwen2.5-VL grids.

    h and w must be multiples of spatial_merge_size (2). pixel_values rows are
    concatenated per item in grid order, matching how the model's
    select_encoder_cudagraph_items slices a batch back apart.
    """
    total_patches = sum(t * h * w for t, h, w in grids)
    # Seeded on the grids so a failure reproduces run to run.
    gen = torch.Generator().manual_seed(
        hash(tuple(tuple(g) for g in grids)) & 0xFFFFFFFF)
    return {
        "pixel_values":
        torch.randn(total_patches,
                    _QWEN25VL_PATCH_FEAT,
                    dtype=torch.bfloat16,
                    generator=gen),
        "image_grid_thw": [tuple(g) for g in grids],
    }


def _qwen25vl_out_tokens(grid) -> int:
    t, h, w = grid
    return t * (h // _QWEN25VL_SPATIAL_MERGE) * (w // _QWEN25VL_SPATIAL_MERGE)


class TestQwen25VLMMEncoderJITIntegration:
    """Correctness of the budget-padded JIT encoder path for Qwen2.5-VL.

    The nightly accuracy job for this model is text-only (gsm8k), so nothing
    else in CI compares the compiled vision path against eager. Two padding
    hazards are covered here:

    * cu_window_seqlens comes back UNPADDED from
      prepare_encoder_cudagraph_replay_buffers, so _pad_to_template must apply
      the model's cumulative-seqlens padding logic (repeat the last offset)
      instead of zero-filling the tail.
    * pixel_values is padded up to the token budget, so the ViT attention must
      keep the pad rows in their own segment. For a batch of exactly
      max_batch_size items cu_seqlens has no trailing empty segment, and a
      "repeat the last id" segment-id construction merges the pad rows into
      the last image.
    """

    # Relative tolerance on max|out|. Measured on the fixture's 2-block
    # Qwen2.5-VL-3B with the sensitivity re-init: the correct padded path lands
    # at 0.0066-0.0071 of max|out| (bf16 + attention over a different padded
    # length); zero-padding cu_window_seqlens instead of repeating its last
    # offset gives 0.3073-0.3186, and filling the trailing segment ids with the
    # last real id gives 0.3194-0.3393. 0.1 sits ~14x above the correct path
    # and ~3x below either bug.
    _REL_TOL = 0.1

    @staticmethod
    def _clear_rope_cache(manager):
        """Drop visual.get_rope_by_thw's lru_cache before a test.

        It is keyed on (t, h, w) only, and the tensors it caches are plain
        torch or torchax depending on which path filled it first, so a
        previous test's entries must not leak into this one.
        """
        manager.model.visual.get_rope_by_thw.cache_clear()

    def _assert_padded_matches_eager(self, manager, mesh, grids):
        self._clear_rope_cache(manager)
        mm_kwargs = _qwen25vl_mm_kwargs(grids)
        with jax.set_mesh(mesh):
            # Run the budget path BEFORE the eager reference. Both go through
            # visual.get_rope_by_thw, which is @lru_cache'd on (t, h, w); the
            # eager path runs inside the torchax env and seeds that cache with
            # torchax tensors, which then blow up in the replay-buffer prep
            # (plain torch, outside the env). This ordering only matters here:
            # _execute_local routes an item to the budget path or to eager
            # purely by its token count, i.e. by its (t, h, w), so in
            # production a geometry that goes eager always goes eager and its
            # cached tensors never reach the replay-buffer prep. This test is
            # the only place both paths see the same geometry.
            padded = manager.execute(mm_kwargs)
            eager = [
                manager.model.encoder_eager_forward(
                    manager.model.select_encoder_cudagraph_items(
                        mm_kwargs, [i])) for i in range(len(grids))
            ]
        jax.block_until_ready((padded, eager))

        assert len(padded) == len(grids)
        for i, grid in enumerate(grids):
            got = np.asarray(padded[i], dtype=np.float32)
            want = np.asarray(eager[i], dtype=np.float32)
            assert got.shape == want.shape, (
                f"item {i} grid={grid}: padded {got.shape} != eager "
                f"{want.shape}")
            scale = float(np.abs(want).max())
            assert scale > 1e-3, (
                f"item {i} grid={grid}: eager output is all ~0 "
                f"(max|out|={scale:.2e}); the comparison below cannot "
                f"discriminate — the model weights are degenerate")
            max_diff = float(np.abs(got - want).max())
            assert max_diff <= self._REL_TOL * scale, (
                f"item {i} grid={grid}: budget-padded encoder output differs "
                f"from eager by {max_diff:.4f} "
                f"({max_diff / scale:.4f} of max|out|={scale:.4f}, allowed "
                f"{self._REL_TOL}); a metadata buffer was padded with zeros "
                f"instead of its registered padding logic, or pad rows leaked "
                f"into a real image's attention segment")

    def test_execute_single_image(self, qwen25vl_mm_encoder):
        """One within-budget image: right output shape and a graph hit."""
        manager, mesh = qwen25vl_mm_encoder
        self._clear_rope_cache(manager)
        grid = (1, 4, 4)
        mm_kwargs = _qwen25vl_mm_kwargs([grid])

        hits_before = manager.graph_hits
        with jax.set_mesh(mesh):
            result = manager.execute(mm_kwargs)
        jax.block_until_ready(result)

        assert len(result) == 1
        assert isinstance(result[0], jax.Array)
        assert result[0].shape == (_qwen25vl_out_tokens(grid),
                                   _QWEN25VL_OUT_HIDDEN)
        assert manager.graph_hits == hits_before + 1

    def test_padded_matches_eager_mixed_grids(self, qwen25vl_mm_encoder):
        """Multi-image batch with different grids: every item must match the
        eager single-item forward."""
        manager, mesh = qwen25vl_mm_encoder
        self._assert_padded_matches_eager(manager, mesh, [(1, 4, 4), (1, 6, 8),
                                                          (1, 8, 8)])

    def test_padded_matches_eager_full_batch(self, qwen25vl_mm_encoder):
        """Exactly max_batch_size items, all below the budget: cu_seqlens has
        no trailing empty segment, so the budget pad rows have nothing to hide
        behind."""
        manager, mesh = qwen25vl_mm_encoder
        assert manager.max_batch_size == 2, (
            "fixture pins max_vision_items_per_batch=2 so this batch is full")
        self._assert_padded_matches_eager(manager, mesh, [(1, 4, 4),
                                                          (1, 6, 6)])
