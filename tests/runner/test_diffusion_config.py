# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

try:
    from tpu_inference.runner.diffusion.config import (
        AttentionPolicy, CanvasPolicy, DiffusionAlgorithm, DiffusionConfig,
        DiffusionModelSpec, DiffusionRuntimeConfig, GenerationStrategy,
        LogitAlignment, NextBlockPolicy, PromptRemainderPolicy,
        register_diffusion_model_adapter, resolve_generation_strategy)
except ModuleNotFoundError:
    config_path = (pathlib.Path(__file__).resolve().parents[2] /
                   "tpu_inference" / "runner" / "diffusion" / "config.py")
    spec = importlib.util.spec_from_file_location(
        "diffusion_config_under_test", config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    AttentionPolicy = module.AttentionPolicy
    CanvasPolicy = module.CanvasPolicy
    DiffusionAlgorithm = module.DiffusionAlgorithm
    DiffusionConfig = module.DiffusionConfig
    DiffusionModelSpec = module.DiffusionModelSpec
    DiffusionRuntimeConfig = module.DiffusionRuntimeConfig
    GenerationStrategy = module.GenerationStrategy
    LogitAlignment = module.LogitAlignment
    NextBlockPolicy = module.NextBlockPolicy
    PromptRemainderPolicy = module.PromptRemainderPolicy
    register_diffusion_model_adapter = module.register_diffusion_model_adapter
    resolve_generation_strategy = module.resolve_generation_strategy


def _vllm_config(additional_config=None, **hf_values):
    return SimpleNamespace(
        additional_config=additional_config or {},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_values)),
    )


def test_autoregressive_is_the_default_and_has_no_diffusion_config():
    resolved = resolve_generation_strategy(_vllm_config())

    assert resolved.strategy is GenerationStrategy.AUTOREGRESSIVE
    assert resolved.diffusion is None


def test_strategy_enum_is_accepted_without_string_coercion():
    resolved = resolve_generation_strategy(
        _vllm_config(
            {"generation_strategy": GenerationStrategy.AUTOREGRESSIVE}))

    assert resolved.strategy is GenerationStrategy.AUTOREGRESSIVE


def test_resolves_seeded_shifted_semantics_from_hf_config():
    resolved = resolve_generation_strategy(
        _vllm_config(
            {
                "generation_strategy": "block_diffusion",
                "diffusion": {
                    "model_adapter": "seeded_shifted"
                },
            },
            bd_size=32,
            mask_id=151669,
        ))

    assert resolved.strategy is GenerationStrategy.BLOCK_DIFFUSION
    assert resolved.diffusion is not None
    assert resolved.diffusion.model == DiffusionModelSpec(
        name="seeded_shifted",
        block_size=32,
        mask_token_id=151669,
        attention_policy=AttentionPolicy.BLOCK_CAUSAL,
        logit_alignment=LogitAlignment.SHIFTED,
        canvas_policy=CanvasPolicy.SEED_AND_MASK,
        prompt_remainder_policy=(
            PromptRemainderPolicy.INCLUDE_IN_FIRST_CANVAS),
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        sub_block_size=8,
        supported_algorithms=(DiffusionAlgorithm.LOW_CONFIDENCE, ),
    )
    assert resolved.diffusion.runtime.cohort_max_wait_ms == 0.0
    assert resolved.diffusion.runtime.cohort_quiet_wait_ms == 0.0


def test_runtime_and_model_overrides_are_separate():
    resolved = resolve_generation_strategy(
        _vllm_config({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "block_size": 16,
                "mask_token_id": 99,
                "sub_block_size": 4,
                "confidence_threshold": 0.75,
                "temperature": 0.2,
                "max_denoise_steps": 12,
                "use_dual_cache": True,
                "cohort_max_wait_ms": 7.5,
                "cohort_quiet_wait_ms": 1.5,
            },
        }))

    assert resolved.diffusion is not None
    assert resolved.diffusion.model.block_size == 16
    assert resolved.diffusion.model.sub_block_size == 4
    assert resolved.diffusion.runtime.confidence_threshold == 0.75
    assert resolved.diffusion.runtime.temperature == 0.2
    assert resolved.diffusion.runtime.max_denoise_steps == 12
    assert resolved.diffusion.runtime.use_dual_cache is True
    assert resolved.diffusion.runtime.trace_acceptance_steps is False
    assert resolved.diffusion.runtime.fp32_partial_rpa is False
    assert resolved.diffusion.runtime.q8_log_confidence_bias == 0.0
    assert resolved.diffusion.runtime.force_q32_anchor_commit is False
    assert resolved.diffusion.runtime.cohort_max_wait_ms == 7.5
    assert resolved.diffusion.runtime.cohort_quiet_wait_ms == 1.5


def test_resolves_opt_in_dual_cache_acceptance_trace():
    resolved = resolve_generation_strategy(
        _vllm_config({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "block_size": 16,
                "mask_token_id": 99,
                "sub_block_size": 4,
                "use_dual_cache": True,
                "trace_acceptance_steps": True,
            },
        }))

    assert resolved.diffusion is not None
    assert resolved.diffusion.runtime.trace_acceptance_steps is True


def test_resolves_opt_in_fp32_partial_rpa():
    resolved = resolve_generation_strategy(
        _vllm_config({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "block_size": 16,
                "mask_token_id": 99,
                "sub_block_size": 4,
                "use_dual_cache": True,
                "fp32_partial_rpa": True,
            },
        }))

    assert resolved.diffusion is not None
    assert resolved.diffusion.runtime.fp32_partial_rpa is True


def test_resolves_opt_in_q8_log_confidence_bias():
    resolved = resolve_generation_strategy(
        _vllm_config({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "block_size": 16,
                "mask_token_id": 99,
                "sub_block_size": 4,
                "confidence_threshold": 0.9,
                "use_dual_cache": True,
                "q8_log_confidence_bias": 0.05,
            },
        }))

    assert resolved.diffusion is not None
    assert resolved.diffusion.runtime.q8_log_confidence_bias == 0.05


def test_resolves_opt_in_forced_q32_anchor_commit():
    resolved = resolve_generation_strategy(
        _vllm_config({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "block_size": 16,
                "mask_token_id": 99,
                "sub_block_size": 4,
                "use_dual_cache": True,
                "force_q32_anchor_commit": True,
            },
        }))

    assert resolved.diffusion is not None
    assert resolved.diffusion.runtime.force_q32_anchor_commit is True


def test_new_model_adapter_does_not_change_strategy_resolution():

    def adapter(_hf_config, _values):
        return DiffusionModelSpec(
            name="test-model",
            block_size=4,
            mask_token_id=7,
            attention_policy=AttentionPolicy.BLOCK_CAUSAL,
            logit_alignment=LogitAlignment.SAME_POSITION,
            canvas_policy=CanvasPolicy.ALL_MASKED,
            prompt_remainder_policy=(
                PromptRemainderPolicy.REQUIRE_BLOCK_ALIGNED),
            next_block_policy=NextBlockPolicy.ALL_MASKED,
            sub_block_size=4,
            supported_algorithms=(DiffusionAlgorithm.LOW_CONFIDENCE, ),
        )

    register_diffusion_model_adapter("test-model", adapter)
    resolved = resolve_generation_strategy(
        _vllm_config({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "test-model"
            },
        }))

    assert resolved.diffusion is not None
    assert resolved.diffusion.model.name == "test-model"
    assert resolved.diffusion.model.logit_alignment is LogitAlignment.SAME_POSITION


@pytest.mark.parametrize(
    "additional_config,match",
    [
        ({
            "generation_strategy": "unknown"
        }, "Unsupported generation_strategy"),
        ({
            "generation_strategy": "block_diffusion"
        }, "model_adapter"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "missing"
            },
        }, "Unknown diffusion model_adapter"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "confidence_threshold": 2.0,
            },
        }, "confidence_threshold"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "trace_acceptance_steps": True,
            },
        }, "requires use_dual_cache"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "fp32_partial_rpa": True,
            },
        }, "fp32_partial_rpa requires use_dual_cache"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "q8_log_confidence_bias": 0.05,
            },
        }, "q8_log_confidence_bias requires use_dual_cache"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "force_q32_anchor_commit": True,
            },
        }, "force_q32_anchor_commit requires use_dual_cache"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "use_dual_cache": True,
                "q8_log_confidence_bias": 0.51,
            },
        }, "must not exceed 0.5"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "use_dual_cache": True,
                "q8_log_confidence_bias": -0.01,
            },
        }, "finite and non-negative"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "use_dual_cache": True,
                "q8_log_confidence_bias": float("inf"),
            },
        }, "finite and non-negative"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "cohort_max_wait_ms": -0.01,
            },
        }, "cohort_max_wait_ms must be finite and non-negative"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "cohort_max_wait_ms": float("inf"),
            },
        }, "cohort_max_wait_ms must be finite and non-negative"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "cohort_max_wait_ms": 100.01,
            },
        }, "cohort_max_wait_ms must not exceed 100"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "cohort_max_wait_ms": 20.0,
                "cohort_quiet_wait_ms": -0.01,
            },
        }, "cohort_quiet_wait_ms must be finite and non-negative"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "cohort_quiet_wait_ms": 2.0,
            },
        }, "cohort_quiet_wait_ms requires positive cohort_max_wait_ms"),
        ({
            "generation_strategy": "block_diffusion",
            "diffusion": {
                "model_adapter": "seeded_shifted",
                "cohort_max_wait_ms": 2.0,
                "cohort_quiet_wait_ms": 2.01,
            },
        }, "cohort_quiet_wait_ms must not exceed cohort_max_wait_ms"),
    ],
)
def test_invalid_configuration_fails_at_resolution(additional_config, match):
    config = _vllm_config(additional_config, bd_size=32, mask_id=151669)

    with pytest.raises(ValueError, match=match):
        resolve_generation_strategy(config)


def test_model_spec_rejects_incompatible_sub_block_size():
    with pytest.raises(ValueError, match="divide block_size"):
        DiffusionModelSpec(
            name="invalid",
            block_size=10,
            mask_token_id=1,
            attention_policy=AttentionPolicy.BLOCK_CAUSAL,
            logit_alignment=LogitAlignment.SAME_POSITION,
            canvas_policy=CanvasPolicy.ALL_MASKED,
            prompt_remainder_policy=(
                PromptRemainderPolicy.REQUIRE_BLOCK_ALIGNED),
            next_block_policy=NextBlockPolicy.ALL_MASKED,
            sub_block_size=4,
            supported_algorithms=(DiffusionAlgorithm.LOW_CONFIDENCE, ),
        )


def test_forced_q32_anchor_commit_rejects_same_position_alignment():
    model = DiffusionModelSpec(
        name="same-position",
        block_size=4,
        mask_token_id=7,
        attention_policy=AttentionPolicy.BLOCK_CAUSAL,
        logit_alignment=LogitAlignment.SAME_POSITION,
        canvas_policy=CanvasPolicy.ALL_MASKED,
        prompt_remainder_policy=(PromptRemainderPolicy.REQUIRE_BLOCK_ALIGNED),
        next_block_policy=NextBlockPolicy.ALL_MASKED,
        sub_block_size=4,
        supported_algorithms=(DiffusionAlgorithm.LOW_CONFIDENCE, ),
    )
    runtime = DiffusionRuntimeConfig(
        use_dual_cache=True,
        force_q32_anchor_commit=True,
    )

    with pytest.raises(ValueError, match="requires shifted logit alignment"):
        DiffusionConfig(model=model, runtime=runtime)


def test_model_spec_rejects_shifted_all_masked_canvas():
    with pytest.raises(ValueError, match="requires seed_and_mask"):
        DiffusionModelSpec(
            name="invalid",
            block_size=4,
            mask_token_id=7,
            attention_policy=AttentionPolicy.BLOCK_CAUSAL,
            logit_alignment=LogitAlignment.SHIFTED,
            canvas_policy=CanvasPolicy.ALL_MASKED,
            prompt_remainder_policy=(
                PromptRemainderPolicy.REQUIRE_BLOCK_ALIGNED),
            next_block_policy=NextBlockPolicy.ALL_MASKED,
            sub_block_size=4,
            supported_algorithms=(DiffusionAlgorithm.LOW_CONFIDENCE, ),
        )
