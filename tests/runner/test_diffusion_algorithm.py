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
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _load_pure_diffusion_modules():
    root = pathlib.Path(__file__).resolve().parents[2]
    module_paths = {
        "tpu_inference.runner.diffusion.config":
        root / "tpu_inference" / "runner" / "diffusion" / "config.py",
        "tpu_inference.runner.diffusion.algorithm":
        root / "tpu_inference" / "runner" / "diffusion" / "algorithm.py",
        "tpu_inference.runner.diffusion.program":
        root / "tpu_inference" / "runner" / "diffusion" / "program.py",
    }
    for package in ("tpu_inference", "tpu_inference.runner",
                    "tpu_inference.runner.diffusion"):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module
    loaded = {}
    for name, path in module_paths.items():
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


_MODULES = _load_pure_diffusion_modules()
_CONFIG = _MODULES["tpu_inference.runner.diffusion.config"]
_ALGORITHM = _MODULES["tpu_inference.runner.diffusion.algorithm"]
_PROGRAM = _MODULES["tpu_inference.runner.diffusion.program"]

LogitAlignment = _CONFIG.LogitAlignment
NextBlockPolicy = _CONFIG.NextBlockPolicy
low_confidence_commit = _ALGORITHM.low_confidence_commit
low_confidence_diagnostics = _ALGORITHM.low_confidence_diagnostics
confidence_threshold_with_log_bias = (
    _ALGORITHM.confidence_threshold_with_log_bias)
denoise_block = _PROGRAM.denoise_block
denoise_block_dual_cache = _PROGRAM.denoise_block_dual_cache
select_aligned_hidden_states = _PROGRAM.select_aligned_hidden_states


def _thresholds(batch_size, value=0.9):
    return jnp.full((batch_size, ), value, dtype=jnp.float32)


def _temperatures(batch_size):
    return jnp.zeros((batch_size, ), dtype=jnp.float32)


def _official_gpu_reference_commit(logits, eligible_mask, active_rows,
                                   confidence_threshold):
    probabilities = jax.nn.softmax(logits, axis=-1)
    token_ids = jnp.argmax(probabilities, axis=-1).astype(jnp.int32)
    selected_probabilities = jnp.take_along_axis(probabilities,
                                                 token_ids[..., None],
                                                 axis=-1)[..., 0]
    thresholds = jnp.asarray(confidence_threshold, dtype=probabilities.dtype)
    eligible = jnp.asarray(eligible_mask, dtype=bool) & active_rows[:, None]
    commit = eligible & (selected_probabilities > thresholds[:, None])

    masked_probabilities = jnp.where(eligible, selected_probabilities,
                                     -jnp.inf)
    forced_indices = jnp.argmax(masked_probabilities, axis=-1)
    forced = jax.nn.one_hot(forced_indices, logits.shape[1], dtype=bool)
    forced &= jnp.any(eligible, axis=-1)[:, None]
    commit |= forced
    return token_ids, eligible & ~commit


def test_log_confidence_bias_preserves_zero_and_lowers_threshold():
    thresholds = _thresholds(2, value=0.9)

    unchanged = confidence_threshold_with_log_bias(thresholds, 0.0)
    biased = confidence_threshold_with_log_bias(thresholds, 0.05)

    assert unchanged is thresholds
    np.testing.assert_array_equal(unchanged, thresholds)
    np.testing.assert_allclose(
        biased,
        np.asarray(thresholds) * np.exp(-0.05),
        rtol=1e-6,
    )
    boundary_thresholds = jnp.array([0.0, 1.0], dtype=jnp.float32)
    np.testing.assert_array_equal(
        confidence_threshold_with_log_bias(boundary_thresholds, 0.5),
        boundary_thresholds,
    )


def test_low_confidence_commit_threshold_and_forced_progress():
    logits = jnp.array([
        [[10.0, 0.0, -10.0], [0.1, 0.0, -10.0], [0.0, 0.1, -10.0]],
        [[0.1, 0.0, -10.0], [0.0, 0.1, -10.0], [10.0, 0.0, -10.0]],
    ])
    eligible = jnp.ones((2, 3), dtype=bool)

    tokens, remaining = low_confidence_commit(
        logits,
        eligible,
        jnp.array([True, True]),
        _thresholds(2),
        _temperatures(2),
        2,
    )

    np.testing.assert_array_equal(tokens, [[0, 0, 1], [0, 1, 0]])
    assert remaining[0].sum() == 2
    assert remaining[1].sum() == 2


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16],
                         ids=["float32", "bfloat16"])
def test_low_confidence_commit_matches_official_gpu_reference(dtype):
    logits = jnp.array([
        [[5.0, 0.0, -4.0, -20.0], [2.5, 0.0, -4.0, -20.0],
         [0.25, 0.0, -4.0, -20.0]],
        [[0.2, 0.0, -4.0, -20.0], [0.4, 0.0, -4.0, -20.0],
         [0.3, 0.0, -4.0, -20.0]],
    ],
                       dtype=dtype)
    eligible = jnp.ones((2, 3), dtype=bool)
    active_rows = jnp.array([True, True])
    thresholds = _thresholds(2)

    actual_tokens, actual_remaining = low_confidence_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
        _temperatures(2),
        3,
    )
    expected_tokens, expected_remaining = _official_gpu_reference_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
    )

    np.testing.assert_array_equal(actual_tokens, expected_tokens)
    np.testing.assert_array_equal(actual_remaining, expected_remaining)


def test_float32_probability_boundary_documents_log_space_rounding_contract():
    threshold = jnp.asarray(0.9, dtype=jnp.float32)
    boundary_gap = jnp.log(jnp.asarray(9.0, dtype=jnp.float32))
    position_logits = jnp.array([
        [8.0, 0.0, -jnp.inf],
        [boundary_gap, 0.0, -jnp.inf],
        [0.0, 0.0, -jnp.inf],
    ],
                                dtype=jnp.float32)
    boundary_probability = jax.nn.softmax(position_logits[1])[0]
    np.testing.assert_array_equal(boundary_probability, threshold)
    thresholds = jnp.array([
        jnp.nextafter(boundary_probability, -jnp.inf),
        boundary_probability,
        jnp.nextafter(boundary_probability, jnp.inf),
    ])
    logits = jnp.broadcast_to(position_logits, (3, 3, 3))
    eligible = jnp.ones((3, 3), dtype=bool)
    active_rows = jnp.ones((3, ), dtype=bool)

    _, official_remaining = _official_gpu_reference_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
    )
    _, current_remaining = low_confidence_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
        _temperatures(3),
        2,
    )

    np.testing.assert_array_equal(
        official_remaining,
        [[False, False, True], [False, True, True], [False, True, True]],
    )
    # At the exactly-equal boundary, the current FP32 log-space comparison
    # rounds differently from the official probability-space comparison.
    np.testing.assert_array_equal(
        current_remaining,
        [[False, False, True], [False, False, True], [False, True, True]],
    )


def test_bfloat16_probability_boundary_matches_official_gpu_reference():
    boundary_gap = jnp.log(jnp.asarray(9.0, dtype=jnp.float32))
    position_logits = jnp.array([
        [8.0, 0.0, -jnp.inf],
        [boundary_gap, 0.0, -jnp.inf],
        [0.0, 0.0, -jnp.inf],
    ],
                                dtype=jnp.bfloat16)
    boundary_probability = jax.nn.softmax(position_logits[1])[0]
    thresholds = jnp.array([
        jnp.nextafter(boundary_probability, -jnp.inf),
        boundary_probability,
        jnp.nextafter(boundary_probability, jnp.inf),
    ],
                           dtype=jnp.bfloat16)
    logits = jnp.broadcast_to(position_logits, (3, 3, 3))
    eligible = jnp.ones((3, 3), dtype=bool)
    active_rows = jnp.ones((3, ), dtype=bool)

    _, official_remaining = _official_gpu_reference_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
    )
    _, current_remaining = low_confidence_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
        _temperatures(3),
        2,
    )

    np.testing.assert_array_equal(
        official_remaining,
        [[False, False, True], [False, True, True], [False, True, True]],
    )
    np.testing.assert_array_equal(current_remaining, official_remaining)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16],
                         ids=["float32", "bfloat16"])
def test_threshold_one_commits_exactly_one_forced_position(dtype):
    logits = jnp.array([
        [[8.0, 0.0, -4.0, -20.0], [2.0, 0.0, -4.0, -20.0],
         [1.0, 0.0, -4.0, -20.0]],
        [[0.2, 0.0, -4.0, -20.0], [0.4, 0.0, -4.0, -20.0],
         [0.3, 0.0, -4.0, -20.0]],
    ],
                       dtype=dtype)
    eligible = jnp.ones((2, 3), dtype=bool)
    active_rows = jnp.ones((2, ), dtype=bool)
    thresholds = _thresholds(2, value=1.0)

    actual_tokens, actual_remaining = low_confidence_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
        _temperatures(2),
        3,
    )
    expected_tokens, expected_remaining = _official_gpu_reference_commit(
        logits,
        eligible,
        active_rows,
        thresholds,
    )

    np.testing.assert_array_equal(actual_tokens, expected_tokens)
    np.testing.assert_array_equal(actual_remaining, expected_remaining)
    committed = np.asarray(eligible & ~actual_remaining)
    np.testing.assert_array_equal(committed.sum(axis=-1), [1, 1])
    np.testing.assert_array_equal(np.argmax(committed, axis=-1), [0, 1])


def test_low_confidence_diagnostics_report_selected_log_probability_margin():
    logits = jnp.array([[[2.0, 0.0, -10.0]]], dtype=jnp.float32)

    diagnostics = low_confidence_diagnostics(
        logits,
        _thresholds(1, value=0.5),
        _temperatures(1),
        2,
    )

    selected_probability = np.exp(
        float(diagnostics.selected_log_confidence[0, 0]))
    np.testing.assert_allclose(selected_probability,
                               np.exp(2.0) / (np.exp(2.0) + 1.0),
                               rtol=1e-6)
    np.testing.assert_allclose(
        diagnostics.threshold_margin[0, 0],
        diagnostics.selected_log_confidence[0, 0] - np.log(0.5),
        rtol=1e-6,
    )


def test_dual_cache_hidden_selection_matches_reference_shift_rules():
    hidden = jnp.arange(32, dtype=jnp.float32).reshape(1, 32, 1)

    full = select_aligned_hidden_states(
        hidden,
        1,
        32,
        jnp.array(8, dtype=jnp.int32),
        8,
        LogitAlignment.SHIFTED,
        local_alignment=False,
    )
    partial = select_aligned_hidden_states(
        hidden[:, :8],
        1,
        8,
        jnp.array(0, dtype=jnp.int32),
        8,
        LogitAlignment.SHIFTED,
        local_alignment=True,
    )

    np.testing.assert_array_equal(full[:, 0], np.arange(7, 15))
    np.testing.assert_array_equal(partial[:, 0], [0, 0, 1, 2, 3, 4, 5, 6])


def test_low_confidence_commit_keeps_inactive_rows_unchanged():
    logits = jnp.ones((2, 3, 4), dtype=jnp.float32)
    eligible = jnp.ones((2, 3), dtype=bool)

    _, remaining = low_confidence_commit(
        logits,
        eligible,
        jnp.array([True, False]),
        _thresholds(2),
        _temperatures(2),
        3,
    )

    assert remaining[0].sum() == 2
    assert remaining[1].sum() == 0


def test_mask_token_is_excluded_from_commits_and_next_anchor():
    mask_token_id = 7
    target_token_id = 5

    def mask_favoring_forward(model_state, canvas, positions, kv_caches,
                              active_rows, forward_context):
        del model_state, positions, active_rows, forward_context
        logits = jnp.zeros((*canvas.shape, 8), dtype=jnp.float32)
        logits = logits.at[..., mask_token_id].set(100.0)
        logits = logits.at[..., target_token_id].set(20.0)
        return logits, kv_caches

    output = denoise_block(
        mask_favoring_forward,
        low_confidence_commit,
        None,
        jnp.array([[4, 7, 7, 7]], dtype=jnp.int32),
        jnp.array([[False, True, True, True]]),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=4,
    )

    np.testing.assert_array_equal(output.canvas, [[4, 5, 5, 5]])
    np.testing.assert_array_equal(output.next_anchor, [target_token_id])


def _position_forward(vocab_size):

    def forward(model_state, canvas, positions, _kv_caches, active_rows,
                forward_context):
        del model_state, active_rows, forward_context
        targets = positions % vocab_size
        logits = jax.nn.one_hot(targets, vocab_size) * 20.0
        return logits, canvas

    return forward


def test_denoise_block_supports_shifted_logits_and_inactive_rows():
    initial_canvas = jnp.array([[7, 15, 15, 15], [9, 8, 7, 6]],
                               dtype=jnp.int32)
    initial_mask = jnp.array([[False, True, True, True],
                              [False, False, False, False]])
    positions = jnp.array([[10, 11, 12, 13], [20, 21, 22, 23]],
                          dtype=jnp.int32)

    output = denoise_block(
        _position_forward(vocab_size=16),
        low_confidence_commit,
        None,
        initial_canvas,
        initial_mask,
        positions,
        jnp.zeros_like(initial_canvas),
        jnp.array([True, False]),
        _thresholds(2),
        _temperatures(2),
        None,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=15,
        sub_block_size=2,
    )

    np.testing.assert_array_equal(output.canvas[0], [7, 10, 11, 12])
    np.testing.assert_array_equal(output.canvas[1], initial_canvas[1])
    assert int(output.next_anchor[0]) == 13
    assert int(output.next_anchor[1]) == 0


def test_shifted_logits_reject_a_masked_seed_position():
    with pytest.raises(ValueError, match="position 0.*unmasked seed"):
        denoise_block(
            _position_forward(vocab_size=16),
            low_confidence_commit,
            None,
            jnp.full((1, 4), 15, dtype=jnp.int32),
            jnp.ones((1, 4), dtype=bool),
            jnp.arange(4, dtype=jnp.int32)[None, :],
            jnp.zeros((1, 4), dtype=jnp.int32),
            jnp.array([True]),
            _thresholds(1),
            _temperatures(1),
            None,
            logit_alignment=LogitAlignment.SHIFTED,
            next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
            mask_token_id=15,
            sub_block_size=2,
        )


def test_sub_blocks_are_denoised_in_order():
    vocab_size = 32

    def dependent_forward(model_state, canvas, positions, _kv_caches,
                          active_rows, forward_context):
        del model_state, positions, active_rows, forward_context
        first_half_committed = jnp.sum(canvas[:, :2], axis=-1) % vocab_size
        targets = jnp.stack([
            jnp.full_like(first_half_committed, 2),
            jnp.full_like(first_half_committed, 3),
            first_half_committed,
            first_half_committed,
        ],
                            axis=1)
        logits = jax.nn.one_hot(targets, vocab_size) * 20.0
        return logits, canvas

    output = denoise_block(
        dependent_forward,
        low_confidence_commit,
        None,
        jnp.full((1, 4), 31, dtype=jnp.int32),
        jnp.ones((1, 4), dtype=bool),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.ALL_MASKED,
        mask_token_id=31,
        sub_block_size=2,
    )

    np.testing.assert_array_equal(output.canvas[0], [2, 3, 5, 5])


def test_final_forward_refreshes_committed_kv_and_next_anchor():
    vocab_size = 32

    def canvas_dependent_forward(model_state, canvas, positions, _kv_caches,
                                 active_rows, forward_context):
        del model_state, positions, active_rows, forward_context
        base_targets = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
        next_target = jnp.sum(canvas, axis=-1) % vocab_size
        targets = base_targets.at[:, -1].set(next_target)
        logits = jax.nn.one_hot(targets, vocab_size) * 20.0
        return logits, canvas

    output = denoise_block(
        canvas_dependent_forward,
        low_confidence_commit,
        None,
        jnp.full((1, 4), 31, dtype=jnp.int32),
        jnp.ones((1, 4), dtype=bool),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.zeros((1, 4), dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=31,
        sub_block_size=4,
    )

    np.testing.assert_array_equal(output.kv_caches, output.canvas)
    assert int(output.next_anchor[0]) == int(output.canvas.sum() % vocab_size)


def test_terminal_block_skips_non_dual_cache_final_forward():

    def forward(model_state, canvas, positions, kv_caches, active_rows,
                forward_context):
        del model_state, positions, active_rows, forward_context
        logits = jnp.zeros((*canvas.shape, 4), dtype=jnp.float32)
        return logits, kv_caches + 1

    initial_canvas = jnp.array([[0, 3, 3, 3]], dtype=jnp.int32)
    common_args = (
        forward,
        low_confidence_commit,
        None,
        initial_canvas,
        initial_canvas == 3,
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.array(0, dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=1.0),
        _temperatures(1),
        None,
    )
    nonterminal = denoise_block(
        *common_args,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=3,
        sub_block_size=4,
        needs_final_forward=jnp.array([True]),
    )
    terminal = denoise_block(
        *common_args,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=3,
        sub_block_size=4,
        needs_final_forward=jnp.array([False]),
    )

    np.testing.assert_array_equal(terminal.canvas, nonterminal.canvas)
    np.testing.assert_array_equal(terminal.denoise_steps,
                                  nonterminal.denoise_steps)
    assert int(terminal.kv_caches) + 1 == int(nonterminal.kv_caches)
    np.testing.assert_array_equal(terminal.next_anchor, [0])


def test_non_dual_cache_stops_after_resolved_eos_prefix():
    mask_token_id = 4
    eos_token_id = 2
    targets = jnp.array([1, eos_token_id, 3, 3], dtype=jnp.int32)

    def forward(model_state, canvas, positions, kv_caches, active_rows,
                forward_context):
        del model_state, canvas, positions, forward_context
        logits = jax.nn.one_hot(targets, 5) * 20.0
        logits = jnp.broadcast_to(logits, (active_rows.shape[0], 4, 5))
        return logits, kv_caches + jnp.any(active_rows).astype(jnp.int32)

    output = denoise_block(
        forward,
        low_confidence_commit,
        None,
        jnp.full((1, 4), mask_token_id, dtype=jnp.int32),
        jnp.ones((1, 4), dtype=bool),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.array(0, dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=1.0),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=4,
        stop_on_eos_rows=jnp.array([True]),
        eos_token_ids=(eos_token_id, ),
    )

    np.testing.assert_array_equal(
        output.canvas, [[1, eos_token_id, mask_token_id, mask_token_id]])
    np.testing.assert_array_equal(output.denoise_steps, [2])
    np.testing.assert_array_equal(output.stopped_rows, [True])
    np.testing.assert_array_equal(output.next_anchor, [0])
    assert int(output.kv_caches) == 2


def test_eos_in_seed_context_does_not_stop_generation():
    mask_token_id = 4
    eos_token_id = 2
    targets = jnp.array([0, 1, 3, 3], dtype=jnp.int32)

    def forward(model_state, canvas, positions, kv_caches, active_rows,
                forward_context):
        del model_state, canvas, positions, active_rows, forward_context
        logits = jax.nn.one_hot(targets, 5) * 20.0
        return logits[None, :, :], kv_caches + 1

    output = denoise_block(
        forward,
        low_confidence_commit,
        None,
        jnp.array(
            [[eos_token_id, mask_token_id, mask_token_id, mask_token_id]],
            dtype=jnp.int32),
        jnp.array([[False, True, True, True]]),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.array(0, dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=1.0),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=4,
        stop_on_eos_rows=jnp.array([True]),
        eos_token_ids=(eos_token_id, ),
    )

    np.testing.assert_array_equal(output.canvas, [[eos_token_id, 1, 3, 3]])
    np.testing.assert_array_equal(output.stopped_rows, [False])


def test_step_cap_force_fills_and_final_forward_refreshes_kv():

    initial_canvas = jnp.array([[4, 7, 7, 7]], dtype=jnp.int32)

    def low_confidence_forward(model_state, canvas, positions, kv_caches,
                               active_rows, forward_context):
        del model_state, positions, forward_context
        logits = jnp.zeros((*canvas.shape, 8), dtype=jnp.float32)
        updated_kv = jnp.where(active_rows[:, None], canvas, kv_caches)
        return logits, updated_kv

    output = denoise_block(
        low_confidence_forward,
        low_confidence_commit,
        None,
        initial_canvas,
        jnp.array([[False, True, True, True]]),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.full((1, 4), -1, dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=0.99),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.ALL_MASKED,
        mask_token_id=7,
        sub_block_size=4,
        max_denoise_steps=1,
    )

    np.testing.assert_array_equal(output.canvas, [[4, 0, 0, 0]])
    np.testing.assert_array_equal(output.kv_caches, output.canvas)
    np.testing.assert_array_equal(output.denoise_steps, [1])

    def next_block_token(cache):
        logits = jax.nn.one_hot(jnp.sum(cache, axis=-1) % 8, 8)
        return jnp.argmax(logits, axis=-1)

    # The last three positions were bulk-committed after the only denoise
    # forward. A stale provisional cache still contains MASK values and changes
    # the next block prediction; the final forward exposes the committed tokens.
    np.testing.assert_array_equal(next_block_token(output.kv_caches), [4])
    np.testing.assert_array_equal(next_block_token(initial_canvas), [1])


def test_dual_cache_acceptance_trace_is_opt_in_and_semantics_preserving():
    mask_token_id = 3
    sub_block_size = 2

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], sub_block_size, 4),
                           dtype=jnp.float32)
        return logits, kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], sub_block_size, 4),
                           dtype=jnp.float32)
        return logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, active_rows, forward_context
        return jnp.zeros((1, 4), dtype=jnp.float32), kv_caches.at[2].add(1)

    def run(trace_acceptance_steps):
        return denoise_block_dual_cache(
            full_forward,
            partial_forward,
            final_forward,
            low_confidence_commit,
            None,
            jnp.array([[0, mask_token_id, mask_token_id, mask_token_id]],
                      dtype=jnp.int32),
            jnp.array([[False, True, True, True]]),
            jnp.arange(40, 44, dtype=jnp.int32)[None, :],
            jnp.zeros((3, ), dtype=jnp.int32),
            jnp.array([True]),
            _thresholds(1, value=1.0),
            _temperatures(1),
            None,
            logit_alignment=LogitAlignment.SHIFTED,
            next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
            mask_token_id=mask_token_id,
            sub_block_size=sub_block_size,
            commit_diagnostics_fn=low_confidence_diagnostics,
            trace_acceptance_steps=trace_acceptance_steps,
        )

    untraced = run(False)
    traced = run(True)

    assert untraced.acceptance_trace is None
    np.testing.assert_array_equal(traced.canvas, untraced.canvas)
    np.testing.assert_array_equal(traced.next_anchor, untraced.next_anchor)
    np.testing.assert_array_equal(traced.denoise_steps, untraced.denoise_steps)
    np.testing.assert_array_equal(traced.kv_caches, untraced.kv_caches)
    assert int(traced.q32_forward_calls) == int(untraced.q32_forward_calls)
    assert int(traced.q8_forward_calls) == int(untraced.q8_forward_calls)

    trace = traced.acceptance_trace
    assert trace is not None
    count = int(trace.count)
    assert count == 3
    assert int(trace.block_start) == 40
    np.testing.assert_array_equal(trace.sub_block_starts[:count], [40, 42, 42])
    np.testing.assert_array_equal(trace.iterations[:count], [0, 0, 1])
    np.testing.assert_array_equal(trace.forward_kinds[:count], [0, 0, 1])
    np.testing.assert_array_equal(
        trace.row0_eligible[:count],
        [[False, True], [True, True], [False, True]],
    )
    np.testing.assert_array_equal(
        trace.row0_commit[:count],
        [[False, True], [True, False], [False, True]],
    )
    np.testing.assert_array_equal(
        trace.row0_remaining[:count],
        [[False, False], [False, True], [False, False]],
    )
    assert np.all(np.isfinite(trace.row0_selected_log_confidence[:count]))
    assert np.all(trace.row0_threshold_margin[:count] < 0.0)


def test_dual_cache_acceptance_trace_requires_diagnostics_function():
    with pytest.raises(ValueError, match="requires commit_diagnostics_fn"):
        denoise_block_dual_cache(
            lambda *args: (jnp.zeros((1, 2, 4)), args[3]),
            lambda *args: (jnp.zeros((1, 2, 4)), args[3]),
            lambda *args: (jnp.zeros((1, 4)), args[3]),
            low_confidence_commit,
            None,
            jnp.array([[0, 3]], dtype=jnp.int32),
            jnp.array([[False, True]]),
            jnp.array([[0, 1]], dtype=jnp.int32),
            jnp.zeros((1, ), dtype=jnp.int32),
            jnp.array([True]),
            _thresholds(1),
            _temperatures(1),
            None,
            logit_alignment=LogitAlignment.SHIFTED,
            next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
            mask_token_id=3,
            sub_block_size=2,
            trace_acceptance_steps=True,
        )


def test_q8_log_confidence_bias_changes_only_partial_commit_threshold():
    mask_token_id = 3
    threshold = 0.9

    def logits_for_probability(probability):
        gap = np.log(probability / (1.0 - probability))
        return jnp.array([gap, 0.0, -20.0, -20.0], dtype=jnp.float32)

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[0].add(1)

    partial_logits = jnp.stack([
        logits_for_probability(0.5),
        logits_for_probability(0.88),
        logits_for_probability(0.87),
        logits_for_probability(0.5),
    ])[None, ...]

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, canvas, positions, active_rows, forward_context, start
        return partial_logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, active_rows, forward_context
        return jnp.zeros((1, 4), dtype=jnp.float32), kv_caches.at[2].add(1)

    def run(q8_log_confidence_bias=0.0):
        return denoise_block_dual_cache(
            full_forward,
            partial_forward,
            final_forward,
            low_confidence_commit,
            None,
            jnp.full((1, 4), mask_token_id, dtype=jnp.int32),
            jnp.ones((1, 4), dtype=bool),
            jnp.arange(4, dtype=jnp.int32)[None, :],
            jnp.zeros((3, ), dtype=jnp.int32),
            jnp.array([True]),
            _thresholds(1, value=threshold),
            _temperatures(1),
            None,
            logit_alignment=LogitAlignment.SAME_POSITION,
            next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
            mask_token_id=mask_token_id,
            sub_block_size=4,
            needs_final_forward=jnp.array([False]),
            commit_diagnostics_fn=low_confidence_diagnostics,
            trace_acceptance_steps=True,
            q8_log_confidence_bias=q8_log_confidence_bias,
        )

    default = run()
    explicit_zero = run(0.0)
    biased = run(0.05)

    assert jax.tree_util.tree_structure(
        default) == jax.tree_util.tree_structure(explicit_zero)
    for default_value, explicit_value in zip(
            jax.tree_util.tree_leaves(default),
            jax.tree_util.tree_leaves(explicit_zero),
            strict=True):
        np.testing.assert_array_equal(default_value, explicit_value)

    assert int(default.q32_forward_calls) == 1
    assert int(biased.q32_forward_calls) == 1
    assert int(default.final_q32_forward_calls) == 0
    assert int(biased.final_q32_forward_calls) == 0
    assert int(default.q8_forward_calls) == 3
    assert int(biased.q8_forward_calls) == 2

    trace = biased.acceptance_trace
    assert trace is not None
    trace_count = int(trace.count)
    q32_steps = np.asarray(trace.forward_kinds[:trace_count]) == 0
    q8_steps = ~q32_steps
    assert np.all(
        np.asarray(trace.row0_threshold_margin[:trace_count])[q32_steps] < 0.0)
    first_q8_log_confidence = np.asarray(
        trace.row0_selected_log_confidence[:trace_count])[q8_steps][0]
    np.testing.assert_allclose(
        first_q8_log_confidence[1:3],
        np.log([0.88, 0.87]),
        atol=1e-6,
    )
    first_q8_margins = np.asarray(
        trace.row0_threshold_margin[:trace_count])[q8_steps][0]
    assert first_q8_margins[1] > 0.0
    assert first_q8_margins[2] > 0.0
    assert first_q8_margins[3] < 0.0


@pytest.mark.parametrize("q8_log_confidence_bias", [0.0, 0.5])
def test_threshold_one_uses_q32_once_then_q8_for_each_sub_block(
        q8_log_confidence_bias):
    block_size = 32
    sub_block_size = 8
    mask_token_id = 3

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], sub_block_size, 4),
                           dtype=jnp.float32)
        return logits, kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], sub_block_size, 4),
                           dtype=jnp.float32)
        return logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, positions, active_rows, forward_context
        logits = jnp.zeros((canvas.shape[0], 4), dtype=jnp.float32)
        return logits, kv_caches.at[2].add(1)

    initial_canvas = jnp.array([[0] + [mask_token_id] * (block_size - 1)],
                               dtype=jnp.int32)
    initial_mask = initial_canvas == mask_token_id
    output = denoise_block_dual_cache(
        full_forward,
        partial_forward,
        final_forward,
        low_confidence_commit,
        None,
        initial_canvas,
        initial_mask,
        jnp.arange(block_size, dtype=jnp.int32)[None, :],
        jnp.zeros((3, ), dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=1.0),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=sub_block_size,
        q8_log_confidence_bias=q8_log_confidence_bias,
    )

    np.testing.assert_array_equal(output.canvas, np.zeros((1, block_size)))
    np.testing.assert_array_equal(output.kv_caches, [4, 27, 1])
    np.testing.assert_array_equal(output.denoise_steps, [31])
    assert int(output.q32_forward_calls) == 5
    assert int(output.q8_forward_calls) == 27
    assert int(output.final_q32_forward_calls) == 1
    assert int(output.q32_forward_calls) * block_size + int(
        output.q8_forward_calls) * sub_block_size == 376


def test_dual_cache_terminal_block_skips_final_q32():
    mask_token_id = 3

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, forward_context
        target = jnp.where(active_rows, 2, 1)
        logits = jax.nn.one_hot(target, 4) * 20.0
        return logits, kv_caches.at[2].add(1)

    initial_canvas = jnp.array(
        [[0, mask_token_id, mask_token_id, mask_token_id]], dtype=jnp.int32)

    def run(needs_final_forward):
        return denoise_block_dual_cache(
            full_forward,
            partial_forward,
            final_forward,
            low_confidence_commit,
            None,
            initial_canvas,
            initial_canvas == mask_token_id,
            jnp.arange(4, dtype=jnp.int32)[None, :],
            jnp.zeros((3, ), dtype=jnp.int32),
            jnp.array([True]),
            _thresholds(1, value=1.0),
            _temperatures(1),
            None,
            logit_alignment=LogitAlignment.SHIFTED,
            next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
            mask_token_id=mask_token_id,
            sub_block_size=4,
            needs_final_forward=jnp.array([needs_final_forward]),
        )

    nonterminal = run(True)
    terminal = run(False)

    np.testing.assert_array_equal(terminal.canvas, nonterminal.canvas)
    np.testing.assert_array_equal(terminal.denoise_steps,
                                  nonterminal.denoise_steps)
    np.testing.assert_array_equal(terminal.kv_caches, [1, 2, 0])
    np.testing.assert_array_equal(nonterminal.kv_caches, [1, 2, 1])
    np.testing.assert_array_equal(terminal.next_anchor, [0])
    np.testing.assert_array_equal(nonterminal.next_anchor, [2])
    assert int(terminal.q32_forward_calls) + 1 == int(
        nonterminal.q32_forward_calls)
    assert int(terminal.q8_forward_calls) == int(nonterminal.q8_forward_calls)
    assert int(terminal.final_q32_forward_calls) == 0
    assert int(nonterminal.final_q32_forward_calls) == 1


def test_dual_cache_stops_after_resolved_eos_prefix():
    mask_token_id = 4
    eos_token_id = 2
    targets = jnp.array([1, eos_token_id, 3, 3], dtype=jnp.int32)

    def block_logits(batch_size):
        logits = jax.nn.one_hot(targets, 5) * 20.0
        return jnp.broadcast_to(logits, (batch_size, 4, 5))

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        return block_logits(canvas.shape[0]), kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        return block_logits(canvas.shape[0]), kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, active_rows, forward_context
        return jnp.zeros((1, 5), dtype=jnp.float32), kv_caches.at[2].add(1)

    output = denoise_block_dual_cache(
        full_forward,
        partial_forward,
        final_forward,
        low_confidence_commit,
        None,
        jnp.full((1, 4), mask_token_id, dtype=jnp.int32),
        jnp.ones((1, 4), dtype=bool),
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.zeros((3, ), dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=1.0),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=4,
        stop_on_eos_rows=jnp.array([True]),
        eos_token_ids=(eos_token_id, ),
    )

    np.testing.assert_array_equal(
        output.canvas, [[1, eos_token_id, mask_token_id, mask_token_id]])
    np.testing.assert_array_equal(output.stopped_rows, [True])
    np.testing.assert_array_equal(output.kv_caches, [1, 1, 0])
    assert int(output.q32_forward_calls) == 1
    assert int(output.q8_forward_calls) == 1
    assert int(output.final_q32_forward_calls) == 0


def test_dual_cache_eos_termination_is_row_local():
    mask_token_id = 4
    eos_token_id = 2
    targets = jnp.array([1, eos_token_id, 3, 3], dtype=jnp.int32)

    def block_logits(batch_size):
        logits = jax.nn.one_hot(targets, 5) * 20.0
        return jnp.broadcast_to(logits, (batch_size, 4, 5))

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        return block_logits(canvas.shape[0]), kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        return block_logits(canvas.shape[0]), kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, forward_context
        target = jnp.where(active_rows, 1, 0)
        return jax.nn.one_hot(target, 5) * 20.0, kv_caches.at[2].add(1)

    output = denoise_block_dual_cache(
        full_forward,
        partial_forward,
        final_forward,
        low_confidence_commit,
        None,
        jnp.full((2, 4), mask_token_id, dtype=jnp.int32),
        jnp.ones((2, 4), dtype=bool),
        jnp.tile(jnp.arange(4, dtype=jnp.int32), (2, 1)),
        jnp.zeros((3, ), dtype=jnp.int32),
        jnp.array([True, True]),
        _thresholds(2, value=1.0),
        _temperatures(2),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=4,
        stop_on_eos_rows=jnp.array([True, False]),
        eos_token_ids=(eos_token_id, ),
    )

    np.testing.assert_array_equal(
        output.canvas,
        [[1, eos_token_id, mask_token_id, mask_token_id],
         [1, eos_token_id, 3, 3]],
    )
    np.testing.assert_array_equal(output.stopped_rows, [True, False])
    np.testing.assert_array_equal(output.next_anchor, [0, 1])
    assert int(output.q32_forward_calls) == 2
    assert int(output.q8_forward_calls) == 3
    assert int(output.final_q32_forward_calls) == 1


def test_dual_cache_mixed_terminal_rows_run_one_batch_final_q32():
    mask_token_id = 3

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, forward_context
        target = jnp.where(active_rows, 2, 1)
        logits = jax.nn.one_hot(target, 4) * 20.0
        return logits, kv_caches.at[2].add(1)

    initial_canvas = jnp.array(
        [[0, mask_token_id, mask_token_id, mask_token_id],
         [0, mask_token_id, mask_token_id, mask_token_id]],
        dtype=jnp.int32)
    output = denoise_block_dual_cache(
        full_forward,
        partial_forward,
        final_forward,
        low_confidence_commit,
        None,
        initial_canvas,
        initial_canvas == mask_token_id,
        jnp.tile(jnp.arange(4, dtype=jnp.int32), (2, 1)),
        jnp.zeros((3, ), dtype=jnp.int32),
        jnp.array([True, True]),
        _thresholds(2, value=1.0),
        _temperatures(2),
        None,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=4,
        needs_final_forward=jnp.array([False, True]),
    )

    np.testing.assert_array_equal(output.canvas, np.zeros((2, 4)))
    np.testing.assert_array_equal(output.next_anchor, [0, 2])
    np.testing.assert_array_equal(output.kv_caches, [1, 2, 1])
    assert int(output.q32_forward_calls) == 2
    assert int(output.q8_forward_calls) == 2
    assert int(output.final_q32_forward_calls) == 1


def test_dual_cache_keeps_q32_until_shifted_sub_block_anchor_is_committed():
    mask_token_id = 3
    sub_block_size = 2

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context
        first_call_for_second_sub_block = (start == 2) & (kv_caches[0] == 1)
        confident_position = jnp.where(first_call_for_second_sub_block, 1, 0)
        logits = jnp.zeros((canvas.shape[0], sub_block_size, 4),
                           dtype=jnp.float32)
        logits = logits.at[:, confident_position, 0].set(20.0)
        return logits, kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], sub_block_size, 4),
                           dtype=jnp.float32)
        return logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, positions, active_rows, forward_context
        logits = jnp.zeros((canvas.shape[0], 4), dtype=jnp.float32)
        return logits, kv_caches.at[2].add(1)

    initial_canvas = jnp.array(
        [[0, mask_token_id, mask_token_id, mask_token_id]], dtype=jnp.int32)
    output = denoise_block_dual_cache(
        full_forward,
        partial_forward,
        final_forward,
        low_confidence_commit,
        None,
        initial_canvas,
        initial_canvas == mask_token_id,
        jnp.arange(4, dtype=jnp.int32)[None, :],
        jnp.zeros((3, ), dtype=jnp.int32),
        jnp.array([True]),
        _thresholds(1, value=0.9),
        _temperatures(1),
        None,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=mask_token_id,
        sub_block_size=sub_block_size,
    )

    np.testing.assert_array_equal(output.canvas, [[0, 0, 0, 0]])
    np.testing.assert_array_equal(output.kv_caches, [3, 0, 1])
    assert int(output.q32_forward_calls) == 4
    assert int(output.q8_forward_calls) == 0


def test_dual_cache_partial_forward_receives_only_rows_with_work():

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[0].add(jnp.sum(active_rows))

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[1].add(jnp.sum(active_rows))

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, active_rows, forward_context
        return jnp.zeros((2, 4), dtype=jnp.float32), kv_caches

    output = denoise_block_dual_cache(
        full_forward,
        partial_forward,
        final_forward,
        low_confidence_commit,
        None,
        jnp.array([[0, 3, 2, 1], [0, 3, 3, 3]], dtype=jnp.int32),
        jnp.array([[False, True, False, False], [False, True, True, True]]),
        jnp.tile(jnp.arange(4, dtype=jnp.int32), (2, 1)),
        jnp.zeros((2, ), dtype=jnp.int32),
        jnp.array([True, True]),
        _thresholds(2, value=1.0),
        _temperatures(2),
        None,
        logit_alignment=LogitAlignment.SHIFTED,
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        mask_token_id=3,
        sub_block_size=4,
    )

    np.testing.assert_array_equal(output.kv_caches, [2, 2])
    np.testing.assert_array_equal(output.denoise_steps, [1, 3])
    assert int(output.q8_forward_calls) == 2


def test_dual_cache_three_row_bucket_matches_four_row_padding():

    def full_forward(model_state, canvas, positions, kv_caches, active_rows,
                     forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[0].add(1)

    def partial_forward(model_state, canvas, positions, kv_caches, active_rows,
                        forward_context, start):
        del model_state, positions, active_rows, forward_context, start
        logits = jnp.zeros((canvas.shape[0], 4, 4), dtype=jnp.float32)
        return logits, kv_caches.at[1].add(1)

    def final_forward(model_state, canvas, positions, kv_caches, active_rows,
                      forward_context):
        del model_state, canvas, positions, forward_context
        targets = jnp.where(active_rows, 2, 1)
        return jax.nn.one_hot(targets, 4) * 20.0, kv_caches.at[2].add(1)

    active_canvas = jnp.array([[0, 3, 3, 3], [0, 3, 2, 1], [0, 3, 3, 2]],
                              dtype=jnp.int32)

    def run(pad_to_four):
        if pad_to_four:
            canvas = jnp.concatenate(
                [active_canvas,
                 jnp.array([[9, 8, 7, 6]], dtype=jnp.int32)])
            active_rows = jnp.array([True, True, True, False])
        else:
            canvas = active_canvas
            active_rows = jnp.ones((3, ), dtype=bool)
        batch_size = canvas.shape[0]
        return denoise_block_dual_cache(
            full_forward,
            partial_forward,
            final_forward,
            low_confidence_commit,
            None,
            canvas,
            canvas == 3,
            jnp.tile(jnp.arange(4, dtype=jnp.int32), (batch_size, 1)),
            jnp.zeros((3, ), dtype=jnp.int32),
            active_rows,
            _thresholds(batch_size, value=0.9),
            _temperatures(batch_size),
            None,
            logit_alignment=LogitAlignment.SHIFTED,
            next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
            mask_token_id=3,
            sub_block_size=4,
            q8_log_confidence_bias=0.05,
        )

    exact = run(False)
    padded = run(True)

    np.testing.assert_array_equal(exact.canvas, padded.canvas[:3])
    np.testing.assert_array_equal(exact.next_anchor, padded.next_anchor[:3])
    np.testing.assert_array_equal(exact.denoise_steps,
                                  padded.denoise_steps[:3])
    np.testing.assert_array_equal(exact.kv_caches, padded.kv_caches)
    assert int(exact.q32_forward_calls) == int(padded.q32_forward_calls)
    assert int(exact.q8_forward_calls) == int(padded.q8_forward_calls)


def test_heterogeneous_rows_preserve_inactive_kv():

    def active_aware_forward(model_state, canvas, positions, kv_caches,
                             active_rows, forward_context):
        del model_state, positions, forward_context
        logits = jnp.zeros((*canvas.shape, 8), dtype=jnp.float32)
        updated_kv = jnp.where(active_rows[:, None], canvas, kv_caches)
        return logits, updated_kv

    initial_canvas = jnp.array([
        [4, 7, 6, 5],
        [3, 7, 7, 7],
        [9, 8, 7, 6],
    ],
                               dtype=jnp.int32)
    initial_kv = jnp.array([
        [90, 91, 92, 93],
        [80, 81, 82, 83],
        [70, 71, 72, 73],
    ],
                           dtype=jnp.int32)
    output = denoise_block(
        active_aware_forward,
        low_confidence_commit,
        None,
        initial_canvas,
        jnp.array([
            [False, True, False, False],
            [False, True, True, True],
            [True, True, True, True],
        ]),
        jnp.broadcast_to(jnp.arange(4, dtype=jnp.int32), (3, 4)),
        initial_kv,
        jnp.array([True, True, False]),
        _thresholds(3, value=0.99),
        _temperatures(3),
        None,
        logit_alignment=LogitAlignment.SAME_POSITION,
        next_block_policy=NextBlockPolicy.ALL_MASKED,
        mask_token_id=7,
        sub_block_size=4,
    )

    np.testing.assert_array_equal(output.canvas[2], initial_canvas[2])
    np.testing.assert_array_equal(output.kv_caches[:2], output.canvas[:2])
    np.testing.assert_array_equal(output.kv_caches[2], initial_kv[2])
    np.testing.assert_array_equal(output.denoise_steps, [1, 3, 0])
