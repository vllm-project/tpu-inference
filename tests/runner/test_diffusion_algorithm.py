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
denoise_block = _PROGRAM.denoise_block


def _thresholds(batch_size, value=0.9):
    return jnp.full((batch_size, ), value, dtype=jnp.float32)


def _temperatures(batch_size):
    return jnp.zeros((batch_size, ), dtype=jnp.float32)


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
