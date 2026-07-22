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
        [[10.0, 0.0], [0.1, 0.0], [0.0, 0.1]],
        [[0.1, 0.0], [0.0, 0.1], [10.0, 0.0]],
    ])
    eligible = jnp.ones((2, 3), dtype=bool)

    tokens, remaining = low_confidence_commit(
        logits,
        eligible,
        jnp.array([True, True]),
        _thresholds(2),
        _temperatures(2),
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
    )

    assert remaining[0].sum() == 2
    assert remaining[1].sum() == 0


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
        sub_block_size=2,
    )

    np.testing.assert_array_equal(output.canvas[0], [7, 10, 11, 12])
    np.testing.assert_array_equal(output.canvas[1], initial_canvas[1])
    assert int(output.next_anchor[0]) == 13
    assert int(output.next_anchor[1]) == 0


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
        sub_block_size=4,
    )

    np.testing.assert_array_equal(output.kv_caches, output.canvas)
    assert int(output.next_anchor[0]) == int(output.canvas.sum() % vocab_size)
