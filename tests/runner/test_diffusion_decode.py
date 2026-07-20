# Copyright 2025 Google LLC
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
"""CPU unit tests for the block-diffusion denoising decode loop.

The whole seed -> forward -> shifted-logit -> threshold-commit -> advance loop
is exercised end-to-end on CPU with a *stub* ``forward_fn`` that returns
deterministic, position-addressed logits. No TPU, real model, RPA kernel, or
paged KV cache is involved (those pieces are marked TPU-required and covered by
AST/inspection checks instead).

The behavioral tests use the REAL ``diffusion_commit`` (the foundation's
threshold-commit sampler). On a full-dependency host they import
``tpu_inference.runner.diffusion_decode`` directly; on a minimal ``jax[cpu]``
venv (no vllm / heavy tpu_inference deps) they fall back to loading the
pure-logic module standalone with the real ``diffusion_commit`` extracted from
source, so the loop still runs for real on CPU.
"""

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TPU_RUNNER = _REPO_ROOT / "tpu_inference" / "runner" / "tpu_runner.py"
_DIFFUSION_DECODE = (_REPO_ROOT / "tpu_inference" / "runner" /
                     "diffusion_decode.py")
_SAMPLING = (_REPO_ROOT / "tpu_inference" / "layers" / "jax" / "sample" /
             "sampling.py")


def _load_diffusion_decode_standalone():
    """Load diffusion_decode.py without the heavy tpu_inference import chain.

    Extracts the REAL ``diffusion_commit`` from ``sampling.py`` source (it is
    pure jax/jnp) and injects a stub ``sampling`` module so the standalone load
    of ``diffusion_decode.py`` binds the real commit function.
    """
    import importlib.util
    import sys
    import types

    # 1. Extract + exec the real diffusion_commit (pure jax/jnp, no deps).
    mod_ast = ast.parse(_SAMPLING.read_text())
    fn = next(
        n for n in mod_ast.body
        if isinstance(n, ast.FunctionDef) and n.name == "diffusion_commit")
    ns = {"jax": jax, "jnp": jnp}
    exec(  # noqa: S102 - executing our own repo source under test
        compile(ast.Module(body=[fn], type_ignores=[]), str(_SAMPLING),
                "exec"), ns)

    # 2. Inject stub parent packages + a sampling module exposing the real fn.
    for pkg in ("tpu_inference", "tpu_inference.layers",
                "tpu_inference.layers.jax", "tpu_inference.layers.jax.sample"):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # mark as package
            sys.modules[pkg] = m
    samp = types.ModuleType("tpu_inference.layers.jax.sample.sampling")
    samp.diffusion_commit = ns["diffusion_commit"]
    sys.modules["tpu_inference.layers.jax.sample.sampling"] = samp

    # 3. Load diffusion_decode.py standalone; its top-level import resolves to
    #    the stub sampling module above.
    spec = importlib.util.spec_from_file_location(
        "diffusion_decode_under_test", _DIFFUSION_DECODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from tpu_inference.runner.diffusion_decode import (  # noqa: F401
        DiffusionDecodeResult, denoise_block, diffusion_decode)
except Exception:  # pragma: no cover - minimal CPU venv path
    _dd = _load_diffusion_decode_standalone()
    DiffusionDecodeResult = _dd.DiffusionDecodeResult
    denoise_block = _dd.denoise_block
    diffusion_decode = _dd.diffusion_decode

# ---- Test constants ---------------------------------------------------------
VOCAB = 16
MASK_ID = 15  # out of the normal token range so leaks are detectable
EOS_ID = 14  # never produced by the normal position->token map below


def _target_array(n: int) -> np.ndarray:
    """Deterministic absolute-position -> token map in [0, VOCAB-3]."""
    return np.array([p % (VOCAB - 2) for p in range(n)], dtype=np.int32)


def _make_stub_forward(global_target: np.ndarray, logit_scale: float):
    """A pure-JAX stub forward_fn addressed by ABSOLUTE position.

    ``full_logits[j]`` peaks on ``global_target[positions[j]]`` with confidence
    controlled by ``logit_scale`` (large -> commits via threshold in one step;
    small -> below threshold so only the forced-argmax position commits).
    The stub ignores the canvas so the per-position argmax is stable, which lets
    the tests assert exact committed values.
    """
    g = jnp.asarray(global_target)

    def forward_fn(canvas, positions, kv_caches):
        target = g[positions]
        logits = jax.nn.one_hot(target, VOCAB, dtype=jnp.float32) * logit_scale
        return logits, kv_caches

    return forward_fn


class TestDenoiseBlock:
    """Direct on-device single-block tests (a, b, c + step-cap cleanup)."""

    def test_a_block_fully_commits_in_one_step_high_confidence(self):
        block_size = 4
        prefix = 10
        g = _target_array(256)
        fwd = _make_stub_forward(g, logit_scale=30.0)
        positions = jnp.arange(block_size, dtype=jnp.int32) + prefix

        canvas, next_seed, steps, _ = denoise_block(
            fwd,
            jnp.array(7, dtype=jnp.int32),
            positions,
            None,
            block_size=block_size,
            mask_id=MASK_ID,
            threshold=0.9,
            temperature=0.0,
            max_denoise_steps=block_size,
        )
        canvas = np.asarray(canvas)
        # High confidence -> everything commits via threshold in a single step.
        assert int(steps) == 1
        # No masked positions leak into the output.
        assert not np.any(canvas == MASK_ID)

    def test_b_forced_argmax_guarantees_one_commit_per_step(self):
        block_size = 4
        prefix = 3
        g = _target_array(256)
        # Below-threshold confidence: only the forced (highest-confidence
        # masked) position commits each iteration -> one commit per step.
        fwd = _make_stub_forward(g, logit_scale=0.05)
        positions = jnp.arange(block_size, dtype=jnp.int32) + prefix

        canvas, _, steps, _ = denoise_block(
            fwd,
            jnp.array(1, dtype=jnp.int32),
            positions,
            None,
            block_size=block_size,
            mask_id=MASK_ID,
            threshold=0.9,
            temperature=0.0,
            max_denoise_steps=block_size,
        )
        canvas = np.asarray(canvas)
        # 3 initially-masked positions, one commit per step -> exactly 3 steps.
        assert int(steps) == block_size - 1
        assert not np.any(canvas == MASK_ID)

    def test_c_shifted_logit_indexing_is_i_from_i_minus_1(self):
        block_size = 4
        prefix = 10
        seed = 7
        g = _target_array(256)
        fwd = _make_stub_forward(g, logit_scale=30.0)
        positions = jnp.arange(block_size, dtype=jnp.int32) + prefix

        canvas, next_seed, _, _ = denoise_block(
            fwd,
            jnp.array(seed, dtype=jnp.int32),
            positions,
            None,
            block_size=block_size,
            mask_id=MASK_ID,
            threshold=0.9,
            temperature=0.0,
            max_denoise_steps=block_size,
        )
        canvas = np.asarray(canvas)
        # Position 0 is the given seed (committed, never overwritten).
        assert int(canvas[0]) == seed
        # Shifted convention: canvas[i] is predicted from hidden i-1, i.e.
        # equals the stub target at ABSOLUTE position (prefix + i - 1).
        assert int(canvas[1]) == int(g[prefix + 0])
        assert int(canvas[2]) == int(g[prefix + 1])
        assert int(canvas[3]) == int(g[prefix + 2])
        # Next block's seed = argmax of the last (unshifted) position's logits
        # = target at the final absolute position.
        assert int(next_seed) == int(g[prefix + block_size - 1])

    def test_step_cap_forces_fill_no_mask_leak(self):
        # Cap denoise steps below what full commitment needs; the post-loop
        # force-fill must still leave no mask_id in the canvas.
        block_size = 4
        prefix = 20
        g = _target_array(256)
        fwd = _make_stub_forward(g, logit_scale=0.05)
        positions = jnp.arange(block_size, dtype=jnp.int32) + prefix

        canvas, _, steps, _ = denoise_block(
            fwd,
            jnp.array(2, dtype=jnp.int32),
            positions,
            None,
            block_size=block_size,
            mask_id=MASK_ID,
            threshold=0.9,
            temperature=0.0,
            max_denoise_steps=1,  # far fewer than the 3 needed
        )
        canvas = np.asarray(canvas)
        assert int(steps) == 1
        assert not np.any(canvas == MASK_ID)
        # Force-filled values still follow the shifted convention.
        assert int(canvas[1]) == int(g[prefix + 0])
        assert int(canvas[2]) == int(g[prefix + 1])
        assert int(canvas[3]) == int(g[prefix + 2])


class TestDiffusionDecodeMultiBlock:
    """Host-side multi-block progression tests (d + truncation + EOS)."""

    def test_d_multi_block_seeds_next_block_first_token(self):
        block_size = 4
        prefix = 5
        first_token = 9
        max_tokens = 10
        g = _target_array(256)
        fwd = _make_stub_forward(g, logit_scale=30.0)

        res = diffusion_decode(
            fwd,
            first_token=first_token,
            prefix_len=prefix,
            kv_caches=None,
            block_size=block_size,
            mask_id=MASK_ID,
            max_tokens=max_tokens,
            threshold=0.9,
            temperature=0.0,
            max_denoise_steps=0,  # -> block_size
            eos_token_id=(),
        )
        tokens = res.tokens
        assert len(tokens) == max_tokens
        # Multiple blocks were needed (block_size=4, max_tokens=10 -> 3 blocks).
        assert res.num_blocks == 3
        # The stream is fully contiguous across block boundaries: token 0 is the
        # seed, and every subsequent token follows the shifted convention
        # token[k] == g[prefix + k - 1]. Contiguity AT k == block_size proves the
        # next block was seeded with the previous block's next_seed.
        assert tokens[0] == first_token
        for k in range(1, len(tokens)):
            assert tokens[k] == int(g[prefix + k - 1]), f"mismatch at {k}"
        # Explicit boundary check: block 2's first token (index block_size).
        assert tokens[block_size] == int(g[prefix + block_size - 1])
        assert res.hit_eos is False

    def test_max_tokens_truncation(self):
        block_size = 4
        g = _target_array(256)
        fwd = _make_stub_forward(g, logit_scale=30.0)
        for max_tokens in (1, 3, 5, 7):
            res = diffusion_decode(
                fwd,
                first_token=8,
                prefix_len=2,
                kv_caches=None,
                block_size=block_size,
                mask_id=MASK_ID,
                max_tokens=max_tokens,
                eos_token_id=(),
            )
            assert len(res.tokens) == max_tokens

    def test_eos_truncates_stream_inclusive(self):
        block_size = 4
        prefix = 2
        first_token = 8
        g = _target_array(256)
        # Plant an EOS at a known absolute position; it surfaces at stream index
        # k where (prefix + k - 1) == that position.
        eos_pos = prefix + 5
        g[eos_pos] = EOS_ID
        fwd = _make_stub_forward(g, logit_scale=30.0)

        res = diffusion_decode(
            fwd,
            first_token=first_token,
            prefix_len=prefix,
            kv_caches=None,
            block_size=block_size,
            mask_id=MASK_ID,
            max_tokens=100,
            eos_token_id=(EOS_ID, ),
        )
        assert res.hit_eos is True
        # EOS at stream index 6 (k with prefix + k - 1 == prefix + 5 -> k = 6).
        assert res.tokens[-1] == EOS_ID
        assert len(res.tokens) == 7
        assert EOS_ID not in res.tokens[:-1]

    def test_disabled_is_noop_zero_tokens(self):
        # max_tokens == 0 -> the loop never runs (the runner-level gate keeps the
        # whole path off when enable_diffusion_decode is False; see AST tests).
        g = _target_array(64)
        fwd = _make_stub_forward(g, logit_scale=30.0)
        res = diffusion_decode(
            fwd,
            first_token=3,
            prefix_len=0,
            kv_caches=None,
            block_size=4,
            mask_id=MASK_ID,
            max_tokens=0,
            eos_token_id=(),
        )
        assert res.tokens == []
        assert res.num_blocks == 0


class TestOnDeviceLoopStructure:
    """AST checks that the denoise loop is on-device and reuses the foundation."""

    def test_denoise_uses_lax_while_loop(self):
        src = _DIFFUSION_DECODE.read_text()
        assert "jax.lax.while_loop" in src

    def test_reuses_diffusion_commit(self):
        src = _DIFFUSION_DECODE.read_text()
        assert ("from tpu_inference.layers.jax.sample.sampling import "
                "diffusion_commit") in src
        assert "diffusion_commit(" in src


class TestRunnerDispatchGating:
    """AST checks for the gated runner dispatch (path is a no-op when off)."""

    def _runner_ast(self):
        return ast.parse(_TPU_RUNNER.read_text())

    def _find_method(self, module, cls, name):
        for node in module.body:
            if isinstance(node, ast.ClassDef):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.FunctionDef) and sub.name == name:
                        return sub
        return None

    def test_execute_model_has_gated_diffusion_branch(self):
        src = _TPU_RUNNER.read_text()
        # Gated exactly on the (default-False) enable_diffusion_decode flag, so
        # AR decode is untouched unless the flag is explicitly enabled.
        assert "is_decode_only and self.enable_diffusion_decode" in src
        assert ("return self._execute_diffusion_decode(scheduler_output)"
                in src)

    def test_execute_diffusion_decode_method_exists(self):
        module = self._runner_ast()
        method = self._find_method(module, "TPUModelRunner",
                                   "_execute_diffusion_decode")
        assert method is not None, "_execute_diffusion_decode not defined"

    def test_mask_id_not_hardcoded_to_151665(self):
        # The Fast_dLLM_Qwen release uses 151669; 151665 is the wrong constant.
        src = _TPU_RUNNER.read_text()
        assert "151665" not in src
        # mask id is read from config, not hardcoded.
        method = self._find_method(self._runner_ast(), "TPUModelRunner",
                                   "_get_diffusion_mask_id")
        assert method is not None, "_get_diffusion_mask_id not defined"
