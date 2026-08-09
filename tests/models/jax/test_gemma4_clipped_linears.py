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
"""In-tree tests for Gemma-4 vision clipped-linears (PR #3355).

These replace the previously private R4 validation suite. They exercise the
serving-side clip-bound contract WITHOUT requiring a gated HF checkpoint:

  1. flag OFF is an exact no-op (clipped einsum == plain einsum)
  2. one HF-parity projection: clamp-in -> linear -> clamp-out matches a numpy oracle
  3. a finite, ordered bound set validates OK
  4. a missing (NaN-sentinel) bound -> validate hard-fails
  5. an Inf bound -> hard-fails
  6. a non-scalar bound -> hard-fails
  7. min > max (empty interval) -> hard-fails
  8. q and k carry DISTINCT bounds (not accidentally shared)
  9. the walker discovers EXACTLY num_blocks*7 modules; ZERO/partial -> hard-fail
 10. 448-bound accounting for a 16-block tower; audio (480 keys) is out of scope

Synthetic tiny modules — fast, no network, CPU-only.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tpu_inference.models.jax.gemma4_mm import Gemma4ClippableEinsum


def _rngs():
    return nnx.Rngs(params=0)


def _set_bounds(m, imin, imax, omin, omax):
    m.input_min = nnx.Param(jnp.asarray(imin, dtype=jnp.float32))
    m.input_max = nnx.Param(jnp.asarray(imax, dtype=jnp.float32))
    m.output_min = nnx.Param(jnp.asarray(omin, dtype=jnp.float32))
    m.output_max = nnx.Param(jnp.asarray(omax, dtype=jnp.float32))


def _make_einsum(use_clipped):
    # y[t,o] = x[t,d] * W[d,o]
    return Gemma4ClippableEinsum(
        "td,do->to", (4, 3), use_clipped_linears=use_clipped, rngs=_rngs())


# ---------------------------------------------------------------------------
# 1. flag OFF exact no-op
# ---------------------------------------------------------------------------
def test_flag_off_is_exact_noop():
    m = _make_einsum(use_clipped=False)
    x = jnp.asarray(np.random.RandomState(0).randn(2, 4), dtype=jnp.float32)
    # OFF: no bound state is created and forward == plain einsum.
    assert not m.use_clipped_linears
    assert not hasattr(m, "input_min")
    y = m(x)
    W = m.weight.value
    expected = jnp.einsum("td,do->to", x, W)
    np.testing.assert_allclose(np.asarray(y), np.asarray(expected), rtol=1e-6, atol=1e-6)
    # validate_clip_bounds is a no-op when off.
    m.validate_clip_bounds()


# ---------------------------------------------------------------------------
# 2. one HF-parity projection: clamp-in -> linear -> clamp-out
# ---------------------------------------------------------------------------
def test_clip_forward_matches_oracle():
    m = _make_einsum(use_clipped=True)
    _set_bounds(m, -0.5, 0.5, -1.0, 1.0)
    x = jnp.asarray(np.random.RandomState(1).randn(3, 4) * 3.0, dtype=jnp.float32)
    y = m(x)
    W = np.asarray(m.weight.value)
    xin = np.clip(np.asarray(x), -0.5, 0.5)
    yout = np.clip(np.einsum("td,do->to", xin, W), -1.0, 1.0)
    np.testing.assert_allclose(np.asarray(y), yout, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. finite ordered bounds validate OK
# ---------------------------------------------------------------------------
def test_valid_bounds_pass():
    m = _make_einsum(use_clipped=True)
    _set_bounds(m, -2.0, 3.0, -1.0, 1.0)
    m.validate_clip_bounds()  # must not raise


# ---------------------------------------------------------------------------
# 4. missing (NaN-sentinel) bound -> fail
# ---------------------------------------------------------------------------
def test_nan_sentinel_bound_fails():
    m = _make_einsum(use_clipped=True)
    # Leave bounds at the NaN sentinel from __init__.
    with pytest.raises(ValueError, match="non-finite"):
        m.validate_clip_bounds()


# ---------------------------------------------------------------------------
# 5. Inf bound -> fail
# ---------------------------------------------------------------------------
def test_inf_bound_fails():
    m = _make_einsum(use_clipped=True)
    _set_bounds(m, -np.inf, 0.5, -1.0, 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        m.validate_clip_bounds()


# ---------------------------------------------------------------------------
# 6. non-scalar bound -> fail
# ---------------------------------------------------------------------------
def test_non_scalar_bound_fails():
    m = _make_einsum(use_clipped=True)
    _set_bounds(m, -0.5, 0.5, -1.0, 1.0)
    m.input_min = nnx.Param(jnp.asarray([-0.5, -0.6], dtype=jnp.float32))
    with pytest.raises(ValueError, match="non-scalar"):
        m.validate_clip_bounds()


# ---------------------------------------------------------------------------
# 7. min > max (empty interval) -> fail
# ---------------------------------------------------------------------------
def test_input_min_gt_max_fails():
    m = _make_einsum(use_clipped=True)
    _set_bounds(m, 0.5, -0.5, -1.0, 1.0)
    with pytest.raises(ValueError, match="input_min.*>.*input_max"):
        m.validate_clip_bounds()


def test_output_min_gt_max_fails():
    m = _make_einsum(use_clipped=True)
    _set_bounds(m, -0.5, 0.5, 1.0, -1.0)
    with pytest.raises(ValueError, match="output_min.*>.*output_max"):
        m.validate_clip_bounds()


# ---------------------------------------------------------------------------
# 8. q and k carry DISTINCT bounds (not accidentally shared)
# ---------------------------------------------------------------------------
def test_q_k_bounds_are_distinct():
    q = _make_einsum(use_clipped=True)
    k = _make_einsum(use_clipped=True)
    _set_bounds(q, -0.5, 0.5, -1.0, 1.0)
    _set_bounds(k, -0.7, 0.9, -1.5, 1.5)
    # Distinct Param objects and distinct values.
    assert q.input_min is not k.input_min
    assert float(q.input_min.value) != float(k.input_min.value)
    assert float(q.output_max.value) != float(k.output_max.value)
    q.validate_clip_bounds()
    k.validate_clip_bounds()


# ---------------------------------------------------------------------------
# 9 + 10. Walker: discovers EXACTLY num_blocks*7; zero/partial -> fail; audio out of scope.
# We build a synthetic tower whose structure mirrors the real one enough for the
# nnx.iter_graph walk, then invoke the SAME validation logic used post-load.
# ---------------------------------------------------------------------------
class _FakeVisionConfig:
    def __init__(self, num_hidden_layers, use_clipped_linears=True):
        self.num_hidden_layers = num_hidden_layers
        self.use_clipped_linears = use_clipped_linears


class _FakeBlock(nnx.Module):
    """7 clipped projections: q,k,v,o + gate,up,down."""

    def __init__(self, finite=True):
        for nm in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            e = _make_einsum(use_clipped=True)
            if finite:
                _set_bounds(e, -1.0, 1.0, -2.0, 2.0)
            setattr(self, nm, e)


class _FakeVisionTower(nnx.Module):
    def __init__(self, num_blocks, finite=True, use_clipped=True):
        self.config = _FakeVisionConfig(num_blocks, use_clipped)
        self.layers = [_FakeBlock(finite=finite) for _ in range(num_blocks)]


class _FakeModel(nnx.Module):
    def __init__(self, vt):
        self.vision_tower = vt


class _Harness(nnx.Module):
    """Exposes the real _validate_vision_clip_bounds via composition."""

    def __init__(self, num_blocks, finite=True, use_clipped=True, vt_none=False):
        if vt_none:
            self.model = _FakeModel(None)
        else:
            self.model = _FakeModel(_FakeVisionTower(num_blocks, finite, use_clipped))

    # Bind the real method under test.
    from tpu_inference.models.jax.gemma4_mm import Gemma4ForConditionalGeneration as _G
    _validate_vision_clip_bounds = _G._validate_vision_clip_bounds


def test_walker_counts_exactly_16x7_448():
    h = _Harness(num_blocks=16, finite=True)
    n = h._validate_vision_clip_bounds()
    assert n == 16 * 7  # 112 modules
    assert n * 4 == 448  # 448 scalar bounds


def test_walker_zero_modules_fails():
    # A tower with zero blocks -> expected_modules == 0 -> hard-fail (not a silent pass).
    h = _Harness(num_blocks=0, finite=True)
    with pytest.raises(ValueError, match="positive expected clip-module count"):
        h._validate_vision_clip_bounds()


def test_walker_partial_bounds_fails():
    # 16 blocks but one bound left at the NaN sentinel -> validate_clip_bounds raises.
    h = _Harness(num_blocks=2, finite=True)
    # Corrupt one projection's bound back to NaN.
    h.model.vision_tower.layers[0].q_proj.input_min = nnx.Param(
        jnp.asarray(jnp.nan, dtype=jnp.float32))
    with pytest.raises(ValueError, match="non-finite"):
        h._validate_vision_clip_bounds()


def test_walker_vision_tower_none_fails():
    h = _Harness(num_blocks=1, vt_none=True)
    with pytest.raises(ValueError, match="vision_tower is None"):
        h._validate_vision_clip_bounds()


def test_walker_clipping_off_is_noop():
    # use_clipped_linears False on the config -> validation is a no-op (returns None).
    h = _Harness(num_blocks=16, finite=True, use_clipped=False)
    assert h._validate_vision_clip_bounds() is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
