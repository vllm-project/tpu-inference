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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.layernorm.fused_gemma_rmsnorm import (
    DEFAULT_BLOCK_ROWS, fused_add_gemma_rms_norm, supports)


def _gemma_rmsnorm_ref(x, residual, weight, eps):
    s = x.astype(jnp.float32)
    if residual is not None:
        s = s + residual.astype(jnp.float32)
    var = jnp.mean(s * s, axis=-1, keepdims=True)
    y = (s * jax.lax.rsqrt(var + eps) *
         (weight.astype(jnp.float32) + 1.0)).astype(x.dtype)
    if residual is not None:
        return y, s.astype(x.dtype)
    return y


def test_supports_bounds():
    assert supports(512, 5120, DEFAULT_BLOCK_ROWS, min_rows=512)
    assert supports(1024, 5120, DEFAULT_BLOCK_ROWS, min_rows=512)
    assert not supports(128, 5120, DEFAULT_BLOCK_ROWS, min_rows=512)
    assert not supports(512, 5100, DEFAULT_BLOCK_ROWS, min_rows=512)
    assert not supports(512, 32768, DEFAULT_BLOCK_ROWS, min_rows=512)


@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("rows,hidden", [(512, 512), (1024, 5120)])
def test_fused_add_gemma_rms_norm_parity(has_residual, rows, hidden):
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas TPU kernel requires TPU backend")
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k1, (rows, hidden), dtype=jnp.bfloat16)
    r = (jax.random.normal(k2, (rows, hidden), dtype=jnp.bfloat16)
         if has_residual else None)
    w = jax.random.normal(k3, (hidden, ), dtype=jnp.bfloat16) * 0.1
    eps = 1e-6

    out = fused_add_gemma_rms_norm(x, r, w, eps=eps)
    ref = _gemma_rmsnorm_ref(x, r, w, eps)
    if has_residual:
        np.testing.assert_allclose(
            np.asarray(out[0], dtype=np.float32),
            np.asarray(ref[0], dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(out[1], dtype=np.float32),
            np.asarray(ref[1], dtype=np.float32),
            atol=0.0,
            rtol=0.0,
        )
    else:
        np.testing.assert_allclose(
            np.asarray(out, dtype=np.float32),
            np.asarray(ref, dtype=np.float32),
            atol=1e-2,
            rtol=1e-2,
        )
