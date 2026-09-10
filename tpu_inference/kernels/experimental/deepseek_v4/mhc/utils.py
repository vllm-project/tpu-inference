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


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

    Shared by the Pallas ``pre_kernel`` and the fused seam op.
    Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
    """
    num_tokens = mixes.shape[0]

    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (mixes[:, hc_mult:2 * hc_mult] * hc_scale[1] +
                   hc_base[hc_mult:2 * hc_mult])
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = (
        mixes[:, 2 * hc_mult:].reshape(num_tokens, hc_mult, hc_mult) *
        hc_scale[2] + hc_base[2 * hc_mult:].reshape(1, hc_mult, hc_mult))
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) +
                           hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-1, keepdims=True) +
                               hc_sinkhorn_eps)
        comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) +
                               hc_sinkhorn_eps)
    return pre_mix, post_mix, comb_mix
