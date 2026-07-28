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
"""Gated activations that take both the gate and the up branch.

`FlaxUtils.ACT2FN` maps a single tensor to a single tensor, so it can only
express activations applied to the gate branch alone (``act(gate) * up``).
Activations that also transform the up branch live here.
"""

import jax
import jax.numpy as jnp

# Kimi-K3 config defaults (HF moonshotai/Kimi-K3 `activation_situ_beta` /
# `activation_situ_linear_beta`).
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def situ_and_mul(gate: jax.Array,
                 up: jax.Array,
                 beta: float = SITU_BETA,
                 linear_beta: float | None = SITU_LINEAR_BETA) -> jax.Array:
    """SiTU ("Sigmoid Tanh Unit"), used by every MLP in Kimi-K3.

        out = beta * tanh(gate / beta) * sigmoid(gate)
              * linear_beta * tanh(up / linear_beta)

    Both branches are soft-clamped: the gate branch saturates at +-beta and
    the up branch at +-linear_beta, which is what makes the activation safe
    for the QAT'd 4-bit expert weights.

    Computed in fp32 regardless of input dtype, matching the reference
    implementation (HF `moonshotai/Kimi-K3` modeling_kimi_linear.py
    ``SituAndMul``), then cast back to the input dtype.
    """
    out_dtype = gate.dtype
    gate_f32 = gate.astype(jnp.float32)
    up_f32 = up.astype(jnp.float32)

    situ = beta * jnp.tanh(gate_f32 / beta) * jax.nn.sigmoid(gate_f32)
    if linear_beta is not None:
        up_f32 = linear_beta * jnp.tanh(up_f32 / linear_beta)
    return (situ * up_f32).astype(out_dtype)
