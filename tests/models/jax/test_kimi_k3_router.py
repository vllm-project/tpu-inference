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
"""Kimi-K3 MoE routing parity against the reference gate.

Kimi-K3 routes with the DeepSeek-style noaux_tc gate but a single expert
group (`num_expert_group=1`), sigmoid scoring, renormalized top-16 weights and
`routed_scaling_factor=1.0` (HF `moonshotai/Kimi-K3` config + the
`KimiMoEGate` implementation in its modeling_kimi_linear.py).

The reference gate is transcribed here in numpy; the tests check the shared
router reproduces both WHICH experts are selected and the weights they get,
using a real K3-shaped checkpoint and the golden activations captured from
the reference implementation when available.

Set `K3_TINY_CKPT` to a K3-tiny checkpoint directory (with goldens_run1.npz)
to run the checkpoint-backed cases; they skip when it is unset.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.models.jax.deepseek_v3 import DeepSeekV3Router

CKPT = os.environ.get("K3_TINY_CKPT", "")
# K3-tiny routed-MoE config (mirrors Kimi-K3 with smaller dims).
NUM_EXPERTS, TOPK, HIDDEN = 32, 4, 1024
ROUTED_SCALING = 1.0


def reference_gate(x,
                   weight,
                   bias,
                   topk=TOPK,
                   renormalize=True,
                   routed_scaling_factor=ROUTED_SCALING):
    """KimiMoEGate from HF moonshotai/Kimi-K3, n_group == 1 path, fp32.

    Selection uses scores + bias; the returned weights come from the
    *un-biased* scores and are renormalized to sum to 1.
    """
    logits = x.astype(np.float32) @ weight.astype(np.float32).T
    scores = 1.0 / (1.0 + np.exp(-logits))
    scores_for_choice = scores + bias.astype(np.float32)[None, :]
    # num_expert_group == 1 -> no group masking, plain top-k.
    topk_idx = np.argsort(-scores_for_choice, axis=-1, kind="stable")[:, :topk]
    topk_weight = np.take_along_axis(scores, topk_idx, axis=-1)
    if topk > 1 and renormalize:
        topk_weight = topk_weight / (topk_weight.sum(-1, keepdims=True) +
                                     1e-20)
    return topk_idx, topk_weight * routed_scaling_factor


@pytest.fixture(scope="module")
def mesh():
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, axis_names=MESH_AXIS_NAMES)


def _build_router(mesh, num_experts=NUM_EXPERTS, hidden=HIDDEN, topk=TOPK):
    """K3's routing config on the shared DeepSeek-style router.

    `moe_backend` is deliberately an unfused backend: the fused backends take
    raw logits and do selection themselves, which is where the noaux_tc bias
    gets dropped. K3 keeps routing eager so the bias is applied here.
    """
    return DeepSeekV3Router(
        hidden_size=hidden,
        num_experts=num_experts,
        num_experts_per_tok=topk,
        n_groups=1,
        topk_groups=1,
        norm_topk_prob=True,
        routed_scaling_factor=ROUTED_SCALING,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
        activation_ffw_td=P(),
        ed_sharding=P(),
        e_sharding=P(),
        scoring_func="sigmoid",
        moe_backend=MoEBackend.DENSE_MAT,
    )


def _load_gate_weights(layer=1):
    from safetensors import safe_open
    pre = f"model.layers.{layer}.block_sparse_moe.gate."
    with safe_open(os.path.join(CKPT, "model.safetensors"), "np") as f:
        w = f.get_tensor(pre + "weight").astype(np.float32)
        b = f.get_tensor(pre + "e_score_correction_bias").astype(np.float32)
    return w, b


def _run_router(mesh, router, x):
    # The router constrains its input's sharding, which requires an
    # explicitly-placed array rather than a bare device array.
    with jax.set_mesh(mesh):
        x_dev = jax.device_put(jnp.asarray(x, jnp.float32),
                               NamedSharding(mesh, P()))
        weights, idx = router(x_dev)
    return np.asarray(idx), np.asarray(weights)


def _assert_matches(idx_jax, w_jax, idx_ref, w_ref, label):
    """Expert *sets* must match exactly; weights must match numerically.

    Top-k tie ordering is unspecified, so compare sorted sets and compare
    weights after sorting each row by expert id.
    """
    np.testing.assert_array_equal(np.sort(idx_jax, -1), np.sort(idx_ref, -1))
    order_jax = np.argsort(idx_jax, -1)
    order_ref = np.argsort(idx_ref, -1)
    w_jax_sorted = np.take_along_axis(w_jax, order_jax, -1)
    w_ref_sorted = np.take_along_axis(w_ref, order_ref, -1)
    err = np.abs(w_jax_sorted - w_ref_sorted).max()
    print(f"[observed] {label}: weight max_abs={err:.3e}")
    np.testing.assert_allclose(w_jax_sorted,
                               w_ref_sorted,
                               atol=1e-5,
                               rtol=1e-5)


def test_router_matches_reference_gate_random_inputs(mesh):
    """Random weights/inputs: selection and weights must match the
    reference gate, with a bias large enough that ignoring it would show."""
    rng = np.random.default_rng(0)
    router = _build_router(mesh)
    weight = rng.standard_normal(
        (NUM_EXPERTS, HIDDEN), dtype=np.float32) * 0.05
    bias = rng.standard_normal(NUM_EXPERTS, dtype=np.float32) * 0.1
    with jax.set_mesh(mesh):
        router.weight.value = jnp.asarray(weight.T)
        router.e_score_correction_bias.value = jnp.asarray(bias)
    x = rng.standard_normal((64, HIDDEN), dtype=np.float32) * 0.1

    idx_ref, w_ref = reference_gate(x, weight, bias)
    idx_jax, w_jax = _run_router(mesh, router, x)
    _assert_matches(idx_jax, w_jax, idx_ref, w_ref, "random")


def test_gate_logits_are_computed_in_full_precision(mesh):
    """The gate matmul itself must run in fp32, not a bf16 pass.

    The reference casts both operands to fp32 before the linear. On TPU an
    fp32 einsum at XLA's default precision is lowered to a single bf16
    multiply, which costs ~4 decimal digits on the logits -- enough to
    reorder experts whose scores are within that of each other, which shows
    up as an expert-set mismatch in the parity tests above. Assert the
    logits directly so the cause is readable when it regresses.
    """
    rng = np.random.default_rng(2)
    router = _build_router(mesh)
    weight = rng.standard_normal(
        (NUM_EXPERTS, HIDDEN), dtype=np.float32) * 0.05
    with jax.set_mesh(mesh):
        router.weight.value = jnp.asarray(weight.T)
    x = rng.standard_normal((64, HIDDEN), dtype=np.float32) * 0.1

    with jax.set_mesh(mesh):
        x_dev = jax.device_put(jnp.asarray(x, jnp.float32),
                               NamedSharding(mesh, P()))
        # The gate matmul on its own, before sigmoid/top-k.
        logits = np.asarray(JaxEinsum.__call__(router, x_dev))
    # fp64 reference for the same contraction.
    ref = x.astype(np.float64) @ weight.astype(np.float64).T
    err = np.abs(logits - ref).max()
    print(f"[observed] gate logit max_abs vs fp64: {err:.3e}")
    # A bf16-rounded contraction over 1024 terms lands around 1e-3 here;
    # fp32 lands near 1e-6. The gate sits between them by a wide margin.
    assert err < 1e-5, (
        f"router gate logits are off by {err:.3e} -- the gate matmul is not "
        "running in fp32 (check `precision` on DeepSeekV3Router)")


def test_ignoring_the_bias_would_fail_this_test(mesh):
    """Guard against a vacuous parity test: the same comparison against a
    reference computed WITHOUT the bias must disagree."""
    rng = np.random.default_rng(1)
    router = _build_router(mesh)
    weight = rng.standard_normal(
        (NUM_EXPERTS, HIDDEN), dtype=np.float32) * 0.05
    bias = rng.standard_normal(NUM_EXPERTS, dtype=np.float32) * 0.1
    with jax.set_mesh(mesh):
        router.weight.value = jnp.asarray(weight.T)
        router.e_score_correction_bias.value = jnp.asarray(bias)
    x = rng.standard_normal((64, HIDDEN), dtype=np.float32) * 0.1

    idx_jax, _ = _run_router(mesh, router, x)
    idx_nobias, _ = reference_gate(x, weight, np.zeros_like(bias))
    changed = (np.sort(idx_jax, -1) != np.sort(idx_nobias, -1)).any(-1)
    print(f"[observed] tokens whose selection depends on the bias: "
          f"{changed.mean():.1%}")
    assert changed.any(), (
        "bias had no observable effect, so the parity test above would pass "
        "even with the bias dropped")


@pytest.mark.skipif(not CKPT, reason="K3_TINY_CKPT unset")
@pytest.mark.parametrize("layer", [1, 4, 8])
def test_router_matches_reference_on_checkpoint_and_golden_inputs(mesh, layer):
    """Real K3-shaped gate weights, driven by the activations the reference
    implementation actually saw at that layer."""
    z = np.load(os.path.join(CKPT, "goldens_run1.npz"))
    weight, bias = _load_gate_weights(layer)
    # Input to the MoE block, captured from the reference forward.
    x = z[f"p48.layers.{layer}.block_sparse_moe.in"][0].astype(np.float32)

    router = _build_router(mesh)
    with jax.set_mesh(mesh):
        router.weight.value = jnp.asarray(weight.T)
        router.e_score_correction_bias.value = jnp.asarray(bias)

    idx_ref, w_ref = reference_gate(x, weight, bias)
    idx_jax, w_jax = _run_router(mesh, router, x)
    _assert_matches(idx_jax, w_jax, idx_ref, w_ref, f"ckpt L{layer}")
    assert idx_jax.shape == (x.shape[0], TOPK)


@pytest.mark.skipif(not CKPT, reason="K3_TINY_CKPT unset")
def test_weights_sum_to_one_and_scaling_is_identity(mesh):
    """K3 renormalizes the top-k weights and uses routed_scaling_factor=1.0,
    so each token's routing weights sum to 1."""
    z = np.load(os.path.join(CKPT, "goldens_run1.npz"))
    weight, bias = _load_gate_weights(1)
    x = z["p48.layers.1.block_sparse_moe.in"][0].astype(np.float32)
    router = _build_router(mesh)
    with jax.set_mesh(mesh):
        router.weight.value = jnp.asarray(weight.T)
        router.e_score_correction_bias.value = jnp.asarray(bias)
    _, w_jax = _run_router(mesh, router, x)
    sums = w_jax.sum(-1)
    print(f"[observed] weight-sum deviation from 1: "
          f"{np.abs(sums - 1).max():.3e}")
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
