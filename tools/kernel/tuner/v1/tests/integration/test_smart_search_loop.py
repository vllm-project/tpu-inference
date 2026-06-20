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
"""End-to-end test of ``run_smart_search`` with a synthetic kernel.

Exercises the full composition — strategy → cost-model → run → verifier →
anti-cheat — without requiring TPU hardware. The synthetic kernel returns
its input ``q`` scaled by a function of ``params``; the oracle returns the
same function. The optimum is a unique ``params`` configuration; the test
asserts the strategy finds it within budget AND that verifier failures
short-circuit candidates so they don't shadow the real winner.

This is the CPU-runnable safety net. The TPU-only RPA v3 smoke test
(``test_rpa_v3_autotune_smoke.py``) covers the real-kernel wire-up.
"""

import dataclasses
import math
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.bench.cost_estimate import CostEstimate, CostModel
from tools.kernel.tuner.v1.common.kernel_tuner_base import (RunResult,
                                                            TuningStatus)
from tools.kernel.tuner.v1.kernel_tuner_runner import (SmartSearchConfig,
                                                       run_smart_search)
from tools.kernel.tuner.v1.search.strategy import IntChoice
from tools.kernel.tuner.v1.verifier.numerics import NumericsReport, check_many


@dataclasses.dataclass
class _SyntheticTuningKey:
    seq_len: int
    head_dim: int


_OPTIMUM = {"bq_sz": 64, "bkv_sz": 512, "bq_csz": 32, "bkv_csz": 512}


def _quadratic_latency(params: dict[str, Any]) -> int:
    # Lower-is-better; minimum at _OPTIMUM. Add a base so even the optimum
    # has a non-zero latency the verifier can see.
    base_ns = 1_000_000  # 1 ms
    penalty = sum((math.log2(params[k] / _OPTIMUM[k]))**2 for k in _OPTIMUM)
    return int(base_ns * (1 + 5 * penalty))


class _SyntheticOracle:

    def compute(self, inputs):
        # Reference: q * 2 (matches the synthetic kernel)
        return inputs["q"] * 2.0

    def dtype_tolerance(self, dtype):
        return 1e-4, 1e-4


class _FakeKernelTuner:
    """Mimics enough of ``KernelTunerBase`` for ``run_smart_search``.

    Drives every code path in the smart-search loop:
        * cost_model rejects bkv_sz < 128 (infeasible)
        * runs the "kernel" (a pure-Python multiply) and reports latency
        * verifier compares against the oracle
        * ``buggy_constant_for_bq_sz`` returns a constant output (anti-cheat
          fires, but numerics catches it first because constant != reference)
        * ``return_shadow_for_bq_sz`` returns an input verbatim that happens
          to equal the reference (numerics passes, anti-cheat fires)
        * ``wrong_scale_for_bq_csz`` returns 0.5*q (numerics fails)
    """

    class _TunerCfg:
        kernel_tuner_name = "fake_kernel_tuner"

    def __init__(
        self,
        *,
        buggy_constant_for_bq_sz: int | None = None,
        return_shadow_for_bq_sz: int | None = None,
        wrong_scale_for_bq_csz: int | None = None,
    ):
        self.tuner_config = _FakeKernelTuner._TunerCfg()
        # Hooks read by ``_try_second_seed_inputs``; we only need them to be
        # attributes for hasattr to pass.
        self._tuning_key = None
        self._kernel_inputs_cache = {}
        self._inputs_seed_used: list[int] = []
        self._buggy_constant_for_bq_sz = buggy_constant_for_bq_sz
        self._return_shadow_for_bq_sz = return_shadow_for_bq_sz
        self._wrong_scale_for_bq_csz = wrong_scale_for_bq_csz

    # ---- Smart-search hooks -------------------------------------------------
    def get_default_tuning_key(self):
        return _SyntheticTuningKey(seq_len=128, head_dim=64)

    def get_search_space(self):
        return {
            "bq_sz": IntChoice("bq_sz", options=[16, 32, 64, 128]),
            "bkv_sz": IntChoice("bkv_sz", options=[64, 128, 256, 512, 1024]),
            "bq_csz": IntChoice("bq_csz", options=[8, 16, 32, 64]),
            "bkv_csz": IntChoice("bkv_csz", options=[128, 256, 512, 1024]),
        }

    def get_oracle(self):
        return _SyntheticOracle()

    def get_cost_model(self):

        def _estimate(_tk, params):
            if params["bkv_sz"] < 128:
                return CostEstimate(reason=f"bkv_sz={params['bkv_sz']} < 128")
            return CostEstimate(vmem_bytes=1024 * params["bq_sz"])

        return CostModel(_estimate)

    def generate_inputs(self, tuning_key):
        if self._tuning_key is None:
            seed = int(np.random.randint(0, 2**31 - 1))
            self._inputs_seed_used.append(seed)
            self._tuning_key = tuning_key
            rng = np.random.default_rng(seed)
            q = jnp.asarray(
                rng.normal(0,
                           1,
                           size=(tuning_key.seq_len,
                                 tuning_key.head_dim)).astype(np.float32))
            inputs = {"q": q}
            # Only seed "shadow" when the test wants to exercise the
            # input-aliasing anti-cheat path; otherwise the correct kernel
            # output (q*2) would always match shadow and get rejected.
            if self._return_shadow_for_bq_sz is not None:
                inputs["shadow"] = q * 2.0
            self._kernel_inputs_cache = inputs
        return self._kernel_inputs_cache

    def build_kernel_fn(self, tuning_key, params, inputs):
        bq_sz = params["bq_sz"]
        bq_csz = params["bq_csz"]
        q = inputs["q"]
        buggy = self._buggy_constant_for_bq_sz
        shadow = self._return_shadow_for_bq_sz
        wrong = self._wrong_scale_for_bq_csz

        def fn():
            if buggy is not None and bq_sz == buggy:
                return jnp.full_like(q, 0.5)  # fails numerics
            if shadow is not None and bq_sz == shadow:
                return inputs["shadow"]  # passes numerics, trips anti-cheat
            if wrong is not None and bq_csz == wrong:
                return q * 0.5  # fails numerics
            return q * 2.0

        return fn

    def run_with_outputs(self, tuning_key, params, iters):
        inputs = self.generate_inputs(tuning_key)
        fn = self.build_kernel_fn(tuning_key, params, inputs)
        out = fn()
        latency = _quadratic_latency(params)
        # Sleep tiny amount so wall-clock is finite but bounded.
        time.sleep(0.001)
        return RunResult(
            status=TuningStatus.SUCCESS,
            avg_latency_ns=float(latency),
            total_latency_ns=float(latency * iters),
            output=out,
            bench=None,
        )

    def verify(self, tuning_key, params, output, *, inputs=None):
        if output is None or inputs is None:
            return NumericsReport(True, 0.0, 1.0, 0, 0)
        oracle = self.get_oracle()
        ref = oracle.compute(inputs)
        atol, rtol = oracle.dtype_tolerance(output.dtype)
        return check_many((output, ), (ref, ), atol=atol, rtol=rtol)


def test_smart_search_finds_near_optimum_with_tpe():
    """TPE with budget 60 should find within ~1-step-of-optimum on each axis
    of a 320-config space — a 6-7x improvement on random's ~30M expected
    latency."""
    tuner = _FakeKernelTuner()
    cfg = SmartSearchConfig(
        search_strategy='tpe',
        trial_budget=60,
        verifier_mode='fast',
        cost_model_prefilter=True,
        interpret_check=False,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["best_params"] is not None
    assert summary["best_score_ns"] is not None
    # Optimum = 1M; 1-step-off on one axis ≈ 6M; 2-steps-off ≈ 11M.
    # TPE with 60 trials and 10 startup should hit ≤ 1-step-off reliably.
    assert summary["best_score_ns"] <= 7_500_000


def test_smart_search_finds_optimum_with_evolutionary():
    """EA with budget 200 explores ~63% of a 320-config space; it should
    find the exact optimum or one step off."""
    tuner = _FakeKernelTuner()
    cfg = SmartSearchConfig(
        search_strategy='evolutionary',
        trial_budget=200,
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["best_params"] is not None
    # Allow up to one-step-off on the worst axis (penalty <= 1.0 → 6M).
    assert summary["best_score_ns"] <= 6_100_000


def test_smart_search_grid_terminates_on_space_exhaustion():
    tuner = _FakeKernelTuner()
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=10_000,  # bigger than the space (4*5*4*4 = 320)
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["best_params"] == _OPTIMUM
    # 320 enumerable; cost-model rejects bkv_sz<128 → 64 of them; remaining
    # ~256 actually evaluate. Plus the rejected ones are still counted as
    # observed trials (with score=inf), so total observed should be 320.
    assert summary["trials_observed"] == 320


def test_smart_search_rejects_buggy_constant_candidates():
    """Constant output fails numerics (constant != q*2 reference)."""
    # Mark bq_sz=64 (the optimum) as the buggy branch; the winner must
    # therefore be a different bq_sz with worse latency.
    tuner = _FakeKernelTuner(buggy_constant_for_bq_sz=64)
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=10_000,
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["best_params"] is not None
    assert summary["best_params"]["bq_sz"] != 64
    # The buggy bq_sz must be rejected via verifier (numerics fires first;
    # ANTI_CHEAT_FAIL would also be acceptable if the order ever reverses).
    rejected_buggy = [
        t for t in summary["trials"]
        if t.get("params", {}).get("bq_sz") == 64 and t["status"] in (
            "NUMERICS_FAIL", "ANTI_CHEAT_FAIL")
    ]
    assert rejected_buggy, "buggy bq_sz=64 candidates were never rejected"


def test_smart_search_routes_to_anti_cheat_path():
    """When a kernel's output bytewise-equals an input that also matches the
    reference, the runner must reach the anti-cheat detector and emit
    ``ANTI_CHEAT_FAIL``. (The detector itself is unit-tested separately —
    here we're proving the runner *reaches* it.)

    Note: because JAX's deterministic compute makes ``q*2`` and a stored
    ``shadow = q*2`` bytewise-identical, normal candidates *also* trip the
    aliasing detector once ``shadow`` is added to the inputs dict. That is
    correct behavior in this synthetic — every candidate aliases ``shadow``.
    The integration assertion is therefore: ``ANTI_CHEAT_FAIL`` must appear
    in the trial log when ``shadow`` is in the input dict and not skipped.
    """
    tuner = _FakeKernelTuner(return_shadow_for_bq_sz=64)
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=10_000,
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
        anti_cheat_input_skip_keys=(),
    )
    summary = run_smart_search(tuner, cfg)
    statuses = [t["status"] for t in summary["trials"]]
    assert "ANTI_CHEAT_FAIL" in statuses, (
        "anti-cheat path not reached; observed statuses: "
        f"{sorted(set(statuses))}")
    # If skip_keys re-introduces shadow, the runner must let the wiring fall
    # through to a winner. Re-run with shadow skipped — expect a non-None
    # winner (any params), proving the runner's verifier-then-anti-cheat
    # order doesn't silently strand a healthy candidate.
    cfg_skipped = dataclasses.replace(cfg,
                                      anti_cheat_input_skip_keys=("shadow", ))
    fresh_tuner = _FakeKernelTuner(return_shadow_for_bq_sz=64)
    summary2 = run_smart_search(fresh_tuner, cfg_skipped)
    assert summary2["best_params"] is not None


def test_smart_search_rejects_wrong_scale_candidates():
    # Mark bq_csz=32 (the optimum) as a wrong-scale kernel; verifier must
    # reject it, so the winner has a different bq_csz.
    tuner = _FakeKernelTuner(wrong_scale_for_bq_csz=32)
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=10_000,
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["best_params"] is not None
    assert summary["best_params"]["bq_csz"] != 32
    statuses = [t["status"] for t in summary["trials"]]
    assert "NUMERICS_FAIL" in statuses


def test_smart_search_cost_model_prefilters_infeasible_params():
    tuner = _FakeKernelTuner()
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=10_000,
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    statuses = [t["status"] for t in summary["trials"]]
    skipped = [
        t for t in summary["trials"] if t["status"] == "COST_MODEL_SKIP"
    ]
    assert "COST_MODEL_SKIP" in statuses
    # Every skip should be a bkv_sz<128 candidate (the cost model's rule).
    assert all(t["params"]["bkv_sz"] < 128 for t in skipped)


def test_smart_search_returns_no_winner_when_all_fail():
    # Cost model rejects everything by setting bkv_sz < 128 threshold above
    # all options. We accomplish this by forcing the cost-model path to
    # reject the entire grid.
    tuner = _FakeKernelTuner()

    def _all_infeasible(*_args):
        return CostEstimate(reason="forced infeasible (test)")

    tuner.get_cost_model = lambda: CostModel(_all_infeasible)
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=100,
        verifier_mode='fast',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["best_params"] is None
    assert summary["best_score_ns"] is None


def test_smart_search_strict_mode_runs_second_seed():
    tuner = _FakeKernelTuner()
    cfg = SmartSearchConfig(
        search_strategy='grid',
        trial_budget=10,
        verifier_mode='strict',
        cost_model_prefilter=True,
        timing_iters=2,
        second_seed=99,
    )
    summary = run_smart_search(tuner, cfg)
    # Strict mode requests two seeded input draws (one default, one with the
    # configured second_seed). Both should appear in the seed log.
    assert len(tuner._inputs_seed_used) >= 2
    assert summary["verifier_mode"] == "strict"
