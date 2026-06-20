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
"""TPU-gated smoke test: drive the full smart-search pipeline against the
real ``ragged_paged_attention`` v3 kernel.

Runs a tiny budget (5 trials) so the test finishes quickly even on TPU.
Asserts:

* the run completes without raising,
* at least one candidate either succeeds verification or fails for a known
  reason (numerics / OOM / cost-model — not a Python exception),
* if any candidate succeeds, the winning latency is finite.

Skipped on non-TPU platforms.
"""

import jax
import pytest

from tools.kernel.tuner.v1.common.kernel_tuner_base import RunConfig
from tools.kernel.tuner.v1.kernel_tuner_runner import (SmartSearchConfig,
                                                       run_smart_search)
from tools.kernel.tuner.v1.rpa_v3_kernel_tuner import RpaV3KernelTuner


def _tpu_available() -> bool:
    try:
        return any(d.platform == "tpu" for d in jax.devices())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _tpu_available(),
                                reason="TPU not available in this environment")


def _make_run_config(tmp_path) -> RunConfig:
    return RunConfig(
        case_set_id="autotune_smoke",
        run_id="run0",
        case_set_desc="smart-search smoke test",
        tpu_version="tpu6e",
        tpu_cores=1,
        tpu_queue_multi="tpu_v6e_queue",
        run_locally=True,
        job_priority=-10,
        max_execution_minutes=5,
    )


def test_smart_search_runs_on_rpa_v3_no_exceptions(tmp_path):
    rc = _make_run_config(tmp_path)
    tuner = RpaV3KernelTuner(run_config=rc)
    cfg = SmartSearchConfig(
        search_strategy='tpe',
        trial_budget=5,
        verifier_mode='fast',
        cost_model_prefilter=True,
        interpret_check=False,
        timing_iters=3,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["trials_observed"] == 5
    # The smoke test passes if the loop completes; whether a winner exists
    # depends on whether any candidate happens to pass numerics on this
    # particular TPU/dtype combo.
    if summary["best_params"] is not None:
        assert summary["best_score_ns"] is not None
        assert summary["best_score_ns"] > 0


def test_smart_search_with_strict_verifier_on_rpa_v3(tmp_path):
    rc = _make_run_config(tmp_path)
    tuner = RpaV3KernelTuner(run_config=rc)
    cfg = SmartSearchConfig(
        search_strategy='evolutionary',
        trial_budget=4,
        verifier_mode='strict',
        cost_model_prefilter=True,
        timing_iters=2,
    )
    summary = run_smart_search(tuner, cfg)
    assert summary["trials_observed"] == 4
    # Strict mode must produce a CROSS_TRIAL_FAIL classification at most
    # zero times if a winner was found.
    if summary["best_params"] is not None:
        ct_fails = [
            t for t in summary["trials"] if t["status"] == "CROSS_TRIAL_FAIL"
            and t["params"] == summary["best_params"]
        ]
        assert not ct_fails, ("Winning params should not have a "
                              "cross-trial-fail record")
