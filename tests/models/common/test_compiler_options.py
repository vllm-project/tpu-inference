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

import pytest

from tpu_inference.models.common.compiler_options import \
    get_step_fn_compiler_options

_RAGGED_A2A_BARRIER = "xla_tpu_force_custom_barrier_for_ragged_all_to_all"


def test_step_fn_options_for_moe_ragged_a2a(monkeypatch: pytest.MonkeyPatch):
    """MOE_GMM_EP_RAGGED_A2A turns off the ragged all-to-all custom barrier,
    which the prefill programs need to compile."""
    # Keep the SparseCore offload threshold off TPU-info lookups.
    monkeypatch.setenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "0")

    monkeypatch.delenv("MOE_GMM_EP_RAGGED_A2A", raising=False)
    assert _RAGGED_A2A_BARRIER not in get_step_fn_compiler_options()

    monkeypatch.setenv("MOE_GMM_EP_RAGGED_A2A", "1")
    assert get_step_fn_compiler_options()[_RAGGED_A2A_BARRIER] == "false"
