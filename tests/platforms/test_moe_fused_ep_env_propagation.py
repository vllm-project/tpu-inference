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
"""The fused expert-parallel MoE settings reach the Ray workers.

The worker process is the one that builds the model, so a setting the
driver reads and the worker does not inverts what the switch promises: the
operator asks for the kernel and every worker quietly runs the general MoE
path instead. The roster is what the Ray executor copies, so every setting
the decision reads has to be on it.
"""

import inspect
import re

from tpu_inference import envs
from tpu_inference.layers.common import moe as moe_module
from tpu_inference.layers.common import moe_fused_ep
from tpu_inference.platforms.tpu_platform import TpuPlatform

# The switch and its threshold, named so that a rename which misses the
# roster leaves a name nothing reads on it.
FUSED_EP_SETTINGS = (
    "USE_MOE_FUSED_EP_KERNEL",
    "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
)


def test_every_setting_the_decision_reads_is_on_the_roster():
    """The pair above is a list somebody has to remember to extend, so the
    names are also scanned out of the source: the two modules below are read
    where the model is BUILT, which is the worker, and a setting added to
    the decision and not to the roster fails here instead of going quiet.

    A missing name is not only a lost optimisation. Two of the settings this
    finds are refusals, and a refusal the worker cannot see turns into the
    run silently being the thing the refusal exists to prevent.
    """
    unknown = [
        name for name in FUSED_EP_SETTINGS
        if name not in envs.environment_variables
    ]
    assert not unknown, f"{unknown} are not declared in tpu_inference.envs"

    read = set(FUSED_EP_SETTINGS)
    for module in (moe_module, moe_fused_ep):
        read |= set(
            re.findall(r"\benvs\.([A-Z][A-Z0-9_]*)",
                       inspect.getsource(module)))
    # Only names this repository declares: a VLLM_-prefixed setting is
    # forwarded by vLLM's own executor rather than by this roster.
    read = {
        name
        for name in read
        if name in envs.environment_variables and not name.startswith("VLLM_")
    }
    assert read >= set(FUSED_EP_SETTINGS), "the source scan is broken"
    missing = sorted(read - set(TpuPlatform.additional_env_vars))
    assert not missing, (
        f"{missing} are read where the model is built but are not in "
        "TpuPlatform.additional_env_vars, so a Ray executor does not copy "
        "them to its workers, and the driver would answer on a setting "
        "every worker builds without")
