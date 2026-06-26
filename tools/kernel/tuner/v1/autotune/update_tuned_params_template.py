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

# type: ignore

from tools.kernel.tuner.v1.common.kernel_tuner_base import TuningCase
from tools.kernel.tuner.v1.storage_management.spanner_database_manager import \
    SpannerStorageManager


# ruff: noqa: F821
def get_tuned_params(tuning_key: TuningKey) -> TunableParams:
    KERNEL_TUNER_NAME = "KERNEL_TUNER_NAME_PLACEHOLDER"
    CASE_SET_ID = "CASE_SET_ID_PLACEHOLDER"
    TPU = "TPU_PLACEHOLDER"
    tunable_params = _get_tuned_params(tuning_key)
    # log the tuning key and the tuned params to a gcp database
    # Id(CaseSetId), CaseKeyValue, KernelTunerName, TPU
    # This case set will use to build all tuning cases later.
    tuning_case = TuningCase(tuning_key, tunable_params, is_baseline=True)
    storage_manager = SpannerStorageManager(
        gcp_project_id='cloud-tpu-inference-test',
        spanner_instance_id='vllm-bm-inst',
        spanner_database_id='tune-gmm')
    storage_manager.add_autotune_case(CASE_SET_ID, str(tuning_case),
                                       KERNEL_TUNER_NAME, TPU)
    storage_manager.close()
    return tunable_params
