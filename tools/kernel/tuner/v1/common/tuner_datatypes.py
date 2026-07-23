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

import json
from dataclasses import asdict, dataclass
from enum import Enum


@dataclass
class TuningKey:
    # Specify the key for tuning case
    pass


@dataclass
class TunableParams:
    # Specify the tiles for tuning case
    pass


class TuningStatus(Enum):
    SUCCESS = 'SUCCESS'
    FAILED_OOM = 'FAILED_OOM'
    XPROF_MEASUREMENT_ERROR = 'XPROF_MEASUREMENT_ERROR'
    UNKNOWN_ERROR = 'UNKNOWN_ERROR'
    SKIPPED = 'SKIPPED'


class TuningCase:

    def __init__(self,
                 tuning_key: TuningKey,
                 tunable_params: TunableParams,
                 is_baseline: bool = False):
        self.tuning_key = tuning_key
        self.tunable_params = tunable_params
        self.is_baseline = is_baseline  # can be used to mark whether this case is the baseline case for the tuning key, which can be used for comparison in the analysis.

    def __str__(self):
        return json.dumps({
            'tuning_key': asdict(self.tuning_key),
            'tunable_params': asdict(self.tunable_params),
            'is_baseline': self.is_baseline
        })

    @classmethod
    def from_string(cls, string, tuning_key_class, tunable_params_class):
        data = json.loads(string)
        tuning_key = tuning_key_class(**data['tuning_key'])
        tunable_params = tunable_params_class(**data['tunable_params'])
        case = TuningCase(tuning_key, tunable_params)
        case.is_baseline = data.get('is_baseline', False)
        return case


@dataclass
class TunerConfig:
    tuning_key_class: any = None
    tunable_params_class: any = None
    kernel_tuner_name: str = None
    # When support autotune and run_config.autotune_mode is True,
    # the kernel tuner will read the cases from spanner using the case_set_id and kernel_tuner_name
    support_autotune: bool = False
    support_bayesian_optimization: bool = False
    jit_kernel_pattern: str = None


@dataclass
class RunConfig:
    case_set_id: str = None
    run_id: str = None
    case_set_desc: str = None
    tpu_version: str = None
    tpu_cores: int = None
    tpu_queue_multi: str = None
    run_locally: bool = False
    job_priority: int = -10
    max_execution_minutes: int = 20
    job_bucket_size: int = 500
    gcp_project_id: str = None
    spanner_instance_id: str = None
    spanner_database_id: str = None
    worker_id: str = None
    autotune_mode: bool = False
    debug: bool = False
