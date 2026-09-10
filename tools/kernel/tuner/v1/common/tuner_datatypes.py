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
from typing import Any, NamedTuple, Optional, Protocol, Type, runtime_checkable


@runtime_checkable
class TuningKey(Protocol):
    """Protocol for any frozen, hashable tuning key dataclass."""

    def __hash__(self) -> int:
        ...


@runtime_checkable
class TunableParams(Protocol):
    """Protocol for any frozen tunable parameters dataclass with comparison operators."""

    def __hash__(self) -> int:
        ...

    def __le__(self, other: Any) -> bool:
        ...

    def __ge__(self, other: Any) -> bool:
        ...


class TuningStatus(Enum):
    SUCCESS = 'SUCCESS'
    FAILED_OOM = 'FAILED_OOM'
    XPROF_MEASUREMENT_ERROR = 'XPROF_MEASUREMENT_ERROR'
    UNKNOWN_ERROR = 'UNKNOWN_ERROR'
    SKIPPED = 'SKIPPED'


class BucketStatus(Enum):
    INITIALIZED = 'INITIALIZED'
    IN_PROGRESS = 'IN_PROGRESS'
    COMPLETED = 'COMPLETED'
    # The Bucket failed to complete due to errors after maximum bucket level retries in worker process
    FAILED = 'FAILED'
    # When run in non local mode, if RunConfig.max_execute_minutes is reached, it will yield to other CI jobs
    YIELDED = 'YIELDED'


class ProcessedCaseStatus(NamedTuple):
    case_id: int
    status: str


@dataclass
class CaseResult:
    case_set_id: str
    run_id: str
    case_id: int
    processed_status: str
    worker_id: str
    latency: int
    warmup_time: int
    total_time: int
    processed_at: int
    tpu: str


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
    tuning_key_class: Type[TuningKey]
    tunable_params_class: Type[TunableParams]
    kernel_tuner_name: str
    # When support autotune and run_config.autotune_mode is True,
    # the kernel tuner will read the cases from spanner using the case_set_id and kernel_tuner_name
    support_autotune: bool = False
    support_bayesian_optimization: bool = False
    jit_kernel_pattern: Optional[str] = None
    # Number of Bayesian optimization trials (optuna) to run per tuning key bucket.
    # Only used when support_bayesian_optimization is True.
    n_bayesian_trials: int = 100
    # Minimum number of total cases for a tuning key search space required to use Bayesian Optimization.
    # If the search space has fewer cases than min_cases_for_bayesian, BO falls back to full sweep search.
    min_cases_for_bayesian: int = 200
    # Early stopping patience (number of consecutive trials without min_delta_ratio relative improvement).
    # None by default (early stopping disabled unless explicitly specified).
    bayesian_early_stopping_patience: Optional[int] = None
    # Early stopping relative improvement threshold ratio (e.g., 0.05 for 5% improvement).
    bayesian_early_stopping_min_delta_ratio: float = 0.10


@dataclass
class RunConfig:
    case_set_id: Optional[str] = None
    run_id: Optional[str] = None
    case_set_desc: Optional[str] = None
    tpu_version: Optional[str] = None
    tpu_cores: Optional[int] = None
    tpu_queue_multi: Optional[str] = None
    run_locally: bool = False
    job_priority: int = -10
    max_execution_minutes: int = 20
    job_bucket_size: int = 500
    gcp_project_id: Optional[str] = None
    spanner_instance_id: Optional[str] = None
    spanner_database_id: Optional[str] = None
    worker_id: Optional[str] = None
    autotune_mode: bool = False
    use_bayesian_optimization: bool = False
    # Runtime override for number of Bayesian trials per tuning key bucket.
    n_bayesian_trials: Optional[int] = None
    # Runtime override for minimum cases required for Bayesian optimization.
    min_cases_for_bayesian: Optional[int] = None
    # Local database directory path (used when run_locally=True).
    local_db_path: Optional[str] = None

    def subbucket_yml_path(self, end_case_id: int) -> str:
        return f'/tmp/kernel_tuning/subbucket_{end_case_id}_pipeline.yml'
