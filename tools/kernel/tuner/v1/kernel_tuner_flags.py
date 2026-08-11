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
"""Shared absl flag definitions for the kernel tuner runner and worker."""

from absl import flags

RUN_LOCALLY = flags.DEFINE_bool(
    'run_locally', False,
    'If true, uses local storage instead of cloud storage.')
AUTOTUNE_MODE = flags.DEFINE_bool(
    'autotune_mode', False,
    'If true, runs the kernel tuner in autotune mode, which reads tuning cases from Spanner and generates Buildkite pipeline YAML for tuning jobs. '
)
KERNEL_TUNER_NAME = flags.DEFINE_string(
    'kernel_tuner_name', 'example_kernel_tuner',
    'Name of the kernel tuner to run, support RpaV3KernelTuner, RpaV3KernelTuner, MlaKernelTuner, BatchedRpaKernelTuner, FlashAttentionKernelTuner and an ExampleKernelTuner so far.'
)
CASE_SET_ID = flags.DEFINE_string('case_set_id', '',
                                  'The case set ID to use for this run.')
RUN_ID = flags.DEFINE_string(
    'run_id', '',
    'The run ID to use for this run. If not specified, a timestamp-based ID will be generated.'
)
CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '',
                                    'Description of the case set.')
GENERATE_BUILDKITE_PIPELINE = flags.DEFINE_bool(
    'generate_buildkite_pipeline', False,
    'If true, generates Buildkite pipeline YAML instead of running tuning jobs.'
)
BEGIN_CASE_ID = flags.DEFINE_integer(
    'begin_case_id', None,
    'The begin case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.'
)
END_CASE_ID = flags.DEFINE_integer(
    'end_case_id', None,
    'The end case ID for tuning. Only used when --generate_buildkite_pipeline is false and --run_locally is false.'
)
GCP_PROJECT_ID = flags.DEFINE_string(
    'gcp_project_id', 'cloud-tpu-inference-test',
    'The GCP project ID to use for Spanner. Only used when --run_locally is false.'
)
SPANNER_INSTANCE_ID = flags.DEFINE_string(
    'spanner_instance_id', 'vllm-bm-inst',
    'The Spanner instance ID to use. Only used when --run_locally is false.')
SPANNER_DATABASE_ID = flags.DEFINE_string(
    'spanner_database_id', 'tune-gmm',
    'The Spanner database ID to use. Only used when --run_locally is false.')
WORKER_ID = flags.DEFINE_string(
    'worker_id', None,
    'The worker ID representing the kernel_tuner_worker process. If not specified, resolves from TPU_WORKER_ID, HOST_NAME, HOSTNAME, or defaults to "0".'
)
TPU_VERSION = flags.DEFINE_string(
    'tpu_version', '',
    'The TPU version to use for tuning. Supported values are "tpu6e" and "tpu7x".'
)

TPU_CORES = flags.DEFINE_integer(
    'tpu_cores', 0,
    'The number of TPU cores to use for tuning. Default is 0. TPU tpu6e has 1 core per chip, TPU tpu7x has 2 cores per chip.'
)

TPU_QUEUE_MULTI = flags.DEFINE_string(
    'tpu_queue_multi', '',
    'The TPU queue to use for tuning. This will be automatically determined based on the TPU version and cores if not specified. Supported values are "tpu_v6e_queue", "tpu_v6e_8_queue", "tpu_v7x_2_queue", "tpu_v7x_8_queue", and "tpu_v7x_16_queue".'
)

JOB_PRIORITY = flags.DEFINE_integer(
    'job_priority', -10,
    'The priority to use for kernel tuning jobs. Higher priority jobs will be scheduled before lower priority ones. Default is -10, which is lower than typical user jobs to avoid impacting them.'
)

MAX_EXECUTION_MINUTES = flags.DEFINE_integer(
    'max_execution_minutes', 20,
    'Only used when the kernel tuning job is scheduled through Buildkite. The maximum execution time in minutes for each kernel tuning job. If the job exceeds this time, it will save the job progresss, generate a new job to be scheduled by Buildkite and exit.'
)

USE_BAYESIAN_OPTIMIZATION = flags.DEFINE_boolean(
    'use_bayesian_optimization', False,
    ' whether to use Bayesian optimization (optuna) instead of sweeping '
    'all tuning cases.  When True, the kernel tuner uses optuna to intelligently '
    'select which tunable-parameter combinations to evaluate.  When False, every '
    'case is evaluated (full sweep).  When not specified (None), the default set '
    'by each kernel tuner\'s TunerConfig.support_bayesian_optimization is used.'
)

N_BAYESIAN_TRIALS = flags.DEFINE_integer(
    'n_bayesian_trials', None,
    'Number of Bayesian optimization trials to run per tuning key bucket. '
    'Overrides default if specified via flag or KERNEL_TUNING_N_BAYESIAN_TRIALS env var.'
)

MIN_CASES_FOR_BAYESIAN = flags.DEFINE_integer(
    'min_cases_for_bayesian', None,
    'Minimum number of total cases for a tuning key search space required to use Bayesian Optimization. '
    'Overrides default if specified via flag or KERNEL_TUNING_MIN_CASES_FOR_BAYESIAN env var.'
)

# ------------------------------------------------------------------
# MLA Kernel Tuner Flags
# ------------------------------------------------------------------
MLA_TOTAL_NUM_PAGES = flags.DEFINE_integer(
    "mla_total_num_pages", 1506, "Total number of pages in the cache.")
MLA_PAGE_SIZE_PER_KV_PACKING = flags.DEFINE_integer(
    "mla_page_size_per_kv_packing", 256, "Page size per KV packing.")
MLA_KV_PACKING = flags.DEFINE_integer("mla_kv_packing", 4,
                                      "Packing factor for KV.")
MLA_MAX_NUM_SEQS = flags.DEFINE_integer(
    "mla_max_num_seqs", 160, "Maximum number of sequences in the batch.")
MLA_PAGES_PER_SEQ = flags.DEFINE_integer("mla_pages_per_seq", 9,
                                         "Number of pages per sequence.")
MLA_ACTUAL_NUM_Q_HEADS = flags.DEFINE_integer("mla_actual_num_q_heads", 128,
                                              "Actual number of Q heads.")
MLA_ACTUAL_LKV_DIM = flags.DEFINE_integer("mla_actual_lkv_dim", 512,
                                          "Actual NOPE head dimension.")
MLA_ACTUAL_R_DIM = flags.DEFINE_integer("mla_actual_r_dim", 64,
                                        "Actual ROPE head dimension.")
MLA_KV_DTYPE = flags.DEFINE_string("mla_kv_dtype", "float8_e4m3fn",
                                   "KV cache data type.")
MLA_Q_DTYPE = flags.DEFINE_string("mla_q_dtype", "float8_e4m3fn",
                                  "Q activation dtype.")


def get_present_flag_args(exclude_flags=()):
    """Serialize explicitly-set shared tuner flags to CLI args.

    Only flags defined in THIS module are forwarded, so process-specific
    flags (worker's --result_path, executor's --run_config_path, absl
    logging flags) can never leak into a child that doesn't define them.
    """
    exclude = set(exclude_flags)
    exclude.add('generate_buildkite_pipeline')
    own = {
        f.name
        for f in flags.FLAGS.flags_by_module_dict().get(__name__, [])
    }
    return [
        flags.FLAGS[name].serialize()  # absl-native '--name=value'
        for name in sorted(own)
        if flags.FLAGS[name].present and name not in exclude
    ]
