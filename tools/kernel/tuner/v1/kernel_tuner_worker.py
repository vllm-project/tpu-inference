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
"""Kernel tuner worker subprocess entry point.

This script is spawned by ``kernel_tuner_runner._invoke_worker_process``
in a fresh Python process so that optimizer state is isolated from the
runner.

Architecture (three-process model):
  * **Runner** → spawns **Worker** (this script)
  * **Worker** → creates optimizer + storage_manager + ExecutorProcessManager
  * **ExecutorProcessManager** → spawns persistent **Executor** subprocess
    for isolated run() / generate_inputs() calls

Communication protocol:
  * **Inputs** — ``RunConfig`` is read from a JSON file whose path is
    passed via ``--run_config_path``.  The kernel tuner name, case-id
    range, and result path are passed as CLI flags.
  * **Outputs** — The worker writes a JSON object with
    ``next_begin_case_id`` to the file at ``--result_path``.
"""

import json
import logging
import signal
import sys

from absl import app, flags
from absl import logging as absl_logging

from tools.kernel.tuner.v1.executor_process_manager import \
    ExecutorProcessManager
from tools.kernel.tuner.v1.kernel_tuner_factory import (create_kernel_tuner,
                                                        create_storage_manager,
                                                        run_config_from_json)
from tools.kernel.tuner.v1.kernel_tuner_flags import (BEGIN_CASE_ID,
                                                      END_CASE_ID,
                                                      KERNEL_TUNER_NAME)

logger = logging.getLogger(__name__)

# Worker-only flags (not shared with runner).
_RUN_CONFIG_PATH = flags.DEFINE_string(
    'run_config_path', None,
    'Path to the JSON file containing the serialized RunConfig.')

_RESULT_PATH = flags.DEFINE_string(
    'result_path', None, 'Path where the worker writes its JSON result '
    '(containing next_begin_case_id).')


def main(argv):
    del argv  # Unused.
    absl_logging.get_absl_handler().setFormatter(
        logging.Formatter(
            '%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%m%d %H:%M:%S',
        ))

    # --- Read RunConfig from JSON file ---
    with open(_RUN_CONFIG_PATH.value, 'r') as f:
        run_config = run_config_from_json(f.read())

    # --- Lightweight KernelTunerBase for config/search_space only ---
    kernel_tuner = create_kernel_tuner(KERNEL_TUNER_NAME.value,
                                       run_config,
                                       lightweight=True)

    # --- Storage manager (lives in worker process, not executor) ---
    storage_manager = create_storage_manager(run_config)

    # --- Executor process manager (persistent executor subprocess) ---
    executor_mgr = ExecutorProcessManager(KERNEL_TUNER_NAME.value, run_config)

    def _handle_signal(signum, frame):
        logger.warning(
            'Worker process received signal %d (%s), shutting down executor process...',
            signum,
            signal.Signals(signum).name)
        executor_mgr.shutdown()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Create optimizer (now lives here, NOT in KernelTunerBase) ---
    if kernel_tuner.use_bayesian_optimization:
        from tools.kernel.tuner.v1.optimizer import BayesianOptimizer
        optimizer = BayesianOptimizer(kernel_tuner, storage_manager,
                                      executor_mgr)
    else:
        from tools.kernel.tuner.v1.optimizer import SweepOptimizer
        optimizer = SweepOptimizer(kernel_tuner, storage_manager, executor_mgr)

    begin_case_id = BEGIN_CASE_ID.value
    end_case_id = END_CASE_ID.value

    try:
        # TODO: measure_latency should take a param called time_budget so the executor can yield
        # when the time_budget is reached. This is useful for the sweep optimizer
        # to avoid running for too long but not for the Bayesian optimizer.
        next_begin_case_id = optimizer.measure_latency(
            begin_case_id=begin_case_id, end_case_id=end_case_id)
    finally:
        executor_mgr.shutdown()

    # --- Write result to file ---
    result = {'next_begin_case_id': next_begin_case_id}
    with open(_RESULT_PATH.value, 'w') as f:
        json.dump(result, f)
    logger.debug('Worker result written to %s: %s', _RESULT_PATH.value, result)


if __name__ == '__main__':
    app.run(main)
