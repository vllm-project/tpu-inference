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
"""Persistent executor subprocess for isolated kernel run() calls.

Spawned by the worker process.  Stays alive across multiple cases to
preserve the generate_inputs() cache.  Restarts only on crash (e.g.
TPU OOM poisons the JAX runtime).

Communication protocol:
  - Reads JSON requests from **stdin** (one JSON object per line).
  - Writes JSON responses to **stdout** (one JSON object per line).
  - Sends a ``{"status": "ready"}`` line on stdout after initialisation.
  - Exits cleanly on ``{"cmd": "shutdown"}`` or EOF on stdin.
"""

import json
import logging
import sys

import jax
from absl import app, flags
from absl import logging as absl_logging

from tools.kernel.tuner.v1.kernel_tuner_factory import (create_kernel_tuner,
                                                        run_config_from_json)
from tools.kernel.tuner.v1.kernel_tuner_flags import KERNEL_TUNER_NAME

_RUN_CONFIG_PATH = flags.DEFINE_string(
    'run_config_path', None,
    'Path to a JSON file containing the serialised RunConfig.')

logger = logging.getLogger(__name__)


def main(argv):
    del argv
    absl_logging.get_absl_handler().setFormatter(
        logging.Formatter(
            '%(levelname).1s%(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%m%d %H:%M:%S',
        ))

    # ---- Load RunConfig and create kernel tuner (full init) ----
    with open(_RUN_CONFIG_PATH.value, 'r') as f:
        run_config = run_config_from_json(f.read())

    kernel_tuner = create_kernel_tuner(KERNEL_TUNER_NAME.value,
                                       run_config,
                                       lightweight=False)

    # Signal readiness to the parent (worker) process.
    sys.stdout.write("__JSON__" + json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()
    logger.info("Executor ready, entering request loop.")

    # ---- Request loop ----
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Executor received invalid JSON: %s", e)
            continue

        cmd = request.get("cmd")
        if cmd == "shutdown":
            logger.info("Executor received shutdown command.")
            break

        if cmd != "run":
            logger.warning("Executor received unknown command: %s", cmd)
            continue

        # ---- Deserialise request ----
        tuning_key = kernel_tuner.tuner_config.tuning_key_class(
            **request["tuning_key"])
        tunable_params = kernel_tuner.tuner_config.tunable_params_class(
            **request["tunable_params"])
        iters = request["iters"]
        use_xprof = request.get("use_xprof", False)

        # ---- Execute run() ----
        try:
            if use_xprof and kernel_tuner.tuner_config.jit_kernel_pattern:
                kernel_tuner._cleanup_xprof_dir()
                with jax.profiler.trace(kernel_tuner.xprof_dir,
                                        create_perfetto_link=False):
                    status, avg_ns, total_ns = kernel_tuner.run(
                        tuning_key, tunable_params, iters)
            else:
                status, avg_ns, total_ns = kernel_tuner.run(
                    tuning_key, tunable_params, iters)

            result = {
                "status": status.value,
                "avg_latency_ns": avg_ns,
                "total_latency_ns": total_ns,
            }
        except Exception as e:
            logger.error("Executor run() raised an exception: %s",
                         e,
                         exc_info=True)
            result = {
                "status": "UNKNOWN_ERROR",
                "avg_latency_ns": 0,
                "total_latency_ns": 0,
            }

        sys.stdout.write("__JSON__" + json.dumps(result) + "\n")
        sys.stdout.flush()

    logger.info("Executor exiting.")


if __name__ == '__main__':
    app.run(main)
