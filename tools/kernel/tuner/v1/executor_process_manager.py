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
"""Manages a persistent executor subprocess for isolated kernel run() calls.

The executor subprocess (kernel_tuner_executor.py) stays alive across
multiple tuning cases to preserve the expensive ``generate_inputs()``
cache.  It only restarts when it crashes (e.g. unrecoverable TPU error
poisons the JAX runtime).

Communication uses stdin/stdout JSON lines — one JSON object per line.
"""

import dataclasses
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING

from tools.kernel.tuner.v1.common.tuner_datatypes import (RunConfig,
                                                          TuningStatus)
from tools.kernel.tuner.v1.kernel_tuner_factory import run_config_to_json
from tools.kernel.tuner.v1.utils import get_subprocess_env

if TYPE_CHECKING:
    from tools.kernel.tuner.v1.common.tuner_datatypes import (TunableParams,
                                                              TuningKey)

logger = logging.getLogger(__name__)


def _set_pdeathsig():
    """Sets PR_SET_PDEATHSIG to SIGKILL on Linux.

    This ensures the child executor process receives SIGKILL if its parent worker
    process terminates (e.g. killed by the runner or container timeout), preventing
    orphaned processes from holding the TPU device lock.
    """
    try:
        import ctypes
        import signal
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


class ExecutorProcessManager:
    """Manages a persistent executor subprocess for isolated run() calls.

    Usage::

        mgr = ExecutorProcessManager("batched_rpa_kernel_tuner", run_config)
        status, avg_ns, total_ns = mgr.execute_run(tuning_key, tunable_params, iters=5)
        mgr.shutdown()

    The executor is started lazily on the first ``execute_run()`` call and
    kept alive for subsequent calls.  If the executor crashes, the next
    ``execute_run()`` transparently spawns a replacement.
    """

    _EXECUTOR_TIMEOUT_SECONDS = 10 * 60  # 10 minutes per run() call

    def __init__(self, kernel_tuner_name: str, run_config: RunConfig):
        self._kernel_tuner_name = kernel_tuner_name
        self._run_config = run_config
        self._proc: subprocess.Popen | None = None
        self._run_config_path: str | None = None  # temp file, created once
        self._stderr_thread: threading.Thread | None = None
        self._restart_count = 0
        self._consecutive_start_failures = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_run(
        self,
        tuning_key: 'TuningKey',
        tunable_params: 'TunableParams',
        iters: int,
        timeout_seconds: int | None = None,
        use_xprof: bool = False,
    ) -> tuple[TuningStatus, float, float]:
        """Send a run request and return (status, avg_latency_ns, total_latency_ns).

        On crash or timeout the executor is killed and UNKNOWN_ERROR is
        returned.  The next ``execute_run()`` call will transparently
        start a fresh executor.
        """
        self._ensure_alive()
        timeout = timeout_seconds or self._EXECUTOR_TIMEOUT_SECONDS

        request = {
            "cmd": "run",
            "tuning_key": dataclasses.asdict(tuning_key),
            "tunable_params": dataclasses.asdict(tunable_params),
            "iters": iters,
            "use_xprof": use_xprof,
        }

        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()

            result = self._read_json_response(timeout)

            return (
                TuningStatus(result["status"]),
                result["avg_latency_ns"],
                result["total_latency_ns"],
            )
        except (BrokenPipeError, OSError) as e:
            logger.error("Executor pipe error: %s", e)
            self._kill()
            return TuningStatus.UNKNOWN_ERROR, 0, 0
        except json.JSONDecodeError as e:
            logger.error("Executor returned invalid JSON: %s", e)
            self._kill()
            return TuningStatus.UNKNOWN_ERROR, 0, 0
        except TimeoutError:
            logger.error("Executor timed out after %d seconds", timeout)
            self._kill()
            return TuningStatus.UNKNOWN_ERROR, 0, 0

    def shutdown(self):
        """Clean shutdown of the executor subprocess."""
        if self._proc is None or self._proc.poll() is not None:
            self._cleanup_temp_file()
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self._proc.stdin.flush()
            self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except Exception:
            logger.warning("Clean shutdown failed, killing executor.")
            self._kill()
        self._cleanup_temp_file()

    @property
    def restart_count(self) -> int:
        """Number of times the executor has been restarted after a crash."""
        return self._restart_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_alive(self):
        """Starts executor if not running.  Auto-restarts after crash."""
        if self._proc is not None and self._proc.poll() is None:
            return  # still alive
        if self._proc is not None:
            logger.warning(
                "Executor process died (rc=%s), restarting... (restart #%d)",
                self._proc.returncode, self._restart_count + 1)
            self._restart_count += 1
        self._start()

    def _start(self):
        """Spawns a fresh executor subprocess with stdin/stdout pipes."""
        if self._consecutive_start_failures >= 3:
            raise RuntimeError(
                f"Executor subprocess failed to start {self._consecutive_start_failures} consecutive times "
                "(unrecoverable TPU/JAX runtime error). Aborting worker process."
            )

        # Write RunConfig to a temp file (reused across restarts).
        if self._run_config_path is None:
            fd, self._run_config_path = tempfile.mkstemp(
                prefix="executor_run_config_", suffix=".json")
            with os.fdopen(fd, 'w') as f:
                f.write(run_config_to_json(self._run_config))

        executor_module = "tools.kernel.tuner.v1.kernel_tuner_executor"
        command = [
            sys.executable,
            "-m",
            executor_module,
            f"--kernel_tuner_name={self._kernel_tuner_name}",
            f"--run_config_path={self._run_config_path}",
        ]

        from tools.kernel.tuner.v1.kernel_tuner_flags import \
            get_present_flag_args
        command.extend(
            get_present_flag_args(exclude_flags={
                'kernel_tuner_name', 'result_path', 'run_config_path'
            }))

        env = get_subprocess_env()

        logger.info(
            f'Starting {executor_module} as subprocess for evaluate single case...'
        )
        logger.debug(f"Command: {' '.join(command)}")
        preexec_fn = _set_pdeathsig if sys.platform.startswith(
            "linux") else None
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            env=env,
            start_new_session=True,  # own process group for clean kill
            preexec_fn=preexec_fn,
        )

        # Stream executor stderr to our logger in a background thread.
        self._stderr_thread = threading.Thread(target=self._stream_stderr,
                                               daemon=True)
        self._stderr_thread.start()

        # Wait for the executor to signal readiness.
        try:
            ready_msg = self._read_json_response(
                timeout=120)  # generous timeout for JAX init
            if ready_msg.get("status") != "ready":
                raise RuntimeError(
                    f"Executor sent unexpected ready message: {ready_msg}")
            self._consecutive_start_failures = 0
            logger.info("Executor subprocess is ready (pid=%d).",
                        self._proc.pid)
        except Exception as e:
            self._consecutive_start_failures += 1
            logger.error(
                "Executor failed to start (consecutive failure %d/3): %s",
                self._consecutive_start_failures, e)
            self._kill()
            raise RuntimeError(
                f"Executor subprocess failed to start: {e}") from e

    def _read_json_response(self, timeout: int) -> dict:
        """Reads lines from executor stdout until a valid protocol JSON line is received.

        Filters out non-JSON log lines that third-party libraries (e.g. vllm, JAX)
        may write to stdout.
        """
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            remaining_timeout = max(1, int(timeout - elapsed))
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Executor did not respond within {timeout} seconds.")

            line = self._read_with_timeout(remaining_timeout)

            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("__JSON__"):
                line_str = line_str[len("__JSON__"):].strip()

            try:
                return json.loads(line_str)
            except json.JSONDecodeError:
                logger.debug(
                    "Ignoring non-JSON output from executor stdout: %s",
                    line_str)

    def _read_with_timeout(self, timeout: int) -> str:
        """Read a single line from executor stdout with a timeout.

        Uses a background thread to avoid blocking the main thread
        indefinitely.
        """
        result_container = [None]
        error_container = [None]

        def _reader():
            try:
                line = self._proc.stdout.readline()
                result_container[0] = line
            except Exception as e:
                error_container[0] = e

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()
        reader_thread.join(timeout=timeout)

        if reader_thread.is_alive():
            raise TimeoutError(
                f"Executor did not respond within {timeout} seconds.")
        if error_container[0] is not None:
            raise error_container[0]
        line = result_container[0]
        if not line:
            raise BrokenPipeError("Executor stdout closed (process died?).")
        return line

    def _stream_stderr(self):
        """Stream executor stderr lines to our logger."""
        try:
            for line in self._proc.stderr:
                line = line.rstrip()
                if line:
                    logger.info("[executor] %s", line)
        except (ValueError, OSError):
            pass  # pipe closed

    def _kill(self):
        """Forcibly kill the executor process group."""
        if self._proc is None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            try:
                self._proc.kill()
            except (ProcessLookupError, OSError):
                pass
        # Close stdio pipes to release OS locks and unblock reader threads.
        for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except (OSError, ValueError):
                    pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass
        self._proc = None

    def _cleanup_temp_file(self):
        """Remove the temporary RunConfig JSON file."""
        if self._run_config_path and os.path.exists(self._run_config_path):
            try:
                os.remove(self._run_config_path)
            except OSError:
                pass
            self._run_config_path = None
