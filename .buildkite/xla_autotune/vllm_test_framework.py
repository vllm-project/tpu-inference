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
"""vllm serve + benchmark_serving runner for the XLA autotune sweep."""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Upper bound on how long `vllm serve` may take to open its port.  This only
# guards against a genuine hang (process alive but never serving); a new flag
# set cold-compiles the full graph on first run, which on the 397B model can
# take well over an hour, so the bound is deliberately generous and overridable.
_STARTUP_TIMEOUT_S = int(
    os.environ.get("AUTOTUNE_SERVER_STARTUP_TIMEOUT_S", "7200"))


@dataclass(frozen=True)
class ModelSpec:
    serve_args: List[str]
    server_env: Dict[str, str]
    benchmark_shapes: List[Dict[str, Any]]


# Production model registry.  Add an entry here to onboard a new model.
MODELS: Dict[str, ModelSpec] = {
    "Qwen/Qwen3.5-397B-A17B-FP8":
    ModelSpec(
        serve_args=[
            "--tensor-parallel-size=8",
            "--max-model-len=9216",
            "--max-num-batched-tokens=1024",
            "--max-num-seqs=64",
            "--gpu-memory-utilization=0.9",
            "--no-enable-prefix-caching",
            "--async-scheduling",
            "--language-model-only",
            "--enable-auto-tool-choice",
            "--tool-call-parser=qwen3_coder",
            "--reasoning-parser=qwen3",
            '--limit-mm-per-prompt={"image": 0, "video": 0}',
            "--kv-cache-dtype=fp8",
            "--enable-expert-parallel",
            '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}',
            "--mamba-ssm-cache-dtype=bfloat16",
        ],
        server_env={
            "MODEL_IMPL_TYPE": "vllm",
            "USE_MOE_EP_KERNEL": "0",
            "ATTN_BUCKETIZED_NUM_REQS": "true",
            "ATTN_CUSTOM_NUM_REQS_BUCKETS": "8,16,32,64",
            "NEW_MODEL_DESIGN": "1",
        },
        benchmark_shapes=[
            {
                "random-input-len": 8192,
                "random-output-len": 1024
            },
        ],
    ),
}

DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_PORT = 8000
# bench_serving is git-cloned into the docker workdir by the shard wrapper.
DEFAULT_BENCHMARK_SCRIPT = "bench_serving/benchmark_serving.py"
DEFAULT_BENCHMARK_ARGS: Dict[str, Any] = {
    "--ignore-eos": True,
    "--dataset-name": "random",
    "--backend": "vllm",
    "--random-range-ratio": "0.8",
    "--num-prompts": "640",
    "--max-concurrency": "512",
    "--percentile-metrics": "ttft,tpot,itl,e2el",
    "--save-result": True,
}


@dataclass
class VLLMTestParam:
    model_name: str = DEFAULT_MODEL
    port: int = DEFAULT_PORT
    # Appended (space-joined) to LIBTPU_INIT_ARGS.  libtpu uses SPACE — not
    # comma — as the flag separator; each entry must be one `--flag=value`.
    extra_libtpu_init_args: List[str] = field(default_factory=list)
    benchmark_script_path: str = DEFAULT_BENCHMARK_SCRIPT
    benchmark_args: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_BENCHMARK_ARGS))
    # Warmup passes per shape (discarded).  The (warmup_runs + 1)th pass is
    # the measured result; vLLM's first-batch latency reflects compile / cache
    # transients, not steady-state throughput.
    warmup_runs: int = 1
    base_log_dir: Optional[str] = None
    tag: str = ""


@dataclass
class TrialResult:
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class VLLMTestFramework:
    """One trial: start `vllm serve`, benchmark every shape, tear down."""

    def __init__(self, param: VLLMTestParam, dry_run: bool = False):
        if param.model_name not in MODELS:
            raise KeyError(f"unknown model {param.model_name!r}; "
                           f"known: {sorted(MODELS)}")
        self.param = param
        self.spec = MODELS[param.model_name]
        self.dry_run = dry_run
        self._server: Optional[subprocess.Popen] = None
        atexit.register(self._stop_server)

        base = param.base_log_dir or os.path.join(os.getcwd(), "logs")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{param.tag}_EXP_{ts}" if param.tag else f"EXP_{ts}"
        self.exp_dir = os.path.join(base, folder)
        os.makedirs(self.exp_dir, exist_ok=True)

        # A per-trial logger with its own handlers.  `basicConfig` is a no-op
        # once the root logger has handlers, so a shared logger would route
        # every trial's output to the first trial's file; key the logger on the
        # trial tag and attach fresh handlers instead.
        self.log = logging.getLogger(f"{__name__}.{param.tag or id(self)}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        for handler in list(self.log.handlers):
            self.log.removeHandler(handler)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (
                logging.FileHandler(
                    os.path.join(self.exp_dir, "framework_main.log")),
                logging.StreamHandler(),
        ):
            handler.setFormatter(fmt)
            self.log.addHandler(handler)
        self.log.info("Experiment dir: %s", self.exp_dir)

    def run_benchmark(self) -> TrialResult:
        """Set up env, launch the server, benchmark every shape, tear down."""
        self._build_env()
        self._start_server()
        try:
            return self._benchmark_all_shapes()
        finally:
            self._stop_server()

    # ---------------------------------------------------------------- env

    def _build_env(self) -> None:
        self.env = os.environ.copy()
        # The GCS-backed JAX compile cache keys on the byte-exact
        # LIBTPU_INIT_ARGS string; the autotuner owns every byte that
        # lands here so cache reuse is predictable.
        self.env["LIBTPU_INIT_ARGS"] = " ".join(
            self.param.extra_libtpu_init_args)
        # Each trial legitimately re-lowers HLO when its flag set changes;
        # vLLM's recompilation guard would otherwise abort the run.
        self.env["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
        for k, v in self.spec.server_env.items():
            self.env[k] = v

        with open(os.path.join(self.exp_dir, "experiment_info.txt"), "w") as f:
            f.write(f"model: {self.param.model_name}\n")
            f.write(f"port: {self.param.port}\n")
            f.write(f"started: {datetime.now()}\n\n--- key env ---\n")
            f.write(f"LIBTPU_INIT_ARGS={self.env['LIBTPU_INIT_ARGS']}\n")
            for k in sorted(self.spec.server_env):
                f.write(f"{k}={self.env.get(k, '')}\n")
            f.write("\n--- full env ---\n")
            for k, v in sorted(self.env.items()):
                f.write(f"{k}={v}\n")

    # -------------------------------------------------------- server lifecycle

    def _start_server(self) -> None:
        cmd = [
            "vllm",
            "serve",
            self.param.model_name,
            "--port",
            str(self.param.port),
            *self.spec.serve_args,
        ]
        with open(os.path.join(self.exp_dir, "vllm_server.cmd"), "w") as f:
            f.write(" ".join(cmd))

        if self.dry_run:
            self.log.info("[dry-run] would start: %s", " ".join(cmd))
            return

        log_path = os.path.join(self.exp_dir, "vllm_server.log")
        self.log.info("Starting vllm serve...")
        self._server = subprocess.Popen(
            cmd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
            errors="replace",
        )

        def tail() -> None:
            with open(log_path, "w") as f:
                for line in iter(self._server.stdout.readline, ""):
                    f.write(line)
                    f.flush()

        threading.Thread(target=tail, daemon=True).start()
        self._wait_for_port()

    def _wait_for_port(self) -> None:
        t0 = time.time()
        while True:
            if self._server.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited during startup (rc={self._server.poll()})"
                )
            if time.time() - t0 > _STARTUP_TIMEOUT_S:
                raise RuntimeError(
                    f"vllm serve did not open port {self.param.port} within "
                    f"{_STARTUP_TIMEOUT_S}s")
            try:
                with socket.create_connection(("127.0.0.1", self.param.port),
                                              timeout=1):
                    self.log.info("Server ready (%ds)", int(time.time() - t0))
                    return
            except OSError:
                time.sleep(10)

    def _stop_server(self) -> None:
        if not self._server:
            return
        try:
            pgid = os.getpgid(self._server.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                self._server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.log.warning("vllm serve did not exit in 20s; SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
                self._server.wait()
        except ProcessLookupError:
            pass
        finally:
            self._server = None

    # ------------------------------------------------------------- benchmark

    def _benchmark_all_shapes(self) -> TrialResult:
        result = TrialResult(success=True)
        passes = max(0, self.param.warmup_runs) + 1
        summary = os.path.join(self.exp_dir, "benchmark_serving_all.log")
        with open(summary, "w") as f:
            f.write(f"=== {datetime.now()} ===\n")

        for shape in self.spec.benchmark_shapes:
            input_len = shape["random-input-len"]
            output_len = shape["random-output-len"]
            base = f"benchmark_{input_len}_{output_len}"
            measured_path: Optional[str] = None

            self.log.info(
                "Shape input=%s output=%s — %d warmup + 1 measured",
                input_len,
                output_len,
                passes - 1,
            )

            for i in range(passes):
                is_measured = i == passes - 1
                name = base if is_measured else f"{base}_warmup{i+1}"
                fname = f"{name}_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

                cmd = ["python3", self.param.benchmark_script_path]
                for k, v in self.param.benchmark_args.items():
                    if isinstance(v, bool):
                        if v:
                            cmd.append(k)  # store_true flag, e.g. --ignore-eos
                        # False → omit; passing --flag=False errors store_true
                    else:
                        cmd.append(f"{k}={v}")
                cmd.extend([
                    f"--random-input-len={input_len}",
                    f"--random-output-len={output_len}",
                    f"--model={self.param.model_name}",
                    f"--port={self.param.port}",
                    f"--result-dir={self.exp_dir}",
                    f"--result-filename={fname}",
                ])

                rc = self._run_subprocess(name, cmd, summary)
                if rc != 0:
                    result.success = False
                    result.error += f"{name} failed rc={rc}; "
                    break
                if is_measured and not self.dry_run:
                    measured_path = os.path.join(self.exp_dir, fname)

            if self.dry_run:
                result.metrics[base] = {
                    "request_throughput": 10.0,
                    "output_throughput": 100.0,
                    "total_token_throughput": 1000.0,
                    "mean_ttft_ms": 50.0,
                    "mean_tpot_ms": 5.0,
                }
            elif result.success and measured_path:
                self._record_measured(measured_path, base, result, passes - 1)

        return result

    def _run_subprocess(self, name: str, cmd: List[str], log_path: str) -> int:
        with open(os.path.join(self.exp_dir, f"task_{name}.cmd"), "w") as f:
            f.write(" ".join(cmd))
        self.log.info("Starting [%s]", name)
        if self.dry_run:
            self.log.info("[dry-run] would run: %s", " ".join(cmd))
            return 0

        with open(log_path, "a") as log_f:
            proc = subprocess.Popen(
                cmd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            try:
                for line in iter(proc.stdout.readline, ""):
                    log_f.write(line)
                    log_f.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                return 130
            proc.wait()

        if self._server and self._server.poll() not in (None, 0):
            raise RuntimeError(
                f"vllm serve crashed (rc={self._server.poll()}) during [{name}]"
            )
        if proc.returncode != 0:
            self.log.error("[%s] FAILED rc=%d", name, proc.returncode)
        return proc.returncode

    def _record_measured(self, path: str, name: str, result: TrialResult,
                         warmup: int) -> None:
        with open(path) as f:
            data = json.load(f)
        data["warmup_runs"] = warmup
        result.metrics[name] = data
        self.log.info("--- %s ---", name)
        for k in (
                "request_throughput",
                "output_throughput",
                "total_token_throughput",
                "mean_ttft_ms",
                "mean_tpot_ms",
        ):
            v = data.get(k)
            s = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
            self.log.info("  %s = %s", k, s)
