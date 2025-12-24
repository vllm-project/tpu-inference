# Copyright 2025 Google LLC
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

# tests/e2e/test_model_loader.py

import os
import re
import signal
import subprocess
import sys
import tempfile
import time

import pytest
import requests
import torch
from flax import nnx
from vllm.model_executor.models.registry import ModelRegistry

from tpu_inference.models.common.model_loader import (_MODEL_REGISTRY,
                                                      register_model)


@pytest.fixture
def cleanup_registries():
    """Cleans up the model registries before and after each test."""
    _MODEL_REGISTRY.clear()
    # vLLM's ModelRegistry uses a class-level dictionary to store model classes.
    # We need to clear it to ensure test isolation.
    if hasattr(ModelRegistry, "models"):
        ModelRegistry.models.clear()
    yield
    _MODEL_REGISTRY.clear()
    if hasattr(ModelRegistry, "models"):
        ModelRegistry.models.clear()


class DummyGoodModel(nnx.Module):
    """A valid model that conforms to the expected interface."""

    def __init__(self, vllm_config=None, rng=None, mesh=None):
        pass

    def __call__(self,
                 kv_caches=None,
                 input_ids=None,
                 attention_metadata=None):
        pass


def test_register_model_success(cleanup_registries):
    """Tests that a valid model is registered successfully."""
    arch = "DummyGoodModelForCausalLM"
    register_model(arch, DummyGoodModel)

    # Check tpu_inference registry
    assert arch in _MODEL_REGISTRY

    class MockModelConfig:

        def __init__(self, architectures):
            self.hf_config = self._MockHfConfig(architectures)
            self.model_impl = "flax_nnx"

        class _MockHfConfig:

            def __init__(self, architectures):
                self.architectures = architectures

    model_config = MockModelConfig(architectures=[arch])
    vllm_compatible_model, _ = ModelRegistry.resolve_model_cls(
        architectures=[arch], model_config=model_config)
    assert vllm_compatible_model is not None
    assert issubclass(vllm_compatible_model, torch.nn.Module)
    assert issubclass(vllm_compatible_model, DummyGoodModel)


try:
    # Attempt to import vLLM's interface validation function
    from vllm.model_executor.models.interfaces_base import is_vllm_model
    VLLM_INTERFACE_CHECK_AVAILABLE = True
except ImportError:
    VLLM_INTERFACE_CHECK_AVAILABLE = False


@pytest.mark.skipif(not VLLM_INTERFACE_CHECK_AVAILABLE,
                    reason="is_vllm_model could not be imported from vllm.")
def test_registered_model_passes_vllm_interface_check(cleanup_registries):
    """
    Ensures the wrapped model passes vLLM's own interface validation.

    This test is future-proof. If vLLM adds new requirements to its
    model interface, this test will fail, signaling that the wrapper
    in `register_model` needs to be updated.
    """
    arch = "DummyGoodModelForCausalLM"
    register_model(arch, DummyGoodModel)

    class MockModelConfig:

        def __init__(self, architectures):
            self.hf_config = self._MockHfConfig(architectures)
            self.model_impl = "flax_nnx"

        class _MockHfConfig:

            def __init__(self, architectures):
                self.architectures = architectures

    model_config = MockModelConfig(architectures=[arch])
    vllm_compatible_model, _ = ModelRegistry.resolve_model_cls(
        architectures=[arch], model_config=model_config)

    # This directly uses vLLM's checker, so it's always up-to-date.
    # We assume is_vllm_model returns True for a valid model, and either
    # returns False or raises an exception for an invalid one.
    assert is_vllm_model(vllm_compatible_model)


def _run_server_and_bench(model_name: str, model_impl_type: str,
                          port: int) -> float:
    env = os.environ.copy()
    env["MODEL_IMPL_TYPE"] = model_impl_type

    # Start server
    server_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model_name,
        "--port",
        str(port),
        "--max-model-len",
        "2048",
        "--tensor-parallel-size",
        "1",
        "--disable-log-requests",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.90",
    ]

    print(f"Starting server ({model_impl_type}) on port {port}...")
    # Use a new process group so we can kill the server and its children
    # Use temporary files for stdout/stderr to avoid pipe buffer deadlocks
    stdout_file = tempfile.TemporaryFile(mode='w+b')
    stderr_file = tempfile.TemporaryFile(mode='w+b')
    server_process = subprocess.Popen(server_cmd,
                                      env=env,
                                      stdout=stdout_file,
                                      stderr=stderr_file,
                                      preexec_fn=os.setsid)

    try:
        # Wait for server to be ready
        start_time = time.time()
        server_ready = False
        while time.time() - start_time < 600:  # 10 minutes timeout
            try:
                if requests.get(
                        f"http://localhost:{port}/health").status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.RequestException:
                pass

            if server_process.poll() is not None:
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read().decode("utf-8", errors="replace")
                stderr = stderr_file.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Server process exited unexpectedly.\nStdout: {stdout}\nStderr: {stderr}"
                )

            time.sleep(5)

        if not server_ready:
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read().decode("utf-8", errors="replace")
            stderr = stderr_file.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Server failed to start within timeout.\nStdout: {stdout}\nStderr: {stderr}"
            )

        print("Server is ready. Running benchmark...")

        # Run benchmark
        bench_cmd = [
            "vllm", "bench", "serve", "--model", model_name, "--port",
            str(port), "--dataset-name", "random", "--random-input-len", "50",
            "--random-output-len", "128", "--num-prompts", "20"
        ]

        result = subprocess.run(bench_cmd,
                                env=env,
                                capture_output=True,
                                text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Benchmark failed.\nStdout: {result.stdout}\nStderr: {result.stderr}"
            )

        # Parse throughput
        # Output example: "Request throughput (req/s): 12.34"
        match = re.search(r"Request throughput \(req/s\):\s+([\d\.]+)",
                          result.stdout)
        if not match:
            raise ValueError(
                f"Could not parse throughput from output:\n{result.stdout}")

        throughput = float(match.group(1))
        return throughput

    finally:
        print("Stopping server...")
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        server_process.wait()
        stdout_file.close()
        stderr_file.close()
        # Wait for TPU cleanup
        time.sleep(5)


def test_flax_nnx_vs_vllm_performance():
    """
    Compares the performance of flax_nnx and vllm model implementations.

    This test ensures that the JAX-native (`flax_nnx`) implementation's
    performance is not significantly different from the vLLM-native PyTorch
    (`vllm`) implementation. It measures the request throughput for both
    backends and asserts that the percentage
    difference is within a reasonable threshold.
    """
    model_name = "Qwen/Qwen3-4B"
    # This should be 2-3% but 6% reduces flakiness.
    percentage_difference_threshold = 0.06

    throughput_vllm = _run_server_and_bench(model_name, "vllm", 8001)
    throughput_flax = _run_server_and_bench(model_name, "flax_nnx", 8002)

    print(f"vLLM (PyTorch) throughput: {throughput_vllm:.2f} req/s.")
    print(f"flax_nnx (JAX) throughput: {throughput_flax:.2f} req/s.")

    percentage_diff = abs(throughput_flax - throughput_vllm) / throughput_vllm
    print(f"Percentage difference in throughput: {percentage_diff:.2%}.")

    assert percentage_diff < percentage_difference_threshold, (
        f"The performance difference between flax_nnx and vllm is too high. "
        f"Difference: {percentage_diff:.2%}, Threshold: {percentage_difference_threshold:.2%}"
    )
