# tests/e2e/test_model_loader.py

import time

import pytest
import torch
from flax import nnx
from vllm import LLM, SamplingParams
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


def _run_inference_and_time(monkeypatch: pytest.MonkeyPatch, model_name: str,
                            model_impl_type: str):
    with monkeypatch.context():
        monkeypatch.setenv("MODEL_IMPL_TYPE", model_impl_type)
        start_time = time.time()
        try:
            llm = LLM(
                model=model_name,
                max_model_len=128,
                tensor_parallel_size=1,
                enable_prefix_caching=False,
                gpu_memory_utilization=0.98,
            )
            prompts = ["Hello, my name is"]
            sampling_params = SamplingParams(max_tokens=16)
            _ = llm.generate(prompts, sampling_params)
        except Exception as e:
            pytest.fail(
                f"{model_impl_type} implementation failed with an exception: {e}"
            )

        end_time = time.time()
        duration = end_time - start_time

        del llm
        import gc
        gc.collect()
        time.sleep(10)  # wait for TPU to be released

        return duration


def test_flax_nnx_vs_vllm_performance(monkeypatch: pytest.MonkeyPatch):
    """
    Compares the performance of flax_nnx and vllm model implementations.

    This test ensures that the JAX-native (`flax_nnx`) implementation's
    performance is not significantly different from the vLLM-native PyTorch
    (`vllm`) implementation. It measures the time taken for model loading and
    a short generation for both backends and asserts that the percentage
    difference is within a reasonable threshold.
    """
    model_name = "Qwen/Qwen3-0.6B"
    # A 10% threshold to avoid flakiness on different machines.
    # This can be adjusted based on typical performance.
    percentage_difference_threshold = 0.1

    duration_vllm = _run_inference_and_time(monkeypatch, model_name, "vllm")
    duration_flax = _run_inference_and_time(monkeypatch, model_name,
                                            "flax_nnx")

    print(f"vLLM (PyTorch) implementation took {duration_vllm:.2f} seconds.")
    print(f"flax_nnx (JAX) implementation took {duration_flax:.2f} seconds.")

    # Calculate the percentage difference
    if duration_vllm == 0:
        # Avoid division by zero if the vLLM part was instantaneous
        # (unlikely, but good practice).
        # In this case, any non-zero duration for flax is a huge difference,
        # but for simplicity, we can pass if both are near-zero.
        assert duration_flax < 1.0, "vLLM was instantaneous, but flax_nnx took over a second."
    else:
        percentage_diff = abs(duration_flax - duration_vllm) / duration_vllm
        print(f"Percentage difference in duration: {percentage_diff:.2%}.")

        assert percentage_diff < percentage_difference_threshold, (
            f"The performance difference between flax_nnx and vllm is too high. "
            f"Difference: {percentage_diff:.2%}, Threshold: {percentage_difference_threshold:.2%}"
        )
