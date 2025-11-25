# tests/e2e/test_model_loader.py

import pytest
import torch
from flax import nnx
from vllm.model_executor.models.registry import ModelRegistry

from tpu_inference.models.common.model_loader import (_MODEL_REGISTRY,
                                                      register_model)


@pytest.fixture(autouse=True)
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


def test_register_model_success():
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
def test_registered_model_passes_vllm_interface_check():
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
