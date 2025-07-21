# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from tpu_commons.di.abstracts import (AbstractKVCacheConfig,
                                      AbstractKVCacheSpec, AbstractLoRARequest,
                                      AbstractModelRunnerOutput,
                                      AbstractSchedulerOutput)
from tpu_commons.di.interfaces import HostInterface
from tpu_commons.worker.base import AbstractTpuWorker

# Mock the abstract types for isolated testing
MockAbstractLoRARequest = MagicMock(spec=AbstractLoRARequest)
MockAbstractSchedulerOutput = MagicMock(spec=AbstractSchedulerOutput)
MockAbstractModelRunnerOutput = MagicMock(spec=AbstractModelRunnerOutput)
MockAbstractKVCacheConfig = MagicMock(spec=AbstractKVCacheConfig)
MockAbstractKVCacheSpec = MagicMock(spec=AbstractKVCacheSpec)


# A concrete class for testing the abstract base class
class ConcreteTPUWorker(AbstractTpuWorker):
    """
    A concrete implementation of AbstractTpuWorker for testing.
    It implements all abstract methods with minimal, verifiable logic.
    """

    def __init__(self, host_interface: HostInterface):
        super().__init__(host_interface)
        self._model = 1
        self.memory_size = 1024 * 1024  # 1 MB
        self.profile_state = None

    def init_device(self):
        pass  # No-op for testing

    def determine_available_memory(self) -> int:
        return self.memory_size

    def execute_model(
        self, scheduler_output: "AbstractSchedulerOutput"
    ) -> "AbstractModelRunnerOutput":
        # Return a mock output if the input is valid
        if scheduler_output:
            return MockAbstractModelRunnerOutput()
        return None

    def profile(self, is_start: bool = True):
        self.profile_state = "started" if is_start else "stopped"

    def add_lora(self, lora_request: "AbstractLoRARequest") -> bool:
        # Mock logic: succeed if a request is provided
        return lora_request is not None

    def load_model(self) -> None:
        pass  # No-op for testing

    def compile_or_warm_up_model(self) -> None:
        pass  # No-op for testing

    def get_model(self):
        return self._model

    def get_kv_cache_spec(self) -> dict[str, "AbstractKVCacheSpec"]:
        # Return a mock spec dictionary
        return {"layer_0": MockAbstractKVCacheSpec()}

    def initialize_from_config(
            self, kv_cache_config: "AbstractKVCacheConfig") -> None:
        pass  # No-op for testing


## Pytest Fixture
@pytest.fixture
def concrete_worker() -> ConcreteTPUWorker:
    """Provides a fresh instance of ConcreteTPUWorker for each test."""
    return ConcreteTPUWorker(host_interface=MagicMock(spec=HostInterface))


## Test Cases
def test_abc_cannot_be_instantiated():
    """
    Verifies that the abstract base class `AbstractTpuWorker` cannot be instantiated.
    This is the expected behavior for an ABC.
    """
    with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class AbstractTpuWorker"):
        # Pass a mock host because the __init__ requires it
        AbstractTpuWorker(MagicMock(spec=HostInterface))


def test_concrete_worker_instantiation(concrete_worker: ConcreteTPUWorker):
    """
    Tests that our concrete implementation can be instantiated successfully.
    """
    assert isinstance(concrete_worker, ConcreteTPUWorker)
    assert isinstance(concrete_worker,
                      AbstractTpuWorker)  # It's also an instance of the ABC


def test_determine_available_memory(concrete_worker: ConcreteTPUWorker):
    """
    Tests the `determine_available_memory` method returns the correct value.
    """
    assert concrete_worker.determine_available_memory() == 1024 * 1024


def test_execute_model(concrete_worker: ConcreteTPUWorker):
    """
    Tests the `execute_model` method's branching logic.
    """
    mock_scheduler_output = MockAbstractSchedulerOutput()
    _ = concrete_worker.execute_model(mock_scheduler_output)

    # Test the case where the input is None
    assert concrete_worker.execute_model(None) is None


def test_profile(concrete_worker: ConcreteTPUWorker):
    """
    Tests that the `profile` method correctly updates its state.
    """
    assert concrete_worker.profile_state is None
    concrete_worker.profile(is_start=True)
    assert concrete_worker.profile_state == "started"
    concrete_worker.profile(is_start=False)
    assert concrete_worker.profile_state == "stopped"


def test_add_lora(concrete_worker: ConcreteTPUWorker):
    """
    Tests the `add_lora` method's mock logic.
    """
    assert concrete_worker.add_lora(MockAbstractLoRARequest()) is True
    assert concrete_worker.add_lora(None) is False


def test_get_model(concrete_worker: ConcreteTPUWorker):
    """
    Tests that `get_model` returns the expected model instance.
    """
    model = concrete_worker.get_model()
    assert model


def test_get_kv_cache_spec(concrete_worker: ConcreteTPUWorker):
    """
    Tests that `get_kv_cache_spec` returns a correctly structured dictionary.
    """
    spec = concrete_worker.get_kv_cache_spec()
    assert isinstance(spec, dict)
    assert "layer_0" in spec


def test_check_health(concrete_worker: ConcreteTPUWorker):
    """
    Tests the non-abstract `check_health` method from the base class.
    It should execute without error and return None.
    """
    assert concrete_worker.check_health() is None


# Test that other methods can be called without raising an exception
def test_noop_methods_run_without_error(concrete_worker: ConcreteTPUWorker):
    """
    Ensures that methods with no-op implementations can be called without error.
    """
    try:
        concrete_worker.init_device()
        concrete_worker.load_model()
        concrete_worker.compile_or_warm_up_model()
        concrete_worker.initialize_from_config(MockAbstractKVCacheConfig())
    except Exception as e:
        pytest.fail(f"A no-op method raised an unexpected exception: {e}")
