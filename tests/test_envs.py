# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import pytest

import tpu_inference.envs as envs
from tpu_inference.envs import enable_envs_cache, environment_variables


def test_getattr_without_cache(monkeypatch: pytest.MonkeyPatch):
    assert envs.JAX_PLATFORMS == ""
    assert envs.PHASED_PROFILING_DIR == ""
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("PHASED_PROFILING_DIR", "/tmp/profiling")
    assert envs.JAX_PLATFORMS == "tpu"
    assert envs.PHASED_PROFILING_DIR == "/tmp/profiling"

    assert envs.TPU_NAME is None
    assert envs.TPU_ACCELERATOR_TYPE is None
    monkeypatch.setenv("TPU_NAME", "my-tpu")
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5litepod-16")
    assert envs.TPU_NAME == "my-tpu"
    assert envs.TPU_ACCELERATOR_TYPE == "v5litepod-16"

    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")


def test_getattr_with_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("TPU_NAME", "my-tpu")

    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    enable_envs_cache()

    # __getattr__ is decorated with functools.cache
    assert hasattr(envs.__getattr__, "cache_info")
    start_hits = envs.__getattr__.cache_info().hits

    # 2 more hits due to JAX_PLATFORMS and TPU_NAME accesses
    assert envs.JAX_PLATFORMS == "tpu"
    assert envs.TPU_NAME == "my-tpu"
    assert envs.__getattr__.cache_info().hits == start_hits + 2

    # All environment variables are cached
    for environment_variable in environment_variables:
        envs.__getattr__(environment_variable)
    assert envs.__getattr__.cache_info(
    ).hits == start_hits + 2 + len(environment_variables)

    # Reset envs.__getattr__ back to non-cached version to
    # avoid affecting other tests
    envs.__getattr__ = envs.__getattr__.__wrapped__


def test_boolean_env_vars(monkeypatch: pytest.MonkeyPatch):
    # Ensure clean environment for boolean vars by setting to default "0"
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "0")
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")
    monkeypatch.setenv("NEW_MODEL_DESIGN", "0")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "0")

    # Test SKIP_JAX_PRECOMPILE (default False)
    assert envs.SKIP_JAX_PRECOMPILE is False
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "1")
    assert envs.SKIP_JAX_PRECOMPILE is True
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "0")
    assert envs.SKIP_JAX_PRECOMPILE is False

    # Test VLLM_XLA_CHECK_RECOMPILATION (default False)
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is False
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "1")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is True
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "0")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is False

    # Test NEW_MODEL_DESIGN (default False)
    assert envs.NEW_MODEL_DESIGN is False
    monkeypatch.setenv("NEW_MODEL_DESIGN", "1")
    assert envs.NEW_MODEL_DESIGN is True

    # Test USE_MOE_EP_KERNEL (default False)
    assert envs.USE_MOE_EP_KERNEL is False
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "1")
    assert envs.USE_MOE_EP_KERNEL is True

    # Test ENABLE_QUANTIZED_MATMUL_KERNEL (default False)
    assert envs.ENABLE_QUANTIZED_MATMUL_KERNEL is False
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    assert envs.ENABLE_QUANTIZED_MATMUL_KERNEL is True


def test_boolean_env_vars_string_values(monkeypatch: pytest.MonkeyPatch):
    """Test that boolean env vars accept string values like 'True' and 'False'"""

    # Test NEW_MODEL_DESIGN with string "True"
    monkeypatch.setenv("NEW_MODEL_DESIGN", "True")
    assert envs.NEW_MODEL_DESIGN is True

    monkeypatch.setenv("NEW_MODEL_DESIGN", "true")
    assert envs.NEW_MODEL_DESIGN is True

    monkeypatch.setenv("NEW_MODEL_DESIGN", "False")
    assert envs.NEW_MODEL_DESIGN is False

    monkeypatch.setenv("NEW_MODEL_DESIGN", "false")
    assert envs.NEW_MODEL_DESIGN is False

    # Test SKIP_JAX_PRECOMPILE with string values
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "True")
    assert envs.SKIP_JAX_PRECOMPILE is True

    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "false")
    assert envs.SKIP_JAX_PRECOMPILE is False

    # Test VLLM_XLA_CHECK_RECOMPILATION with string values
    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "TRUE")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is True

    monkeypatch.setenv("VLLM_XLA_CHECK_RECOMPILATION", "FALSE")
    assert envs.VLLM_XLA_CHECK_RECOMPILATION is False

    # Test USE_MOE_EP_KERNEL with string values
    monkeypatch.setenv("USE_MOE_EP_KERNEL", "true")
    assert envs.USE_MOE_EP_KERNEL is True

    monkeypatch.setenv("USE_MOE_EP_KERNEL", "False")
    assert envs.USE_MOE_EP_KERNEL is False


def test_boolean_env_vars_invalid_values(monkeypatch: pytest.MonkeyPatch):
    """Test that boolean env vars raise errors for invalid values"""

    # Test invalid value for NEW_MODEL_DESIGN
    monkeypatch.setenv("NEW_MODEL_DESIGN", "yes")
    with pytest.raises(
            ValueError,
            match="Invalid boolean value 'yes' for NEW_MODEL_DESIGN"):
        _ = envs.NEW_MODEL_DESIGN

    monkeypatch.setenv("NEW_MODEL_DESIGN", "2")
    with pytest.raises(ValueError,
                       match="Invalid boolean value '2' for NEW_MODEL_DESIGN"):
        _ = envs.NEW_MODEL_DESIGN

    # Test invalid value for SKIP_JAX_PRECOMPILE
    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "invalid")
    with pytest.raises(
            ValueError,
            match="Invalid boolean value 'invalid' for SKIP_JAX_PRECOMPILE"):
        _ = envs.SKIP_JAX_PRECOMPILE


def test_boolean_env_vars_empty_string(monkeypatch: pytest.MonkeyPatch):
    """Test that empty string returns default value"""

    monkeypatch.setenv("NEW_MODEL_DESIGN", "")
    assert envs.NEW_MODEL_DESIGN is False  # Should return default

    monkeypatch.setenv("SKIP_JAX_PRECOMPILE", "")
    assert envs.SKIP_JAX_PRECOMPILE is False  # Should return default


def test_integer_env_vars(monkeypatch: pytest.MonkeyPatch):
    # Ensure clean environment for integer vars by setting to defaults
    monkeypatch.setenv("PYTHON_TRACER_LEVEL", "1")
    monkeypatch.setenv("NUM_SLICES", "1")

    assert envs.PYTHON_TRACER_LEVEL == 1
    monkeypatch.setenv("PYTHON_TRACER_LEVEL", "3")
    assert envs.PYTHON_TRACER_LEVEL == 3
    monkeypatch.setenv("PYTHON_TRACER_LEVEL", "0")
    assert envs.PYTHON_TRACER_LEVEL == 0

    # Test NUM_SLICES (default 1)
    assert envs.NUM_SLICES == 1
    monkeypatch.setenv("NUM_SLICES", "2")
    assert envs.NUM_SLICES == 2
    monkeypatch.setenv("NUM_SLICES", "4")
    assert envs.NUM_SLICES == 4


def test_model_impl_type_choices(monkeypatch: pytest.MonkeyPatch):
    # Test case sensitive choices
    monkeypatch.setenv("MODEL_IMPL_TYPE", "flax_nnx")
    assert envs.MODEL_IMPL_TYPE == "flax_nnx"

    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")
    assert envs.MODEL_IMPL_TYPE == "vllm"


def test_string_env_vars_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("PREFILL_SLICES", raising=False)
    monkeypatch.delenv("DECODE_SLICES", raising=False)

    assert envs.JAX_PLATFORMS == ""
    assert envs.PREFILL_SLICES == ""
    assert envs.DECODE_SLICES == ""
    assert envs.PHASED_PROFILING_DIR == ""


def test_none_default_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)

    assert envs.TPU_ACCELERATOR_TYPE is None
    assert envs.TPU_NAME is None
    assert envs.TPU_WORKER_ID is None


def test_ray_env_vars(monkeypatch: pytest.MonkeyPatch):
    assert envs.RAY_USAGE_STATS_ENABLED == "0"
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "1")
    assert envs.RAY_USAGE_STATS_ENABLED == "1"

    assert envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE == "shm"


def test_invalid_attribute_raises_error():
    with pytest.raises(AttributeError,
                       match="has no attribute 'NONEXISTENT_VAR'"):
        _ = envs.NONEXISTENT_VAR


def test_dir_returns_all_env_vars():
    env_vars = envs.__dir__()
    assert isinstance(env_vars, list)
    assert len(env_vars) == len(environment_variables)
    assert "JAX_PLATFORMS" in env_vars
    assert "TPU_NAME" in env_vars
    assert "SKIP_JAX_PRECOMPILE" in env_vars
    assert "VLLM_XLA_CHECK_RECOMPILATION" in env_vars
    assert "MODEL_IMPL_TYPE" in env_vars


def test_tpu_multihost_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    assert envs.TPU_WORKER_ID == "0"

    monkeypatch.setenv("TPU_MULTIHOST_BACKEND", "ray")
    assert envs.TPU_MULTIHOST_BACKEND == "ray"


def test_disaggregated_serving_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PREFILL_SLICES", "0,1,2,3")
    assert envs.PREFILL_SLICES == "0,1,2,3"

    monkeypatch.setenv("DECODE_SLICES", "4,5,6,7")
    assert envs.DECODE_SLICES == "4,5,6,7"


def test_model_impl_type_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_IMPL_TYPE", raising=False)
    assert envs.MODEL_IMPL_TYPE == "auto"


def test_cache_preserves_values_across_env_changes(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")

    enable_envs_cache()

    assert envs.JAX_PLATFORMS == "tpu"

    # Change environment variable
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    # Cached value should still be "tpu"
    assert envs.JAX_PLATFORMS == "tpu"

    # Reset envs.__getattr__ back to non-cached version
    envs.__getattr__ = envs.__getattr__.__wrapped__

    # Now it should reflect the new value
    assert envs.JAX_PLATFORMS == "cpu"
