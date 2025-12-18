# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    JAX_PLATFORMS: str = ""
    TPU_ACCELERATOR_TYPE: str | None = None
    TPU_NAME: str | None = None
    TPU_WORKER_ID: str | None = None
    TPU_MULTIHOST_BACKEND: str = ""
    PREFILL_SLICES: str = ""
    DECODE_SLICES: str = ""
    SKIP_JAX_PRECOMPILE: bool = False
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    MODEL_IMPL_TYPE: str = "auto"
    NEW_MODEL_DESIGN: bool = False
    PHASED_PROFILING_DIR: str = ""
    PYTHON_TRACER_LEVEL: int = 1
    USE_MOE_EP_KERNEL: bool = False
    NUM_SLICES: int = 1
    RAY_USAGE_STATS_ENABLED: str = "0"
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "shm"
    ENABLE_QUANTIZED_MATMUL_KERNEL: bool = False


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(f"Invalid value '{value}' for {env_name}. "
                             f"Valid options: {actual_choices}.")

        return value

    return _get_validated_env


def env_bool(env_name: str, default: bool = False) -> Callable[[], bool]:
    """
    Accepts both numeric strings ("0", "1") and boolean strings
    ("true", "false", "True", "False").

    Args:
        env_name: Name of the environment variable
        default: Default boolean value if not set
    """

    def _get_bool_env() -> bool:
        value = os.getenv(env_name)
        if value is None or value == "":
            return default

        value_lower = value.lower()
        if value_lower in ("true", "1"):
            return True
        elif value_lower in ("false", "0"):
            return False
        else:
            raise ValueError(
                f"Invalid boolean value '{value}' for {env_name}. "
                f"Valid options: '0', '1', 'true', 'false', 'True', 'False'.")

    return _get_bool_env


environment_variables: dict[str, Callable[[], Any]] = {
    # JAX platform selection (e.g., "tpu", "cpu", "proxy")
    "JAX_PLATFORMS":
    lambda: os.getenv("JAX_PLATFORMS", "").lower(),
    # TPU accelerator type (e.g., "v5litepod-16", "v4-8")
    "TPU_ACCELERATOR_TYPE":
    lambda: os.getenv("TPU_ACCELERATOR_TYPE", None),
    # Name of the TPU resource
    "TPU_NAME":
    lambda: os.getenv("TPU_NAME", None),
    # Worker ID for multi-host TPU setups
    "TPU_WORKER_ID":
    lambda: os.getenv("TPU_WORKER_ID", None),
    # Backend for multi-host communication on TPU
    "TPU_MULTIHOST_BACKEND":
    env_with_choices("TPU_MULTIHOST_BACKEND", "", ["ray"]),
    # Slice configuration for disaggregated prefill workers
    "PREFILL_SLICES":
    lambda: os.getenv("PREFILL_SLICES", ""),
    # Slice configuration for disaggregated decode workers
    "DECODE_SLICES":
    lambda: os.getenv("DECODE_SLICES", ""),
    # Skip JAX precompilation step during initialization
    "SKIP_JAX_PRECOMPILE":
    env_bool("SKIP_JAX_PRECOMPILE", default=False),
    # Check for XLA recompilation during execution
    "VLLM_XLA_CHECK_RECOMPILATION":
    env_bool("VLLM_XLA_CHECK_RECOMPILATION", default=False),
    # Model implementation type (e.g., "flax_nnx")
    "MODEL_IMPL_TYPE":
    env_with_choices("MODEL_IMPL_TYPE", "auto",
                     ["auto", "vllm", "flax_nnx", "jetpack"]),
    # Enable new experimental model design
    "NEW_MODEL_DESIGN":
    env_bool("NEW_MODEL_DESIGN", default=False),
    # Directory to store phased profiling output
    "PHASED_PROFILING_DIR":
    lambda: os.getenv("PHASED_PROFILING_DIR", ""),
    # Python tracer level for profiling
    "PYTHON_TRACER_LEVEL":
    lambda: int(os.getenv("PYTHON_TRACER_LEVEL") or "1"),
    # Use custom expert-parallel kernel for MoE (Mixture of Experts)
    "USE_MOE_EP_KERNEL":
    env_bool("USE_MOE_EP_KERNEL", default=False),
    # Number of TPU slices for multi-slice mesh
    "NUM_SLICES":
    lambda: int(os.getenv("NUM_SLICES") or "1"),
    # Enable/disable Ray usage statistics collection
    "RAY_USAGE_STATS_ENABLED":
    lambda: os.getenv("RAY_USAGE_STATS_ENABLED", "0"),
    # Ray compiled DAG channel type for TPU
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    env_with_choices("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "shm", ["shm"]),
    "ENABLE_QUANTIZED_MATMUL_KERNEL":
    lambda: bool(int(os.getenv("ENABLE_QUANTIZED_MATMUL_KERNEL") or "0")),
}


def __getattr__(name: str) -> Any:
    """
    Gets environment variables lazily.

    NOTE: After enable_envs_cache() invocation (which triggered after service
    initialization), all environment variables will be cached.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def enable_envs_cache() -> None:
    """
    Enables caching of environment variables by wrapping the module's __getattr__
    function with functools.cache(). This improves performance by avoiding
    repeated re-evaluation of environment variables.

    NOTE: This should be called after service initialization. Once enabled,
    environment variable values are cached and will not reflect changes to
    os.environ until the process is restarted.
    """
    # Tag __getattr__ with functools.cache
    global __getattr__
    __getattr__ = functools.cache(__getattr__)

    # Cache all environment variables
    for key in environment_variables:
        __getattr__(key)


def __dir__() -> list[str]:
    return list(environment_variables.keys())
