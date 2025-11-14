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
    MODEL_IMPL_TYPE: str = "flax_nnx"
    NEW_MODEL_DESIGN: bool = False
    PHASED_PROFILING_DIR: str = ""
    PYTHON_TRACER_LEVEL: int = 1
    USE_MOE_EP_KERNEL: bool = False
    RAY_USAGE_STATS_ENABLED: str = "0"
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "shm"

environment_variables: dict[str, Callable[[], Any]] = {
    # JAX platform selection (e.g., "tpu", "cpu", "proxy")
    "JAX_PLATFORMS":
    lambda: os.getenv("JAX_PLATFORMS", ""),
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
    lambda: os.getenv("TPU_MULTIHOST_BACKEND", "").lower(),
    # Slice configuration for disaggregated prefill workers
    "PREFILL_SLICES":
    lambda: os.getenv("PREFILL_SLICES", ""),
    # Slice configuration for disaggregated decode workers
    "DECODE_SLICES":
    lambda: os.getenv("DECODE_SLICES", ""),
    # Skip JAX precompilation step during initialization
    "SKIP_JAX_PRECOMPILE":
    lambda: bool(int(os.getenv("SKIP_JAX_PRECOMPILE", "0"))),
    # Model implementation type (e.g., "flax_nnx")
    "MODEL_IMPL_TYPE":
    lambda: os.getenv("MODEL_IMPL_TYPE", "flax_nnx").lower(),
    # Enable new experimental model design
    "NEW_MODEL_DESIGN":
    lambda: bool(int(os.getenv("NEW_MODEL_DESIGN", "0"))),
    # Directory to store phased profiling output
    "PHASED_PROFILING_DIR":
    lambda: os.getenv("PHASED_PROFILING_DIR", ""),
    # Python tracer level for profiling
    "PYTHON_TRACER_LEVEL":
    lambda: int(os.getenv("PYTHON_TRACER_LEVEL", "1")),
    # Use custom expert-parallel kernel for MoE (Mixture of Experts)
    "USE_MOE_EP_KERNEL":
    lambda: bool(int(os.getenv("USE_MOE_EP_KERNEL", "0"))),
    # Enable/disable Ray usage statistics collection
    "RAY_USAGE_STATS_ENABLED":
    lambda: os.getenv("RAY_USAGE_STATS_ENABLED", "0"),
    # Ray compiled DAG channel type for TPU
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    lambda: os.getenv("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "shm"),
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
