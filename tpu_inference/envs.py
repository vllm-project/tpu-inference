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
    DRAFT_MODEL_IMPL_TYPE: str = "auto"
    NEW_MODEL_DESIGN: bool = False
    PHASED_PROFILING_DIR: str = ""
    PYTHON_TRACER_LEVEL: int = 1
    USE_MOE_EP_KERNEL: bool = False
    USE_UNFUSED_MEGABLOCKS: bool = False
    USE_DENSE_MOE: bool = False
    NUM_SLICES: int = 1
    RAY_USAGE_STATS_ENABLED: bool = False
    VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: str = "shm"
    ENABLE_QUANTIZED_MATMUL_KERNEL: bool = False
    REQUANTIZE_BLOCK_SIZE: int | None = None
    REQUANTIZE_WEIGHT_DTYPE: str = "float8_e4m3fn"
    MOE_REQUANTIZE_BLOCK_SIZE: int | None = None
    MOE_REQUANTIZE_WEIGHT_DTYPE: str = "float8_e4m3fn"
    LAYOUT_Q_PROJ_AS_NDH: bool = False
    USE_JAX_PROFILER_SERVER: bool = False
    JAX_PROFILER_SERVER_PORT: int = 9999
    USE_BATCHED_RPA_KERNEL: bool = False
    FORCE_MOE_RANDOM_ROUTING: bool = False
    SC_KERNEL_THRESHOLD: int = 16777216
    SC_KERNEL_COL_CHUNK_SIZE: int = 1024
    JITTED_MM_MODULE_KEYS: list[str] = []
    REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES: list[str] = []
    RAGGED_GATED_DELTA_RULE_IMPL: str = "ragged_gated_delta_rule_chunked"
    MOE_ALL_GATHER_ACTIVATION_DTYPE: str = ""
    TPU_OFFLOAD_SKIP_JAX_PRECOMPILE: bool = False
    TPU_OFFLOAD_DECODE_SAVE: bool = False
    TPU_OFFLOAD_NUM_CPU_CHUNKS: int = 1024
    TPU_OFFLOAD_NUM_STAGING_BLOCKS: int = 128
    TPU_OFFLOAD_SAVE_THREADS: int = 1
    TPU_OFFLOAD_BATCHED_SAVE: bool = False
    TPU_OFFLOAD_METRICS_LOG_INTERVAL: int = 5


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
    allow_csv: bool = False,
) -> Callable[[], str | None]:
    """
    Create a lambda that validates environment variable against allowed choices

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive
        allow_csv: Whether to allow comma-separated values, validating each
            part individually against the choices

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
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_choices = actual_choices

        parts = value.split(",") if allow_csv else [value]
        for part in parts:
            check_part = part.lower() if not case_sensitive else part
            if check_part not in check_choices:
                raise ValueError(f"Invalid value '{part}' for {env_name}. "
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


def env_str_list(env_name: str) -> Callable[[], list[str]]:
    """
    Accepts a comma-separated string and returns a list of strings.

    Args:
        env_name: Name of the environment variable
        default: Default list of strings if not set
    """

    def _get_str_list_env() -> list[str]:
        value = os.getenv(env_name)
        if value is None or value == "":
            return []

        return [v.strip() for v in value.split(",")]

    return _get_str_list_env


environment_variables: dict[str, Callable[[], Any]] = {
    # JAX platform selection (e.g., "tpu", "cpu", "proxy", "proxy,cpu")
    "JAX_PLATFORMS":
    env_with_choices("JAX_PLATFORMS",
                     "", ["", "tpu", "cpu", "proxy"],
                     allow_csv=True),
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
    "DRAFT_MODEL_IMPL_TYPE":
    env_with_choices("DRAFT_MODEL_IMPL_TYPE", "auto",
                     ["auto", "vllm", "flax_nnx"]),
    # Enable 2D tensor parallelism, shard attention heads across multiple axes
    "USE_2D_TP":
    env_bool("USE_2D_TP", default=False),
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
    # Enable megablocks for JAX sparse matmul for MoE (Mixture of Experts)
    # using Unfused weights
    "USE_UNFUSED_MEGABLOCKS":
    env_bool("USE_UNFUSED_MEGABLOCKS", default=False),
    # Enable the dense backend for Jax MoE (Mixture of Experts)
    # NOTE: this is a naive implementation and should not be used in production
    "USE_DENSE_MOE":
    env_bool("USE_DENSE_MOE", default=False),
    # Number of TPU slices for multi-slice mesh
    "NUM_SLICES":
    lambda: int(os.getenv("NUM_SLICES") or "1"),
    # Enable/disable Ray usage statistics collection
    "RAY_USAGE_STATS_ENABLED":
    env_bool("RAY_USAGE_STATS_ENABLED"),
    # Ray compiled DAG channel type for TPU
    "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE":
    env_with_choices("VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE", "shm", ["shm"]),
    "ENABLE_QUANTIZED_MATMUL_KERNEL":
    env_bool("ENABLE_QUANTIZED_MATMUL_KERNEL"),
    # Specify block quantization size
    "REQUANTIZE_BLOCK_SIZE":
    lambda: int(block_size) if
    (block_size := os.getenv("REQUANTIZE_BLOCK_SIZE")) is not None else None,
    # Specify dtype for quantized linear weights
    "REQUANTIZE_WEIGHT_DTYPE":
    lambda: os.getenv("REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn"),
    # Specify dtype for quantized MoE weights
    "MOE_REQUANTIZE_WEIGHT_DTYPE":
    lambda: os.getenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "float8_e4m3fn"),
    # Specify requantization block size for MoE weights
    "MOE_REQUANTIZE_BLOCK_SIZE":
    lambda: int(block_size) if (block_size := os.getenv(
        "MOE_REQUANTIZE_BLOCK_SIZE")) is not None else None,
    # dictates whether to layout q-proj as NDH (q-heads, model dim, head dim)
    # or DNH (model dim, q-heads, head dim), which is the default (False)
    "LAYOUT_Q_PROJ_AS_NDH":
    env_bool("LAYOUT_Q_PROJ_AS_NDH"),
    "USE_JAX_PROFILER_SERVER":
    env_bool("USE_JAX_PROFILER_SERVER"),
    "JAX_PROFILER_SERVER_PORT":
    lambda: int(os.getenv("JAX_PROFILER_SERVER_PORT") or "9999"),
    "USE_BATCHED_RPA_KERNEL":
    env_bool("USE_BATCHED_RPA_KERNEL"),
    # Force random expert routing in MoE layers (for testing purposes only)
    "FORCE_MOE_RANDOM_ROUTING":
    env_bool("FORCE_MOE_RANDOM_ROUTING", default=False),
    "SC_KERNEL_THRESHOLD":
    lambda: int(os.getenv("SC_KERNEL_THRESHOLD") or "16777216"),
    "SC_KERNEL_COL_CHUNK_SIZE":
    lambda: int(os.getenv("SC_KERNEL_COL_CHUNK_SIZE") or "3072"),
    "JITTED_MM_MODULE_KEYS":
    env_str_list("JITTED_MM_MODULE_KEYS"),
    "REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES":
    env_str_list("REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES"),
    "RAGGED_GATED_DELTA_RULE_IMPL":
    env_with_choices("RAGGED_GATED_DELTA_RULE_IMPL",
                     "ragged_gated_delta_rule_chunked", [
                         "ragged_gated_delta_rule_ref",
                         "ragged_gated_delta_rule_chunked", "fused_gdn_kernel"
                     ]),
    "MOE_ALL_GATHER_ACTIVATION_DTYPE":
    lambda: os.getenv("MOE_ALL_GATHER_ACTIVATION_DTYPE", ""),
    # kv offload to dram: skip pre-compiling swap-related jax functions
    "TPU_OFFLOAD_SKIP_JAX_PRECOMPILE":
    lambda: bool(int(os.getenv("TPU_OFFLOAD_SKIP_JAX_PRECOMPILE", "0"))),
    # kv offload to dram: save kv in the decode phase
    "TPU_OFFLOAD_DECODE_SAVE":
    lambda: bool(int(os.getenv("TPU_OFFLOAD_DECODE_SAVE", "0"))),
    # kv offload to dram: dram space size in # of chunks / blocks
    "TPU_OFFLOAD_NUM_CPU_CHUNKS":
    lambda: int(os.getenv("TPU_OFFLOAD_NUM_CPU_CHUNKS", "1024")),
    # kv offload to dram: size of staging buffer (hbm) for swap
    "TPU_OFFLOAD_NUM_STAGING_BLOCKS":
    lambda: int(os.getenv("TPU_OFFLOAD_NUM_STAGING_BLOCKS", "128")),
    # kv offload to dram: number of threads for asynchronous TPU -> CPU data transfer
    "TPU_OFFLOAD_SAVE_THREADS":
    lambda: int(os.getenv("TPU_OFFLOAD_SAVE_THREADS", "1")),
    # kv offload to dram: batch multiple requests' save operations into a single swap call
    "TPU_OFFLOAD_BATCHED_SAVE":
    lambda: bool(int(os.getenv("TPU_OFFLOAD_BATCHED_SAVE", "0"))),
    # kv offload to dram: prometheus metrics log interval in seconds
    "TPU_OFFLOAD_METRICS_LOG_INTERVAL":
    lambda: int(os.getenv("TPU_OFFLOAD_METRICS_LOG_INTERVAL", "10")),
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
