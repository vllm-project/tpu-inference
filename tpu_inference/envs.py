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
    TPU_MULTIPROCESS_DP: bool | None = None
    PREFILL_SLICES: str = ""
    DECODE_SLICES: str = ""
    SKIP_JAX_PRECOMPILE: bool = False
    VLLM_XLA_CHECK_RECOMPILATION: bool = False
    MODEL_IMPL_TYPE: str = "auto"
    DRAFT_MODEL_IMPL_TYPE: str = "auto"
    NEW_MODEL_DESIGN: bool = False
    PHASED_PROFILING_DIR: str = ""
    AGGREGATED_STATS_DIR: str = ""
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
    MOE_REQUANTIZE_WEIGHT_DTYPE: str = ""
    MOE_REQUANTIZE_CLIP_PERCENTILE: float | None = None
    ATTN_BUCKETIZED_NUM_REQS: bool = False
    ATTN_CUSTOM_NUM_REQS_BUCKETS: list[int] = []
    LAYOUT_Q_PROJ_AS_NDH: bool = False
    USE_JAX_PROFILER_SERVER: bool = False
    JAX_PROFILER_SERVER_PORT: int = 9999
    USE_BATCHED_RPA_KERNEL: bool = False
    USE_BATCHED_RPA_SEQ_ON_LANE: bool = False
    FORCE_MOE_RANDOM_ROUTING: bool = False
    JITTED_MM_MODULE_KEYS: list[str] = []
    REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES: list[str] = []
    MOE_ALL_GATHER_ACTIVATION_DTYPE: str = ""
    TPU_OFFLOAD_SKIP_JAX_PRECOMPILE: bool = False
    TPU_OFFLOAD_DECODE_SAVE: bool = False
    TPU_OFFLOAD_NUM_CPU_CHUNKS: int = 1024
    TPU_OFFLOAD_NUM_STAGING_BLOCKS: int = 128
    TPU_OFFLOAD_SAVE_THREADS: int = 1
    TPU_OFFLOAD_BATCHED_SAVE: bool = False
    TPU_OFFLOAD_METRICS_LOG_INTERVAL: int = 5
    TPU_OFFLOAD_USE_UNPINNED_HOST: bool = False
    TPU_OFFLOAD_BLOCK_SIZE_BUCKETS: list[int] = []
    MOE_APPROX_TOPK: bool = False
    MOE_APPROX_TOPK_RECALL_TARGET: float | None = None
    VLLM_TPU_PATCH_MM_EMBEDDINGS: bool = False
    ENABLE_RS_KERNEL: bool = False
    NUM_PRECOMPILE_WORKERS: int = 1
    DP_SCHED_BATCH_PREFILL: bool = False
    DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS: int = 10000
    VLLM_MOE_CHUNK_SIZE: int = 0
    ONEHOT_MOE_PERMUTE_THRESHOLD: int = 0
    PROFILE_SINGLE_DEVICE: bool = False
    LORA_MODULE_PATH: str = ""
    SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES: str = "auto"
    SLICE_ROPE_CACHE: bool = False
    MIN_TOKEN_BUCKET: int = 16
    MOE_ROUTE_PADDING_TO_EXPERT0: bool = False
    VLLM_TPU_BUCKET_PADDING_GAP: int = 0
    VLLM_INCREMENTAL_FP8_LOADING: bool = False
    TPU_MESH_SORT_BY_COORDS: bool = False


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


def env_bool(env_name: str,
             default: bool | None = False,
             requires: list[str] | None = None) -> Callable[[], bool | None]:
    """
    Accepts both numeric strings ("0", "1") and boolean strings
    ("true", "false", "True", "False").

    Args:
        env_name: Name of the environment variable
        default: Default value if not set. Pass None for a tri-state flag
            (unset -> None) that callers resolve themselves.
        requires: List of environment variables that must be set if this is True.
    """

    def _get_bool_env() -> bool | None:
        value = os.getenv(env_name)
        if value is None or value == "":
            parsed_value = default
        else:
            value_lower = value.lower()
            if value_lower in ("true", "1"):
                parsed_value = True
            elif value_lower in ("false", "0"):
                parsed_value = False
            else:
                raise ValueError(
                    f"Invalid boolean value '{value}' for {env_name}. "
                    f"Valid options: '0', '1', 'true', 'false', 'True', 'False'."
                )

        if parsed_value and requires:
            for req in requires:
                if not os.getenv(req):
                    raise ValueError(
                        f"{env_name} can only be set if {req} is set.")

        return parsed_value

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


def env_int_list(env_name: str) -> Callable[[], list[int]]:
    """
    Accepts a comma-separated string and returns a list of strings.

    Args:
        env_name: Name of the environment variable
        default: Default list of strings if not set
    """

    def _get_int_list_env() -> list[int]:
        value = os.getenv(env_name)
        if value is None or value == "":
            return []

        return [int(v.strip()) for v in value.split(",")]

    return _get_int_list_env


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
    # Use vLLM-native multi-process data parallelism (one engine process per
    # DP rank, single load-balanced API endpoint) instead of tpu-inference's
    # single-process SPMD data parallelism. Each DP rank is pinned to a
    # disjoint set of TPU chips. Unset (None) means "auto", and
    # TpuPlatform.check_and_update_config resolves it to a concrete value
    # (on for online `vllm serve` with DP > 1; off for offline, attention DP,
    # and Pathways).
    "TPU_MULTIPROCESS_DP":
    env_bool("TPU_MULTIPROCESS_DP", default=None),
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
    lambda: os.getenv("MOE_REQUANTIZE_WEIGHT_DTYPE", ""),
    # Specify requantization block size for MoE weights
    "MOE_REQUANTIZE_BLOCK_SIZE":
    lambda: int(block_size)
    if (block_size := os.getenv("MOE_REQUANTIZE_BLOCK_SIZE")) else None,
    # Clip outlier weights before requantization at the given percentile
    # (e.g. 99.9). Reduces quantization error for large block sizes by
    # preventing extreme outliers from inflating the per-block scale.
    "MOE_REQUANTIZE_CLIP_PERCENTILE":
    lambda: float(pct)
    if (pct := os.getenv("MOE_REQUANTIZE_CLIP_PERCENTILE")) else None,
    # By default, it only use max_reqs for attentions. But if set true, it
    # will precompile max_reqs to power-of-twos between min and max reqs,
    # and attention will have the num_reqs closer to actual num_reqs. This
    # makes attention more efficient for num_reqs less than max_reqs but at
    # the cost of longer model precompilation time.
    "ATTN_BUCKETIZED_NUM_REQS":
    env_bool("ATTN_BUCKETIZED_NUM_REQS"),
    # ATTN_BUCKETIZED_NUM_REQS set to true but the compilation time is too
    # long, user can set a list of custom buckets (num_reqs to precompile)
    # separated by comma. The max_reqs will alwasy be added to the buckets
    "ATTN_CUSTOM_NUM_REQS_BUCKETS":
    env_int_list("ATTN_CUSTOM_NUM_REQS_BUCKETS"),
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
    "USE_BATCHED_RPA_SEQ_ON_LANE":
    env_bool("USE_BATCHED_RPA_SEQ_ON_LANE"),
    # Force random expert routing in MoE layers (for testing purposes only)
    "FORCE_MOE_RANDOM_ROUTING":
    env_bool("FORCE_MOE_RANDOM_ROUTING", default=False),
    "JITTED_MM_MODULE_KEYS":
    env_str_list("JITTED_MM_MODULE_KEYS"),
    "REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES":
    env_str_list("REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES"),
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
    # kv offload to dram: Whether to use unpinned_host for KV cache tensors on host dram.
    "TPU_OFFLOAD_USE_UNPINNED_HOST":
    lambda: bool(int(os.getenv("TPU_OFFLOAD_USE_UNPINNED_HOST", "0"))),
    "AGGREGATED_STATS_DIR":
    lambda: os.getenv("AGGREGATED_STATS_DIR", ""),
    # kv offload to dram: buckets of sizes for pre-compilation
    "TPU_OFFLOAD_BLOCK_SIZE_BUCKETS":
    lambda: env_int_list("TPU_OFFLOAD_BLOCK_SIZE_BUCKETS")
    () or [1, 2, 4, 8, 16, 32, 64],
    # MoE: whether to use approximate top-k for expert selection.
    # Enabling this may speedup the expert selection at the risk of accuracy loss.
    "MOE_APPROX_TOPK":
    env_bool("MOE_APPROX_TOPK", default=False),
    # MoE: the target recall rate for approximate top-k expert selection.
    # A higher rate increases accuracy at the cost of slower speed.
    # A lower rate can speedup expert selection at the risk of higher accuracy loss.
    "MOE_APPROX_TOPK_RECALL_TARGET":
    lambda: float(os.getenv("MOE_APPROX_TOPK_RECALL_TARGET", "0.9")),
    "DISABLE_WEIGHT_REQUANTIZATION":
    env_bool("DISABLE_WEIGHT_REQUANTIZATION", default=False),
    "VLLM_TPU_PATCH_MM_EMBEDDINGS":
    env_bool("VLLM_TPU_PATCH_MM_EMBEDDINGS", default=False),
    "DISABLE_MLA_Q_ACTIVATION_QUANTIZATION":
    env_bool("DISABLE_MLA_Q_ACTIVATION_QUANTIZATION", default=False),
    # Enable hierarchical reduce-scatter kernel for MoE
    "ENABLE_RS_KERNEL":
    env_bool("ENABLE_RS_KERNEL", default=False),
    # Number of worker threads for parallel XLA precompilation.
    "NUM_PRECOMPILE_WORKERS":
    lambda: int(os.getenv("NUM_PRECOMPILE_WORKERS") or "1"),
    # DP scheudler: hold and batch incoming requests (prefills) to
    # cluster and dispatch prefills together.
    "DP_SCHED_BATCH_PREFILL":
    env_bool("DP_SCHED_BATCH_PREFILL", default=True),
    # DP scheduler: timeout (ms) to force flush pending requests.
    "DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS":
    lambda: int(os.getenv("DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS", "30000")),
    "MLA_XPOSE_N_TILE_SIZE":
    lambda: int(os.getenv("MLA_XPOSE_N_TILE_SIZE", "160")),
    "VLLM_MOE_CHUNK_SIZE":
    lambda: int(os.getenv("VLLM_MOE_CHUNK_SIZE", "0")),
    # Use Onehot+Matmul for permute and unpermute before and after moe
    # when the batch size <= this threshold. When set to 0, this feature
    # is effectively disabled.
    "ONEHOT_MOE_PERMUTE_THRESHOLD":
    lambda: int(os.getenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "0")),
    # Profile a single device instead of all devices.
    "PROFILE_SINGLE_DEVICE":
    env_bool("PROFILE_SINGLE_DEVICE", default=False),
    "LORA_MODULE_PATH":
    lambda: os.getenv("LORA_MODULE_PATH", ""),
    "MLA_KV_PACKING_SIZE":
    lambda: int(os.getenv("MLA_KV_PACKING_SIZE", "32")),
    # When set to a value, override XLA SparseCore offload minimum size (in Bytes) for all-reduce
    # and all-gather. When set to 0, use default XLA offload threshold. When set to auto,
    # use VMEM size as the threshold.
    "SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES":
    lambda: os.getenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", "auto"),
    # Slice the rotary cos_sin_cache to max_model_len at load (saves HBM and a
    # per-step layout copy of the full max_position_embeddings table). Applies
    # to text / 1-D RoPE only; ignored for MRoPE models, whose (video)
    # positions can exceed max_model_len.
    "SLICE_ROPE_CACHE":
    env_bool("SLICE_ROPE_CACHE", default=False),
    "MLA_TRANSPOSE_KV_CACHE":
    env_bool("MLA_TRANSPOSE_KV_CACHE", default=False),
    # Minimum max num of batched tokens.
    "MIN_TOKEN_BUCKET":
    lambda: int(os.getenv("MIN_TOKEN_BUCKET") or "16"),
    # Route padding tokens to expert 0 instead of picking other experts, to
    # avoid activating unneeded experts and speed up the GMM kernel by not
    # loading unnecessary weights. Only applies when DP attention size is 1
    # (pure TP attention, e.g. TP8_EP), since under DP attention the padding
    # is interleaved per rank and a single valid-token count cannot describe it.
    "MOE_ROUTE_PADDING_TO_EXPERT0":
    env_bool("MOE_ROUTE_PADDING_TO_EXPERT0", default=False),
    # Gap between token-bucket padding sizes for TPU precompilation. When 0,
    # buckets grow as powers of two; otherwise buckets increase by this gap
    # once past the power-of-two ramp. Previously provided by vllm.envs, which
    # removed it upstream as an orphaned var, so it now lives here.
    "VLLM_TPU_BUCKET_PADDING_GAP":
    lambda: int(os.getenv("VLLM_TPU_BUCKET_PADDING_GAP", "0")),
    # Sort devices by coords and core_on_chip when constructing a tpu mesh.
    # Currently, it only supports a single host set up.
    "TPU_MESH_SORT_BY_COORDS":
    env_bool("TPU_MESH_SORT_BY_COORDS", default=False),
    # Controls whether FP8 layers perform incremental weight loading and sharding.
    "VLLM_INCREMENTAL_FP8_LOADING":
    env_bool("VLLM_INCREMENTAL_FP8_LOADING", default=False),
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
