# SPDX-License-Identifier: Apache-2.0

import random
from typing import TYPE_CHECKING, Optional, Tuple, Union

import jax.numpy as jnp
import numpy
import torch
import vllm.envs as vllm_envs
from vllm.platforms.interface import Platform, PlatformEnum

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger

# TODO(weiyulin): These dummy ops bypass vLLM's eager CUDA-specific imports during
# Sequence Parallelism initialization. Our TPU SP implementation (see
# vllmQuantLinearConfig) is independent of upstream compilation logic.
# Revisit to see if these imports can be guarded or disabled for TPU.

try:
    import vllm._C  # noqa: F401
except ImportError:
    # Ensure the _C namespace exists
    if not hasattr(torch.ops, "_C"):
        torch.library.define("_C::dummy", "() -> ()")

    def _register_dummy(name: str, schema: str):
        if not hasattr(torch.ops._C, name):
            torch.library.define(f"_C::{name}", schema)
            torch.library.impl(f"_C::{name}", "default",
                               lambda *args, **kwargs: None)

    # Register the ops vLLM expects
    _register_dummy("rms_norm",
                    "(Tensor input, Tensor weight, float epsilon) -> Tensor")
    _register_dummy(
        "fused_add_rms_norm",
        "(Tensor input, Tensor residual, Tensor weight, float epsilon) -> (Tensor, Tensor)"
    )
    _register_dummy(
        "rotary_embedding",
        "(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()"
    )
    _register_dummy("static_scaled_fp8_quant",
                    "(Tensor input, Tensor scale) -> Tensor")
    _register_dummy("dynamic_scaled_fp8_quant",
                    "(Tensor input, Tensor scale) -> Tensor")
    _register_dummy("dynamic_per_token_scaled_fp8_quant",
                    "(Tensor input, Tensor scale) -> Tensor")
    _register_dummy("silu_and_mul", "(Tensor input) -> Tensor")
    _register_dummy(
        "rms_norm_static_fp8_quant",
        "(Tensor input, Tensor weight, Tensor scale, float epsilon) -> Tensor")
    _register_dummy(
        "fused_add_rms_norm_static_fp8_quant",
        "(Tensor input, Tensor residual, Tensor weight, Tensor scale, float epsilon) -> (Tensor, Tensor)"
    )
    _register_dummy(
        "rms_norm_dynamic_per_token_quant",
        "(Tensor input, Tensor weight, Tensor scale, float epsilon) -> Tensor")

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.cache import BlockSize
    from vllm.inputs import ProcessorInputs, PromptType
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams, SamplingType
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    BlockSize = None
    ModelConfig = None
    VllmConfig = None
    PoolingParams = None
    AttentionBackendEnum = None
    SamplingParams = None
    SamplingType = None
    PromptType = None
    ProcessorInputs = None

logger = init_logger(__name__)


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TPU"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "openxla"

    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "awq", "fp8", "gpt_oss_mxfp4",
        "modelopt_fp4"
    ]

    additional_env_vars: list[str] = [
        "PHASED_PROFILING_DIR",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND",
        "VLLM_MLA_DISABLE",
        "TPU_BACKEND_TYPE",
        "NEW_MODEL_DESIGN",
        "MODEL_IMPL_TYPE",
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
        "MOE_REQUANTIZE_BLOCK_SIZE",
        "MOE_REQUANTIZE_WEIGHT_DTYPE",
        "USE_JAX_PROFILER_SERVER",
        "JAX_PROFILER_SERVER_PORT",
        "ENABLE_RS_KERNEL",
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: "AttentionBackendEnum",
                             attn_selector_config: "AttentionSelectorConfig",
                             **kwargs) -> str:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        use_mla = attn_selector_config.use_mla
        if use_mla:
            selected_backend = AttentionBackendEnum.FLASH_ATTN_MLA
        elif selected_backend != AttentionBackendEnum.FLASH_ATTN:
            logger.info("Cannot use %s backend on TPU. Setting to FLASH_ATTN.",
                        selected_backend)
            selected_backend = AttentionBackendEnum.FLASH_ATTN
        logger.info("Using %s backend.", selected_backend.name)
        return selected_backend.get_path()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        try:
            if vllm_envs.VLLM_TPU_USING_PATHWAYS:
                # Causes mutliprocess accessing IFRT when calling jax.devices()
                return "TPU v6 lite"
            else:
                # The tpu_info package, upon being imported, executes
                # _initialize_libtpu_safely(), which attempts to start a new
                # process (process.start()). Python's multiprocessing module
                # forbids starting new processes, resulting in error.
                # So import tpu_info here instead.
                from tpu_info import device
                chip_type, _ = device.get_local_chips()
                return f"TPU {chip_type.name}"
        except Exception as e:
            logger.warning(f"Error getting device name: {e}")
            return 'TPU'

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        if cls.get_device_name().lower() == "tpu v6e":
            logger.info(
                "Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.")
            return torch.float8_e5m2
        return torch.float8_e4m3fn

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU"

    @classmethod
    def get_infinity_values(cls, dtype: jnp.dtype) -> Tuple[float, float]:
        return jnp.finfo(dtype).min, jnp.finfo(dtype).max

    @classmethod
    def can_update_inplace(cls):
        return False

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        return 1

    @classmethod
    def inference_mode(cls):
        return True

    @classmethod
    def _initialize_sharding_config(cls, vllm_config: VllmConfig) -> None:

        sharding_config = ShardingConfigManager.from_vllm_config(vllm_config)
        vllm_config.sharding_config = sharding_config
        logger.info(f"Initialized sharding configuration: {sharding_config}")

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            assert not vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
                "VLLM_ENABLE_V1_MULTIPROCESSING must be 0 when using Pathways(JAX_PLATFORMS=proxy)"
            )

        if vllm_config.model_config and vllm_config.model_config.use_mla:
            if not envs.NEW_MODEL_DESIGN or not vllm_config.additional_config.get(
                    "sharding", {}).get("sharding_strategy", {}).get(
                        "enable_dp_attention", False):
                raise ValueError(
                    "MLA models require both the NEW_MODEL_DESIGN=1 environment "
                    "variable to be set and DP attention set via: --additional_config \'{\"sharding\": {\"sharding_strategy\": {\"enable_dp_attention\": true}}}\'"
                )
        cls._initialize_sharding_config(vllm_config)

        from vllm.config import CompilationMode

        compilation_config = vllm_config.compilation_config

        # TPU only supports DYNAMO_TRACE_ONCE compilation level
        # NOTE(xiang): the compilation_config is not used by jax.
        if compilation_config.mode != CompilationMode.DYNAMO_TRACE_ONCE:
            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and not cache_config.user_specified_block_size:
            if vllm_config.model_config:
                if vllm_config.model_config.use_mla:
                    from tpu_inference.layers.vllm.backends.flash_attn_mla import \
                        PallasMLAttentionBackend
                    cache_config.block_size = PallasMLAttentionBackend.get_page_size(
                        vllm_config)  # type: ignore[assignment]
                else:
                    from tpu_inference.layers.vllm.backends.flash_attn import \
                        PallasAttentionBackend
                    cache_config.block_size = PallasAttentionBackend.get_page_size(
                        vllm_config)  # type: ignore[assignment]
                    min_page_size = PallasAttentionBackend.get_min_page_size(
                        vllm_config)
                    if min_page_size > cache_config.block_size:
                        logger.warning(
                            "Increase the page size from %s to %s to avoid SMEM OOM",
                            cache_config.block_size,
                            min_page_size,
                        )
                        cache_config.block_size = min_page_size  # type: ignore[assignment]
            if envs.USE_BATCHED_RPA_KERNEL and cache_config.block_size < 256:
                cache_config.block_size = 256
            logger.info(
                f"Using KV cache block size: {cache_config.block_size}")

        if cache_config and envs.TPU_MAMBA_SSM_CACHE_DTYPE:
            override = envs.TPU_MAMBA_SSM_CACHE_DTYPE
            current = cache_config.mamba_ssm_cache_dtype
            if current != override:
                logger.info(
                    "TPU_MAMBA_SSM_CACHE_DTYPE=%s overriding "
                    "cache_config.mamba_ssm_cache_dtype (was %r)", override,
                    current)
                cache_config.mamba_ssm_cache_dtype = override

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config.worker_cls = \
                        "tpu_inference.worker.tpu_worker.TPUWorker"

        multihost_backend = envs.TPU_MULTIHOST_BACKEND
        if not multihost_backend:  # Single host
            if parallel_config.pipeline_parallel_size == 1:
                logger.info("Force using UniProcExecutor for JAX on "
                            "single host without pipeline parallelism.")
                parallel_config.distributed_executor_backend = "uni"
            else:
                logger.info("Force using MultiprocExecutor for JAX on "
                            "single host with pipeline parallelism.")
                from tpu_inference.executors.multiproc_executor import \
                    MultiprocExecutor
                parallel_config.distributed_executor_backend = MultiprocExecutor
        elif multihost_backend == "ray":
            from tpu_inference.executors.ray_distributed_executor import \
                RayDistributedExecutor
            parallel_config.distributed_executor_backend = RayDistributedExecutor
            logger.info(
                "Force using RayDistributedExecutor for JAX on multihost.")
        else:
            logger.warning(
                f"Unknown TPU multihost backend: {multihost_backend}. "
                "Using uniproc_executor.")
            parallel_config.distributed_executor_backend = "uni"

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"
                           " without setting `--disable_chunked_mm_input`. "
                           "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            allowed = ("TPUConnector", "TPUConnectorHMA",
                       "TPUOffloadConnector")
            if kv_transfer_config.kv_connector not in allowed:
                raise ValueError(
                    f"Unsupported kv_connector "
                    f"'{kv_transfer_config.kv_connector}' for the TPU "
                    f"platform. Expected one of {allowed}.")
        # Late initialization to avoid circular import.
        from tpu_inference.core.sched.dp_scheduler import \
            update_vllm_config_for_dp_scheduler
        update_vllm_config_for_dp_scheduler(vllm_config)

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        # TODO: TPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.

        # vLLM uses `tensor_parallel_size` to calculate the number of KV heads
        # per partition. When data parallelism is enabled, the global
        # `tensor_parallel_size` (total workers) is larger than the actual
        # `tp_size` used.
        # https://github.com/vllm-project/tpu-inference/blob/618dea5f5c0ca556a6c76a2e1cc130ff6a30893c/tpu_inference/layers/common/sharding.py#L196
        # Use the sharding calculated `tp_size` for block size calculations.
        orig_tp_size = vllm_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.tensor_parallel_size = vllm_config.sharding_config.tp_size
        try:
            if vllm_config.model_config.is_hybrid:
                backend_cls = cls._find_non_ssm_backend(vllm_config)
                if backend_cls is not None:
                    # Align block/mamba sizes for hybrid model (may override
                    # user settings).
                    cls._align_hybrid_block_size(vllm_config, backend_cls)
        finally:
            vllm_config.parallel_config.tensor_parallel_size = orig_tp_size

    @classmethod
    def is_pin_memory_available(cls):
        logger.warning("Pin memory is not supported on TPU.")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        # V1 support on TPU is experimental
        return True

    @classmethod
    def validate_request(
        cls,
        processed_inputs: ProcessorInputs,
        params: Union["SamplingParams", PoolingParams],
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams, SamplingType

        if isinstance(params, SamplingParams):
            if params.sampling_type == SamplingType.RANDOM_SEED:
                raise ValueError("JAX does not support per-request seed.")

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    model_config: ModelConfig) -> bool:
        return True

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        """
        Returns if the current platform needs to sync weight loader.
        """
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def current_device(cls) -> torch.device:
        """
        Get the current device for the current platform.

        This is mostly a placeholder since this method isn't
        currently called from TPU Inference but instead
        from upstream vLLM.  This won't be an issue,
        however, because we'll manually place tensors
        on the TPU device(s).
        """
        return torch.device("cpu")

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        random.seed(seed)
        numpy.random.seed(seed)
