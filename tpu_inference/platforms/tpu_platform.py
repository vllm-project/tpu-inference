# SPDX-License-Identifier: Apache-2.0

import traceback
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import jax.numpy as jnp
import torch
import vllm.envs as vllm_envs
from vllm.platforms.interface import Platform, PlatformEnum

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger

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
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4"
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
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: "AttentionBackendEnum",
                             attn_selector_config: "AttentionSelectorConfig",
                             **kwargs) -> str:
        logger.debug("Enter TpuPlatform.get_attn_backend_cls")
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # Invoke @register_backend in the module.
        import tpu_inference.layers.vllm.attention  # noqa: F401
        if selected_backend != AttentionBackendEnum.FLASH_ATTN:
            logger.info("Cannot use %s backend on TPU. Setting to FLASH_ATTN.",
                        selected_backend)
            selected_backend = AttentionBackendEnum.FLASH_ATTN
        logger.info("Using %s backend.", selected_backend.name)
        ret = selected_backend.get_path()
        logger.debug("Exit TpuPlatform.get_attn_backend_cls")
        return ret

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        logger.debug("Enter TpuPlatform.get_device_name")
        try:
            if vllm_envs.VLLM_TPU_USING_PATHWAYS:
                # Causes mutliprocess accessing IFRT when calling jax.devices()
                logger.debug("Exit TpuPlatform.get_device_name")
                return "TPU v6 lite"
            else:
                # The tpu_info package, upon being imported, executes
                # _initialize_libtpu_safely(), which attempts to start a new
                # process (process.start()). Python's multiprocessing module
                # forbids starting new processes, resulting in error.
                # So import tpu_info here instead.
                from tpu_info import device
                chip_type, _ = device.get_local_chips()
                ret = f"TPU {chip_type.name}"
                logger.debug("Exit TpuPlatform.get_device_name")
                return ret
        except Exception as e:
            logger.warning(f"Error getting device name: {e}")
            logger.debug("Exit TpuPlatform.get_device_name")
            return 'TPU'

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        logger.debug("Enter TpuPlatform.fp8_dtype")
        if cls.get_device_name().lower() == "tpu v6e":
            logger.info(
                "Automatically using fp8_e5m2 for FP8 KV cache on TPU v6e.")
            logger.debug("Exit TpuPlatform.fp8_dtype")
            return torch.float8_e5m2
        logger.debug("Exit TpuPlatform.fp8_dtype")
        return torch.float8_e4m3fn

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        logger.debug("Enter TpuPlatform.get_device_total_memory")
        logger.debug("Exit TpuPlatform.get_device_total_memory")
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        logger.debug("Enter TpuPlatform.is_async_output_supported")
        logger.debug("Exit TpuPlatform.is_async_output_supported")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        logger.debug("Enter TpuPlatform.get_punica_wrapper")
        logger.debug("Exit TpuPlatform.get_punica_wrapper")
        return "tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU"

    @classmethod
    def get_infinity_values(cls, dtype: jnp.dtype) -> Tuple[float, float]:
        logger.debug("Enter TpuPlatform.get_infinity_values")
        ret = jnp.finfo(dtype).min, jnp.finfo(dtype).max
        logger.debug("Exit TpuPlatform.get_infinity_values")
        return ret

    @classmethod
    def can_update_inplace(cls):
        logger.debug("Enter TpuPlatform.can_update_inplace")
        logger.debug("Exit TpuPlatform.can_update_inplace")
        return False

    @classmethod
    def get_lora_vocab_padding_size(cls) -> int:
        logger.debug("Enter TpuPlatform.get_lora_vocab_padding_size")
        logger.debug("Exit TpuPlatform.get_lora_vocab_padding_size")
        return 1

    @classmethod
    def inference_mode(cls):
        logger.debug("Enter TpuPlatform.inference_mode")
        logger.debug("Exit TpuPlatform.inference_mode")
        return True

    @classmethod
    def _initialize_sharding_config(cls, vllm_config: VllmConfig) -> None:
        logger.debug("Enter TpuPlatform._initialize_sharding_config")
        sharding_config = ShardingConfigManager.from_vllm_config(vllm_config)
        vllm_config.sharding_config = sharding_config
        logger.info("sharding_config: %s", sharding_config)
        logger.info(f"Initialized sharding configuration: {sharding_config}")
        logger.debug("Exit TpuPlatform._initialize_sharding_config")

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        logger.debug("Enter TpuPlatform.check_and_update_config")
        logger.debug(f"Received vllm_config: type {type(vllm_config)} : {vllm_config}")
        logger.info(f"TpuPlatform.check_and_update_config stack trace: {''.join(traceback.format_stack())}")
        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            assert not vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
                "VLLM_ENABLE_V1_MULTIPROCESSING must be 0 when using Pathways(JAX_PLATFORMS=proxy)"
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
        logger.debug(f"compilation_config after update: {compilation_config}")

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 16)
            if vllm_config.model_config:
                from tpu_inference.layers.vllm.attention import \
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
        logger.debug(f"cache_config after update: {cache_config}")
        logger.info(
                f"Using KV cache block size: {cache_config.block_size}")

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
            if parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    "PP on Ray is disabled due to a pending change on vLLM.")
        else:
            logger.warning(
                f"Unknown TPU multihost backend: {multihost_backend}. "
                "Using uniproc_executor.")
            parallel_config.distributed_executor_backend = "uni"
        logger.debug("parallel_config after update: %s", parallel_config)

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"
                           " without setting `--disable_chunked_mm_input`. "
                           "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True
        logger.debug("scheduler_config after update: %s", scheduler_config)

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert kv_transfer_config.kv_connector == "TPUConnector"
        logger.debug("kv_transfer_config after update: %s", kv_transfer_config)
        # Late initialization to avoid circular import.
        from tpu_inference.core.sched.dp_scheduler import \
            update_vllm_config_for_dp_scheduler
        update_vllm_config_for_dp_scheduler(vllm_config)
        logger.debug("Exit TpuPlatform.check_and_update_config")

    @classmethod
    def is_pin_memory_available(cls):
        logger.debug("Enter TpuPlatform.is_pin_memory_available")
        logger.warning("Pin memory is not supported on TPU.")
        logger.debug("Exit TpuPlatform.is_pin_memory_available")
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        logger.debug("Enter TpuPlatform.get_device_communicator_cls")
        logger.debug("Exit TpuPlatform.get_device_communicator_cls")
        return "vllm.distributed.device_communicators.tpu_communicator.TpuCommunicator"  # noqa

    @classmethod
    def use_all_gather(cls) -> bool:
        logger.debug("Enter TpuPlatform.use_all_gather")
        logger.debug("Exit TpuPlatform.use_all_gather")
        return True

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        logger.debug("Enter TpuPlatform.supports_v1")
        # V1 support on TPU is experimental
        logger.debug("Exit TpuPlatform.supports_v1")
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union["SamplingParams", PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        logger.debug("Enter TpuPlatform.validate_request")
        """Raises if this request is unsupported on this platform"""
        from vllm.sampling_params import SamplingParams, SamplingType

        if isinstance(params, SamplingParams):
            if params.sampling_type == SamplingType.RANDOM_SEED:
                logger.debug("Exit TpuPlatform.validate_request")
                raise ValueError("JAX does not support per-request seed.")
        logger.debug("Exit TpuPlatform.validate_request")

    @classmethod
    def is_kv_cache_dtype_supported(cls, kv_cache_dtype: str,
                                    model_config: ModelConfig) -> bool:
        logger.debug("Enter TpuPlatform.is_kv_cache_dtype_supported")
        logger.debug("Exit TpuPlatform.is_kv_cache_dtype_supported")
        return True

    @classmethod
    def use_sync_weight_loader(cls) -> bool:
        logger.debug("Enter TpuPlatform.use_sync_weight_loader")
        """
        Returns if the current platform needs to sync weight loader.
        """
        logger.debug("Exit TpuPlatform.use_sync_weight_loader")
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        logger.debug("Enter TpuPlatform.support_hybrid_kv_cache")
        logger.debug("Exit TpuPlatform.support_hybrid_kv_cache")
        return True
