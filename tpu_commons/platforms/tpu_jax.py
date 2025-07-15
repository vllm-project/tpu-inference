# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import jax.numpy as jnp
import vllm.envs as envs
from tpu_info import device
from vllm.inputs import ProcessorInputs, PromptType
from vllm.platforms.interface import Platform, PlatformEnum, _Backend
from vllm.sampling_params import SamplingParams, SamplingType

from tpu_commons.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import BlockSize, ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
else:
    BlockSize = None
    ModelConfig = None
    VllmConfig = None
    PoolingParams = None

logger = init_logger(__name__)

_DTYPE: dict[str, jnp.dtype] = {
    "bfloat16": jnp.bfloat16,
    "float": jnp.float32,
    "float32": jnp.float32,
}


class TpuPlatform(Platform):
    _enum = PlatformEnum.TPU
    device_name: str = "tpu"
    device_type: str = "tpu"
    dispatch_key: str = "XLA"
    ray_device_key: str = "TPU"
    device_control_env_var: str = "TPU_VISIBLE_CHIPS"
    simple_compile_backend: str = "openxla"

    supported_quantization: list[str] = ["tpu_int8", "compressed-tensors"]

    additional_env_vars: list[str] = [
        "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: jnp.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if (selected_backend != _Backend.PALLAS
                and selected_backend != _Backend.PALLAS_VLLM_V1):
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        if use_v1:
            logger.info("Using Pallas V1 backend.")
            return "vllm.v1.attention.backends.pallas.PallasAttentionBackend"
        else:
            logger.info("Using Pallas backend.")
            return "vllm.attention.backends.pallas.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        chip_type, _ = device.get_local_chips()
        return f"TPU {chip_type.name}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return not envs.VLLM_USE_V1

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_tpu.PunicaWrapperTPU"

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
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if not envs.VLLM_USE_V1:
            raise RuntimeError("VLLM_USE_V1=1 must be set for JAX backend.")

        from vllm.config import CompilationLevel

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 16)
        compilation_config = vllm_config.compilation_config

        # TPU only supports DYNAMO_ONCE compilation level
        if compilation_config.level != CompilationLevel.DYNAMO_ONCE:
            logger.info("[TPU] Forcing DYNAMO_ONCE compilation level")
            compilation_config.level = CompilationLevel.DYNAMO_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        assert vllm_config.speculative_config is None, \
            "TPU does not support speculative decoding"

        # NOTE(xiang): convert dtype to jnp.dtype
        if not isinstance(vllm_config.model_config.dtype, str):
            logger.warning(
                "The model dtype is not properly set for JAX backend. "
                "Overwriting it to jnp.bfloat16")
            vllm_config.model_config.dtype = jnp.bfloat16
        else:
            vllm_config.model_config.dtype = _DTYPE.get(
                vllm_config.model_config.dtype, jnp.bfloat16)

        if envs.VLLM_USE_V1:
            from vllm.v1.attention.backends.pallas import \
                PallasAttentionBackend
            cache_config.block_size = PallasAttentionBackend.get_page_size(
                vllm_config)  # type: ignore[assignment]
            min_page_size = PallasAttentionBackend.get_min_page_size(
                vllm_config)
            if min_page_size > cache_config.block_size:
                logger.warning(
                    "Increase the page size from %s to %s to make sure there's"
                    "no SMEM OOM",
                    cache_config.block_size,
                    min_page_size,
                )
                cache_config.block_size = min_page_size  # type: ignore[assignment]

        if os.getenv("EXP_SCHEDULER") is not None:
            from tpu_commons.core.experimental_scheduler_config import \
                ExperimentalSchedulerConfig
            experimental_scheduler_config = ExperimentalSchedulerConfig.initialize_from_config(
                vllm_config.scheduler_config, {})
            vllm_config.scheduler_config = experimental_scheduler_config

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config.worker_cls = \
                        "tpu_commons.worker.tpu_worker_jax.TPUWorker"

        # TODO(xiang): fix this for multi-host case
        print('===========================')
        print("distributed_executor_backend: ",
              parallel_config.distributed_executor_backend)
        from tpu_commons.executors.tpu_distributed_executor import TpuDistributedExecutor
        parallel_config.distributed_executor_backend = TpuDistributedExecutor
        # if parallel_config.distributed_executor_backend != "uni":
        #     logger.warning(
        #         "JAX requires to use uniproc_executor for single host.")
        #     parallel_config.distributed_executor_backend = "uni"

        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TPU backend")

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

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
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        if isinstance(params, SamplingParams):
            if params.guided_decoding is not None and not envs.VLLM_USE_V1:
                raise ValueError("Structured output is not supported on "
                                 f"{cls.device_name} V0.")
            if params.sampling_type == SamplingType.RANDOM_SEED:
                raise ValueError("JAX does not support per-request seed.")
