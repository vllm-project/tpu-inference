# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import jax.numpy as jnp
import vllm.envs as vllm_envs
from torchax.ops.mappings import j2t_dtype
from tpu_info import device
from vllm.inputs import ProcessorInputs, PromptType
from vllm.platforms.interface import Platform, PlatformEnum
from vllm.sampling_params import SamplingParams, SamplingType

from tpu_inference import envs
from tpu_inference.layers.jax.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.registry import _Backend
    from vllm.config import BlockSize, ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
else:
    BlockSize = None
    ModelConfig = None
    VllmConfig = None
    PoolingParams = None
    _Backend = None

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

    supported_quantization: list[str] = [
        "tpu_int8", "compressed-tensors", "awq", "fp8", "mxfp4"
    ]

    additional_env_vars: list[str] = [
        "PHASED_PROFILING_DIR", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND", "VLLM_MLA_DISABLE", "TPU_BACKEND_TYPE"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: "_Backend", head_size: int,
                             dtype: jnp.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool, use_mla: bool,
                             has_sink: bool, use_sparse: bool) -> str:
        from vllm.attention.backends.registry import _Backend
        if selected_backend != _Backend.PALLAS:
            logger.info("Cannot use %s backend on TPU.", selected_backend)

        if use_v1:
            logger.info("Using Pallas V1 backend.")
            return "tpu_inference.layers.vllm.attention.PallasAttentionBackend"
        else:
            logger.info("Using Pallas backend.")
            return "vllm.attention.backends.pallas.PallasAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        try:
            if vllm_envs.VLLM_TPU_USING_PATHWAYS:
                # Causes mutliprocess accessing IFRT when calling jax.devices()
                return "TPU v6 lite"
            else:
                chip_type, _ = device.get_local_chips()
                return f"TPU {chip_type.name}"
        except Exception as e:
            logger.warning(f"Error getting device name: {e}")
            return 'TPU'

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return not vllm_envs.VLLM_USE_V1

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
        if not vllm_envs.VLLM_USE_V1:
            raise RuntimeError("VLLM_USE_V1=1 must be set for JAX backend.")

        if vllm_envs.VLLM_TPU_USING_PATHWAYS:
            assert not vllm_envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
                "VLLM_ENABLE_V1_MULTIPROCESSING must be 0 when using Pathways(JAX_PLATFORMS=proxy)"
            )
        cls._initialize_sharding_config(vllm_config)

        from vllm.config import CompilationMode

        cache_config = vllm_config.cache_config
        # For v0, the default block size is 16.
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = cast(BlockSize, 16)
        compilation_config = vllm_config.compilation_config

        # TPU only supports DYNAMO_TRACE_ONCE compilation level
        # NOTE(xiang): the compilation_config is not used by jax.
        if compilation_config.mode != CompilationMode.DYNAMO_TRACE_ONCE:
            compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE

        if compilation_config.backend == "":
            compilation_config.backend = "openxla"

        # If we use vLLM's model implementation in PyTorch, we should set it with torch version of the dtype.
        impl = envs.MODEL_IMPL_TYPE

        # NOTE(xiang): convert dtype to jnp.dtype
        # NOTE(wenlong): skip this logic for mm model preprocessing
        # For mm model preprocessors, it may need the output dtype to be torch.
        # In order to avoid a PR to vLLM, we postpone the dtype checking during tpu_worker initialization
        if not vllm_config.scheduler_config.is_multimodal_model or impl == "vllm":
            if not isinstance(vllm_config.model_config.dtype, str):
                logger.warning(
                    "The model dtype is not properly set for JAX backend. "
                    "Overwriting it to jnp.bfloat16")
                vllm_config.model_config.dtype = jnp.bfloat16
            else:
                vllm_config.model_config.dtype = _DTYPE.get(
                    vllm_config.model_config.dtype, jnp.bfloat16)

        if impl == "vllm":
            vllm_config.model_config.dtype = j2t_dtype(
                vllm_config.model_config.dtype.dtype)

        if vllm_envs.VLLM_USE_V1:
            # TODO(cuiq): remove this dependency.
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

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config.worker_cls = \
                        "tpu_inference.worker.tpu_worker.TPUWorker"

        multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "").lower()
        if not multihost_backend:  # Single host
            logger.info("Force using UniProcExecutor for JAX on single host.")
            parallel_config.distributed_executor_backend = "uni"
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
            logger.warning("TPU does not support running Multimodal models"\
            " without setting `--disable_chunked_mm_input`. " \
            "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert kv_transfer_config.kv_connector == "TPUConnector"
        # Late initialization to avoid circular import
        from tpu_inference.models.jax.utils.quantization.quantization_utils import \
            update_vllm_config_for_qwix_quantization

        update_vllm_config_for_qwix_quantization(vllm_config)

        from tpu_inference.core.sched.dp_scheduler import \
            update_vllm_config_for_dp_scheduler
        update_vllm_config_for_dp_scheduler(vllm_config)

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
            if params.structured_outputs is not None and not vllm_envs.VLLM_USE_V1:
                raise ValueError("Structured output is not supported on "
                                 f"{cls.device_name} V0.")
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
