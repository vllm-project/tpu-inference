# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

import jax.numpy as jnp
import torch
import vllm.envs as vllm_envs
from tpu_info import device
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
        "PHASED_PROFILING_DIR", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
        "TPU_MULTIHOST_BACKEND", "VLLM_MLA_DISABLE", "TPU_BACKEND_TYPE",
        "NEW_MODEL_DESIGN"
    ]

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: "AttentionBackendEnum",
                             attn_selector_config: "AttentionSelectorConfig",
                             **kwargs) -> str:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # Invoke @register_backend in the module.
        import tpu_inference.layers.vllm.attention  # noqa: F401

        # Allow both FLASH_ATTN and FLASH_ATTN_MLA for TPU
        allowed_backends = {AttentionBackendEnum.FLASH_ATTN}
        has_mla = hasattr(AttentionBackendEnum, 'FLASH_ATTN_MLA')
        if has_mla:
            allowed_backends.add(AttentionBackendEnum.FLASH_ATTN_MLA)

        # If selected_backend is None, try to detect MLA model and select appropriate backend
        if selected_backend is None:
            # Check if model has MLA (kv_lora_rank attribute indicates MLA)
            is_mla_model = False

            # Debug: log what we received
            logger.info("[MLA_DETECT] attn_selector_config=%s, type=%s",
                       attn_selector_config, type(attn_selector_config).__name__)
            logger.info("[MLA_DETECT] kwargs keys=%s", list(kwargs.keys()) if kwargs else None)

            # Try to get model_config from attn_selector_config
            if attn_selector_config is not None:
                logger.info("[MLA_DETECT] attn_selector_config attrs: %s",
                           [a for a in dir(attn_selector_config) if not a.startswith('_')])

                model_config = getattr(attn_selector_config, 'model_config', None)
                logger.info("[MLA_DETECT] model_config=%s", model_config)

                if model_config is not None:
                    logger.info("[MLA_DETECT] model_config attrs: %s",
                               [a for a in dir(model_config) if not a.startswith('_')])

                    hf_config = getattr(model_config, 'hf_config', None)
                    logger.info("[MLA_DETECT] hf_config=%s, type=%s",
                               hf_config, type(hf_config).__name__ if hf_config else None)

                    if hf_config is not None:
                        kv_lora_rank = getattr(hf_config, 'kv_lora_rank', None)
                        logger.info("[MLA_DETECT] kv_lora_rank=%s", kv_lora_rank)
                        is_mla_model = kv_lora_rank is not None

            # Also check kwargs for model config (alternative path)
            if not is_mla_model and kwargs:
                for key in ['model_config', 'vllm_config']:
                    if key in kwargs:
                        cfg = kwargs[key]
                        logger.info("[MLA_DETECT] Found %s in kwargs: %s", key, cfg)
                        if hasattr(cfg, 'hf_config'):
                            hf_config = cfg.hf_config
                            kv_lora_rank = getattr(hf_config, 'kv_lora_rank', None)
                            logger.info("[MLA_DETECT] kwargs path: kv_lora_rank=%s", kv_lora_rank)
                            if kv_lora_rank is not None:
                                is_mla_model = True
                                break
                        elif hasattr(cfg, 'model_config'):
                            model_config = cfg.model_config
                            if hasattr(model_config, 'hf_config'):
                                hf_config = model_config.hf_config
                                kv_lora_rank = getattr(hf_config, 'kv_lora_rank', None)
                                logger.info("[MLA_DETECT] kwargs nested path: kv_lora_rank=%s", kv_lora_rank)
                                if kv_lora_rank is not None:
                                    is_mla_model = True
                                    break

            # Fallback: check environment variable FORCE_MLA_BACKEND
            if not is_mla_model:
                force_mla = envs.FORCE_MLA_BACKEND
                logger.info("[MLA_DETECT] FORCE_MLA_BACKEND env var=%s", force_mla)
                if force_mla:
                    is_mla_model = True
                    logger.info("[MLA_DETECT] Forcing MLA backend via FORCE_MLA_BACKEND=1")

            logger.info("[MLA_DETECT] Final: is_mla_model=%s, has_mla=%s", is_mla_model, has_mla)

            if is_mla_model and has_mla:
                logger.info("Detected MLA model, selecting FLASH_ATTN_MLA backend.")
                selected_backend = AttentionBackendEnum.FLASH_ATTN_MLA
            else:
                logger.info("Backend is None, defaulting to FLASH_ATTN.")
                selected_backend = AttentionBackendEnum.FLASH_ATTN
        elif selected_backend not in allowed_backends:
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

        if scheduler_config.is_multimodal_model and not \
            scheduler_config.disable_chunked_mm_input:
            logger.warning("TPU does not support running Multimodal models"
                           " without setting `--disable_chunked_mm_input`. "
                           "Forcing --disable_chunked_mm_input.")
            scheduler_config.disable_chunked_mm_input = True

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert kv_transfer_config.kv_connector == "TPUConnector"
        # Late initialization to avoid circular import.
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
        params: Union["SamplingParams", PoolingParams],
        processed_inputs: ProcessorInputs,
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
