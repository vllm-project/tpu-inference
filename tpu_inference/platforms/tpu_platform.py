# SPDX-License-Identifier: Apache-2.0

import os
import random
from typing import TYPE_CHECKING, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy
import torch
import vllm.envs as vllm_envs
from vllm.platforms.interface import Platform, PlatformEnum

from tpu_inference import envs, tpu_info
from tpu_inference.layers.common.sharding import ShardingConfigManager
from tpu_inference.logger import init_logger

# Monkeypatch torch.accelerator.empty_cache to ignore device_allocator error on TPU.
if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "empty_cache"):
    _orig_empty_cache = torch.accelerator.empty_cache

    def _patched_empty_cache(*args, **kwargs):
        try:
            _orig_empty_cache(*args, **kwargs)
        except RuntimeError as e:
            if "Allocator for jax is not a DeviceAllocator" in str(e):
                pass
            else:
                raise e

    torch.accelerator.empty_cache = _patched_empty_cache

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
    # Bypass torch.compile; torchax defers all compilation to JAX
    simple_compile_backend: str = "eager"

    supported_quantization: list[str] = [
        "compressed-tensors", "auto_awq", "fp8", "gpt_oss_mxfp4",
        "modelopt_fp4", "deepseek_v4_fp8"
    ]

    def set_device(self, device: torch.device) -> None:
        # No-op on TPU since JAX/libtpu handles device management internally.
        pass

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
        "MOE_ALL_GATHER_ACTIVATION_DTYPE",
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
            accelerator_type = tpu_info.get_tpu_type()
            chip_type = accelerator_type.split("-", maxsplit=1)[0].lower()
            chip_type = {
                "tpu7x": "v7x",
                "v5litepod": "v5e",
            }.get(chip_type, chip_type)
            return f"TPU {chip_type}"
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
    def mem_get_info(cls) -> Tuple[int, int]:
        """
        Returns (free_memory, total_memory) in bytes for the specified TPU device.
        """
        # Fetch TPU memory statistics via JAX
        # On TPU SPMD, we need to aggregate both the limit and usage across all
        # local devices because global tensor dimensions are used for budget calculations.
        total_memory = 0
        bytes_in_use = 0
        for d in jax.local_devices():
            stats = d.memory_stats()
            total_memory += stats.get('bytes_limit', 0)
            bytes_in_use += stats.get('bytes_in_use', 0)

        free_memory = total_memory - bytes_in_use
        return free_memory, total_memory

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
    def _resolve_multiprocess_dp(cls, vllm_config: VllmConfig) -> None:
        """vLLM-native multi-process DP only works for online `vllm serve` (which
        sets _api_process_rank to -1) with DP > 1, and not with attention DP or
        on Pathways.
        """
        pc = vllm_config.parallel_config
        if pc.data_parallel_size <= 1:
            return
        enable_dp_attention = vllm_config.additional_config.get(
            "sharding", {}).get("sharding_strategy",
                                {}).get("enable_dp_attention", False)
        incompatible = enable_dp_attention or vllm_envs.VLLM_TPU_USING_PATHWAYS

        requested = envs.TPU_MULTIPROCESS_DP
        if requested is not None:
            if requested and incompatible:
                raise ValueError(
                    "TPU_MULTIPROCESS_DP=1 is not supported with attention DP "
                    "(enable_dp_attention) or on Pathways. Set "
                    "TPU_MULTIPROCESS_DP=0 to use single-process SPMD DP.")
            return

        # Unset: only the `vllm serve` launcher (which sets _api_process_rank to
        # -1) auto-enables it; offline LLM() (rank 0) falls back to SPMD.
        online_serving = getattr(pc, "_api_process_rank", 0) == -1
        os.environ["TPU_MULTIPROCESS_DP"] = ("1" if online_serving
                                             and not incompatible else "0")
        logger.info("Resolved TPU_MULTIPROCESS_DP=%s",
                    os.environ["TPU_MULTIPROCESS_DP"])

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        requested_data_parallel_size = \
            vllm_config.parallel_config.data_parallel_size
        cls._resolve_multiprocess_dp(vllm_config)

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
            # Check if we should use Ray Executor V2 (V1-multiproc compatible)
            if vllm_envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from tpu_inference.executors.ray_distributed_executor_v2 import \
                    RayDistributedExecutorV2
                parallel_config.distributed_executor_backend = RayDistributedExecutorV2
                logger.info(
                    "Force using RayDistributedExecutorV2 for JAX on multihost."
                )
            else:
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
            logger.warning(
                "TPU does not support running Multimodal models"
                " without setting `--disable_chunked_mm_input`. "
                "If you are serving a multimodal model, please explicitly add the "
                "`--disable-chunked-mm-input` flag to your server command to avoid execution failures."
            )

        kv_transfer_config = vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            allowed = ("TPUConnector", "TPUConnectorHMA",
                       "TPUOffloadConnector", "RaidenOffloadConnector")
            if kv_transfer_config.kv_connector not in allowed:
                raise ValueError(
                    f"Unsupported kv_connector "
                    f"'{kv_transfer_config.kv_connector}' for the TPU "
                    f"platform. Expected one of {allowed}.")

        enable_continue_decode = vllm_config.additional_config.get(
            "enable_continue_decode", False)
        from tpu_inference.runner.diffusion.config import (
            GenerationStrategy, resolve_generation_strategy)
        generation_strategy = resolve_generation_strategy(vllm_config)
        enable_block_diffusion = (generation_strategy.strategy
                                  is GenerationStrategy.BLOCK_DIFFUSION)
        if enable_continue_decode and enable_block_diffusion:
            raise ValueError(
                "continue_decode and block_diffusion are mutually exclusive")
        if enable_block_diffusion:
            assert generation_strategy.diffusion is not None
            diffusion = generation_strategy.diffusion
            from tpu_inference.runner.diffusion.request_validation import \
                patch_vllm_input_processor_for_block_diffusion
            patch_vllm_input_processor_for_block_diffusion()
            from tpu_inference.runner.utils import MIN_NUM_SEQS
            diffusion_batch_capacity = (
                scheduler_config.max_num_batched_tokens //
                diffusion.model.block_size)
            if diffusion_batch_capacity < MIN_NUM_SEQS:
                raise ValueError(
                    "block_diffusion requires max_num_batched_tokens to fit "
                    f"the minimum padded batch of {MIN_NUM_SEQS} diffusion "
                    "blocks")
            scheduler_config.max_num_seqs = min(
                scheduler_config.max_num_seqs,
                diffusion_batch_capacity,
            )
            from tpu_inference.core.sched.utils import \
                MULTI_TOKEN_LOOKAHEAD_CONFIG
            vllm_config.additional_config[
                MULTI_TOKEN_LOOKAHEAD_CONFIG] = diffusion.model.block_size - 1
        is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        # Late initialization to avoid circular import.
        from tpu_inference.core.sched.dp_scheduler import \
            update_vllm_config_for_dp_scheduler
        update_vllm_config_for_dp_scheduler(vllm_config)

        if enable_continue_decode or enable_block_diffusion:
            mode = ("continue_decode"
                    if enable_continue_decode else "block_diffusion")
            if parallel_config.pipeline_parallel_size > 1:
                raise ValueError(
                    f"{mode} is not supported with pipeline parallelism")
            if is_pooling_model:
                raise ValueError(f"{mode} is not supported for pooling models")

            if enable_block_diffusion:
                if kv_transfer_config is not None:
                    raise ValueError(
                        "block_diffusion does not support KV transfer")
                if scheduler_config.async_scheduling:
                    raise ValueError(
                        "block_diffusion is not supported with async scheduling"
                    )
                if vllm_config.speculative_config is not None:
                    raise ValueError(
                        "block_diffusion is not supported with speculative decoding"
                    )
                if vllm_config.lora_config is not None:
                    raise ValueError(
                        "block_diffusion is not supported with LoRA")
                if vllm_config.model_config.is_multimodal_model:
                    raise ValueError(
                        "block_diffusion is not supported for multimodal models"
                    )
                total_dp_size = getattr(vllm_config.sharding_config,
                                        "total_dp_size", 1)
                if (requested_data_parallel_size > 1 or
                    (isinstance(total_dp_size, int) and total_dp_size > 1)):
                    raise ValueError(
                        "block_diffusion currently requires data_parallel_size=1"
                    )
                if getattr(scheduler_config, "enable_chunked_prefill", False):
                    raise ValueError(
                        "block_diffusion currently requires chunked prefill "
                        "to be disabled")
                if envs.USE_BATCHED_RPA_KERNEL:
                    raise ValueError(
                        "block_diffusion requires the default RPA kernel")
                hf_config = vllm_config.model_config.hf_config
                text_config = getattr(hf_config, "text_config", hf_config)
                head_dim = getattr(text_config, "head_dim", None)
                if head_dim is None:
                    hidden_size = getattr(text_config, "hidden_size", None)
                    num_heads = getattr(text_config, "num_attention_heads",
                                        None)
                    if hidden_size is not None and num_heads:
                        head_dim = hidden_size // num_heads
                if head_dim == 64:
                    raise ValueError(
                        "block_diffusion is not supported with head_dim=64")
                if getattr(vllm_config.cache_config, "enable_prefix_caching",
                           False):
                    raise ValueError(
                        "block_diffusion is not supported with prefix caching")

            if enable_continue_decode:
                from tpu_inference.core.sched.utils import \
                    patch_vllm_scheduler_for_continue_decode
                patch_vllm_scheduler_for_continue_decode()
            else:
                from tpu_inference.core.sched.utils import \
                    patch_vllm_scheduler_for_multi_token_decode
                patch_vllm_scheduler_for_multi_token_decode()

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None:
        # TODO: TPU still sets block_size in check_and_update_config.
        # Move that logic here so block_size is chosen by the backend.
        logger.info(f"Using cache_config.block_size: "
                    f"{vllm_config.cache_config.block_size} "
                    f"instead of overriding with _align_hybrid_block_size() "
                    f"since we set mamba_page_size_padded in "
                    f"kv_cache_manager.py")
        pass

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
