# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import torch
from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)


class NativeSequenceClassificationModel:
    """Explicit capability for models with a native replicated score head."""


def supports_native_sequence_classification(vllm_config, mesh: Mesh) -> bool:
    """Limit native selection to supported LAST classification configurations."""
    model_config = vllm_config.model_config
    if getattr(model_config.hf_config, "architectures",
               []) != ["Gemma4ForSequenceClassification"]:
        return False
    pooler_config = getattr(model_config, "pooler_config", None)
    sharding_config = getattr(vllm_config, "sharding_config", None)
    return (
        model_config.runner_type == "pooling"
        and model_config.convert_type == "classify"
        and model_config.dtype == torch.bfloat16
        and model_config.head_dtype == torch.float32
        and model_config.quantization is None
        and not getattr(model_config.hf_config, "quantization_config", None)
        and not getattr(getattr(model_config.hf_config, "text_config", None),
                        "quantization_config", None)
        and not vllm_config.additional_config.get("quantization")
        and vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
        and not envs.MOE_REQUANTIZE_WEIGHT_DTYPE
        and not envs.BF16_LINEAR_REQUANTIZE_PATTERNS
        and get_mesh_shape_product(mesh, ShardingAxisName.MLP_TENSOR) == 1
        and pooler_config is not None
        and pooler_config.task in (None, "classify")
        and pooler_config.seq_pooling_type == "LAST"
        and vllm_config.lora_config is None
        and vllm_config.speculative_config is None
        and getattr(sharding_config, "total_dp_size", None) == 1
        and getattr(sharding_config, "prefill_cp_size", None) == 1
        and vllm_config.parallel_config.pipeline_parallel_size == 1)


def classify_rows(rows: jax.Array, weight: jax.Array) -> jax.Array:
    """Project BF16 hidden rows with FP32 multiplication and accumulation."""
    return jnp.matmul(
        rows.astype(jnp.float32),
        weight.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )


class NativeSequenceClassificationPooler:
    """Apply vLLM classifier calibration and activation to device logits."""

    def __init__(self, model, pooler, hidden_size: int):
        from vllm.model_executor.layers.pooler import DispatchPooler
        from vllm.model_executor.layers.pooler.seqwise import (
            ClassifierPoolerHead, LastPool, SequencePooler)

        classify_pooler = (pooler.poolers_by_task.get("classify")
                           if isinstance(pooler, DispatchPooler) else None)
        if not (isinstance(classify_pooler, SequencePooler)
                and isinstance(classify_pooler.pooling, LastPool)
                and isinstance(classify_pooler.head, ClassifierPoolerHead)):
            raise TypeError("Native sequence classification requires a LAST "
                            "ClassifierPoolerHead")
        self.head = classify_pooler.head
        score_weight = getattr(getattr(model, "score", None), "weight", None)
        if score_weight is None or score_weight.get_value().ndim != 2:
            raise RuntimeError("Native sequence classifier requires a rank-2 "
                               "score.weight")
        if not score_weight.get_metadata("_is_loaded", False):
            raise RuntimeError("Native sequence classifier score.weight "
                               "was not loaded")
        weight = score_weight.get_value()
        if weight.shape != (hidden_size, model.num_labels):
            raise RuntimeError("Native sequence-classifier score.weight must "
                               f"be [{hidden_size}, {model.num_labels}]; "
                               f"got {weight.shape}")

        logger.info(
            "Native JAX sequence classification enabled: score=%s labels=%d",
            tuple(weight.shape), weight.shape[1])

    def __call__(self, logits_dev: jax.Array, pooling_metadata):
        import torchax
        from torchax.interop import torch_view

        if not all(task == "classify" for task in pooling_metadata.tasks):
            raise RuntimeError("Native sequence classification requires "
                               "classify-only batches")
        num_reqs = len(pooling_metadata.pooling_params)
        if not num_reqs:
            return []
        with torchax.default_env():
            logits = torch_view(logits_dev).to("cpu",
                                               non_blocking=True)[:num_reqs]

        out = self.head(logits, pooling_metadata)
        return list(out.unbind(0)) if isinstance(out,
                                                 torch.Tensor) else list(out)
