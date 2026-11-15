# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec

from tpu_inference import envs
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_torch_dtype

logger = init_logger(__name__)


def supports_native_sequence_classification(
        vllm_config,
        architecture: str = "Gemma4ForSequenceClassification") -> bool:
    """Limit native selection to supported LAST classification configurations."""
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    if getattr(hf_config, "architectures", []) != [architecture]:
        return False
    text_config = getattr(hf_config, "text_config", None)
    pooler_config = getattr(model_config, "pooler_config", None)
    sharding_config = getattr(vllm_config, "sharding_config", None)
    return (model_config.runner_type == "pooling"
            and model_config.convert_type == "classify"
            and to_torch_dtype(model_config.dtype) == torch.bfloat16
            and to_torch_dtype(model_config.head_dtype) == torch.float32
            and model_config.quantization is None
            and not getattr(hf_config, "quantization_config", None)
            and not getattr(text_config, "quantization_config", None)
            and not getattr(hf_config, "classifier_from_token", None)
            and not getattr(text_config, "classifier_from_token", None)
            and not getattr(hf_config, "method", None)
            and not getattr(text_config, "method", None)
            and not vllm_config.additional_config.get("quantization")
            and vllm_config.cache_config.cache_dtype in ("auto", "bfloat16")
            and not envs.MOE_REQUANTIZE_WEIGHT_DTYPE
            and not envs.BF16_LINEAR_REQUANTIZE_PATTERNS
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
    """Apply vLLM classifier calibration and activation to device logits.

    ``pools_device_logits`` tells the runner that this pooler consumes the
    ``[num_reqs, num_labels]`` logits produced on device instead of the full
    hidden states, and ``prewarm`` compiles that path at startup.
    """

    pools_device_logits = True

    def __init__(self, head):
        self.head = head

    def prewarm(self, runner) -> None:
        """Warm the request buckets and their small device-to-host transfers."""
        import torchax
        from torchax.interop import torch_view

        hidden_size = runner.model_config.get_hidden_size()
        # Match the PartitionSpec(ATTN_DATA, None) sharding produced by
        # _select_from_array_fn (already precompiled across all token/request
        # bucket pairs by CompilationManager._precompile_select_from_array).
        rows_sharding = NamedSharding(
            runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
        with torchax.default_env():
            for padded_reqs in runner.num_reqs_paddings:
                rows = jax.device_put(
                    jnp.zeros((padded_reqs, hidden_size), dtype=jnp.bfloat16),
                    rows_sharding)
                logits = runner.compute_logits_fn(runner.state_leaves, rows)
                torch_view(logits).to("cpu", non_blocking=False)

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


class NativeSequenceClassificationModel:
    """Mixin for native models with a replicated ``[hidden, labels]`` head.

    ``model_loader`` reads the generic hooks below instead of branching on
    the concrete class: ``supports_vllm_config`` gates native selection,
    ``logits_partition_spec`` keeps the label axis replicated, and
    ``build_pooler`` returns the device-logits pooler.
    """

    logits_partition_spec = PartitionSpec(ShardingAxisName.MLP_DATA, None)

    @classmethod
    def supports_vllm_config(cls, vllm_config) -> bool:
        return supports_native_sequence_classification(vllm_config,
                                                       cls.__name__)

    @classmethod
    def build_pooler(cls, vllm_config) -> NativeSequenceClassificationPooler:
        from vllm.model_executor.layers.pooler.seqwise import \
            pooler_for_classify

        # The head only calibrates and activates the logits computed on
        # device, so no classifier is attached.
        head = pooler_for_classify(vllm_config.model_config.pooler_config,
                                   classifier=None).head
        return NativeSequenceClassificationPooler(head)
