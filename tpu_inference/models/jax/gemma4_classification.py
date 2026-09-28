# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any, ClassVar

import jax
from flax import nnx
from jax.sharding import Mesh
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxLinear
from tpu_inference.models.jax.gemma4 import Gemma4Model, init_fn
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.sequence_classification import (
    NativeSequenceClassificationModel, classify_rows)
from tpu_inference.models.jax.utils.weight_utils import (JaxAutoWeightsLoader,
                                                         LoadableWithIterator,
                                                         StandardWeightLoader)


class Gemma4ForSequenceClassification(JaxModule, LoadableWithIterator,
                                      NativeSequenceClassificationModel):
    """Native JAX Gemma-4 sequence-classification model.

    The transformer backbone is identical to :class:`Gemma4ForCausalLM`, but
    the large vocabulary LM head is replaced by the checkpoint's tiny
    ``score.weight`` matrix. LAST-token selection remains runner-owned so it
    can reuse vLLM's chunked-prefill bookkeeping and padded request buckets.
    """

    packed_modules_mapping: ClassVar[dict[str, list[str]]] = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }
    WeightLoader = StandardWeightLoader

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.language_model = Gemma4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model.language_model",
        )

        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        hidden_size = hf_config.text_config.hidden_size
        self.num_labels = int(getattr(hf_config, "num_labels", 2))

        # Replicate the small head. The weight loader transposes the checkpoint
        # score.weight from [labels, hidden] to [hidden, labels].
        self.score = JaxLinear(
            hidden_size,
            self.num_labels,
            use_bias=False,
            param_dtype=model_config.dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
            quant_config=None,
            prefix="score",
        )

    def load_weights(self, weights: Iterable[tuple[str, Any]]):
        # Checkpoint backbone keys are model.language_model.* while the Python
        # root attribute is language_model. score.weight is already a root key
        # and therefore passes through unchanged.
        mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})
        loader = JaxAutoWeightsLoader(
            self,
            skip_substrs=["vision", "audio", "multi_modal"],
        )

        def checked_weights():
            expected = (self.num_labels,
                        self.score.weight.get_value().shape[0])
            for name, weight in mapper.apply(weights):
                # Class labels cannot use the loader's vocabulary padding.
                if name == "score.weight" and tuple(weight.shape) != expected:
                    raise ValueError(
                        f"score.weight must have shape {expected}; "
                        f"got {tuple(weight.shape)}")
                yield name, weight

        return loader.load_weights(checked_weights())

    def __call__(
        self,
        kv_caches: list[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: jax.Array | None = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> tuple[list[jax.Array], jax.Array | JaxIntermediateTensors,
               list[jax.Array], jax.Array | None]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]

        layer_name_to_kv_cache = dict(
            _layer_name_to_kv_cache) if _layer_name_to_kv_cache else None
        kv_caches, hidden_states, expert_indices = self.language_model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
            is_multimodal=None,
        )

        if not is_last_rank:
            hidden_states = JaxIntermediateTensors(
                tensors={"hidden_states": hidden_states})

        return kv_caches, hidden_states, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        """Return raw FP32 class logits; calibration belongs to the pooler."""
        return classify_rows(hidden_states, self.score.weight.get_value())
