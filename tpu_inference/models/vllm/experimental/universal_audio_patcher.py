# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Universal runtime patcher to register custom multimodal audio model support into vLLM on TPUs."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP, MultiModalEmbeddings
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix, AutoWeightsLoader
from vllm.sequence import IntermediateTensors


class GenericAudioProcessor:
    """Universal mock processor interceptor to satisfy framework warmup requirements for remote code repositories lacking standard upstream Processing classes."""
    def __init__(self, config: Any, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self.target_kwarg = getattr(self.config, "audio_kwarg_name", "audio_features")

    def _merge_kwargs(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
        merged = dict(kwargs)
        for arg in args:
            if isinstance(arg, dict):
                merged.update(arg)
            elif hasattr(arg, "keys"):
                try:
                    merged.update(dict(arg))
                except Exception:
                    pass
        return merged

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from transformers import BatchFeature
        inputs = {}
        text = kwargs.get("text")
        if text is None and len(args) > 0:
            text = args[0]

        if text is not None and hasattr(self.tokenizer, "__call__"):
            try:
                if not isinstance(text, list):
                    text = [text]
                inputs.update(self.tokenizer(text))
            except Exception:
                pass

        audio_input = (
            kwargs.get("audios")
            or kwargs.get("audio")
            or kwargs.get("audio_features")
            or kwargs.get("input_features")
            or kwargs.get("input_values")
        )
        if audio_input is None:
            for arg in args:
                if isinstance(arg, torch.Tensor) or (isinstance(arg, list) and len(arg) > 0 and isinstance(arg[0], torch.Tensor)):
                    audio_input = arg
                    break

        if audio_input is not None:
            if isinstance(audio_input, list) and len(audio_input) > 0 and isinstance(audio_input[0], torch.Tensor):
                inputs[self.target_kwarg] = torch.stack(audio_input)
            else:
                inputs[self.target_kwarg] = audio_input
        return BatchFeature(inputs)


class GenericAudioProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> GenericAudioProcessor:
        return GenericAudioProcessor(self.ctx.get_hf_config(), self.ctx.get_tokenizer())

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        max_inputs = getattr(self.ctx.get_hf_config(), "max_audio_inputs", 1)
        return {"audio": max_inputs}

    def get_num_audio_tokens(self, audio_features: Optional[Any] = None, dummy_seq_len: int = 128) -> int:
        if audio_features is not None:
            if isinstance(audio_features, list):
                audio_features = audio_features[0]
            if isinstance(audio_features, torch.Tensor) and len(audio_features.shape) >= 2:
                # Universal sequence length extraction from feature shape (e.g., [batch, seq_len, dim] or [seq_len, dim])
                return audio_features.shape[-2]
        return dummy_seq_len


class GenericAudioDummyInputsBuilder(BaseDummyInputsBuilder[GenericAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return ""
            
        hf_config = self.info.ctx.get_hf_config()
        begin_tag = getattr(hf_config, "media_begin_tag", "<|im_media_begin|>")
        end_tag = getattr(hf_config, "media_end_tag", "<|im_media_end|>")
        
        dummy_block = f"{begin_tag}{end_tag}" if end_tag else begin_tag
        return dummy_block * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, BaseDummyOptions]] = None,
    ) -> Mapping[str, Any]:
        num_audios = mm_counts.get("audio", 0)
        if num_audios == 0:
            return {}

        dummy_seq_len = self.info.get_num_audio_tokens()
        hf_config = self.info.ctx.get_hf_config()
        feature_dim = getattr(hf_config, "hidden_size", 4096)

        # Construct continuous dummy audio feature tensors
        dummy_audio = torch.zeros((num_audios, dummy_seq_len, feature_dim), dtype=torch.float16)
        return {"audio": dummy_audio} if num_audios > 0 else {}


class UniversalAudioMultiModalProcessor(BaseMultiModalProcessor[GenericAudioProcessingInfo]):
    def _validate_mm_kwargs(self, mm_kwargs: Mapping[str, Any], mm_item_counts: Mapping[str, int]) -> None:
        pass

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, Any],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        target_kwarg = getattr(self.info.ctx.get_hf_config(), "audio_kwarg_name", "audio_features")
        return {
            target_kwarg: MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        tokenizer = self.info.ctx.get_tokenizer()
        hf_config = self.info.ctx.get_hf_config()
        begin_tag = getattr(hf_config, "media_begin_tag", "<|im_media_begin|>")
        end_tag = getattr(hf_config, "media_end_tag", "<|im_media_end|>")
        
        media_begin_id = None
        media_end_id = None

        if tokenizer is not None:
            try:
                vocab = tokenizer.get_vocab()
                media_begin_id = vocab.get(begin_tag)
                if end_tag:
                    media_end_id = vocab.get(end_tag)
            except Exception:
                pass

        def get_replacement(item_idx: int):
            num_tokens = self.info.get_num_audio_tokens()

            if media_begin_id is not None:
                full_seq = [media_begin_id] + [0] * num_tokens + ([media_end_id] if media_end_id is not None else [])
                return PromptUpdateDetails.select_token_id(full_seq, 0)
            else:
                full_seq = [0] * num_tokens
                return PromptUpdateDetails.select_token_id(full_seq, 0)

        target = [media_begin_id, media_end_id] if media_begin_id is not None and media_end_id is not None else f"{begin_tag}{end_tag if end_tag else ''}"

        return [
            PromptReplacement(
                modality="audio",
                target=target,
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    UniversalAudioMultiModalProcessor,
    info=GenericAudioProcessingInfo,
    dummy_inputs=GenericAudioDummyInputsBuilder,
)
class UniversalRemoteAudioForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        internal_arch = getattr(config, "internal_architecture_map", ["MiMoForCausalLM"])

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=internal_arch,
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality == "audio":
            return "<|im_media_begin|><|im_media_end|>"
        return None

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Optional[Any]:
        target_kwarg = getattr(self.config, "audio_kwarg_name", "audio_features")
        return (
            kwargs.get(target_kwarg)
            or kwargs.get("audio_features")
            or kwargs.get("input_features")
            or kwargs.get("input_values")
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        try:
            import torchax
            torchax.enable_globally()
        except Exception:
            pass

        audio_features = self._parse_and_validate_audio_input(**kwargs)
        if audio_features is None:
            return []
        if isinstance(audio_features, list):
            return audio_features
        return [audio_features]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        try:
            import torchax
            torchax.enable_globally()
        except Exception:
            pass

        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        try:
            import torchax
            torchax.enable_globally()
        except Exception:
            pass

        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import WeightsMapper

        default_mapping = {
            "model.mimo_layers.": "language_model.model.layers.",
            "mimo_layers.": "language_model.model.layers.",
            "model.mimo_norm.": "language_model.model.norm.",
            "mimo_norm.": "language_model.model.norm.",
            "model.mimo_lm_head.": "language_model.lm_head.",
            "mimo_lm_head.": "language_model.lm_head.",
            "model.vq_adaptor.": None,
            "vq_adaptor.": None,
            "audio_tower.": None,
            "vision_tower.": None,
            "connector.": None,
            "multi_modal_projector.": None,
            "projector.": None,
            "speech_encoder.": None,
            "encoder.": None,
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
        custom_mapping = getattr(self.config, "custom_weights_mapper", default_mapping)

        valid_prefixes = tuple(set(k.split(".")[0] + "." for k in custom_mapping.keys() if k))
        filtered_weights = (
            (name, data)
            for name, data in weights
            if any(name.startswith(p) for p in valid_prefixes)
        )

        mapper = WeightsMapper(orig_to_new_prefix=custom_mapping)
        loader = AutoWeightsLoader(self)
        try:
            import torchax
            torchax.disable_globally()
        except Exception:
            pass

        try:
            loaded = loader.load_weights(filtered_weights, mapper=mapper)
            try:
                import torchax
                torchax.enable_globally()
            except Exception:
                pass
            return loaded
        except Exception as e:
            import sys
            print(f"CRITICAL ERROR during UniversalRemoteAudio weight loading: {e}", file=sys.stderr)
            raise e


def apply_global_tpu_audio_patches():
    """Apply global environment monkeypatches mandatory for running JAX audio models on TPUs."""
    # 1. Prevent JAX dynamic concretization tracing crashes during XLA compilation
    try:
        import vllm.model_executor.models.utils
        vllm.model_executor.models.utils.cast_overflow_tensors = lambda x: x
        import vllm.model_executor.models.whisper
        if hasattr(vllm.model_executor.models.whisper, 'cast_overflow_tensors'):
            vllm.model_executor.models.whisper.cast_overflow_tensors = lambda x: x
    except Exception:
        pass

    # 2. Ensure robust audio container decoding using soundfile (float32 mono-aligned), bypassing broken PyAV FFmpeg container codecs
    try:
        import io
        import soundfile as sf
        import vllm.multimodal.media.audio
        
        def custom_load_bytes(self, data: bytes):
            audio, sr = sf.read(io.BytesIO(data), dtype='float32')
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio, float(sr)
            
        vllm.multimodal.media.audio.AudioMediaIO.load_bytes = custom_load_bytes
    except Exception as e:
        import logging
        logging.warning(f"Failed to monkeypatch AudioMediaIO.load_bytes: {e}")


def register_remote_audio_model(arch_name: str, wrapper_class: Any = UniversalRemoteAudioForConditionalGeneration):
    """Explicitly register a remote audio model architecture string into vLLM's ModelRegistry at runtime."""
    try:
        from vllm.model_executor.models import ModelRegistry
        ModelRegistry.register_model(arch_name, wrapper_class)
    except Exception as e:
        import logging
        logging.warning(f"Failed to register {arch_name} with ModelRegistry: {e}")
