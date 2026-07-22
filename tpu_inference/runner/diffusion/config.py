# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class GenerationStrategy(str, Enum):
    AUTOREGRESSIVE = "autoregressive"
    BLOCK_DIFFUSION = "block_diffusion"


class AttentionPolicy(str, Enum):
    BLOCK_CAUSAL = "block_causal"


class LogitAlignment(str, Enum):
    SAME_POSITION = "same_position"
    SHIFTED = "shifted"


class CanvasPolicy(str, Enum):
    ALL_MASKED = "all_masked"
    SEED_AND_MASK = "seed_and_mask"


class PromptRemainderPolicy(str, Enum):
    INCLUDE_IN_FIRST_CANVAS = "include_in_first_canvas"
    REQUIRE_BLOCK_ALIGNED = "require_block_aligned"


class NextBlockPolicy(str, Enum):
    LAST_LOGIT_ANCHOR = "last_logit_anchor"
    ALL_MASKED = "all_masked"


class DiffusionAlgorithm(str, Enum):
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class DiffusionModelSpec:
    name: str
    block_size: int
    mask_token_id: int
    attention_policy: AttentionPolicy
    logit_alignment: LogitAlignment
    canvas_policy: CanvasPolicy
    prompt_remainder_policy: PromptRemainderPolicy
    next_block_policy: NextBlockPolicy
    sub_block_size: int
    supported_algorithms: tuple[DiffusionAlgorithm, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Diffusion model spec name must not be empty")
        if self.block_size < 1:
            raise ValueError("Diffusion block_size must be positive")
        if self.mask_token_id < 0:
            raise ValueError("Diffusion mask_token_id must be non-negative")
        if self.sub_block_size < 1:
            raise ValueError("Diffusion sub_block_size must be positive")
        if self.block_size % self.sub_block_size:
            raise ValueError(
                "Diffusion sub_block_size must divide block_size exactly")
        if not self.supported_algorithms:
            raise ValueError(
                "Diffusion model spec must support at least one algorithm")


@dataclass(frozen=True)
class DiffusionRuntimeConfig:
    algorithm: DiffusionAlgorithm = DiffusionAlgorithm.LOW_CONFIDENCE
    confidence_threshold: float = 0.9
    temperature: float = 0.0
    max_denoise_steps: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "Diffusion confidence_threshold must be between 0 and 1")
        if self.temperature < 0.0:
            raise ValueError("Diffusion temperature must be non-negative")
        if self.max_denoise_steps < 0:
            raise ValueError(
                "Diffusion max_denoise_steps must be non-negative")


@dataclass(frozen=True)
class DiffusionConfig:
    model: DiffusionModelSpec
    runtime: DiffusionRuntimeConfig

    def __post_init__(self) -> None:
        if self.runtime.algorithm not in self.model.supported_algorithms:
            raise ValueError(
                f"Diffusion algorithm {self.runtime.algorithm.value!r} is not "
                f"supported by model adapter {self.model.name!r}")


@dataclass(frozen=True)
class GenerationStrategyConfig:
    strategy: GenerationStrategy
    diffusion: DiffusionConfig | None = None

    def __post_init__(self) -> None:
        has_diffusion = self.diffusion is not None
        expects_diffusion = self.strategy is GenerationStrategy.BLOCK_DIFFUSION
        if has_diffusion != expects_diffusion:
            raise ValueError(
                "A diffusion config is required only for block_diffusion")


DiffusionModelAdapter = Callable[[Any, Mapping[str, Any]], DiffusionModelSpec]
_MODEL_ADAPTERS: dict[str, DiffusionModelAdapter] = {}


def register_diffusion_model_adapter(
    name: str,
    adapter: DiffusionModelAdapter,
) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Diffusion model adapter name must not be empty")
    if normalized in _MODEL_ADAPTERS:
        raise ValueError(
            f"Diffusion model adapter {normalized!r} is already registered")
    _MODEL_ADAPTERS[normalized] = adapter


def _read_int(
    config: Mapping[str, Any],
    key: str,
    hf_config: Any,
    hf_keys: tuple[str, ...],
    *,
    default: int | None = None,
) -> int:
    value = config.get(key)
    if value is None:
        for hf_key in hf_keys:
            value = getattr(hf_config, hf_key, None)
            if value is not None:
                break
    if value is None:
        value = default
    if value is None:
        raise ValueError(
            f"Diffusion config {key!r} is required by the model adapter")
    return int(value)


def _dgr2_adapter(
    hf_config: Any,
    config: Mapping[str, Any],
) -> DiffusionModelSpec:
    block_size = _read_int(config, "block_size", hf_config, ("bd_size", ))
    mask_token_id = _read_int(config, "mask_token_id", hf_config,
                              ("mask_token_id", "mask_id"))
    sub_block_size = _read_int(config,
                               "sub_block_size",
                               hf_config, ("sub_block_size", ),
                               default=min(8, block_size))
    return DiffusionModelSpec(
        name="dgr2",
        block_size=block_size,
        mask_token_id=mask_token_id,
        attention_policy=AttentionPolicy.BLOCK_CAUSAL,
        logit_alignment=LogitAlignment.SHIFTED,
        canvas_policy=CanvasPolicy.SEED_AND_MASK,
        prompt_remainder_policy=(
            PromptRemainderPolicy.INCLUDE_IN_FIRST_CANVAS),
        next_block_policy=NextBlockPolicy.LAST_LOGIT_ANCHOR,
        sub_block_size=sub_block_size,
        supported_algorithms=(DiffusionAlgorithm.LOW_CONFIDENCE, ),
    )


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def resolve_generation_strategy(vllm_config: Any) -> GenerationStrategyConfig:
    additional_config = _as_mapping(
        getattr(vllm_config, "additional_config", None),
        "additional_config",
    )
    raw_strategy = additional_config.get(
        "generation_strategy", GenerationStrategy.AUTOREGRESSIVE.value)
    try:
        strategy = GenerationStrategy(raw_strategy)
    except ValueError as exc:
        supported = ", ".join(item.value for item in GenerationStrategy)
        raise ValueError(
            f"Unsupported generation_strategy {raw_strategy!r}; expected one "
            f"of: {supported}") from exc

    if strategy is GenerationStrategy.AUTOREGRESSIVE:
        return GenerationStrategyConfig(strategy=strategy)

    diffusion_values = _as_mapping(additional_config.get("diffusion"),
                                   "additional_config['diffusion']")
    adapter_name = str(diffusion_values.get("model_adapter",
                                            "")).strip().lower()
    if not adapter_name:
        raise ValueError("block_diffusion requires diffusion.model_adapter")
    try:
        adapter = _MODEL_ADAPTERS[adapter_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODEL_ADAPTERS)) or "<none>"
        raise ValueError(
            f"Unknown diffusion model_adapter {adapter_name!r}; registered "
            f"adapters: {supported}") from exc

    hf_config = getattr(getattr(vllm_config, "model_config", None),
                        "hf_config", None)
    model_spec = adapter(hf_config, diffusion_values)
    runtime = DiffusionRuntimeConfig(
        algorithm=DiffusionAlgorithm(
            diffusion_values.get("algorithm",
                                 DiffusionAlgorithm.LOW_CONFIDENCE.value)),
        confidence_threshold=float(
            diffusion_values.get("confidence_threshold", 0.9)),
        temperature=float(diffusion_values.get("temperature", 0.0)),
        max_denoise_steps=int(diffusion_values.get("max_denoise_steps", 0)),
    )
    return GenerationStrategyConfig(
        strategy=strategy,
        diffusion=DiffusionConfig(model=model_spec, runtime=runtime),
    )


register_diffusion_model_adapter("dgr2", _dgr2_adapter)
