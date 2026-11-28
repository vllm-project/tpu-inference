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
"""Mamba prefix-caching checks that do not depend on the vLLM version.

Up to vllm-project/vllm#63830, vLLM recorded Mamba prefix caching as
`CacheConfig.mamba_cache_mode`: `"align"` when prefix caching was on for a
hybrid (attention + Mamba/linear-attention) model, `"none"` otherwise. #63830
removed the field; vLLM now checks `cache_config.enable_prefix_caching` (and
`model_config.is_hybrid` where the model may not be hybrid).
"""

from typing import Any


def _has_mamba_cache_mode(cache_config: Any) -> bool:
    """Whether `cache_config` carries the pre-#63830 `mamba_cache_mode`."""
    if "mamba_cache_mode" in getattr(type(cache_config),
                                     "__dataclass_fields__", {}):
        return True
    return isinstance(getattr(cache_config, "mamba_cache_mode", None), str)


def mamba_prefix_caching_enabled(cache_config: Any,
                                 model_config: Any = None) -> bool:
    """Whether Mamba prefix caching (the former "align" mode) is on.

    Args:
        cache_config: vLLM `CacheConfig`.
        model_config: vLLM `ModelConfig`. Pass it unless the caller only runs
            for hybrid models: without `mamba_cache_mode`, prefix caching on a
            non-hybrid model does not mean Mamba prefix caching.
    """
    if cache_config is None:
        return False
    if getattr(cache_config, "enable_prefix_caching", False) is not True:
        return False
    if _has_mamba_cache_mode(cache_config):
        return cache_config.mamba_cache_mode == "align"
    return model_config is None or getattr(model_config, "is_hybrid",
                                           False) is True


def set_mamba_prefix_caching(cache_config: Any, enabled: bool) -> None:
    """Sets prefix caching, and the legacy `mamba_cache_mode` if present."""
    cache_config.enable_prefix_caching = enabled
    fields = getattr(type(cache_config), "__dataclass_fields__", None)
    if fields is None or _has_mamba_cache_mode(cache_config):
        # A pre-#63830 `CacheConfig`, an instance carrying the legacy
        # attribute, or a duck-typed stand-in.
        cache_config.mamba_cache_mode = "align" if enabled else "none"


def mamba_manager_params(manager_cls: Any, enabled: bool) -> dict[str, Any]:
    """`manager_cls.layer_config()` kwargs for Mamba prefix caching on/off.

    The `MambaManager.Params` field is `mamba_cache_mode` before
    vllm-project/vllm#63830 and `enable_prefix_caching` after it.
    """
    fields = getattr(manager_cls.Params, "__dataclass_fields__", {})
    if "enable_prefix_caching" in fields:
        return {"enable_prefix_caching": enabled}
    return {"mamba_cache_mode": "align" if enabled else "none"}
