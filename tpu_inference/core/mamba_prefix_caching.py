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
"""Mamba prefix caching helpers for vLLM with or without `mamba_cache_mode`.

Older vLLM sets `CacheConfig.mamba_cache_mode` to "align" (on) or "none" (off).
Newer vLLM drops it and uses `enable_prefix_caching` on hybrid models.
"""

from typing import Any


def _has_mamba_cache_mode(cache_config: Any) -> bool:
    """Whether `cache_config` has the legacy `mamba_cache_mode` field."""
    if "mamba_cache_mode" in getattr(type(cache_config),
                                     "__dataclass_fields__", {}):
        return True
    return isinstance(getattr(cache_config, "mamba_cache_mode", None), str)


def mamba_prefix_caching_enabled(cache_config: Any,
                                 model_config: Any = None) -> bool:
    """Whether Mamba prefix caching is on.

    Pass `model_config` unless the model is known to be hybrid.
    """
    if cache_config is None:
        return False
    if _has_mamba_cache_mode(cache_config):
        return cache_config.mamba_cache_mode == "align"
    if getattr(cache_config, "enable_prefix_caching", False) is not True:
        return False
    return model_config is None or getattr(model_config, "is_hybrid",
                                           False) is True


def set_mamba_prefix_caching(cache_config: Any, enabled: bool) -> None:
    """Sets prefix caching, and the legacy `mamba_cache_mode` if present."""
    cache_config.enable_prefix_caching = enabled
    fields = getattr(type(cache_config), "__dataclass_fields__", None)
    if fields is None or "mamba_cache_mode" in fields:
        cache_config.mamba_cache_mode = "align" if enabled else "none"


def mamba_manager_params(manager_cls: Any, enabled: bool) -> dict[str, Any]:
    """`MambaManager.layer_config()` kwargs for Mamba prefix caching."""
    fields = getattr(manager_cls.Params, "__dataclass_fields__", {})
    if "enable_prefix_caching" in fields:
        return {"enable_prefix_caching": enabled}
    return {"mamba_cache_mode": "align" if enabled else "none"}
