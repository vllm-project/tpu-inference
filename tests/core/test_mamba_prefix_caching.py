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

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tpu_inference.core.mamba_prefix_caching import (
    mamba_manager_params, mamba_prefix_caching_enabled,
    set_mamba_prefix_caching)


@dataclasses.dataclass
class _LegacyCacheConfig:
    """`CacheConfig` with `mamba_cache_mode`."""
    enable_prefix_caching: bool = False
    mamba_cache_mode: str = "none"


@dataclasses.dataclass
class _CacheConfig:
    """`CacheConfig` without `mamba_cache_mode`."""
    enable_prefix_caching: bool = False


def _model_config(is_hybrid):
    return SimpleNamespace(is_hybrid=is_hybrid)


@pytest.mark.parametrize("mode, expected", [("align", True), ("none", False)])
def test_legacy_mode_decides(mode, expected):
    cache_config = _LegacyCacheConfig(enable_prefix_caching=True,
                                      mamba_cache_mode=mode)
    assert mamba_prefix_caching_enabled(cache_config,
                                        _model_config(True)) is expected


@pytest.mark.parametrize("enable_prefix_caching, model_config, expected", [
    (True, None, True),
    (True, _model_config(True), True),
    (True, _model_config(False), False),
    (False, _model_config(True), False),
])
def test_prefix_caching_decides_without_legacy_mode(enable_prefix_caching,
                                                    model_config, expected):
    cache_config = _CacheConfig(enable_prefix_caching=enable_prefix_caching)
    assert mamba_prefix_caching_enabled(cache_config, model_config) is expected


def test_none_cache_config():
    assert mamba_prefix_caching_enabled(None) is False


@pytest.mark.parametrize("enabled", [True, False])
def test_set_on_legacy_config(enabled):
    cache_config = _LegacyCacheConfig()
    set_mamba_prefix_caching(cache_config, enabled)
    assert cache_config.enable_prefix_caching is enabled
    assert cache_config.mamba_cache_mode == ("align" if enabled else "none")
    assert mamba_prefix_caching_enabled(cache_config) is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_set_on_config_without_legacy_mode(enabled):
    cache_config = _CacheConfig()
    set_mamba_prefix_caching(cache_config, enabled)
    assert cache_config.enable_prefix_caching is enabled
    assert not hasattr(cache_config, "mamba_cache_mode")
    assert mamba_prefix_caching_enabled(cache_config) is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_set_on_mock(enabled):
    cache_config = MagicMock()
    set_mamba_prefix_caching(cache_config, enabled)
    assert mamba_prefix_caching_enabled(cache_config, MagicMock()) is enabled


def test_unset_mock_is_off():
    assert mamba_prefix_caching_enabled(MagicMock(), MagicMock()) is False


@pytest.mark.parametrize("enabled", [True, False])
def test_manager_params(enabled):

    @dataclasses.dataclass(frozen=True)
    class LegacyParams:
        mamba_cache_mode: str = "none"

    @dataclasses.dataclass(frozen=True)
    class Params:
        enable_prefix_caching: bool = False

    legacy = mamba_manager_params(SimpleNamespace(Params=LegacyParams),
                                  enabled)
    assert legacy == {"mamba_cache_mode": "align" if enabled else "none"}
    current = mamba_manager_params(SimpleNamespace(Params=Params), enabled)
    assert current == {"enable_prefix_caching": enabled}
