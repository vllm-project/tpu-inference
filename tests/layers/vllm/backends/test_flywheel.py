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
"""--attention-backend flywheel, registered by the register_layers plugin."""

from types import SimpleNamespace

import pytest
from vllm.config.attention import AttentionConfig
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tpu_inference.layers.common import attention_backend
from tpu_inference.layers.common.flywheel_attention import \
    FlywheelAttentionBackend
from tpu_inference.layers.common.flywheel_linear_attention import \
    FlywheelLinearAttentionBackend
from tpu_inference.layers.vllm import register_layers
from tpu_inference.layers.vllm.backends import flywheel
from tpu_inference.platforms.tpu_platform import TpuPlatform


@pytest.fixture(autouse=True)
def registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention_backend, "_REGISTERED_BACKENDS", {})
    register_layers()


def test_the_plugin_lets_vllm_parse_the_name():
    # vLLM runs the plugins in every process, and more than once in one.
    register_layers()

    backend = AttentionConfig.validate_backend_before("flywheel")
    assert backend is AttentionBackendEnum.FLYWHEEL
    assert attention_backend.attention_backend_name(
        SimpleNamespace(attention_config=AttentionConfig(
            backend=backend))) == flywheel.FLYWHEEL


def test_the_name_selects_the_flywheel_backends():
    assert isinstance(
        attention_backend.create_attention_backend(flywheel.FLYWHEEL),
        FlywheelAttentionBackend)
    assert isinstance(
        attention_backend.create_linear_attention_backend(flywheel.FLYWHEEL),
        FlywheelLinearAttentionBackend)


def test_the_attention_layers_run_flywheel():
    path = TpuPlatform.get_attn_backend_cls(AttentionBackendEnum.FLYWHEEL,
                                            SimpleNamespace(use_mla=False))

    vllm_backend = resolve_obj_by_qualname(path)
    assert vllm_backend is flywheel.FlywheelPallasAttentionBackend
    assert vllm_backend.get_impl_cls().attention_backend == flywheel.FLYWHEEL
