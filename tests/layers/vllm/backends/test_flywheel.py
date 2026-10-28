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
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tpu_inference.layers.common import attention_backend
from tpu_inference.layers.common.flywheel_attention import \
    FlywheelAttentionBackend
from tpu_inference.layers.common.flywheel_linear_attention import \
    FlywheelLinearAttentionBackend
from tpu_inference.layers.vllm import register_layers
from tpu_inference.layers.vllm.backends import flywheel
from tpu_inference.layers.vllm.custom_ops.gdn_attention_op import \
    VllmGatedDeltaNetAttention
from tpu_inference.platforms.tpu_platform import TpuPlatform

FLYWHEEL_VLLM_BACKEND = f"{flywheel.__name__}.FlywheelPallasAttentionBackend"


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
            backend=backend))) == flywheel.FLYWHEEL_BACKEND_NAME


def test_the_name_selects_the_flywheel_backends():
    assert isinstance(
        attention_backend.create_attention_backend(
            flywheel.FLYWHEEL_BACKEND_NAME), FlywheelAttentionBackend)
    assert isinstance(
        attention_backend.create_linear_attention_backend(
            flywheel.FLYWHEEL_BACKEND_NAME), FlywheelLinearAttentionBackend)


def test_the_attention_layers_run_flywheel():
    path = TpuPlatform.get_attn_backend_cls(AttentionBackendEnum.FLYWHEEL,
                                            SimpleNamespace(use_mla=False))

    vllm_backend = resolve_obj_by_qualname(path)
    assert vllm_backend is flywheel.FlywheelPallasAttentionBackend
    assert vllm_backend.get_impl_cls(
    ).attention_backend == flywheel.FLYWHEEL_BACKEND_NAME


def make_vllm_config(
        architecture: str = "Qwen3_5ForConditionalGeneration"
) -> SimpleNamespace:
    return SimpleNamespace(
        attention_config=AttentionConfig(
            backend=flywheel.FLYWHEEL_BACKEND_NAME),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            architectures=[architecture])))


def test_the_gdn_layers_run_flywheel(monkeypatch):
    monkeypatch.setattr(QwenGatedDeltaNetAttention, "__init__",
                        lambda self, config, vllm_config, *args: None)
    layer = VllmGatedDeltaNetAttention.__new__(VllmGatedDeltaNetAttention)

    VllmGatedDeltaNetAttention.__init__(layer, None, make_vllm_config())

    assert type(
        layer.linear_attention_backend) is FlywheelLinearAttentionBackend


def test_register_rejects_a_vllm_backend_name():
    with pytest.raises(ValueError, match="FLASH_ATTN"):
        attention_backend.register_attention_backend(
            "flash_attn",
            attention=FlywheelAttentionBackend,
            vllm_backend=FLYWHEEL_VLLM_BACKEND)


def test_the_platform_rejects_the_name_for_mla():
    with pytest.raises(ValueError, match="does not support MLA"):
        TpuPlatform.get_attn_backend_cls(AttentionBackendEnum.FLYWHEEL,
                                         SimpleNamespace(use_mla=True))


@pytest.mark.parametrize(("env", "value"), [
    ("TPU_MULTIHOST_BACKEND", "ray"),
    ("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
])
def test_rejects_a_config_that_crosses_a_process_boundary(
        monkeypatch, env, value):
    monkeypatch.setenv(env, value)

    with pytest.raises(ValueError, match=f"{env}={value}"):
        attention_backend.check_attention_backend_config(make_vllm_config())


@pytest.mark.parametrize(("impl_type", "architecture"), [
    ("flax_nnx", "Qwen3_5ForConditionalGeneration"),
    ("auto", "Qwen3ForCausalLM"),
])
def test_rejects_a_jax_model(monkeypatch, impl_type, architecture):
    monkeypatch.setenv("MODEL_IMPL_TYPE", impl_type)

    with pytest.raises(ValueError,
                       match=f"MODEL_IMPL_TYPE={impl_type} resolves to "
                       "flax_nnx"):
        attention_backend.check_attention_backend_config(
            make_vllm_config(architecture))


@pytest.mark.parametrize(("impl_type", "architecture"), [
    ("auto", "Qwen3_5ForConditionalGeneration"),
    ("vllm", "Qwen3ForCausalLM"),
])
def test_accepts_a_vllm_model(monkeypatch, impl_type, architecture):
    monkeypatch.setenv("MODEL_IMPL_TYPE", impl_type)
    monkeypatch.setattr(FlywheelAttentionBackend, "check_config",
                        lambda self, vllm_config: None)

    attention_backend.check_attention_backend_config(
        make_vllm_config(architecture))


def test_config_check_runs_the_backend_check(monkeypatch):
    checked = []
    monkeypatch.setattr(FlywheelAttentionBackend, "check_config",
                        lambda self, vllm_config: checked.append(vllm_config))
    vllm_config = make_vllm_config()

    attention_backend.check_attention_backend_config(vllm_config)

    assert checked == [vllm_config]
