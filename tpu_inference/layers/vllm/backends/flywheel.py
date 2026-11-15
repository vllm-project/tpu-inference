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
"""--attention-backend flywheel: the FlyWheel TPU kernels (flywheel_tpu) for
the attention and the GDN linear attention layers."""

from tpu_inference.layers.common.attention_backend import (
    AttentionBackend, LinearAttentionBackend, register_attention_backend)
from tpu_inference.layers.vllm.backends.flash_attn import (
    PallasAttentionBackend, PallasAttentionBackendImpl)

FLYWHEEL_BACKEND_NAME = "flywheel"


class FlywheelPallasAttentionBackendImpl(PallasAttentionBackendImpl):
    attention_backend = FLYWHEEL_BACKEND_NAME


class FlywheelPallasAttentionBackend(PallasAttentionBackend):
    """The vLLM attention backend TpuPlatform.get_attn_backend_cls gives the
    attention layers for --attention-backend flywheel."""

    @staticmethod
    def get_name() -> str:
        return FLYWHEEL_BACKEND_NAME.upper()

    @staticmethod
    def get_impl_cls() -> type[FlywheelPallasAttentionBackendImpl]:
        return FlywheelPallasAttentionBackendImpl


def create_flywheel_attention_backend() -> AttentionBackend:
    # Note: imported here so that registering the name does not load the
    # flywheel_tpu kernels.
    from tpu_inference.layers.common.flywheel_attention import \
        FlywheelAttentionBackend
    return FlywheelAttentionBackend()


def create_flywheel_linear_attention_backend() -> LinearAttentionBackend:
    from tpu_inference.layers.common.flywheel_linear_attention import \
        FlywheelLinearAttentionBackend
    return FlywheelLinearAttentionBackend()


def register_flywheel_attention_backend() -> None:
    """Make --attention-backend flywheel run the FlyWheel kernels. Called from
    the register_layers vllm.general_plugins entry point."""
    register_attention_backend(
        FLYWHEEL_BACKEND_NAME,
        attention=create_flywheel_attention_backend,
        vllm_backend=f"{__name__}.FlywheelPallasAttentionBackend",
        linear_attention=create_flywheel_linear_attention_backend)
