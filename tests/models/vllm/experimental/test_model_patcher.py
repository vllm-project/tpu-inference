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

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torchax
from torchax.interop import jax_view, torch_view

from tpu_inference.models.vllm.experimental.model_patcher import patch_mm_model
from tpu_inference.runner.mm_encoder_jit_manager import \
    _TorchaxEncoderModelAdapter


class _DummyEncoder(nn.Module):

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.register_buffer("scale", torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * self.scale


class _DummyVisionTower(nn.Module):

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.encoder = _DummyEncoder(hidden_size)


class _DummyVllmModel(nn.Module):

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.vision_tower = _DummyVisionTower(hidden_size)

    def encoder_cudagraph_forward(
            self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vision_tower.encoder(inputs["pixel_values"])

    def encoder_eager_forward(self,
                              inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.vision_tower.encoder(inputs["pixel_values"])


class _DummyRunner(nn.Module):

    def __init__(self, vllm_model: nn.Module):
        super().__init__()
        self.vllm_model = vllm_model

    def forward(self, **kwargs):
        method = getattr(self.vllm_model, kwargs["call_method"])
        return method(*kwargs.get("call_args", ()),
                      **kwargs.get("call_kwargs", {}))


def _build_patched_runner_and_params(hidden_size: int = 8):
    vllm_model = _DummyVllmModel(hidden_size)
    runner = _DummyRunner(vllm_model)
    params_and_buffers = {
        "vllm_model.vision_tower.encoder.proj.weight":
        torch_view(jnp.full((hidden_size, hidden_size), 2.0,
                            dtype=jnp.float32)),
        "vllm_model.vision_tower.encoder.proj.bias":
        torch_view(jnp.full((hidden_size, ), 0.5, dtype=jnp.float32)),
        "vllm_model.vision_tower.encoder.scale":
        torch_view(jnp.full((hidden_size, ), 3.0, dtype=jnp.float32)),
    }
    runner, params_and_buffers = patch_mm_model(
        runner,
        params_and_buffers,
        jitted_mm_module_keys=["model.vision_tower.encoder"],
        register_mm_module_custom_pytree_classes=[],
    )
    return runner, jax_view(params_and_buffers)


def test_patch_mm_model_does_not_inline_weights_in_outer_jit():
    hidden_size = 8
    runner, params_jax = _build_patched_runner_and_params(hidden_size)
    adapter = _TorchaxEncoderModelAdapter(runner.vllm_model, runner,
                                          params_jax)

    values_jax = {
        "pixel_values": jnp.ones((4, hidden_size), dtype=jnp.float32)
    }
    with torchax.default_env():
        hlo_text = adapter._jit_forward.lower(params_jax, values_jax).as_text()

    assert "stablehlo.constant dense<2.000000e+00> : tensor<8x8xf32>" not in hlo_text
    assert "stablehlo.constant dense<5.000000e-01> : tensor<8xf32>" not in hlo_text
    assert "stablehlo.constant dense<3.000000e+00> : tensor<8xf32>" not in hlo_text


def test_patch_mm_model_respects_functional_call_parameters():
    hidden_size = 4
    runner, params_jax = _build_patched_runner_and_params(hidden_size)
    adapter = _TorchaxEncoderModelAdapter(runner.vllm_model, runner,
                                          params_jax)

    x_torch = {
        "pixel_values": torch.ones((2, hidden_size), dtype=torch.float32)
    }
    out_jit = adapter.run_budget_forward(x_torch)
    out_eager = adapter.encoder_eager_forward(x_torch)
    np.testing.assert_allclose(np.asarray(out_jit), 25.5, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out_eager), 25.5, rtol=1e-5)

    with torchax.default_env():
        x_direct = torch_view(jnp.ones((2, hidden_size), dtype=jnp.float32))
        out_direct = jax_view(runner.vllm_model.vision_tower.encoder(x_direct))
    np.testing.assert_allclose(np.asarray(out_direct), 25.5, rtol=1e-5)

    updated_params = dict(params_jax)
    updated_params[
        "vllm_model.vision_tower.encoder._model.proj.weight"] = jnp.full(
            (hidden_size, hidden_size), 1.5, dtype=jnp.float32)
    updated_params["vllm_model.vision_tower.encoder._model.scale"] = jnp.full(
        (hidden_size, ), 2.0, dtype=jnp.float32)
    adapter._params = updated_params
    out_jit_updated = adapter.run_budget_forward(x_torch)
    out_eager_updated = adapter.encoder_eager_forward(x_torch)
    np.testing.assert_allclose(np.asarray(out_jit_updated), 13.0, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out_eager_updated), 13.0, rtol=1e-5)


def test_patch_mm_model_fallback_ignores_stale_init_torchax_tensor():
    hidden_size = 4
    vllm_model = _DummyVllmModel(hidden_size)
    vllm_model.vision_tower.encoder.proj.weight = nn.Parameter(
        torch_view(
            jnp.full((hidden_size, hidden_size), 99.0, dtype=jnp.float32)),
        requires_grad=False,
    )
    runner = _DummyRunner(vllm_model)
    params_and_buffers = {
        "vllm_model.vision_tower.encoder.proj.weight":
        torch_view(jnp.full((hidden_size, hidden_size), 2.0,
                            dtype=jnp.float32)),
        "vllm_model.vision_tower.encoder.proj.bias":
        torch_view(jnp.full((hidden_size, ), 0.5, dtype=jnp.float32)),
        "vllm_model.vision_tower.encoder.scale":
        torch_view(jnp.full((hidden_size, ), 3.0, dtype=jnp.float32)),
    }
    runner, _ = patch_mm_model(
        runner,
        params_and_buffers,
        jitted_mm_module_keys=["model.vision_tower.encoder"],
        register_mm_module_custom_pytree_classes=[],
    )
    with torchax.default_env():
        x_direct = torch_view(jnp.ones((2, hidden_size), dtype=jnp.float32))
        out_direct = jax_view(runner.vllm_model.vision_tower.encoder(x_direct))
    np.testing.assert_allclose(np.asarray(out_direct), 25.5, rtol=1e-5)
