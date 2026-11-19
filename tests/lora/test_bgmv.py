# Copyright 2025 Google LLC
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

import jax
import pytest
import torch
import torchax
from torchax.interop import jax_jit

from tpu_inference.lora.torch_lora_ops import bgmv_torch
from tpu_inference.lora.torch_punica_tpu import PunicaWrapperTPU


def test_bgmv_torch():
    num_tokens = 16
    hidden_size = 128
    max_loras = 9
    max_lora_rank = 8

    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        inputs = torch.rand(num_tokens, hidden_size, device='jax')
        loras = torch.rand(max_loras,
                           1,
                           max_lora_rank,
                           hidden_size,
                           device='jax')
        idxs = torch.randint(0, max_loras, (num_tokens, ), device='jax')

        actual = bgmv_torch(inputs, loras, idxs)
        expected = _ref_bgmv_torch(inputs, loras, idxs)
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=1e-3)


def _ref_bgmv_torch(inputs, loras, idxs):
    if len(loras.shape) == 4:
        loras = loras.squeeze(axis=1)

    # Another equivalent ref impl is as the 2 lines below.
    # selected_loras = loras[idxs]
    # return torch.einsum('td,tld->tl', inputs, selected_loras)
    num_tokens, _ = inputs.shape
    outputs = []
    for i in range(num_tokens):
        input = inputs[i]  # [hidden_size]
        lora = loras[idxs[i]]  # [max_lora_rank, hidden_size]
        out = torch.matmul(lora, input)
        outputs.append(out)

    return torch.stack(outputs, axis=0)


@pytest.mark.parametrize("offset_start", [0, 2])
@pytest.mark.parametrize("output_slices", [(3, ), (3, 2)])
@pytest.mark.parametrize("add_inputs", [False, True])
def test_add_expand_preserves_output_slices(offset_start, output_slices,
                                            add_inputs):
    num_tokens, rank, num_loras = 3, 4, 2
    output_size = offset_start + sum(output_slices) + 2
    y = torch.arange(num_tokens * output_size, dtype=torch.float32).reshape(
        num_tokens, output_size) / 8
    x = torch.arange(len(output_slices) * num_tokens * rank,
                     dtype=torch.float32).reshape(len(output_slices),
                                                  num_tokens, rank) / 16
    weights = tuple(
        torch.arange(num_loras * size * rank, dtype=torch.float32).reshape(
            num_loras, 1, size, rank) / 8 for size in output_slices)
    indices = torch.tensor([0, 1, 0], dtype=torch.int32)
    expected = y.clone()
    offset = offset_start
    for slice_idx, size in enumerate(output_slices):
        for token_idx, lora_idx in enumerate(indices.tolist()):
            delta = weights[slice_idx][lora_idx, 0] @ x[slice_idx, token_idx]
            if add_inputs:
                expected[token_idx, offset:offset + size] += delta
            else:
                expected[token_idx, offset:offset + size] = delta
        offset += size

    class DummyWrapper:

        def _get_token_lora_indices(self, inputs):
            return self.indices

    with torchax.default_env(), jax.default_device(jax.devices()[0]), \
            jax.default_matmul_precision("highest"):
        wrapper = DummyWrapper()
        wrapper.indices = indices.to('jax')
        y_device = y.to('jax')

        def expand(y, x, weights):
            return PunicaWrapperTPU.add_expand(wrapper,
                                               y,
                                               x,
                                               weights,
                                               output_slices,
                                               offset_start=offset_start,
                                               add_inputs=add_inputs)

        for apply in (expand, jax_jit(expand)):
            actual = apply(y_device, x.to('jax'),
                           tuple(w.to('jax') for w in weights))
            torch.testing.assert_close(actual.torch(),
                                       expected,
                                       rtol=1e-5,
                                       atol=1e-5)
            torch.testing.assert_close(y_device.torch(), y, rtol=0, atol=0)
