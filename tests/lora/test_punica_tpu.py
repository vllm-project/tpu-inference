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

import jax
import pytest
import torch
import torchax

from tpu_inference.lora.torch_punica_tpu import PunicaWrapperTPU

tpu_available = False
try:
    if jax.devices("tpu"):
        tpu_available = True
except RuntimeError:
    pass

pytestmark = pytest.mark.skipif(not tpu_available, reason="No TPU found")


def test_add_lora_embedding():

    class DummyWrapper:

        def __init__(self, idxs):
            self.idxs = idxs

        def _get_token_lora_indices(self, x):
            return self.idxs

    num_tokens = 16
    hidden_size = 128
    max_loras = 9
    max_lora_rank = 8

    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        x = torch.rand(num_tokens, max_lora_rank, device='jax')
        lora_b_stacked = torch.rand(max_loras,
                                    1,
                                    hidden_size,
                                    max_lora_rank,
                                    device='jax')
        y = torch.zeros(num_tokens, hidden_size, device='jax')
        idxs = torch.randint(0,
                             max_loras, (num_tokens, ),
                             device='jax',
                             dtype=torch.int32)

        wrapper = DummyWrapper(idxs)

        actual = PunicaWrapperTPU.add_lora_embedding(wrapper, y.clone(), x,
                                                     lora_b_stacked)

        expected = y.clone()
        for i in range(num_tokens):
            lora_idx = idxs[i].item()
            lora_b = lora_b_stacked[lora_idx, 0]
            expected[i] += torch.matmul(lora_b, x[i])

        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=1e-3)


def test_add_lora_logits():

    class DummyWrapper:

        def __init__(self, idxs):
            self.idxs = idxs

        def _get_sampler_indices(self, x):
            return self.idxs

    num_tokens = 16
    hidden_size = 128
    max_loras = 9
    max_lora_rank = 8

    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        x = torch.rand(num_tokens, hidden_size, device='jax')
        lora_a_stacked = torch.rand(max_loras,
                                    1,
                                    max_lora_rank,
                                    hidden_size,
                                    device='jax')
        lora_b_stacked = torch.rand(max_loras,
                                    1,
                                    hidden_size,
                                    max_lora_rank,
                                    device='jax')
        y = torch.zeros(num_tokens, hidden_size, device='jax')
        idxs = torch.randint(0,
                             max_loras, (num_tokens, ),
                             device='jax',
                             dtype=torch.int32)

        wrapper = DummyWrapper(idxs)

        actual = PunicaWrapperTPU.add_lora_logits(wrapper,
                                                  y.clone(),
                                                  x,
                                                  lora_a_stacked,
                                                  lora_b_stacked,
                                                  scale=1.0)

        expected = y.clone()
        for i in range(num_tokens):
            lora_idx = idxs[i].item()
            lora_a = lora_a_stacked[lora_idx, 0]
            lora_b = lora_b_stacked[lora_idx, 0]

            buffer = torch.matmul(lora_a, x[i])
            expected[i] += torch.matmul(lora_b, buffer)

        torch.testing.assert_close(actual, expected, atol=5e-1, rtol=1e-2)
