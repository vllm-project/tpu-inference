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

import torch

from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import \
    _patched_flatten_embeddings


def test_patched_flatten_embeddings_2d():
    # Standard 2D embedding tensor (num_tokens, hidden_dim)
    t = torch.ones((5, 10))
    res = _patched_flatten_embeddings(t)

    # Should flatten all but the last dimension.
    # For 2D (5, 10), all but last dim is just dim 0 (size 5).
    # Result shape should be (5, 10).
    assert res.shape == (5, 10)
    assert torch.allclose(res, t)


def test_patched_flatten_embeddings_3d():
    # 3D embedding tensor (batch, num_tokens, hidden_dim)
    t = torch.ones((2, 5, 10))
    res = _patched_flatten_embeddings(t)

    # Should flatten dimensions 0 and 1 -> 2 * 5 = 10.
    # Result shape should be (10, 10).
    assert res.shape == (10, 10)


def test_patched_flatten_embeddings_nested_list():
    # Nested list/tuple of tensors
    t1 = torch.ones((2, 5))
    t2 = torch.ones((3, 5))
    embeddings = [t1, t2]

    res = _patched_flatten_embeddings(embeddings)
    # Should flatten each and concatenate them -> (2+3, 5) = (5, 5)
    assert res.shape == (5, 5)


def test_patched_flatten_embeddings_empty_and_1d():
    # 1D tensor of shape (L,) representing edge cases / 1D sequences
    t = torch.ones((5, ))
    res = _patched_flatten_embeddings(t)

    # ndim = 1. start_dim = 0, end_dim = -2.
    # start_dim = 0. end_dim = -2 + 1 = -1.
    # Returns t.flatten(0, -1) which is a 1D tensor.
    assert res.shape == (5, )

    # Empty 1D tensor
    t_empty = torch.ones((0, ))
    res_empty = _patched_flatten_embeddings(t_empty)
    assert res_empty.shape == (0, )


def test_patched_get_deepstack_none():
    from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import \
        _patched_get_deepstack

    class MockModel:
        pass

    model = MockModel()

    # Case 1: orig_get_deepstack returns None, and there are no cached deepstack tensors
    def orig_get_deepstack(tokens):
        return None

    res = _patched_get_deepstack(model, orig_get_deepstack, 10)
    assert res is None


def test_patched_get_deepstack_with_cache():
    from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import \
        _patched_get_deepstack

    class MockModel:
        _deepstack_tensors = {"deepstack_input_embeds_0": torch.ones((5, 10))}

    model = MockModel()

    # Case 2: cached deepstack tensors exist, orig_get_deepstack should not be called
    def orig_get_deepstack(tokens):
        raise AssertionError("Should not be called")

    res = _patched_get_deepstack(model, orig_get_deepstack, 10)
    assert res is not None
    assert "deepstack_input_embeds_0" in res.tensors


def test_patched_get_deepstack_fallback():
    from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import \
        _patched_get_deepstack

    class MockModel:
        pass

    model = MockModel()

    # Case 3: no cached deepstack tensors, orig_get_deepstack returns a dictionary
    t1 = torch.ones((5, 10))

    def orig_get_deepstack(tokens):
        return {"layer_0": t1}

    res = _patched_get_deepstack(model, orig_get_deepstack, 10)
    assert res is not None
    assert "layer_0" in res.tensors
