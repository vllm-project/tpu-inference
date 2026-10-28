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
"""Device sequence classification (mean pool + linear head + softmax) vs
vLLM's CPU classify pooler. Runs anywhere (CPU is fine)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.model_executor.layers.pooler.activations import PoolerClassify
from vllm.model_executor.layers.pooler.seqwise.heads import \
    ClassifierPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import MeanPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from tpu_inference.models.common.pooling import (DeviceSeqClassifier,
                                                 classify_from_seq_lens,
                                                 pool_on_host)

HIDDEN = 64
LABELS = 2


def _head(seed=1):
    rng = np.random.default_rng(seed)
    kernel = rng.standard_normal((HIDDEN, LABELS)).astype(np.float32) * 0.3
    bias = rng.standard_normal((LABELS, )).astype(np.float32)
    lin = torch.nn.Linear(HIDDEN, LABELS)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(kernel.T.copy()))
        lin.bias.copy_(torch.from_numpy(bias))
    return jnp.asarray(kernel), jnp.asarray(bias), lin


def _cpu_pooler(lin):
    head = ClassifierPoolerHead(classifier=lin,
                                head_dtype=torch.float32,
                                activation=PoolerClassify())
    return DispatchPooler({"classify": SequencePooler(MeanPool(), head)})


def _metadata(prompt_lens, use_activation=None, tasks=None):
    if not isinstance(use_activation, list):
        use_activation = [use_activation] * len(prompt_lens)
    tasks = tasks or ["classify"] * len(prompt_lens)
    params = [
        PoolingParams(task=t, use_activation=a)
        for t, a in zip(tasks, use_activation)
    ]
    return PoolingMetadata(
        prompt_lens=torch.from_numpy(np.array(prompt_lens, np.int32)),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=params,
        pooling_states=[PoolingStates() for _ in prompt_lens])


def _step(lens, num_tokens, seed=0):
    rng = np.random.default_rng(seed)
    hidden = rng.standard_normal((num_tokens, HIDDEN)).astype(np.float32)
    hidden[sum(lens):] = np.nan  # Padding rows: unspecified contents.
    return jnp.asarray(hidden), np.array(lens, np.int32)


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:1]), ("x", ))


CASES = [([5], 16), ([1024], 1024), ([1000, 1000], 2048), ([7, 1, 30, 2], 64)]


@pytest.mark.parametrize("lens,num_tokens", CASES)
@pytest.mark.parametrize("use_activation", [None, True, False, "mixed"])
def test_device_classifier_matches_cpu_pooler(mesh, lens, num_tokens,
                                              use_activation):
    kernel, bias, lin = _head()
    act = ([i % 2 == 0 for i in range(len(lens))]
           if use_activation == "mixed" else use_activation)
    hidden, nst = _step(lens, num_tokens)
    want = pool_on_host(_cpu_pooler(lin), hidden, _metadata(lens, act), nst,
                        nst)
    clf = DeviceSeqClassifier.maybe_create(kernel, bias, mesh, 256)
    got = clf(hidden, _metadata(lens, act), nst)
    assert got is not None and len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == (LABELS, ) and g.dtype == torch.float32
        torch.testing.assert_close(g, w.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("lens,num_tokens", CASES)
def test_fused_outputs_match_standalone(mesh, lens, num_tokens):
    kernel, bias, lin = _head()
    hidden, nst = _step(lens, num_tokens)
    seq_lens = jnp.asarray(np.pad(nst, (0, 32 - len(nst))))
    aux = classify_from_seq_lens(hidden, seq_lens, kernel, bias, 32)
    assert aux[0].shape == (32, LABELS) and aux[1].shape == (32, LABELS)
    clf = DeviceSeqClassifier.maybe_create(kernel, bias, mesh, 256,
                                           fused_max_reqs=32)
    want = clf(hidden, _metadata(lens, True), nst)
    got = clf(hidden, _metadata(lens, True), nst, aux_hidden_states=list(aux))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-6, atol=1e-6)


def test_declines_other_tasks_and_partial_prefill(mesh):
    kernel, bias, _ = _head()
    clf = DeviceSeqClassifier.maybe_create(kernel, bias, mesh, 256)
    hidden, nst = _step([5, 3], 16)
    assert clf(hidden, _metadata([5, 3], tasks=["embed", "classify"]),
               nst) is None
    # Partial prefill: scheduled tokens != prompt length.
    assert clf(hidden, _metadata([9, 3]), nst) is None


def test_calibrated_pooler_config_falls_back(mesh):
    kernel, bias, _ = _head()
    assert DeviceSeqClassifier.maybe_create(kernel, bias, mesh, 256,
                                            calibrated=True) is None
