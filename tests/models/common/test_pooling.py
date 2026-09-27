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
"""Pooling host fast path and device mean pooling vs the original pooler.

Runs anywhere (CPU is fine): the reference is the original
`compute_pooler_output` body, i.e. vLLM's embed pooler (mean pooling +
normalization) under torchax's dispatch modes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from jax.sharding import Mesh
from torchax.interop import torch_view
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import CLSPool, MeanPool
from vllm.model_executor.layers.pooler.seqwise.poolers import SequencePooler
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from tpu_inference.models.common.pooling import DeviceMeanPooler, pool_on_host

HIDDEN = 64


def _pooler(pooling=None):
    head = EmbeddingPoolerHead(projector=None,
                               head_dtype=torch.float32,
                               activation=PoolerNormalize())
    return DispatchPooler(
        {"embed": SequencePooler(pooling or MeanPool(), head)})


def _metadata(prompt_lens, dimensions=None, normalize=None):
    params = [
        PoolingParams(task="embed",
                      dimensions=dimensions,
                      use_activation=normalize) for _ in prompt_lens
    ]
    return PoolingMetadata(
        prompt_lens=torch.from_numpy(np.array(prompt_lens, np.int32)),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=params,
        pooling_states=[PoolingStates() for _ in prompt_lens])


def _original(pooler, hidden, metadata, seq_lens, num_scheduled_tokens):
    """The pre-fast-path body of model_loader.compute_pooler_output."""
    torch_states = torch_view(hidden)
    with torchax.default_env():
        torch_states = torch_states.to('cpu', non_blocking=True)
        metadata.build_pooling_cursor(num_scheduled_tokens,
                                      torch.tensor(seq_lens),
                                      device=torch_states.device)
        return pooler(torch_states, metadata)


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
def test_host_fast_path_is_bitwise_identical(lens, num_tokens):
    pooler = _pooler()
    hidden, nst = _step(lens, num_tokens)
    want = _original(pooler, hidden, _metadata(lens), nst, nst)
    got = pool_on_host(pooler, hidden, _metadata(lens), nst, nst)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert torch.equal(g, w)


@pytest.mark.parametrize("lens,num_tokens", CASES)
@pytest.mark.parametrize("dimensions,normalize", [(None, None), (32, None),
                                                  (None, False)])
def test_device_mean_pooling_matches(mesh, lens, num_tokens, dimensions,
                                     normalize):
    pooler = _pooler()
    device_pooler = DeviceMeanPooler.maybe_create(pooler, mesh, 256)
    assert device_pooler is not None
    hidden, nst = _step(lens, num_tokens)
    want = _original(pooler, hidden, _metadata(lens, dimensions, normalize),
                     nst, nst)
    got = device_pooler(hidden, _metadata(lens, dimensions, normalize), nst)
    assert got is not None and len(got) == len(want)
    for g, w in zip(got, want):
        assert g.shape == w.shape and g.dtype == w.dtype
        assert torch.isfinite(g).all()
        np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=1e-5, atol=1e-6)


def test_device_mean_pooling_falls_back(mesh):
    pooler = _pooler()
    device_pooler = DeviceMeanPooler.maybe_create(pooler, mesh, 16)
    hidden, nst = _step([10, 6], 16)
    # Partial prefill: left to vLLM's pooler (which rejects it).
    assert device_pooler(hidden, _metadata([10, 9]), nst) is None
    # More requests than the device path is compiled for.
    many = np.ones(32, np.int32)
    assert device_pooler(jnp.zeros((32, HIDDEN)), _metadata([1] * 32),
                         many) is None
    # Not mean pooling: no device pooler at all.
    assert DeviceMeanPooler.maybe_create(_pooler(CLSPool()), mesh, 16) is None


def test_device_mean_pooling_precompile(mesh):
    device_pooler = DeviceMeanPooler.maybe_create(_pooler(), mesh, 64)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    device_pooler.precompile([16, 128], HIDDEN, jnp.float32, sharding)
    assert list(device_pooler._req_buckets(16)) == [8, 16]
    assert list(device_pooler._req_buckets(2048)) == [8, 16, 32, 64]
