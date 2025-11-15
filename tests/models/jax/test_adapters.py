from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig
from vllm.config.pooler import PoolerConfig

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.pool.pooler import (CLSPoolingMethod,
                                                  LastPoolingMethod,
                                                  MeanPoolingMethod)
from tpu_inference.layers.jax.pool.pooling_metadata import (
    TPUSupportedPoolingMetadata, )
from tpu_inference.models.jax.adapters import as_embedding_model
from tpu_inference.models.jax.qwen3 import Qwen3ForCausalLM
from tpu_inference.runner.kv_cache import create_kv_caches


class MockVllmConfig:

    def __init__(self, model: str, pooling_type: str):
        self.model_config = ModelConfig(model=model)
        self.model_config.dtype = jnp.bfloat16
        self.model_config.pooler_config = PoolerConfig(
            pooling_type=pooling_type, normalize=False)
        self.cache_config = MagicMock(cache_dtype="auto")
        self.load_config = MagicMock()
        self.load_config.download_dir = None


@pytest.fixture(scope="module")
def mesh():
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    device_mesh = devices.reshape((len(devices), 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m


@pytest.fixture
def rng() -> PRNGKey:
    return jax.random.PRNGKey(0)


@pytest.fixture
def mock_model_inputs():
    num_tokens = 6
    num_reqs = 1
    max_num_blocks_per_req = 4
    input_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.ones((num_reqs, ), dtype=jnp.int32)
    query_start_loc = jnp.arange(num_reqs + 1, dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 0], dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )

    return input_ids, attention_metadata


TEST_MODELS = [
    ("Qwen/Qwen3-0.6B", Qwen3ForCausalLM),
]


@pytest.mark.parametrize(
    ("model_id", "model_cls", "pooling_type", "pooling_cls"),
    [
        (model_id, model_cls, pooling_type, pooling_cls)
        for model_id, model_cls in TEST_MODELS
        for pooling_type, pooling_cls in [
            ("LAST", LastPoolingMethod),
            ("CLS", CLSPoolingMethod),
            ("MEAN", MeanPoolingMethod),
        ]
    ],
)
def test_embedding_adapter(model_id, model_cls, pooling_type, pooling_cls, rng,
                           mesh, mock_model_inputs):
    EmbeddingModel = as_embedding_model(model_cls)
    vllm_config = MockVllmConfig(model_id, pooling_type)
    model = EmbeddingModel(vllm_config, rng, mesh)

    assert isinstance(model.pooler.pooling, pooling_cls)
    assert model.is_pooling_model
    assert isinstance(model.pooler.head, nnx.Module)

    model.load_weights(rng)

    hf_config = vllm_config.model_config.hf_config
    head_dim = 128
    kv_caches = create_kv_caches(
        num_blocks=4,
        block_size=32,
        num_kv_heads=hf_config.num_key_value_heads,
        head_size=head_dim,
        mesh=mesh,
        layer_names=["layer"] * hf_config.num_hidden_layers,
        cache_dtype=jnp.bfloat16,
    )

    input_ids, attention_metadata = mock_model_inputs
    kv_caches, hidden_states, _ = model(kv_caches, input_ids,
                                        attention_metadata)

    num_tokens = input_ids.shape[0]
    pooling_metadata = TPUSupportedPoolingMetadata(
        prompt_lens=jnp.array([num_tokens], dtype=jnp.int32),
        first_token_indices=jnp.array([0], dtype=jnp.int32),
        last_token_indices=jnp.array([num_tokens - 1], dtype=jnp.int32),
        num_scheduled_tokens=jnp.array([num_tokens], dtype=jnp.int32),
    )

    embeddings = model.pooler(hidden_states, pooling_metadata)
    assert embeddings.shape == (1, hf_config.hidden_size)

    hidden_np = np.array(hidden_states, dtype=np.float32)
    last_index = int(pooling_metadata.last_token_indices[0])
    first_index = int(pooling_metadata.first_token_indices[0])
    if pooling_type == "LAST":
        expected = hidden_np[last_index]
    elif pooling_type == "CLS":
        expected = hidden_np[first_index]
    else:
        start = first_index
        end = last_index + 1
        expected = hidden_np[start:end].mean(axis=0)

    np.testing.assert_allclose(np.array(embeddings[0]), expected, rtol=1e-5,
                               atol=1e-5)
