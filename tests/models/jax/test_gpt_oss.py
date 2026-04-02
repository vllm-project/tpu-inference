from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.attention.gpt_oss_attention import GptOssAttention
from tpu_inference.layers.jax.moe.gpt_oss_moe import GptOssMoE
from tpu_inference.models.jax.gpt_oss import GptOss
from tpu_inference.runner.kv_cache import create_kv_caches


class MockHfConfig:
    """Mocks the HuggingFace config object with small values for testing."""

    def __init__(self):
        self.num_hidden_layers: int = 2
        self.num_local_experts: int = 4
        self.vocab_size: int = 1024
        self.num_attention_heads: int = 64
        self.num_key_value_heads: int = 8
        self.head_dim: int = 64
        self.hidden_size: int = self.num_attention_heads * self.head_dim
        self.intermediate_size: int = 256
        self.num_experts_per_tok: int = 2
        self.rms_norm_eps: float = 1e-5
        self.swiglu_limit: float = 0.0
        self.rope_theta: float = 10000.0
        self.rope_scaling = {
            "factor": 1.0,
            "beta_slow": 1.0,
            "beta_fast": 1.0,
            "original_max_position_embeddings": 2048,
        }
        self.sliding_window: int | None = None


class MockVllmConfig:
    """
    Mocks the VllmConfig object, providing a mock hf_config and
    setting 'random_weights' to True to avoid downloading real weights.
    """

    def __init__(self, model: str, kv_cache_dtype: str):
        self.model_config = MagicMock()
        self.model_config.hf_config = MockHfConfig()
        self.model_config.model = model
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock(download_dir=None)
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.additional_config = {"random_weights": True, "is_verbose": False}


@pytest.fixture(scope="module")
def mesh():
    """
    Creates and globally activates a mesh for the entire test module.
    This is necessary to satisfy the jax.jit in `create_param`.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1))

    # Create the mesh
    m = Mesh(device_mesh, axis_names=('data', 'model'))

    jax.set_mesh(m)

    yield m

    try:
        empty_devices = np.empty((0, 0), dtype=devices.dtype)
        jax.set_mesh(Mesh(empty_devices, axis_names=()))
    except Exception:
        pass


@pytest.fixture
def mock_model_inputs():
    """Provides mock inputs for a forward pass."""
    num_tokens = 8
    num_reqs = 1
    max_num_blocks_per_req = 4
    input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
    positions = jnp.arange(0, num_tokens, dtype=jnp.int32)
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.array([num_tokens], dtype=jnp.int32)
    query_start_loc = jnp.array([0, num_tokens], dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 0], dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    indices_do_sample = jnp.array([num_tokens - 1], dtype=jnp.int32)

    return (input_ids, attention_metadata, indices_do_sample)


@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)


class TestGptOss:

    @pytest.mark.parametrize("mock_vllm_config", [
        MockVllmConfig("mock/gpt-oss-small", "auto"),
    ])
    def test_gpt_oss_init_and_forward(self, mock_vllm_config, rng, mesh,
                                      mock_model_inputs):
        """Tests model init, weight loading (mocked), and a forward pass."""

        # Test model init
        hf_config = mock_vllm_config.model_config.hf_config

        model = GptOss(mock_vllm_config, rng, mesh)

        assert model.mesh.shape == {"data": 1, "model": 1}
        assert isinstance(model.rng, nnx.Rngs)
        assert len(model.layers) == hf_config.num_hidden_layers

        # Check key submodule shapes
        assert model.embedder.input_embedding_table_VD.shape == (
            hf_config.vocab_size, hf_config.hidden_size)

        layer_0 = model.layers[0]
        attn = layer_0.attn
        assert isinstance(attn, GptOssAttention)
        assert attn.kernel_q_DNH.shape == (hf_config.hidden_size,
                                           hf_config.num_attention_heads,
                                           hf_config.head_dim)

        moe_mlp = layer_0.custom_module
        assert isinstance(moe_mlp, GptOssMoE)
        assert moe_mlp.mlp1_weight_EDF2.shape == (hf_config.num_local_experts,
                                                  hf_config.hidden_size,
                                                  hf_config.intermediate_size *
                                                  2)

        assert model.final_norm.scale.shape == (hf_config.hidden_size, )
        assert model.lm_head.input_embedding_table_DV.shape == (
            hf_config.hidden_size, hf_config.vocab_size)

        # Test model load
        with patch("tpu_inference.models.jax.gpt_oss.model_weights_generator",
                   return_value=iter([])):
            model.load_weights(rng)

        # Test model forward
        num_key_value_heads = int(hf_config.num_key_value_heads / 2)
        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=32,
            num_kv_heads=num_key_value_heads,
            head_size=hf_config.head_dim,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            cache_dtype=jnp.float8_e4m3fn
            if mock_vllm_config.cache_config.cache_dtype == "fp8" else
            jnp.bfloat16)

        input_ids, attention_metadata, indices_do_sample = mock_model_inputs

        kv_caches, hidden_states, aux_hidden_states = model(
            kv_caches, input_ids, attention_metadata)

        # Check output shapes
        assert hidden_states.shape == (8, hf_config.hidden_size)
        assert aux_hidden_states == []

        # Test logits computation
        hidden_states = hidden_states[indices_do_sample, :]
        assert hidden_states.shape == (1, hf_config.hidden_size)

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (1, hf_config.vocab_size)
