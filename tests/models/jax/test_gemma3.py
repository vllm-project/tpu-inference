from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import ModelConfig
from vllm.model_executor.model_loader import LoadConfig, get_model_loader

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.models.jax.gemma3 import Gemma3ForCausalLM
from tpu_inference.models.jax.utils.qwix.qwix_utils import \
    apply_qwix_quantization
from tpu_inference.runner.kv_cache import create_kv_caches
from tpu_inference import utils


class MockVllmConfig:

    def __init__(self, model: str, kv_cache_dtype: str):
        self.model_config = ModelConfig(model)
        self.model_config.dtype = jnp.bfloat16
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.cache_config = MagicMock(cache_dtype=kv_cache_dtype)
        self.quant_config = None
        self.additional_config = {}


@pytest.fixture(scope="module")
def mesh():
    """
    Creates a mesh with 1 device.
    """
    if not jax.devices():
        pytest.skip("No JAX devices available for mesh creation.")

    devices = np.array(jax.local_devices()[:1])
    num_devices = len(devices)
    assert num_devices == 1
    device_mesh = devices.reshape((num_devices, 1, 1, 1))

    with Mesh(device_mesh,
              axis_names=('data', 'attn_dp', 'expert', 'model')) as m:
        yield m

@pytest.fixture
def mock_model_inputs():
    num_tokens = 8
    num_reqs = 1
    max_num_blocks_per_req = 4
    input_ids = jnp.ones((num_tokens, ), dtype=jnp.int32)
    positions = jnp.ones((num_tokens, ), dtype=jnp.int32)
    block_tables = jnp.zeros((num_reqs, max_num_blocks_per_req),
                             dtype=jnp.int32).reshape(-1)
    seq_lens = jnp.ones((num_reqs, ), dtype=jnp.int32)
    query_start_loc = jnp.ones((num_reqs + 1, ), dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 0], dtype=jnp.int32)

    attention_metadata = AttentionMetadata(
        input_positions=positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    indices_do_sample = jnp.ones((num_reqs, ), dtype=jnp.int32)

    return (input_ids, attention_metadata, indices_do_sample)

@pytest.fixture
def rng() -> PRNGKey:
    """Provides a reusable JAX PRNGKey."""
    return jax.random.PRNGKey(42)

@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.models.jax.gemma3.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield

class TestGemma3ForCausalLM:

    @pytest.mark.parametrize("model_name", ["google/gemma-3-1b-it"])
    @pytest.mark.parametrize("kv_cache_type", ["auto", "fp8"])
    @pytest.mark.parametrize("qwix_rules", [
        None,
        [{
            "module_path": ".*",
            "weight_qtype": "int8",
            "act_qtype": "int8"
        }]
    ])
    def test_gemma3_1B(self, model_name, kv_cache_type, qwix_rules, rng, mesh,
                        mock_model_inputs):
        """Tests model init and model forward for the 1B model variant."""
        mock_vllm_config = MockVllmConfig(model_name, kv_cache_type)
        if qwix_rules:
            mock_vllm_config.additional_config["quanntization"] = dict(
                qwix=dict(rules=qwix_rules))

        # Test model init
        with jax.set_mesh(mesh):
            model = Gemma3ForCausalLM(mock_vllm_config, rng, mesh)

        model_config = mock_vllm_config.model_config
        hf_config = model_config.hf_config

        assert model.mesh.shape == {
            "data": 1,
            "attn_dp": 1,
            "expert": 1,
            "model": 1
        }

        layers = model.model.layers
        assert len(layers) == hf_config.num_hidden_layers

        hidden_size = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        head_dim_original = hf_config.head_dim
        head_dim = utils.get_padded_head_dim(head_dim_original)
        intermediate_size = hf_config.intermediate_size
        sliding_window = hf_config.sliding_window
        rope_scaling = getattr(hf_config, "rope_scaling", None)

        # Derive layer types from config
        layer_types = hf_config.layer_types

        # Find first sliding and first full attention layer dynamically
        sliding_layer_idx = None
        full_layer_idx = None
        for idx, layer_type in enumerate(layer_types):
            if layer_type == "sliding_attention" and sliding_layer_idx is None:
                sliding_layer_idx = idx
            if layer_type == "full_attention" and full_layer_idx is None:
                full_layer_idx = idx
            if sliding_layer_idx is not None and full_layer_idx is not None:
                break

        # Test sliding attention layer
        if sliding_layer_idx is not None:
            attn = layers[sliding_layer_idx].self_attn
            assert attn.hidden_size == hidden_size
            assert attn.num_heads == num_heads
            assert attn.num_kv_heads == num_kv_heads
            assert attn.head_dim_original == head_dim_original
            assert attn.head_dim == head_dim
            assert attn.sliding_window == sliding_window
            assert attn.is_sliding is True

            # Check rope config for sliding layer
            rope_parameters = getattr(hf_config, "rope_parameters", {})
            if "sliding_attention" in rope_parameters:
                expected_theta = rope_parameters["sliding_attention"].get(
                    "rope_theta", hf_config.rope_theta)
            else:
                expected_theta = hf_config.rope_local_base_freq
            assert attn.rope_theta == expected_theta

            # Check rope_scaling
            if "sliding_attention" in rope_parameters:
                expected_scaling = rope_parameters["sliding_attention"].get(
                    "rope_scaling", rope_scaling)
            else:
                expected_scaling = rope_scaling
            assert attn.rope_scaling == expected_scaling

            assert attn.q_proj.weight.shape == (hidden_size, num_heads, head_dim)
            assert attn.k_proj.weight.shape == (hidden_size, num_kv_heads, head_dim)
            assert attn.v_proj.weight.shape == (hidden_size, num_kv_heads, head_dim)
            assert attn.o_proj.weight.shape == (num_heads, head_dim, hidden_size)

            mlp = layers[sliding_layer_idx].mlp
            assert mlp.gate_proj.weight.shape == (hidden_size, intermediate_size)
            assert mlp.up_proj.weight.shape == (hidden_size, intermediate_size)
            assert mlp.down_proj.weight.shape == (intermediate_size, hidden_size)

        # Test full attention layer
        if full_layer_idx is not None:
            attn = layers[full_layer_idx].self_attn
            assert attn.hidden_size == hidden_size
            assert attn.num_heads == num_heads
            assert attn.num_kv_heads == num_kv_heads
            assert attn.head_dim_original == head_dim_original
            assert attn.head_dim == head_dim
            assert attn.sliding_window is None
            assert attn.is_sliding is False

            # Check rope config for full attention layer
            rope_parameters = getattr(hf_config, "rope_parameters", {})
            if "full_attention" in rope_parameters:
                expected_theta = rope_parameters["full_attention"].get(
                    "rope_theta", hf_config.rope_theta)
            else:
                expected_theta = hf_config.rope_theta
            assert attn.rope_theta == expected_theta

            if "full_attention" in rope_parameters:
                expected_scaling = rope_parameters["full_attention"].get(
                    "rope_scaling", rope_scaling)
            else:
                expected_scaling = rope_scaling
            assert attn.rope_scaling == expected_scaling

        # Test model load
        with jax.set_mesh(mesh):
            loader = get_model_loader(LoadConfig(load_format="hf"))
            loader.load_weights(model, model_config)

        # Apply qwix quantization, no-op if rules are not given.
        model = apply_qwix_quantization(mock_vllm_config,
                                        model,
                                        rng,
                                        mesh,
                                        apply_to_abstract_model=False)

        # Test model forward
        kv_caches = create_kv_caches(
            num_blocks=4,
            block_size=32,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            mesh=mesh,
            layer_names=["layer"] * hf_config.num_hidden_layers,
            cache_dtype=jnp.float8_e4m3fn
            if mock_vllm_config.cache_config.cache_dtype == "fp8" else
            jnp.bfloat16
        )
        # 1 seq with 8 tokens
        input_ids, attention_metadata, indices_do_sample = mock_model_inputs
        kv_caches, hidden_states, aux_hidden_states = model(
            kv_caches, input_ids, attention_metadata)
        assert hidden_states.shape == (8, hidden_size)
        assert len(aux_hidden_states) == 0

        hidden_states = hidden_states[indices_do_sample]
        assert hidden_states.shape == (1, hidden_size)

        logits = model.compute_logits(hidden_states)
        assert logits.shape == (1, hf_config.vocab_size)