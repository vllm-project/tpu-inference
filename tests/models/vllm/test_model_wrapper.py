# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import jax
import jax.numpy as jnp
import pytest
import torch
import torchax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs

from tpu_commons.attention.backends.pallas_torchax import PallasMetadata
from tpu_commons.distributed.tpu_distributed_utils import \
    create_torchax_tensor_with_partition_spec
from tpu_commons.models.jax import model_loader as jax_model_loader
from tpu_commons.models.jax.attention_metadata import \
    AttentionMetadata as JaxAttentionMetadata
from tpu_commons.models.torchax.torchax_wrapper import (wrap_model,
                                                        wrap_model_func)
from tpu_commons.models.torchax.tpu import TPUModelLoader


def _setup_environment(model):
    engine_args = EngineArgs(model=model, )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.level = CompilationLevel.NO_COMPILATION
    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        # Under single worker mode, full model is init first and then
        # partitioned using GSPMD.
        ensure_model_parallel_initialized(1, 1)
    return vllm_config


def prepare_test_inputs():
    """Prepare test inputs for the model.
    Dump from 'offline_inference.py', prompt: "A robot may not injure a human
    being"
    """
    # Pallas metadata
    slot_mapping = jnp.zeros((3, 16), dtype=jnp.int32)
    slot_mapping.at[0, 0].set(16)
    slot_mapping.at[2, 0].set(9)
    block_tables = jnp.zeros((8, 8), dtype=jnp.int32)
    block_tables.at[0, 0].set(1)
    context_lens = jnp.array([9, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 9, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    num_seqs = jnp.array([1], dtype=jnp.int32)
    num_slices = jnp.array([1], dtype=jnp.int32)
    attn_metadata = PallasMetadata(
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        num_slices=num_slices,
    )

    input_id = jnp.array([
        32, 12305, 1231, 537, 5811, 552, 264, 3738, 1660, 0, 0, 0, 0, 0, 0, 0
    ],
                         dtype=jnp.int32)
    position_ids = jnp.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)

    num_tokens = 16

    logits_indices = jnp.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    return attn_metadata, input_id, position_ids, num_tokens, \
            logits_indices


# SINGLE_CHIP_TEST_MODELS = set()
SINGLE_CHIP_TEST_MODELS = {
    "Qwen/Qwen2-1.5B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
}

# MULTI_CHIP_TEST_MODELS = set()
MULTI_CHIP_TEST_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
}

ALL_TEST_MODELS = SINGLE_CHIP_TEST_MODELS | MULTI_CHIP_TEST_MODELS


@pytest.mark.parametrize("model", list(ALL_TEST_MODELS))
def test_tpu_model_loader(model):
    use_mesh = model in MULTI_CHIP_TEST_MODELS
    # TODO: torchax runner requires env var to work with SPMD.
    os.environ["VLLM_XLA_USE_SPMD"] = "1" if use_mesh else "0"

    vllm_config = _setup_environment(model)
    mesh = Mesh(jax.devices(), axis_names=("x", )) if use_mesh else None

    with set_current_vllm_config(vllm_config):
        # Load Model
        loader = TPUModelLoader(load_config=vllm_config.load_config)
        model = loader.load_model(vllm_config=vllm_config,
                                  model_config=vllm_config.model_config,
                                  mesh=mesh)
        attn_metadata, input_ids, position_ids, num_tokens, \
            logits_indices = prepare_test_inputs()

        layers_dict = get_layers_from_vllm_config(vllm_config, Attention)
        layer_names = layers_dict.keys()
        # Attention metadata
        attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }

        # KV Cache params
        n_blocks = 512
        block_size = 16
        kv_cache_dtype = jnp.bfloat16
        attn_module = list(layers_dict.values())[0]
        num_kv_heads = attn_module.num_kv_heads
        head_size = attn_module.head_size
        kv_cache_shape = (n_blocks, block_size, num_kv_heads * 2, head_size)

        kv_cache_sharding = NamedSharding(mesh, P(
            None, None, 'x', None)) if use_mesh else jax.devices()[0]
        kv_caches = {
            layer_name:
            jnp.zeros(kv_cache_shape,
                      dtype=kv_cache_dtype,
                      device=kv_cache_sharding)
            for layer_name in layer_names
        }

        wrapped_model_func = wrap_model(
            model, vllm_config,
            vllm_config.compilation_config.static_forward_context)

        params, buffers = extract_all_buffers(model)
        params_and_buffers = {**params, **buffers}
        for name, tensor in params_and_buffers.items():
            if not isinstance(tensor, torchax.tensor.Tensor):
                params_and_buffers[name] = \
                    create_torchax_tensor_with_partition_spec(tensor, mesh,
                                                              ())
        params_and_buffers = pytree.tree_map_only(torch.Tensor,
                                                  lambda x: x.jax(),
                                                  params_and_buffers)

        hidden_states, new_kv_caches = wrapped_model_func(
            params_and_buffers, (input_ids, position_ids), kv_caches,
            attn_metadata, num_tokens)

        compute_logits_func = wrap_model_func(model, "compute_logits")
        _ = compute_logits_func(params_and_buffers, hidden_states, None)
        # Check that new_kv_caches are not all zero
        for layer_name, kv_cache in new_kv_caches.items():
            assert not jnp.all(
                kv_cache == 0), f"new_kv_caches for {layer_name} is all zero"
        # Check that old kv caches are donated
        for layer_name, kv_cache in kv_caches.items():
            assert kv_cache.is_deleted(
            ), f"kv_caches for {layer_name} is not donated"

        del model, params_and_buffers, kv_caches


@pytest.mark.parametrize("model", [
    "meta-llama/Llama-3.1-8B-Instruct",
])
def test_jax_model_wrapper(model):
    rng = jax.random.PRNGKey(42)
    mesh = Mesh(jax.devices(), axis_names=("model", ))

    engine_args = EngineArgs(model=model)
    vllm_config = engine_args.create_engine_config()
    # JAX Loader expect model config dtype to be a jax dtype.
    vllm_config.model_config.dtype = jnp.bfloat16

    attn_metadata, input_ids, position_ids, num_tokens, \
            logits_indices = prepare_test_inputs()

    jax_attn_metadata = JaxAttentionMetadata(
        input_positions=position_ids,
        slot_mapping=attn_metadata.slot_mapping,
        block_tables=attn_metadata.block_tables,
        seq_lens=attn_metadata.context_lens,
        query_start_loc=attn_metadata.query_start_loc,
        num_seqs=attn_metadata.num_seqs,
        num_slices=attn_metadata.num_slices,
    )

    model_fn, compute_logits_fn = jax_model_loader.get_vllm_model(
        vllm_config, rng, mesh)

    n_layers = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config)
    kv_caches = []
    # Right now we don't have a good way to get the num_kv_heads and page sizes.
    # So we hard code the kv_cache shape in this test.
    kv_cache_shape = (512, 16, 16, 128)
    kv_cache_dtype = jnp.bfloat16
    for _ in range(n_layers):
        kv_caches.append(jnp.zeros(kv_cache_shape, dtype=kv_cache_dtype))
    inputs = (kv_caches, input_ids, jax_attn_metadata)
    new_kv_caches, hidden_states = model_fn(*inputs)
    _ = compute_logits_fn(hidden_states)

    # Check that new_kv_caches are not all zero
    for kv_cache in new_kv_caches:
        assert not jnp.all(kv_cache == 0), "new_kv_caches is all zero"
    # Check that old kv caches are donated
    for kv_cache in kv_caches:
        assert kv_cache.is_deleted(), "kv_caches is not donated"
