# SPDX-License-Identifier: Apache-2.0
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from torch.utils import _pytree as pytree
from torchax.interop import extract_all_buffers
from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, get_current_vllm_config,
                         get_layers_from_vllm_config, set_current_vllm_config)
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs

from tpu_commons.attention.backends.pallas_torchax import PallasMetadata
from tpu_commons.distributed.tpu_distributed_utils import \
    create_torchax_tensor_with_partition_spec
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

    vllm_config = get_current_vllm_config()
    layer_names = get_layers_from_vllm_config(vllm_config, Attention).keys()

    per_layer_attn_metadata = {
        layer_name: attn_metadata
        for layer_name in layer_names
    }

    kv_cache_shape = (58884, 16, 4, 128)
    # kv_cache_shape = (8192, 16, 4, 128)
    kv_cache_dtype = jnp.bfloat16

    kv_caches = {
        layer_name: jnp.zeros(kv_cache_shape, dtype=kv_cache_dtype)
        for layer_name in layer_names
    }

    input_id = jnp.array([
        32, 12305, 1231, 537, 5811, 552, 264, 3738, 1660, 0, 0, 0, 0, 0, 0, 0
    ],
                         dtype=jnp.int32)
    position_ids = jnp.array(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)

    num_tokens = 16

    logits_indices = jnp.array([8, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    return per_layer_attn_metadata, kv_caches, input_id, position_ids, num_tokens, \
            logits_indices


def torch_cpu_from_jax_array(j_arr):
    if j_arr.dtype == jnp.bfloat16:
        j_arr = j_arr.astype(jnp.float32)
    np_arr = np.asarray(j_arr)
    torch_t = torch.from_numpy(np_arr)
    if j_arr.dtype == jnp.bfloat16:
        torch_t = torch_t.to(torch.bfloat16)
    return torch_t


@pytest.mark.parametrize("model", [
    "Qwen/Qwen2-1.5B-Instruct",
])
def test_tpu_model_loader(model):
    vllm_config = _setup_environment(model)
    # vllm_config_2 = copy.deepcopy(vllm_config)
    mesh = None

    with set_current_vllm_config(vllm_config):
        loader = TPUModelLoader(load_config=vllm_config.load_config)
        model = loader.load_model(vllm_config=vllm_config,
                                  model_config=vllm_config.model_config,
                                  mesh=mesh)
        attn_metadata, kv_caches, input_ids, position_ids, num_tokens, \
            logits_indices = prepare_test_inputs()
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
        # for name, tensor in params_and_buffers.items():
        #         if not isinstance(tensor, torchax.tensor.Tensor):
        #             print("name {}, tensor {}".format(name, tensor))
        params_and_buffers = pytree.tree_map_only(torch.Tensor,
                                                  lambda x: x.jax(),
                                                  params_and_buffers)

        # Run in eager mode.
        torchax_env = torchax.default_env()
        attn_metadata_pt, input_ids_pt, position_ids_pt, kv_caches_pt = \
            pytree.tree_map_only(jax.Array,
                                 lambda x: torchax.tensor.Tensor(x, torchax_env),
                                 (attn_metadata, input_ids, position_ids,
                                  kv_caches))

        # with torchax_env:
        #     with set_forward_context(attn_metadata_pt, vllm_config,
        #                              num_tokens=num_tokens):
        #         hidden_states_eager = model(input_ids=input_ids_pt,
        #                                     positions=position_ids_pt)

        # print("check params_and_buffers ", params_and_buffers)
        hidden_states, new_kv_caches = wrapped_model_func(
            params_and_buffers, (input_ids, position_ids), kv_caches,
            attn_metadata, num_tokens)
        print("check hidden_states %s", hidden_states)
        breakpoint()

        selected_hidden = hidden_states[logits_indices]

        compute_logits_func = wrap_model_func(model, "compute_logits")
        logits = compute_logits_func(params_and_buffers, selected_hidden, None)
        sampled_logits = jnp.argmax(logits, axis=-1, keepdims=True)
        breakpoint()
        print(sampled_logits)

        # with torchax.default_env():
        #     out2 = model(torchax.tensor.Tensor(input_id, torchax.default_env()),
        #                 torchax.tensor.Tensor(position_ids, torchax.default_env()))

    # vllm_config_2.device_config.device = torch.device('cpu')
    # vllm_config_2.device_config.device_type = 'cpu'
    # vllm.platforms._current_platform = CpuPlatform()
    # vllm.model_executor.layers.rotary_embedding._ROPE_DICT = {}
    # with set_current_vllm_config(vllm_config_2):
    #     breakpoint()
    #     vllm_loader = get_model_loader(vllm_config_2.load_config)
    #     model = vllm_loader.load_model(vllm_config=vllm_config_2,
    #                                    model_config=vllm_config_2.model_config)
    #     input_id, position_ids, kv_caches = \
    #         pytree.tree_map_only(jax.Array,
    #                              lambda x : torch_cpu_from_jax_array(x),
    #                              (input_id, position_ids, new_kv_caches))
    #     bind_kv_cache(kv_caches,
    #                   vllm_config_2.compilation_config.static_forward_context,
    #                   [])
    #     # model = model.cpu()
    #     params, buffers = extract_all_buffers(model)
    #     params_and_buffers = {**params, **buffers}
    #     for name, tensor in model.named_buffers():
    #         if isinstance(tensor, torchax.tensor.Tensor):
    #             print("name {}, tensor {}".format(name, tensor))
    #     breakpoint()
    #     out = model(input_id, position_ids)
    #     print(out)

    # torchax.enable_globally()
    # model = model.to('jax')
    # print(model)

    # attn_metadata, input_ids, position_ids = \
    #     _load_dump('/home/lsiyuan/torchax_dump/attn_metadata.pt')

    # assert isinstance(model.model.layers[0].self_attn.attn, Attention)

    # n_blocks = 1024
    # block_size = 16
    # num_kv_heads = model.model.layers[0].self_attn.attn.num_kv_heads
    # head_size = model.model.layers[0].self_attn.attn.head_size
    # kv_dtype = torch.bfloat16

    # kv_caches = dict()
    # kv_cache_shape = (n_blocks, block_size, num_kv_heads * 2, head_size)
    # for i in range(len(model.model.layers)):
    #     key = f"model.layers.{i}.self_attn.attn"
    #     kv_caches[key] = torch.zeros(kv_cache_shape, dtype=kv_dtype)

    # # kv_caches = torch.load('/home/lsiyuan/torchax_dump/kv_caches.pt')
    # # for key, value in kv_caches.items():
    # #     kv_caches[key] = value[:1024]
    # #     print(f"key {key}: value {kv_caches[key].shape}")
    # kv_caches = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
    #                                  kv_caches)

    # # Simulate bind kv cache
    # static_forward_context = \
    #     vllm_config.compilation_config.static_forward_context

    # wrapped_func = wrapped_module(model, vllm_config, static_forward_context)
    # params, buffers = extract_all_buffers(model)
    # params, buffers = pytree.tree_map_only(torch.Tensor, lambda x: x.to('jax'),
    #                                        (params, buffers))
    # params_and_buffers = {**params, **buffers}
    # input_args = (input_ids, position_ids)
    # num_tokens = 9  # Not used?
    # hidden_states, new_kv_caches = wrapped_func(params_and_buffers, input_args,
    #                                             kv_caches, attn_metadata,
    #                                             num_tokens)
    # print(hidden_states)
    # for new_kv_cache in new_kv_caches.values():
    #     # Ensure kv cache is updated.
    #     assert torch.count_nonzero(new_kv_cache[0]) > 0

    # @jax_jit
    # def wrapped_compute_logits(params, buffers, hidden_states, sampling_param):
    #     return functional_call(model, "compute_logits", params, buffers,
    #                            hidden_states, sampling_param)

    # logits = wrapped_compute_logits(params, buffers, hidden_states, None)
    # # logits = model.compute_logits(hidden_states, None)
    # print(logits)
