import jax
import jax.numpy as jnp
import pytest
import torch
import torchax
import utils as test_utils
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j, t2j_dtype
from vllm.attention import Attention as VllmAttention
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

from tpu_commons.kernels.ragged_paged_attention.v3.kernel import \
    ref_ragged_paged_attention
from tpu_commons.models.jax.attention import get_kv_cache_shape_with_mesh
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.vllm.jax_attention import JaxAttention
from tpu_commons.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)

P = PartitionSpec


def generate_attention_metadata(num_tokens, mesh) -> AttentionMetadata:
    input_positions = None  # not used in test, doesn't matter
    block_tables = jnp.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
         [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=jnp.int32)
    seq_lens = jnp.array([num_tokens, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, num_tokens, 1, 1, 1, 1, 1, 1, 1],
                                dtype=jnp.int32)
    request_distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    input_positions = jax.device_put(input_positions,
                                     device=NamedSharding(
                                         mesh, PartitionSpec(None)))
    block_tables = jax.device_put(block_tables.reshape(-1),
                                  device=NamedSharding(mesh,
                                                       PartitionSpec(None)))
    seq_lens = jax.device_put(seq_lens,
                              device=NamedSharding(mesh, PartitionSpec(None)))
    query_start_loc = jax.device_put(query_start_loc,
                                     device=NamedSharding(
                                         mesh, PartitionSpec(None)))

    attention_metadata = AttentionMetadata(
        input_positions=input_positions,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    return attention_metadata


def generate_kv_caches(num_kv_heads, head_size, mesh, dtype):
    cache_shape = get_kv_cache_shape_with_mesh(mesh, 1024, 16, num_kv_heads,
                                               head_size, dtype)
    sharding = NamedSharding(mesh, PartitionSpec())

    def _allocate():
        return jnp.empty(
            shape=cache_shape,
            dtype=t2j_dtype(dtype),
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    return [sharded_allocate()]


@pytest.mark.parametrize("mesh", [test_utils.get_spmd_mesh()])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_size", [96, 128])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("num_tokens", [15, 63])
def test_jax_attention(mesh, num_heads, head_size, num_kv_heads, num_tokens):
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = dtype
    with set_current_vllm_config(vllm_config):
        attention = VllmAttention(
            num_heads=num_heads,
            head_size=head_size,
            scale=float('nan'),  # doesn't matter
            num_kv_heads=num_kv_heads,
            prefix="test_jax_attention.layer.0",
        )

    scale = float(1.0 / (head_size**0.5))
    qkv = torch.empty(num_tokens,
                      num_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype)
    qkv.uniform_(-scale, scale)
    q, k, v = qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=1)

    # reshape q,k,v into vLLM convention
    vllm_q = q.view(num_tokens, num_heads * head_size)
    vllm_k = k.view(num_tokens, num_kv_heads * head_size)
    vllm_v = v.view(num_tokens, num_kv_heads * head_size)

    # Set jax default device to workaround a layout bug in JAX 0.7.0 and earlier
    with torchax.default_env(), jax.default_device(jax.devices("tpu")[0]):
        jax_attention = JaxAttention(attention, mesh=mesh)
        vllm_q = torch_view(t2j(vllm_q))
        vllm_q.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        vllm_k = torch_view(t2j(vllm_k))
        vllm_k.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        vllm_v = torch_view(t2j(vllm_v))
        vllm_v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        q = t2j(q)
        q = jax.device_put(q, NamedSharding(mesh, P()))

    md = generate_attention_metadata(num_tokens, mesh)
    kv_caches = generate_kv_caches(num_kv_heads, head_size, mesh, dtype)

    with torchax.default_env(), set_vllm_model_wrapper_context(
            kv_caches=kv_caches,
            attention_metadata=md,
    ):
        jax_output = jax_attention(vllm_q, vllm_k, vllm_v)

        # reshape from vLLM convention
        jax_output = jax_output.view(num_tokens, num_heads, head_size)
        # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
        jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

        # the above jax_attetion call also updates the kv cache
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        kv_cache = vllm_model_wrapper_context.kv_caches[0]

    ref_output = ref_ragged_paged_attention(q,
                                            k,
                                            v,
                                            kv_cache,
                                            md.seq_lens,
                                            md.block_tables,
                                            md.query_start_loc,
                                            md.request_distribution,
                                            sm_scale=scale)
    ref_output = j2t(ref_output.astype(jnp.float32)).to(dtype)

    torch.testing.assert_close(ref_output, jax_output, atol=1e-2, rtol=1e-5)
