import jax
import jax.numpy as jnp
import pytest
import torch
import torchax
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import j2t, t2j, t2j_dtype
from vllm.attention import Attention as VllmAttention
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs

from tpu_commons import utils_jax as utils
from tpu_commons.kernels.ragged_paged_attention.kernel import \
    ref_ragged_paged_attention
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.vllm.jax_attention import JaxAttention
from tpu_commons.models.vllm.vllm_model_wrapper_context import (
    get_vllm_model_wrapper_context, set_vllm_model_wrapper_context)

P = PartitionSpec


@pytest.fixture(scope="module", autouse=True)
def setup_torchax():
    """Enable torchax globally before all tests, disable after all tests."""
    torchax.enable_globally()
    yield
    torchax.disable_globally()


def _get_spmd_mesh():
    axis_names = ("data", "model")
    mesh_shape = (1, len(jax.devices()))
    return jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())


def generate_attention_metadata(num_tokens, mesh) -> AttentionMetadata:
    input_positions = None  # not used in test, doesn't matter
    slot_mapping = jnp.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [num_tokens, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=jnp.int32)
    block_tables = jnp.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
         [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=jnp.int32)
    seq_lens = jnp.array([num_tokens, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, num_tokens, 1, 1, 1, 1, 1, 1, 1],
                                dtype=jnp.int32)
    num_seqs = jnp.array([1], dtype=jnp.int32)
    num_slices = jnp.array([1], dtype=jnp.int32)

    input_positions = jax.device_put(input_positions,
                                     device=NamedSharding(
                                         mesh, PartitionSpec(None)))
    slot_mapping = jax.device_put(slot_mapping,
                                  device=NamedSharding(mesh,
                                                       PartitionSpec(None)))
    block_tables = jax.device_put(block_tables,
                                  device=NamedSharding(mesh,
                                                       PartitionSpec(None)))
    seq_lens = jax.device_put(seq_lens,
                              device=NamedSharding(mesh, PartitionSpec(None)))
    query_start_loc = jax.device_put(query_start_loc,
                                     device=NamedSharding(
                                         mesh, PartitionSpec(None)))
    num_seqs = jax.device_put(num_seqs,
                              device=NamedSharding(mesh, PartitionSpec(None)))
    num_slices = jax.device_put(num_slices,
                                device=NamedSharding(mesh,
                                                     PartitionSpec(None)))

    attention_metadata = AttentionMetadata(
        input_positions=input_positions,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        num_slices=num_slices,
    )
    return attention_metadata


def generate_kv_caches(num_kv_heads, head_size, mesh, dtype):
    cache_shape = (
        1024,  # num_blocks
        16,  # block_size
        num_kv_heads * 2,
        utils.get_padded_head_dim(head_size),
    )
    sharding = NamedSharding(mesh, PartitionSpec(None, None, "model", None))

    def _allocate():
        return jnp.empty(
            shape=cache_shape,
            dtype=t2j_dtype(dtype),
        )

    sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
    return [sharded_allocate()]


@pytest.mark.parametrize("mesh", [_get_spmd_mesh()])
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
    jax_attention = JaxAttention(attention, mesh=mesh)

    scale = float(1.0 / (head_size**0.5))
    qkv = torch.empty(num_tokens,
                      num_heads + 2 * num_kv_heads,
                      head_size,
                      dtype=dtype)
    qkv.uniform_(-scale, scale)
    q, k, v = qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=1)

    q = torch_view(t2j(q))
    q.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, None)))
    k = torch_view(t2j(k))
    k.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, None)))
    v = torch_view(t2j(v))
    v.apply_jax_(jax.device_put, NamedSharding(mesh, P(None, None)))

    md = generate_attention_metadata(num_tokens, mesh)
    kv_caches = generate_kv_caches(num_kv_heads, head_size, mesh, dtype)

    with set_vllm_model_wrapper_context(
            kv_caches=kv_caches,
            attention_metadata=md,
    ):
        # reshape q,k,v into vLLM convention
        vllm_q = q.view(num_tokens, num_heads * head_size)
        vllm_k = k.view(num_tokens, num_kv_heads * head_size)
        vllm_v = v.view(num_tokens, num_kv_heads * head_size)

        jax_output = jax_attention(vllm_q, vllm_k, vllm_v)

        # reshape from vLLM convention
        jax_output = jax_output.view(num_tokens, num_heads, head_size)

        # the above jax_attetion call also updates the kv cache
        vllm_model_wrapper_context = get_vllm_model_wrapper_context()
        kv_cache = vllm_model_wrapper_context.kv_caches[0]

    # j2t() doens't support bfloat16, so we cast it into float32 as an intermedate step.
    jax_output = j2t(jax_output.to(torch.float32)).to(dtype)

    if head_size % 128 != 0:
        padded_head_size = utils.get_padded_head_dim(head_size)
        pad_width = [(0, 0), (0, 0), (0, padded_head_size - head_size)]
        q = jnp.pad(q,
                    pad_width=pad_width,
                    mode='constant',
                    constant_values=0.0)
    ref_output = ref_ragged_paged_attention(jax_view(q),
                                            kv_cache,
                                            md.seq_lens,
                                            md.block_tables,
                                            md.query_start_loc,
                                            md.num_seqs,
                                            sm_scale=scale)
    if head_size % 128 != 0:
        ref_output = ref_output[:, :, :head_size]
    ref_output = j2t(ref_output.astype(jnp.float32)).to(dtype)

    torch.testing.assert_close(ref_output, jax_output, atol=1e-2, rtol=1e-5)
