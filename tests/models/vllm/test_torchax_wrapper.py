import jax
import jax.numpy as jnp
import pytest
import torch
import torchax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.utils import _pytree as pytree
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import get_forward_context

from tpu_commons.models.torchax.torchax_wrapper import (
    get_cpu_tensor_from_torchax_tensor, with_torchax_global, wrap_model,
    wrap_model_func)


def test_with_torchax_global_normal_execution():
    """Test that with_torchax_global enables/disables torchax around function call."""
    import unittest.mock as mock

    with mock.patch('torchax.enable_globally') as mock_enable, \
         mock.patch('torchax.disable_globally') as mock_disable:

        @with_torchax_global
        def test_func(x, y=10):
            return x + y

        result = test_func(5, y=15)

        assert result == 20
        mock_enable.assert_called_once()
        mock_disable.assert_called_once()


def test_with_torchax_global_exception_handling():
    """Test that with_torchax_global disables torchax even when function raises exception."""
    import unittest.mock as mock

    with mock.patch('torchax.enable_globally') as mock_enable, \
         mock.patch('torchax.disable_globally') as mock_disable:

        @with_torchax_global
        def test_func():
            raise ValueError("Test exception")

        with pytest.raises(ValueError, match="Test exception"):
            test_func()

        mock_enable.assert_called_once()
        mock_disable.assert_called_once()


@pytest.mark.skip(
    reason="b/440250062. Delete the test when deleting torchax-pt path.")
@pytest.mark.parametrize("tensor_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_mesh", [True, False])
def test_get_cpu_tensor_from_torchax_tensor(tensor_dtype, use_mesh):

    if use_mesh:
        mesh = Mesh(jax.devices(), axis_names=('x', ))
    else:
        mesh = None

    with torchax.default_env():
        t = torch.zeros(32, dtype=tensor_dtype)
        if mesh is not None:
            t.apply_jax_(jax.device_put, NamedSharding(mesh, P('x')))
    assert isinstance(t, torchax.tensor.Tensor)

    cpu_torch_t = get_cpu_tensor_from_torchax_tensor(t)
    assert isinstance(cpu_torch_t, torch.Tensor)
    assert cpu_torch_t.device == torch.device('cpu')


@pytest.fixture
def vllm_config():
    """Setup distributed environment for tests that need it."""
    # This is a fake config used for init dist env.
    # QKVParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )

    vllm_config = engine_args.create_engine_config()
    return vllm_config


class DummyAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.kv_cache = []

    def forward(self, hidden_states):
        # Dummy forward to simulate attention
        kv_cache = self.kv_cache[0]
        kv_cache = kv_cache + 1
        self.kv_cache[0] = kv_cache
        return hidden_states + kv_cache


class M(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.attn = DummyAttention()
        self.linear = torch.nn.Linear(10, 10, bias=False)
        self.param = torch.nn.Parameter(torch.zeros(10, 10))
        self.register_buffer('buffer', torch.zeros(10))

    def forward(self, input_ids, positions):
        fwd_context = get_forward_context()
        attn_metadata = fwd_context.attn_metadata
        _ = self.attn(input_ids)
        return self.linear(input_ids) + self.linear(
            positions) + self.buffer, attn_metadata['m']

    def model_func(self, hidden_states, positions=None):
        res = hidden_states @ self.param
        if positions is not None:
            res += positions @ self.param
        return res


@torch.no_grad()
def test_wrap_model(vllm_config):
    """Test wrapping a model with wrap_model."""
    m = M()
    static_forward_context = vllm_config.compilation_config.static_forward_context

    static_forward_context['attn'] = m.attn

    with torchax.default_env():
        m = m.to('jax')
        from torchax.interop import extract_all_buffers
        params, buffers = extract_all_buffers(m)
        params_and_buffers = {**params, **buffers}
        params_and_buffers_jax = pytree.tree_map_only(torch.Tensor,
                                                      lambda x: x.jax(),
                                                      params_and_buffers)
        inputs = (torch.zeros(5, 10).to('jax').jax(),
                  torch.zeros(5, 10).to('jax').jax())
        kv_caches = {
            'attn': torch.zeros(5, 10).to('jax').jax(),
        }
        attn_meata_data_tensor = torch.zeros(5, 10).to('jax').jax()
        attn_metadata = {'m': attn_meata_data_tensor}
        num_tokens = 20

        wrapped_func = wrap_model(m, vllm_config, static_forward_context)

        result, new_kv_cache = wrapped_func(params_and_buffers_jax, inputs,
                                            kv_caches, attn_metadata,
                                            num_tokens)

        assert isinstance(result, tuple)
        assert len(result) == 2
        # KV cache is updated
        updated_cache = new_kv_cache['attn'][0]
        assert isinstance(updated_cache, jax.Array)
        assert jnp.all(updated_cache == jnp.ones_like(updated_cache))
        # KV cache is donated
        assert kv_caches['attn'].is_deleted()
        # Check attn metatdata is passed through
        assert jnp.all(result[1] == attn_meata_data_tensor)


@torch.no_grad()
def test_model_func_wrapper():
    m = M()
    with torchax.default_env():
        m = m.to('jax')

        from torchax.interop import extract_all_buffers
        params, buffers = extract_all_buffers(m)
        params_and_buffers = {**params, **buffers}
        params_and_buffers_jax = pytree.tree_map_only(torch.Tensor,
                                                      lambda x: x.jax(),
                                                      params_and_buffers)
        input_ids = torch.zeros(5, 10).to('jax')
        positions = torch.zeros(5, 10).to('jax')
        args = (input_ids.jax(), )
        kwargs = {"positions": positions.jax()}

        method_name = 'model_func'
        wrapped_func = wrap_model_func(m, method_name)

        result = wrapped_func(params_and_buffers_jax, *args, **kwargs)
        assert isinstance(result, jax.Array)
        expected = m.model_func(input_ids, positions)
        assert jax.numpy.allclose(result, expected.jax(), atol=1e-6), \
            f"Expected {expected}, got {result}"
