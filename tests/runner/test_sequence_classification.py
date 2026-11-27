# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh, PartitionSpec

from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)
from tpu_inference.models.jax.sequence_classification import \
    NativeSequenceClassificationPooler
from tpu_inference.runner.compilation_manager import CompilationManager
from tpu_inference.runner.tpu_runner import TPUModelRunner
from tpu_inference.utils import DeviceBuffer


def _make_single_device_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:1], dtype=object).reshape(
        (1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, MESH_AXIS_NAMES)


class _ForbidCompileSpy:
    """Re-entrant stand-in for ForbidCompile() / nullcontext()."""

    def __init__(self):
        self.entries = 0

    def __enter__(self):
        self.entries += 1

    def __exit__(self, *exc):
        return False


class _DeviceLogitsPooler:
    pools_device_logits = True

    def __call__(self, logits, metadata):
        return [
            torch.tensor([1.0, 2.0]),
            torch.tensor([5.0, 6.0]),
        ]


def _make_runner(mesh, pooler_fn):
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["req-done", "req-partial"],
        num_computed_tokens_cpu=np.array([4, 2], dtype=np.int32),
        num_prompt_tokens=np.array([8, 8], dtype=np.int32),
        get_pooling_metadata=lambda: "meta",
    )
    return SimpleNamespace(
        input_batch=batch,
        mesh=mesh,
        maybe_forbid_compile=_ForbidCompileSpy(),
        vllm_config=SimpleNamespace(sharding_config=SimpleNamespace(
            prefill_cp_size=1)),
        state_leaves="leaves",
        _select_from_array_fn=lambda arr, idx, m, pcp: arr[idx],
        compute_logits_fn=lambda leaves, rows: rows.astype(jnp.float32),
        pooler_fn=pooler_fn,
    )


def test_pool_device_logits_chunked_completion():
    mesh = _make_single_device_mesh()
    hidden = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                         dtype=jnp.bfloat16)
    logits_indices = jnp.asarray([0, 2], dtype=jnp.int32)
    runner = _make_runner(mesh, _DeviceLogitsPooler())
    scheduler_output = SimpleNamespace(num_scheduled_tokens={
        "req-done": 4,
        "req-partial": 3
    })

    out = TPUModelRunner._pool(runner, hidden, logits_indices, None,
                               scheduler_output)
    assert runner.maybe_forbid_compile.entries == 1
    assert len(out.pooler_output) == 2
    assert torch.equal(out.pooler_output[0], torch.tensor([1.0, 2.0]))
    assert out.pooler_output[1] is None

    # A second step re-enters the same context manager.
    TPUModelRunner._pool(runner, hidden, logits_indices, None,
                         scheduler_output)
    assert runner.maybe_forbid_compile.entries == 2

    with pytest.raises(RuntimeError, match="without in-process DP reordering"):
        TPUModelRunner._pool(runner, hidden, logits_indices, np.array([0, 1]),
                             scheduler_output)


def test_pool_host_pooler_keeps_decoupled_interface():
    mesh = _make_single_device_mesh()
    hidden = jnp.ones((3, 2), dtype=jnp.bfloat16)
    calls = []

    def host_pooler(hidden_states, metadata, seq_lens, num_scheduled_tokens):
        calls.append((metadata, list(seq_lens), list(num_scheduled_tokens)))
        return ["a", "b"]

    runner = _make_runner(mesh, host_pooler)
    runner.max_num_reqs = 4
    # Stale values: _pool must not read seq_lens from a fresh buffer view.
    runner.device_buffer = SimpleNamespace(
        get_view=lambda shape, key: np.array([8, 8, 0, 0], dtype=np.int32))
    scheduler_output = SimpleNamespace(num_scheduled_tokens={
        "req-done": 4,
        "req-partial": 3
    })

    out = TPUModelRunner._pool(runner, hidden, None, None, scheduler_output)
    # seq_lens = num_computed_tokens + num_scheduled_tokens = [4+4, 2+3].
    assert calls == [("meta", [8, 5], [4, 3])]
    assert out.pooler_output == ["a", "b"]
    assert runner.maybe_forbid_compile.entries == 0


def _dirty_device_buffer(max_num_reqs):
    """A DeviceBuffer after a longer step and then a shorter step, in the
    reset() / get_view() / build() order _prepare_inputs uses."""
    buf = DeviceBuffer(initial_capacity=64)
    buf.reset()
    buf.get_view((40, ), key="input_ids")[:] = 1024
    buf.get_view((max_num_reqs, ), key="seq_lens")[:] = 0
    buf.build()
    buf.reset()
    buf.get_view((4, ), key="input_ids")[:] = 5
    seq_lens = buf.get_view((max_num_reqs, ), key="seq_lens")
    seq_lens[:2] = [8, 5]
    seq_lens[2:] = 0
    buf.build()
    return buf


def test_pool_host_pooler_seq_lens_ignore_device_buffer_leftovers():
    mesh = _make_single_device_mesh()
    hidden = jnp.ones((7, 2), dtype=jnp.bfloat16)
    captured = []

    def host_pooler(hidden_states, metadata, seq_lens, num_scheduled_tokens):
        captured.append(np.asarray(seq_lens).tolist())
        return ["a", None]

    runner = _make_runner(mesh, host_pooler)
    runner.max_num_reqs = 4
    runner.device_buffer = _dirty_device_buffer(runner.max_num_reqs)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={
        "req-done": 4,
        "req-partial": 3
    })

    TPUModelRunner._pool(runner, hidden, None, None, scheduler_output)
    assert captured == [[8, 5]]


def test_pool_token_wise_all_pool_returns_completed_prompt():
    """With chunked prefill, vLLM's AllPool returns a request's output only
    when prompt_lens == seq_lens, so seq_lens must be this step's values."""
    from vllm.model_executor.layers.pooler.tokwise import methods
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

    with mock.patch.object(
            methods,
            "get_current_vllm_config",
            return_value=SimpleNamespace(scheduler_config=SimpleNamespace(
                enable_chunked_prefill=True, async_scheduling=False))):
        all_pool = methods.AllPool()

    metadata = PoolingMetadata(
        prompt_lens=torch.tensor([8, 8]),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task="token_embed")] * 2,
        pooling_states=[PoolingStates(), PoolingStates()])

    def host_pooler(hidden_states, pooling_metadata, seq_lens,
                    num_scheduled_tokens):
        # Same calls as compute_pooler_output in models/common/model_loader.py.
        states = torch.from_numpy(np.asarray(hidden_states, np.float32))
        # No pinned host memory in this CPU-only test.
        with mock.patch("vllm.v1.pool.metadata.PIN_MEMORY", False):
            pooling_metadata.build_pooling_cursor(num_scheduled_tokens,
                                                  torch.tensor(seq_lens),
                                                  device=states.device)
        return all_pool(states, pooling_metadata)

    mesh = _make_single_device_mesh()
    runner = _make_runner(mesh, host_pooler)
    runner.input_batch.get_pooling_metadata = lambda: metadata
    runner.max_num_reqs = 4
    runner.device_buffer = _dirty_device_buffer(runner.max_num_reqs)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={
        "req-done": 4,
        "req-partial": 3
    })
    hidden = jnp.arange(14, dtype=jnp.float32).reshape(7, 2)

    out = TPUModelRunner._pool(runner, hidden, None, None, scheduler_output)
    # req-done finishes its 8-token prompt in this step (4 + 4); req-partial
    # is at 5 of 8 tokens.
    assert out.pooler_output[0] is not None
    assert torch.equal(out.pooler_output[0],
                       torch.arange(8, dtype=torch.float32).reshape(4, 2))
    assert out.pooler_output[1] is None


@pytest.mark.parametrize("num_reqs_paddings", [[8, 16], [1, 4, 32]])
def test_prewarm_compiles_every_request_bucket(num_reqs_paddings):
    mesh = _make_single_device_mesh()
    seen = []

    def compute_logits_fn(leaves, rows):
        assert leaves == "leaves"
        assert rows.dtype == jnp.bfloat16
        # Same sharding _select_from_array_fn produces at serving time, so
        # the warm-up hits the jit cache entry used by _pool.
        assert rows.sharding.spec == PartitionSpec(ShardingAxisName.ATTN_DATA,
                                                   None)
        seen.append(rows.shape)
        return jnp.zeros((rows.shape[0], 2), dtype=jnp.float32)

    runner = SimpleNamespace(
        mesh=mesh,
        num_reqs_paddings=num_reqs_paddings,
        model_config=SimpleNamespace(get_hidden_size=lambda: 4),
        state_leaves="leaves",
        compute_logits_fn=compute_logits_fn,
        pooler_fn=NativeSequenceClassificationPooler(head=None),
    )
    # Dispatch through CompilationManager like the real startup path does.
    CompilationManager._precompile_compute_pooling(
        SimpleNamespace(runner=runner))
    assert seen == [(n, 4) for n in num_reqs_paddings]
