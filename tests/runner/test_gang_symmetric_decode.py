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
"""Gang-symmetric continue_decode keeps the stock per-request safety window.

GANG_SYMMETRIC_DISPATCH bounds the fused decode loop by
static_max_decode_steps instead of the per-process safe window. With the EOS
early exit off (CONTINUE_DECODE_EOS_CHECK_INTERVAL <= 0) that bound is the
trip count, so every process of a gang runs the same number of loop
iterations; with an EOS check the loop still stops after any row hits EOS and
the trip count follows the data, so the runner refuses that combination at
init. A request may still only advance within the stock window, which is
bounded by max_model_len and, in Mamba align mode, by Mamba block boundaries.
These tests run the real runner + decode loop with a small stand-in model on
CPU and check that the gang path never writes past those bounds, produces the
same tokens, positions and KV as the stock path, and fails closed where it
cannot: at runner or KV cache init for static configs, and per window (on
the process concerned) for the data-dependent Mamba case. The window tests
call _execute_continue_decode directly, without runner init, so some of them
also run the gang path with an EOS check to show the loop still matches the
stock path there.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.runner import decode_loop
from tpu_inference.runner.decode_loop import TpuSamplingState, continue_decode
from tpu_inference.runner.tpu_runner import TPUModelRunner

EOS = 1
PAD = 0
VOCAB = 64
# Extra KV columns past max_model_len so that out-of-range writes are recorded
# instead of being clipped away.
KV_SLACK = 64
GANG_ENV = "GANG_SYMMETRIC_DISPATCH"


@pytest.fixture(autouse=True)
def _cpu_compatible_decode_core(monkeypatch):
    """XLA:CPU rejects the TPU-only compiler options the fused loop is built
    with; drop them off-TPU so the loop semantics can be checked anywhere."""
    if jax.default_backend() != "tpu":
        monkeypatch.setattr(decode_loop, "get_step_fn_compiler_options",
                            lambda: {})
        monkeypatch.setattr(decode_loop, "get_sampling_compiler_options",
                            lambda: {})
    decode_loop._get_decode_core.cache_clear()
    yield
    decode_loop._get_decode_core.cache_clear()


def _fake_model(batch_size, dp_size, eos_pos):
    """Stand-in for model_fn / compute_logits_fn / sample over paged KV.

    kv_caches is [slots, max_pos, calls, folded]:
      slots[r, p]  token whose KV row r wrote at position p (-1: never),
      max_pos[r]   highest position row r ever wrote,
      calls        number of model invocations,
      folded[r]    tokens folded into row r's recurrent (Mamba-like) state.
    The next token depends on the current token, its position and the KV
    context before it, so a corrupted KV slot changes later tokens. Row r
    emits EOS when it processes position eos_pos[r].
    """
    eos_pos = jnp.asarray(eos_pos, dtype=jnp.int32)
    rows_per_dp = batch_size // dp_size

    def model_fn(state, kv_caches, input_ids, attn_metadata, *args, **kwargs):
        slots, max_pos, calls, folded = kv_caches
        pos = attn_metadata.input_positions
        seq_lens = attn_metadata.seq_lens.reshape(dp_size, -1)
        real = seq_lens[:, :rows_per_dp].ravel() > 0
        rows = jnp.arange(batch_size)
        width = slots.shape[1]
        before = jnp.arange(width)[None, :] < pos[:, None]
        context = jnp.sum(jnp.where(before, slots, 0), axis=1)
        next_tokens = (input_ids * 7 + pos * 13 + context) % (VOCAB - 2) + 2
        next_tokens = jnp.where(pos == eos_pos, EOS, next_tokens)
        col = jnp.clip(pos, 0, width - 1)
        slots = slots.at[rows,
                         col].set(jnp.where(real, input_ids, slots[rows, col]))
        max_pos = jnp.where(real, jnp.maximum(max_pos, pos), max_pos)
        folded = folded + real.astype(jnp.int32)
        return ([slots, max_pos, calls + 1,
                 folded], next_tokens.astype(jnp.float32), None, None)

    def compute_logits_fn(state, hidden_states, lora_metadata):
        return jax.nn.one_hot(hidden_states.astype(jnp.int32), VOCAB)

    def sample_fn(rng, mesh, logits, sampling_metadata, **kwargs):
        return jnp.argmax(logits, axis=-1).astype(jnp.int32), logits

    # continue_decode unwraps the stock sampler via __wrapped__.
    sample_fn.__wrapped__ = sample_fn
    return model_fn, compute_logits_fn, sample_fn


def _init_kv(batch_size, width):
    return [
        jnp.full((batch_size, width), -1, jnp.int32),
        jnp.full((batch_size, ), -1, jnp.int32),
        jnp.zeros((), jnp.int32),
        jnp.zeros((batch_size, ), jnp.int32),
    ]


@dataclass
class Req:
    # Tokens so far (prompt + generated). The last one is fed next, at
    # position num_tokens - 1.
    num_tokens: int
    # Physical row of the request in the padded batch.
    row: int
    # Position at which the model emits EOS for this request (-1: never).
    eos_pos: int = -1


def _make_runner(reqs,
                 *,
                 max_model_len,
                 static_steps,
                 dp_size=1,
                 batch_size=8,
                 has_mamba=False,
                 mamba_cache_mode="none",
                 mamba_block_size=None,
                 eos_check_interval=1,
                 async_scheduling=False):
    width = max_model_len + KV_SLACK
    num_reqs = len(reqs)
    kv = _init_kv(batch_size, width)
    slots = np.asarray(kv[0]).copy()
    positions = np.zeros(batch_size, np.int32)
    seq_lens = np.zeros(batch_size, np.int32)
    input_ids = np.full(batch_size, PAD, np.int32)
    logits_indices = np.full(batch_size, -1, np.int32)
    eos_pos = np.full(batch_size, -1, np.int32)
    for r in reqs:
        n = r.num_tokens
        slots[r.row, :n - 1] = 2 + (np.arange(n - 1) * 5 + r.row) % (VOCAB - 2)
        positions[r.row] = n - 1
        seq_lens[r.row] = n
        input_ids[r.row] = 2 + (n * 3 + r.row) % (VOCAB - 2)
        logits_indices[r.row] = r.row
        eos_pos[r.row] = r.eos_pos
    kv[0] = jnp.asarray(slots)
    model_fn, compute_logits_fn, sample_fn = _fake_model(
        batch_size, dp_size, eos_pos)

    runner = MagicMock()
    runner.vllm_config.parallel_config.data_parallel_size = 1
    runner.vllm_config.parallel_config.use_sequence_parallel_moe = False
    runner.vllm_config.parallel_config.is_moe_model = False
    runner.scheduler_config.async_scheduling = async_scheduling
    runner._pre_async_results = None
    runner.max_model_len = max_model_len
    runner.max_num_reqs = batch_size
    runner.static_max_decode_steps = static_steps
    runner.continue_decode_eos_check_interval = eos_check_interval
    runner.dp_size = dp_size
    runner.eos_token_id = (EOS, )
    runner.pad_token_id = PAD
    runner.is_first_rank = True
    runner.is_last_rank = True
    runner.aux_output_worker = None
    runner.model_config.max_logprobs = 0
    runner.model_config.logprobs_mode = "raw"
    runner.layer_name_to_kvcache_index = {}
    runner.lora_utils.extract_lora_metadata.return_value = None
    runner.state_leaves = {}
    runner.model = SimpleNamespace(step_fn_no_options=model_fn)
    runner.compute_logits_fn = compute_logits_fn
    runner.rng_params_for_sampling = jax.random.PRNGKey(0)
    runner.kv_caches = kv
    runner.kv_cache_config.has_mamba_layers = has_mamba
    runner.cache_config.mamba_cache_mode = mamba_cache_mode
    runner.cache_config.mamba_block_size = mamba_block_size
    runner.block_size = 16

    req_ids = [f"req{i}" for i in range(num_reqs)]
    runner.input_batch.num_reqs = num_reqs
    runner.input_batch.req_ids = req_ids
    runner.input_batch.req_id_to_index = {
        rid: i
        for i, rid in enumerate(req_ids)
    }
    num_tokens = np.zeros(batch_size, np.int32)
    num_tokens[:num_reqs] = [r.num_tokens for r in reqs]
    runner.input_batch.num_tokens = num_tokens.copy()
    runner.input_batch.num_tokens_no_spec = num_tokens.copy()
    runner.input_batch.token_ids_cpu = np.zeros((batch_size, max_model_len),
                                                np.int32)
    runner.requests = {
        rid: SimpleNamespace(output_token_ids=[])
        for rid in req_ids
    }
    # The real safety window computation, not a mock.
    runner._get_min_remaining_slots = (
        lambda: TPUModelRunner._get_min_remaining_slots(runner))

    attn_metadata = AttentionMetadata(
        input_positions=jnp.asarray(positions),
        block_tables=jnp.zeros((batch_size * 4, ), jnp.int32),
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.arange(batch_size + dp_size, dtype=jnp.int32),
        request_distribution=jnp.zeros((3 * dp_size, ), jnp.int32),
        mamba_state_indices=None,
    )
    runner._prepare_inputs.return_value = (
        jnp.asarray(input_ids),
        None,
        attn_metadata,
        None,
        jnp.asarray(logits_indices),
        None,
        None,
        batch_size,
        None,
        None,
        [r.row for r in reqs],
        None,
    )
    return runner, sample_fn


def _run_window(monkeypatch, gang, reqs, catch=False, **config):
    """Runs one continue_decode window through the runner."""
    if gang:
        monkeypatch.setenv(GANG_ENV, "1")
    else:
        monkeypatch.delenv(GANG_ENV, raising=False)
    runner, sample_fn = _make_runner(reqs, **config)
    error = None
    with patch("tpu_inference.layers.jax.sample.sampling.sample", sample_fn):
        try:
            TPUModelRunner._execute_continue_decode(runner, MagicMock())
        except RuntimeError as e:
            if not catch:
                raise
            error = e
    slots, max_pos, calls, folded = (np.asarray(x) for x in runner.kv_caches)
    rows = [r.row for r in reqs]
    return SimpleNamespace(
        runner=runner,
        error=error,
        tokens=[
            list(runner.requests[f"req{i}"].output_token_ids)
            for i in range(len(reqs))
        ],
        num_tokens=runner.input_batch.num_tokens[:len(reqs)].tolist(),
        slots=slots[rows],
        max_pos=max_pos[rows],
        calls=int(calls),
        folded=folded[rows],
    )


def _assert_same_as_stock(gang, stock):
    assert gang.tokens == stock.tokens
    assert gang.num_tokens == stock.num_tokens
    np.testing.assert_array_equal(gang.max_pos, stock.max_pos)
    np.testing.assert_array_equal(gang.slots, stock.slots)


def test_gang_decode_stops_at_max_model_len(monkeypatch):
    """max_model_len 512, seq len 500, max_decode_steps 32: the safe window
    is 12 steps and no KV may be written at position >= 512."""
    reqs = [Req(num_tokens=500, row=0)]
    config = dict(max_model_len=512, static_steps=32)
    stock = _run_window(monkeypatch, False, reqs, **config)
    gang = _run_window(monkeypatch, True, reqs, **config)

    assert stock.calls == 12
    assert len(stock.tokens[0]) == 12
    assert int(gang.max_pos.max()) < 512, (
        f"gang-symmetric decode wrote KV at position {int(gang.max_pos.max())}"
        " >= max_model_len 512")
    _assert_same_as_stock(gang, stock)
    assert gang.num_tokens == [512]
    # The trip count is still static_max_decode_steps: no row hits EOS, so
    # the EOS check never ends the loop early.
    assert gang.calls == 32


def test_gang_decode_mamba_align_boundary_fails_closed(monkeypatch):
    """Mamba align mode, block boundary 128, seq len 126, max_decode_steps
    16: the safe window is 2 steps. Masked extra iterations would advance the
    recurrent state past the boundary, so the gang path must refuse before
    any device step instead of running 16 steps. KV cache init already
    rejects align mode under the flag (see
    test_gang_kv_cache_init_rejects_mamba_align); this is the per-window
    backstop."""
    reqs = [Req(num_tokens=126, row=0)]
    config = dict(max_model_len=1024,
                  static_steps=16,
                  has_mamba=True,
                  mamba_cache_mode="align",
                  mamba_block_size=128)
    stock = _run_window(monkeypatch, False, reqs, **config)
    assert stock.calls == 2
    assert 126 + int(stock.folded[0]) == 128

    gang = _run_window(monkeypatch, True, reqs, catch=True, **config)
    folded = int(gang.folded[0])
    assert 126 + folded <= 128, (
        f"gang-symmetric decode advanced the recurrent state to "
        f"{126 + folded} tokens, past the Mamba block boundary at 128")
    assert isinstance(gang.error, RuntimeError)
    assert GANG_ENV in str(gang.error)
    assert gang.calls == 0


def test_gang_decode_mamba_context_limit_fails_closed(monkeypatch):
    """Without align mode the window is still bounded by max_model_len; a
    Mamba model cannot be padded with masked iterations either."""
    reqs = [Req(num_tokens=500, row=0)]
    config = dict(max_model_len=512,
                  static_steps=32,
                  has_mamba=True,
                  mamba_cache_mode="none")
    gang = _run_window(monkeypatch, True, reqs, catch=True, **config)
    folded = int(gang.folded[0])
    assert 500 + folded <= 512, (
        f"gang-symmetric decode advanced the recurrent state to "
        f"{500 + folded} tokens, past max_model_len 512")
    assert isinstance(gang.error, RuntimeError)
    assert gang.calls == 0


def test_gang_decode_mamba_full_window_matches_stock(monkeypatch):
    """A Mamba model whose window is not shortened needs no masking, so the
    gang path runs and is identical to the stock path."""
    reqs = [Req(num_tokens=100, row=0), Req(num_tokens=300, row=1)]
    config = dict(max_model_len=1024,
                  static_steps=16,
                  has_mamba=True,
                  mamba_cache_mode="none")
    stock = _run_window(monkeypatch, False, reqs, **config)
    gang = _run_window(monkeypatch, True, reqs, **config)
    assert gang.error is None
    assert stock.calls == gang.calls == 16
    _assert_same_as_stock(gang, stock)
    np.testing.assert_array_equal(gang.folded, stock.folded)


def test_gang_decode_mixed_lengths_across_dp_ranks(monkeypatch):
    """Two DP ranks with different numbers of active requests and different
    lengths. The window is the minimum over the whole batch (12 here), every
    request stays inside it, and all outputs match the stock path."""
    reqs = [
        Req(num_tokens=500, row=0),
        Req(num_tokens=100, row=1),
        Req(num_tokens=300, row=4),
    ]
    config = dict(max_model_len=512, static_steps=32, dp_size=2)
    stock = _run_window(monkeypatch, False, reqs, **config)
    gang = _run_window(monkeypatch, True, reqs, **config)

    assert stock.calls == 12
    for i, r in enumerate(reqs):
        last_stock_write = r.num_tokens - 1 + 12 - 1
        assert int(gang.max_pos[i]) <= last_stock_write < 512, (
            f"request {i} wrote KV at position {int(gang.max_pos[i])}, past "
            f"its safe window ending at {last_stock_write}")
    _assert_same_as_stock(gang, stock)
    assert gang.num_tokens == [512, 112, 312]
    assert all(len(t) == 12 for t in gang.tokens)
    assert gang.calls == 32


def test_gang_decode_trip_count_is_data_independent(monkeypatch):
    """Gang members whose batches allow different windows (12 vs 32 steps)
    run the same number of loop iterations (no row hits EOS here), and each
    still matches its own stock output."""
    config = dict(max_model_len=512, static_steps=32)
    calls = []
    for reqs in ([Req(num_tokens=500, row=0)], [Req(num_tokens=100, row=0)]):
        stock = _run_window(monkeypatch, False, reqs, **config)
        gang = _run_window(monkeypatch, True, reqs, **config)
        assert int(gang.max_pos.max()) < 512
        _assert_same_as_stock(gang, stock)
        calls.append((stock.calls, gang.calls))
    assert calls == [(12, 32), (32, 32)]


@pytest.mark.parametrize("eos_check_interval", [1, -1])
def test_gang_decode_eos_inside_window(monkeypatch, eos_check_interval):
    """A request hitting EOS inside the window stops there (its tokens end
    with EOS and its position stays frozen) exactly as in the stock path,
    while the window-limiting request still stops at max_model_len. With an
    EOS check (interval 1) both loops exit early, so the gang trip count
    follows the data; runner init rejects that combination under
    GANG_SYMMETRIC_DISPATCH (test_gang_runner_init_requires_no_eos_exit)."""
    reqs = [
        Req(num_tokens=500, row=0),
        # Processes positions 199, 200, ...; emits EOS at position 203.
        Req(num_tokens=200, row=1, eos_pos=203),
    ]
    config = dict(max_model_len=512,
                  static_steps=32,
                  eos_check_interval=eos_check_interval)
    stock = _run_window(monkeypatch, False, reqs, **config)
    gang = _run_window(monkeypatch, True, reqs, **config)

    assert int(gang.max_pos.max()) < 512, (
        f"gang-symmetric decode wrote KV at position {int(gang.max_pos.max())}"
        " >= max_model_len 512")
    _assert_same_as_stock(gang, stock)
    assert len(gang.tokens[1]) == 5 and gang.tokens[1][-1] == EOS
    assert int(gang.max_pos[1]) == 203
    if eos_check_interval > 0:
        # Early exit right after the EOS step, in both paths.
        assert stock.calls == gang.calls == 5
        assert len(gang.tokens[0]) == 5
    else:
        assert (stock.calls, gang.calls) == (12, 32)
        assert len(gang.tokens[0]) == 12


def test_gang_decode_async_scheduling_respects_window(monkeypatch):
    """Async scheduling books placeholder tokens for the window before the
    outputs are known; it must book the safe window, not the static one."""
    reqs = [Req(num_tokens=500, row=0)]
    config = dict(max_model_len=512, static_steps=32)
    stock = _run_window(monkeypatch, False, reqs, **config)
    gang = _run_window(monkeypatch,
                       True,
                       reqs,
                       async_scheduling=True,
                       **config)

    pre = gang.runner._pre_async_results
    assert pre.continue_decode_max_steps == 12
    assert gang.num_tokens == [512]
    assert int(np.asarray(pre.continue_decode_actual_steps)) == 12
    output = gang.runner._continue_decode_output.get_output()
    assert output.sampled_token_ids == stock.tokens
    assert int(gang.max_pos.max()) < 512
    np.testing.assert_array_equal(gang.slots, stock.slots)


def test_flags_off_dispatch_is_unchanged(monkeypatch):
    """With GANG_SYMMETRIC_DISPATCH unset the runner dispatches exactly the
    stock loop: the clamped step count and no fixed trip count."""
    reqs = [Req(num_tokens=500, row=0)]
    with patch("tpu_inference.runner.tpu_runner.continue_decode",
               wraps=continue_decode) as spy:
        stock = _run_window(monkeypatch,
                            False,
                            reqs,
                            max_model_len=512,
                            static_steps=32)
    kwargs = spy.call_args.kwargs
    assert int(kwargs["max_decode_steps"]) == 12
    assert kwargs["static_max_decode_steps"] == 32
    assert kwargs.get("fixed_trip_count", False) is False
    assert stock.calls == 12


def _loop_inputs(batch_size, width, reqs):
    slots = np.full((batch_size, width), -1, np.int32)
    positions = np.zeros(batch_size, np.int32)
    seq_lens = np.zeros(batch_size, np.int32)
    tokens = np.full(batch_size, PAD, np.int32)
    active = np.zeros(batch_size, bool)
    for row, n in reqs:
        slots[row, :n - 1] = 2 + np.arange(n - 1) % (VOCAB - 2)
        positions[row], seq_lens[row] = n - 1, n
        tokens[row], active[row] = 2 + n % (VOCAB - 2), True
    kv = _init_kv(batch_size, width)
    kv[0] = jnp.asarray(slots)
    init_state = TpuSamplingState(
        current_tokens=jnp.asarray(tokens),
        active_mask=jnp.asarray(active),
        attn_metadata=AttentionMetadata(
            input_positions=jnp.asarray(positions),
            block_tables=jnp.zeros((batch_size * 4, ), jnp.int32),
            seq_lens=jnp.asarray(seq_lens),
            query_start_loc=jnp.arange(batch_size + 1, dtype=jnp.int32),
            request_distribution=jnp.zeros((3, ), jnp.int32),
            mamba_state_indices=None,
        ),
        step_counter=jnp.array(0, jnp.int32),
    )
    return kv, init_state


@pytest.mark.parametrize("max_decode_steps", [1, 5, 9])
def test_fixed_trip_count_loop_matches_stock_loop(max_decode_steps):
    """continue_decode(fixed_trip_count=True) returns what the stock loop
    returns for the same window while always running the static trip count."""
    batch_size, width, static_steps = 4, 64, 9
    reqs = [(0, 20), (2, 40)]
    eos_pos = [-1, -1, 42, -1]  # row 2 emits EOS on its 4th step
    model_fn, compute_logits_fn, sample_fn = _fake_model(
        batch_size, 1, eos_pos)

    results = {}
    for fixed in (False, True):
        kv, init_state = _loop_inputs(batch_size, width, reqs)
        results[fixed] = continue_decode(
            state={},
            model_fn=model_fn,
            compute_logits_fn=compute_logits_fn,
            sample_fn=sample_fn,
            init_state=init_state,
            kv_caches=kv,
            max_decode_steps=jnp.array(max_decode_steps, jnp.int32),
            static_max_decode_steps=static_steps,
            eos_token_id=(EOS, ),
            padding_token_id=PAD,
            rng=jax.random.PRNGKey(0),
            mesh=None,
            sampling_metadata=None,
            continue_decode_eos_check_interval=-1,
            fixed_trip_count=fixed,
        )
    (tokens_s, kv_s, state_s, rng_s, _, _) = results[False]
    (tokens_f, kv_f, state_f, rng_f, _, _) = results[True]

    np.testing.assert_array_equal(tokens_f, tokens_s)
    assert int(state_f.step_counter) == int(state_s.step_counter)
    for name in ("current_tokens", "active_mask"):
        np.testing.assert_array_equal(getattr(state_f, name),
                                      getattr(state_s, name))
    for name in ("input_positions", "seq_lens"):
        np.testing.assert_array_equal(getattr(state_f.attn_metadata, name),
                                      getattr(state_s.attn_metadata, name))
    np.testing.assert_array_equal(np.asarray(rng_f), np.asarray(rng_s))
    # Same KV slots and positions written; only the call count differs...
    np.testing.assert_array_equal(kv_f[0], kv_s[0])
    np.testing.assert_array_equal(kv_f[1], kv_s[1])
    assert int(kv_s[2]) == max_decode_steps
    assert int(kv_f[2]) == static_steps
    # ...which is why recurrent state is not preserved by the masked
    # iterations (and why the runner rejects Mamba models that would need
    # them): every call folds one more token into it.
    real = np.asarray([1, 0, 1, 0])
    np.testing.assert_array_equal(kv_s[3], real * max_decode_steps)
    np.testing.assert_array_equal(kv_f[3], real * static_steps)


def test_fixed_trip_count_loop_compiles_once_for_all_windows():
    """The window is a traced value: one executable serves every window."""
    batch_size, width, static_steps = 4, 64, 8
    model_fn, compute_logits_fn, sample_fn = _fake_model(
        batch_size, 1, [-1] * batch_size)
    core = decode_loop._get_decode_core()
    sizes = []
    for window in (2, 5, 8):
        kv, init_state = _loop_inputs(batch_size, width, [(1, 30)])
        _, kv_out, state, _, _, _ = continue_decode(
            state={},
            model_fn=model_fn,
            compute_logits_fn=compute_logits_fn,
            sample_fn=sample_fn,
            init_state=init_state,
            kv_caches=kv,
            max_decode_steps=jnp.array(window, jnp.int32),
            static_max_decode_steps=static_steps,
            eos_token_id=(EOS, ),
            padding_token_id=PAD,
            rng=jax.random.PRNGKey(0),
            mesh=None,
            sampling_metadata=None,
            fixed_trip_count=True,
        )
        assert int(state.step_counter) == window
        assert int(kv_out[2]) == static_steps
        sizes.append(core._cache_size())
    assert sizes[0] == sizes[1] == sizes[2]


def test_check_gang_symmetric_decode_window():
    from tpu_inference.runner.tpu_runner import \
        _check_gang_symmetric_decode_window as check

    # Attention-only models: masking is always safe.
    for safe in (1, 12, 32):
        check(safe_steps=safe,
              static_max_decode_steps=32,
              has_mamba_layers=False,
              mamba_cache_mode="none")
    # Mamba without align mode: fine while the window is full.
    check(safe_steps=32,
          static_max_decode_steps=32,
          has_mamba_layers=True,
          mamba_cache_mode="none")
    with pytest.raises(RuntimeError, match=GANG_ENV):
        check(safe_steps=12,
              static_max_decode_steps=32,
              has_mamba_layers=True,
              mamba_cache_mode="none")
    # Align mode bounds windows at every block boundary: always rejected.
    with pytest.raises(RuntimeError, match="align"):
        check(safe_steps=16,
              static_max_decode_steps=16,
              has_mamba_layers=True,
              mamba_cache_mode="align")


@pytest.mark.parametrize("gang", [False, True])
def test_precompile_uses_the_runtime_trip_count_mode(monkeypatch, gang):
    """The warmed-up continue_decode variant must be the one the runner
    dispatches, or serving pays a compile (or trips ForbidCompile)."""
    from jax.sharding import Mesh

    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
    from tpu_inference.runner.compilation_manager import CompilationManager

    if gang:
        monkeypatch.setenv(GANG_ENV, "1")
    else:
        monkeypatch.delenv(GANG_ENV, raising=False)
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    runner = MagicMock()
    runner.mesh = Mesh(devices, MESH_AXIS_NAMES)
    runner.vllm_config.sharding_config.total_dp_size = 1
    runner.vllm_config.additional_config = {"max_decode_steps": 8}
    runner.num_reqs_paddings_per_dp = [8]
    runner.attn_num_reqs_paddings_per_dp = [8]
    runner.uses_mrope = False
    runner.max_num_reqs = 8
    runner.kv_cache_config.has_mamba_layers = False
    runner.kv_cache_config.kv_cache_groups = [
        SimpleNamespace(layer_names=["layer.0"])
    ]
    runner.input_batch.block_table = {
        0: SimpleNamespace(max_num_blocks_per_req=4)
    }
    runner.lora_utils.extract_lora_metadata.return_value = None
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = runner
    manager._run_compilation = lambda name, fn, *args, **kwargs: fn(*args)

    with patch("tpu_inference.runner.compilation_manager.continue_decode",
               return_value=(None, ) * 6) as spy:
        CompilationManager._precompile_continue_decode(manager)

    assert spy.call_count > 0
    assert {
        c.kwargs.get("fixed_trip_count", False)
        for c in spy.call_args_list
    } == {gang}


def test_gang_decode_mamba_window_check_is_per_process(monkeypatch):
    """Outside align mode the Mamba check depends on the batch: it raises on
    the gang member whose request shortens the window and not on a member
    whose window is full, which keeps decoding."""
    config = dict(max_model_len=512,
                  static_steps=32,
                  has_mamba=True,
                  mamba_cache_mode="none")
    short = _run_window(monkeypatch,
                        True, [Req(num_tokens=500, row=0)],
                        catch=True,
                        **config)
    full = _run_window(monkeypatch,
                       True, [Req(num_tokens=100, row=0)],
                       catch=True,
                       **config)
    assert isinstance(short.error, RuntimeError)
    assert short.calls == 0
    assert full.error is None
    assert full.calls == 32


def _real_runner(monkeypatch, *, gang, enable_continue_decode,
                 eos_check_interval):
    """A real TPUModelRunner, with model loading and the device mesh stubbed
    the way tests/runner/test_tpu_runner.py does it."""
    from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                             SchedulerConfig, VllmConfig)

    if gang:
        monkeypatch.setenv(GANG_ENV, "1")
    else:
        monkeypatch.delenv(GANG_ENV, raising=False)
    monkeypatch.setenv("CONTINUE_DECODE_EOS_CHECK_INTERVAL",
                       str(eos_check_interval))
    devices = [MagicMock(coords=0)]
    device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
    mesh = jax.make_mesh(device_array.shape,
                         ("data", "attn_dp", "expert", "model"))
    with patch("jax.devices", return_value=devices), \
         patch("jax.make_mesh", return_value=mesh), \
         patch("jax.random.key", return_value=MagicMock()), \
         patch("tpu_inference.runner.tpu_runner.get_model",
               return_value=MagicMock()), \
         patch("tpu_inference.runner.tpu_runner.make_optimized_mesh",
               return_value=mesh):
        vllm_config = VllmConfig(
            model_config=ModelConfig(tokenizer_mode="auto",
                                     trust_remote_code=False,
                                     seed=0,
                                     dtype="bfloat16"),
            cache_config=CacheConfig(block_size=16,
                                     gpu_memory_utilization=0.9,
                                     cache_dtype="auto"),
            scheduler_config=SchedulerConfig(max_num_seqs=16,
                                             max_model_len=1024,
                                             is_encoder_decoder=False),
            parallel_config=ParallelConfig(pipeline_parallel_size=1,
                                           tensor_parallel_size=1),
            observability_config={},
            additional_config={
                "enable_continue_decode": enable_continue_decode
            },
        )
        return TPUModelRunner(vllm_config, devices=devices)


@pytest.mark.parametrize(
    "gang, enable_continue_decode, eos_check_interval, rejected",
    [
        # The EOS early exit makes the trip count follow each process's data.
        (True, True, 1, True),
        (True, True, 4, True),
        # No EOS early exit: the trip count is static_max_decode_steps.
        (True, True, 0, False),
        (True, True, -1, False),
        # Without continue_decode the fused loop never runs.
        (True, False, 1, False),
        # Flag unset: unchanged.
        (False, True, 1, False),
    ])
def test_gang_runner_init_requires_no_eos_exit(monkeypatch, gang,
                                               enable_continue_decode,
                                               eos_check_interval, rejected):
    """Runner init fails closed when GANG_SYMMETRIC_DISPATCH and
    continue_decode are combined with the EOS early exit, before any
    request, so every gang member fails alike."""
    kwargs = dict(gang=gang,
                  enable_continue_decode=enable_continue_decode,
                  eos_check_interval=eos_check_interval)
    if rejected:
        with pytest.raises(ValueError,
                           match="CONTINUE_DECODE_EOS_CHECK_INTERVAL <= 0"):
            _real_runner(monkeypatch, **kwargs)
    else:
        runner = _real_runner(monkeypatch, **kwargs)
        assert (
            runner.continue_decode_eos_check_interval == eos_check_interval)


class _KVCacheInitReached(Exception):
    pass


@pytest.mark.parametrize(
    "gang, enable_continue_decode, has_mamba, mamba_cache_mode, rejected",
    [
        (True, True, True, "align", True),
        # Other Mamba modes are checked per window instead.
        (True, True, True, "none", False),
        (True, True, False, "align", False),
        (True, False, True, "align", False),
        # Flag unset: unchanged.
        (False, True, True, "align", False),
    ])
def test_gang_kv_cache_init_rejects_mamba_align(monkeypatch, gang,
                                                enable_continue_decode,
                                                has_mamba, mamba_cache_mode,
                                                rejected):
    """Mamba align mode is a static config, so KV cache init rejects it under
    GANG_SYMMETRIC_DISPATCH with continue_decode, before allocating the KV
    cache, instead of the first continue_decode window."""
    runner = _real_runner(monkeypatch,
                          gang=gang,
                          enable_continue_decode=enable_continue_decode,
                          eos_check_interval=0)
    runner.cache_config.mamba_cache_mode = mamba_cache_mode
    runner.kv_cache_manager = MagicMock()
    runner.kv_cache_manager.initialize_kv_cache.side_effect = (
        _KVCacheInitReached)
    kv_cache_config = SimpleNamespace(has_mamba_layers=has_mamba,
                                      kv_cache_groups=[])
    if rejected:
        with pytest.raises(ValueError, match="mamba_cache_mode='align'"):
            runner.initialize_kv_cache(kv_cache_config)
        runner.kv_cache_manager.initialize_kv_cache.assert_not_called()
    else:
        with pytest.raises(_KVCacheInitReached):
            runner.initialize_kv_cache(kv_cache_config)
