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
"""Gang-symmetric continue_decode: a static window, per-request limits.

Under GANG_SYMMETRIC_DISPATCH every continue_decode call gets the window
static_max_decode_steps and, with the EOS early exit off
(CONTINUE_DECODE_EOS_CHECK_INTERVAL <= 0), runs exactly that many loop
iterations; with an EOS check the loop still stops after any row hits EOS and
the trip count follows the data, so the runner refuses that combination at
init. Each request still advances only within its own window, bounded by
max_model_len and by its max_tokens: past it the request is masked and only
repeats its last in-window step, while requests with more room keep decoding.
The number of calls per request thus follows the longest request rather than
the shortest one of each process, so gang members whose longest requests
match make the same calls. Mamba models fail closed where a window would be
shortened: in align mode at KV cache init, and per window (on the process
concerned) when max_model_len shortens one.

These tests run the real runner + decode loop with a small stand-in model on
CPU and check that no request decodes or writes KV past its own limit, that
each request gets the tokens, positions and KV of the stock path (per call
for the same inputs, and over whole requests with greedy decoding), that gang
members with different batches make the same calls with the same dispatch
signature, and that the runner fails closed where it cannot stay exact. The
window tests call _execute_continue_decode directly, without runner init, so
some of them also run the gang path with an EOS check.
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
    context before it, so a corrupted KV slot changes later tokens. Like
    ragged paged attention, a row writes its KV at position seq_len - 1, so
    the attention seq_lens a step is given decide where its KV lands. Row r
    emits EOS when it processes position eos_pos[r].
    """
    eos_pos = jnp.asarray(eos_pos, dtype=jnp.int32)
    rows_per_dp = batch_size // dp_size

    def model_fn(state, kv_caches, input_ids, attn_metadata, *args, **kwargs):
        slots, max_pos, calls, folded = kv_caches
        pos = attn_metadata.input_positions
        seq_lens = attn_metadata.seq_lens.reshape(dp_size,
                                                  -1)[:, :rows_per_dp].ravel()
        real = seq_lens > 0
        rows = jnp.arange(batch_size)
        width = slots.shape[1]
        before = jnp.arange(width)[None, :] < pos[:, None]
        context = jnp.sum(jnp.where(before, slots, 0), axis=1)
        next_tokens = (input_ids * 7 + pos * 13 + context) % (VOCAB - 2) + 2
        next_tokens = jnp.where(pos == eos_pos, EOS, next_tokens)
        write_pos = seq_lens - 1
        col = jnp.clip(write_pos, 0, width - 1)
        slots = slots.at[rows,
                         col].set(jnp.where(real, input_ids, slots[rows, col]))
        max_pos = jnp.where(real, jnp.maximum(max_pos, write_pos), max_pos)
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


def _cpu_mesh():
    from jax.sharding import Mesh

    from tpu_inference.layers.common.sharding import MESH_AXIS_NAMES
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    return Mesh(devices, MESH_AXIS_NAMES)


def _row_steps_sharding(mesh):
    from jax.sharding import NamedSharding, PartitionSpec

    from tpu_inference.layers.common.sharding import ShardingAxisName
    return NamedSharding(mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))


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
    # The request's max_tokens (None: unset).
    max_tokens: int | None = None
    # Prompt length (None: num_tokens - 1, i.e. one token generated).
    num_prompt: int | None = None
    # Token fed next (None: a synthetic one).
    last_token: int | None = None


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
                 async_scheduling=False,
                 pad_token_id=PAD,
                 kv=None,
                 model=None):
    width = max_model_len + KV_SLACK
    num_reqs = len(reqs)
    synthetic_kv = kv is None
    if synthetic_kv:
        kv = _init_kv(batch_size, width)
    slots = np.asarray(kv[0]).copy()
    positions = np.zeros(batch_size, np.int32)
    seq_lens = np.zeros(batch_size, np.int32)
    input_ids = np.full(batch_size, pad_token_id, np.int32)
    logits_indices = np.full(batch_size, -1, np.int32)
    eos_pos = np.full(batch_size, -1, np.int32)
    for r in reqs:
        n = r.num_tokens
        if synthetic_kv:
            prefix = (np.arange(n - 1) * 5 + r.row) % (VOCAB - 2)
            slots[r.row, :n - 1] = 2 + prefix
        positions[r.row] = n - 1
        seq_lens[r.row] = n
        input_ids[r.row] = (2 + (n * 3 + r.row) % (VOCAB - 2)
                            if r.last_token is None else r.last_token)
        logits_indices[r.row] = r.row
        eos_pos[r.row] = r.eos_pos
    kv = list(kv)
    kv[0] = jnp.asarray(slots)
    if model is None:
        model = _fake_model(batch_size, dp_size, eos_pos)
    model_fn, compute_logits_fn, sample_fn = model

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
    runner.pad_token_id = pad_token_id
    runner.mesh = _cpu_mesh()
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
    num_prompt = np.zeros(batch_size, np.int32)
    num_prompt[:num_reqs] = [
        r.num_tokens - 1 if r.num_prompt is None else r.num_prompt
        for r in reqs
    ]
    runner.input_batch.num_prompt_tokens = num_prompt
    runner.input_batch.token_ids_cpu = np.zeros((batch_size, max_model_len),
                                                np.int32)
    runner.requests = {}
    for rid, r in zip(req_ids, reqs):
        params = SimpleNamespace(max_tokens=r.max_tokens)
        runner.requests[rid] = SimpleNamespace(output_token_ids=[],
                                               sampling_params=params)
    # The real window computations, not mocks.
    runner._get_min_remaining_slots = (
        lambda: TPUModelRunner._get_min_remaining_slots(runner))
    runner._get_gang_row_decode_steps = (
        lambda: TPUModelRunner._get_gang_row_decode_steps(runner))

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


def _assert_rows_match_stock_alone(monkeypatch, gang, reqs, windows, **config):
    """Request i of a gang window decoded windows[i] tokens and got what the
    stock path gives it when it is alone in the batch (so that its own
    window, as far as max_model_len allows, is the batch's): the same tokens,
    as far as it went, the same KV for the positions it wrote, and no KV
    write at or past position num_tokens - 1 + windows[i]."""
    for i, (r, window) in enumerate(zip(reqs, windows)):
        alone = _run_window(monkeypatch, False, [r], **config)
        assert len(gang.tokens[i]) == window
        assert gang.tokens[i] == alone.tokens[0][:window]
        end = r.num_tokens - 1 + window
        assert int(gang.max_pos[i]) == end - 1
        np.testing.assert_array_equal(gang.slots[i][:end],
                                      alone.slots[0][:end])
        assert (gang.slots[i][end:] == -1).all(), (
            f"request {i} wrote KV at or past position {end}")


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


@pytest.mark.parametrize("others", [False, True])
def test_gang_decode_mamba_align_boundary_fails_closed(monkeypatch, others):
    """Mamba align mode, block boundary 128, seq len 126, max_decode_steps
    16: the request's window is 2 steps. Masked iterations would advance its
    recurrent state past the boundary, so the gang path must refuse before
    any device step, also when other requests with room share the batch.
    KV cache init already rejects align mode under the flag (see
    test_gang_kv_cache_init_rejects_mamba_align); this is the per-window
    backstop."""
    reqs = [Req(num_tokens=126, row=0)]
    if others:
        reqs += [Req(num_tokens=300, row=1), Req(num_tokens=40, row=2)]
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
    assert (gang.folded == 0).all()
    assert all(t == [] for t in gang.tokens)


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
    lengths. Each request decodes its own window in the same call (12 steps
    next to max_model_len, 32 with room), never writes past it, and gets the
    stock tokens and KV for it."""
    reqs = [
        Req(num_tokens=500, row=0),
        Req(num_tokens=100, row=1),
        Req(num_tokens=300, row=4),
    ]
    config = dict(max_model_len=512, static_steps=32, dp_size=2)
    gang = _run_window(monkeypatch, True, reqs, **config)

    _assert_rows_match_stock_alone(monkeypatch, gang, reqs, [12, 32, 32],
                                   **config)
    assert gang.num_tokens == [512, 132, 332]
    assert gang.calls == 32


@pytest.mark.parametrize("pad_token_id", [PAD, EOS])
def test_gang_decode_each_request_keeps_its_own_window(monkeypatch,
                                                       pad_token_id):
    """max_model_len 512, max_decode_steps 32: the request at 500 tokens
    decodes 12 tokens and writes KV only below position 512 (its last KV
    write is at 510; the token at 511, its last one, is never fed), while
    the requests with room decode all 32 steps in the same call. Tokens past
    a request's window are padding, which must not be taken for an EOS when
    the padding id is the EOS id."""
    reqs = [
        Req(num_tokens=500, row=0),
        Req(num_tokens=100, row=1),
        Req(num_tokens=300, row=3),
    ]
    config = dict(max_model_len=512,
                  static_steps=32,
                  pad_token_id=pad_token_id,
                  eos_check_interval=0)
    with patch("tpu_inference.runner.tpu_runner.continue_decode",
               wraps=continue_decode) as spy:
        gang = _run_window(monkeypatch, True, reqs, **config)

    kwargs = spy.call_args.kwargs
    assert int(kwargs["max_decode_steps"]) == 32
    assert kwargs["fixed_trip_count"] is True
    np.testing.assert_array_equal(kwargs["row_decode_steps"],
                                  [12, 32, 32, 32, 32, 32, 32, 32])
    assert kwargs["row_decode_steps"].sharding == _row_steps_sharding(
        gang.runner.mesh)
    assert int(gang.max_pos[0]) == 510
    assert (gang.slots[0][511:] == -1).all()
    assert all(EOS not in t for t in gang.tokens)
    _assert_rows_match_stock_alone(monkeypatch, gang, reqs, [12, 32, 32],
                                   **config)
    assert gang.num_tokens == [512, 132, 332]
    assert gang.calls == 32


def test_gang_decode_stops_a_request_at_its_max_tokens(monkeypatch):
    """A request 5 tokens short of its max_tokens decodes exactly 5 tokens
    and writes no KV past them (the stock loop decodes its whole window and
    leaves the cut to the scheduler), while the other request decodes 32."""
    reqs = [
        Req(num_tokens=200, row=0, num_prompt=150, max_tokens=55),
        Req(num_tokens=100, row=1, max_tokens=500),
    ]
    config = dict(max_model_len=512, static_steps=32)
    gang = _run_window(monkeypatch, True, reqs, **config)

    _assert_rows_match_stock_alone(monkeypatch, gang, reqs, [5, 32], **config)
    assert gang.num_tokens == [205, 132]
    stock = _run_window(monkeypatch, False, reqs, **config)
    assert [len(t) for t in stock.tokens] == [32, 32]


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


def test_gang_decode_async_scheduling_books_each_request_window(monkeypatch):
    """Async scheduling books placeholder tokens before the outputs are
    known. Each request must get its own window (12 next to max_model_len,
    32 with room), never the static window past max_model_len, and the
    resolved and reconciled outputs must be each request's real tokens."""
    # num_prompt == num_tokens: no earlier output tokens to keep.
    reqs = [
        Req(num_tokens=500, row=0, num_prompt=500),
        Req(num_tokens=100, row=1, num_prompt=100),
    ]
    config = dict(max_model_len=512, static_steps=32)
    gang = _run_window(monkeypatch,
                       True,
                       reqs,
                       async_scheduling=True,
                       **config)
    alone = [
        _run_window(monkeypatch, False, [r], **config).tokens[0] for r in reqs
    ]

    pre = gang.runner._pre_async_results
    assert pre.continue_decode_row_steps.tolist() == [12, 32]
    assert pre.continue_decode_max_steps == 32
    assert int(np.asarray(pre.continue_decode_actual_steps)) == 32
    assert gang.num_tokens == [512, 132]
    assert gang.tokens == [[0] * 12, [0] * 32]
    np.testing.assert_array_equal(
        np.asarray(pre.next_tokens_in_tpu)[[0, 1]], [t[-1] for t in alone])
    output = gang.runner._continue_decode_output.get_output()
    assert output.sampled_token_ids == alone
    assert int(gang.max_pos.max()) < 512

    TPUModelRunner._modify_prev_results(gang.runner)
    for i, r in enumerate(reqs):
        state = gang.runner.requests[f"req{i}"]
        assert state.output_token_ids == alone[i]
        assert state.num_computed_tokens == r.num_tokens + len(alone[i]) - 1
    assert gang.runner.input_batch.num_tokens[:2].tolist() == [512, 132]


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
    assert kwargs.get("row_decode_steps") is None
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


def _loop_inputs_dp(dp_size, rows_per_dp, seq_cols, width, reqs):
    """Loop inputs for dp_size ranks of rows_per_dp rows, with seq_lens
    padded to seq_cols entries per rank. reqs: (row, num_tokens) pairs."""
    batch_size = dp_size * rows_per_dp
    slots = np.full((batch_size, width), -1, np.int32)
    positions = np.zeros(batch_size, np.int32)
    seq_lens = np.zeros((dp_size, seq_cols), np.int32)
    tokens = np.full(batch_size, PAD, np.int32)
    active = np.zeros(batch_size, bool)
    for row, n in reqs:
        slots[row, :n - 1] = 2 + np.arange(n - 1) % (VOCAB - 2)
        positions[row] = n - 1
        seq_lens[row // rows_per_dp, row % rows_per_dp] = n
        tokens[row], active[row] = 2 + n % (VOCAB - 2), True
    kv = _init_kv(batch_size, width)
    kv[0] = jnp.asarray(slots)
    init_state = TpuSamplingState(
        current_tokens=jnp.asarray(tokens),
        active_mask=jnp.asarray(active),
        attn_metadata=AttentionMetadata(
            input_positions=jnp.asarray(positions),
            block_tables=jnp.zeros((batch_size * 4, ), jnp.int32),
            seq_lens=jnp.asarray(seq_lens.ravel()),
            query_start_loc=jnp.arange(batch_size + dp_size, dtype=jnp.int32),
            request_distribution=jnp.zeros((3 * dp_size, ), jnp.int32),
            mamba_state_indices=None,
        ),
        step_counter=jnp.array(0, jnp.int32),
    )
    return kv, init_state


def test_row_windows_match_the_stock_loop_row_by_row():
    """continue_decode(fixed_trip_count=True, row_decode_steps=w) gives each
    row r what the stock loop gives it with max_decode_steps=w[r]: tokens,
    final state and KV. Past its window a row records padding and writes no
    KV past position pos0 + w[r] - 1, while rows with longer windows keep
    decoding, and the loop always runs static_max_decode_steps steps. Two DP
    ranks with padded seq_lens check the seq_lens layout, and one row hits
    EOS inside its window. Row 0 is the Mamba block-boundary case (126
    tokens, boundary 128, 16 steps: a 2-step window). Its recurrent state is
    still run on every step, which is why the runner refuses Mamba models
    whose windows would need masking."""
    dp_size, rows_per_dp, seq_cols, width, static_steps = 2, 3, 4, 200, 16
    reqs = [(0, 126), (1, 40), (3, 100), (4, 60), (5, 20)]
    windows = {0: 2, 1: 16, 3: 7, 4: 16, 5: 1}
    # Row 4 processes positions 59, 60, ... and emits EOS at 63.
    eos_pos = [-1, -1, -1, -1, 63, -1]
    batch_size = dp_size * rows_per_dp
    row_steps = np.full(batch_size, static_steps, np.int32)
    for row, window in windows.items():
        row_steps[row] = window
    model_fn, compute_logits_fn, sample_fn = _fake_model(
        batch_size, dp_size, eos_pos)

    def run(max_decode_steps, fixed, row_decode_steps=None):
        kv, init_state = _loop_inputs_dp(dp_size, rows_per_dp, seq_cols, width,
                                         reqs)
        return continue_decode(
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
            dp_size=dp_size,
            continue_decode_eos_check_interval=0,
            fixed_trip_count=fixed,
            row_decode_steps=row_decode_steps,
        )

    tokens_g, kv_g, state_g, rng_g, _, _ = run(static_steps, True,
                                               jnp.asarray(row_steps))
    tokens_g = np.asarray(tokens_g)
    seq_g = np.asarray(state_g.attn_metadata.seq_lens).reshape(
        dp_size, seq_cols)
    assert int(state_g.step_counter) == static_steps
    assert int(kv_g[2]) == static_steps
    np.testing.assert_array_equal(
        np.asarray(rng_g),
        np.asarray(
            jax.random.split(jax.random.PRNGKey(0),
                             static_steps + 1)[static_steps]))
    for row, window in windows.items():
        tokens_s, kv_s, state_s, _, _, _ = run(window, False)
        tokens_s = np.asarray(tokens_s)
        seq_s = np.asarray(state_s.attn_metadata.seq_lens).reshape(
            dp_size, seq_cols)
        np.testing.assert_array_equal(tokens_g[:window, row], tokens_s[:window,
                                                                       row])
        assert (tokens_g[window:, row] == PAD).all()
        for name in ("current_tokens", "active_mask"):
            assert (np.asarray(getattr(state_g, name))[row] == np.asarray(
                getattr(state_s, name))[row])
        assert (np.asarray(
            state_g.attn_metadata.input_positions)[row] == np.asarray(
                state_s.attn_metadata.input_positions)[row])
        cell = (row // rows_per_dp, row % rows_per_dp)
        assert seq_g[cell] == seq_s[cell]
        np.testing.assert_array_equal(
            np.asarray(kv_g[0])[row],
            np.asarray(kv_s[0])[row])
        assert int(kv_g[1][row]) == int(kv_s[1][row])
        if row != 4:
            assert int(kv_g[1][row]) == dict(reqs)[row] - 1 + window - 1
        assert int(kv_g[3][row]) == static_steps
    # Row 4 stopped at its EOS (its 5th token), inside its window.
    assert tokens_g[4, 4] == EOS
    assert (tokens_g[5:, 4] == PAD).all()
    # The Mamba-like row stops at the block boundary: 126 + 2 = 128 tokens.
    assert int(seq_g[0, 0]) == 128
    # Seq_lens padding entries are untouched.
    assert (seq_g[:, rows_per_dp:] == 0).all()


def test_row_decode_steps_requires_fixed_trip_count():
    """Per-row windows exist only for the fixed-trip loop; the stock loop
    keeps its signature and program."""
    batch_size, width = 4, 64
    model_fn, compute_logits_fn, sample_fn = _fake_model(
        batch_size, 1, [-1] * batch_size)
    kv, init_state = _loop_inputs(batch_size, width, [(1, 30)])
    with pytest.raises(ValueError, match="fixed_trip_count"):
        continue_decode(
            state={},
            model_fn=model_fn,
            compute_logits_fn=compute_logits_fn,
            sample_fn=sample_fn,
            init_state=init_state,
            kv_caches=kv,
            max_decode_steps=jnp.array(4, jnp.int32),
            static_max_decode_steps=8,
            eos_token_id=(EOS, ),
            padding_token_id=PAD,
            rng=jax.random.PRNGKey(0),
            mesh=None,
            sampling_metadata=None,
            row_decode_steps=jnp.full((batch_size, ), 3, jnp.int32),
        )


@dataclass
class SimReq:
    # Tokens after prefill: the prompt plus the first generated token.
    num_tokens: int
    max_tokens: int | None = None
    # Prompt length (None: num_tokens - 1).
    num_prompt: int | None = None
    # Position at which the model emits EOS (-1: never).
    eos_pos: int = -1


def _dispatch_signature(kwargs):
    """What a continue_decode call hands the device: the program variant,
    its input shapes and its traced window."""

    def sig(x):
        return None if x is None else (tuple(x.shape), str(x.dtype))

    state = kwargs["init_state"]
    attn = state.attn_metadata
    return (kwargs.get("fixed_trip_count",
                       False), kwargs["static_max_decode_steps"],
            int(kwargs["max_decode_steps"]),
            kwargs.get("continue_decode_eos_check_interval"),
            sig(state.current_tokens), sig(state.active_mask),
            sig(attn.input_positions), sig(attn.seq_lens),
            sig(kwargs.get("row_decode_steps")))


def _simulate(monkeypatch, gang, reqs, rows=None, **config):
    """Decodes every request to its end through repeated continue_decode
    windows (greedy, no EOS early exit), finishing requests the way vLLM's
    scheduler does: at EOS, max_model_len or max_tokens, dropping any tokens
    past the stop. Each request keeps its physical row; a finished request
    leaves its row empty. Returns per-request tokens, final lengths and KV,
    the tokens each request got per call, and one dispatch signature per
    call."""
    if gang:
        monkeypatch.setenv(GANG_ENV, "1")
    else:
        monkeypatch.delenv(GANG_ENV, raising=False)
    max_model_len = config["max_model_len"]
    batch_size = config.get("batch_size", 8)
    rows = list(range(len(reqs))) if rows is None else rows
    eos_pos = np.full(batch_size, -1, np.int32)
    for r, row in zip(reqs, rows):
        eos_pos[row] = r.eos_pos
    model = _fake_model(batch_size, config.get("dp_size", 1), eos_pos)
    num_tokens = [r.num_tokens for r in reqs]
    num_prompt = [
        r.num_tokens - 1 if r.num_prompt is None else r.num_prompt
        for r in reqs
    ]
    outputs = [[] for _ in reqs]
    done = [False] * len(reqs)
    kv = None
    signatures, advanced = [], []
    with patch("tpu_inference.runner.tpu_runner.continue_decode",
               wraps=continue_decode) as spy, \
         patch("tpu_inference.layers.jax.sample.sampling.sample", model[2]):
        while not all(done):
            assert len(signatures) < 64
            live = [i for i in range(len(reqs)) if not done[i]]
            batch = [
                Req(num_tokens=num_tokens[i],
                    row=rows[i],
                    eos_pos=reqs[i].eos_pos,
                    max_tokens=reqs[i].max_tokens,
                    num_prompt=num_prompt[i],
                    last_token=outputs[i][-1] if outputs[i] else None)
                for i in live
            ]
            runner, _ = _make_runner(batch,
                                     kv=kv,
                                     model=model,
                                     eos_check_interval=0,
                                     **config)
            TPUModelRunner._execute_continue_decode(runner, MagicMock())
            kv = runner.kv_caches
            signatures.append(_dispatch_signature(spy.call_args.kwargs))
            step = {}
            for j, i in enumerate(live):
                new_tokens = runner.requests[f"req{j}"].output_token_ids
                step[i] = len(new_tokens)
                max_tokens = reqs[i].max_tokens
                for token in new_tokens:
                    outputs[i].append(token)
                    num_tokens[i] += 1
                    if (token == EOS or num_tokens[i] >= max_model_len or
                        (max_tokens is not None
                         and num_tokens[i] - num_prompt[i] >= max_tokens)):
                        done[i] = True
                        break
            advanced.append(step)
    return SimpleNamespace(tokens=outputs,
                           num_tokens=num_tokens,
                           signatures=signatures,
                           advanced=advanced,
                           slots=np.asarray(kv[0])[rows],
                           max_pos=np.asarray(kv[1])[rows])


def _assert_same_requests(stock, gang):
    """Every request got the same tokens on both paths and the same KV at
    every position it computed. On the gang path no request wrote KV past
    its last computed position (num_tokens - 2: the last token is never
    fed). The slot of a request's EOS step is left out: the stock loop
    overwrites it with the padding token's KV when a step follows the EOS
    in the same window, so it depends on where the windows end."""
    assert gang.tokens == stock.tokens
    assert gang.num_tokens == stock.num_tokens
    for i, (tokens, n) in enumerate(zip(gang.tokens, gang.num_tokens)):
        computed = n - 2 if tokens[-1] == EOS else n - 1
        np.testing.assert_array_equal(gang.slots[i][:computed],
                                      stock.slots[i][:computed])
        assert int(gang.max_pos[i]) == n - 2
        assert (gang.slots[i][n - 1:] == -1).all(), (
            f"request {i} wrote KV past its last computed position {n - 2}")


def test_gang_calls_follow_the_longest_request(monkeypatch):
    """The TPU trace pattern: two gang members with different numbers of
    requests and different short requests, but the same longest one (101
    tokens to go). With every window set by the member's shortest request
    (the stock path, and the gang path before per-request windows) they make
    5 and 6 continue_decode calls, which desyncs a gang whose model steps
    hold collectives. With per-request windows both make ceil(101 / 32) = 4
    calls with one dispatch signature, and every request still gets the
    tokens and KV of the stock path."""
    config = dict(max_model_len=512, static_steps=32)
    members = [
        # 101, 70, 45 and 37 tokens to go.
        [SimReq(411), SimReq(442),
         SimReq(467), SimReq(475)],
        # 101 and 1 tokens to go, 2 by max_tokens, and 3.
        [
            SimReq(411),
            SimReq(511),
            SimReq(301, num_prompt=300, max_tokens=3),
            SimReq(509)
        ],
    ]
    stock = [_simulate(monkeypatch, False, m, **config) for m in members]
    gang = [_simulate(monkeypatch, True, m, **config) for m in members]

    assert [len(s.signatures) for s in stock] == [5, 6]
    assert [len(g.signatures) for g in gang
            ] == [4, 4
                  ], ("continue_decode call counts differ across gang members")
    assert len({sig for g in gang for sig in g.signatures}) == 1
    for s, g in zip(stock, gang):
        _assert_same_requests(s, g)
    assert gang[0].advanced[0] == {0: 32, 1: 32, 2: 32, 3: 32}
    assert gang[1].advanced[0] == {0: 32, 1: 1, 2: 2, 3: 3}
    assert gang[0].num_tokens == [512] * 4
    assert gang[1].num_tokens == [512, 512, 303, 512]


def test_gang_eos_mid_window_and_mixed_lengths_across_dp_ranks(monkeypatch):
    """Two DP ranks holding 3 and 1 requests. In the first call one request
    hits EOS mid-window and stops there with its EOS, one stops at
    max_model_len and one at its max_tokens, while the longest decodes the
    whole window. The calls follow the longest request with one signature,
    and every request gets the tokens and KV of the stock path."""
    config = dict(max_model_len=512, static_steps=32, dp_size=2)
    reqs = [
        SimReq(300, eos_pos=310),  # EOS as its 12th token
        SimReq(500),  # 12 tokens to max_model_len
        SimReq(200, num_prompt=180, max_tokens=40),  # 20 to max_tokens
        SimReq(420),  # 92 tokens to go
    ]
    rows = [0, 1, 2, 4]
    stock = _simulate(monkeypatch, False, reqs, rows=rows, **config)
    gang = _simulate(monkeypatch, True, reqs, rows=rows, **config)

    assert gang.advanced[0] == {0: 12, 1: 12, 2: 20, 3: 32}
    assert len(gang.tokens[0]) == 12 and gang.tokens[0][-1] == EOS
    assert len(gang.signatures) == 3
    assert len(set(gang.signatures)) == 1
    assert len(stock.signatures) == 4
    _assert_same_requests(stock, gang)
    assert gang.num_tokens == [312, 512, 220, 512]


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
    # The per-request windows have the shape and sharding the runner gives
    # them (see test_gang_decode_each_request_keeps_its_own_window).
    for c in spy.call_args_list:
        rows = c.kwargs.get("row_decode_steps")
        if not gang:
            assert rows is None
            continue
        tokens = c.kwargs["init_state"].current_tokens
        assert rows.shape == tokens.shape and rows.dtype == jnp.int32
        assert rows.sharding == _row_steps_sharding(runner.mesh)


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
