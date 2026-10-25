# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Tests for the opt-in DPScheduler routing mirror (DP_SCHED_ROUTE_MIRROR).

Every DP rank runs the installed vLLM Scheduler in a forked worker process, as
in production; only the DP head's config is a stub. A trace is replayed on a
stock DPScheduler and on a DPScheduler with the mirror installed (a private
copy of dp_scheduler.py imported with DP_SCHED_ROUTE_MIRROR=1, the production
install path), and every routing decision, unfinished count, finished request
and abort result is compared.
"""

import collections
import contextlib
import importlib.util
import itertools
import json
import logging
import os
import random
import signal
import sys
import types

import pytest
import torch
import vllm
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import \
    register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tpu_inference.core.sched import dp_scheduler as stock_dp

BLOCK = 16
BLOCKS_PER_RANK = 512
MAX_NUM_SEQS = 8
MAX_BATCHED_TOKENS = 256
MAX_MODEL_LEN = 2048
_COPIES = itertools.count()

# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _real_vllm_config(model_dir, prefix_caching):
    model_config = ModelConfig(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        seed=0,
        max_model_len=MAX_MODEL_LEN,
    )
    vllm_config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_BATCHED_TOKENS,
            max_model_len=MAX_MODEL_LEN,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        model_config=model_config,
        cache_config=CacheConfig(
            block_size=BLOCK,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=prefix_caching,
        ),
        parallel_config=ParallelConfig(),
    )
    register_all_kvcache_specs(vllm_config)
    return vllm_config


@pytest.fixture(scope="module")
def real_configs(tmp_path_factory):
    """Real vLLM configs for the rank schedulers (a tiny local model config)."""
    model_dir = tmp_path_factory.mktemp("tiny_model")
    with open(model_dir / "config.json", "w") as fh:
        json.dump(
            {
                "model_type": "opt",
                "architectures": ["OPTForCausalLM"],
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "ffn_dim": 32,
                "vocab_size": 128,
                "max_position_embeddings": MAX_MODEL_LEN,
                "word_embed_proj_dim": 16,
                "torch_dtype": "float32",
                "eos_token_id": 2,
                "bos_token_id": 2,
                "pad_token_id": 1,
            },
            fh,
        )
    init_none_hash(sha256)
    return {
        False: _real_vllm_config(str(model_dir), prefix_caching=False),
        True: _real_vllm_config(str(model_dir), prefix_caching=True),
    }


class _RankScheduler(Scheduler):
    """The installed vLLM Scheduler, built in the worker from a real config.

    The DP head only sees a stub config; `real_config` is set on a per-test
    subclass before the workers are forked.
    """

    real_config = None

    def __init__(
        self,
        vllm_config,
        kv_cache_config,
        structured_output_manager,
        block_size,
        hash_block_size=None,
        mm_registry=None,
        include_finished_set=False,
        log_stats=False,
    ):
        cfg = type(self).real_config
        cfg.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        super().__init__(
            vllm_config=cfg,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(cfg),
            block_size=block_size,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )


class _RemoteKVRankScheduler(_RankScheduler):
    """Puts requests that ask for a remote prefill into the async KV load state.

    This is what vLLM's scheduler does when a KV connector reports an
    asynchronous load: the request becomes WAITING_FOR_REMOTE_KVS and waits,
    deferred, in kv_holding_waiting until the load finishes. No connector is
    configured here, so the load never finishes and the request stays there
    until it is aborted. Only schedule() is overridden, not a method the
    mirror models.
    """

    def schedule(self, *args, **kwargs):
        for req in list(self.waiting):
            params = req.kv_transfer_params or {}
            if params.get("do_remote_prefill"
                          ) and req.status == RequestStatus.WAITING:
                self.waiting.remove_request(req)
                req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                self.deferred_waiting.add(req)
                self.kv_holding_waiting.add_request(req)
        return super().schedule(*args, **kwargs)


class _OverridesAddRequest(_RankScheduler):
    """A scheduler class that overrides a method the mirror models."""

    def add_request(self, request):
        super().add_request(request)


def _rank_scheduler_cls(real_configs, prefix_caching=False, base=None):
    base = base or _RankScheduler
    return type(base.__name__, (base, ),
                {"real_config": real_configs[prefix_caching]})


def _head_config(dp, sched_cls, prefix_caching=False, kv_transfer=None):
    return types.SimpleNamespace(
        sharding_config=types.SimpleNamespace(total_dp_size=dp),
        scheduler_config=types.SimpleNamespace(
            _original_scheduler_cls=sched_cls,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=MAX_BATCHED_TOKENS,
            async_scheduling=False,
        ),
        cache_config=types.SimpleNamespace(
            enable_prefix_caching=prefix_caching,
            prefix_caching_hash_algo="sha256",
            mamba_num_blocks=None,
        ),
        additional_config={},
        kv_transfer_config=kv_transfer,
    )


def _kv_cache_config(dp, blocks_per_rank):
    spec = FullAttentionSpec(block_size=BLOCK,
                             num_kv_heads=1,
                             head_size=1,
                             dtype=torch.float32)
    return KVCacheConfig(
        num_blocks=blocks_per_rank * dp,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )


def _mirrored_module(monkeypatch, **env):
    """Import a private copy of dp_scheduler.py with DP_SCHED_ROUTE_MIRROR=1.

    This runs the production install path (the guarded block at the end of
    dp_scheduler.py), on a DPScheduler class of its own.
    """
    monkeypatch.setenv("DP_SCHED_ROUTE_MIRROR", "1")
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    name = "tpu_inference.core.sched._dp_scheduler_route_mirror_test_%d" % (
        next(_COPIES))
    spec = importlib.util.spec_from_file_location(name, stock_dp.__file__)
    module = importlib.util.module_from_spec(spec)
    # Registered so the workers unpickle its SchedulerCommand by reference.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _dp_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DP_SCHED_"):
            monkeypatch.delenv(key)
    # Per-request routing (batched prefill keeps the stock path by design).
    monkeypatch.setenv("DP_SCHED_BATCH_PREFILL", "0")
    assert not getattr(stock_dp.DPScheduler, "_route_mirror_installed",
                       False), "stock dp_scheduler imported with the mirror"


class _Engine:
    """Drives a DPScheduler the way EngineCore does (one CPU model step)."""

    def __init__(self,
                 dp_cls,
                 dp,
                 sched_cls,
                 blocks_per_rank=BLOCKS_PER_RANK,
                 **head_kwargs):
        self.sched = dp_cls(
            vllm_config=_head_config(dp, sched_cls, **head_kwargs),
            kv_cache_config=_kv_cache_config(dp, blocks_per_rank),
            structured_output_manager=None,
            block_size=BLOCK,
            log_stats=False,
        )
        self.prompt_len = {}
        self.done = []  # every request id reported finished or aborted
        self.resumed = 0  # requests resumed after a preemption
        self.stock_picks = 0
        pick = self.sched._find_best_rank_for_request

        def counting_pick(request):
            self.stock_picks += 1
            return pick(request)

        self.sched._find_best_rank_for_request = counting_pick

    @property
    def mirror(self):
        return getattr(self.sched, "_route_mirror", None)

    def add(self, request):
        self.prompt_len[request.request_id] = request.num_prompt_tokens
        self.sched.add_request(request)
        return self.sched.assigned_dp_rank[request.request_id]

    def step(self):
        out = self.sched.schedule()
        computed = {
            r.req_id: r.num_computed_tokens
            for r in out.scheduled_new_reqs
        }
        cached = out.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        self.resumed += len(cached.resumed_req_ids)
        req_ids = list(out.num_scheduled_tokens)
        sampled = [
            ([7] if computed[rid] +
             out.num_scheduled_tokens[rid] >= self.prompt_len[rid] else [])
            for rid in req_ids
        ]
        result = self.sched.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={
                    rid: i
                    for i, rid in enumerate(req_ids)
                },
                sampled_token_ids=sampled,
            ),
        )
        finished = sorted(o.request_id for eco in result.values()
                          for o in eco.outputs if o.finish_reason is not None)
        self.done.extend(finished)
        return out.total_num_scheduled_tokens, finished

    def abort(self, request_ids):
        infos = self.sched.finish_requests(request_ids,
                                           RequestStatus.FINISHED_ABORTED)
        aborted = sorted(i.request_id for i in infos)
        self.done.extend(aborted)
        return aborted

    def count(self):
        return self.sched.get_num_unfinished_requests()

    def pause(self, state):
        self.sched.set_pause_state(state)

    def shutdown(self):
        self.sched.shutdown()


@pytest.fixture
def engines():
    made = []

    def make(*args, **kwargs):
        engine = _Engine(*args, **kwargs)
        made.append(engine)
        return engine

    yield make
    for engine in made:
        with contextlib.suppress(Exception):
            engine.shutdown()


def _prompt(rid, length):
    rng = random.Random(rid)
    return [rng.randrange(3, 128) for _ in range(length)]


def _request(spec, block_hasher=None):
    rid, length, max_tokens, kind = spec
    structured = (StructuredOutputsParams(
        json={"type": "object"}) if kind == "grammar" else None)
    extra = ({
        "kv_transfer_params": {
            "do_remote_prefill": True
        }
    } if kind == "remote_kv" else None)
    req = Request(
        rid,
        _prompt(rid if kind != "shared" else "shared", length),
        SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=True,
            structured_outputs=structured,
            extra_args=extra,
        ),
        None,
        resumable=kind == "resumable",
        block_hasher=block_hasher,
    )
    if kind == "kv_arrival":
        # Arrives already holding KV blocks (remote KV load in flight).
        req.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    return req


_LENGTHS = (1, 7, 16, 33, 64, 150, 300)


def _trace(seed, n_ops, kinds=("plain", ), max_burst=6, pauses=False):
    """A deterministic trace of EngineCore-level operations."""
    rng = random.Random(seed)
    ops, added = [], []
    for _ in range(n_ops):
        r = rng.random()
        if r < 0.4:
            for _ in range(rng.randint(1, max_burst)):
                spec = (
                    "s%d-r%d" % (seed, len(added)),
                    rng.choice(_LENGTHS),
                    rng.randint(1, 6),
                    rng.choice(kinds),
                )
                ops.append(("add", spec))
                added.append(spec[0])
        elif r < 0.8:
            ops.append(("step", ))
        elif r < 0.9 and added:
            ops.append(("abort", [rng.choice(added)]))
        elif pauses:
            ops.append((
                "pause",
                rng.choice([
                    PauseState.PAUSED_NEW,
                    PauseState.PAUSED_ALL,
                    PauseState.UNPAUSED,
                ]),
            ))
        else:
            ops.append(("step", ))
    if pauses:
        ops.append(("pause", PauseState.UNPAUSED))
    return ops, added


def _apply(engine, op, block_hasher=None):
    kind = op[0]
    if kind == "add":
        return engine.add(_request(op[1], block_hasher))
    if kind == "step":
        return engine.step()
    if kind == "abort":
        return engine.abort(op[1])
    if kind == "pause":
        return engine.pause(op[1])
    raise AssertionError(op)


def _replay(stock,
            mirror,
            ops,
            added,
            block_hasher=None,
            max_drain=200,
            reported_per_turn=()):
    """Replay `ops` on both engines; every result must be identical.

    `reported_per_turn`: resumable requests, reported with a finish reason at
    the end of every turn and once more when aborted.
    """
    for i, op in enumerate(ops):
        got = _apply(mirror, op, block_hasher)
        want = _apply(stock, op, block_hasher)
        assert got == want, "op %d %r: mirror=%r stock=%r" % (i, op, got, want)
        assert mirror.sched.assigned_dp_rank == stock.sched.assigned_dp_rank
        assert mirror.count() == stock.count(), "op %d %r" % (i, op)
    # Drain like EngineCore: step while requests remain; abort what cannot
    # finish (blocked grammar, remote KV) once nothing is scheduled.
    for _ in range(max_drain):
        n = stock.count()
        assert mirror.count() == n
        if n == 0:
            break
        got, want = mirror.step(), stock.step()
        assert got == want
        if want[0] == 0:
            assert mirror.abort(None) == stock.abort(None)
    else:
        raise AssertionError("trace did not drain")
    # No request lost or reported twice.
    assert sorted(mirror.done) == sorted(stock.done)
    once = collections.Counter(stock.done)
    assert sorted(once) == sorted(added)
    assert all(n == 1 for rid, n in once.items()
               if rid not in reported_per_turn)


def _assert_mirror_ran(mirror, expect_stock_picks=0):
    st = mirror.mirror
    assert st is not None
    assert st.failed == "", st.failed
    assert st.route_on, st.route_reason
    assert mirror.stock_picks == expect_stock_picks


class _DeadlineExceeded(BaseException):
    """Raised by _deadline.

    Not an Exception (TimeoutError would be an OSError), so neither a
    pytest.raises for the error a test expects nor an `except Exception` in
    the code under test (DPScheduler._get_result wraps those into a
    RuntimeError) can turn a hang into a pass.
    """


@contextlib.contextmanager
def _deadline(seconds):
    """Fail instead of hanging (a deferred-ack deadlock would hang)."""

    def on_alarm(signum, frame):
        raise _DeadlineExceeded("no progress within %ss" % seconds)

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# --------------------------------------------------------------------------
# Install on the pinned vLLM, and the install-time verification
# --------------------------------------------------------------------------


def test_install_succeeds_on_installed_vllm(monkeypatch):
    module = _mirrored_module(monkeypatch)
    assert module.DPScheduler._route_mirror_installed
    for name in (
            "_route_and_forward_request",
            "get_num_unfinished_requests",
            "_get_result",
            "_send_command",
            "shutdown",
    ):
        assert hasattr(getattr(module.DPScheduler, name), "__wrapped__")


def test_install_is_idempotent_per_class(monkeypatch):
    from tpu_inference.core.sched import dp_route_mirror

    module = _mirrored_module(monkeypatch)
    wrapped = module.DPScheduler._route_and_forward_request
    dp_route_mirror.install(logging.getLogger(__name__), module=module)
    assert module.DPScheduler._route_and_forward_request is wrapped


def _skipped_waiting_enqueue(self, request):
    # The queue layout of an older vLLM: blocked requests in a separate queue.
    if self._is_blocked_waiting_status(request.status):
        self.skipped_waiting.add_request(request)
    else:
        self.waiting.add_request(request)


def _blocked_not_waiting_enqueue(self, request):
    # Blocked requests left out of `waiting`, KV holders in it.
    if self._is_blocked_waiting_status(request.status):
        self.kv_holding_waiting.add_request(request)
    else:
        self.waiting.add_request(request)


def _unfinished_without_kv_holding(self):
    return len(self.waiting) + len(self.running)


def _add_request_with_new_state(self, request, _orig=Scheduler.add_request):
    _orig(self, request)
    self.added_since_probe = True


@pytest.mark.parametrize(
    "method, replacement, expect",
    [
        ("_enqueue_waiting_request", _skipped_waiting_enqueue,
         "skipped_waiting"),
        (
            "_enqueue_waiting_request",
            _blocked_not_waiting_enqueue,
            "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
        ),
        (
            "get_num_unfinished_requests",
            _unfinished_without_kv_holding,
            "get_num_unfinished_requests=",
        ),
        ("add_request", _add_request_with_new_state, "added_since_probe"),
    ],
    ids=[
        "older-skipped-waiting-queue",
        "blocked-left-out-of-waiting",
        "unfinished-count-shape",
        "unmodelled-scheduler-state",
    ],
)
def test_install_rejects_unsupported_vllm_scheduler(monkeypatch, method,
                                                    replacement, expect):
    monkeypatch.setattr(Scheduler, method, replacement)
    with pytest.raises(RuntimeError) as err:
        _mirrored_module(monkeypatch)
    msg = str(err.value)
    assert "does not match the modelled shape" in msg
    assert "installed vLLM %s Scheduler" % vllm.__version__ in msg
    assert expect in msg
    assert "unset DP_SCHED_ROUTE_MIRROR" in msg


def test_install_rejects_window_above_maximum(monkeypatch):
    with pytest.raises(RuntimeError, match="above the maximum 64"):
        _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_WINDOW=65)


def test_flags_off_leaves_dp_scheduler_untouched(monkeypatch):
    from tpu_inference.core.sched import dp_route_mirror
    monkeypatch.delenv("DP_SCHED_ROUTE_MIRROR", raising=False)
    name = "tpu_inference.core.sched._dp_scheduler_flags_off_test"
    spec = importlib.util.spec_from_file_location(name, stock_dp.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    for dp in (module.DPScheduler, stock_dp.DPScheduler):
        assert not hasattr(dp, "_route_mirror_installed")
        for value in vars(dp).values():
            code = getattr(value, "__code__", None)
            assert code is None or code.co_filename != dp_route_mirror.__file__
    with open(stock_dp.__file__) as fh:
        src = fh.read()
    # The only use of the mirror module is the block guarded by the flag.
    guarded = src[src.index("if envs.DP_SCHED_ROUTE_MIRROR:"):]
    assert src.count("import dp_route_mirror") == 1
    assert "import dp_route_mirror" in guarded


# --------------------------------------------------------------------------
# Routing and counts against the stock DPScheduler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dp", [2, 3])
@pytest.mark.parametrize("seed", [0, 1])
def test_mixed_prompt_bursts_match_stock(monkeypatch, real_configs, engines,
                                         dp, seed):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, dp, cls)
    mirror = engines(module.DPScheduler, dp, cls)
    ops, added = _trace(seed, 60)
    _replay(stock, mirror, ops, added)
    _assert_mirror_ran(mirror)
    assert mirror.mirror.bursts > 5


def test_tied_ranks_in_one_burst_match_stock(monkeypatch, real_configs,
                                             engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 4, cls)
    mirror = engines(module.DPScheduler, 4, cls)
    # Twelve identical prompts from idle: every pick is a tie on load and
    # in-flight count, broken towards the lowest rank.
    ops = [("add", ("t%d" % i, 32, 2, "plain")) for i in range(12)]
    ops += [("step", ), ("add", ("u0", 32, 2, "plain")),
            ("add", ("u1", 32, 2, "plain"))]
    ranks = [mirror.add(_request(op[1])) for op in ops[:12]]
    assert ranks == [stock.add(_request(op[1])) for op in ops[:12]]
    assert ranks == [0, 1, 2, 3] * 3
    _replay(stock, mirror, ops[12:],
            [op[1][0] for op in ops if op[0] == "add"])
    _assert_mirror_ran(mirror)


def test_grammar_waiting_requests_match_stock(monkeypatch, real_configs,
                                              engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops, added = _trace(2, 60, kinds=("plain", "grammar", "grammar"))
    assert any(op[0] == "add" and op[1][3] == "grammar" for op in ops)
    _replay(stock, mirror, ops, added)
    _assert_mirror_ran(mirror)


def test_remote_kv_loading_requests_match_stock(monkeypatch, real_configs,
                                                engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs, base=_RemoteKVRankScheduler)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops, added = _trace(3,
                        60,
                        kinds=("plain", "remote_kv", "kv_arrival", "grammar"))
    _replay(stock, mirror, ops, added)
    _assert_mirror_ran(mirror)


def test_preemption_under_kv_pressure_matches_stock(monkeypatch, real_configs,
                                                    engines):
    # 24 blocks per rank: prompts of 4 blocks that decode 3 more blocks each
    # do not all fit, so running requests are preempted back to `waiting`
    # and resumed later, between and during bursts.
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls, blocks_per_rank=24)
    mirror = engines(module.DPScheduler, 2, cls, blocks_per_rank=24)
    ops, added = [], []
    for burst in range(5):
        for i in range(4):
            rid = "p%d-%d" % (burst, i)
            ops.append(("add", (rid, 64 - 8 * (i % 2), 48, "plain")))
            added.append(rid)
        ops += [("step", )] * 6
    _replay(stock, mirror, ops, added, max_drain=1000)
    assert stock.resumed and mirror.resumed == stock.resumed
    _assert_mirror_ran(mirror)


def test_cancel_inside_and_between_bursts_match_stock(monkeypatch,
                                                      real_configs, engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops = [("add", ("c%d" % i, n, 3, "plain"))
           for i, n in enumerate((40, 10, 90, 5))]
    # Abort a request whose ADD ack is still deferred, mid-burst, then keep
    # adding; then abort ids that are unknown or already aborted.
    ops += [
        ("abort", ["c2"]),
        ("add", ("c4", 20, 3, "plain")),
        ("add", ("c5", 70, 3, "plain")),
        ("abort", ["c2", "nope"]),
        ("step", ),
        ("abort", ["c0", "c5"]),
        ("add", ("c6", 8, 3, "plain")),
        ("step", ),
    ]
    _replay(stock, mirror, ops, ["c%d" % i for i in range(7)])
    _assert_mirror_ran(mirror)


def test_pause_resume_match_stock(monkeypatch, real_configs, engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs, base=_RemoteKVRankScheduler)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops, added = _trace(4,
                        80,
                        kinds=("plain", "plain", "remote_kv"),
                        pauses=True)
    assert any(op[0] == "pause" and op[1] == PauseState.PAUSED_NEW
               for op in ops)
    assert any(op[0] == "pause" and op[1] == PauseState.PAUSED_ALL
               for op in ops)
    state, adds_while_paused = PauseState.UNPAUSED, 0
    for op in ops:
        if op[0] == "pause":
            state = op[1]
        adds_while_paused += op[0] == "add" and state != PauseState.UNPAUSED
    assert adds_while_paused
    _replay(stock, mirror, ops, added)
    _assert_mirror_ran(mirror)


def test_resumable_requests_take_stock_path(monkeypatch, real_configs,
                                            engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops, added = _trace(5, 50, kinds=("plain", "plain", "resumable"))
    n_resumable = sum(1 for op in ops
                      if op[0] == "add" and op[1][3] == "resumable")
    assert n_resumable
    resumable = [
        op[1][0] for op in ops if op[0] == "add" and op[1][3] == "resumable"
    ]
    _replay(stock, mirror, ops, added, reported_per_turn=resumable)
    _assert_mirror_ran(mirror, expect_stock_picks=n_resumable)
    assert not mirror.mirror.count_on
    assert mirror.mirror.count_reason == "resumable request"


def test_prefix_caching_keeps_stock_routing(monkeypatch, real_configs,
                                            engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs, prefix_caching=True)
    stock = engines(stock_dp.DPScheduler, 2, cls, prefix_caching=True)
    mirror = engines(module.DPScheduler, 2, cls, prefix_caching=True)
    hasher = get_request_block_hasher(BLOCK, sha256)
    ops, added = _trace(6, 50, kinds=("plain", "shared", "shared"))
    _replay(stock, mirror, ops, added, block_hasher=hasher)
    st = mirror.mirror
    assert not st.route_on and st.route_reason == "prefix_caching=on"
    assert st.count_on and st.failed == ""
    assert mirror.stock_picks == stock.stock_picks == len(added)


@pytest.mark.parametrize("case", ["kv_transfer", "override"])
def test_unverified_configs_keep_stock_code(monkeypatch, real_configs, engines,
                                            case):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    if case == "kv_transfer":
        cls = _rank_scheduler_cls(real_configs)
        kwargs = {"kv_transfer": types.SimpleNamespace(kv_connector="x")}
        reason = "kv_transfer_config set"
    else:
        cls = _rank_scheduler_cls(real_configs, base=_OverridesAddRequest)
        kwargs = {}
        reason = "overrides Scheduler.add_request"
    stock = engines(stock_dp.DPScheduler, 2, cls, **kwargs)
    mirror = engines(module.DPScheduler, 2, cls, **kwargs)
    ops, added = _trace(7, 30)
    _replay(stock, mirror, ops, added)
    st = mirror.mirror
    assert not st.route_on and reason in st.route_reason
    assert mirror.stock_picks == len(added)
    if case == "override":
        assert not st.count_on and reason in st.count_reason


def test_check_bursts_pass_on_every_burst(monkeypatch, real_configs, engines):
    # CHECK covering every burst: each mirror pick is compared with the
    # stock picker and with a fresh query of every rank.
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=10**6)
    cls = _rank_scheduler_cls(real_configs, base=_RemoteKVRankScheduler)
    stock = engines(stock_dp.DPScheduler, 3, cls)
    mirror = engines(module.DPScheduler, 3, cls)
    ops, added = _trace(8,
                        60,
                        kinds=("plain", "grammar", "remote_kv", "kv_arrival"),
                        pauses=True)
    _replay(stock, mirror, ops, added)
    st = mirror.mirror
    assert st.failed == "" and st.route_on and st.count_on


def test_large_burst_with_full_window(monkeypatch, real_configs, engines):
    module = _mirrored_module(monkeypatch,
                              DP_SCHED_ROUTE_MIRROR_CHECK=0,
                              DP_SCHED_ROUTE_MIRROR_WINDOW=64)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops = [("add", ("w%d" % i, 1 + (i * 37) % 300, 1, "plain"))
           for i in range(400)]
    with _deadline(120):
        _replay(stock, mirror, ops, [op[1][0] for op in ops], max_drain=2000)
    _assert_mirror_ran(mirror)


# --------------------------------------------------------------------------
# Worker failures during deferred ADD acks
# --------------------------------------------------------------------------


def _fail_in_worker(head_pid):
    if os.getpid() != head_pid:
        raise RuntimeError("injected worker-side ADD failure")
    return _FailsInWorker(head_pid)


class _FailsInWorker:
    """Pickles in the DP head; unpickling it in a worker raises."""

    def __init__(self, head_pid):
        self.head_pid = head_pid

    def __reduce__(self):
        return (_fail_in_worker, (self.head_pid, ))


def test_worker_add_failure_surfaces_from_deferred_ack(monkeypatch,
                                                       real_configs, engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)

    def bad_request():
        req = _request(("bad", 16, 2, "plain"))
        req.trace_headers = {"x": _FailsInWorker(os.getpid())}
        return req

    # Stock: the worker error is raised by add_request itself.
    with pytest.raises(stock_dp.SchedulerWorkerError,
                       match="injected worker-side ADD failure"):
        stock.add(bad_request())

    # Mirror: the ack is deferred; the error is raised by the next read
    # from that rank, with a note naming the request, and is not hidden.
    with _deadline(60):
        assert mirror.add(bad_request()) == 0
        assert mirror.add(_request(("ok1", 16, 2, "plain"))) == 1
        assert mirror.add(_request(("ok2", 16, 2, "plain"))) == 0
        st = mirror.mirror
        assert list(st.outstanding[0]) == ["bad", "ok2"]
        with pytest.raises(module.SchedulerWorkerError,
                           match="injected worker-side ADD failure") as err:
            mirror.step()
    notes = " ".join(getattr(err.value, "__notes__", []))
    assert "'bad'" in notes and "rank 0" in notes
    # The failed ack was consumed; the later ack is still queued in order.
    assert list(st.outstanding[0]) == ["ok2"]


def test_worker_death_with_deferred_acks_is_raised(monkeypatch, real_configs,
                                                   engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    for engine in (stock, mirror):
        for i in range(6):
            engine.add(_request(("k%d" % i, 20, 2, "plain")))
    st = mirror.mirror
    assert list(st.outstanding[0]) == ["k0", "k2", "k4"]
    for engine in (stock, mirror):
        proc = engine.sched.processes[0]
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=10)
        assert proc.exitcode == -signal.SIGKILL
    with _deadline(60):
        # The local count needs no IPC.
        assert mirror.count() == 6
        # schedule() fails on the dead rank the same way as the stock one.
        for engine in (stock, mirror):
            with pytest.raises(BrokenPipeError):
                engine.step()
        # Reading from the dead rank (what every DPScheduler query does
        # after its send) goes through its deferred acks and raises the
        # stock pipe error naming the dead worker.
        with pytest.raises(
                RuntimeError,
                match=(r"Pipe error for rank 0: Worker process \(PID=%d\) "
                       r"terminated with exit code -9" %
                       mirror.sched.processes[0].pid),
        ):
            mirror.sched._get_result(0)
    assert not st.outstanding[0]
    assert list(st.outstanding[1]) == ["k1", "k3", "k5"]


def _failing_request(rid):
    req = _request((rid, 16, 2, "plain"))
    req.trace_headers = {"x": _FailsInWorker(os.getpid())}
    return req


def _assert_shut_down(sched, ranks):
    """The stock shutdown ran to its end for these ranks."""
    for rank in ranks:
        proc = sched.processes[rank]
        # exited on its SHUTDOWN command (exit 0), not killed by the join
        assert not proc.is_alive() and proc.exitcode == 0, proc.exitcode
    assert all(conn.closed for conn in sched.input_conns + sched.output_conns)


def test_shutdown_raises_deferred_ack_error_after_shutting_down_all_ranks(
        monkeypatch, real_configs, engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    mirror = engines(module.DPScheduler, 3, cls)
    st = mirror.mirror
    with _deadline(60):
        assert mirror.add(_failing_request("bad")) == 0
        for i in range(5):
            mirror.add(_request(("s%d" % i, 16, 2, "plain")))
        queued = [list(q) for q in st.outstanding]
        assert queued == [["bad", "s2"], ["s0", "s3"], ["s1", "s4"]]
        # The worker error is first read by shutdown().
        with pytest.raises(module.SchedulerWorkerError,
                           match="injected worker-side ADD failure") as err:
            mirror.shutdown()
    notes = " ".join(getattr(err.value, "__notes__", []))
    assert "'bad' on rank 0" in notes and "shutdown()" in notes
    assert not any(st.outstanding)
    _assert_shut_down(mirror.sched, range(3))


@pytest.mark.parametrize("worker_error", [True, False],
                         ids=["worker-error", "no-error"])
def test_shutdown_reads_deferred_acks_of_dead_rank(monkeypatch, real_configs,
                                                   engines, worker_error):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    mirror = engines(module.DPScheduler, 2, cls)
    sched, st = mirror.sched, mirror.mirror
    first = (_failing_request("bad") if worker_error else _request(
        ("bad", 16, 2, "plain")))
    with _deadline(60):
        assert mirror.add(first) == 0
        assert mirror.add(_request(("d1", 16, 2, "plain"))) == 1
        assert mirror.add(_request(("d2", 16, 2, "plain"))) == 0
        assert list(st.outstanding[0]) == ["bad", "d2"]
        # Rank 0 answers its first ADD, then its worker dies before
        # shutdown(), which skips dead ranks in the stock code.
        assert sched.output_conns[0].poll(30)
        proc = sched.processes[0]
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=10)
        if worker_error:
            # Not lost with the dead worker: raised after the shutdown.
            with pytest.raises(module.SchedulerWorkerError,
                               match="injected worker-side ADD failure"):
                mirror.shutdown()
        else:
            # As in the stock code, a dead rank alone does not raise.
            mirror.shutdown()
    _assert_shut_down(sched, [1])


# --------------------------------------------------------------------------
# Negative controls: stale or corrupted mirror state
# --------------------------------------------------------------------------


def _corrupt_after_first_add(mirror):
    st = mirror.mirror
    assert st.snap is not None
    st.snap[0][1] += 10**6  # rank 1 looks loaded to the mirror only


def test_corrupted_snapshot_without_check_diverges(monkeypatch, real_configs,
                                                   engines):
    # Shows the stock comparison detects a stale snapshot at all.
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    assert (mirror.add(_request(
        ("a", 30, 2, "plain"))) == stock.add(_request(
            ("a", 30, 2, "plain"))) == 0)
    _corrupt_after_first_add(mirror)
    assert stock.add(_request(("b", 30, 2, "plain"))) == 1
    assert mirror.add(_request(("b", 30, 2, "plain"))) == 0


def test_corrupted_snapshot_with_check_falls_back_to_stock(
        monkeypatch, real_configs, engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=1)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    assert (mirror.add(_request(
        ("a", 30, 2, "plain"))) == stock.add(_request(
            ("a", 30, 2, "plain"))) == 0)
    _corrupt_after_first_add(mirror)
    ops, added = _trace(9, 40)
    ops = [("add", ("b", 30, 2, "plain"))] + ops
    _replay(stock, mirror, ops, ["a", "b"] + added)
    st = mirror.mirror
    assert st.failed.startswith("route req=b")
    assert not st.route_on and not st.count_on


def test_stale_low_count_never_ends_engine_early(monkeypatch, real_configs,
                                                 engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops = [("add", ("l%d" % i, 50, 4, "plain")) for i in range(5)]
    for op in ops:
        assert _apply(mirror, op) == _apply(stock, op)
    mirror.mirror.live.clear()  # every request forgotten by the mirror
    assert mirror.count() == stock.count() == 5
    assert mirror.mirror.failed.startswith("count local=0 stock=5")
    _replay(stock, mirror, [], [op[1][0] for op in ops])


def test_stale_high_count_does_not_keep_idle_engine_stepping(
        monkeypatch, real_configs, engines):
    module = _mirrored_module(monkeypatch, DP_SCHED_ROUTE_MIRROR_CHECK=0)
    cls = _rank_scheduler_cls(real_configs)
    stock = engines(stock_dp.DPScheduler, 2, cls)
    mirror = engines(module.DPScheduler, 2, cls)
    ops = [("add", ("h%d" % i, 20, 2, "plain")) for i in range(4)]
    for op in ops:
        assert _apply(mirror, op) == _apply(stock, op)
    mirror.mirror.live["phantom"] = 0  # a request the workers never had
    steps = 0
    while mirror.count() > 0:
        steps += 1
        assert steps <= 10, "idle engine kept stepping on a stale count"
        mirror.step()
    while stock.count() > 0:
        stock.step()
    assert sorted(mirror.done) == sorted(stock.done) == sorted(op[1][0]
                                                               for op in ops)
    assert mirror.mirror.failed.startswith("count local=1 stock=0")
