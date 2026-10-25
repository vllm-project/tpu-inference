# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""DP request routing without per-request IPC (env DP_SCHED_ROUTE_MIRROR=1, default off).

Installed by tpu_inference/core/sched/dp_scheduler.py at import time when DP_SCHED_ROUTE_MIRROR=1 (a wrap of the
DPScheduler class methods listed below; nothing changes when the flag is off). The per-rank scheduler worker
processes are NOT touched: they run `_scheduler_worker_process` unchanged and see the same command stream, minus
read-only queries. Only engines with a DPScheduler (data_parallel_size > 1) are affected.

The install checks source anchors in dp_scheduler.py, and runs the installed vLLM Scheduler.add_request and
get_num_unfinished_requests on probe requests (VLLM SCHEDULER MODEL below); it refuses to install on any mismatch.
The CHECK bursts below then compare the mirror with the stock code at runtime. HAS_REQUEST is treated as a
read-only query when present. Under DP_SCHED_ROUTING=round_robin the routing part stays off by design (the stock
round-robin picker needs no IPC), and so does the ack deferral: every add takes the stock path (ADD_REQUEST, then
a blocking wait for its ack). Only the unfinished count (3. below) still applies. Routing also stays on the stock
path with prefix caching, batched prefill, a KV transfer config (the connector's add hook is not modelled), or a
worker scheduler class that overrides a modelled Scheduler method; the per-engine log line names the reason.

What it replaces (tpu_inference/core/sched/dp_scheduler.py):
  * _find_best_rank_for_request: for EVERY added request, 3 queries x dp_size ranks over the pipes
    (pending prefill tokens, (running, waiting), min remaining output), then
    argmin over ranks of (pending + num_tokens, running + waiting, min_remaining).
  * _route_and_forward_request: ADD_REQUEST, then a blocking wait for its ack.
  * get_num_unfinished_requests: dp_size round trips on every EngineCore.step (has_requests()).

VLLM SCHEDULER MODEL (the shape of the vLLM commit in .buildkite/vllm_lkg.version, ef5bcaa3876):
  Scheduler.add_request (new, non-resumable request) -> Scheduler._enqueue_waiting_request:
    - request holding KV blocks (num_computed_tokens > 0 or WAITING_FOR_REMOTE_KVS) -> `kv_holding_waiting`
    - every other request -> `waiting`, including the blocked statuses (grammar, remote KV, streaming input),
      which are only additionally recorded in the `deferred_waiting` set
  Scheduler.get_num_unfinished_requests (UNPAUSED) counts every such request once.
  The DP worker's GET_PENDING_PREFILL_TOKENS and GET_REQUEST_COUNTS read `running` and `waiting` only.
  Any other behaviour of the installed vLLM is an unsupported shape: install raises RuntimeError naming the
  installed vLLM version, its scheduler.py and every probe case that did not match.

The mirror (value-exact: same rank for every request, same count for every query):
  1. ROUTING. A burst is a run of ADD_REQUESTs with no state-changing command in between. At the first add
     of a burst every rank is queried once, with the stock commands in the stock order. After that, the
     mirror applies to the three queried quantities exactly what the worker's Scheduler.add_request does:
       - request lands in `waiting` -> pending += num_prompt_tokens, inflight += 1
       - request lands in `kv_holding_waiting` -> nothing changes (the DP worker does not read it)
       - GET_MIN_REMAINING_OUTPUT reads `running` only, so it never changes during a burst.
     The argmin expression is the stock one verbatim (min(range(dp), key=(load, inflight, min_rem))).
     Any command other than ADD_REQUEST and the read-only queries ends the burst and drops the snapshot.
  2. ACKS. The ADD acks (cloudpickle(None), one small message each) stay in each rank's output pipe and are
     drained, oldest first, by the wrapped _get_result before ANY other read from that rank, or as soon as a
     rank has W outstanding (DP_SCHED_ROUTE_MIRROR_WINDOW, default and maximum 64, which stays well inside a
     default socket send buffer, so a worker never blocks on its ack write while we block on a send). A worker
     error in place of an ack is raised by that drain, with a note naming the request. shutdown() first reads
     the outstanding acks of every rank (also of a rank whose worker already exited; the stock shutdown skips
     those), then runs the stock shutdown, and only then raises the first worker error found in them.
  3. UNFINISHED COUNT. A dict of live request ids: + on every ADD_REQUEST, - on every EngineCoreOutput with a
     finish_reason (and every id in EngineCoreOutputs.finished_requests) returned by update_from_output, and
     on every FinishedRequestInfo returned by finish_requests. A local 0 is always confirmed with the stock IPC
     (so a stale-low count can never stop the engine early), and so is the count after a step that scheduled
     no token (so a stale-high count cannot keep an idle engine stepping). Paused scheduler or a resumable
     request -> stock.

CHECK (DP_SCHED_ROUTE_MIRROR_CHECK=N, default 3): during the first N bursts every add ALSO runs the stock picker
(`_pick_rank_for_request`, i.e. the full stock IPC query) plus a fresh state query of every rank, and every
get_num_unfinished_requests call until burst N+1 begins also runs the stock IPC. The STOCK result is the one
used. Any mismatch logs `[dp-route-mirror] ... check=FAIL` and turns the whole mirror off for the rest of the
process (stock methods from then on; outstanding acks are still drained in order).

Other knobs (all default to the stock-equivalent behaviour): DP_SCHED_ROUTE_MIRROR_ROUTE=1,
DP_SCHED_ROUTE_MIRROR_COUNT=1, DP_SCHED_ROUTE_MIRROR_WINDOW=64 (0 = wait for every ack like stock),
DP_SCHED_ROUTE_MIRROR_LOG_EVERY=1 (log every Nth burst; bursts <= CHECK+1 and failures are always logged).
DP_SCHED_ROUTE_MIRROR_FAULT_INJECT=route|state|count is TEST-ONLY (a positive control for CHECK); leave it unset.

Log lines: `[dp-route-mirror] ON: installed ...` (anchors), `[dp-route-mirror] engine dp=... route=ON count=ON ...`
(per DPScheduler instance), then one `[dp-route-mirror] burst=K n=... mode=mirror dist=[...] ... check=PASS(...)`
line per burst. Install raises RuntimeError when a check does not pass, so an unverified setup never runs.
"""

import collections
import functools
import hashlib
import inspect
import time

from tpu_inference import envs

_READ_ONLY_NAMES = (
    "GET_PENDING_PREFILL_TOKENS",
    "GET_REQUEST_COUNTS",
    "GET_MIN_REMAINING_OUTPUT",
    "PROBE_COMPUTED_BLOCKS",
    "GET_NUM_UNFINISHED_REQUESTS",
    "HAS_FINISHED_REQUESTS",
    "GET_PAUSE_STATE",
    "GET_TOKEN_COUNT",
)
# Read-only commands that newer revisions add; used when present.
_READ_ONLY_OPTIONAL_NAMES = ("HAS_REQUEST", )

# (object getter, required substrings). Any missing anchor refuses the install and names what broke.
_ANCHORS_DP = {
    "DPScheduler._route_and_forward_request": (
        "rank = self._pick_rank_for_request(request)",
        "self.assigned_dp_rank[request.request_id] = rank",
        "self._send_command(rank, SchedulerCommand.ADD_REQUEST, request)",
        "self._get_result(rank, SchedulerCommand.ADD_REQUEST)",
    ),
    "DPScheduler._find_best_rank_for_request": (
        "self._send_command(rank,\n                               SchedulerCommand.GET_PENDING_PREFILL_TOKENS)",
        "self._send_command(rank, SchedulerCommand.GET_REQUEST_COUNTS)",
        "self._send_command(rank, SchedulerCommand.GET_MIN_REMAINING_OUTPUT)",
        "loads[rank] = pending + max(0, num_tokens - cached)",
        "inflight[rank] = running + waiting",
        "key=lambda r: (loads[r], inflight[r], min_remaining[r]))",
    ),
    "DPScheduler.get_num_unfinished_requests": (
        "total = len(self._pending_new_requests)",
        "SchedulerCommand.GET_NUM_UNFINISHED_REQUESTS",
    ),
    "DPScheduler.finish_requests":
    ("FinishedRequestInfo(req_id, client_index)", ),
    "_scheduler_worker_process": (
        "scheduler.add_request(request)\n                    _send_result(None)  # Signal completion",
        "for req in scheduler.waiting:\n                        pending += req.num_prompt_tokens",
        "waiting = len(scheduler.waiting)",
        "min(r.max_tokens - len(r.output_token_ids)\n                                for r in scheduler.running)",
        "result = scheduler.get_num_unfinished_requests()",
    ),
}
# Largest accepted DP_SCHED_ROUTE_MIRROR_WINDOW: deferred acks are small messages queued in the worker's output
# socket, and 64 of them stay well inside a default AF_UNIX send buffer, so the worker never blocks on an ack write.
_MAX_WINDOW = 64

# vLLM Scheduler methods whose behaviour the mirror models (see VLLM SCHEDULER MODEL). A worker scheduler class that
# overrides one of them is not covered by the install probe, so that engine keeps the stock DPScheduler code.
_MODELLED_SCHEDULER_METHODS = (
    "add_request",
    "_enqueue_waiting_request",
    "_holds_kv_blocks",
    "get_request_counts",
    "get_num_unfinished_requests",
)
_PROBE_STATUSES = (
    "WAITING",
    "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR",
    "WAITING_FOR_REMOTE_KVS",
    "WAITING_FOR_STREAMING_REQ",
    "PREEMPTED",
)
_SUPPORTED_VLLM = (
    "add_request -> _enqueue_waiting_request: KV-holding requests (computed tokens or WAITING_FOR_REMOTE_KVS) "
    "to kv_holding_waiting, all others (blocked statuses included) to waiting; get_num_unfinished_requests "
    "counts each such request once (vLLM ef5bcaa3876, .buildkite/vllm_lkg.version)"
)


def _counted_in_waiting(request, remote_kvs_status):
    """Whether the worker's vLLM Scheduler.add_request puts this new request in `waiting`.

    `waiting` (with `running`) is all the DP worker's GET_PENDING_PREFILL_TOKENS and GET_REQUEST_COUNTS read.
    A request holding KV blocks goes to `kv_holding_waiting` instead; a blocked status alone does not move it.
    install() checks this against the installed vLLM (_verify_vllm_scheduler).
    """
    return request.num_computed_tokens <= 0 and request.status != remote_kvs_status


def _verify_vllm_scheduler(VS):
    """Run the installed Scheduler.add_request / get_num_unfinished_requests on probe requests.

    Returns the list of mismatches with VLLM SCHEDULER MODEL (empty when the installed vLLM matches it). The probe
    scheduler has only the attributes the modelled shape uses, so a vLLM whose add_request reads or writes other
    state is reported as unsupported instead of being guessed at.
    """
    try:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.interface import PauseState
        from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                                      create_request_queue)
        from vllm.v1.request import Request, RequestStatus
    except ImportError as exc:
        return ["probe imports failed: %r" % (exc, )]

    missing = [
        n for n in _MODELLED_SCHEDULER_METHODS
        if not callable(getattr(VS, n, None))
    ]
    missing += [
        "RequestStatus." + n for n in _PROBE_STATUSES
        if not hasattr(RequestStatus, n)
    ]
    if missing:
        return ["vLLM Scheduler lacks %s" % ", ".join(missing)]
    probe = object.__new__(VS)
    vars(probe).update(
        requests={},
        waiting=create_request_queue(SchedulingPolicy.FCFS),
        kv_holding_waiting=create_request_queue(SchedulingPolicy.FCFS),
        deferred_waiting=set(),
        running=[],
        num_waiting_for_streaming_input=0,
        _pause_state=PauseState.UNPAUSED,
        connector=None,
        log_stats=False,
        spec_decode_metrics_level="none",
        num_spec_tokens=0,
    )
    attrs = set(vars(probe))
    remote_kvs = RequestStatus.WAITING_FOR_REMOTE_KVS
    bad = []
    added = 0
    for status in _PROBE_STATUSES:
        for computed in (0, 16):
            case = "status=%s num_computed_tokens=%d" % (status, computed)
            req = Request("dp-route-mirror-probe-%d" % added, [0] * 32,
                          SamplingParams(max_tokens=1), None)
            req.status = getattr(RequestStatus, status)
            req.num_computed_tokens = computed
            try:
                VS.add_request(probe, req)
                unfinished = VS.get_num_unfinished_requests(probe)
            except Exception as exc:  # noqa: BLE001 - reported as unsupported
                return bad + ["add_request(%s) raised %r" % (case, exc)]
            added += 1
            in_waiting = any(r is req for r in probe.waiting)
            if in_waiting != _counted_in_waiting(req, remote_kvs):
                bad.append("add_request(%s): in waiting=%s, modelled %s" %
                           (case, in_waiting, not in_waiting))
            if probe.running:
                bad.append("add_request(%s) put a request in running" % case)
            if unfinished != added:
                bad.append(
                    "get_num_unfinished_requests=%d after %d adds (%s)" %
                    (unfinished, added, case))
    if set(vars(probe)) != attrs:
        bad.append("add_request set unmodelled scheduler state %s" %
                   sorted(set(vars(probe)) - attrs))
    return bad


_G = {"on": False, "log": None}
# what the DP worker's GET_MIN_REMAINING_OUTPUT returns for an empty `running`
_IDLE_MINREM = 2**31 - 1


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _resolve(root, dotted):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class _St:
    """Per-DPScheduler-instance mirror state (created by the wrapped __init__)."""

    def __init__(self, dp, cfg):
        self.dp = dp
        self.cfg = cfg
        self.route_on = False
        self.route_reason = ""
        self.count_on = False
        self.count_reason = ""
        self.failed = ""
        self.paused = False
        # per rank: ids of the requests whose ADD ack is still in the pipe, oldest first
        self.outstanding = [collections.deque() for _ in range(dp)]
        # the last update_from_output was for a step that scheduled no token
        self.idle_step = False
        self.snap = (
            # [pending[], inflight[], minrem[]] while a mirror burst is open
            None)
        self.mirror_send = False
        self.in_burst = False
        self.bursts = 0
        self.live = {}  # request_id -> rank
        # bursts whose first add found a rank with pending, waiting or running requests
        self.busy_bursts = 0
        self._burst_reset()
        self._unf_reset()

    def _burst_reset(self):
        self.b_n = 0
        self.b_dist = [0] * self.dp
        self.b_mode = ""
        self.b_queries = 0
        self.b_deferred = 0
        self.b_t = 0.0
        self.b_ck_route = 0
        self.b_ck_state = 0
        self.b_fail = ""
        self.b_base = "-"

    def _unf_reset(self):
        self.u_local = 0
        self.u_stock = 0
        self.u_zero = 0
        self.u_ck = 0

    def checking(self):
        return self.bursts <= self.cfg["check"]


def install(log, module=None):
    """Install the mirror on tpu_inference's DPScheduler.

    Raises RuntimeError when a dp_scheduler.py source anchor does not match, when the installed vLLM Scheduler
    does not behave as VLLM SCHEDULER MODEL describes, or when DP_SCHED_ROUTE_MIRROR_WINDOW is out of range.
    `module` is the dp_scheduler module object (dp_scheduler.py passes itself while it is being imported).
    """
    if module is None:
        from tpu_inference.core.sched import dp_scheduler as module
    M = module
    import vllm
    from vllm.v1.core.sched import scheduler as VSM
    from vllm.v1.request import RequestStatus

    DP, SC, VS = M.DPScheduler, M.SchedulerCommand, VSM.Scheduler
    if getattr(DP, "_route_mirror_installed", False):
        log.info("[dp-route-mirror] ON: already installed on %s.DPScheduler",
                 M.__name__)
        return
    f_dp, f_vs = inspect.getsourcefile(M), inspect.getsourcefile(VSM)
    bad = []
    s_dp = _sha(f_dp) if f_dp else "?"
    s_vs = _sha(f_vs) if f_vs else "?"
    vllm_version = getattr(vllm, "__version__", "?")
    for name, needles in _ANCHORS_DP.items():
        try:
            src = inspect.getsource(_resolve(M, name))
        except (AttributeError, OSError, TypeError) as exc:
            bad.append("source unavailable: %s (%r)" % (name, exc))
            continue
        for needle in needles:
            if needle not in src:
                bad.append("anchor missing in %s: %r" %
                           (name, needle.split("\n")[0]))
    vllm_bad = _verify_vllm_scheduler(VS)
    if vllm_bad:
        bad.append(
            "installed vLLM %s Scheduler (%s, sha256 %s) does not match the modelled shape [%s]: %s"
            % (vllm_version, f_vs, s_vs[:12], _SUPPORTED_VLLM,
               "; ".join(vllm_bad)))
    if envs.DP_SCHED_ROUTE_MIRROR_WINDOW > _MAX_WINDOW:
        bad.append("DP_SCHED_ROUTE_MIRROR_WINDOW=%d is above the maximum %d" %
                   (envs.DP_SCHED_ROUTE_MIRROR_WINDOW, _MAX_WINDOW))
    # The three methods this module REPLACES must be the module's own code, not an earlier wrapper.
    for name in (
            "_route_and_forward_request",
            "_find_best_rank_for_request",
            "get_num_unfinished_requests",
    ):
        fn = DP.__dict__.get(name)
        code = getattr(fn, "__code__", None)
        if (fn is None or hasattr(fn, "__wrapped__") or code is None
                or code.co_filename != f_dp):
            bad.append("DPScheduler.%s is not the original method" % name)
    ro = []
    for n in _READ_ONLY_NAMES:
        if not hasattr(SC, n):
            bad.append("SchedulerCommand.%s missing" % n)
        else:
            ro.append(getattr(SC, n))
    for n in _READ_ONLY_OPTIONAL_NAMES:
        if hasattr(SC, n):
            ro.append(getattr(SC, n))
    WORKER_ERROR = getattr(M, "SchedulerWorkerError", None)
    if not (isinstance(WORKER_ERROR, type)
            and issubclass(WORKER_ERROR, Exception)):
        bad.append("SchedulerWorkerError missing")
    if bad:
        raise RuntimeError(
            "[dp-route-mirror] FATAL: %s; refusing to install (unset DP_SCHED_ROUTE_MIRROR for the stock "
            "DPScheduler)" % "; ".join(bad))

    cfg = dict(
        check=max(0, envs.DP_SCHED_ROUTE_MIRROR_CHECK),
        route=envs.DP_SCHED_ROUTE_MIRROR_ROUTE != 0,
        count=envs.DP_SCHED_ROUTE_MIRROR_COUNT != 0,
        window=max(0, envs.DP_SCHED_ROUTE_MIRROR_WINDOW),
        log_every=max(0, envs.DP_SCHED_ROUTE_MIRROR_LOG_EVERY),
        fault=(envs.DP_SCHED_ROUTE_MIRROR_FAULT_INJECT or "").strip(),
    )
    READ_ONLY = frozenset(ro)
    ADD = SC.ADD_REQUEST
    GPPT, GRC, GMRO = (
        SC.GET_PENDING_PREFILL_TOKENS,
        SC.GET_REQUEST_COUNTS,
        SC.GET_MIN_REMAINING_OUTPUT,
    )
    LEAST_LOADED = DP.RoutingPolicy.LEAST_LOADED
    UNPAUSED = M.PauseState.UNPAUSED
    REMOTE_KVS = RequestStatus.WAITING_FOR_REMOTE_KVS

    o_init = DP.__init__
    o_send = DP._send_command
    o_get = DP._get_result
    o_rafr = DP._route_and_forward_request
    o_gnur = DP.get_num_unfinished_requests
    o_ufo = DP.update_from_output
    o_fin = DP.finish_requests
    o_sps = DP.set_pause_state
    o_shut = DP.shutdown

    # ------------------------------------------------------------------ helpers
    def _route_static_reason(self):
        if not cfg["route"]:
            return "DP_SCHED_ROUTE_MIRROR_ROUTE=0"
        if self._routing_policy != LEAST_LOADED:
            return "routing=%s" % getattr(self._routing_policy, "value",
                                          self._routing_policy)
        if self.vllm_config.cache_config.enable_prefix_caching:
            return "prefix_caching=on"
        if self._batch_prefills:
            return "batch_prefill=on"
        if getattr(self.vllm_config, "kv_transfer_config", None) is not None:
            return "kv_transfer_config set"
        return ""

    def _model_reason(self):
        """Why the workers' scheduler class is not the one install() verified ("" when it is)."""
        cls = getattr(self.vllm_config.scheduler_config,
                      "_original_scheduler_cls", None)
        if not (isinstance(cls, type) and issubclass(cls, VS)):
            return "scheduler class %r is not a vLLM Scheduler" % (cls, )
        for name in _MODELLED_SCHEDULER_METHODS:
            if getattr(cls, name) is not getattr(VS, name):
                return "%s overrides Scheduler.%s" % (cls.__name__, name)
        return ""

    def _fail(self, st, what):
        st.failed = what
        st.b_fail = what
        st.route_on = False
        st.count_on = False
        st.snap = None
        log.warning(
            "[dp-route-mirror] check=FAIL burst=%d %s -> route mirror OFF for this process; stock DPScheduler code from now on",
            st.bursts,
            what,
        )

    def _drain(self, st, rank, errors=None):
        """Read the deferred ADD acks of `rank`, oldest first.

        A worker error in place of an ack is raised with a note naming the request. With `errors` (shutdown) a
        worker error is appended to it instead and the later acks are still read; pipe errors are raised.
        """
        pending = st.outstanding[rank]
        while pending:
            # the message is consumed even if it carries a worker error
            rid = pending.popleft()
            try:
                o_get(self, rank, ADD)
            except Exception as exc:
                keep_reading = errors is not None and isinstance(
                    exc, WORKER_ERROR)
                if keep_reading:
                    where = "read by shutdown(), raised once every rank was shut down"
                else:
                    where = "%d later acks still queued on that rank" % len(
                        pending)
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "[dp-route-mirror] raised by the deferred ADD_REQUEST ack of request %r on rank %d (%s)"
                        % (rid, rank, where))
                if not keep_reading:
                    raise
                errors.append(exc)

    def _query(self):
        """The stock least_loaded query (prefix caching off): same commands, same order."""
        dp = self.dp_size
        for rank in range(dp):
            self._send_command(rank, GPPT)
            self._send_command(rank, GRC)
            self._send_command(rank, GMRO)
        pending, inflight, minrem = [0] * dp, [0] * dp, [0] * dp
        for rank in range(dp):
            pending[rank] = self._get_result(rank, GPPT)
            running, waiting = self._get_result(rank, GRC)
            inflight[rank] = running + waiting
            minrem[rank] = self._get_result(rank, GMRO)
        return [pending, inflight, minrem]

    def _decide(dp, snap, request):
        pending, inflight, minrem = snap
        num_tokens = request.num_tokens
        cached = 0  # prefix caching is off whenever the mirror routes
        loads = [pending[r] + max(0, num_tokens - cached) for r in range(dp)]
        return min(range(dp), key=lambda r: (loads[r], inflight[r], minrem[r]))

    def _begin_burst(st):
        if not st.in_burst:
            st.in_burst = True
            st.bursts += 1
            st._burst_reset()

    def _end_burst(self, st):
        st.in_burst = False
        st.snap = None
        k = st.bursts
        every = cfg["log_every"]
        if not (st.b_fail or k <= cfg["check"] + 1 or
                (every and k % every == 0)):
            st._unf_reset()
            return
        if st.b_fail:
            ck = "FAIL(%s)" % st.b_fail
        elif st.b_ck_route:
            ck = "PASS(route=%d,state=%d)" % (st.b_ck_route, st.b_ck_state)
        else:
            ck = "-"
        uck = "PASS(%d)" % st.u_ck if st.u_ck else "-"
        log.info(
            "[dp-route-mirror] burst=%d n=%d mode=%s base=%s dist=%s queries=%d acks_deferred=%d t_add_ms=%.2f check=%s "
            "| unfinished since last burst: local=%d stock=%d zero_confirm=%d check=%s | live=%d%s",
            k,
            st.b_n,
            st.b_mode or "-",
            st.b_base,
            st.b_dist,
            st.b_queries,
            st.b_deferred,
            1e3 * st.b_t,
            ck,
            st.u_local,
            st.u_stock,
            st.u_zero,
            uck,
            len(st.live),
            " OFF=" + st.failed if st.failed else "",
        )
        st._unf_reset()

    # ------------------------------------------------------------------ wrappers
    @functools.wraps(o_init)
    def __init__(self, *a, **k):
        o_init(self, *a, **k)
        self._route_mirror = None
        try:
            st = _St(int(self.dp_size), cfg)
            model_reason = _model_reason(self)
            reason = model_reason or _route_static_reason(self)
            st.route_on = not reason
            st.route_reason = reason
            st.count_on = cfg["count"] and not model_reason
            st.count_reason = ("DP_SCHED_ROUTE_MIRROR_COUNT=0"
                               if not cfg["count"] else model_reason)
            self._route_mirror = st
            sc = self.vllm_config.scheduler_config
            log.info(
                "[dp-route-mirror] engine dp=%d policy=%s prefix_cache=%s batch_prefill=%s async=%s sched=%s -> "
                "route=%s count=%s acks=%s check_bursts=%d",
                st.dp,
                getattr(self._routing_policy, "value", self._routing_policy),
                self.vllm_config.cache_config.enable_prefix_caching,
                self._batch_prefills,
                getattr(sc, "async_scheduling", "?"),
                getattr(getattr(sc, "_original_scheduler_cls", None),
                        "__name__", "?"),
                "ON" if st.route_on else "OFF(%s)" % st.route_reason,
                "ON" if st.count_on else "OFF(%s)" % st.count_reason,
                (("deferred(window=%d)" %
                  cfg["window"]) if cfg["window"] else "per-add(stock)"),
                cfg["check"],
            )
        except Exception as exc:  # noqa: BLE001 - never break engine construction; stock path
            self._route_mirror = None
            log.warning(
                "[dp-route-mirror] engine state init failed (%r) -> stock DPScheduler code",
                exc,
            )

    @functools.wraps(o_send)
    def _send_command(self, rank, command, data=None):
        st = getattr(self, "_route_mirror", None)
        if st is not None:
            if command is ADD:
                _begin_burst(st)
                if not st.mirror_send:
                    st.snap = None  # an ADD the mirror did not account for
                if getattr(data, "resumable", False):
                    st.count_on = False
                    st.count_reason = "resumable request"
                rid = getattr(data, "request_id", None)
                if rid is not None:
                    st.live[rid] = rank
            elif command not in READ_ONLY and st.in_burst:
                _end_burst(self, st)
        return o_send(self, rank, command, data)

    @functools.wraps(o_get)
    def _get_result(self, rank, command=None):
        st = getattr(self, "_route_mirror", None)
        if st is not None and st.outstanding[rank]:
            _drain(self, st, rank)
        return o_get(self, rank, command)

    @functools.wraps(o_rafr)
    def _route_and_forward_request(self, request):
        st = getattr(self, "_route_mirror", None)
        if st is None:
            return o_rafr(self, request)
        t0 = time.perf_counter()
        try:
            if st.route_on and not st.failed:
                reason = _route_static_reason(self)
                if reason:
                    st.route_on, st.route_reason = False, reason
            if not st.route_on or st.failed or getattr(request, "resumable",
                                                       False):
                _begin_burst(st)
                st.b_mode = "stock:" + (st.failed and "failed"
                                        or st.route_reason or "resumable")
                o_rafr(self, request)
                rank = self.assigned_dp_rank.get(request.request_id)
                if rank is not None:
                    st.b_dist[rank] += 1
                st.b_n += 1
                return None
            _begin_burst(st)
            st.b_mode = "mirror"
            if st.snap is None:
                st.snap = _query(self)
                st.b_queries += 1
                idle = not any(st.snap[0]) and not any(st.snap[1]) and all(
                    m == _IDLE_MINREM for m in st.snap[2])
                st.b_base = "idle" if idle else "busy"
                if not idle:
                    st.busy_bursts += 1
            dp = self.dp_size
            rank = _decide(dp, st.snap, request)
            counted = _counted_in_waiting(request, REMOTE_KVS)
            if cfg["fault"] == "route" and st.b_n == 5:
                rank = (rank + 1) % dp  # TEST-ONLY positive control for CHECK
            if st.checking():
                stock_rank = self._pick_rank_for_request(request)
                fresh = _query(self)
                st.b_queries += 2
                if stock_rank != rank or fresh != st.snap:
                    _fail(
                        self,
                        st,
                        "route req=%s idx=%d mirror_rank=%d stock_rank=%d mirror_state=%s fresh_state=%s"
                        % (
                            request.request_id,
                            st.b_n,
                            rank,
                            stock_rank,
                            st.snap,
                            fresh,
                        ),
                    )
                    st.b_mode = "stock:failed"
                    o_rafr(
                        self, request
                        # stock query + ADD + wait; same state -> stock_rank again
                    )
                    st.b_dist[self.assigned_dp_rank[request.request_id]] += 1
                    st.b_n += 1
                    return None
                st.b_ck_route += 1
                st.b_ck_state += 1
                rank = stock_rank
            self.assigned_dp_rank[request.request_id] = rank
            st.mirror_send = True
            try:
                self._send_command(rank, ADD, request)
            finally:
                st.mirror_send = False
            if cfg["window"]:
                st.outstanding[rank].append(request.request_id)
                st.b_deferred += 1
                if len(st.outstanding[rank]) >= cfg["window"]:
                    _drain(self, st, rank)
            else:
                self._get_result(rank, ADD)
            # Mirror of the worker's scheduler.add_request on the three queried quantities.
            # (fault=state, TEST-ONLY positive control: forget the 8th add of every burst.)
            if counted and not (cfg["fault"] == "state" and st.b_n == 7):
                st.snap[0][rank] += request.num_prompt_tokens
                st.snap[1][rank] += 1
            st.b_dist[rank] += 1
            st.b_n += 1
            return None
        finally:
            st.b_t += time.perf_counter() - t0

    @functools.wraps(o_gnur)
    def get_num_unfinished_requests(self):
        st = getattr(self, "_route_mirror", None)
        if st is None or not st.count_on or st.failed or st.paused:
            if st is not None:
                st.u_stock += 1
            return o_gnur(self)
        local = len(self._pending_new_requests) + len(st.live)
        if cfg["fault"] == "count" and st.u_local + st.u_ck == 3:
            local += 1  # TEST-ONLY positive control for CHECK
        checking = st.checking()
        if checking or local == 0 or st.idle_step:
            stock = o_gnur(self)
            st.u_stock += 1
            if stock != local:
                _fail(
                    self,
                    st,
                    "count local=%d stock=%d live=%d" %
                    (local, stock, len(st.live)),
                )
                return stock
            if checking:
                st.u_ck += 1
            else:
                st.u_zero += 1
            return stock
        st.u_local += 1
        return local

    @functools.wraps(o_ufo)
    def update_from_output(self, scheduler_output, model_runner_output):
        out = o_ufo(self, scheduler_output, model_runner_output)
        st = getattr(self, "_route_mirror", None)
        if st is not None:
            # unknown -> treated as idle: the next count is confirmed with the stock IPC
            st.idle_step = getattr(scheduler_output,
                                   "total_num_scheduled_tokens", 0) == 0
        if st is not None and st.live:
            live = st.live
            for eco in out.values():
                for o in getattr(eco, "outputs", None) or ():
                    if o.finish_reason is not None:
                        live.pop(o.request_id, None)
                fr = getattr(eco, "finished_requests", None)
                if fr:
                    for rid in fr:
                        live.pop(rid, None)
        return out

    @functools.wraps(o_fin)
    def finish_requests(self, request_ids, finished_status):
        out = o_fin(self, request_ids, finished_status)
        st = getattr(self, "_route_mirror", None)
        if st is not None and out:
            for info in out:
                st.live.pop(info.request_id, None)
        return out

    @functools.wraps(o_sps)
    def set_pause_state(self, pause_state):
        o_sps(self, pause_state)
        st = getattr(self, "_route_mirror", None)
        if st is not None:
            st.paused = pause_state != UNPAUSED

    @functools.wraps(o_shut)
    def shutdown(self):
        st = getattr(self, "_route_mirror", None)
        if st is None or not any(st.outstanding):
            return o_shut(self)
        # Read every rank's outstanding acks before the stock shutdown, including those of a rank whose worker
        # already exited (they stay readable in the pipe; the stock shutdown skips dead ranks). A worker error found
        # there must not stop the stock shutdown half-way (SHUTDOWN acks unread, processes not joined, pipes left
        # open), so it is raised after the stock shutdown has run.
        errors = []
        for rank in range(len(st.outstanding)):
            try:
                _drain(self, st, rank, errors)
            except (RuntimeError, OSError) as exc:
                # a pipe error, handled like the stock shutdown handles one on a SHUTDOWN ack
                st.outstanding[rank].clear()
                log.warning(
                    "[dp-route-mirror] rank %d: deferred ADD_REQUEST acks not read at shutdown: %s",
                    rank, exc)
        for exc in errors:
            log.error("[dp-route-mirror] shutdown: %s | %s", exc,
                      " ".join(getattr(exc, "__notes__", ())))
        o_shut(self)
        if errors:
            raise errors[0]

    DP.__init__ = __init__
    DP._send_command = _send_command
    DP._get_result = _get_result
    DP._route_and_forward_request = _route_and_forward_request
    DP.get_num_unfinished_requests = get_num_unfinished_requests
    DP.update_from_output = update_from_output
    DP.finish_requests = finish_requests
    DP.set_pause_state = set_pause_state
    DP.shutdown = shutdown
    DP._route_mirror_installed = True
    _G["on"] = True
    _G["log"] = log
    log.info(
        "[dp-route-mirror] ON: installed on %s.DPScheduler | dp_scheduler.py sha=%s | vllm %s scheduler.py "
        "sha=%s | anchors=%d ok | vllm scheduler probe ok (%d cases) | check_bursts=%d route=%d count=%d "
        "window=%d log_every=%d%s",
        M.__name__,
        s_dp[:12],
        vllm_version,
        s_vs[:12],
        sum(len(v) for v in _ANCHORS_DP.values()),
        2 * len(_PROBE_STATUSES),
        cfg["check"],
        cfg["route"],
        cfg["count"],
        cfg["window"],
        cfg["log_every"],
        " FAULT_INJECT=" + cfg["fault"] if cfg["fault"] else "",
    )
