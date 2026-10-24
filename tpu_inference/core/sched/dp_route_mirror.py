# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""DP request routing without per-request IPC (env DP_SCHED_ROUTE_MIRROR=1, default off).

Installed by tpu_inference/core/sched/dp_scheduler.py at import time when DP_SCHED_ROUTE_MIRROR=1 (a wrap of the
DPScheduler class methods listed below; nothing changes when the flag is off). The per-rank scheduler worker
processes are NOT touched: they run `_scheduler_worker_process` unchanged and see the same command stream, minus
read-only queries. Only engines with a DPScheduler (data_parallel_size > 1) are affected.

The install checks source anchors in dp_scheduler.py and in the installed vLLM Scheduler and refuses to install
on a mismatch; the CHECK bursts below then compare the mirror with the stock code at runtime. HAS_REQUEST is
treated as a read-only query when present. Under DP_SCHED_ROUTING=round_robin the routing part stays off by
design (the stock round-robin picker needs no IPC), and so does the ack deferral: every add takes the stock
path (ADD_REQUEST, then a blocking wait for its ack). Only the unfinished count (3. below) still applies.

What it replaces (tpu_inference/core/sched/dp_scheduler.py):
  * _find_best_rank_for_request: for EVERY added request, 3 queries x dp_size ranks over the pipes
    (pending prefill tokens, (running, waiting), min remaining output), then
    argmin over ranks of (pending + num_tokens, running + waiting, min_remaining).
  * _route_and_forward_request: ADD_REQUEST, then a blocking wait for its ack.
  * get_num_unfinished_requests: dp_size round trips on every EngineCore.step (has_requests()).

The mirror (value-exact: same rank for every request, same count for every query):
  1. ROUTING. A burst is a run of ADD_REQUESTs with no state-changing command in between. At the first add
     of a burst every rank is queried once, with the stock commands in the stock order. After that, the
     mirror applies to the three queried quantities exactly what the worker's vLLM Scheduler.add_request does
     (Scheduler.add_request -> Scheduler._enqueue_waiting_request):
       - status not blocked -> request appended to `waiting`: pending += num_prompt_tokens, inflight += 1
       - blocked status (grammar / remote KV / streaming) -> `skipped_waiting`: the DP worker's
         GET_PENDING_PREFILL_TOKENS and GET_REQUEST_COUNTS read `waiting` only, so nothing changes
       - GET_MIN_REMAINING_OUTPUT reads `running` only, so it never changes during a burst.
     The argmin expression is the stock one verbatim (min(range(dp), key=(load, inflight, min_rem))).
     Any command other than ADD_REQUEST and the read-only queries ends the burst and drops the snapshot.
  2. ACKS. The ADD acks (cloudpickle(None), 8 bytes with the length header) stay in each rank's output pipe
     and are drained, oldest first, by the wrapped _get_result before ANY other read from that rank, or as soon
     as a rank has W outstanding (DP_SCHED_ROUTE_MIRROR_WINDOW, default 64 -> at most 512 bytes queued per rank,
     far below the minimum socket buffer, so a worker can never block on its ack write while we block on a send).
  3. UNFINISHED COUNT. A dict of live request ids: + on every ADD_REQUEST, - on every EngineCoreOutput with a
     finish_reason (and every id in EngineCoreOutputs.finished_requests) returned by update_from_output, and
     on every FinishedRequestInfo returned by finish_requests. A local 0 is always confirmed with the stock IPC
     (so a stale-low count can never stop the engine early). Paused scheduler or a resumable request -> stock.

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
line per burst. Install raises RuntimeError when an anchor does not match, so an unverified setup never runs.
"""

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
_ANCHORS_VLLM = {
    "Scheduler.add_request": ("self._enqueue_waiting_request(request)", ),
    "Scheduler._enqueue_waiting_request": (
        "if self._is_blocked_waiting_status(request.status):",
        "self.skipped_waiting.add_request(request)",
        "self.waiting.add_request(request)",
    ),
    "Scheduler.get_num_unfinished_requests": (
        "if self._pause_state == PauseState.PAUSED_ALL:",
        "- self.num_waiting_for_streaming_input",
        "return num_waiting + len(self.running)",
    ),
}

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
        self.outstanding = [0] * dp
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
    """Install the mirror on tpu_inference's DPScheduler. Raises when a source anchor does not match.

    `module` is the dp_scheduler module object (dp_scheduler.py passes itself while it is being imported).
    """
    if _G["on"]:
        log.info("[dp-route-mirror] ON: already installed")
        return
    if module is None:
        from tpu_inference.core.sched import dp_scheduler as module
    M = module
    from vllm.v1.core.sched import scheduler as VSM

    DP, SC, VS = M.DPScheduler, M.SchedulerCommand, VSM.Scheduler
    f_dp, f_vs = inspect.getsourcefile(M), inspect.getsourcefile(VSM)
    bad = []
    s_dp = _sha(f_dp) if f_dp else "?"
    s_vs = _sha(f_vs) if f_vs else "?"
    for root, table in ((M, _ANCHORS_DP), (VSM, _ANCHORS_VLLM)):
        for name, needles in table.items():
            try:
                src = inspect.getsource(_resolve(root, name))
            except (AttributeError, OSError, TypeError) as exc:
                bad.append("source unavailable: %s (%r)" % (name, exc))
                continue
            for needle in needles:
                if needle not in src:
                    bad.append("anchor missing in %s: %r" %
                               (name, needle.split("\n")[0]))
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
    if getattr(DP, "_route_mirror_installed", False):
        bad.append("DPScheduler already carries a route mirror")
    if bad:
        raise RuntimeError("[dp-route-mirror] FATAL: %s; refusing to install" %
                           "; ".join(bad))

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
    is_blocked = VS._is_blocked_waiting_status

    o_init = DP.__init__
    o_send = DP._send_command
    o_get = DP._get_result
    o_rafr = DP._route_and_forward_request
    o_gnur = DP.get_num_unfinished_requests
    o_ufo = DP.update_from_output
    o_fin = DP.finish_requests
    o_sps = DP.set_pause_state

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

    def _drain(self, st, rank):
        while st.outstanding[rank] > 0:
            st.outstanding[
                rank] -= 1  # the message is consumed even if it carries a worker error
            o_get(self, rank, ADD)

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
            reason = _route_static_reason(self)
            st.route_on = not reason
            st.route_reason = reason
            st.count_on = cfg["count"]
            st.count_reason = "" if cfg[
                "count"] else "DP_SCHED_ROUTE_MIRROR_COUNT=0"
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
                st.outstanding[rank] += 1
                st.b_deferred += 1
                if st.outstanding[rank] >= cfg["window"]:
                    _drain(self, st, rank)
            else:
                self._get_result(rank, ADD)
            # Mirror of the worker's scheduler.add_request on the three queried quantities.
            # (fault=state, TEST-ONLY positive control: forget the 8th add of every burst.)
            if not is_blocked(request.status) and not (cfg["fault"] == "state"
                                                       and st.b_n == 7):
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
        if checking or local == 0:
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

    DP.__init__ = __init__
    DP._send_command = _send_command
    DP._get_result = _get_result
    DP._route_and_forward_request = _route_and_forward_request
    DP.get_num_unfinished_requests = get_num_unfinished_requests
    DP.update_from_output = update_from_output
    DP.finish_requests = finish_requests
    DP.set_pause_state = set_pause_state
    DP._route_mirror_installed = True
    _G["on"] = True
    _G["log"] = log
    log.info(
        "[dp-route-mirror] ON: installed on %s.DPScheduler | dp_scheduler.py sha=%s | vllm scheduler.py "
        "sha=%s | anchors=%d ok | check_bursts=%d route=%d count=%d window=%d log_every=%d%s",
        M.__name__,
        s_dp[:12],
        s_vs[:12],
        sum(len(v) for v in _ANCHORS_DP.values()) +
        sum(len(v) for v in _ANCHORS_VLLM.values()),
        cfg["check"],
        cfg["route"],
        cfg["count"],
        cfg["window"],
        cfg["log_every"],
        " FAULT_INJECT=" + cfg["fault"] if cfg["fault"] else "",
    )
