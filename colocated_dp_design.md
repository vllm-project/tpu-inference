# Colocated-Python Data Parallelism on Pathways — Design

> **2026-05-22 update — major revision.** Empirical finding: inside a
> `colocated_python` program on Pathways, `jax.devices()` returns only the
> colocated CPU devices — TPU devices are not addressable from inside.
> This invalidates the original v1 design in which each `EngineShard` was a
> full vLLM `EngineCore` (because the EngineCore's `TPUModelRunner` /
> `model_fn` / KV cache need TPU device access).
>
> The redesign **splits engine work along the CPU/TPU line**:
> per-DP-rank Scheduler (pure CPU/numpy) lives in a `colocated_python_class`
> instance on the colocated host; per-DP-rank TPU execution (model_fn, KV
> cache, sampling) stays on the controller. See §11 below for the new
> architecture. §§2–10 are kept as historical context and are partially
> superseded.


## 1. Problem & goals

We need data parallelism (DP) for `tpu-inference` that:

1. Works on **Pathways** (single-controller / single-process; `JAX_PLATFORMS=proxy`).
2. Lets each DP engine **run independently** — no lock-step, no global padded batch.
3. Supports **both** offline (`LLM().generate()`) and online (`vllm serve`).

Neither existing approach satisfies all three:

| | SPMD DP (`DPScheduler`) | MPMD DP (PR #2700, `TPU_MULTIPROCESS_DP`) |
|---|---|---|
| Mechanism | 1 process; `DPScheduler` forks `dp_size` *scheduler* subprocesses; **one** `TPUModelRunner` runs all ranks in **one SPMD program** over a `"data"` mesh axis | vLLM spawns `dp_size` `DPEngineCoreProc` **OS processes**, each a full engine, chips isolated via libtpu env vars |
| Independence | ❌ Lock-step: every step batches `padded_max(all ranks) × dp_size`; idle ranks burn compute on padding (`tpu_runner.py:_prepare_input_metadata`) | ✅ Engines fully independent |
| Pathways | ✅ single-process | ❌ multi-process breaks single-controller (explicitly disabled: `sharding.py` raises if `VLLM_TPU_USING_PATHWAYS`) |
| Offline | ✅ | ❌ "`LLM().generate()` will hang — use `vllm serve`" |

**Goal: get MPMD's independence on Pathways' single controller** by replacing the OS-process+ZMQ transport with `jax.experimental.colocated_python` — which is JAX's official mechanism for running host Python code on remote device hosts, portable across single- and multi-controller backends. (Verified available: jax 0.10.0 exposes `colocated_python_class`, `colocated_cpu_devices`.)

## 2. The central question: where to insert the colocated-python seam

The vLLM stack:

```
LLM / AsyncLLM  →  EngineCoreClient  →  EngineCore  →  Executor  →  TPUWorker  →  TPUModelRunner  →  jitted model_fn
   (front-end:                          (Scheduler   (UniProc)                      (KV cache,
    tokenize,                            lives here)                                 input batch)
    detokenize)
```

Candidate seams, evaluated:

### ❌ Option C — wrap `ModelRunner` / `model_fn`
Lowest level. KV cache, input batch, attention metadata would cross the boundary every step. Maximally lock-step. Reject.

### ❌ Option B — wrap `Executor` / `Worker`
Controller keeps **one** `EngineCore` + **one** `Scheduler` (or today's `DPScheduler`); a "colocated DP executor" fans `execute_model` out to `dp_size` colocated workers. Smaller boundary, reuses `DPScheduler`'s load balancer.

**Fatal flaw:** `EngineCore.step()` is `schedule() → execute_model() → update_from_output()` — strictly synchronous, one batch per step. A single `step()` loop *cannot* let ranks advance at different rates. Option B removes SPMD padding but **keeps lock-step**. It does not meet goal #2.

### ✅ Option A — wrap `EngineCore` (recommended)

Each DP rank is a **complete, independent vLLM `EngineCore`** (its own stock `Scheduler`, `UniProcExecutor`, `TPUWorker`, `TPUModelRunner`), running as a `colocated_python` class instance on the CPU host colocated with that rank's TPU chips. The controller holds a thin **`ColocatedDPEngineCore`** that owns the `dp_size` shard handles and routes requests.

Why this is the right seam:

- **It is the seam vLLM itself uses for DP.** vLLM's own DP unit is the per-rank `DPEngineCoreProc`. We keep that boundary and only swap the *transport*: OS-process + ZMQ → colocated-python. The colocated `EngineShard` is morally a `DPEngineCoreProc`; our `ColocatedDPEngineCore` replaces the (DP client + DP coordinator + load balancer) trio.
- **Independence falls out for free.** Each shard has its own `step()` loop, so ranks advance at their own pace — exactly goal #2. The `TPUModelRunner` inside each shard sees `dp_size=1`, so the entire SPMD DP code path (`_prepare_input_metadata` padding to `padded_max × dp_size`, the `"data"` mesh axis) is simply *not used*. No padding waste.
- **A precedent already exists in this repo.** `DisaggEngineCore` (`core_tpu.py`) is exactly this shape: a vLLM-facing `EngineCore` subclass that owns a *list of sub-`EngineCore`s* + an orchestrator with one thread per sub-engine + an output funnel queue. We mirror its structure; `_DisaggOrchestrator` is the template for the router.
- **Offline + online unified.** Both `LLMEngine` (offline) and `AsyncLLM` (online) call the same engine-core surface (`add_request(EngineCoreRequest)` / `step()`). The colocated boundary lives *inside* our engine core, below tokenization — the front-end is untouched. Offline works naturally because the single controller *is* the process; we need no DP coordinator, which is exactly what made MPMD hang offline.

**Recommendation: Option A.** The rest of this document details it.

## 3. Proposed architecture

```
                          SINGLE CONTROLLER PROCESS (Pathways)
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  LLMEngine (offline)  /  AsyncLLM + API server (online)                    │
 │        │  add_request(EngineCoreRequest) / step()                          │
 │        ▼                                                                   │
 │  ColocatedDPEngineCore  (subclass of vllm EngineCore)                       │
 │    • partitions jax.devices() into dp_size TPU groups                       │
 │    • Router: req_id → shard  (round-robin / least-outstanding)              │
 │    • req_id → shard map (for abort/finish)                                  │
 │    • dp_size driver threads (1 per shard ⇒ colocated calls run concurrent)  │
 │    • output funnel Queue → step() drains it                                 │
 │        │                  │                  │                            │
 │   colocated-python    colocated-python    colocated-python   (uint8 blob    │
 │   class handle 0      class handle 1      class handle N      I/O, §6)      │
 └───────┼──────────────────┼──────────────────┼─────────────────────────────┘
         ▼                  ▼                  ▼
   ┌───────────┐      ┌───────────┐      ┌───────────┐
   │EngineShard│      │EngineShard│      │EngineShard│   ← real instance per
   │  host 0   │      │  host 1   │      │  host N   │     colocated CPU host
   │ EngineCore│      │ EngineCore│      │ EngineCore│
   │ +Scheduler│      │ +Scheduler│      │ +Scheduler│
   │ +busy loop│      │ +busy loop│      │ +busy loop│
   └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
     TPU group 0        TPU group 1        TPU group N   (disjoint chips)
```

Each `EngineShard` is independent: own scheduler, own KV cache (profiled on its own chips), own compiled `model_fn` over a mesh covering **only its TPU group**.

## 4. Components

### 4.1 `EngineShard` — runs on the colocated host

Wraps a vLLM `EngineCore`. Reuse vLLM's `EngineCoreProc` machinery as much as possible: it already has an input queue, output queue, and `run_busy_loop()` — we just feed those queues from colocated-python method calls instead of ZMQ sockets.

```python
class EngineShard:                       # wrapped by colocated_python_class
    def __init__(self, vllm_config, device_group_id):
        # vllm_config has device_config.slice → this shard's TPU devices,
        # data_parallel_size = 1  (so TPUModelRunner runs single-rank)
        self._core = vLLMEngineCore(vllm_config, UniProcExecutor, log_stats=...)
        self._inq, self._outq = Queue(), Queue()
        threading.Thread(target=self._busy_loop, daemon=True).start()

    def _busy_loop(self):                 # independent step loop — no lock-step
        while True:
            self._drain_inq_into_scheduler()
            if self._core.scheduler.has_requests():
                for cidx, out in self._core.step()[0].items():
                    self._outq.put((cidx, out))

    # colocated-python-exposed methods. Args carry a tiny `pin` JAX array on
    # this shard's colocated-CPU sharding (selects the host); the requests/
    # outputs ride alongside as ordinary cloudpickled pytree leaves (§6).
    # Uint8-blob fallback only if Spike S2 disproves direct object args.
    def submit(self, pin, requests):  [self._inq.put(r) for r in requests]
    def poll(self, pin):  -> outputs: return drain(self._outq)
    def control(self, pin, op, payload): # abort/reset_prefix_cache/stats/shutdown
```

Device scoping: the shard builds `TPUModelRunner` over `device_config.slice` (its TPU group only). We **reuse the existing `device_config.slice` mechanism** that `DisaggExecutor._init_executor` (`disagg_executor.py:24`) already uses to hand a device subset to a `TPUWorker`. With `data_parallel_size=1`, `ShardingConfigManager` builds a `(1,1,1,expert,tp,dcp)` mesh over just that group — ordinary single-rank execution, no `"data"` axis.

### 4.2 `ColocatedDPEngineCore` — runs on the controller

Subclass of vLLM `EngineCore` (offline / `InprocClient`), mirroring `DisaggEngineCore`:

```python
class ColocatedDPEngineCore(vLLMEngineCore):
    def __init__(self, vllm_config, executor_class, log_stats, ...):
        groups   = partition(jax.devices(), dp_size)        # disjoint TPU groups
        ShardCls = colocated_python.colocated_python_class(EngineShard)
        self._shards = []
        for g in groups:
            cpu = colocated_python.colocated_cpu_devices(g)  # the colocated host
            cfg = clone_config(vllm_config, slice=g, data_parallel_size=1)
            self._shards.append((ShardCls(cfg, g.id), cpu_sharding(cpu)))
        self._router      = Router(self._shards)             # req → shard
        self._owner       = {}                               # req_id → shard idx
        self._output_q    = queue.Queue()
        self._threads     = [Thread(target=self._drive, args=(i,)) ...]  # 1/shard

    def add_request(self, request):                          # EngineCoreRequest
        i = self._router.pick(request); self._owner[request.request_id] = i
        self._shards[i].pending.append(request)

    def _drive(self, i):                                     # one per shard
        shard, sharding = self._shards[i]
        while self.live:
            if shard.pending:  shard.submit(pack(drain(shard.pending), sharding))
            for cidx, out in unpack(shard.poll()): self._output_q.put((cidx, out))

    def step(self):                                          # vLLM contract
        cidx, out = self._output_q.get(); return {cidx: out}, True
```

Key points:
- **One driver thread per shard** — colocated-python runs calls from *different threads* concurrently (and program-order within a thread), so the `dp_size` engines genuinely run in parallel; per-shard call ordering stays correct.
- **First method call pins the instance.** A `colocated_python_class` instance is created on the host of the devices used in its *first* method call. Every shard method takes a blob arg placed on that shard's colocated-CPU sharding ⇒ first `submit`/`poll` pins `EngineShard i` to host `i`. (Class methods aren't specializable yet, so we cannot `.specialize(devices=)`; placement-via-first-call is the supported path.)
- Aggregation methods (`get_num_unfinished_requests`, `has_unfinished_requests`, stats) fan out over shards and combine — the `DPScheduler` already shows the combine logic.
- `finish_requests`/abort routed via `self._owner[req_id]`.

### 4.3 `ColocatedDPEngineCoreProc` — online parity

Mirror of `DisaggEngineCoreProc` for the `vllm serve` path. On Pathways `VLLM_ENABLE_V1_MULTIPROCESSING` is forced 0 (`tpu_platform.check_and_update_config`), so the engine core runs in the controller process regardless — the in-process `ColocatedDPEngineCore` is the primary target. **Open question (spike):** confirm exactly which client/core class `AsyncLLM` instantiates for `vllm serve` under Pathways, and whether a separate `*Proc` subclass is needed at all or `ColocatedDPEngineCore` suffices for both.

## 5. Routing & load balancing

Scheduling (token-level batching, prefix cache) lives **inside** each shard's stock `Scheduler`. The controller only does coarse request→shard assignment:

- **v1:** least-outstanding-requests, i.e. vLLM's P2C score `waiting*4 + running` (`DPLBAsyncMPClient`), tracked from each shard's reported counts. Sticky: a `req_id` stays on its shard for its lifetime.
- **later:** prefix-cache-aware routing — probe shards for cache hits (the `DPScheduler._find_best_rank_for_request` two-tier strategy), at the cost of an extra round-trip.

We do **not** use vLLM's DP coordinator or front-end LB: from vLLM's view `data_parallel_size=1`, so none is created. The controller owns balancing — simpler, and identical offline and online.

## 6. The colocated boundary — serialization

Two independent pieces of evidence say this is *not* the linchpin we feared:

**(a) Stateful Python objects on the colocated host are a first-class, supported pattern.**
JAX's own test suite verifies that non-serializable Python state (including a `TemporaryFile`!) can live in module globals on the colocated host and persist across calls; the test deliberately "poisons" the colocated_python module to prove module globals are *never* shipped/serialized during a call. The same applies, by construction, to `colocated_python_class` instance state — the wrapper class docstring states: *"The actual object will persist while the wrapper object is alive, and will be destroyed asynchronously when the wrapper object is destroyed."*

**Consequence for our design:** the `EngineShard`'s vLLM `EngineCore` — with its compiled `model_fn` references, KV cache handles, scheduler state, request dicts — lives safely on the colocated host as instance state, untouched between calls. There is no need to (re)materialize it across the boundary. This is the heaviest state in the system and it sits entirely on the right side.

**(b) Per-call arguments can almost certainly include arbitrary picklable Python objects.** `jax.experimental.colocated_python.serialization` is built on **cloudpickle**, and `colocated_python.func` flattens `(args, kwargs)` with `tree_util.tree_leaves` and runs the leaves through `_serialize`. This is the same cloudpickle path the `DPScheduler` already uses to round-trip `Request` objects across a process boundary (`dp_scheduler.py:_cloudpickle_dumps`).

**Consequence for our design:** the planned `shard.submit(reqs)` / `shard.poll()` calls likely take/return `EngineCoreRequest` / `EngineCoreOutputs` lists directly (mixed inside an args pytree alongside a tiny JAX array placed on the shard's colocated-CPU sharding so colocated_python knows *which* host to ship to). The uint8 JAX-array blob fallback from earlier drafts is no longer the default plan.

**What still needs a quick spike (S2 below):** confirm two specifics:
- Picklable non-array leaves are accepted in an args pytree (vs. all leaves needing to be arrays).
- The dispatch device is inferred from the array leaves even when most of the pytree is plain Python — so a single 1-byte JAX array on group-r's colocated CPU sharding is enough to pin the call to shard r.

If both hold, the boundary code is just `shard.submit(pin_array, requests)` and `shard.poll(pin_array) → outputs` — no blob packing.

**Net effect on the risk register:** §6 was previously listed as the #1 risk; with this evidence it drops to a low-effort confirmation spike. The remaining real risks shift up: boundary *latency* (item 3) and instance-pinning ergonomics under threading.

## 7. Configuration & registration

- New env var `TPU_COLOCATED_DP` (mirrors `TPU_MULTIPROCESS_DP`). When set with `data_parallel_size > 1`:
  - In `sharding.py`: set per-engine `data_parallel_size = 1` / `total_dp_size = 1` (so each shard runs single-rank), **and skip `update_vllm_config_for_dp_scheduler`** so `DPScheduler` is *not* installed. The controller-level `ColocatedDPEngineCore` is the only thing that knows `dp_size`.
- Selection of the engine-core class: follow the `DisaggEngineCore` precedent — `patch("vllm.v1.engine.core.EngineCore", ColocatedDPEngineCore)` (and `EngineCoreProc`), gated on the env flag, ideally centralized in a helper alongside `update_vllm_config_for_dp_scheduler` rather than per-example monkeypatching.

## 8. Reuse vs. new

| Reused unchanged | New |
|---|---|
| stock vLLM `Scheduler` (inside each shard) | `EngineShard` (colocated-host class) |
| `TPUWorker`, `TPUModelRunner`, `UniProcExecutor` | `ColocatedDPEngineCore` (+ `…Proc`) |
| `device_config.slice` device-subset mechanism (`disagg_executor.py`) | blob pack/unpack helpers |
| `ShardingConfigManager` (with `dp_size=1`) | `Router` (req→shard) |
| `_DisaggOrchestrator` thread/queue *pattern* | `TPU_COLOCATED_DP` env wiring |
| front-end (tokenizer, detokenizer, API server) — untouched | |

`DPScheduler` and the runner's SPMD-DP path stay for the non-Pathways SPMD mode; the colocated path bypasses both.

## 9. Risks & open questions — spike before building

1. **Serialization boundary (§6).** Downgraded — JAX tests + colocated_python source (cloudpickle-based `serialization.py`) show stateful instance state is first-class and picklable arg leaves are likely accepted. Spike S2 just confirms the two specifics above.
2. **`jax.devices()` scope inside a colocated instance.** Does the shard see the global pod or just its host? Determines whether explicit device-subset mesh scoping (via `device_config.slice`) is sufficient — almost certainly yes, but confirm.
3. **Boundary latency.** Every `submit`/`poll` is a host↔host hop. The in-shard busy loop (§4.1) decouples step cadence from boundary frequency, but measure the floor; batch aggressively.
4. **`vllm serve` client wiring on Pathways** (§4.3).
5. **Multi-host-per-DP-rank.** v1 scopes a DP rank to a chip subset whose colocated CPUs we treat as one host (matching MPMD PR's single-host chip isolation). DP rank spanning multiple hosts ⇒ `colocated_python` creates the instance per-host and the shard's engine must itself be multi-host — defer.
6. **`colocated_python` is experimental** — API may shift (JAX explicitly disclaims its compat policy).
7. Feature surface inside shards — structured output, spec decode, multimodal, LoRA — should work since each shard is a stock `EngineCore`, but needs validation.

## 10. Phased plan

**Phase 0 — Spikes (de-risk, ~no product code)**
- S1: On Pathways, `colocated_cpu_devices` for N TPU groups; run a `colocated_python_class` instance per group; confirm each runs a jitted matmul on *its* TPU subset concurrently; measure call latency.
- S2: Round-trip `EngineCoreRequest` / `EngineCoreOutputs` across a colocated method *as direct picklable args* (mixed with a 1-byte pin array). Confirm: (i) non-array picklable leaves are accepted; (ii) the pin array alone determines dispatch host. Fall back to the uint8 blob only if either fails.
- S3: Build a real `TPUModelRunner` mesh over a device subgroup *inside* a colocated instance; run `model_fn`.

**Phase 1 — Single shard (dp_size=1), plumbing only**
- `EngineShard` wrapping a stock `EngineCore`; `ColocatedDPEngineCore` holding one shard; `llm.generate()` end-to-end through the colocated boundary. Proves the seam.

**Phase 2 — Multi-shard DP, offline**
- Device partitioning, `dp_size` shards, `Router`, per-shard driver threads, output funnel. `llm.generate()` on Pathways with `dp_size>1`, engines provably async/independent. Per-shard internal busy loop (§4.1) included from the start.

**Phase 3 — Online serving**
- `vllm serve` path (§4.3); integrate with the vLLM front-end; resolve open question #4.

**Phase 4 — Performance & balancing**
- Boundary-latency tuning, request/output batching; prefix-cache-aware routing.

**Phase 5 — Hardening**
- Stats aggregation, abort/finish, shutdown/cleanup, error propagation across the boundary; structured output / spec decode / multimodal / LoRA validation.

---

## 11. Revised architecture (v2) — CPU/TPU split

### 11.1 The new constraint

Inside a `colocated_python` program on Pathways, `jax.devices()` returns only
the colocated CPU devices passed in — not the host's local TPU chips. The
colocated process cannot:

- build a JAX mesh over TPU devices
- `jax.device_put` an array onto TPU
- jit-compile or dispatch a program that runs on TPU
- hold a TPU-resident KV cache

It *can* do everything that's pure-CPU / numpy / Python: scheduling logic,
KV-cache-manager bookkeeping (block tables, free list, prefix index), request
queue management, input-batch numpy prep.

### 11.2 Why split: motivation for the redesign

A pure "drive everything from the controller, N meshes / N engines / N
driver threads, all in one process" design works too (and is simpler). The
reason to keep colocated_python in the loop is to **distribute the per-rank
CPU scheduling work across N hosts** so a single controller CPU isn't the
bottleneck at large `dp_size`, and to physically co-locate per-rank
host-side bookkeeping with the TPUs it manages.

### 11.3 The split

For each DP rank `i`:

| Lives on | Component | Why |
|---|---|---|
| Colocated CPU host `i` | `ColocatedScheduler[i]` = stock vLLM `Scheduler` (with its KV cache *manager* / block tables / prefix index / request queue) | Pure CPU/Python. One CPU per rank — scales horizontally. |
| Controller | `RankExecutor[i]` = `TPUModelRunner` + KV cache `jax.Array` + compiled `model_fn`, mesh over rank-i's TPU chips | Needs TPU device access. |
| Controller | Driver thread `i` | Runs the per-step protocol below, glueing the two halves together. |
| Controller | `DPEngineCore` | vLLM `EngineCore` subclass; partitions devices into N groups, owns the N (scheduler-handle, executor) pairs and N driver threads, fans `add_request` to a chosen rank, drains a funnel queue in `step()`. |

### 11.4 Per-step protocol (one driver thread, one rank)

```
loop:
    sched_out = colocated_scheduler[i].schedule(pin)        # boundary X-ing #1
    if sched_out.total_num_scheduled_tokens == 0:
        sleep_small(); continue
    model_out = rank_executor[i].execute(sched_out)         # TPU compute, controller-local
    engine_outs = colocated_scheduler[i].update_from_output( # boundary X-ing #2
        pin, sched_out, model_out)
    funnel.put((i, engine_outs))
```

Two colocated calls per step (down from "≥1 per request" in the original
design's blob-style polling). The objects crossing the boundary are vLLM
data structures (`SchedulerOutput`, `ModelRunnerOutput`, `EngineCoreOutputs`)
— msgspec/dataclass, fully picklable, a few hundred KB at most per step.

`add_request` and `abort` also cross the boundary; they go to the chosen
rank's scheduler directly.

### 11.5 Latency / overlap

Two boundary crossings per step are not free. Mitigations, ordered cheapest
first:

1. **Pipeline scheduling with TPU compute** — while step N's `model_fn` runs
   on TPU (async via `non_block=True`), fan out step N+1's `schedule()`
   call on the colocated host. The boundary cost overlaps the TPU step.
2. **Batch boundary calls** — group `add_request`s in microbatches; drain
   `update_from_output` outputs in bulk. Less applicable to the schedule
   → execute round-trip which is inherently per-step.
3. **Skip path** — if `sched_out` is small, executing locally on the
   controller (with the scheduler also on the controller for that rank)
   could be cheaper. Not worth complexity in v1.

If per-step boundary latency dominates after pipelining, the fallback is the
"pure controller, N meshes" alternative in §11.7 — no colocated_python at
all.

### 11.6 Component changes vs the v1 code already on this branch

The code under `tpu_inference/core/colocated_dp_engine.py` and the
`examples/colocated_dp/*` are a v1 implementation of the **old** design
(EngineShard owns full EngineCore). Under v2:

- `EngineShard` → `ColocatedScheduler`: drops `_engine` (no vLLMEngineCore
  inside), drops the internal busy-loop thread, exposes
  `add_request`/`schedule`/`update_from_output`/`abort`/state-query
  surfaces only.
- New: `RankExecutor` on the controller. Wraps a `TPUModelRunner` scoped to
  one DP rank's TPU chips (reuse `device_config.slice` mechanism from
  `DisaggExecutor`). Owns the KV cache `jax.Array`. Exposes `execute(sched_out)
  → ModelRunnerOutput`.
- `ColocatedDPEngineCore` → `DPEngineCore`: orchestrator now runs the
  per-step protocol in driver threads (schedule → execute → update), not a
  poll loop.
- Per-rank `vllm_config` mutation: same as before — `data_parallel_size=1`
  for each rank's executor; `ShardingConfigManager` rebuild over rank-i's
  devices.
- Registration / env wiring (`TPU_COLOCATED_DP`, sharding override,
  platform patch): unchanged.

### 11.7 Alternative considered: no colocated_python at all

"Multi-mesh single-controller DP": same `DPEngineCore` orchestrator, but
each rank's `EngineCore` lives entirely on the controller (mesh over its
chip group). No colocated boundary, no scheduler-on-host. Simpler, lower
per-step latency, mirrors `DisaggEngineCore` exactly.

Trade-off: one CPU/GIL does all N schedulers' work — may bottleneck at
large `dp_size`.

The v2 design (§§11.3–11.5) is preferred when the goal is to scale CPU
scheduling work across hosts. The alternative is preferred when boundary
latency proves problematic in practice, or `dp_size` is small enough that
one CPU is fine.

### 11.8 Phase plan (v2)

**P0 — spikes**: same as before (S2: object args across boundary;
S3: now reframed to verify the scheduler-only pattern — instantiate a vLLM
`Scheduler` inside a colocated_python_class instance, call `add_request` /
`schedule` / `update_from_output` with picklable args, confirm round-trip).
Also need a new spike S4: confirm Pathways supports N concurrent
controller-driven jit'd programs over disjoint device groups (driver-thread
concurrency).

**P1 — single rank**: ColocatedScheduler + RankExecutor for dp_size=1.
Driver thread runs the per-step protocol. Prove `LLM.generate()` works
end-to-end.

**P2 — multi-rank offline**: N (scheduler, executor) pairs + N driver
threads + router + funnel. `LLM.generate()` on Pathways with `dp_size>1`,
true independence (no lock-step).

**P3 — online serving**: same as v1.

**P4 — perf**: pipeline schedule with execute (§11.5 mitigation 1);
prefix-cache-aware routing.

**P5 — hardening**: same as v1.
