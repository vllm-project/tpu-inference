# Migrating TPU CI to Kubernetes: does it match bare metal?

## Verdict

**Yes. Setup is an order of magnitude faster, execution is slightly faster, and
the chips are used better. Every per-push step bare metal runs, we run.**

All figures are medians: five consecutive full-suite runs on the shipped
configuration under one pinned image (builds
[255](https://buildkite.com/tpu-commons/kube-dev/builds/255),
[257](https://buildkite.com/tpu-commons/kube-dev/builds/257),
[258](https://buildkite.com/tpu-commons/kube-dev/builds/258),
[259](https://buildkite.com/tpu-commons/kube-dev/builds/259),
[260](https://buildkite.com/tpu-commons/kube-dev/builds/260)) against three
`NIGHTLY`-unset bare-metal builds
([24341](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/24341),
[24337](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/24337),
[24333](https://buildkite.com/tpu-commons/tpu-inference-ci/builds/24333)).

| question | measure | Kubernetes | bare metal | |
|---|---|---|---|---|
| Do the tests pass? | 16 steps, five runs | 16/16 | passing | level |
| Step setup | container start + mount, vs docker pull + rsync | **0.1m** | 1.5-2.3m | **~20x faster** |
| Test execution | 15 matched steps, medians | **169.4m** | 176.5m | **0.96x** |
| Chip efficiency | overhead on chip-minutes held | **12-13%** | 38.4% | **better** |
| Can it run without bare metal? | self-built compilation cache | 28.0m | 27.8m seeded | level |
| Suite wall clock | makespan | ~85m / 16 steps | 80-81m / 17-19 steps | comparable |

### Per step

Median of five runs, with the spread. Single runs vary enough to mislead —
part2 alone spans 78.4-80.0m — so no figure here rests on one observation.

| step | kube median | range | bare metal | ratio |
|---|---|---|---|---|
| JAX unit tests part2 | 79.1m | 78.4-80.0 | 79.6m | 0.99x |
| E2E speculative decoding | 27.8m | 27.0-28.2 | 29.6m | 0.94x |
| JAX unit tests part1 | 15.4m | 15.2-15.8 | 14.1m | 1.09x |
| E2E MLPerf JAX models | 7.4m | 7.2-7.7 | 8.1m | 0.91x |
| E2E MLPerf JAX + vLLM | 6.6m | 6.5-7.0 | 7.3m | 0.90x |
| Accuracy Qwen2.5-VL-7B | 6.4m | 6.2-7.0 | 6.4m | 1.00x |
| lora e2e multi chip | 3.9m | 3.7-4.0 | 6.5m | 0.60x |
| Runai streamer Torchax Ray | 3.7m | 3.5-3.9 | 6.3m | 0.59x |
| E2E disagg single host | 3.6m | 3.6-3.9 | 6.1m | 0.59x |
| E2E disagg multi host | 3.5m | 3.3-4.4 | 5.4m | 0.65x |
| Runai streamer JAX Uniproc | 3.0m | 2.6-3.0 | 3.2m | 0.94x |
| Runai streamer Torchax Uniproc | 2.8m | 2.5-2.8 | 3.4m | 0.82x |
| E2E MPMD data parallelism | 2.3m | 2.2-2.4 | 5.3m | 0.43x |
| lora unit tests multi chip | 2.1m | 2.0-2.4 | 4.8m | 0.44x |
| lora unit tests single chip | 1.8m | 1.5-1.9 | 3.1m | 0.58x |
| **sum of medians** | **169.4m** | | **176.5m** | **0.96x** |

Only `part1` is materially slower. The two long steps that dominate the suite -
part2 and speculative decoding - are at parity or better, and the short
multi-chip steps are roughly twice as fast.

### Scope

Bare metal's per-push run schedules 41 tpu6e jobs and executes 17-19; the rest
are `NIGHTLY`-gated, as they are here. We run 16 of them. What we do not run:

| | |
|---|---|
| `JAX unit tests combine and report` | we do run it - see coverage below |
| `TPU Test Notification` | not a test |

`JAX unit tests - kernels` and `- collective kernels` are gated on both sides,
by `NIGHTLY` or by `RUN_KERNEL_TESTS` / `RUN_KERNEL_COLLECTIVES_TESTS`, so they
are conditional rather than missing.

Multi-host disaggregation was the last real gap and is now covered. Bare metal
runs it as eight docker containers on one 8-chip VM - "multi-host" meaning
eight TPU processes, not eight machines - and a pod holding all 8 chips is the
same environment with the container boundaries removed. It has passed in five
consecutive suites at a median of 3.5m against bare metal's 5.4m. The launcher's
JobSet path remains unexercised, because nothing here needs a slice spanning
nodes.

### What each side is charged for

**Setup.** Bare metal's wall time already contains its checkout, docker pull and
cache rsync, because the agent runs on the TPU VM: part2 reaches first test
output 2.3m into a 78.9m job, speculative decoding 1.5m into 28.8m. The
Kubernetes equivalent - container start and the gcsfuse mount - is **0.08m
median, 0.34m worst** across pods that landed on an existing node. Image
streaming is on and the mount overlaps container start, where bare metal pulls
and then rsyncs in sequence.

**Node creation is not setup.** Kueue admits a workload before the node exists,
so the gap between admission and first output is 2.0-3.8m whenever
cluster-autoscaler has to build one, and 0.1m when it does not. The launcher
says which: `pod not scheduled yet: Unschedulable`. Charging that against bare
metal's docker pull charges us for building hardware bare metal never stops
paying for.

Nodes being built mid-suite is intentional, not an accident of a small fleet:
the pools scale to zero (or near it) precisely so that every build exercises
the autoscaler, and it works. Even the worst case - a pod whose node is
created from scratch - reaches first output in 2.0-3.8m, on par with the
1.5-2.3m a *warm* bare-metal host spends on its docker pull and rsync. The
fair comparison for a from-scratch node is a fresh bare-metal host, whose
models and caches exist in GCS but not yet on its disk: it would pay the full
image pull, the full cache rsync and cold model downloads before its first
test, several times what our cold path costs.

**Queue is not symmetric.** On Kubernetes the wait for capacity happens inside
the Buildkite job; on bare metal it happens before the job starts and never
appears in its duration. Both belong in makespan.

### Coverage and artifacts

Bare metal's per-push suite uploads two artifacts, the coverage shards, and
merges them in a CPU step. That works here, by a different route.

`artifact_paths` cannot: it collects from the agent's working directory, and the
agent is in a different pod from the test. So the CI image carries the
buildkite-agent CLI and the test uploads its own artifacts - the pod has
`BUILDKITE_AGENT_ACCESS_TOKEN`, `_BUILD_ID` and `_JOB_ID`, which is all
`artifact upload` needs, and the result is attached to the step's Buildkite job
exactly as an agent-side upload would be.

The merge step installs `coverage` with pip rather than the pex `pipeline_jax`
uses: that pex release declares python <3.13 and the launcher image runs 3.14,
so it cannot bootstrap. pex exists there to avoid installing into a long-lived
VM's interpreter, which is not a concern for a pod discarded after the step.

## What it took

Three changes carry nearly all of it, and only one is about Kubernetes:

1. **`VLLM_XLA_CHECK_RECOMPILATION=1`.** Bare metal sets it on every step and
   Kubernetes never did. Without it JAX persists nothing small or quick to
   compile, so a self-built cache plateaus at 57.7m on speculative decoding and
   stays there. With it, 28.0m - level with a cache bare metal filled.
2. **A file cache large enough to hold a checkpoint, with an eviction
   threshold.** 84.6m of part2 went on refetching shards that could not fit.
   Both halves matter: `fileCacheCapacity: -1` removes the threshold rather
   than lifting it, and cost 26.6m where explicit capacities cost 9.1m.
3. **Test ordering.** `test_gemma4.py` cycled four ~30GB checkpoints
   round-robin, each fetched four times with the other three in between.
   Grouping them took loading from 23.8m to 8.8m. **Bare metal pays this too** -
   its page cache thrashes on the same cycle. We found it only because our
   configuration made it expensive enough to chase.

Both caches ship as PersistentVolumeClaims over GCS FUSE in the cluster's own
region, writable, with the flag set on every step. Writable is the point: the
cache fills itself, and what a test compiles survives its pod - which is what
makes the self-seeding numbers above possible. The `clone` backend measured
marginally cheaper in chip-minutes and ships anyway as the alternative it is
not: it writes nothing back, so its efficiency is borrowed from whatever filled
its golden disk.

Sizing follows three rules:

* the pod's `gke-gcsfuse-cache` volume is RAM-backed and sized per machine type
  by the launcher, since host memory ranges from 176 GB on `ct6e-standard-1t`
  to 1440 GB on `ct6e-standard-8t`;
* `fileCacheCapacity` is explicit on both mounts and fixed, because a
  PersistentVolume is one object per cluster and every profile binds the same
  claim - so it has to hold on the smallest shape;
* the two capacities sum to less than the volume, or the tmpfs reaches its
  `sizeLimit` before either mount reaches its own eviction threshold.

The shipped figures are 65Gi of models and 8Gi of compilation cache in a volume
of half the node's memory, leaving ~57 GiB of a 176 GB machine to the tests.
Past ~56Gi the model cache stops mattering - 73Gi gives 9.5m of loading against
56Gi's 8.8m - so the working set fits and further capacity is headroom.

## How we got there

What follows is the climb: the measurements that led to the configuration
above, in roughly the order they were made.

### The metric

A TPU chip is unavailable to anyone else from the moment a job takes it until
that job lets go. What matters is therefore **chip-minutes held**, not how long
a suite takes on the wall clock, and not per-step duration.

Two consequences:

**Steps are weighted by chips.** An eight-chip step that runs for ten minutes
costs 80 chip-minutes; a single-chip step of the same length costs 10. Five of
the thirteen steps here are eight-chip, so unweighted totals badly understate
what the multi-chip steps cost.

**Everything a job does while holding a chip counts**, testing or not. On bare
metal the Buildkite agent runs *on the TPU VM*, so the chips are held through
checkout, the docker pull, the test, and the cache rsync back to GCS. On
Kubernetes the chip is held from the moment the pod is placed on the node until
the job completes.

Wall clock is deliberately not the metric. Build [200](https://buildkite.com/tpu-commons/kube-dev/builds/200) did 95 minutes of work and
finished in 29; build [199](https://buildkite.com/tpu-commons/kube-dev/builds/199) did 85 minutes and took 36, because it waited 15
minutes longer for free chips. Queueing is a capacity question, not an
efficiency one.

### The compilation cache: Kubernetes can seed its own

A cache that Kubernetes builds itself, with no bare-metal seeding, on
speculative decoding:

| | cold | warm | second warm |
|---|---|---|---|
| without the flag | 161.5m | 57.8m | 57.7m |
| with `VLLM_XLA_CHECK_RECOMPILATION=1` | 133.0m | **28.0m** | - |
| reference: cache seeded by bare metal | - | 27.8m | - |

Without the flag it plateaus. Builds
[216](https://buildkite.com/tpu-commons/kube-dev/builds/216) and
[217](https://buildkite.com/tpu-commons/kube-dev/builds/217) land within 0.1m
of each other, so no number of passes closes the gap - the entries are skipped
deterministically. With it, a self-built cache reaches parity with one bare
metal filled.

The flag's documented job is to fail a test that recompiles at runtime. Its
second effect is what matters here: `CompilationManager` only lowers
`jax_persistent_cache_min_compile_time_secs` and `min_entry_size_bytes` to -1
when it is set, and without that JAX keeps its defaults and never writes
anything small or quick to compile.

The cold run is the price: 133.0m against 27.8m warm, paid once per cache
generation.

### The model cache: capacity, then ordering

part2 is the only step with a large model working set, and it was 1.40x bare
metal. Two independent causes, contributing about equally:

| | execution | checkpoint loading |
|---|---|---|
| 20Gi cache, round-robin | 111.8m | 84.6m |
| 56Gi cache, round-robin | 79.1m | 23.8m |
| 56Gi cache, grouped | 71.6m | 8.8m |
| 73Gi cache, grouped, on the shipped PVC path | 73.3m | 9.5m |
| 65Gi cache, grouped, shipped, in a full suite | **71.8m** | **9.1m** |
| bare metal | 79.6m | - |

The last row is the configuration that ships. It also bounds the curve: 73Gi
buys nothing over 56Gi, so the working set fits at either and further capacity
is headroom rather than speed.

**Capacity.** `cache-file-for-range-read` ingests the whole object on a partial
read, and gcsfuse will not cache an object that does not fit the remaining
capacity. part2 reads gemma-4 checkpoints a few layers at a time, so at 20Gi a
~30GB shard was refetched every time while evicting the small models that would
have fit.

**Ordering.** Stacked `parametrize` varies the topmost decorator fastest, so
four checkpoints cycled round-robin and each was fetched four times. Grouping
them made one checkpoint resident at a time. This costs bare metal too.

Two mechanisms worth remembering:

* `fileCacheCapacity` is the threshold gcsfuse evicts against. Setting it to
  `-1` does not mean "use everything" in any useful sense - it means there is
  no threshold, so the volume fills and writes fail. Build
  [241](https://buildkite.com/tpu-commons/kube-dev/builds/241) spent 26.6m on
  loading that way against build 234's 8.8m in a *smaller* volume.
* The capacity is nominal unless the pod declares a `gke-gcsfuse-cache` volume
  at least as large. Without one the sidecar falls back to 5GiB of ephemeral
  storage; build [239](https://buildkite.com/tpu-commons/kube-dev/builds/239)
  spent 113.6m loading, worse than having no tuning at all.

GKE's managed gcsfuse profiles would have replaced the hand-derived capacities
with StorageClasses that size the cache from the node and the bucket. Measured
on the same image and ordering, build
[240](https://buildkite.com/tpu-commons/kube-dev/builds/240) ran part2 in
115.9m with 91.3m of loading, against 71.6m and 8.8m: a profile does not set
the readahead, parallel-download and chunk-size options these mounts are tuned
with, and it gates the pod on a bucket scan that authenticates as the GKE
service agent rather than the workload identity. The capacities stay explicit.

### Where bare metal's 38% goes

Summed across the 13 steps, bare metal spends 1.0 minute on checkout, 16.4 on
the docker pull and 6.1 pushing the cache back — 23.4 minutes of wall time, but
**114 chip-minutes** once weighted, because the eight-chip hosts pull the same
multi-gigabyte image while holding eight chips each.

Kubernetes does not pay this. The image is the pod, and GKE image streaming
means the container starts before the image has finished transferring. The
entire fixed cost drops to between 13 and 49 chip-minutes depending on what
gets mounted.

That single difference is most of the efficiency gain, and it is structural
rather than something we tuned.

### Where Kubernetes' overhead goes

**Mounting GCS FUSE costs ~36 chip-minutes** over a plain local disk (49 against
13). That is the metadata prefetch walking 24,441 cache entries at mount. It
scales with cache size, so partitioning the namespace by topology would reduce
it — entries are already distinct per shape, so splitting them loses no sharing.

**Pulling the cache onto an empty disk costs 133 chip-minutes** (`rsync`), which
is why it is the only same-region configuration worse than bare metal. Starting
from a clone and pulling only the delta costs 26.

**Cross-region latency was the largest single effect at the time.** Measured
from a pod:

| operation | us-central1 | same region |
|---|---|---|
| cache miss | 502 ms | **36 ms** |
| cold read | 513 ms | 1.7 ms |
| stat, cached | 2.9 ms | 2.8 ms |
| read, cached | 1.8 ms | 1.7 ms |

A compile-heavy step asks whether an entry exists far more often than it reads
one, so the miss cost dominates. `fuse` holds 474 chip-minutes cross-region and
238 in-region — same bucket contents, same image, same tests.

**The compilation cache itself is worth 190 chip-minutes** — `nocache` holds
503 against `clone`'s 313 on the same bucket. Nothing else measured comes
close.

### So the cache lives in the cluster's region

The cross-region numbers made the bucket's region the single largest lever, so
the shipped configuration puts a cache in each cluster's own region. On the two
platforms that is very different work.

**On Kubernetes the cache is a mount, and mounts are per-cluster.** A GCS FUSE
PersistentVolume holds the bucket name; the pod refers to a claim by a name that
is identical in every region. Adding a region means adding an entry to the
`worker_clusters` map — the same `for_each` that already creates the cluster,
its node pools and its queues creates the bucket, the PV and the PVC alongside
them. No manifest changes, no pipeline changes, no step knows which bucket it
is using. Because the claim is resolved at mount time in whichever cluster the
workload landed in, this also survives MultiKueue dispatching a job to a
different region than the one it was submitted from.

**On bare metal the cache is a path on a VM's disk, and the bucket is a
literal.** `run_in_docker.sh` hardcodes `GCS_CACHE_BASE`, and each agent keeps
its own copy under `/mnt/disks/persist`. Per-region would mean per-agent
configuration, and the model cache in particular is purely local to each VM and
never shared — so a new region starts cold on every host and has no mechanism to
be seeded. What looks like one cache is really N independent local caches with
a bucket loosely behind them.

The difference is where the region binding lives: infrastructure that is already
generated per cluster, against application configuration replicated per host.

It is also cheaper, structurally, because storage cost scales with different
things on the two platforms:

**Bare metal pays per instance, for provisioned capacity.** Every agent needs a
persistent disk big enough for the compilation cache and every model any test on
that host might load, plus headroom, because a disk that fills is an outage
rather than a slowdown. That is `instances × (cache + models + buffer)`, billed
on the size allocated whether or not it is used.

**A bucket pays per region, for consumed capacity.** GCS bills what is stored,
so the buffer disappears: there is nothing to size and nothing to run out of.
That is `regions × cache`, and regions are a much smaller number than instances
and grow far more slowly.

For scale, the compilation cache namespace measured 26 GB. On bare metal that
number is multiplied by the fleet and rounded up; on GCS it is multiplied by the
number of regions and not rounded at all.

**The clone variant inherits bare metal's cost shape**, which is worth noting
because it is otherwise the fastest option. Each pod provisions a 50 GiB disk
from the golden, so a full build allocates up to eleven of them concurrently —
transient, but provisioned and billed as capacity, on top of the golden itself
and whatever the bucket already holds. `fuse` allocates nothing per pod.

**Same-region also removes egress billing**, separately from the latency it
fixed. Reading a us-central1 bucket from southamerica-west1 is inter-region
egress on every miss and every fetch; reading a bucket in the cluster's own
region is not.

### Downloading models instead of mounting them

`clone-nomodel` is the most efficient cross-region configuration at 0.76×,
because it skips the cross-region model mount entirely. Its steps completed
normally — identical test counts to every other run, 233 passed and 148 skipped
on part1, 42 passed on the lora unit tests, 11 on speculative decoding — and it
logged zero HTTP 429 responses.

Rate limiting is still real, on narrower evidence. One step, once: `JAX unit
tests part2` in build [160](https://buildkite.com/tpu-commons/kube-dev/builds/160), with no model cache, hit
`2500 api requests per 5 minutes` and failed. It resolves far more models than
any other step and is excluded from every comparison here, so this set cannot
say whether downloading is safe at full-suite scale.

With a same-region model mount the question does not arise, and the mount costs
nothing measurable against downloading. The reason to prefer it is that it
keeps a third-party request budget off the critical path — not that we measured
that budget being exceeded.

### Scheduling: profiles, borrowing and autoscaling

The fleet is **10 single-chip nodes plus one eight-chip node — 18 chips**. Two
profiles use it:

| profile | chips | topology | nodes | nominal quota |
|---|---|---|---|---|
| `v6e-1-1x1` | 1 | 1x1 | 0–10, min 2 | 10 chips |
| `v6e-8-2x4` | 8 | 2x4 | 0–1 | 8 chips |

Both sit in one Kueue cohort, so either can borrow the other's quota and have it
reclaimed. Eight of the thirteen steps are single-chip and five are eight-chip.

#### Borrowing was measured directly, then deliberately abandoned

Before the suite existed, dedicated builds tested borrowing with `sleep` jobs
against the original quotas — v6e-1 nominal **2**, v6e-8 nominal **8**, so six
single-chip jobs could only run by borrowing four chips from the other shape.

| what | build | result |
|---|---|---|
| scale up from zero | [123](https://buildkite.com/tpu-commons/kube-dev/builds/123), [125](https://buildkite.com/tpu-commons/kube-dev/builds/125), [126](https://buildkite.com/tpu-commons/kube-dev/builds/126), [127](https://buildkite.com/tpu-commons/kube-dev/builds/127) | first chip **2.2–3.6 min** after submit |
| borrow 4 chips beyond quota | [123](https://buildkite.com/tpu-commons/kube-dev/builds/123) | all 6 admitted within **0.5 min** of each other |
| reclaim after borrowers release | [124](https://buildkite.com/tpu-commons/kube-dev/builds/124) | **4.6 min** to hand 8 chips to the other shape |
| reclaim under preemption | [128](https://buildkite.com/tpu-commons/kube-dev/builds/128) | **5.0 min** after the borrowers ended |
| two builds competing | [126](https://buildkite.com/tpu-commons/kube-dev/builds/126), [127](https://buildkite.com/tpu-commons/kube-dev/builds/127) | admission spread grows to **11.2** and **17.3 min** |

Borrowing itself works and is fast. **What costs is swapping a flavor.** In
[build 124](https://buildkite.com/tpu-commons/kube-dev/builds/124) an eight-chip job was submitted at 00:40:17 while six
single-chip jobs held the cohort; those released at 00:46:26 and the eight-chip
job did not start until 00:51:03. Its total wait was 10.8 minutes, of which 4.6
came after the chips were already free — the time to scale ten single-chip
nodes down and one eight-chip node up. [Build 128](https://buildkite.com/tpu-commons/kube-dev/builds/128) waited 28.3 minutes
for the same swap under preemption.

The pools are physically separate hardware. Borrowed capacity cannot be handed
back by reassigning it; the nodes holding it have to be destroyed and different
ones built.

**That is why the current configuration does not borrow.** `v6e-1-1x1` now has
`nominal_nodes = 10`, its own ten chips, with nominal equal to max. The tfvars
records the reason: in [build 153](https://buildkite.com/tpu-commons/kube-dev/builds/153), eight single-chip steps were
admitted within six seconds of each other and every one paid a 4–13 minute cold
start, because they had all been queued behind the eight-chip lane and were
released at once. With nine concurrent single-chip steps per suite there was
nothing to gain from reaching into the other shape's capacity.

#### What the two profiles cost now

Averaged per step over the nine complete suite builds, under the
no-borrowing configuration:

| profile | queue | node scale-up | provisioning | running tests | waited for a node |
|---|---|---|---|---|---|
| `v6e-1-1x1` | **0.4 min** | 2.6 min | 1.6 min | 13.5 min | 59/72 |
| `v6e-8-2x4` | **12.5 min** | 1.3 min | 1.2 min | 4.2 min | 22/45 |

Single-chip steps are admitted almost immediately because the lane owns enough
chips for all of them. The eight-chip steps wait 12.5 minutes for a different
reason than before: **there is one eight-chip node and five steps that need it**,
so they serialise. Their combined runtime is essentially their wall-clock span —
15.6 minutes of work in a 16.2 minute window for `clone` — where the
single-chip steps overlap freely.

That queue is mostly capacity rather than contention with the other shape -
though not purely. Later builds log `Preempted to accommodate a workload due
to reclamation within the cohort` and `insufficient unused quota for
google.com/tpu in flavor v6e`, so reclamation does occur: thirteen steps
wanting 48 chip-equivalents against a cohort of 18 oversubscribe it. A second
eight-chip node would remove most of the queue. Whether it is worth one is a separate
question: those five steps total about 15 node-minutes per suite.

#### Nodes are created for most steps

82% of single-chip steps (59 of 72) waited for a node to be created, against
49% of eight-chip steps. Scale-up costs 2.6 minutes per single-chip step.

This is the cost of scaling to zero, and paying it on most steps is a choice:
it keeps the fleet from holding idle chips between builds, and it means the
autoscaler path is exercised dozens of times per suite rather than trusted. It
is charged before any chip is held, which is why it sits outside the
utilisation figures. It is real for time-to-result, and it is the reason
`min_nodes = 2` exists on the single-chip pool — the first two steps of a
build skip it.

#### Fleet occupancy during a build

Chip-minutes integrated over each same-region build, with the build window
measured from the first pod scheduled to the last job completing:

| config | window | mean chips held | chip-minutes |
|---|---|---|---|
| `clone` | 35.4 min | 5.5 of 18 | 196 |
| `clone-rsync` | 30.0 min | 7.2 of 18 | 216 |
| `fuse` | 28.7 min | 8.3 of 18 | 239 |
| `rsync` | 32.5 min | 10.3 of 18 | 334 |

The integrated totals agree with the phase sums to within 1%, which is the
check that the reconstruction is sound. Instantaneous peaks are not reported:
the launcher's completion message lags actual pod termination, so a departing
pod can appear to overlap an arriving one.

Mean occupancy is not a quality measure here — `rsync` holds the most chips
because it wastes the most, not because it packs best. What it does show is
that no configuration comes close to saturating the fleet during a single
suite; the constraint on wall clock is the serialised eight-chip steps and the
27-minute critical path, not total capacity.

#### The autoscaler will take a node out from under a running test

Job pods are evictable by default. `OPTIMIZE_UTILIZATION` consolidates an
underused node roughly ten minutes after it drains, and the last long step of a
build is a lone pod on an otherwise idle node — exactly the shape it removes.

In build [178](https://buildkite.com/tpu-commons/kube-dev/builds/178), `JAX unit tests part2` was still printing passing tests at
05:25:39 and its job was gone five seconds later: 65 minutes into a 180-minute
budget, with everything else finished at 05:05. Build [175](https://buildkite.com/tpu-commons/kube-dev/builds/175) lost the same step the
same way. It presents as a job that vanished, not as an eviction, which is why
it took three builds to notice.

The fix is `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"` on the
workload pods. `OPTIMIZE_UTILIZATION` is kept rather than switched to
`BALANCED`: the two pools are physically separate, so the eight-chip node must
scale down before single-chip nodes can scale up, and slower scale-down starves
the other shape.

#### Two effects this does not quantify

Both favour Kubernetes and neither is in the tables.

**Bare-metal VMs hold their chips continuously.** A TPU VM that exists holds its
accelerators whether or not a test is running, so its real utilisation over a
day is worse than the 38% per-suite overhead suggests. The Kubernetes pools
scale to zero on the eight-chip shape and to two nodes on the single-chip one.

**Borrowing has no bare-metal equivalent**, though we do not currently use it:
an idle eight-chip host cannot serve single-chip work on either platform, but
Kueue can at least be asked, and the cost of asking is measured above.

Sizing either needs bare-metal fleet uptime against test time, which we do not
have.

## What changes when bare metal goes away

**Age-based retention starts deleting live entries.** The bucket has a four-day
lifecycle rule, which is safe today because bare metal's persistent disk is the
real cache and GCS is only distribution. When pods are ephemeral, GCS *is* the
cache — and nothing refreshes an object's timestamp when it is read, because a
cache hit is a read. An entry used every day still expires on day four.

**Version bumps become a thundering herd.** A JAX or vLLM bump empties the
namespace. Today bare metal absorbs the first cold run; afterwards ten pods
would each compile the same modules simultaneously. The cold run is now
measured rather than estimated - 133.0m against 27.8m warm on speculative
decoding - so the cost of a bump is roughly one suite at five times its usual
length, once, per namespace.

**There is no second copy.** Bare metal's VM disks are an accidental backup. If
the bucket is emptied, the fleet runs at `nocache` efficiency until it refills.

## Open questions

- **Cache failures are silent.** Missing, stale, empty or wrong-path all
  produce no error, only a slower run and a chip held longer. An assertion at
  pod start that the cache directory exists and is non-empty would turn a slow
  week into a failed build.
- **What does the push cost?** The rsync variants push from a sidecar after the
  workload exits, which is after the launcher stops collecting logs, so it is
  absent from every figure above.
- **Does topology partitioning pay?** It would shrink the 36 chip-minute
  prefetch. Unmeasured.
- **How is the regional bucket kept current?** It is a manual snapshot today.
  Whatever replaces bare metal as producer has to keep it fresh, per region.

## Reproducing

Today, with the comparison finished and the losing backends removed:

```
bk build create --pipeline tpu-commons/kube-dev --branch <branch> \
  --env TEST_TYPE=jax \
  --env CACHE_BACKEND=<clone|pvc> \
  --env WORKLOAD_IMAGE=<pinned image>
```

`TEST_TYPE=part2` or `spec` runs one long step on its own, which is how the
storage work was iterated without paying for the other thirteen. Setting
`CACHE_NAMESPACE` to an unused name gives a cold compilation cache without
disturbing the real one - that is how the self-seeding pair was measured.

The runs in the table below used backends that no longer exist
(`fuse`, `rsync`, `clone-rsync`, `clone-nomodel`, `nocache`, `none`, `inline`,
`pvc-profile`). They were deleted once the question they answered was settled;
the builds remain readable.

```
# what those runs looked like at the time
bk build create --pipeline tpu-commons/kube-dev --branch <branch> \
  --env TEST_TYPE=jax --env SKIP_PART2=1 \
  --env CACHE_BACKEND=<one of the above> \
  --env GCS_CACHE_BUCKET=<bucket> \
  --env WORKLOAD_IMAGE=<pinned image>
```

Pin the image. Without it each run builds its own, the HLO changes, the cache
hit rate moves with it, and the storage backend cannot be isolated.

Every configuration's steps were checked for identical pytest counts before its
timings were used, so a step that aborted early cannot masquerade as a fast
one.

Phases come from the launcher's own log timestamps, so nothing instruments the
test: `submitting Job` → `admitted` is queue, → `ContainerCreating` is node
scale-up, → first workload output is provisioning, → `job/… completed` is the
test. Queue and scale-up are excluded from the tables; they are capacity, not
efficiency.

| build | configuration |
|---|---|
| [188](https://buildkite.com/tpu-commons/kube-dev/builds/188)–[193](https://buildkite.com/tpu-commons/kube-dev/builds/193) | us-central1: fuse, clone, clone-rsync, rsync, clone-nomodel, nocache |
| [199](https://buildkite.com/tpu-commons/kube-dev/builds/199)–[202](https://buildkite.com/tpu-commons/kube-dev/builds/202) | same region: clone, fuse, clone-rsync, rsync |
| [177](https://buildkite.com/tpu-commons/kube-dev/builds/177), [198](https://buildkite.com/tpu-commons/kube-dev/builds/198) | per-operation latency, both buckets |

## Appendix: per-step wall time

Minutes of wall time per step, not weighted by chips. The tables above measure
chip-minutes because that is what utilisation means; these are the underlying
durations, for anyone asking how long a given step actually took.

Kubernetes figures are `provisioning + execution`. Queue and node scale-up are
excluded, as above, and are given separately at the end.

### Same-region bucket

| step | chips | bare metal | `clone` | `clone-rsync` | `fuse` | `rsync` |
|---|---|---|---|---|---|---|
| E2E speculative decoding | 1 | 28.3 | 26.4 | 26.5 | 28.7 | 27.8 |
| JAX unit tests part1 | 1 | 14.0 | 14.3 | 14.7 | 14.9 | 17.4 |
| E2E MLPerf \| JAX models | 1 | 8.1 | 7.8 | 8.3 | 8.0 | 10.7 |
| E2E MLPerf \| JAX + vLLM models | 1 | 7.4 | 7.0 | 7.5 | 7.3 | 10.0 |
| Accuracy \| Qwen2.5-VL-7B-Instruct | 1 | 7.2 | 5.9 | 6.1 | 8.1 | 7.9 |
| lora e2e \| multi chip | 8 | 6.3 | 4.4 | 4.7 | 5.2 | 7.2 |
| Runai streamer \| Torchax RayDistributedExecutor | 8 | 6.2 | 4.3 | 5.2 | 3.7 | 7.2 |
| E2E \| Single host DCN P/D disaggregation | 8 | 5.7 | 2.3 | 2.7 | 6.2 | 4.7 |
| E2E \| MPMD data parallelism | 8 | 4.8 | 2.4 | 2.7 | 3.1 | 5.8 |
| lora unit tests \| multi chip | 8 | 4.6 | 2.2 | 2.5 | 2.1 | 5.1 |
| Runai streamer \| JAX UniProcExecutor | 1 | 4.0 | 3.4 | 3.8 | 2.7 | 6.1 |
| Runai streamer \| Torchax UniProcExecutor | 1 | 3.4 | 2.7 | 2.9 | 3.1 | 6.0 |
| lora unit tests \| single chip | 1 | 2.8 | 2.2 | 2.5 | 2.2 | 5.0 |
| **total** | | **102.8** | **85.4** | **89.9** | **95.3** | **120.9** |

Bare metal's per-step figure is checkout + docker pull + test run + cache push;
summed those are 1.0, 16.4, 79.4 and 6.1 minutes.

| config | provisioning | execution | total |
|---|---|---|---|
| `clone` | 3.9 | 81.4 | 85.4 |
| `clone-rsync` | 7.6 | 82.2 | 89.9 |
| `fuse` | 7.4 | 87.9 | 95.3 |
| `rsync` | 36.9 | 84.0 | 120.9 |

Execution is within 8% across all four. Almost the whole difference between
them is provisioning: mounting FUSE, or copying the cache onto a disk.

### us-central1 bucket

| step | chips | bare metal | `clone-nomodel` | `clone` | `fuse` | `nocache` | `rsync` |
|---|---|---|---|---|---|---|---|
| E2E speculative decoding | 1 | 28.3 | 30.7 | 35.4 | 59.0 | 160.4 | 40.9 |
| JAX unit tests part1 | 1 | 14.0 | 16.8 | 26.3 | 39.4 | 28.5 | 31.1 |
| E2E MLPerf \| JAX models | 1 | 8.1 | 10.2 | 12.2 | 17.1 | 18.2 | 17.8 |
| E2E MLPerf \| JAX + vLLM models | 1 | 7.4 | 8.9 | 27.2 | 17.6 | 17.6 | 17.4 |
| Accuracy \| Qwen2.5-VL-7B-Instruct | 1 | 7.2 | 6.6 | 8.6 | 22.9 | 11.5 | 13.2 |
| lora e2e \| multi chip | 8 | 6.3 | 5.1 | 7.1 | 11.5 | 10.3 | 11.6 |
| Runai streamer \| Torchax RayDistributedExecutor | 8 | 6.2 | 4.7 | 5.1 | 5.7 | 6.0 | 11.2 |
| E2E \| Single host DCN P/D disaggregation | 8 | 5.7 | 2.4 | 3.3 | 7.7 | 3.4 | 8.4 |
| E2E \| MPMD data parallelism | 8 | 4.8 | 3.7 | 5.2 | 6.7 | 7.9 | 10.3 |
| lora unit tests \| multi chip | 8 | 4.6 | 2.0 | 3.3 | 5.4 | 3.8 | 8.7 |
| Runai streamer \| JAX UniProcExecutor | 1 | 4.0 | 4.5 | 4.8 | 7.1 | 6.8 | 10.0 |
| Runai streamer \| Torchax UniProcExecutor | 1 | 3.4 | 3.4 | 3.6 | 7.1 | 5.5 | 10.0 |
| lora unit tests \| single chip | 1 | 2.8 | 2.0 | 3.5 | 7.1 | 3.5 | 9.3 |
| **total** | | **102.8** | **101.0** | **145.7** | **214.4** | **283.4** | **199.9** |

### Queue and node scale-up

Excluded from every figure above because they measure capacity rather than
efficiency, and vary by a factor of three between runs of the same
configuration. Recorded here so the exclusion is auditable.

| build | config | queue | node scale-up | steps that waited for a node |
|---|---|---|---|---|
| [199](https://buildkite.com/tpu-commons/kube-dev/builds/199) | `clone` same-region | 47.8 | 55.7 | 9/13 |
| [201](https://buildkite.com/tpu-commons/kube-dev/builds/201) | `clone-rsync` same-region | 51.1 | 33.5 | 10/13 |
| [200](https://buildkite.com/tpu-commons/kube-dev/builds/200) | `fuse` same-region | 32.8 | 14.5 | 9/13 |
| [202](https://buildkite.com/tpu-commons/kube-dev/builds/202) | `rsync` same-region | 59.5 | 26.9 | 10/13 |
| [192](https://buildkite.com/tpu-commons/kube-dev/builds/192) | `clone-nomodel` us-central1 | 54.0 | 33.1 | 10/13 |
| [189](https://buildkite.com/tpu-commons/kube-dev/builds/189) | `clone` us-central1 | 64.9 | 26.8 | 8/13 |
| [188](https://buildkite.com/tpu-commons/kube-dev/builds/188) | `fuse` us-central1 | 98.8 | 16.1 | 9/13 |
| [193](https://buildkite.com/tpu-commons/kube-dev/builds/193) | `nocache` us-central1 | 61.8 | 21.7 | 9/13 |
| [191](https://buildkite.com/tpu-commons/kube-dev/builds/191) | `rsync` us-central1 | 121.6 | 13.6 | 7/13 |
