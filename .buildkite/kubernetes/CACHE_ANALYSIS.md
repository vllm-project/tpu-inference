# TPU utilisation on Kubernetes, against bare metal

> **Preliminary.** The storage numbers below are current as of build
> [241](https://buildkite.com/tpu-commons/kube-dev/builds/241). The cache was
> resized afterwards - 73Gi of models and 16Gi of compilation cache in a 98Gi
> volume, against 241's unbounded capacities - and build
> [242](https://buildkite.com/tpu-commons/kube-dev/builds/242) will settle
> part2's figure. Everything else stands.

Does running TPU CI on Kubernetes use the accelerators at least as efficiently
as the bare-metal pipeline it replaces?

**Yes, and the suite now finishes sooner.** Bare metal spends 38% of its TPU
time not testing; Kubernetes spends 12%. Step for step the two are level -
0.98x across the ten steps that exist on both - and the whole suite completes
in 85 minutes against 114 for the same work in the first green Kubernetes run.

Two changes account for nearly all of it, and neither is about Kubernetes:

* **`VLLM_XLA_CHECK_RECOMPILATION=1`**, which bare metal has always set and
  Kubernetes never did. Without it JAX declines to persist anything small or
  quick to compile, so a self-built cache plateaus at 57.7m on speculative
  decoding no matter how many times it runs. With it, 28.0m - level with a
  cache that bare metal filled.
* **The file cache being large enough to hold a checkpoint**, and the tests
  that read those checkpoints being ordered so one is resident at a time.
  Together they took part2's checkpoint loading from 84.6m to 8.8m.

The ordering fix is not a Kubernetes fix at all. `tests/models/jax/test_gemma4.py`
cycled four ~30GB checkpoints round-robin, so each was fetched four times with
the other three in between - the worst case for any cache, including the page
cache in front of bare metal's local disk. Bare metal pays it too. We only
found it because the Kubernetes configuration made it expensive enough to
chase.

Every number is from a real run of the same v6e steps, with one image pinned
across configurations so the storage backend is the only variable. Bare metal
is the median of the most recent passing nightlies.

## The metric

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

## Result

**Whole suite, 14 steps.** Build [241](https://buildkite.com/tpu-commons/kube-dev/builds/241),
the first green run on the configuration we ship, against build
[230](https://buildkite.com/tpu-commons/kube-dev/builds/230), the first green
run at all.

| | 230 | 241 |
|---|---|---|
| result | 14/14 | 14/14 |
| makespan | 114.4m | **85.4m** |
| JAX unit tests part2 | 111.8m | 81.9m |
| E2E speculative decoding | 28.8m | 30.3m |
| E2E disagg | failed | 3.6m |

**Step for step against bare metal.** Execution time on Kubernetes - first
workload output to job completion - against bare-metal wall time, which is the
right comparison because a bare-metal VM holds its chips through checkout, the
docker pull and the cache rsync.

| step | kube 241 | bare metal | ratio |
|---|---|---|---|
| JAX unit tests part2 | 81.9m | 79.6m | 1.03x |
| E2E speculative decoding | 30.3m | 29.6m | 1.02x |
| JAX unit tests part1 | 15.1m | 14.1m | 1.07x |
| Accuracy Qwen2.5-VL-7B | 8.7m | 6.4m | **1.37x** |
| E2E MLPerf JAX models | 8.2m | 8.1m | 1.01x |
| E2E MLPerf JAX + vLLM | 7.2m | 7.3m | 0.98x |
| lora e2e multi chip | 4.3m | 6.5m | **0.66x** |
| E2E MPMD data parallelism | 2.5m | 5.3m | **0.47x** |
| lora unit multi chip | 2.2m | 4.8m | **0.45x** |
| lora unit single chip | 1.9m | 3.1m | 0.61x |
| **sum of 10 matched** | **162.3m** | **164.8m** | **0.98x** |

Four steps - three Runai streamer variants and disagg - have no bare-metal
equivalent under the same name and are excluded. The Accuracy step is the one
genuine regression and has not been investigated.

**Chip-minutes.** Durations weighted by chips, from build 230:

| | executing | held | overhead |
|---|---|---|---|
| bare metal | 183 | 297 | 38.4% |
| Kubernetes | 320 | 363 | **11.9%** |

The two are not on the same base - 230 ran 14 steps where the bare-metal figure
covers 13 - so read the overhead column, not the totals.

**The efficiency problem is no longer chip-minutes.** At 11.9% there is little
left to win there. What remains is makespan: every step except part2 finishes
inside 32 minutes, and part2 then runs alone for another 82. Sharding it is
worth more than any further storage work, and costs no extra chip-minutes.

## Where bare metal's 38% goes

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

## Where Kubernetes' overhead goes

**Mounting GCS FUSE costs ~36 chip-minutes** over a plain local disk (49 against
13). That is the metadata prefetch walking 24,441 cache entries at mount. It
scales with cache size, so partitioning the namespace by topology would reduce
it — entries are already distinct per shape, so splitting them loses no sharing.

**Pulling the cache onto an empty disk costs 133 chip-minutes** (`rsync`), which
is why it is the only same-region configuration worse than bare metal. Starting
from a clone and pulling only the delta costs 26.

**Cross-region latency is the largest single effect in the table.** Measured
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

## Downloading models instead of mounting them

`clone-nomodel` is the most efficient cross-region configuration at 0.76×,
because it skips the cross-region model mount entirely. Its steps completed
normally — identical test counts to every other run, 233 passed and 148 skipped
on part1, 42 passed on the lora unit tests, 11 on speculative decoding — so the
timings are real rather than truncated by an early abort.

**It logged zero HTTP 429 responses.** An earlier draft reported 698
rate-limit events for this configuration. That figure came from matching the
digits `429` anywhere in the logs and was wrong; there was no throttling in
these 13 steps.

Rate limiting is real but the evidence is narrower than that claim. One step,
once: `JAX unit tests part2` in build [160](https://buildkite.com/tpu-commons/kube-dev/builds/160), with no model cache, hit
`2500 api requests per 5 minutes` and failed. It resolves far more models than
any other step and is excluded from every comparison here, so this set cannot
say whether downloading is safe at full-suite scale.

With a same-region model mount the question does not arise, and the mount costs
nothing measurable against downloading. The reason to prefer it is that it
keeps a third-party request budget off the critical path — not that we measured
that budget being exceeded.

## Scheduling: profiles, borrowing and autoscaling

The fleet is **10 single-chip nodes plus one eight-chip node — 18 chips**. Two
profiles use it:

| profile | chips | topology | nodes | nominal quota |
|---|---|---|---|---|
| `v6e-1-1x1` | 1 | 1x1 | 0–10, min 2 | 10 chips |
| `v6e-8-2x4` | 8 | 2x4 | 0–1 | 8 chips |

Both sit in one Kueue cohort, so either can borrow the other's quota and have it
reclaimed. Eight of the thirteen steps are single-chip and five are eight-chip.

### Borrowing was measured directly, then deliberately abandoned

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

### What the two profiles cost now

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

### Nodes are created for most steps

82% of single-chip steps (59 of 72) waited for a node to be created, against
49% of eight-chip steps. Scale-up costs 2.6 minutes per single-chip step.

This is the cost of scaling to zero, and it is charged before any chip is held,
which is why it sits outside the utilisation figures. It is real for
time-to-result, and it is the reason `min_nodes = 2` exists on the single-chip
pool — the first two steps of a build skip it.

### Fleet occupancy during a build

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

### The autoscaler will take a node out from under a running test

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

### Two effects this does not quantify

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

## Why per-region caches are natural here and awkward on bare metal

The measurements above turn on the cache being in the cluster's region. That is
worth making a design point rather than a one-off fix, because the two
platforms make it very different work.

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

### It is also cheaper, structurally

Storage cost scales with different things on the two platforms.

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

Exact figures would need the bare-metal disk size and instance count, which we
do not have, so this is a statement about shape rather than a total.


## The compilation cache: Kubernetes can seed its own

This was open in the first version of this document, and withdrawn from the
second because two variables had moved. It is now measured.

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

## The model cache: capacity, then ordering

part2 is the only step with a large model working set, and it was 1.40x bare
metal. Two independent causes, contributing about equally:

| | execution | checkpoint loading |
|---|---|---|
| 20Gi cache, round-robin | 111.8m | 84.6m |
| 56Gi cache, round-robin | 79.1m | 23.8m |
| 56Gi cache, grouped | **71.6m** | **8.8m** |
| bare metal | ~79.6m | - |

**Capacity.** `cache-file-for-range-read` ingests the whole object on a partial
read, and gcsfuse will not cache an object that does not fit the remaining
capacity. part2 reads gemma-4 checkpoints a few layers at a time, so at 20Gi a
~30GB shard was refetched every time while evicting the small models that would
have fit.

**Ordering.** Stacked `parametrize` varies the topmost decorator fastest, so
four checkpoints cycled round-robin and each was fetched four times. Grouping
them made one checkpoint resident at a time. This costs bare metal too.

Two mechanisms worth remembering, both learned the hard way:

* `fileCacheCapacity` is the threshold gcsfuse evicts against. Setting it to
  `-1` does not mean "use everything" in any useful sense - it means there is
  no threshold, so the volume fills and writes fail. Build
  [241](https://buildkite.com/tpu-commons/kube-dev/builds/241) spent 26.6m on
  loading that way against build 234's 8.8m in a *smaller* volume.
* The capacity is nominal unless the pod declares a `gke-gcsfuse-cache` volume
  at least as large. Without one the sidecar falls back to 5GiB of ephemeral
  storage; build [239](https://buildkite.com/tpu-commons/kube-dev/builds/239)
  spent 113.6m loading, worse than having no tuning at all.

## What we tried and rejected

**GKE gcsfuse profiles.** Managed StorageClasses that size the cache from the
node and the bucket, which would have replaced a capacity figure derived by
hand from one machine type. Measured against the hand-tuned mounts on the same
image and ordering, build
[240](https://buildkite.com/tpu-commons/kube-dev/builds/240) ran part2 in
115.9m with 91.3m of loading, against 71.6m and 8.8m - individual checkpoint
loads of 21 minutes. A profile does not set the readahead, parallel-download
and chunk-size options these mounts were tuned with, and adding them back on
top is hand-tuning with a StorageClass attached.

Profiles also gate the pod on a bucket scan while it runs, and the scan
authenticates as the GKE service agent rather than the workload identity - a
second reader that the bucket IAM had to be widened for.

## Recommendation

**Both caches as PersistentVolumeClaims over GCS FUSE, in the cluster's own
region, with `VLLM_XLA_CHECK_RECOMPILATION=1` set on every step.**

The claim names are identical in every region and the bucket lives in the
PersistentVolume, so a workload manifest never carries a bucket name and
survives MultiKueue placing it in a region it was not submitted from. The mount
is writable, so the cache fills itself and what a test compiles survives its
pod - which, with the flag set, is now enough to reach bare metal's numbers
without bare metal.

Sizing is the part that needs care rather than the backend:

* the pod's `gke-gcsfuse-cache` volume is RAM-backed and sized per machine type
  by the launcher, since host memory ranges from 176 GB on `ct6e-standard-1t`
  to 1440 GB on `ct6e-standard-8t`;
* `fileCacheCapacity` is explicit on both mounts and fixed, because a
  PersistentVolume is one object per cluster and every profile binds the same
  claim - so it has to hold on the smallest shape;
* the two capacities sum to less than the volume, or the tmpfs reaches its
  `sizeLimit` before either mount reaches its own eviction threshold.

The `clone` backend remains marginally cheaper in chip-minutes and is still not
the recommendation, for the reason it never was: it writes nothing back, so its
efficiency is borrowed from whatever filled the golden disk. That argument used
to be theoretical. It is not any more - the self-seeding measurement above is
what a writable cache buys.

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

## Corrections

Recorded because each cost real time and the failure mode will recur.

**Builds [178](https://buildkite.com/tpu-commons/kube-dev/builds/178) and [180](https://buildkite.com/tpu-commons/kube-dev/builds/180) ran with no compilation cache at all.**
`JAX_COMPILATION_CACHE_DIR` was set to `/cache/jax/jax0.11.0_tputpu6e` to match
the bucket layout when the cache moved to FUSE, and kept when it moved back to a
clone — but the golden's entries were at `/cache/jax`, from before the namespace
existed. JAX read an empty directory. The slowdown was attributed first to
cross-region model reads, then to staleness; it was neither.

**A silent default overrode two experiments.** `GCS_CACHE_BUCKET` had a default
in the pipeline's `env:` block, which in Buildkite wins over
`bk build create --env`. Builds [196](https://buildkite.com/tpu-commons/kube-dev/builds/196) and [197](https://buildkite.com/tpu-commons/kube-dev/builds/197) measured the old bucket and reported
it as the new one.

**A cache refresh filled a disk with another cluster's entries.** An unscoped
`gcloud storage rsync` pulled 23 GB of `jax0.11.0_tputpu7x` onto a 50 GiB disk
serving only v6e. `gcloud` exited 0 having failed 21,708 writes.

**A rate-limit figure was fabricated by a bad regex.** An earlier draft
reported 698 throttling events for the no-model-cache configuration, from
matching `429` anywhere in the logs. The true count was zero. The claim that
downloading models is "fast but not viable" rested on it and did not survive
checking.

**`JAX unit tests part2` has never produced a valid number**, and its
durations are lower bounds rather than completed runs: it executes under
`pytest -x`, so a single failure ends it early. In [203](https://buildkite.com/tpu-commons/kube-dev/builds/203) it stopped
after 1796 of its tests on an environment assumption - GKE injects
TPU_ACCELERATOR_TYPE into every TPU pod and the test asserted it absent.
Before that it was evicted by the cluster autoscaler in builds [175](https://buildkite.com/tpu-commons/kube-dev/builds/175)
and [178](https://buildkite.com/tpu-commons/kube-dev/builds/178) — job pods are evictable by default and
it is the only step still running when a build drains — and cold-cached in 180.

Every configuration's steps were checked for identical pytest counts before
its timings were used, so a step that aborted early cannot masquerade as a fast
one. All 13 steps match across all runs.

The common thread: **every cache failure here is silent.** Missing, stale, empty
or wrong-path all produce no error, only a slower run and a chip held longer. An
assertion at pod start that the cache directory exists and is non-empty would
have caught all of them in one build rather than four.

**`fileCacheCapacity: -1` was read as "use the whole volume".** It removes the
eviction threshold instead. Build 241 spent three times longer loading
checkpoints than build 234 did in a smaller volume with explicit capacities.

**Profile-backed volumes were first measured without a cache volume.** Build
239's 113.6m of loading was a 5GiB ephemeral fallback, not a profile - the
variant had dropped `gke-gcsfuse-cache` on the incorrect assumption that a
profile provisions its own. Build 240 is the valid comparison.

**Step ordering was assumed to be irrelevant to storage.** It was worth more
than the cache sizing: 23.8m of loading became 8.8m without touching the mount.

## Open questions

- ~~Can Kubernetes sustain its own cache?~~ **Answered** - see "The compilation
  cache" above. It can, but only with `VLLM_XLA_CHECK_RECOMPILATION=1`; without
  it a self-built cache plateaus at roughly twice the seeded time and stays
  there.
- **What does the push cost?** The rsync variants push from a sidecar after the
  workload exits, which is after the launcher stops collecting logs, so it is
  absent from every figure above.
- **Does topology partitioning pay?** It would shrink the 36 chip-minute
  prefetch. Unmeasured.
- **How is the regional bucket kept current?** It is a manual snapshot today.
  Whatever replaces bare metal as producer has to keep it fresh, per region.

## Reproducing

```
bk build create --pipeline tpu-commons/kube-dev --branch <branch> \
  --env TEST_TYPE=jax --env SKIP_PART2=1 \
  --env CACHE_BACKEND=<clone|fuse|rsync|clone-rsync|clone-nomodel|nocache|none> \
  --env GCS_CACHE_BUCKET=<bucket> \
  --env WORKLOAD_IMAGE=<pinned image>
```

Pin the image. Without it each run builds its own, the HLO changes, the cache
hit rate moves with it, and the storage backend cannot be isolated.

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
