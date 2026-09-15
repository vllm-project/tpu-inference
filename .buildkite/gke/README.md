# GKE TPU Benchmarking & Diagnostic Toolkit

This directory contains production automation scripts, diagnostic tools, and templates for deploying, managing, and monitoring vLLM LLM inference workloads on Google Cloud TPUs (v7x) using Kubernetes JobSets and Kueue.

> 📖 **Cluster Reference**: These benchmarks and tests run on the shared GKE TPU v7x cluster (`bodaborg-tpu7x-nap` in project `cloud-tpu-shared-capacity`, region `us-central1`), as documented in the [TPU v7x Shared Cluster User Guide](https://docs.google.com/document/d/1qgrT8aW0MlPCcCqtQNlr7HqOn3wpv9BaFGFr0H7ynbU/).

* 🌟 **Preferred**: For the unified **Helm chart implementation** (Monolithic & Disaggregated), see [helm/README.md](helm/README.md).
* For direct **Kubernetes JobSet template execution** (`kubectl apply`), see [kubectl/README.md](kubectl/README.md).

---

## Table of Contents

- [Scripts Overview](#scripts-overview)
- [1. Cluster Capacity & Admission Diagnostics (`bin/cluster_status.sh`)](#1-cluster-capacity--admission-diagnostics-bincluster_statussh)
  - [Pre-Flight Admission Check (`-c`)](#pre-flight-admission-check--c)
  - [Cluster Health & Capacity Dashboard](#cluster-health--capacity-dashboard)
- [2. Workload & Pod Diagnostics (`bin/job_status.sh`)](#2-workload--pod-diagnostics-binjob_statussh)
- [3. Log Streaming & Performance Reports (`bin/tee_logs.sh`)](#3-log-streaming--performance-reports-bintee_logssh)
- [4. Running Benchmarks](#4-running-benchmarks)
- [5. Teardown & Resource Cleanup (`bin/cleanup.sh`)](#5-teardown--resource-cleanup-bincleanupsh)

---

## Scripts Overview

All operational, diagnostic, and monitoring scripts are located in [`gke/bin/`](./bin):

| Script | Purpose |
| :--- | :--- |
| [`bin/cluster_status.sh`](./bin/cluster_status.sh) | Inspects cluster-wide Kueue queues, GCE reservation headroom, idle TPU slices, traffic-light admission guide, and cluster anomalies. |
| [`bin/job_status.sh`](./bin/job_status.sh) (symlink: `bin/check_status.sh`) | Deep inspection of a specific JobSet/workload, showing pod roles, live startup phases, real-time CPU/memory metrics (`kubectl top`), and node pool placement. |
| [`bin/tee_logs.sh`](./bin/tee_logs.sh) (symlink: `bin/log.sh`) | Multiplexes live logs from all components (Prefill, Decode, Proxy, Client) into local files and auto-generates summary reports via `generate_summary.py`. |
| [`bin/generate_summary.py`](./bin/generate_summary.py) | Parses benchmark client logs and produces structured latency percentiles (TTFT, TPOT, ITL) and throughput metrics. |
| [`helm/run_benchmark.sh`](./helm/run_benchmark.sh) *(Preferred)* | Unified Helm-powered CLI runner supporting both monolithic and disaggregated benchmarks. See [helm/README.md](helm/README.md). |
| [`kubectl/run_benchmark.sh`](./kubectl/run_benchmark.sh) | CLI runner for deploying monolithic vLLM benchmarks using direct Kubernetes JobSet templates. See [kubectl/README.md](kubectl/README.md). |
| [`kubectl/run_disagg.sh`](./kubectl/run_disagg.sh) | CLI runner for deploying disaggregated (Prefill/Decode) vLLM benchmarks using direct JobSet templates. See [kubectl/README.md](kubectl/README.md). |
| [`bin/cleanup.sh`](./bin/cleanup.sh) | Safely tears down JobSets, pods, configmaps, and services for a release. |

---

## 1. Cluster Capacity & Admission Diagnostics (`bin/cluster_status.sh`)

[`bin/cluster_status.sh`](./bin/cluster_status.sh) provides a unified 7-section diagnostic dashboard to check physical hardware, cloud reservations, queue quotas, running jobs, and cluster health.

```bash
# Check full cluster status & workload admission guide:
./gke/bin/cluster_status.sh

# Live watch mode (refreshes every 10 seconds):
./gke/bin/cluster_status.sh -w

# Filter by a specific queue (e.g. 'default'):
./gke/bin/cluster_status.sh -q default

# Show only anomalies, failures, and warning signals:
./gke/bin/cluster_status.sh -a
```

### Pre-Flight Admission Check (`-c`)

Before submitting a workload, use `-c` to verify whether your workload can start immediately, needs autoscaling, or will get stuck in queue:

```bash
# Check a single topology (e.g. 2x2x4 / 16 chips):
./gke/bin/cluster_status.sh -c 2x2x4 -q default

# Check by chip count (e.g. 64 chips):
./gke/bin/cluster_status.sh -c 64 -q default

# Check a multi-topology / multi-slice disaggregated workload:
./gke/bin/cluster_status.sh -c "1x2x2x1, 2x2x2" -q default

# Check directly against a Helm values file:
./gke/bin/cluster_status.sh -c gke/helm/values-disagg-llama70b.yaml -q default
```

**Sample Output**:
```text
🔍 PRE-FLIGHT ADMISSION CHECK for '1x2x2x1, 2x2x2' in queue 'default':
----------------------------------------------------------------------------------------------------------------------------------
 Component       Topology     Count    Chips/Slice    Total Chips    Idle Slices in GKE   Action Required
 -------------   ----------   ------   ------------   ------------   ------------------   ----------------------------
 Prefill         2x2x1        1        4              4              2 slice(s) free      🟢 Instant Start (Use idle slice)
 Decode          2x2x2        2        8              16             1 slice(s) free      🟡 1 ready, 1 needs NAP scale-up
 -------------   ----------   ------   ------------   ------------   ------------------   ----------------------------
 Total Workload  -            3        -              20             -                    (Scale-up needed: 8 chips)

 Quota & Hardware Verification:
 [1] Kueue Queue Quota   : ✅ PASS (275 chips available in queue quota >= 20 chips requested)
 [2] GCE Cloud Headroom  : ✅ PASS (11 chips headroom in GCE reservation >= 8 chips to scale-up)

 🎯 VERDICT:
   🟡 SCALE-UP START (~2-3m Wait): Some slices need provisioning, but GCE reservation has enough headroom.
```

### Cluster Health & Capacity Dashboard

Running `./gke/bin/cluster_status.sh` displays:
1. **Physical Hardware**: VMs and TPU chips by topology (`2x2x1`, `2x2x2`, `2x2x4`, `4x4x4`, `4x4x8`), allocated vs free.
2. **GCE Cloud Reservations**: Total reserved chips in the cloud reservation (`cloudtpu-...`), in-use chips, remaining headroom for autoscaling, and utilization %.
3. **Kueue Shared Quotas**: Nominal, Used, Borrowed, and Remaining quota per flavor (`tpu7x-flavor`, `cpu-user`, etc.).
4. **Workload Admission Guide**: Traffic-light guidance (🟢 Instant, 🟡 Scale-up, 🔴 Queued) for each topology.
5. **Queued Workloads**: Workloads currently pending, requested TPUs, wait duration, and blocking reasons.
6. **Admitted & Running Workloads**: Active workloads consuming TPU capacity and their runtimes.
7. **Cluster Anomalies**: Nodes under DiskPressure, abnormal pods (ExitCode 137, CrashLoopBackOff), broken multi-host slices, and warning events.

---

## 2. Workload & Pod Diagnostics (`bin/job_status.sh`)

[`bin/job_status.sh`](./bin/job_status.sh) (and symlink `bin/check_status.sh`) provides deep inspection of an active or past workload:

```bash
# Automatically diagnose the latest active JobSet:
./gke/bin/job_status.sh

# Continuous live watch mode (refreshes every 5 seconds):
./gke/bin/job_status.sh -w

# Inspect a specific release or JobSet by name:
./gke/bin/job_status.sh my-70b-disagg
```

**Diagnostic Capabilities**:
* **Component Breakdown**: Prefill (`p`), Decode (`d`), Proxy Router (`x`), and Benchmark Client (`client`).
* **Live Startup Phases**: Detects container states: image pulling, GCS cache verification, Ray cluster initialization, XLA compilation, and serving readiness.
* **Live Resource Metrics**: Runs `kubectl top` to show real-time CPU cores and memory (RAM) usage per container.
* **Cluster Placement**: Identifies the exact GKE node and nodepool for each pod, and flags any scale-down/NAP activity.
* **Strict Column Alignment**: Formats long pod and node names with middle-truncation (`prefix...suffix`) to keep terminal tables aligned.

---

## 3. Log Streaming & Performance Reports (`bin/tee_logs.sh`)

[`bin/tee_logs.sh`](./bin/tee_logs.sh) multiplexes live pod logs from all distributed components into local files while preserving real-time terminal viewing:

```bash
# Automatically attach to the latest running benchmark:
./gke/bin/tee_logs.sh

# Attach to a specific release and assign log suffix:
./gke/bin/tee_logs.sh xibin-test-xe55u 12
```

Logs are saved under the [`log/`](./log) directory:
* `log/client.log<N>`: Client benchmark metrics (TTFT, TPOT, ITL, request throughput across 5 stages).
* `log/prefill.log<N>`: Prefill slice logs (Ray head/worker setup, JAX mesh, P2P KV transfers).
* `log/decode.log<N>`: Decode slice logs (weight loading, KV cache allocation, token generation).
* `log/proxy.log<N>`: Proxy router request dispatching and health probe logs.
* `log/log_summary.<N>`: Comprehensive performance report generated by [`bin/generate_summary.py`](./bin/generate_summary.py).

---

## 4. Running Benchmarks

> 💡 **Recommendation**: Running benchmarks using the **Helm chart** is preferred and recommended for all standard and production workflows. Direct `kubectl` template execution is provided as a lightweight alternative.

Workload deployment, benchmarking, and direct `kubectl` operations are detailed in their respective folder guides:

* 🚀 **[Helm Chart Guide (`helm/README.md`)](helm/README.md)** *(Preferred)*: Production-grade Helm chart supporting Monolithic and Disaggregated architectures, automated TPU topology calculations, Ray cluster management, compact placement policies, and progressive benchmark suites via [`helm/run_benchmark.sh`](helm/run_benchmark.sh).
* ⚙️ **[Direct Kubectl Guide (`kubectl/README.md`)](kubectl/README.md)**: Lightweight template-driven runners for Monolithic ([`kubectl/run_benchmark.sh`](kubectl/run_benchmark.sh)) and Disaggregated ([`kubectl/run_disagg.sh`](kubectl/run_disagg.sh)) workloads using raw Kubernetes manifests and `envsubst` without Helm. Includes direct `kubectl` pod inspection, log streaming, and teardown commands.

---

## 5. Teardown & Resource Cleanup (`bin/cleanup.sh`)

Safely deletes all Kubernetes resources (JobSets, component pods, ConfigMaps, and Helm releases) associated with a benchmark run:

```bash
# Clean up a specific release or workload:
./gke/bin/cleanup.sh <RELEASE_NAME>

# Example:
./gke/bin/cleanup.sh xibin-test-xe55u
```
