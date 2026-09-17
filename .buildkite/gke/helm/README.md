# TPU vLLM Benchmark Helm Chart

A unified, production-grade Helm chart for deploying, orchestrating, and benchmarking large language models (LLMs) with **vLLM** on **Google Cloud TPUs (v7x)** via Kubernetes **JobSets** and **Kueue**.

Supports both **Monolithic (Aggregated)** and **Disaggregated (Prefill/Decode P/D)** inference architectures from a single, templated chart.

> For direct **Kubernetes JobSet template execution** (`kubectl apply`), see [../kubectl/README.md](../kubectl/README.md).
> For cluster diagnostic tools ([`bin/cluster_status.sh`](../bin/cluster_status.sh)), live workload monitoring ([`bin/job_status.sh`](../bin/job_status.sh)), log streaming ([`bin/tee_logs.sh`](../bin/tee_logs.sh)), and cleanup automation, see the parent [GKE Operations Guide](../README.md).

---

## Table of Contents

- [Chart Structure](#chart-structure)
- [Architecture & Template Implementation](#architecture--template-implementation)
  - [1. Monolithic Mode (`mode: "aggregated"`)](#1-monolithic-mode-mode-aggregated)
  - [2. Disaggregated Mode (`mode: "disaggregated"`)](#2-disaggregated-mode-mode-disaggregated)
  - [3. Exclusive Topology Affinity & Slice Colocation](#3-exclusive-topology-affinity--slice-colocation)
  - [4. Helper Templates (`_helpers.tpl`)](#4-helper-templates-_helperstpl)
- [Storage Modes & Caching](#storage-modes--caching)
- [Values Files Catalog](#values-files-catalog)
- [Values Schema Reference](#values-schema-reference)
- [Helm Automation Runner (`run_benchmark.sh`)](#helm-automation-runner-run_benchmarksh)
  - [Prerequisites & Environment Setup](#prerequisites--environment-setup)
  - [Usage & Options](#usage--options)
  - [Container Image Quick Switches](#container-image-quick-switches)
  - [Example Commands](#example-commands)
- [Helm CLI Workflow](#helm-cli-workflow)
  - [Install / Deploy](#install--deploy)
  - [Render / Dry-Run (`helm template`)](#render--dry-run-helm-template)
  - [Uninstall / Teardown](#uninstall--teardown)

---

## Chart Structure

```text
gke/helm/
├── Chart.yaml                          # Helm chart metadata (v0.1.0)
├── values.yaml                         # Base default configuration
├── buildkite_to_helm.py                # Buildkite CI pipeline to Helm values converter
├── run_benchmark.sh                    # Automated Helm benchmark deployment script
├── values-llama8b.yaml                 # Monolithic Llama-3.1-8B (2x2x1, 1 VM)
├── values-llama8b-ci.yaml              # Llama-3.1-8B CI Benchmark runner (Buildkite Parity Test)
├── values-llama70b.yaml                # Monolithic Llama-3.1-70B (2x2x4, 4 VMs via Ray)
├── values-qwen4b.yaml                  # Monolithic Qwen3.5-4B (2x2x1, 1 VM)
├── values-disagg-llama8b.yaml          # Disaggregated 8B (Prefill 2x2x1, Decode 2x2x2)
├── values-disagg-llama70b.yaml         # Disaggregated 70B (Prefill 2x2x4, 2x Decode 2x2x4 = 12 VMs, 96 TPUs)
├── values-disagg-symmetric.yaml        # Disaggregated 8B (Prefill 2x2x1, Decode 2x2x1)
├── values-disagg-2decode.yaml          # Disaggregated 8B (Prefill 2x2x1, 2x Decode 2x2x1)
└── templates/
    ├── _helpers.tpl                    # Hardware sizing macros, TP math & placement calculations
    ├── configmap.yaml                  # Embedded startup scripts, health probes & benchmark runner
    └── jobset.yaml                     # Unified JobSet manifest (aggregated, disaggregated & script)

```

---

## Architecture & Template Implementation

### 1. Monolithic Mode (`mode: "aggregated"`)

In monolithic mode, `templates/jobset.yaml` renders two `replicatedJobs`:

```
                   +-------------------------------------------------------------+
                   |                     Kubernetes JobSet                       |
                   |                                                             |
                   |   +-----------------------------------------------------+   |
                   |   |               ReplicatedJob: server                 |   |
                   |   |  (Single-host: 1 VM | Multi-host: N VMs via Ray)    |   |
                   |   |  - Step 0: Pre-cache weights (GCS / HF / RAM)       |   |
                   |   |  - Step 1: TPU XLA compilation & graph warmup       |   |
                   |   |  - Step 2: vLLM OpenAI-compatible API Server (:8000)|   |
                   |   +--------------------------^--------------------------+   |
                   |                              | (Health polling & HTTP)      |
                   |   +--------------------------v--------------------------+   |
                   |   |               ReplicatedJob: client                 |   |
                   |   |          (Runs on dedicated CPU nodes)              |   |
                   |   |  - Progressive 5-stage benchmark suite (128-3072 tok)|  |
                   |   |  - All stages finished -> Exit 0 -> Teardown JobSet |   |
                   |   +-----------------------------------------------------+   |
                   +-------------------------------------------------------------+
```

- **Server Job (`replicatedJob: server`)**: Runs on TPU nodes (`cloud.google.com/gke-tpu-accelerator: tpu7x`). For multi-host topologies (e.g. `2x2x4`), Worker 0 acts as the Ray Head, and Workers 1..N join as Ray Workers.
- **Client Job (`replicatedJob: client`)**: Runs on CPU nodes (`cloud.google.com/gke-nodepool: cpu-np`), polls server readiness, runs the 5-stage progressive benchmark, and terminates the JobSet upon completion.

---

### 2. Disaggregated Mode (`mode: "disaggregated"`)

Separates prefill computation and decode token generation across independent TPU slices linked via high-speed P2P KV connectors (`TPUConnector` or `TPURaidenConnector`). `templates/jobset.yaml` renders four `replicatedJobs`:

```
       +-----------------------------------------------------------------------------------------+
       |                                   Kubernetes JobSet                                     |
       |                                                                                         |
       |  +--------------------------------+             +------------------------------------+  |
       |  |     ReplicatedJob 1: p         |             |        ReplicatedJob 2: d          |  |
       |  |  Prefill Slice (KV Producer)   |             |   Decode Slice (KV Consumer)       |  |
       |  |  TPU v7 (e.g. 2x2x1 / TP=4)    |             |   TPU v7 (e.g. 2x2x2 / TP=8)       |  |
       |  |  Serves on Port 8400           |             |   Serves on Port 9400              |  |
       |  +---------------^----------------+             +-----------------^------------------+  |
       |                  |                                                |                     |
       |                  |        High-Speed P2P KV Transfer (:9100)      |                     |
       |                  +================================================+                     |
       |                                                                                         |
       |                                          ^                                              |
       |                                          |                                              |
       |                       +------------------v-------------------+                          |
       |                       |        ReplicatedJob 3: x            |                          |
       |                       |      Disaggregated Proxy Router      |                          |
       |                       |    (CPU Pod - Port 8000 via HTTP)    |                          |
       |                       +------------------^-------------------+                          |
       |                                          |                                              |
       |                       +------------------v-------------------+                          |
       |                       |       ReplicatedJob 4: client        |                          |
       |                       |       Progressive Benchmark Suite    |                          |
       |                       +--------------------------------------+                          |
       +-----------------------------------------------------------------------------------------+
```

- **Prefill Job (`replicatedJob: p`)**: Handles prompt processing and KV cache generation, serving internal requests on port 8400.
- **Decode Job (`replicatedJob: d`)**: Handles autoregressive token generation on port 9400. Configurable with multiple replicas (`decode.replicas: 2`).
- **Proxy Router (`replicatedJob: x`)**: Lightweight CPU pod running `toy_proxy_server.py` on port 8000. It dynamically inspects request lengths and routes them to prefill and decode instances.
- **Benchmark Client (`replicatedJob: client`)**: Sends traffic to proxy port 8000.

---

### 3. Exclusive Topology Affinity & Slice Colocation

Multi-host TPU pods must be scheduled on the same physical TPU slice. If pods from a single multi-host slice are scheduled across different physical node pools, the Inter-Chip Interconnect (ICI) initialization fails with `SLICE_FAILURE_CHIP_DRIVER_ERROR`.

In `templates/jobset.yaml`, exclusive topology is conditionally applied:

```yaml
{{- if or (gt (include "tpu.numVMs" .Values.tpu.topology | int) 1) (and (eq .Values.mode "disaggregated") (or (gt (include "tpu.numVMs" .Values.prefill.tpu.topology | int) 1) (gt (include "tpu.numVMs" .Values.decode.tpu.topology | int) 1))) }}
alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
{{- end }}
```

- **TPU Jobs (`server`, `p`, `d`)**: Scheduled with 1:1 mapping to dedicated TPU node pools created by GKE Node Auto-Provisioning (NAP).
- **CPU Jobs (`x`, `client`)**: Explicitly override node affinity via `cloud.google.com/gke-nodepool: cpu-np`, allowing them to run in the shared CPU pool without blocking or triggering new TPU node pool allocations.

---

### 4. Helper Templates (`_helpers.tpl`)

The template library calculates all hardware and distributed parameters automatically:

| Helper Function | Purpose | Example |
| :--- | :--- | :--- |
| `tpu.numVMs` | Computes VM count from topology string: $(X \times Y \times Z) / 4$ | `2x2x4` &rarr; `4` VMs |
| `tpu.tpSize` | Computes Tensor Parallel size: $X \times Y \times Z$ | `2x2x4` &rarr; `16` TP |
| `tpu.processBounds` | Calculates JAX multi-host process bounds | `2x2x4` &rarr; `1,1,4` |
| `tpu.chipsPerProcessBounds` | Calculates chips-per-process bounds | Standard v7x &rarr; `2,2,1` |
| `tpu.placementPolicy` | Queries/formats GCE Compact Placement Policy for multi-host VMs | e.g. `tpu7x-2x2x4-compact` |

---

## Storage Modes & Caching

Configured via `storage.type` in values:

| Mode | Volume Type | Best Used When | Configuration |
| :--- | :--- | :--- | :--- |
| **`ramdisk`** | In-Memory `emptyDir` (`medium: Memory`) | Ephemeral testing directly from Hugging Face Hub. | `storage.type: "ramdisk"`<br>`storage.ramCacheLimit: "80Gi"` |
| **`gcs-cache`** | GCS FUSE CSI Mount | Warm boots across runs by caching Hugging Face downloads to GCS. | `storage.type: "gcs-cache"`<br>`storage.bucketName: "<bucket>"` |
| **`gcs-direct`** | Direct GCS Mount | Serving pre-downloaded weights already stored in GCS. | `storage.type: "gcs-direct"`<br>`model.name: "gs://<bucket>/<path>"` |

> **Cache Verification Speedup**: In `gcs-cache` mode, `templates/configmap.yaml` downloads weights using `ignore_patterns=['original/*']`. This ignores redundant 140 GB PyTorch `.pth` un-sharded checkpoints, cutting GCS cache verification from **2+ hours down to ~45 seconds**.

---

## Values Files Catalog

| Preset File | Architecture | Model | TPU Slices & Hardware | Notes |
| :--- | :--- | :--- | :--- | :--- |
| [`values.yaml`](./values.yaml) | Monolithic | Llama-3.1-8B | `2x2x1` (1 VM, 4 chips) | Base default values |
| [`values-llama8b.yaml`](./values-llama8b.yaml) | Monolithic | Llama-3.1-8B | `2x2x1` (1 VM, 4 chips) | Monolithic baseline |
| [`values-llama8b-ci.yaml`](./values-llama8b-ci.yaml) | Script Runner (`mode: "script"`) | Llama-3.1-8B | `2x1x1` (1 VM, 2 chips, `tpu7x-2`) | Direct Buildkite CI test runner (executes `benchmark.sh`, sonnet dataset, TP=2, threshold >= 10.77 req/s) |
| [`values-llama70b.yaml`](./values-llama70b.yaml) | Monolithic | Llama-3.1-70B | `2x2x4` (4 VMs, 16 chips via Ray) | Multi-host monolithic benchmark |
| [`values-qwen4b.yaml`](./values-qwen4b.yaml) | Monolithic | Qwen3.5-4B | `2x2x1` (1 VM, 4 chips) | Lightweight 4B test |
| [`values-disagg-symmetric.yaml`](./values-disagg-symmetric.yaml) | Disaggregated | Llama-3.1-8B | Prefill `2x2x1`, Decode `2x2x1` | 1:1 symmetric serving (10 RPS) |
| [`values-disagg-llama8b.yaml`](./values-disagg-llama8b.yaml) | Disaggregated | Llama-3.1-8B | Prefill `2x2x1`, Decode `2x2x2` | 1:2 asymmetric serving (35 RPS) |
| [`values-disagg-llama70b.yaml`](./values-disagg-llama70b.yaml) | Disaggregated | Llama-3.1-70B | Prefill `2x2x4`, 2x Decode `2x2x4` | 12 VMs, 96 TPU v7x chips with topology colocation |
| [`values-disagg-2decode.yaml`](./values-disagg-2decode.yaml) | Disaggregated | Llama-3.1-8B | Prefill `2x2x1`, 2x Decode `2x2x1` | 1 Prefill + 2 Decode replicas |

---

## Values Schema Reference

Key parameters in `values.yaml`:

```yaml
mode: "aggregated"                    # "aggregated" (monolithic) or "disaggregated"

model:
  name: "meta-llama/Llama-3.1-8B-Instruct"  # HuggingFace ID or gs:// URI
  servedName: ""                      # Override tokenizer/model name

image:
  repository: "docker.io/vllm/vllm-tpu"
  tag: "v0.27.0"

# Monolithic settings:
tpu:
  topology: "2x2x1"                   # TPU topology shape
  placementPolicy: ""                 # Compact placement policy (required if VMs > 1)
  resources:
    cpu: "32"
    memory: "64Gi"
    tpu: 4

# Disaggregated settings (used when mode: "disaggregated"):
prefill:
  replicas: 1
  tpu:
    topology: "2x2x1"
    placementPolicy: ""
  port: 8400

decode:
  replicas: 1                         # Number of decode slice replicas
  tpu:
    topology: "2x2x2"
    placementPolicy: ""
  port: 9400
  connector: "TPUConnector"           # "TPUConnector" or "TPURaidenConnector"

proxy:
  port: 8000
  resources:
    cpu: "8"
    memory: "16Gi"

storage:
  type: "gcs-cache"                   # "ramdisk", "gcs-cache", or "gcs-direct"
  bucketName: "my-gcs-bucket"
  cacheSubpath: "hf-cache"
  ramCacheLimit: "80Gi"

job:
  declaredDurationMinutes: 90         # Max Kueue reservation duration
  backoffLimit: 2                     # Retry tolerance for transient network reconnects
  priorityClass: "medium"

benchmark:
  requestRate: 35.0                   # Benchmark request rate in req/s
  stages:                             # Progressive prompt/output evaluation stages
    - inputLen: 128
      outputLen: 128
      numPrompts: 100
    - inputLen: 512
      outputLen: 256
      numPrompts: 100
    - inputLen: 1024
      outputLen: 512
      numPrompts: 100
    - inputLen: 2048
      outputLen: 512
      numPrompts: 100
    - inputLen: 3072
      outputLen: 512
      numPrompts: 100
```

---

## Helm Automation Runner (`run_benchmark.sh`)

[`run_benchmark.sh`](./run_benchmark.sh) is a Helm-powered automation runner that dynamically calculates TPU hardware sizing (VMs, Tensor Parallel size, chip/process bounds, placement policies), formats a custom `values.yaml`, and automatically deploys the chart via `helm install` or renders manifests via `helm template`.

### Prerequisites & Environment Setup

Before running benchmarks with `run_benchmark.sh`, ensure your local workstation or Cloud Shell environment has the necessary tools, Google Cloud authentication, cluster access, and Hugging Face credentials configured.

#### 1. Install Required Tools

Ensure the following command-line tools are installed on your system:

| Tool | Minimum Version | Description & Verification |
| :--- | :--- | :--- |
| **`gcloud`** | Latest | Google Cloud SDK (`gcloud version`) |
| **`gke-gcloud-auth-plugin`** | Latest | Required GKE authentication plugin for `kubectl` |
| **`kubectl`** | v1.26+ | Kubernetes CLI (`kubectl version --client`) |
| **`helm`** | v3.10+ | Helm package manager (`helm version`) |
| **`jq` & `bc`** | Any | Required by diagnostic and runner scripts for JSON parsing and math |

**Quick Installation (Debian/Ubuntu/gLinux):**

```bash
# Install gke-gcloud-auth-plugin and kubectl via gcloud
gcloud components install kubectl gke-gcloud-auth-plugin

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install jq and bc utilities
sudo apt-get update && sudo apt-get install -y jq bc
```

#### 2. Authenticate with Google Cloud

Authenticate both your user credentials and Application Default Credentials (ADC) used by Google Cloud SDKs and GCS storage clients:

```bash
# 1. Log in to your Google Cloud user account
gcloud auth login

# 2. Generate Application Default Credentials (ADC) for GCS bucket storage & APIs
gcloud auth application-default login

# 3. Set your target GCP project
gcloud config set project cloud-tpu-shared-capacity
```

#### 3. Connect to the GKE TPU Cluster

Obtain cluster credentials to point `kubectl` and `helm` to the target GKE cluster:

```bash
# Fetch credentials for the TPU cluster (example: bodaborg-tpu7x-nap in us-central1)
gcloud container clusters get-credentials bodaborg-tpu7x-nap \
  --region us-central1 \
  --project cloud-tpu-shared-capacity

# Verify cluster connection and TPU node availability
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x
```

#### 4. Configure Hugging Face Secret

For gated or access-controlled models (e.g. `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct`), the runner automatically mounts a Kubernetes Secret named `${CLEAN_USER}-test-token` (where `${CLEAN_USER}` is your lowercase alphanumeric username):

```bash
# Determine your clean username prefix (e.g. 'johndoe' -> 'johndoe-test-token')
CLEAN_USER=$(echo "${USER}" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')

# Create or update the secret in your Kubernetes namespace
kubectl create secret generic "${CLEAN_USER}-test-token" \
  --from-literal=HF_TOKEN="hf_your_actual_token_here" \
  --dry-run=client -o yaml | kubectl apply -f -
```

> [!TIP]
> Verify your secret exists with:
>
> ```bash
> kubectl get secret "${CLEAN_USER}-test-token"
> ```

#### 5. Verify Cluster Headroom & TPU Quota

Before submitting a benchmark job, check physical TPU slice availability and Kueue queue quotas:

```bash
# View cluster capacity dashboard, reservation headroom, and queue status:
./gke/bin/cluster_status.sh

# Or test instant admission for your target topology (e.g. 2x2x1):
./gke/bin/cluster_status.sh -c 2x2x1 -q default
```

### Usage & Options

```bash
./gke/helm/run_benchmark.sh [OPTIONS]
```

| Flag | Category | Description | Default |
| :--- | :--- | :--- | :--- |
| `-m, --model <model>` | Common | Model name, HF repo ID, or `gs://` URI | `meta-llama/Llama-3.1-8B-Instruct` |
| `-i, --image <image>` | Common | Docker container image or alias (`torchtpu`, `tpu-inference`) | `docker.io/vllm/vllm-tpu:v0.27.0` |
| `--torchtpu` | Common | Quick switch to Google TorchTPU production image (`torchtpu-vllm-prod:latest`) | `false` |
| `--tpu-inference` | Common | Quick switch to upstream `tpu-inference` image (`vllm-tpu:v0.27.0`) | `true` (default) |
| `-b, --bucket <bucket>` | Common | GCS bucket path for caching (e.g. `gs://<bucket>/hf-cache`) | `""` (RAM-Disk mode) |
| `-r, --release <name>` | Common | Custom Helm release name | Auto-generated (`${USER}-test-<rand>`) |
| `-s, --sa <sa>` | Common | Kubernetes ServiceAccount | `vllm-sa` |
| `-R, --rate <rps>` | Common | Benchmark request rate in req/s | `10.0` |
| `-L, --prefix-len <N>` | Common / APC | Fixed shared prompt prefix length across requests (enables APC test) | `0` (pure random) |
| `-D, --duration <mins>` | Common | Maximum Kueue reservation duration in minutes | `90` |
| `-o, --output <file>` | Common | Save dynamically generated `values.yaml` to file (skips deployment) | `""` |
| `-v, --values-only` | Common | Print generated `values.yaml` to stdout without deploying | `false` |
| `-g, --dry-run` | Common | Render full Kubernetes JobSet manifest (`helm template`) to stdout | `false` |
| `-t, --topology <topo>` | Monolithic | TPU v7 topology shape (`2x2x1`, `2x2x4`, etc.) | `2x2x1` |
| `-P, --policy <name>` | Monolithic | GCE Compact Placement Policy name (auto-queried on multi-host) | `""` |
| `--async-scheduling` / `--no-async-scheduling` | Features | Enable/disable async CPU/TPU scheduling | `true` (enabled) |
| `--prefix-caching` / `--no-prefix-caching` | Features | Enable/disable Automatic Prefix Caching (APC) | `true` (enabled) |
| `--chunked-prefill` / `--no-chunked-prefill` | Features | Enable/disable chunked prefill | `true` (enabled) |
| `--max-batched-tokens <N>` | Features | Max tokens per chunked prefill iteration | `2048` |
| `--wide-ep, --ep` | Features | Enable Wide Expert Parallelism for MoE models | `false` |
| `--spec-decode, --draft-model <repo>` | Features | Enable Speculative Decoding (Eagle3) with draft model | `false` |
| `--spec-tokens <N>` | Features | Number of candidate tokens proposed per step | `3` |
| `--kv-offload, --no-kv-offload` | Features | Enable Host DRAM KV Cache Offloading | `false` |
| `--offload-connector <name>` | Features | KV Cache Offload connector (`RaidenOffloadConnector` / `TPUOffloadConnector`) | `RaidenOffloadConnector` |
| `--structured-output` | Features | Enable grammar-guided structured decoding (JSON schema) | `false` |
| `--disagg` | Disaggregated | Enable Disaggregated Prefill/Decode architecture | `false` |
| `-p, --prefill <topo>` | Disaggregated | Prefill TPU v7 topology | `2x2x1` |
| `-d, --decode <topo>` | Disaggregated | Decode TPU v7 topology | `2x2x2` |
| `-N, --decode-replicas <N>` | Disaggregated | Number of Decode slices to scale horizontally | `1` |
| `-c, --connector <name>` | Disaggregated | KV Connector: `TPUConnector` or `TPURaidenConnector` | Auto-detected |

### Container Image Quick Switches

The chart defaults to the official upstream `tpu-inference` image, but allows instant toggling between both TPU container environments:

```bash
# 1. Default: Upstream tpu-inference image (docker.io/vllm/vllm-tpu:v0.27.0)
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct

# 2. Quick switch to Google TorchTPU image (torchtpu-vllm-prod:latest)
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --torchtpu
# Or using the -i alias:
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -i torchtpu

# 3. Explicitly re-select tpu-inference
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --tpu-inference
# Or:
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -i tpu-inference
```

### Example Commands

```bash
# A/B Prefix Caching Benchmark:
# 1. Baseline (Prefix Caching disabled, full 1024 cold prefill every time):
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --no-prefix-caching -L 1024

# 2. Optimized (Prefix Caching enabled, reuses 1024 prefix tokens across all requests):
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --prefix-caching -L 1024

# 3. Run APC benchmark on Google's TorchTPU image:
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --torchtpu --prefix-caching -L 1024

# Monolithic serving with Speculative Decoding (Eagle3):
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --spec-decode --draft-model yuhuili/EAGLE-LLaMA3-Instruct-8B -b gs://my-bucket/hf-cache

# Disable prefix caching & chunked prefill for baseline comparison:
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --no-prefix-caching --no-chunked-prefill

# MoE model with Wide Expert Parallelism:
./gke/helm/run_benchmark.sh -t 2x2x4 -m Qwen/Qwen3.5-35B-A3B-FP8 --wide-ep

# Disaggregated serving with Async Scheduling & Chunked Prefill:
./gke/helm/run_benchmark.sh --disagg -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct --async-scheduling --chunked-prefill --max-batched-tokens 1024

# 1. Monolithic Single-host (TPU 2x2x1, 1 VM):
./gke/helm/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -b gs://my-bucket/hf-cache

# 2. Monolithic Multi-host (TPU 2x2x4, 4 VMs via Ray):
./gke/helm/run_benchmark.sh -t 2x2x4 -m meta-llama/Llama-3.1-70B-Instruct -b gs://my-bucket/hf-cache

# 3. Disaggregated Asymmetric Serving (Prefill 2x2x1, Decode 2x2x2 @ 35 RPS):
./gke/helm/run_benchmark.sh --disagg -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct -R 35.0 -b gs://my-bucket/hf-cache

# 4. Disaggregated Horizontal Decode Scaling (1x Prefill 2x2x1, 2x Decode 2x2x1 @ 35 RPS):
./gke/helm/run_benchmark.sh --disagg -p 2x2x1 -d 2x2x1 -N 2 -m meta-llama/Llama-3.1-8B-Instruct -R 35.0 -b gs://my-bucket/hf-cache

# 5. Generate and inspect values.yaml without deploying:
./gke/helm/run_benchmark.sh --disagg -p 2x2x4 -d 2x2x4 -N 2 -m meta-llama/Llama-3.1-70B-Instruct -v

# 6. Dry-run render full Kubernetes manifest (helm template):
./gke/helm/run_benchmark.sh --disagg -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct -g
```

---

## Helm CLI Workflow

### Install / Deploy

```bash
# 1. Deploy Asymmetric Disaggregated 8B Benchmark:
helm install my-disagg-run ./gke/helm -f ./gke/helm/values-disagg-llama8b.yaml

# 2. Deploy Multi-Host 70B Disaggregated Benchmark (96 chips):
helm install my-70b-disagg ./gke/helm -f ./gke/helm/values-disagg-llama70b.yaml

# 3. Deploy Monolithic 70B Multi-Host Benchmark:
helm install my-70b-mono ./gke/helm -f ./gke/helm/values-llama70b.yaml

# 4. Deploy with inline value overrides:
helm install my-run ./gke/helm \
  -f ./gke/helm/values-llama8b.yaml \
  --set benchmark.requestRate=50.0 \
  --set storage.type=ramdisk
```

### Render / Dry-Run (`helm template`)

To inspect the generated Kubernetes JobSet YAML without deploying:

```bash
# Render full manifest:
helm template my-test ./gke/helm -f ./gke/helm/values-disagg-llama70b.yaml

# Verify only JobSet resource:
helm template my-test ./gke/helm -f ./gke/helm/values-disagg-llama70b.yaml -s templates/jobset.yaml
```

### Uninstall / Teardown

```bash
# Uninstall release via Helm:
helm uninstall my-70b-disagg

# Or use the comprehensive cleanup script:
./gke/bin/cleanup.sh my-70b-disagg
```

---

## Buildkite CI Pipeline Converter (`buildkite_to_helm.py`)

Converts any Buildkite model pipeline YAML (from `.buildkite/models/*.yml`) into GKE TPU Helm `values.yaml` files, filtering steps that invoke `.buildkite/scripts/run_in_docker.sh` and mapping them into Kubernetes JobSet test runners.

### Features
- **Automatic Step Filtering**: Extracts only `run_in_docker.sh` steps (`UnitTest`, `Accuracy`, `Benchmark`), skipping non-containerized steps like `record_step_result.sh`.
- **Unified `scriptJobs` Architecture**: Always generates ReplicatedJobs under the `scriptJobs` array. Configured with `startupPolicyOrder: InOrder` and fail-fast `failurePolicy` so multi-job pipelines execute sequentially without resource race.
- **RFC 1123 Compliant Job Naming**: ReplicatedJob names are cleanly derived from the substring after the last `_` of the Buildkite step key (e.g. `benchmark`, `unittest`, `accuracy`), lowercased, and length-bounded to guarantee full compliance with Kubernetes DNS label and Pod naming limits.
- **Dynamic Accelerator Replacement**: Dynamically resolves `${TPU_VERSION:-...}` to the `--accelerator` parameter (defaults to `tpu7x`).
- **Target Step Key Filtering (`--step <step_key>`)**: Matches against target step keys, sanitized names, or stages with validation and provides a list of available steps if unmatched.
- **Base Values Inheritance**: Directly inherits `image.tpuInferenceCommit`, `image.vllmCommit`, storage, and secrets from the base values template.
- **Environment Variable Resolution**: Automatically parses Bash expansions (e.g. `${TENSOR_PARALLEL_SIZE_SINGLE:-1}`) and supports `--tensor-parallel-size` overrides.

### Usage Examples

```bash
# 1. Convert all qualifying steps in Buildkite pipeline to multi-job Helm values:
python3 gke/helm/buildkite_to_helm.py \
  -b /path/to/tpu-inference/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \
  -o gke/helm/values-llama8b-ci.yaml

# 2. Extract only a specific step by matching its step key:
python3 gke/helm/buildkite_to_helm.py \
  -b /path/to/tpu-inference/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \
  --step tpu7x_meta-llama_Llama-3_1-8B-Instruct_Benchmark \
  --tensor-parallel-size 2 \
  -o gke/helm/values-llama8b-ci.yaml

# 3. Preview generated YAML on stdout:
python3 gke/helm/buildkite_to_helm.py \
  -b /path/to/tpu-inference/.buildkite/models/meta-llama_Llama-3_1-8B-Instruct.yml \
  -v
```
