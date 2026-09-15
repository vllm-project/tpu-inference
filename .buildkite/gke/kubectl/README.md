# Direct Kubernetes JobSet Runners (`gke/kubectl/`)

Lightweight, template-driven automation runners that deploy **vLLM** inference and progressive multi-stage benchmarks directly to **Google Cloud TPUs (v7x)** via Kubernetes **JobSets** and **Kueue** using `kubectl apply` and `envsubst`.

Supports both **Monolithic Serving** ([`run_benchmark.sh`](./run_benchmark.sh)) and **Disaggregated Prefill/Decode Serving** ([`run_disagg.sh`](./run_disagg.sh)) without requiring Helm.

> For the unified **Helm chart implementation** (which combines both modes into a single parameterized chart with automatic placement policies and values catalogs), see [../helm/README.md](../helm/README.md).  
> For cluster diagnostic tools ([`bin/cluster_status.sh`](../bin/cluster_status.sh)), live workload monitoring ([`bin/job_status.sh`](../bin/job_status.sh)), log streaming ([`bin/tee_logs.sh`](../bin/tee_logs.sh)), and cleanup automation, see the parent [GKE Operations Guide](../README.md).

---

## Table of Contents

- [Directory Structure](#directory-structure)
- [Architectures Supported](#architectures-supported)
  - [1. Monolithic Mode (`run_benchmark.sh`)](#1-monolithic-mode-run_benchmarksh)
  - [2. Disaggregated Mode (`run_disagg.sh`)](#2-disaggregated-mode-run_disaggsh)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Monolithic Runner Usage & Options (`run_benchmark.sh`)](#monolithic-runner-usage--options-run_benchmarksh)
- [Disaggregated Runner Usage & Options (`run_disagg.sh`)](#disaggregated-runner-usage--options-run_disaggsh)
- [Container Image Quick Switches](#container-image-quick-switches)
- [Example Commands](#example-commands)
  - [Monolithic Single-Host Benchmark (2x2x1)](#monolithic-single-host-benchmark-2x2x1)
  - [Monolithic Multi-Host Benchmark via Ray (2x2x4)](#monolithic-multi-host-benchmark-via-ray-2x2x4)
  - [Disaggregated Asymmetric Serving (Prefill 2x2x1, Decode 2x2x2)](#disaggregated-asymmetric-serving-prefill-2x2x1-decode-2x2x2)
  - [Disaggregated Horizontal Decode Scaling (1x Prefill, 2x Decode)](#disaggregated-horizontal-decode-scaling-1x-prefill-2x-decode)
  - [Automatic Prefix Caching (APC) A/B Test](#automatic-prefix-caching-apc-ab-test)
  - [Chunked Prefill & Latency Tuning](#chunked-prefill--latency-tuning)
  - [Dry-Run & Manifest Inspection](#dry-run--manifest-inspection)
- [Monitoring & Teardown](#monitoring--teardown)

---

## Directory Structure

```text
gke/kubectl/
├── README.md                              # Direct Kubernetes runner documentation
├── run_benchmark.sh                       # Monolithic JobSet runner & hardware calculator
├── run_disagg.sh                          # Disaggregated (P/D) JobSet runner
├── benchmark_singlehost_template.yaml     # Single-host TPU template (2x2x1 / 2x2x2, 1 VM)
├── benchmark_multihost_template.yaml      # Multi-host TPU template (2x2x4+, N VMs via Ray)
├── benchmark_disagg_template.yaml         # Disaggregated TPU template (Prefill, Decode, Proxy)
└── benchmark_multihost_template.original.yaml
```

---

## Architectures Supported

### 1. Monolithic Mode (`run_benchmark.sh`)

In monolithic mode, both prefill and token generation execute inside the same vLLM instance on a single TPU slice:

* **Single-Host (`2x2x1`, `2x2x2`)**: Renders [`benchmark_singlehost_template.yaml`](./benchmark_singlehost_template.yaml) with 1 TPU VM (`TP=8`).
* **Multi-Host (`2x2x4`, `4x4x4`, `4x4x8`)**: Renders [`benchmark_multihost_template.yaml`](./benchmark_multihost_template.yaml) with $N$ worker VMs coordinated via a Ray Head/Worker cluster.
* **Client Job**: Deploys a dedicated CPU client running a progressive 5-stage token benchmark suite ($128 \to 3072$ tokens).

### 2. Disaggregated Mode (`run_disagg.sh`)

Separates prefill computation and decode token generation across independent, dedicated TPU slices linked via high-speed P2P KV connectors:

* **Prefill Slice (`replicatedJob: p`)**: High-throughput KV producer on Port 8400.
* **Decode Slice (`replicatedJob: d`)**: Low-latency KV consumer on Port 9400 (supports horizontal scaling via `-N`).
* **Proxy Router (`replicatedJob: x`)**: Dispatches user queries to prefill, directs decode steps, and handles client requests.
* **KV Connectors**: Auto-detects `TPUConnector` (upstream `vllm-tpu`) or `TPURaidenConnector` (Google `torchtpu-vllm-prod`).

---

## Prerequisites & Environment Setup

Before running benchmarks, ensure your local workstation or Cloud Shell environment is configured:

### 1. Install Required Tools

| Tool | Minimum Version | Description & Verification |
| :--- | :--- | :--- |
| **`kubectl`** | v1.26+ | Kubernetes CLI (`kubectl version --client`) |
| **`gcloud`** | Latest | Google Cloud SDK (`gcloud version`) |
| **`gke-gcloud-auth-plugin`** | Latest | Required GKE authentication plugin for `kubectl` |
| **`envsubst`** | Any | Variable interpolation utility (`envsubst --version`, part of `gettext-base`) |
| **`jq` & `bc`** | Any | Required for JSON processing and math |

**Quick Installation (Debian/Ubuntu/gLinux):**
```bash
gcloud components install kubectl gke-gcloud-auth-plugin
sudo apt-get update && sudo apt-get install -y gettext-base jq bc
```

### 2. Authenticate with Google Cloud

```bash
# 1. Authenticate user credentials
gcloud auth login

# 2. Authenticate Application Default Credentials (ADC) for GCS bucket storage access
gcloud auth application-default login

# 3. Set the active GCP project
gcloud config set project cloud-tpu-shared-capacity
```

### 3. Connect to the GKE TPU Cluster

```bash
# Obtain cluster credentials (example: bodaborg-tpu7x-nap in us-central1)
gcloud container clusters get-credentials bodaborg-tpu7x-nap \
  --region us-central1 \
  --project cloud-tpu-shared-capacity

# Verify cluster connection
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x
```

### 4. Configure Hugging Face Secret

For gated models (e.g. `meta-llama/Llama-3.1-8B-Instruct`), the scripts automatically mount a Kubernetes Secret named `${CLEAN_USER}-test-token` containing the `HF_TOKEN` key:

```bash
# Determine your clean username prefix (e.g. 'johndoe' -> 'johndoe-test-token')
CLEAN_USER=$(echo "${USER}" | tr '[:upper:]' '[:lower:]' | tr -dc 'a-z0-9')

# Create or update the secret
kubectl create secret generic "${CLEAN_USER}-test-token" \
  --from-literal=HF_TOKEN="hf_your_actual_token_here" \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## Monolithic Runner Usage & Options (`run_benchmark.sh`)

```bash
./gke/kubectl/run_benchmark.sh [OPTIONS]
```

| Flag | Category | Description | Default |
| :--- | :--- | :--- | :--- |
| `-m, --model <model>` | Common | Model name, HF repo ID, or `gs://` URI | `meta-llama/Llama-3.1-8B-Instruct` |
| `-i, --image <image>` | Common | Container image or alias (`torchtpu`, `tpu-inference`) | `docker.io/vllm/vllm-tpu:v0.27.0` |
| `--torchtpu` | Common | Quick switch to Google TorchTPU production image | `false` |
| `--tpu-inference` | Common | Quick switch to upstream `tpu-inference` image | `true` (default) |
| `-t, --topology <topo>` | Hardware | TPU v7 topology shape (`2x2x1`, `2x2x4`, etc.) | `2x2x1` |
| `-b, --bucket <bucket>` | Storage | GCS bucket path for caching (e.g. `gs://<bucket>/hf-cache`) | `""` (RAM-Disk mode) |
| `-s, --sa <sa>` | Security | Kubernetes ServiceAccount | `vllm-sa` |
| `-R, --rate <rps>` | Traffic | Benchmark request rate in req/s | `10.0` |
| `-L, --prefix-len <N>` | APC | Shared prompt prefix length across requests (enables APC test) | `0` (pure random) |
| `-P, --policy <name>` | Placement | GCE Compact Placement Policy name (multi-host) | `""` |
| `-D, --duration <mins>` | Scheduling | Kueue reservation duration in minutes | `90` |
| `-d, --dry-run` | Execution | Render full Kubernetes JobSet YAML to stdout without deploying | `false` |
| `-o, --output <file>` | Execution | Save generated YAML to file without deploying | `""` |
| `--async-scheduling` / `--no-async-scheduling` | Features | Enable/disable async CPU/TPU scheduling | `true` (enabled) |
| `--prefix-caching` / `--no-prefix-caching` | Features | Enable/disable Automatic Prefix Caching (APC) | `true` (enabled) |
| `--chunked-prefill` / `--no-chunked-prefill` | Features | Enable/disable chunked prefill | `true` (enabled) |
| `--max-batched-tokens <N>` | Features | Max tokens per chunked prefill iteration | `2048` |

---

## Disaggregated Runner Usage & Options (`run_disagg.sh`)

```bash
./gke/kubectl/run_disagg.sh [OPTIONS]
```

| Flag | Category | Description | Default |
| :--- | :--- | :--- | :--- |
| `-m <model>` | Common | Model name from Hugging Face or GCS | `meta-llama/Llama-3.1-8B-Instruct` |
| `-p <topology>` | Hardware | Prefill slice TPU v7 topology | `2x2x1` |
| `-d <topology>` | Hardware | Decode slice TPU v7 topology | `2x2x2` |
| `-N, --decode-replicas <N>` | Hardware | Number of Decode slices to scale horizontally | `1` |
| `-c <connector>` | KV Transfer | KV Connector (`TPUConnector` or `TPURaidenConnector`) | Auto-detected |
| `-i <image>` | Container | Docker container image | `docker.io/vllm/vllm-tpu:v0.27.0` |
| `-b <bucket>` | Storage | GCS bucket path for caching | `""` (RAM-Disk mode) |
| `-s <sa>` | Security | Kubernetes ServiceAccount name | `vllm-sa` |
| `-r <rps>` | Traffic | Benchmark request rate in req/s | `10.0` |
| `-D <mins>` | Scheduling | Kueue reservation duration in minutes | `90` |
| `-g, --dry-run` | Execution | Render full Kubernetes JobSet YAML without deploying | `false` |
| `-o <file>` | Execution | Output filepath to save the rendered YAML | `""` |

---

## Container Image Quick Switches

The runners default to the upstream `tpu-inference` image (`docker.io/vllm/vllm-tpu:v0.27.0`), and allow quick switches between environments:

```bash
# 1. Run with default upstream tpu-inference image:
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct

# 2. Quick switch to Google TorchTPU production image (torchtpu-vllm-prod:latest):
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --torchtpu
# Or using the -i alias:
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -i torchtpu

# 3. Explicitly re-select upstream tpu-inference:
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct --tpu-inference
# Or:
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -i tpu-inference
```

---

## Example Commands

### Monolithic Single-Host Benchmark (2x2x1)

Deploy an 8B model on a single `2x2x1` VM (4 chips, TP=8) using RAM-disk storage:

```bash
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct
```

### Monolithic Multi-Host Benchmark via Ray (2x2x4)

Deploy a 70B model across a multi-host slice (4 VMs, 16 chips via Ray) with GCS weight caching:

```bash
./gke/kubectl/run_benchmark.sh -t 2x2x4 -m meta-llama/Llama-3.1-70B-Instruct -b gs://my-bucket/hf-cache
```

### Disaggregated Asymmetric Serving (Prefill 2x2x1, Decode 2x2x2)

Deploy disaggregated serving with 1 Prefill VM and 2 Decode VMs @ 35 req/s:

```bash
./gke/kubectl/run_disagg.sh -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct -r 35.0 -b gs://my-bucket/hf-cache
```

### Disaggregated Horizontal Decode Scaling (1x Prefill, 2x Decode)

Deploy 1 Prefill slice (`2x2x1`) and 2 independent Decode slices (`2x2x1` each) behind the proxy router:

```bash
./gke/kubectl/run_disagg.sh -p 2x2x1 -d 2x2x1 -N 2 -m meta-llama/Llama-3.1-8B-Instruct -r 35.0 -b gs://my-bucket/hf-cache
```

### Automatic Prefix Caching (APC) A/B Test

Compare cold prompt prefill versus warm prefix reuse across identical traffic:

```bash
# 1. Baseline without caching (cold prefill of all 1024 prompt tokens every time):
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct \
  --no-prefix-caching -L 1024

# 2. Optimized with Prefix Caching (reusing 1024 prefix tokens across all requests):
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct \
  --prefix-caching -L 1024
```

### Chunked Prefill & Latency Tuning

Evaluate TTFT vs TPOT trade-offs:

```bash
# Monolithic un-chunked baseline (lowest TTFT for long prompts):
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct \
  --no-chunked-prefill --no-prefix-caching

# Chunked prefill with custom token budget (protects decode stream latency):
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct \
  --chunked-prefill --max-batched-tokens 1024
```

### Dry-Run & Manifest Inspection

Inspect the interpolated YAML manifest without applying to the cluster:

```bash
# Monolithic dry-run:
./gke/kubectl/run_benchmark.sh -t 2x2x1 -m meta-llama/Llama-3.1-8B-Instruct -d

# Disaggregated dry-run:
./gke/kubectl/run_disagg.sh -p 2x2x1 -d 2x2x2 -m meta-llama/Llama-3.1-8B-Instruct -g -o disagg-job.yaml
```

---

## Monitoring & Teardown

After deploying with either runner, monitor execution using the parent GKE toolkit:

```bash
# 1. Watch pods spinning up:
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=<JOB_NAME> -w

# 2. Deep diagnostic status inspection:
./gke/bin/job_status.sh <JOB_NAME> -w

# 3. Stream & tee all component logs:
./gke/bin/tee_logs.sh <JOB_NAME> [LOG_SUFFIX]

# 4. Stream Prefill logs (Disaggregated):
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<JOB_NAME>,jobset.sigs.k8s.io/replicatedjob-name=p -c vllm-tpu -f

# 5. Stream Decode logs (Disaggregated):
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<JOB_NAME>,jobset.sigs.k8s.io/replicatedjob-name=d -c vllm-tpu -f

# 6. Stream Monolithic server logs:
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<JOB_NAME>,jobset.sigs.k8s.io/replicatedjob-name=server -f

# 7. Stream benchmark client results:
kubectl logs -l jobset.sigs.k8s.io/jobset-name=<JOB_NAME>,jobset.sigs.k8s.io/replicatedjob-name=client -f

# 8. Teardown & cleanup:
./gke/bin/cleanup.sh <JOB_NAME>
```
