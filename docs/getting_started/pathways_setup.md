# Pathways Setup Guide

## What is Pathways?

[Pathways](https://github.com/AI-Hypercomputer/pathways-utils) is a distributed runtime developed by Google that enables efficient execution of machine learning workloads across one or multiple TPU slices. Pathways decouples the client (your Python program) from the TPU accelerators by introducing a resource manager and a proxy-based communication layer. This allows:

- **Multi-slice scaling:** Seamlessly scale your workloads across multiple TPU slices without modifying your model code.
- **Improved resource utilization:** Pathways manages TPU resources centrally, enabling better scheduling and fault tolerance.
- **Transparent JAX integration:** Your JAX programs communicate with TPU hardware through a gRPC proxy, requiring only minimal environment variable configuration.

Pathways is particularly useful for large-scale inference and training workloads that need to span more TPU chips than a single slice provides.

---

## Prerequisites

- Access to a Google Cloud project with TPU quota.
- [xpk](https://github.com/AI-Hypercomputer/xpk) CLI installed and configured.

---

## Usage

### 1. Create a Pathways Cluster

Before launching Pathways workloads, you need a GKE cluster with the appropriate TPU node pools. Use `xpk` to create one:

```bash
xpk cluster create-pathways \
  --cluster <CLUSTER_NAME> \
  --tpu-type=<TPU_TYPE> \
  --num-slices=<NUM_SLICES> \
  --zone=<ZONE> \
  --project=<PROJECT_ID>
```

**Example:**

```bash
xpk cluster create-pathways \
  --cluster my-v5p-cluster \
  --tpu-type=v5p-128 \
  --num-slices=1 \
  --zone=europe-west4-b \
  --project=my-gcp-project
```

### 2. Prepare a Script Directory

Pathways workloads use `--script-dir` to upload a local directory of scripts and files into the workload container. This directory should contain your launch script and any additional files needed at runtime.

Create a script directory with a launch script (e.g., `run_pathways.sh`):

```
script_dir/
├── run_pathways.sh          # Entry-point script
├── offline_inference.py     # Your inference script (or other workload files)
└── ...                      # Any other files your workload needs
```

A default `run_pathways.sh` template is provided in [`scripts/pathways/run_pathways.sh`](../../scripts/pathways/run_pathways.sh). Copy and customize it for your workload:

```bash
cp scripts/pathways/run_pathways.sh my_script_dir/run_pathways.sh
```

> **Note:** Only files inside your `script_dir` are uploaded to the container. If your launch script references other files (e.g., `offline_inference.py`), you must copy them into the `script_dir` as well:
>
> ```bash
> cp examples/offline_inference.py my_script_dir/
> ```

See [run_pathways.sh](#run_pathwayssh-template) below for the full template.

### 3. Launch a Pathways Workload

Use `xpk workload create-pathways` to submit your inference or training job. The `--base-docker-image` flag specifies the base container image, and `--script-dir` points to your local directory of scripts to be uploaded into the container:

```bash
xpk workload create-pathways \
  --workload <WORKLOAD_NAME> \
  --base-docker-image <DOCKER_IMAGE> \
  --script-dir <PATH_TO_SCRIPT_DIR> \
  --cluster <CLUSTER_NAME> \
  --tpu-type=<TPU_TYPE> \
  --num-slices=<NUM_SLICES> \
  --zone=<ZONE> \
  --project=<PROJECT_ID> \
  --priority=<PRIORITY> \
  --command "bash ./run_pathways.sh"
```

**Example:**

```bash
xpk workload create-pathways \
  --workload my-workload \
  --base-docker-image vllm/vllm-tpu:latest \
  --script-dir ./script_dir \
  --cluster my-v5p-cluster \
  --tpu-type=v5p-128 \
  --num-slices=1 \
  --zone=europe-west4-b \
  --project=my-gcp-project \
  --priority=very-high \
  --command "bash ./run_pathways.sh"
```

**Expected output:**

On successful submission you will see output similar to:

```
[XPK] Follow your Pathways workload and other resources here : https://console.cloud.google.com/logs/query;query=resource.type%3D"k8s_container"%0Aresource.labels.project_id%3D"<PROJECT_ID>"%0Aresource.labels.location%3D"<REGION>"%0Aresource.labels.cluster_name%3D"<CLUSTER_NAME>"%0Aresource.labels.pod_name:"<WORKLOAD_NAME>-"%0Aseverity>%3DDEFAULT
[XPK] Exiting XPK cleanly
```

Click the link to open **Cloud Logging** and view your workload logs. To see JAX-specific logs, add the filter `default/jax-tpu` in the Logs Explorer.

#### Required Environment Variables

When running with Pathways, you **must** set the following environment variables so that JAX communicates through the Pathways proxy instead of directly accessing TPU devices:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_PLATFORMS=proxy,cpu
export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000
```

These are already included in the default `run_pathways.sh` template (see below).

#### Multi-Slice Configuration

If you are running across **multiple slices**, you must also set the `NUM_SLICES` environment variable to the number of slices:

```bash
export NUM_SLICES=<NUMBER_OF_SLICES>
```

For example, for a 2-slice workload:

```bash
NUM_SLICES=2 \
JAX_PLATFORMS=proxy,cpu \
JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 \
python examples/offline_inference.py
```

### 4. Delete a Workload

Once your workload is complete (or you want to cancel it), clean up with:

```bash
xpk workload delete \
  --workload <WORKLOAD_NAME> \
  --cluster <CLUSTER_NAME> \
  --zone=<ZONE> \
  --project=<PROJECT_ID>
```

**Example:**

```bash
xpk workload delete \
  --workload my-workload \
  --cluster my-v5p-cluster \
  --zone=europe-west4-b \
  --project=my-gcp-project
```

---

## Monitoring & Debugging

Use `kubectl` to inspect the status of your workload:

- **List all pods** to verify your workload has been submitted:

  ```bash
  kubectl get pods
  ```

- **Describe a specific pod** to check detailed status, events, and any errors:

  ```bash
  kubectl describe pod <POD_NAME>
  ```

- **View logs** from a running or completed pod:

  ```bash
  kubectl logs <POD_NAME>
  ```

---

## `run_pathways.sh` Template

The default launch script template is located at [`scripts/pathways/run_pathways.sh`](../../scripts/pathways/run_pathways.sh). It sets the required environment variables and runs an offline inference example:

```bash
#!/bin/bash
# Default Pathways launch script for vLLM TPU inference.
# Copy this file into your script_dir and customize as needed.

# Required: Authenticate with Hugging Face to download gated models.
# Set HF_TOKEN in your environment or replace below.
export HF_TOKEN="${HF_TOKEN}"

# Required: Pathways environment variables.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export JAX_PLATFORMS=proxy,cpu
export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000

# Optional: Set NUM_SLICES for multi-slice workloads.
# export NUM_SLICES=2

# Run your workload.
python ./offline_inference.py "$@"
```

Customize the script to point to your own inference or serving entry point, add additional `pip install` commands, or set model-specific flags.

---

## Further Reading

- [xpk Documentation](https://github.com/AI-Hypercomputer/xpk)
- [Cloud TPU Pathways Overview](https://cloud.google.com/tpu/docs/pathways-overview)
- [vLLM TPU Quickstart](quickstart.md)
