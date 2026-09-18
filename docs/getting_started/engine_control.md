# TPU Engine Lifecycle Control & In-Band Draining (`tpu-ctl`)

When operating TPU inference workloads under container checkpoint/restore mechanisms (such as **GKE Pod Snapshots / GPS**, CRIU, or live host migration), pausing the workload must satisfy two strict hardware invariants:

1. **Hardware Pipeline Quiescence**: All in-flight tensor matrix multiplications, DMA operations, and inter-chip ICI collective communications must complete so the hardware reaches an idle, consistent boundary.
2. **Zero-Loss HBM Retention**: Model weights and prefix KV-cache blocks must **never be flushed or deallocated** from TPU High-Bandwidth Memory (HBM). Flushing HBM forces multi-gigabyte checkpoint reloading from GCS on resumption, defeating the fast-warmup purpose of container snapshotting.

---

## 1. Enabling Control Endpoints

Engine control endpoints are administrative and disabled by default. To enable them, set the environment variable:

```bash
export ENABLE_ENGINE_CONTROL_ENDPOINTS=true
export CONTROL_DRAIN_TIMEOUT_SECONDS=60 # Optional, default 60s
```

When enabled, the serving server exposes administrative endpoints under `/ctl`:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/ctl/pause` | `POST` | Halts new scheduling, drains in-flight requests, synchronizes TPU hardware, and keeps HBM intact. |
| `/ctl/resume` | `POST` | Resumes scheduling with **0ms warmup** and **zero weight reloading**. |
| `/ctl/status` | `GET` | Returns engine status (`SERVING` or `PAUSED`), queue depth, and HBM retention status. |
| `/ctl/health/ready` | `GET` | Returns `200 OK` while serving; returns `503 Service Unavailable` while paused to drain K8s ingress routing. |
| `/ctl/health/live` | `GET` | Returns `200 OK` process liveness check. |

---

## 2. In-Band Draining Mechanism (`clear_cache=False`)

Under `tpu-inference`, pausing uses non-destructive scheduling suspension:

```python
await engine.pause_generation(mode="keep", clear_cache=False)
await engine.model_executor.synchronize_device()
```

- **`mode="keep"`**: Preserves in-flight requests and token sequences in memory. Requests submitted while paused are buffered in the waiting queue rather than aborted.
- **`clear_cache=False`**: Prevents flushing the KV-cache and model parameters from TPU HBM.
- **`synchronize_device()`**: Invokes `jax.effects_barrier()` to block until all active DMA and ICI torus transfers settle.

---

## 3. Operator CLI: `tpu-ctl`

The `tpu-ctl` utility is installed directly with `tpu-inference`:

```bash
pip install -e .
```

### Multi-Host Coordination
In multi-host TPU topologies (e.g. 4 hosts with 4 chips each = 16 chips in a $4 \times 4$ torus):
- **Worker Pods (Rank 1..N)**: Automatically detect their rank from `TPU_WORKER_ID`, `JOB_COMPLETION_INDEX`, or hostname and perform a **silent local no-op (`return 0`)**.
- **Master Pod (Rank 0)**: Dispatches HTTP calls to `/ctl/pause` and `/ctl/resume`, ensuring coordinated execution across the cluster.

### CLI Commands:
```bash
# 1. Inspect engine status and queue metrics
tpu-ctl status

# 2. Pause serving before container snapshot / host migration
tpu-ctl pause --timeout=60

# 3. Resume serving after snapshot / restore
tpu-ctl resume

# 4. Kubernetes probe evaluation
tpu-ctl probe --type=readiness
tpu-ctl probe --type=liveness
```
