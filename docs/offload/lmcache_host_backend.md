# LMCache host backend for TPUOffloadConnector

Use [LMCache](https://github.com/lmcache/lmcache) as a persistent, tiered host-side
KV storage layer **behind** the native `TPUOffloadConnector`, without rewriting
LMCache's CUDA/XPU/HPU GPU connectors for JAX.

## Why this design

`TPUOffloadConnector` already owns the TPU-specific hard part: moving KV blocks
between HBM (JAX device arrays) and host memory, staging/scatter/gather, and the
scheduler-side content hashing (`LRUCacheManager`). Its host store is
`LocalCPUBackend` — a bounded in-memory `OrderedDict`.

This integration keeps that exact store interface but backs it with LMCache:

```
vLLM engine
  └─ TPUOffloadConnector           (owns HBM<->host JAX-array movement)
       └─ host store               (add/get/reclaim by chunk_id)
            ├─ hot CPU tier         (OrderedDict, == LocalCPUBackend)
            └─ LMCache tier         (disk today; remote/P2P/cross-instance next)
                 via kv_bridge (jax.Array <-> raw bytes, bit-exact incl. bf16)
```

Two alternatives were rejected:
1. **LMCache as a TPU `KVConnector`** — `tpu_platform.py` hard-allowlists the
   connector to `{TPUConnector, TPUConnectorHMA, TPUOffloadConnector,
   RaidenOffloadConnector}`, and LMCache's device detection has no `torch_xla`
   branch (all its connectors move torch tensors out of paged CUDA/XPU/HPU
   memory). TPU KV lives as JAX arrays in an XLA mesh.
2. Reimplementing LMCache's GPU connector for JAX — large surface, duplicates
   the movement `TPUOffloadConnector` already does correctly.

## Enabling it

All flags are off by default → **identical behavior to upstream** (stock
`LocalCPUBackend`).

| Env var | Default | Meaning |
|---|---|---|
| `TPU_OFFLOAD_LMCACHE` | `0` | `1` enables the LMCache host backend |
| `TPU_OFFLOAD_LMCACHE_BACKEND` | `file` | `file` / `memory` (reference) or `lmcache` (real StorageManager tiers) |
| `TPU_OFFLOAD_LMCACHE_HOT_CHUNKS` | `0` | Host-RAM hot-tier capacity in chunks; `0` = `TPU_OFFLOAD_NUM_CPU_CHUNKS` |
| `TPU_OFFLOAD_LMCACHE_PATH` | `/tmp/tpu_lmcache_kv` | Filesystem root for `file` / `lmcache` disk tier |

Example (single-host disk spill, no LMCache install required):

```bash
export TPU_OFFLOAD_LMCACHE=1
export TPU_OFFLOAD_LMCACHE_BACKEND=file
export TPU_OFFLOAD_LMCACHE_HOT_CHUNKS=2048
vllm serve <model> \
  --kv-transfer-config '{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector.TPUOffloadConnector","kv_role":"kv_both"}'
```

For the real LMCache tier (`TPU_OFFLOAD_LMCACHE_BACKEND=lmcache`), install the
LMCache fork with the TPU integration (`lmcache.integration.tpu`).

## Components

| File | Role |
|---|---|
| `tpu_inference/offload/kv_bridge.py` | Bit-exact `jax.Array` (host) ⟷ raw bytes, incl. bfloat16 |
| `tpu_inference/offload/lmcache_host_backend.py` | `LMCacheHostBackend`: drop-in for `LocalCPUBackend` + spill tier + Depth-2 content-hash hook |
| `tpu_inference/offload/lmcache_kv_store.py` | Raw-bytes KV store protocol + reference `File`/`Memory` stores |
| `tpu_inference/offload/host_backend_factory.py` | Env-gated selection (default = stock `LocalCPUBackend`) |

## Correctness

- The value bridge is **bit-exact** for float32/float16/bfloat16/int8/int32
  (raw byte view; no lossy cast — bfloat16 survives via `ml_dtypes`).
- `LMCacheHostBackend` preserves `LocalCPUBackend`'s `add`/`get`/
  `reclaim_unoccupied_chunks`/`num_saved_cpu_chunks` contract, including the
  invalid-`chunk_id` `ValueError`.
- Tests (`tests/offload/kv_bridge_test.py`,
  `tests/offload/lmcache_host_backend_test.py`) run on `jax[cpu]` — no TPU.

## Roadmap

- **Depth 1 (this change):** persistent disk spill keyed by `chunk_id`. KV
  survives beyond hot-tier RAM.
- **Depth 2:** thread the scheduler's `BlockHash` to the worker
  (`set_chunk_hash_hint` is the hook) → content-addressed `CacheEngineKey` →
  cross-replica prefix sharing (e.g. a DP rollout fleet), remote backends
  (Redis / Mooncake / NIXL), and reuse across restarts.
- **Depth 3:** CacheBlend on TPU (out of scope; blend forward is separate work).

## Validated on real hardware

Bit-for-bit correctness confirmed on a **tpu7x 2x2x1** slice via
`examples/offload/offline_inference_kv_cache_verification.py`:
- stock (`TPU_OFFLOAD_LMCACHE=0`) → all runs passed, exit 0;
- LMCache enabled (`backend=file`, `hot_chunks=2`, forced spill) → all runs passed,
  exit 0, backend confirmed live in the worker log;
- direct probe: 5 chunks / hot=2 → 3 evicted + reloaded from disk, all bit-exact
  (incl. bfloat16).
