# TPU KV Cache Host Offload

The `TPUOffloadConnector` extends vLLM's KV-cache prefix cache into host (CPU)
DRAM, so blocks evicted from HBM can be restored on a future prefix-cache hit
instead of being recomputed.

It supports both **non-hybrid** models (Qwen3, Llama, ...) and **hybrid
attention + Mamba/GDN** models (Qwen3.5, ...) via the
[`SupportsHMA`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
interface — for hybrid models, each saved chunk carries both the attention
block payload AND the request's mamba state, and both are scattered back on
load.

## Quick start

### Non-hybrid (Qwen3, Llama, …)

```bash
python examples/offline_inference.py \
  --model Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-model-len 1024 --max-num-batched-tokens 1024 --block-size 128 \
  --kv-transfer-config '{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector","kv_role":"kv_both"}'
```

### Hybrid attention+Mamba (Qwen3.5)

Two extra flags are required so vLLM does not auto-disable the hybrid KV cache
manager when the connector is present, plus two env vars to bypass the
per-bucket swap precompile (which OOMs on hybrid models).

```bash
SKIP_JAX_PRECOMPILE=1 TPU_OFFLOAD_SKIP_JAX_PRECOMPILE=1 \
python examples/offline_inference.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 5120 --max-num-batched-tokens 16384 --block-size 256 \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector","kv_role":"kv_both"}'
```

To exercise real D2H/H2D events on Qwen3.5 (i.e. an actual prefix-cache hit
restoring blocks from CPU), add `--enable-prefix-caching` and the
`TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY=1` escape hatch (see
[Known limitations](#known-limitations) for why this is needed).

```bash
SKIP_JAX_PRECOMPILE=1 TPU_OFFLOAD_SKIP_JAX_PRECOMPILE=1 \
TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY=1 \
python examples/offline_inference.py \
  --model Qwen/Qwen3.5-35B-A3B-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 5120 --max-num-batched-tokens 16384 --block-size 256 \
  --no-disable-hybrid-kv-cache-manager --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"TPUOffloadConnector","kv_connector_module_path":"tpu_inference.offload.tpu_offload_connector","kv_role":"kv_both"}'
```

For a 397B-class hybrid model on a v6e/v7-8 host, set `--tensor-parallel-size 8`.

## Verify on your machine

`verify_offload.sh` runs three sanity checks against the no-connector baseline
and diffs the generated text. All three passing means the connector handles
both non-hybrid and hybrid (attn + mamba) end-to-end.

```bash
bash examples/offload/verify_offload.sh
```

The three cases are:
1. **Qwen3-0.6B (non-hybrid)** — attn-only fast path; bit-exact vs baseline.
2. **Qwen3.5-35B (hybrid, no prefix caching)** — HMA mode; bit-exact vs baseline.
3. **Qwen3.5-35B + prefix caching** — HMA mode; chunks carry mamba state
   alongside attention; bit-exact vs baseline (proves the per-group save/load
   round-trip preserves correctness).

## Configuration

Tunables (env vars, all read in `tpu_inference/envs.py`):

| Env var | Default | Meaning |
|---|---|---|
| `TPU_OFFLOAD_NUM_CPU_CHUNKS` | 1024 | LRU cache size in chunks (each chunk = 1 attn block + per-request mamba state for hybrid). |
| `TPU_OFFLOAD_NUM_STAGING_BLOCKS` | 128 | HBM staging buffer size (in blocks) for the in-flight D2H/H2D pipeline. |
| `TPU_OFFLOAD_SAVE_THREADS` | 4 | Worker thread pool for async D2H. |
| `TPU_OFFLOAD_DECODE_SAVE` | False | Save during decode in addition to prefill. |
| `TPU_OFFLOAD_BATCHED_SAVE` | False | Use batched (multi-request) save path. |
| `TPU_OFFLOAD_SKIP_JAX_PRECOMPILE` | False | Skip per-bucket swap kernel precompile. **Required for hybrid models on TPU** (precompile OOMs otherwise). |
| `TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY` | False | Override the TPU platform's force-disable of `chunked_mm_input` so prefix caching can be enabled on hybrid Qwen3.5 (which is a multimodal class but used text-only here). See [Known limitations](#known-limitations). |

The `--kv-transfer-config` JSON also accepts the standard vLLM
`KVConnectorBase_V1` fields:
- `kv_connector`: must be `"TPUOffloadConnector"`
- `kv_connector_module_path`: `"tpu_inference.offload.tpu_offload_connector"`
- `kv_role`: `"kv_both"` for combined producer/consumer (the only mode this
  connector currently exposes; pure-`kv_consumer` would require pairing with
  another producer).

## How it works

### Non-hybrid path (single attention group)

- **Save**: when vLLM evicts attn blocks (or hits the save watermark during
  prefill / `decode_save`), the worker uses
  `stack_kv_cache_cross_layers` to gather the chosen blocks across all
  attention layers into a contiguous HBM staging buffer, D2Hs to a pinned-host
  array per chunk, and registers the chunks with `LocalCPUBackend` (an
  LRU-evicting `OrderedDict[chunk_id → list[jax.Array]]`).
- **Load**: on a prefix-cache hit, the scheduler asks `LocalCPUBackend` for
  the chunks, the worker H2Ds them onto the device, and `update_kv_caches_one`
  scatters them into the destination attention blocks vLLM allocated.
- This path is what
  [`examples/offload/gke/benchmarks/deploy-cpu-offload.yaml`](gke/benchmarks/deploy-cpu-offload.yaml)
  exercises for production serving (e.g.
  `Qwen3-235B-A22B-Instruct-2507-FP8`).

### Hybrid path (attn + mamba groups)

The connector inherits `SupportsHMA` and overrides `request_finished_all_groups`,
so vLLM passes the full per-group block IDs (`tuple[list[int], ...]`) on every
hook. Per-group block IDs are tracked end-to-end:

- `RequestTracker.block_ids_per_group: list[list[int]]`
- `SaveSpec.src_block_ids_per_group` / `LoadSpec.dst_block_ids_per_group`
- `TPUReqMeta.local_block_ids_per_group`

On the worker side:
- `_init_hma_tracking(...)` discovers `num_groups`, `group_is_mamba`,
  `layer_to_group_id`, `_kv_cache_treedef`, plus per-array host/device
  shardings for the mamba leaves.
- During **save**, after the existing attention gather,
  `_gather_mamba_state_for_request(...)` reads the request's singleton
  mamba slot (1 block per mamba group), and the D2H step packs each chunk's
  CPU value as `(attn_chunk, mamba_arrs_list)`. The same mamba state is
  replicated across all of that request's chunks (mamba is request-scoped,
  not block-scoped — see [Known limitations](#known-limitations)).
- During **load**, when the cached chunk is a `(attn, mamba)` tuple,
  `_scatter_mamba_state_for_request(...)` writes the mamba slices into the
  request's destination mamba blocks via a JIT'd
  `state.at[block_ids].set(slice)` per layer (handles `(conv, ssm)` tuples
  for layers like Qwen3.5 GatedDeltaNet).

### Comparison with vLLM main GPU connectors

| | GPU `OffloadingConnector` | GPU `SimpleCPUOffloadConnector` | TPU `TPUOffloadConnector` |
|---|---|---|---|
| Supports `SupportsHMA` | ❌ | ✅ | ✅ |
| Multi-group worker register | ✅ (#37853) | ✅ | ✅ |
| Multi-group lookup | ✅ (#39401) | ✅ | ✅ |
| Multi-group **load** | ✅ (#39402) | ✅ | ✅ |
| Multi-group **store** | ❌ ([#39403](https://github.com/vllm-project/vllm/pull/39403) open; "does not support sliding window/SSMs") | ✅ | ✅ |
| Mamba state transfer | ❌ ([#38261](https://github.com/vllm-project/vllm/pull/38261) open) | ✅ (CPU mirror) | ✅ (chunk-keyed; per-request mamba in chunk payload) |
| Hybrid model startup | ❌ asserts single-group at scheduler.py:212/277/319/336 → crashes | ✅ | ✅ |

## Outstanding TODOs (search for `TODO(mamba-alignment)` in code)

These are tracked in-source so they're discoverable from the call sites:

- `tpu_inference/offload/tpu_offload_connector.py` —
  `_gather_mamba_state_for_request` and the chunk-packing site in
  `_transfer_and_register_cpu_chunks` both note that once upstream
  `Qwen3NextForCausalLM` declares `SupportsMambaPrefixCaching` (vllm PR
  [#38261](https://github.com/vllm-project/vllm/pull/38261)), we should
  store per-attn-block mamba state in each chunk rather than the
  end-of-prefix replicated state.
- `tpu_inference/platforms/tpu_platform.py` — the
  `TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY` escape hatch should be
  re-evaluated once the upstream mamba prefix-cache fix lands.

When the upstream PRs merge, run `git grep TODO\(mamba-alignment\)` to find
every site that needs updating.

## Known limitations

1. **Qwen3.5 prefix-cache hits are gated by upstream `Qwen3Next` lacking
   `SupportsMambaPrefixCaching`.** vLLM's hybrid coordinator
   ([`kv_cache_coordinator.py:497–529`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_coordinator.py))
   reduces the candidate hit length monotonically per group, so a
   non-trivial prefix-cache hit fires only when *every* group has coverage.
   Today, `Qwen3NextForCausalLM` raises `NotImplementedError` on
   `mamba_cache_mode="all"`, forcing `"align"` mode, which only checkpoints
   mamba state at LCM(group block sizes) boundaries — for Qwen3.5 with
   attn block size 256 and mamba effectively `max_model_len`-scoped, that
   means cross-request hits really only fire for repeated identical-length
   prompts. The TPU connector already transfers mamba state correctly on
   every save/load, so once the upstream model class declares
   `SupportsMambaPrefixCaching` (its `gated_delta_net_state_copy_func` is
   identical to `Mamba2`'s, which already supports `"all"` mode) the
   speedup arrives with no further connector changes.

2. **`TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY=1` is needed for Qwen3.5 +
   prefix caching.** Without it, vLLM's `mamba_cache_mode="align"` codepath
   asserts `chunked_mm_input` is on, but the TPU platform force-disables
   this for any model the registry classifies as multimodal (Qwen3.5 is
   multimodal-capable even when used text-only). The env var is an opt-in
   override.

3. **Mamba state is replicated across all of a request's chunks.** Because
   vLLM exposes only 1 mamba block per request to the connector, every
   chunk this request saves carries the *current* mamba state at save
   time. On load, the last chunk's mamba state wins — which is correct
   for end-of-prefix matches, but for partial prefix matches the mamba
   state restored will correspond to the original save's prefix endpoint,
   not the matched length. With `mamba_cache_mode="all"` upstream, vLLM
   would update mamba state per-block boundary and this behavior becomes
   exact at every save event.

4. **TPU XLA cannot do slice/scatter on pinned-host arrays.** All host
   storage is per-block (one chunk per block), not contiguous. The chunk
   pool is a Python `OrderedDict`, not a vLLM `BlockPool` mirror.

5. **Per-bucket swap kernel precompile OOMs on hybrid models.** Set
   `SKIP_JAX_PRECOMPILE=1 TPU_OFFLOAD_SKIP_JAX_PRECOMPILE=1` for any
   hybrid run; the connector falls back to per-call compilation, which
   pays a small first-iteration latency cost but does not OOM.

## Files

- [`tpu_inference/offload/tpu_offload_connector.py`](../../tpu_inference/offload/tpu_offload_connector.py) — the connector (scheduler + worker), `SupportsHMA` mixin, per-group transfer.
- [`tpu_inference/offload/utils.py`](../../tpu_inference/offload/utils.py) — JIT'd transfer kernels including `gather_kv_blocks_per_group` / `scatter_kv_blocks_per_group` and the mamba-aware scatter.
- [`tpu_inference/offload/cpu_backend.py`](../../tpu_inference/offload/cpu_backend.py) — `LocalCPUBackend` (LRU `OrderedDict`).
- [`tpu_inference/offload/offload_manager.py`](../../tpu_inference/offload/offload_manager.py) — chunk allocation + hash-keyed index.
- [`tpu_inference/offload/metrics.py`](../../tpu_inference/offload/metrics.py) — OTel metrics for D2H/H2D bytes, latency, hit rate.
- [`tpu_inference/envs.py`](../../tpu_inference/envs.py) — `TPU_OFFLOAD_*` env vars.
- [`tpu_inference/platforms/tpu_platform.py`](../../tpu_inference/platforms/tpu_platform.py) — `TPUOffloadConnector` allowlist + `TPU_ALLOW_CHUNKED_MM_INPUT_FOR_TEXT_ONLY` override.
- [`tests/offload/test_hma_tracking.py`](../../tests/offload/test_hma_tracking.py) — HMA-specific unit tests (incl. mamba round-trip).
- [`tests/offload/tpu_offload_*_test.py`](../../tests/offload/) — pre-existing connector unit tests.
- [`gke/benchmarks/`](gke/benchmarks/) — production GKE deployment + benchmark harness for non-hybrid serving.
