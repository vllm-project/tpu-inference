# Running tpu-inference in float32

`--dtype float32` did not work on the vLLM/torchax model path. One bug blocked it;
this page records what the bug was, what changed, and what you do *not* need to
change (a few things that look like blockers turn out not to be).

Verified on a `tpu7x-8` host (8 devices x 101 GB HBM, 944 GB host RAM).

## Short version

```bash
python examples/offline_inference.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --dtype float32 \
  --tensor-parallel-size 8 \
  --max-model-len 1024
```

Leave `--kv-cache-dtype` alone. That is the whole invocation -- no
`--additional-config`, no env vars.

## The one real blocker: freeing host weights after `t2j`

### Symptom

```
File ".../tpu_inference/layers/vllm/quantization/unquantized.py", line 380, in _load_linear_weights
  layer.weight.untyped_storage().resize_(0)
RuntimeError: Trying to resize storage that is not resizable
```

Raised on the first linear layer, so the engine never finishes loading. Failing
on `weight` and then, once that is fixed, on `bias`, `w13_weight`, `w2_weight`,
... -- it is systemic, not one bad call site.

### Cause

`unquantized.py` imports `t2j` from **torchax**, not from `tpu_inference.utils`.
torchax's `t2j` ends in:

```python
if t.dtype in NUMPY_UNSUPPORTED_DTYPES:
    nparray = t.cpu().detach().to(torch.float32).numpy()   # bf16, fp8: COPY
else:
    nparray = t.cpu().detach().numpy()                     # fp32, fp16: ALIAS
```

numpy has no scalar type for bf16 or the fp8 formats, so those get widened to
float32 first -- and `.to(torch.float32)` allocates, so numpy is handed the
*copy's* storage. float32 needs no widening (`.to(torch.float32)` is a no-op on
a tensor that is already float32), so numpy is handed the **parameter's own
storage**.

Handing torch storage to numpy marks that storage non-resizable *permanently* --
it does not come back when the numpy array is freed:

```python
p = torch.nn.Parameter(torch.empty(64, 64, dtype=torch.float32))
p.untyped_storage().resizable()     # True
_ = jnp.asarray(p.detach().numpy())
p.untyped_storage().resizable()     # False -- and stays False after del + gc
```

So the very next line, `resize_(0)`, raises. bf16 never noticed because the
widening copy shields the parameter. This is a dtype-dependent hazard, not a
Qwen3.5 one: float16 hits the identical failure, and the JAX-native model path
(`Qwen3ForCausalLM` and friends) is unaffected because it never takes this route.

`_host_numpy_view` in the same file already documents this exact trap, and
`fp8.py` already carried the right helper for it -- the unquantized path just
was not using it.

### Fix

Share `fp8.py`'s `_free_torch_storage` (moved to `unquantized.py`, which `fp8.py`
already imports from) and use it at the six unguarded `resize_(0)` sites in the
unquantized linear and MoE loaders. It falls back to
`set_(torch.storage.UntypedStorage())`, which drops the storage reference and
frees the buffer just the same.

One addition was needed to make the fallback usable: `set_` is an in-place op and
torch refuses it on a leaf that still requires grad, which most vLLM parameters
do. Without a `torch.no_grad()` wrapper the fallback trades one error for
another:

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

Files touched:

- `tpu_inference/layers/vllm/quantization/unquantized.py` -- add
  `_free_torch_storage` (with the `no_grad` fallback), use it in
  `_load_linear_weights` (weight, bias) and the MoE loader (`w13_weight`,
  `w2_weight`, `w13_bias`, `w2_bias`).
- `tpu_inference/layers/vllm/quantization/fp8.py` -- drop the duplicate
  definition, import the shared one.

Memory behaviour is unchanged: the `set_()` path frees the same buffer eagerly,
so fp32 weights do not accumulate on the host.

## Things that look like blockers but are not

**The KV cache does accept fp32 -- just don't ask for it by name.**
`--kv-cache-dtype` has no `float32` choice; its options are `auto`, `float16`,
`bfloat16`, and the fp8/quantized variants. But `auto` means *follow the model
dtype*, and both `tpu_runner.py` and `sharding.py` resolve it that way, so
`--dtype float32` already gives you an fp32 cache. Confirmed by the page size
doubling on Qwen3.5-4B: `attn_page=524288` in bf16 -> `attn_page=1048576` in
fp32. No `--additional-config` override is required.

**The attention kernels already handle fp32.** `ragged_paged_attention` picks
`out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16`, and the
GDN/mamba kernels accumulate in fp32 regardless of storage dtype.

**`TpuPlatform` does not reject fp32.** `is_kv_cache_dtype_supported` returns
`True` unconditionally, and there is no dtype validation in
`check_and_update_config`.

**The hybrid GDN + MoE + vision stack needs nothing extra.** Qwen3.5-35B-A3B
(40 layers, linear attention with full attention every 4th layer, 256 experts,
vision tower) runs in fp32 with only the loader fix.

## What it costs

For Qwen3.5-35B-A3B on `tpu7x-8`:

| | bf16 | fp32 |
|---|---|---|
| Weights | ~67 GB | ~134 GB |
| KV page (attention) | 524288 B | 1048576 B |
| KV cache capacity @ `max_model_len=1024` | -- | 2,285,614 tokens |
| Weight load (storage -> TPU) | -- | 52.7 s |

fp32 doubles both weights and KV cache. 813 GB of aggregate HBM absorbs that
comfortably here; on smaller TPU generations fp32 is where a model stops fitting
long before anything else breaks.

## Verified

| Model | Path | dtype | Result |
|---|---|---|---|
| Qwen3.5-35B-A3B | vLLM/torchax | float32 | passes, coherent output |
| Qwen3.5-4B | vLLM/torchax | float32 | passes |
| Qwen3.5-4B | vLLM/torchax | float16 | loads (same bug, same fix), then fails in a GDN kernel -- see below |
| Qwen3.5-4B | vLLM/torchax | bfloat16 | unchanged, output byte-identical to pre-fix |
| Qwen3-0.6B | JAX-native | float32 | passes, no changes needed |

## bf16 weights with fp32 activations

fp32 doubles the weights, which is usually the part you did not want. Set
`WEIGHT_STORAGE_DTYPE` to keep unquantized linear and MoE weights in a narrower
dtype while everything else -- activations, KV cache, norms, the residual stream
-- stays fp32:

```bash
WEIGHT_STORAGE_DTYPE=bfloat16 python examples/offline_inference.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --dtype float32 \
  --tensor-parallel-size 8 \
  --max-model-len 1024
```

This is a *storage* choice, not a numerical one. The matmuls promote the weight
back to the activation dtype, so the arithmetic is fp32 either way -- what you
trade is the bits the cast throws away for the HBM they occupied. Rounding fp32
weights to bf16 and promoting them back is numerically the same thing, so this
also covers the precision-study case without paying for fp32 storage.

No matmul changes were needed. The unquantized linear paths multiply with `@`
(via `sharded_matmul`) and `jnp.einsum`, both of which promote; `lax.dot_general`
would not have, and `_matmul_fused` / `_matmul_split` are the documented seam if
a kernel ever needs an explicit cast.

Measured on Qwen3.5-35B-A3B, `tpu7x-8`, against the same run in plain fp32:

| | fp32 weights | bf16 weights |
|---|---|---|
| Weight-resident HBM | 213.75 GiB | 110.17 GiB |
| Attention KV blocks | 49,105 | 59,710 |

Slightly over half remains rather than exactly half: embeddings, norms and the
vision tower are not covered by this flag and stay fp32.

## float16 is still blocked, for an unrelated reason

The loader fix gets fp16 all the way through weight loading, but it then dies
compiling a GDN kernel:

```
jax.errors.JaxRuntimeError: INTERNAL: Mosaic failed to compile TPU kernel
  ... HLO name `fused_conv1d_gdn_batched.1`: Invalid vector type for load
```

That is a Mosaic/Pallas limitation in the fp16 path of the GDN conv1d kernel,
independent of anything here -- fp32 compiles the same kernel fine. Fixing it
would mean either widening the kernel's fp16 loads or forcing the GDN conv to
fp32 while the rest of the model stays fp16; neither was in scope for this
change, so fp16 remains unsupported on hybrid GDN models.
