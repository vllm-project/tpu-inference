# Fused MoE kernels (experimental)

Two TPU Pallas kernels for expert-parallel (EP) Mixture-of-Experts. They demonstrate
fusing the EP collective communication directly into the GMM compute.

## `fused_moe/` — full fused MoE kernel (ready for usage)

`fused_moe_func_rs` runs the entire EP MoE in a single Pallas call:

```
gather -> GMM1 -> activation -> GMM2 -> ICI A2A
```

The A2A is fused into the kernel, replacing the usual post-kernel
all-reduce/psum in EP MoE. It supports bf16 and quantized (fp8) weights, an
optional post-expert RMSNorm, and sequence-parallel in/out (toggle with
`TPU_MOE_ENABLE_SP`). This is the production-oriented kernel — use it as the MoE
layer entry point.

```python
from tpu_inference.kernels.experimental.fused_moe import fused_moe_func_rs

out = fused_moe_func_rs(
    hidden_states, w1, w2, w1_scale, w2_scale, w1_bias, w2_bias,
    gating_output, topk, renormalize, mesh, activation, scoring_fn,
)
```

## `gmm_fused/` — AG+GMM1 kernel (example)

`gmm_v2_ag_gmm1` is a smaller, instructional kernel: a push-based all-gather
fused with `GMM1 + activation`, driven by a precomputed per-round send schedule
(`per_round_schedule.py`). It illustrates how to overlap and fuse the EP
all-gather into the grouped matmul — the same idea the full fused kernel above
builds on. It is provided as a reference/example rather than a complete MoE
layer; the paired GMM2 + ICI reduce-scatter kernels live in the same module.

## Future work

- **Fuse the upstream all-gather** into the kernel so it overlaps with compute,
  using a persistent token cache (each token crosses to a receiver at most once)
  and neighbor-relay routing (origin → nearest → next-nearest, link-local hops).
  Biggest win at decode with large hidden size. The `gmm_fused/` example is a
  first step.
- **Offload random-access work to SparseCore** — the routed-token gather, per-row
  all-to-all, and cache lookups — keeping the dense matmul on the TensorCore. And we need to use both Tensorcore and sparsecore in one kernel. This could make the current MoE kernel even >20% faster in prefill/large_batch

## Notes

- Device-specific tuned block-size tables were removed for this release;
  `tuned_block_sizes.py` returns caller-supplied defaults. Retune per device
  for best performance.
- These kernels target TPU and require a multi-device mesh to exercise the EP
  collectives.
