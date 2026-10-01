# DAPO/GRPO RL on TPU v7x — Throughput Recipe & Measured Ladder

A reproducible recipe for fast DAPO/GRPO reinforcement-learning **rollout+train** steps on
Ironwood TPU v7x (64 chips, 4x4x4, DP16×TP4), Qwen3-0.6B, using the vLLM-on-TPU (`tpu-inference`)
rollout engine. All numbers are median clean-step wall time (Init-kv → Init-kv), n≥5 warm steps,
same-window controls, STRICT-BF16 unless noted.

## Result ladder (measured)

| Config | Median step | Class | Notes |
|---|---|---|---|
| baseline (no continue_decode) | 616s | — | host-dispatch-bound rollout |
| + continue_decode (mds=128) | 369s | STRICT-BF16 | on-device fused multi-step decode (−40%) |
| + RPA-v3 decode-block split `1,16384,1,4096` | 327s | STRICT-BF16 | fetch/compute block tuning (see PR #3102) |
| + EOS-check interval 8 (stacked) | **299s** | **STRICT-BF16** | removes the slow-decode regime; std 2–6s, n=11 |
| + response-cap trim 8192→6144 | **245s** | recipe-candidate | −70s same-window; reward-neutral in-window |

**Headline (STRICT-BF16, no RL-problem change): 299s** — 2.06× over the 616s baseline, entirely from
rollout-engine + kernel-tuning levers (no precision or reward changes).

**Recipe-candidate: 245s** — a further −54s from trimming the response cap 8192→6144. Reward-neutral
in-window (rewards/mean in the 8192 control's band; loss parity 0.006–0.018). This CHANGES the RL
problem (max response length) and should be validated by the reward owner over 50+ steps + a
length-growth curriculum before adoption. The cap mostly truncates degenerate non-terminating
generations (repetitive `</think>` loops), not genuine reasoning (verified from completions + KV usage ≤~38%).

## Levers (all config / env-gated, default-off)

1. **continue_decode** — `vllm_additional_config: {enable_continue_decode: true, max_decode_steps: 128}`.
   Fuses N decode iterations on-device via a JAX while-loop, amortizing the per-token host-dispatch gap
   that dominates TPU rollout. Requires `async_scheduling: true` + a libtpu with the v15 RPA/continue_decode
   Mosaic kernels. Single biggest lever (−40%).
2. **RPA-v3 decode-block split** — `GRR_RPA_DECODE_BLOCKS=1,16384,1,4096`. Ragged Paged Attention v3
   fetch/compute block-size override (fetch≠compute) on the decode path. Upstream env plumbing: PR #3102.
3. **EOS-check interval** — `GRR_CD_EOS_CHECK_INTERVAL=8`. How often the fused loop checks EOS. Interval 8
   removes a bimodal slow regime; >8 regresses (wastes burst compute on finished seqs). Saturated at 8.

## Phase split @ 299s (single 64-chip mesh, interleaved server)
- decode ~199s (64%) — ~100k tok/s aggregate; drains to a single-sequence tail; ~83% batch util
- train 75s (24%) — FLOP-bound at this batch (near-linear scaling)
- setup 37s (12%) — dominated by per-step KV-cache realloc around weight sync

## Tested to destruction (honest negatives)
- **shared-prefix cascade kernel**: RPA-v3 already dedups shared physical KV pages; a composed cascade
  measured 0.96–0.99× (no win). The "8× prefix re-read" premise was false on hardware.
- **response-cap < 6144** (5120): tied with 6144 within variance (233↔243s) — 6144 already cut the
  degenerate tail; diminishing returns.
- **KV-cache persist across weight sync**: not viable — KV is ~57GB/chip, deliberately freed so the
  HBM-heavy weight reshard fits (115→58GB); persisting OOMs the reshard.
- **on-device decode/train overlap on a single mesh**: impossible — the interleaved server time-shares
  all devices; only host-side work (~12s) is already overlapped by the stock rollout/train queue.

## Reproduce
`config_template.yaml` + `launch.sh` here. One command: set the three env levers on a PR-image with the
v15 kernels, launch a DP16×TP4 4x4x4 jobset. The `enable_continue_decode` flag is consumed by the
`tpu-inference` rollout engine; the RPA/EOS env vars tune the decode kernel.
