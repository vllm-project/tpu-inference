# Qwen3.5-397B Benchmarks

This branch serves Qwen3.5-397B prefill-heavy at 4,945 total tokens
per second per chip at eight-bit weights on a TPU v7x node. Three
scripts reproduce it. run_server.sh serves the model, and run_client.sh
and run_sweep.sh measure it, defaulting to the original benchmark
commands. EVAL=inferencex switches them to the public InferenceX
protocol, described under Evaluation Protocol.

## Optimizations

The serving results come from four groups of changes.

- A fused expert-parallel MoE kernel. Dispatch, expert matmuls, and
  combine run as one program across the expert-parallel mesh. The kernel
  carries its own top-k selection and pushes each result row directly to
  the chip that owns the token, so the combine transport runs inside the
  kernel rather than as a separate collective. Off by default, engaged
  per step at or above a token threshold set by
  MOE_FUSED_EP_KERNEL_MIN_TOKENS. This recipe sets 1024, and the
  right threshold can differ by model and configuration.
- The MoE feed-forward's grouped matmuls fused into one kernel. Gate and
  up projection, activation, and down projection run in a single program,
  and the intermediate never leaves it. The layer asks the kernel per
  shape and falls back to the existing pair with the reason logged when
  a shape is not supported. The kernel buckets rows per tile on
  every call.
- The gated delta network's convolution state passed in its stored
  layout and dtype, the recurrent state stored in BF16, widened on
  load and rounded on writeback. The stored-layout handling
  is always on, the flag governs only the BF16 state.
- The rope table slice extended to multimodal models under
  language-model-only serving (the slice itself is upstream). A new
  flag, on by default, compiles the logits gather in its conservative
  mode so the head weight is not retiled per step.

## Serving Results

Throughput rows are total tokens per second per chip at eight-bit
weights, and each row names its workload and tier.

| Workload | Before | After |
|---|---|---|
| Prefill Heavy, 8K In / 1K Out, Concurrency 64 | 3,982 | 4,945 (+24%) |
| Decode Heavy, 1K In / 8K Out, Concurrency 64 | 701 | 795 (+13%) |
| MMLU-Pro, 50 Questions Per Subject | 0.833 | 0.833 |

Before is the previous build measured on the same workloads, and the
MMLU-Pro pair shows quality held.

Run-to-run spread. The scheduler collects incoming requests into groups
of eight, one per rank. Cells at this workload once read 4 to 5 percent
low when a group formed late in the opening burst and waited for its
eighth member. The release rules now free a waiting group at the first
idle rank or after the pinned five-second deadline, and the
prefill-heavy cell holds within about half a percent, with the worst
first-token time near 8 seconds. A run far outside that band means the
batching settings did not engage, and the engagement checks refuse such
a server.

Reproducing each row.

- Prefill heavy, `./run_client.sh` at concurrency 64, or `./run_sweep.sh`
  for the tier walk.
- Decode heavy, the same command with the lengths swapped,
  `ISL=1024 OSL=8192 ./run_client.sh`.
- MMLU-Pro, lm-eval (from pip) against the running server, chat
  template on, thinking off, 50 questions per subject, seed 42.
  Quality baseline is 0.833, measured the same way at eight-bit
  weights on a server without this branch's optimizations.

## Benchmark Commands

```bash
# install per docs/getting_started/installation.md, then the checkpoint,
# hf download Qwen/Qwen3.5-397B-A17B-FP8   (406 GB, HF_HOME needs 450 GB free)
# pip install aiohttp tqdm transformers lm-eval   (client modules, MMLU row)
# the first run fetches the pinned InferenceX benchmark client with git clone
cd scripts/benchmarking/qwen35

# terminal 1, wait for "Application startup complete"
./run_server.sh              # eight-bit weights, as the checkpoint ships
WEIGHTS=fp4 ./run_server.sh  # four-bit MoE weights

# terminal 2, the measurement at concurrency 64
./run_client.sh                     # the original commands, the default
EVAL=inferencex ./run_client.sh     # the InferenceX protocol

# or the tier walk, a fresh server per tier
./run_sweep.sh               # tiers 64, 128, 256, 512
# run_sweep.sh starts its own servers, stop any server on the port first
```

## Evaluation Protocol

Both styles share the same base command. A random dataset at 8192
input and 1024 output with range ratio 0.8, concurrency 64, request
rate infinite, ignore-eos, seed 0. run_client.sh issues the invocation
itself and records it beside each result in protocol.txt. The
differences are the four knobs below.

| | EVAL=current | EVAL=inferencex |
|---|---|---|
| Prompts | 640 always | 10 times concurrency |
| Warmups | none | 2 times concurrency, before the clock |
| Chat Template | off | on |
| Sweep Server Cap | 64 per rank at every tier | rises with the tier |

EVAL=inferencex matches the public InferenceX client flag for flag
against their pinned copy, and the server keeps this branch's serving
recipe at every tier. The protocol is defined in the InferenceX repository at
<https://github.com/SemiAnalysisAI/InferenceX>, in
`benchmarks/benchmark_lib.sh` (`run_benchmark_serving`) and
`benchmarks/single_node/fixed_seq_len/qwen3.5_fp8_b300.sh`, and
published curves are at <https://inferencex.com>. Their curve for this
model sweeps concurrency 4 through 256, and run_sweep.sh walks 64
through 512.

The two styles measure the same serving speed. In matched pairs on the
same server the template and warmup knobs each move the number by under
half a percent, so the table above reads the same under either style
within the half-percent band the Serving Results section describes.

## Serving Recipe

| | |
|---|---|
| Model | `Qwen/Qwen3.5-397B-A17B-FP8`, one checkpoint for both weight forms |
| Mesh | 8 devices (4 chips of 2 cores), one server |
| Max Model Length | 9236 (8K input, 1K output, 20 headroom, the InferenceX context-length convention), batched tokens 1024 and sequences 64 per attention rank, 8192 and 512 across the node |
| KV Cache | FP8, block size 256, prefix caching off |

Attention runs data parallel across the devices, the MoE runs expert
parallel, and the serve flag --tensor-parallel-size=8 names the device
count, not classic tensor sharding. run_server.sh carries the exact
serve line. The fused kernel's token
threshold is compared against the node-wide token count, 8192 at a full
prefill step here, so the kernel owns prefill with an 8x margin and
decode, at most 512 node-wide, stays below it.

With FP4 weights the MoE experts are requantized to E2M1 at block 512
during load, so the FP4 server starts slower than the FP8 one. A cold
start compiles for about an hour, a start whose shapes are already
cached takes about half an hour, and each new tier in run_sweep.sh
compiles its own batch shapes once. The scripts keep the compile cache
in ./jax-cache beside them, and switching weight forms recompiles.
