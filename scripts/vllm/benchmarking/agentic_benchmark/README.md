# GRPO Multi-Turn RL Inference Benchmark

This benchmark simulates large-scale asynchronous Reinforcement Learning (RL) training loops (such as Group Relative Policy Optimization - GRPO) against a running vLLM engine on TPU hardware.

## 1. What is the Agentic RL GRPO Benchmark?

In modern RL training workflows (e.g., GRPO), the inference engine serves rollouts where:
1. **Shared Prefixes**: A single prompt is sharded across a group of $G$ streams ($G=16$). These streams consume the initial prompt concurrently.
2. **Multi-Turn Interactions**: Each stream interacts with an external environment across multiple conversational turns ($10\text{--}100$ turns).
3. **Alternating Rollouts**: In each turn, the model generates output tokens ($200\text{--}2\text{k}$ tokens), and the environment appends feedback ($10\text{--}100$ tokens) before launching the next generation step.

This benchmark uses asynchronous scheduling to maximize TPU hardware utilization, and prefix caching to speed up turn-1 prefill latency for rollout streams.

---

## 2. How to Run the Benchmark (for Qwen3-4B)

### Step 1: Start the vLLM Server
Launch the vLLM OpenAI-compatible API server on your TPU host.

```bash
# Set environment variables for huggingface
export HF_TOKEN="your_hf_token_here"
export HF_HOME="~/.cache/huggingface"

# Start the vLLM server
vllm serve Qwen/Qwen3-4B \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enable-prefix-caching \
    --async-scheduling \
    --max-num-seqs 256 \
    --max-model-len 32768
```

### Step 2: Run the Benchmark Script
Run the multi-turn benchmark script against the server. Configure the active concurrency to scale the request rate.

#### Run with 128 Concurrent Streams (8 Groups)

```bash
python benchmarks/benchmark_agentic.py \
    --model-path-or-id Qwen/Qwen3-4B \
    --model Qwen/Qwen3-4B \
    --num-groups 8 \
    -g 16 \
    --initial-prompt-len-min 500 \
    --initial-prompt-len-max 1000 \
    --turns-min 3 \
    --turns-max 5 \
    --output-len-min 100 \
    --output-len-max 200 \
    --env-len-min 10 \
    --env-len-max 20 \
    --concurrency 8
```

#### Run with 256 Concurrent Streams (16 Groups)

```bash
python benchmarks/benchmark_agentic.py \
    --model-path-or-id Qwen/Qwen3-4B \
    --model Qwen/Qwen3-4B \
    --num-groups 16 \
    -g 16 \
    --initial-prompt-len-min 500 \
    --initial-prompt-len-max 1000 \
    --turns-min 3 \
    --turns-max 5 \
    --output-len-min 100 \
    --output-len-max 200 \
    --env-len-min 10 \
    --env-len-max 20 \
    --concurrency 16
```

---

## 3. Replaying a Recorded Rollout Trace

The flags above sample turn counts and lengths from uniform distributions. Real
agent rollouts do not look like that: turn count is a *cap* that most
trajectories reach rather than a spread, assistant turns are short tool calls,
and environment observations are large and heavy tailed. Measured on SWE-agent
rollouts of `Qwen3.5-397B-A17B`:

| quantity | uniform default | real rollouts (min / median / max) |
| --- | --- | --- |
| assistant turns | `uniform(5, 50)` | 12 / **30** / 30 |
| output tokens per turn | `uniform(200, 1000)` | 3 / **82** / 9,076 |
| observation tokens per turn | `uniform(10, 100)` | 23 / **220** / 26,625 |
| initial prompt | `uniform(2k, 8k)` | 7,546 / **7,650** / 7,753 |

Pass `--trace-file` to replay measured trajectories instead. The offered load is
then exactly the recorded one and is identical on every run, which makes small
throughput differences between server configurations readable.

### Trace file format

One JSON object per line, one line per trajectory:

```json
{"group": 0, "stream": 0, "prompt_len": 7570, "turns": 30, "final_len": 37975, "reward": 1.0, "out_lens": [399, 65, 57], "obs_lens": [23, 942, 2126]}
```

| field | meaning |
| --- | --- |
| `group` | Problem index. Streams sharing a `group` share one initial prompt, which is what exercises prefix caching. |
| `stream` | Generation index within the group, `0..G-1`. |
| `prompt_len` | Initial prompt length in tokens. Taken from the first stream of each group. |
| `out_lens` | Assistant tokens generated on each turn, in order. Its length sets the turn count. |
| `obs_lens` | Environment observation tokens appended after each turn, in order. |
| `turns`, `final_len`, `reward` | Optional, for filtering and validation. Ignored during replay. |

`out_lens` and `obs_lens` are the only required per-turn fields. No prompt or
completion text is needed, so a trace of 1,024 trajectories is only a few
hundred KB and carries no task content.

To build one from RL rollout dumps, walk each trajectory's assistant-token mask
and record the length of each alternating run: runs of generated tokens become
`out_lens`, the runs between them become `obs_lens`, and the leading run of
non-generated tokens is `prompt_len`. Truncate to the unpadded trajectory length
first, or trailing padding is counted as one enormous observation. A trace is
correct when `prompt_len + sum(out_lens) + sum(obs_lens)` equals the recorded
trajectory length for every row.

### Global prefix

Real agents share a system prompt and tool definitions across *every*
trajectory, not just within a group — 6,476 tokens in the rollouts above, with
roughly 1,100 more shared per problem. Pass `--global-prefix-len` to reproduce
that; without it, groups share nothing with each other and prefix caching is
understated.

```bash
python benchmark_agentic.py \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --model-path-or-id Qwen/Qwen3.5-397B-A17B-FP8 \
    --trace-file trace.jsonl \
    --global-prefix-len 6476 \
    --num-groups 16 \
    --concurrency 16
```

`--num-groups` limits how much of the trace is replayed; omit it to replay every
group. Concurrency is counted in groups, so `--concurrency 16` with a group size
of 16 runs 256 concurrent streams.
