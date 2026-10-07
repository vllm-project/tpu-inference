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

Pass `--trace-file` to replay measured trajectories.

### Trace file format

One JSON object per line, one line per trajectory:

```json
{"group": 0, "stream": 0, "prompt_len": 7570, "out_lens": [399, 65, 57], "obs_lens": [23, 942, 2126]}
```

| field | meaning |
| --- | --- |
| `group` | Problem index. Streams sharing a `group` share one initial prompt, which is what exercises prefix caching. |
| `stream` | Generation index within the group, `0..G-1`. |
| `prompt_len` | Initial prompt length in tokens. Taken from the first stream of each group. |
| `out_lens` | Assistant tokens generated on each turn, in order. Its length sets the turn count. |
| `obs_lens` | Environment observation tokens appended after each turn, in order. |

---

## 4. Staggered Batch Scheduling & Metrics Output

In realistic RL training setups, rollout requests arrive in distinct batches (e.g. global training steps or iterations), where each batch launches `concurrency` number of groups, and successive batches are dispatched `time_between_batches` seconds apart.

### Run with Batches Dispatched on a Cadence

```bash
python scripts/vllm/benchmarking/agentic_benchmark/benchmark_agentic.py \
    --model-path-or-id Qwen/Qwen3-4B \
    --model Qwen/Qwen3-4B \
    --concurrency 8 \
    --num-batches 4 \
    --time-between-batches 30.0 \
    -g 16 \
    --save-trajectory-file trajectories.jsonl
```

- `--num-batches`: Number of batches to simulate (total groups = `num_batches * concurrency`).
- `--time-between-batches`: Delay in seconds between launching consecutive batches to the server.
- `--concurrency`: Number of groups running concurrently within each batch.
- `--save-trajectory-file`: Path to write turn-by-turn and trajectory-level metrics (JSONL), compatible with downstream trajectory visualization tools.

