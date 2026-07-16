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

## 3. Running on Real SWE Task Content (R2E-Gym)

By default the benchmark builds prompts and environment replies from random token IDs. That is fine for fixing token counts, but the content is meaningless. To drive the same workload from real agentic content, `prepare_r2e_dataset.py` converts [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) SWE tasks into a scripted multi-turn dataset.

### Why a conversion step is needed

R2E-Gym ships **task definitions, not trajectories** — every row's `input` field is empty, so there is nothing to replay. Each row does carry real content: a GitHub-issue style problem statement, the pre-fix source files, and the recorded stdout of the test suite before and after the fix. The converter assembles those into an agent prompt plus a fixed sequence of environment observations.

Running the real environment instead would mean per-task containers, test execution, and a reward loop (NVIDIA measures ~20 min per training step, ~1 CPU core per task instance). That measures the harness, not the serving stack. Here the model still generates **every assistant turn live against the server**, so the generated token distribution is real; only the environment side is pre-baked.

### Step 1: Build the dataset

```bash
python prepare_r2e_dataset.py \
    --dataset hfilaretov/Benchmark-R2E-Gym-Easy \
    --split val \
    --model-path-or-id Qwen/Qwen3-4B \
    --output r2e_easy_val.jsonl
```

The converter prints the resulting token distribution. For the `val` split with the Qwen3-4B tokenizer:

| Metric | min | p50 | p99 | max |
| --- | --- | --- | --- | --- |
| Initial prompt tokens | 4,409 | 8,552 | — | 15,490 |
| Env observation tokens | 17 | 410 | 2,181 | 2,407 |

Use `--input-jsonl` to convert an already-downloaded file, and `--num-instances` to convert a subset.

### Step 2: Run the benchmark against it

```bash
python benchmark_agentic.py \
    --model-path-or-id Qwen/Qwen3-4B \
    --model Qwen/Qwen3-4B \
    --dataset r2e_easy_val.jsonl \
    --num-groups 16 \
    -g 16 \
    --concurrency 16
```

One task instance backs one GRPO group, matching real RL where a group of rollouts shares a single task and therefore a single prefix. In this mode prompt lengths and turn counts come from the dataset, so `--initial-prompt-len-*`, `--turns-*` and `--env-len-*` are ignored. Omit `--dataset` for the original random-token workload.

### What changes versus random tokens

Real environment observations have a **p50 of ~410 tokens**, against the `10-20` the random-mode examples above use. Real observations are pytest output, file views, and traceback text — roughly 20-40x larger. Since every observation is prefilled on the next turn, this materially raises per-turn prefill cost and shifts the prefill/decode balance. Prompts also share a genuine long prefix (system prompt plus repo context) rather than random tokens, which is what prefix caching actually sees in production.
