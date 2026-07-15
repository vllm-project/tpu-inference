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

## 3. Performance Results Summary (TPU v6e-8, TP=8)

Here are the side-by-side comparative metrics for **Qwen3-1.7B-base** and **Qwen3-4B** run under high-concurrency conditions using explicit `--async-scheduling`.

> [!NOTE]
> All benchmark metrics below were collected on a **Google Cloud TPU v6e-8 VM (8 TPU Cores)** with Tensor Parallelism size set to **`TP=8`** (`--tensor-parallel-size 8`).

### Throughput Metrics

| Model Architecture | Concurrency Scale | Total Run Time | Output Throughput | Total Throughput | Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-1.7B-base** | **128 Streams** | 30.24 s | 2,504.13 tokens/s | 24,156.69 tokens/s | 100% |
| **Qwen3-1.7B-base** | **256 Streams** | 52.95 s | 2,797.90 tokens/s | 26,423.03 tokens/s | 100% |
| **Qwen3-4B** | **128 Streams** | 36.42 s | 2,076.71 tokens/s | 19,987.43 tokens/s | 100% |
| **Qwen3-4B** | **256 Streams** | 66.55 s | 2,245.07 tokens/s | 20,918.13 tokens/s | 100% |

### Latency Profiles (Average)

| Model Architecture | Concurrency Scale | Turn 1 Prefill (Miss) | Turn 1 Prefill (Hit) | Turns 2+ TTFT | TPOT (Time per Output Token) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-1.7B-base** | **128 Streams** | 4.10 s | 3.02 s | 398.58 ms | **30.69 ms** |
| **Qwen3-1.7B-base** | **256 Streams** | 14.16 s | 13.05 s | 1.24 s | **32.01 ms** |
| **Qwen3-4B** | **128 Streams** | 3.94 s | 3.16 s | 602.61 ms | **38.12 ms** |
| **Qwen3-4B** | **256 Streams** | 18.80 s | 17.43 s | 1.19 s | **39.88 ms** |
