# Benchmarking TPU Commons

## Setup
In order to begin benchmarking TPU Commons (via vLLM), you'll want to copy all of the scripts in this directory (also listed below for your convenience) to the [`benchmarks`](https://github.com/vllm-project/vllm/tree/main/benchmarks) directory in your local copy of vLLM:

* `scripts/vllm/benchmarking/backend_request_func.py`
* `scripts/vllm/benchmarking/benchmark_dataset.py`
* `scripts/vllm/benchmarking/benchmark_serving.py`
* `scripts/vllm/benchmarking/benchmark_utils.py`

## Running the Benchmarks
In order to run the benchmarks, navigate to your vLLM root directory and spin up a server to serve your model, for example:

```
TPU_BACKEND_TYPE=jax vllm serve meta-llama/Meta-Llama-3-8B-Instruct --max-model-len=1024 --disable-log-requests --tensor-parallel-size 8 --max-num-batched-tokens 8196 --max-num-seqs=1
```

Once the server has begun -- you should see a message such as `INFO:     Application startup complete.` -- you can now run the client benchmarking command (in a new terminal) in your local vLLM repo:

```
 python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset-name mlperf \
    --dataset-path /mnt/disks/jacobplatin/loadgen_run_data/processed-data.pkl \
    --num-prompts 50 \
    --run_eval
```

Note that you can also specify `mmlu` for the `dataset-name`.

If all goes well, you should an output similar to:

```
============ Serving Benchmark Result ============
Successful requests:                     50
Benchmark duration (s):                  68.70
Total input tokens:                      2956
Total generated tokens:                  7422
Request throughput (req/s):              0.73
Output token throughput (tok/s):         108.03
Total Token throughput (tok/s):          151.06
---------------Time to First Token----------------
Mean TTFT (ms):                          38954.14
Median TTFT (ms):                        32214.06
P99 TTFT (ms):                           67659.68
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.75
Median TPOT (ms):                        5.80
P99 TPOT (ms):                           30.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.16
Median ITL (ms):                         5.77
P99 ITL (ms):                            6.31
==================================================
Evaluating MLPerf...

Results

{'rouge1': 35.5177, 'rouge2': 16.4534, 'rougeL': 24.0036, 'rougeLsum': 34.2323, 'gen_num': 50}
```
