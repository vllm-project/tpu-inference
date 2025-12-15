# Benchmarking TPU Inference

## Setup
In order to begin benchmarking TPU Inference (via vLLM), you'll want to copy all of the scripts in this directory (also listed below for your convenience) to the [`benchmarks`](https://github.com/vllm-project/vllm/tree/main/benchmarks) directory in your local copy of vLLM:

* `scripts/vllm/benchmarking/backend_request_func.py`
* `scripts/vllm/benchmarking/benchmark_dataset.py`
* `scripts/vllm/benchmarking/benchmark_serving.py`
* `scripts/vllm/benchmarking/benchmark_utils.py`

You'll also want to install all the necessary requirements via `pip install -r requirements_benchmarking.txt`

## Downloading the Data
To download and extract the MLPerf data, you can run:

```
curl https://rclone.org/install.sh | bash
rclone config create mlc-inference s3 provider Cloudflare access_key_id f65ba5eef400db161ea49967de89f47b secret_access_key fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P
gzip -d open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
dataset_path=open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
```

To download and extract the MMLU data, you can run:

```
mkdir -p mmlu
cd mmlu
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P .
tar -xvf data.tar
dataset_path=data/test/
```

## Running the Benchmarks
In order to run the benchmarks, navigate to your vLLM root directory and spin up a server to serve your model, for example:

```
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len=1024 --disable-log-requests --tensor-parallel-size 8 --max-num-batched-tokens 8192
```

Once the server has begun -- you should see a message such as `INFO:     Application startup complete.` -- you can now run the client benchmarking command (in a new terminal) in your local vLLM repo:

```
 python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name mlperf \
    --dataset-path [local path to your dataset]\
    --num-prompts 50 \
    --run_eval
```

Note that you can also specify `mmlu` for the `dataset-name` (but your results will be different).

If all goes well, you should an output similar to:

```
============ Serving Benchmark Result ============
Successful requests:                     XX
Benchmark duration (s):                  XX
Total input tokens:                      XX
Total generated tokens:                  XX
Request throughput (req/s):              XX
Output token throughput (tok/s):         XX
Total token throughput (tok/s):          XX
---------------Time to First Token----------------
Mean TTFT (ms):                          XX
Median TTFT (ms):                        XX
P99 TTFT (ms):                           XX
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          XX
Median TPOT (ms):                        XX
P99 TPOT (ms):                           XX
---------------Inter-token Latency----------------
Mean ITL (ms):                           XX
Median ITL (ms):                         XX
P99 ITL (ms):                            XX
==================================================
Evaluating MLPerf...

Results

{'rouge1': XX, 'rouge2': XX, 'rougeL': XX, 'rougeLsum': XX, 'gen_num': XX}
```
