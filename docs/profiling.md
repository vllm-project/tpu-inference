# Profiling

There are currently three ways to profile your workload:

## Using `examples/tpu_profiling.py`

### vLLM TPU Profiling Script

This script is a utility for profiling the performance of the vLLM engine on TPU VMs. It uses the JAX profiler to capture detailed performance traces.

The profiling results can be visualized using tools like TensorBoard (with the `tensorboard-plugin-profile` package) or Perfetto UI.

### How to Use

#### Prerequisites
You must install the TensorBoard profile plugin to visualize the results:

```bash
pip install tensorboard-plugin-profile
```

#### Basic Command
The script is run from the command line, specifying the workload parameters and any necessary vLLM engine arguments.

```bash
python3 examples/tpu_profiling.py --model <your-model-name> [OPTIONS]
```

#### Key Arguments
* `--model`: (Required) The name or path of the model to profile.
* `--input-len`: The length of the input prompt tokens per request
* `--output-len`: The number of tokens to generate per request.
* `--batch-size`: The number of requests.
* `--profile-result-dir`: The directory where the JAX profiler output will be saved.
* The script also accepts all standard vLLM `EngineArgs` (e.g., `--tensor-parallel-size`, `--dtype`).

#### Examples

**1. Profile a Prefill Operation:**
To profile a single request with a long input prompt (e.g., 1024 tokens), set `--input-len` high and `--batch-size` to 1.

```bash
python3 examples/tpu_profiling.py \
  --model google/gemma-2b \
  --input-len 1024 \
  --output-len 1 \
  --batch-size 1
```

**2. Profile a Decoding Operation:**
To profile a large batch of single-token decoding steps, set `--input-len` and `--output-len` to 1 and use a large `--batch-size`.

```bash
python3 examples/tpu_profiling.py \
  --model google/gemma-2b \
  --input-len 1 \
  --output-len 1 \
  --batch-size 256
```

## Using `PHASED_PROFILING_DIR`
If you set the following environment variable:

```

PHASED_PROFILING_DIR=<DESIRED PROFILING OUTPUT DIR>

```

we will automatically capture profiles during three phases of your workload (assuming they are encountered):
1. Prefill-heavy (the quotient of prefill / total scheduled tokens for the given batch is => 0.9)
2. Decode-heavy (the quotient of prefill / total scheduled tokens for the given batch is <= 0.2)
3. Mixed (the quotient of prefill / total scheduled tokens for the given batch is between 0.4 and 0.6)

To aid in your analysis, we will also log the batch composition for the profiled batches.

## Using `USE_JAX_PROFILER_SERVER`
If you set the following environment variable:

```

USE_JAX_PROFILER_SERVER=True

```

you can instead manually decide when to capture a profile and for how long, which can helpful if your workload (e.g. E2E benchmarking) is
large and taking a profile of the entire workload (i.e. using the above method) will generate a massive tracing file.

You can additionally set the desired profiling port (default is `9999`):

```

JAX_PROFILER_SERVER_PORT=XXXX

```

In order to use this approach, you can do the following:

1. Run your typical `vllm serve` or `offline_inference` command (making sure to set `USE_JAX_PROFILER_SERVER=True`)
2. Run your benchmarking command (`python benchmark_serving.py...`)
3. Once the warmup has completed and your benchmark is running, start a new tensorboard instance with your `logdir` set to the desired output location of your profiles (e.g. `tensorboard --logdir=profiles/llama3-mmlu/`)
4. Open the tensorboard instance and navigate to the `profile` page (e.g. `http://localhost:6006/#profile`)
5. Click `Capture Profile` and, in the `Profile Service URL(s) or TPU name` box, enter `localhost:XXXX` where `XXXX` is your `JAX_PROFILER_SERVER_PORT` (default is `9999`)

6. Enter the desired amount of time (in ms)
