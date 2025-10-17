# Get started with vLLM TPU

Google Cloud TPUs (Tensor Processing Units) accelerate machine learning workloads. vLLM supports TPU v6e and v5e. For architecture, supported topologies, and more, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture) and specific TPU version pages ([v5e](https://cloud.google.com/tpu/docs/v5e) and [v6e](https://cloud.google.com/tpu/docs/v6e)).

---

# Serve Llama3.1 with vLLM on TPU VMs

In this guide, we show how to serve
[Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[Llama3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct).

> **Note:** Access to Llama models on Hugging Face requires accepting the Community License Agreement and awaiting approval before you can download and serve them.

## Step 0: Install `gcloud cli`

You can reproduce this experiment from your dev environment
(e.g. your laptop).
You need to install `gcloud` locally to complete this tutorial.

To install `gcloud cli` please follow this guide:
[Install the gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

Once it is installed, you can login to GCP from your terminal with this
command: `gcloud auth login`.

## Step 1: Create a v6e TPU instance

We create a single VM. For Llama3.1-8B, 1 chip is sufficient and for the 70B
model, at least 8 chips are required. If you need a different number of
chips, you can set a different value for `--topology` such as `1x1`,
`2x4`, etc.

To learn more about topologies: [v6e VM Types](https://cloud.google.com/tpu/docs/v6e#vm-types).

> **Note:** Acquiring on-demand TPUs can be challenging due to high demand. We recommend using [Queued Resources](https://cloud.google.com/tpu/docs/queued-resources) to ensure you get the required capacity.

### Option 1: Create an on-demand TPU VM

This command attempts to create a TPU VM immediately.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

# this command creates a tpu vm with 8 Trillium (v6e) chips - adjust it to suit your needs
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
    --type v6e --topology 2x4 \
    --project $PROJECT --zone $ZONE --version v2-alpha-tpuv6e
```

### Option 2: Use Queued Resources (Recommended)

With Queued Resources, you submit a request for TPUs and it gets fulfilled
when capacity is available.

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project
export QR_ID=your-queued-resource-id # e.g. my-qr-request

# This command requests a v6e-8 (8 chips). Adjust accelerator-type for different sizes. For 1 chip (Llama3.1-8B), use --accelerator-type v6e-1.
gcloud alpha compute tpus queued-resources create $QR_ID \
    --node-id $TPU_NAME \
    --project $PROJECT --zone $ZONE \
    --accelerator-type v6e-8 \
    --runtime-version v2-alpha-tpuv6e
```

You can check the status of your request with:

```bash
gcloud alpha compute tpus queued-resources list --project $PROJECT --zone $ZONE
```

Once the state is `ACTIVE`, your TPU VM is ready and you can proceed to the next steps.

## Step 2: ssh to the instance

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 3: Use the latest vllm docker image for TPU

```bash
export DOCKER_URI=vllm/vllm-tpu:latest
```

## Step 4: Run the docker container in the TPU instance

```bash
sudo docker run -it --rm --name $USER-vllm --privileged --net=host \
    -v /dev/shm:/dev/shm \
    --shm-size 150gb \
    -p 8000:8000 \
    --entrypoint /bin/bash ${DOCKER_URI}
```

> **Note:** 150GB should be sufficient for the 70B model. For the 8B model allocate at least 17GB for the weights.

> **Note:** See [this guide](https://cloud.google.com/tpu/docs/attach-durable-block-storage) for attaching durable block storage to TPUs.

## Step 5: Set up env variables

Export your hugging face token along with other environment variables inside
the container.

```bash
export HF_HOME=/dev/shm
export HF_TOKEN=<your HF token>
```

## Step 6: Serve the model

Now we start the vllm server.
Make sure you keep this terminal open for the entire duration of this experiment.

```bash
export MAX_MODEL_LEN=4096
export TP=8 # number of chips

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --seed 42 \
    --disable-log-requests \
    --gpu-memory-utilization 0.98 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 256 \
    --tensor-parallel-size $TP \
    --max-model-len $MAX_MODEL_LEN
```

For the 8B model on a v6e-1 (1-chip) instance, we recommend `--max-num-batched-tokens 1024 --max-num-seqs 128`.

It takes a few minutes depending on the model size to prepare the server.
Once you see the below snippet in the logs, it means that the server is ready
to serve requests or run benchmarks:

```bash
INFO:     Started server process [7]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 7: Prepare the test environment

Open a new terminal to test the server and run the benchmark (keep the previous terminal open).

First, we ssh into the TPU vm via the new terminal:

```bash
export TPU_NAME=your-tpu-name
export ZONE=your-tpu-zone
export PROJECT=your-tpu-project

gcloud compute tpus tpu-vm ssh $TPU_NAME --project $PROJECT --zone=$ZONE
```

## Step 8: access the running container

```bash
sudo docker exec -it $USER-vllm bash
```

## Step 9: Test the server

Let's submit a test request to the server. This helps us to see if the server is launched properly and we can see legitimate response from the model.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "prompt": "I love the mornings, because ",
        "max_tokens": 200,
        "temperature": 0
    }'
```

## Step 10: Preparing the test image

You will need to install datasets as it's not available in the base vllm
image.

```bash
pip install datasets
```

## Step 11: Run the benchmarking

Finally, we are ready to run the benchmark:

```bash
export MAX_INPUT_LEN=1800
export MAX_OUTPUT_LEN=128
export HF_TOKEN=<your HF token>

cd /workspace/vllm

vllm bench serve \
    --backend vllm \
    --model "meta-llama/Llama-3.1-70B-Instruct"  \
    --dataset-name random \
    --num-prompts 1000 \
    --random-input-len=$MAX_INPUT_LEN \
    --random-output-len=$MAX_OUTPUT_LEN \
    --seed 100
```

The snippet below is what you’d expect to see - the numbers vary based on the vllm version, the model size and the TPU instance type/size.

```bash
============ Serving Benchmark Result ============
Successful requests:                     xxxxxxx
Benchmark duration (s):                  xxxxxxx
Total input tokens:                      xxxxxxx
Total generated tokens:                  xxxxxxx
Request throughput (req/s):              xxxxxxx
Output token throughput (tok/s):         xxxxxxx
Total Token throughput (tok/s):          xxxxxxx
---------------Time to First Token----------------
Mean TTFT (ms):                          xxxxxxx
Median TTFT (ms):                        xxxxxxx
P99 TTFT (ms):                           xxxxxxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxxxxxx
Median TPOT (ms):                        xxxxxxx
P99 TPOT (ms):                           xxxxxxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxxxxxx
Median ITL (ms):                         xxxxxxx
P99 ITL (ms):                            xxxxxxx
==================================================
```

## For further reading

* [Documentation](https://github.com/vllm-project/tpu-inference/tree/main/docs)
* [Examples](https://github.com/vllm-project/tpu-inference/tree/main/examples)
* [Recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM)
* [GKE serving with vLLM TPU](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-vllm-tpu)
