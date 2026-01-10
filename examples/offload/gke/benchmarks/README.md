# Benchmarks using SGLang bench_serving tool

This guide outlines the steps to deploy a vLLM serving instance on Google Kubernetes Engine (GKE) with TPUs, create a service to expose it, and then run the SGLang `bench_serving.py` benchmark against it. Two deployment options for vLLM are provided: a baseline without host offload and one with TPU host offload for KV cache.

## Prerequisites

* `kubectl` configured to connect to your GKE cluster.
* `gcloud` CLI installed and authenticated.
* A GKE cluster with TPU nodes (the below steps have been verified with `ct6e-standard-8t` GKE node)
* Access to Llama-3.3-70B model on Hugging Face

## 1. Create Hugging Face Token Secret

A Hugging Face token is required to pull the model. Create a Kubernetes secret with your token:

```bash
kubectl create secret generic hf-token-secret --from-literal=token='<YOUR_HF_TOKEN>'
```

Replace `<YOUR_HF_TOKEN>` with your actual Hugging Face token.

## 2. Deploy vLLM Pod (Choose One)

Choose one of the following deployment options for your vLLM pod. Ensure the right container image is used in the pod spec

### Option A: Baseline vLLM (No Host Offload)

This deployment uses a standard vLLM setup without any specific TPU host offload connector. The KV cache will reside entirely on the TPU HBM.

```bash
kubectl apply -f deploy-baseline.yaml
```

### Option B: vLLM with TPU Host Offload

This deployment configures vLLM to use a `TPUOffloadConnector` for KV cache offload to the host CPU memory. This is specified by the `--kv-transfer-config` argument.
The following environment variables are used to configure the offload behavior:

* `TPU_OFFLOAD_CPU_BUF_SIZE_GB`: Sets the size (in GB) of the CPU memory buffer for the KV cache.
* `TPU_OFFLOAD_STAGING_BUF_SIZE_GB`: Sets the size (in GB) of the on-device HBM staging buffer used for swap.

```bash
kubectl apply -f deploy-cpu-offload.yaml
```

## 3. Deploy Service

Deploy a LoadBalancer service to expose your vLLM deployment. This will provide an external IP address to send benchmark requests to.

```bash
kubectl apply -f service.yaml
```

After deployment, get the external IP of the service:

```bash
kubectl get service tpu-offline-inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

This command will directly output the external IP address. It might take a few minutes for the IP to be provisioned.

## 4. Run Benchmark

Instead of installing SGLang locally, we can run the benchmark from within the Kubernetes cluster using a dedicated pod. This approach avoids local dependency management and ensures the benchmark runs in a consistent environment.

### a. Configure the Benchmark Pod

A sample pod specification is provided in `benchmark-pod.yaml`. Before deploying it, you need to configure the environment variables within the file, especially the `IP` of the vLLM service.

Open `benchmark-pod.yaml` and replace `<Your service EXTERNAL-IP>` with the actual external IP address of your `tpu-offline-inference` service obtained in step 3.

You can also adjust the following benchmark parameters via environment variables in the `benchmark-pod.yaml` file:

* `GSP_NUM_GROUPS`: The number of unique system prompts.
* `GSP_PROMPTS_PER_GROUP`: The number of questions per system prompt.
* `GSP_SYSTEM_PROMPT_LEN`: The token length of the system prompt.
* `GSP_QUESTION_LEN`: The token length of the question.
* `GSP_OUTPUT_LEN`: The desired output token length.
* `MODEL`: The model to benchmark.

### b. Deploy the Benchmark Pod

Once configured, deploy the benchmark pod:

```bash
kubectl apply -f benchmark-pod.yaml
```

The pod will start, clone the SGLang repository, install dependencies, and run the benchmark.

### c. Monitor the Benchmark

You can monitor the progress of the benchmark by checking the logs of the pod:

```bash
kubectl logs -f sglang-benchmark
```

The pod is configured with `restartPolicy: Never`, so it will run the benchmark once and then complete.

## 5. Understanding `generated-shared-prefix` Dataset

The `generated-shared-prefix` dataset is designed to benchmark serving performance for workloads where multiple requests share a common, long prefix. This is common in applications using system prompts or few-shot examples.

**How it works:**

1. **System Prompt Generation:** A specified number of unique "system prompts" are generated. Each is a long sequence of random tokens.
2. **Question Generation:** Shorter "questions" (random tokens) are generated.
3. **Prompt Combination:** Each system prompt is combined with multiple unique questions to form final prompts. This creates groups of prompts where each prompt in a group shares the exact same system prompt as a prefix.
4. **Request Creation:** Each final prompt is packaged with its desired output length.
5. **Shuffling:** The entire set of generated requests is randomly shuffled. This interleaves requests from different groups, simulating realistic traffic where shared prefixes are not necessarily processed sequentially.
6. **Caching:** The generated dataset is cached locally for faster subsequent runs with the same parameters.

**Key Parameters for `generated-shared-prefix`:**

* `--gsp-num-groups`: The number of unique system prompts to generate. Each system prompt forms a "group" of requests.
* `--gsp-prompts-per-group`: The number of unique questions that will be appended to each system prompt. This determines how many requests will share a given system prompt.
* `--gsp-system-prompt-len`: The length (in tokens) of each generated system prompt.
* `--gsp-question-len`: The length (in tokens) of each generated question.
* `--gsp-output-len`: The desired length (in tokens) of the generated output for each request.
* `--seed`: (Optional) An integer seed for random number generation, ensuring reproducible prompt generation and request shuffling across runs.
