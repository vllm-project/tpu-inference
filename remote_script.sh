cd tpu-inference
MODEL_IMPL_TYPE=vllm ../vllm_env/bin/python examples/multi_modal_inference.py --model Qwen/Qwen3-VL-8B-Instruct --test-multi-image --max-model-len 8192 --gpu-memory-utilization 0.85 2>&1 | tee output.log
