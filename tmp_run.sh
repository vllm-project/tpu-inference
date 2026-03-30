#!/bin/bash
set -ex

gcloud compute tpus tpu-vm scp --recurse tpu_inference muskansh-v6e-1:tpu-inference/ --zone us-central2-b --project tpu-prod-env-one-vm
gcloud compute tpus tpu-vm scp --recurse examples muskansh-v6e-1:tpu-inference/ --zone us-central2-b --project tpu-prod-env-one-vm

cat << 'EOF' > remote_script.sh
cd tpu-inference
MODEL_IMPL_TYPE=vllm ../vllm_env/bin/python examples/multi_modal_inference.py --model Qwen/Qwen3-VL-8B-Instruct --test-multi-image --max-model-len 8192 --gpu-memory-utilization 0.85 2>&1 | tee output.log
EOF

gcloud compute tpus tpu-vm scp remote_script.sh muskansh-v6e-1:tpu-inference/ --zone us-central2-b --project tpu-prod-env-one-vm
gcloud compute tpus tpu-vm ssh muskansh-v6e-1 --zone us-central2-b --project tpu-prod-env-one-vm --command "bash tpu-inference/remote_script.sh"
