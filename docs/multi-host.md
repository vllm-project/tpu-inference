# Run JAX model with Ray-based multi-host serving

Run `Llama 3.1 70B Instruct` offline inference on 4 hosts (v6e-16) in multi-host mode:

1. Deploy Ray cluster and containers:

```
~/tpu_commons/scripts/multihost/deploy_cluster.sh \
    -s ~/tpu_commons/scripts/multihost/run_cluster.sh \
    -d "<your_docker_image>" \
    -c "<path_on_remote_hosts_for_hf_cache>" \
    -t "<your_hugging_face_token>" \
    -H "<head_node_public_ip>" \
    -i "<head_node_private_ip>" \
    -W "<worker1_public_ip>,<worker2_public_ip>,<etc...>"
```

This script will build the Ray cluster and install/run the container image on every single host.

1. On the head node, use `sudo docker exec -it node /bin/bash` to enter the container. And then execute:

```
HF_TOKEN=<huggingface_token> python /workspace/tpu_commons/examples/offline_inference.py \
    --model=meta-llama/Llama-3.1-70B  \
    --tensor_parallel_size=16  \
    --max_model_len=1024
```
