from tpu_commons.logger import init_logger

logger = init_logger(__name__)

MODEL_MATMUL_FUSION_TRUTH_TABLE = {
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 1, "QKVParallelLinear"):
    True,
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 1, "MergedColumnParallelLinear"):
    False,
    ("Qwen/Qwen2.5-7B-Instruct", 2048, 1, "QKVParallelLinear"):
    False,
    ("Qwen/Qwen2.5-7B-Instruct", 2048, 1, "MergedColumnParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 1024, 1, "QKVParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 1024, 1, "MergedColumnParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 2048, 1, "QKVParallelLinear"):
    False,
    ("meta-llama/Llama-3.1-8B-Instruct", 2048, 1, "MergedColumnParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 1024, 1, "QKVParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 1024, 1, "MergedColumnParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 2048, 1, "QKVParallelLinear"):
    False,
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", 2048, 1, "MergedColumnParallelLinear"):
    False,
}


def get_model_matmul_fusion_assignment(model_name: str, batch_size: int,
                                       tp_size: int, layer_name: str):
    key = (model_name, batch_size, tp_size, layer_name)
    fuse_matmuls = MODEL_MATMUL_FUSION_TRUTH_TABLE.get(key, None)
    if fuse_matmuls is None:
        logger.info(f"No matmul fusion overwrite for {key=}, default to True.")
        return True
    else:
        return fuse_matmuls
