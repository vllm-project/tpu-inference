"""Utilities for downloading model weights from HuggingFace."""

import glob
import os
from typing import Any, Generator, Optional

import jax
from safetensors import safe_open

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.utils import file_utils

logger = init_logger(__name__)

HF_WEIGHTS_FORMAT = "*.safetensors"
FULL_DOWNLOAD_DISK_RATIO = 0.9


def hf_model_weights_iterator(
    model_name: str,
    framework: str,
    cache_dir: Optional[str] = None,
) -> Generator[tuple, Any, None]:
    weights_files = []
    weights_location = "local"
    if os.path.isdir(model_name):
        weights_files = glob.glob(os.path.join(model_name, HF_WEIGHTS_FORMAT))
    elif file_utils.is_gcs_path(model_name):
        local_free_disk_size = file_utils.get_free_disk_size()
        model_size = file_utils.get_gcs_model_weights_size(
            model_name, HF_WEIGHTS_FORMAT)
        if model_size < local_free_disk_size * FULL_DOWNLOAD_DISK_RATIO:
            logger.info(f"Downloading weights from GCS {model_name}")
            weights_files = file_utils.download_model_weights_from_gcs(
                model_name, HF_WEIGHTS_FORMAT)
        else:
            weights_files = file_utils.list_gcs_dir(model_name,
                                                    HF_WEIGHTS_FORMAT)
            weights_location = "gcs"
    elif file_utils.is_hf_repo(model_name):
        local_free_disk_size = file_utils.get_free_disk_size()
        model_size = file_utils.get_hf_model_weights_size(
            model_name, HF_WEIGHTS_FORMAT)
        if model_size < local_free_disk_size * FULL_DOWNLOAD_DISK_RATIO:
            logger.info(f"Downloading weights from HF {model_name}")
            weights_files = file_utils.download_model_weights_from_hf(
                model_name, HF_WEIGHTS_FORMAT)
        else:
            weights_files = file_utils.list_hf_repo(model_name,
                                                    HF_WEIGHTS_FORMAT)
            weights_location = "hf"
    else:
        raise ValueError(
            f"{model_name} must be a local path, or a gcs path, or a HF model id."
        )

    if len(weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any {HF_WEIGHTS_FORMAT} files in {model_name}.")

    if weights_location != "local":
        logger.warning(
            "Weights files are not downloaded to local disk at once due to insufficient disk space. "
            "They will be downloaded on the fly during loading.")

    # Sort to ensure the order of files is consistent.
    weights_files.sort()

    for st_file in weights_files:
        logger.info(f"Loading weights from {st_file}")
        if weights_location == "gcs":
            st_file = file_utils.download_model_weights_from_gcs(
                model_name, os.path.basename(st_file))[0]
        elif weights_location == "hf":
            st_file = file_utils.download_model_weights_from_hf(
                model_name, os.path.basename(st_file))[0]
        # NOTE: We enforce loading tensors on CPU here.
        # Because otherwise the tensor will be loaded on TPU:0 by default,
        # although the tensor would eventually be sharded across multiple TPUs,
        # it would lead to OOM on TPU:0 for large models.
        with jax.default_device(jax.devices("cpu")[0]):
            with safe_open(st_file, framework=framework) as f:
                for name in f.keys():
                    weight_tensor = f.get_tensor(name)
                    yield name, weight_tensor
        if weights_location != "local":
            file_utils.delete_file(st_file)


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


def get_num_q_heads_by_tp(num_q_heads: int, num_kv_heads: int,
                          tp_size: int) -> int:
    num_kv_heads_by_tp = get_num_kv_heads_by_tp(num_kv_heads, tp_size)
    kv_repeats = num_kv_heads_by_tp // num_kv_heads
    q_repeats = 1
    if num_q_heads % tp_size != 0:
        if (num_q_heads * kv_repeats) % tp_size != 0:
            raise ValueError(
                f"Cannot make q_heads divisible by TP={tp_size} properly. Consider other padding strategies."
            )
        q_repeats = kv_repeats

    return q_repeats * num_q_heads
