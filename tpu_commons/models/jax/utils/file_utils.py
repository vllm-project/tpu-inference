import base64
import glob
import hashlib
import os
import shutil
import subprocess
from typing import List

import filelock
import huggingface_hub.constants
from huggingface_hub import HfFileSystem, snapshot_download
from tqdm.auto import tqdm

from tpu_commons.logger import init_logger

logger = init_logger(__name__)
# Do not set the HuggingFace token here, it should be set via the env `HF_TOKEN`.
hfs = HfFileSystem()

LOCK_DIR = "/tmp/lock"


#####  Local file utils  #####
def run_cmd(cmd: str, *args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd.split(), *args, **kwargs)


def delete_file(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)
    else:
        logger.error(f"Trying to delete non-existing file: {path}")


def list_files(dir: str, pattern: str = "*") -> List[str]:
    files = glob.glob(os.path.join(dir, pattern))
    return files


def get_lock(model_name_or_path: str):
    lock_dir = LOCK_DIR
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock


def get_free_disk_size(path: str = "/") -> int:
    free_bytes = shutil.disk_usage(path)[2]
    return free_bytes


#####  GCS file utils  #####

# TODO(xiang): Unify GCS path and HF path.

GCS_PREFIX = "gs://"
LOCAL_MODEL_DIR = os.getenv("HF_HOME", "/tmp/model")
LOCAL_LORA_DIR = "/tmp/lora"


def _get_local_model_dir(remote_dir: str) -> str:
    return os.path.join(LOCAL_MODEL_DIR, os.path.basename(remote_dir))


def _encode_names(names: List[str]) -> str:
    encoded = "#".join(names).encode()
    encoded = base64.b64encode(encoded).decode()
    return encoded[:8]


def file_lock(local_name) -> filelock.FileLock:
    lock_file = os.path.join(LOCK_DIR, local_name + ".lock")
    lock = filelock.FileLock(lock_file)
    return lock


def is_gcs_path(input_path: str) -> bool:
    return input_path.startswith(GCS_PREFIX)


def list_gcs_dir(gcs_dir: str, pattern: str = "*") -> List[str]:
    gcs_files = run_cmd(
        f"gcloud storage ls {os.path.join(gcs_dir, pattern)}",
        capture_output=True,
        encoding="utf-8",
    ).stdout.split()
    return gcs_files


def download_gcs_dir(gcs_dir: str, local_dir: str, pattern: str = "*") -> None:
    with file_lock(_encode_names([gcs_dir, local_dir, pattern])):
        os.makedirs(local_dir, exist_ok=True)
        gcs_files = os.path.join(gcs_dir, pattern)
        # The `-n` skips existing files.
        run_cmd(f"gcloud storage cp -n {gcs_files} {local_dir}/", check=True)


def download_config_from_gcs(model_path: str) -> str:
    local_dir = _get_local_model_dir(model_path)
    download_gcs_dir(model_path, local_dir, "config.json")
    return local_dir


def download_preprocessor_config_from_gcs(model_path: str) -> str:
    local_dir = _get_local_model_dir(model_path)
    download_gcs_dir(model_path, local_dir, "preprocessor_config.json")
    return os.path.join(local_dir, "preprocessor_config.json")


def download_tokenizer_from_gcs(tokenizer_path: str) -> str:
    local_dir = _get_local_model_dir(tokenizer_path)
    download_gcs_dir(tokenizer_path, local_dir, "tokenizer*")
    download_gcs_dir(tokenizer_path, local_dir, "special_tokens_map.json")
    return local_dir


def get_gcs_model_weights_size(model_path: str, weights_format: str) -> int:
    ret = run_cmd(
        f"gsutil du -s {os.path.join(model_path, weights_format)}",
        capture_output=True,
        encoding="utf-8",
    )
    size = int(ret.stdout.split(GCS_PREFIX)[0])
    return size


def download_model_weights_from_gcs(model_path: str,
                                    weights_format: str) -> str:
    local_dir = _get_local_model_dir(model_path)
    download_gcs_dir(model_path, local_dir, weights_format)
    local_files = list_files(local_dir, weights_format)
    return local_files


#####  HuggingFace file utils  #####


def is_hf_repo(repo_id: str) -> bool:
    return hfs.exists(repo_id)


def list_hf_repo(repo_id: str, pattern: str = "**") -> List[str]:
    repo_files = hfs.glob(os.path.join(repo_id, pattern))
    return repo_files


def get_hf_model_weights_size(repo_id: str, weights_format: str) -> int:
    weights_paths = list_hf_repo(repo_id, weights_format)
    weights_size = 0
    for weights_path in weights_paths:
        weights_size += int(hfs.info(weights_path)["size"])
    return weights_size


class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def download_model_weights_from_hf(model_path: str,
                                   weights_format: str) -> str:
    with get_lock(model_path):
        local_dir = snapshot_download(
            model_path,
            cache_dir=None,  # can be specified by HF_HOME or HF_HUB_CACHE
            allow_patterns=weights_format,
            tqdm_class=DisabledTqdm,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
    local_files = list_files(local_dir, weights_format)
    return local_files
