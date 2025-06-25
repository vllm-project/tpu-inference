import base64
import glob
import hashlib
import os
import shutil
import subprocess
from typing import List

import filelock
from google.cloud import storage
from huggingface_hub import HfFileSystem, snapshot_download

from tpu_commons.logger import init_logger

logger = init_logger(__name__)
# Do not set the HuggingFace token here, it should be set via the env `HF_TOKEN`.
hfs = HfFileSystem()

GCS_PREFIX = "gs://"
LOCAL_MODEL_DIR = os.getenv("HF_HOME", "/tmp/model")
LOCAL_LORA_DIR = "/tmp/lora"
LOCK_DIR = "/tmp/lock"


#####  Local file utils  #####
def run_cmd(cmd: str, *args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd.split(), *args, **kwargs)


def _get_local_model_dir(remote_dir: str) -> str:
    return os.path.join(LOCAL_MODEL_DIR, os.path.basename(remote_dir))


def _encode_names(names: List[str]) -> str:
    encoded = "#".join(names).encode()
    encoded = base64.b64encode(encoded).decode()
    return encoded


def delete_file(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)
    else:
        logger.error(f"Trying to delete non-existing file: {path}")


def list_files(dir: str, pattern: str = "*") -> List[str]:
    files = glob.glob(os.path.join(dir, pattern))
    return files


def file_lock(local_name) -> filelock.FileLock:
    lock_file = os.path.join(LOCK_DIR, local_name + ".lock")
    lock = filelock.FileLock(lock_file)
    return lock


def get_free_disk_size(path: str = "/") -> int:
    free_bytes = shutil.disk_usage(path)[2]
    return free_bytes


def get_md5_hash_of_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).digest()
        return base64.b64encode(file_hash).decode()


def get_md5_hash_of_local_dir(local_dir: str) -> str:
    if not os.path.isdir(local_dir):
        raise ValueError(f"Local directory {local_dir} not found.")
    dir_hash = ""
    for file_path in sorted(glob.glob(os.path.join(local_dir, "*"))):
        if os.path.isfile(file_path):
            file_hash = get_md5_hash_of_file(file_path)
            dir_hash += f"{file_hash}_"
    return dir_hash


#####  GCS file utils  #####


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


def download_lora_weights_from_gcs(lora_path: str,
                                   download_folder: str) -> str:
    local_dir = os.path.join(LOCAL_LORA_DIR, download_folder)
    download_gcs_dir(lora_path, local_dir)
    return local_dir


# TODO: Re-implement this function using the `gcloud storage hash -m` command.
def get_md5_hash_of_gcs_dir(gcs_dir: str) -> str:
    # The GCS directory is structured as: gs://bucket_name/dir
    bucket_name = gcs_dir.split("/")[2]
    prefix = gcs_dir[len(GCS_PREFIX + bucket_name):].strip("/")
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)  # Ordered by name.
    if not blobs:
        raise ValueError(f"No GCS blobs found in {gcs_dir}.")
    dir_hash = ""
    for blob in blobs:
        if blob.name[-1] == "/":
            continue
        dir_hash += f"{blob.md5_hash}_"
    return dir_hash


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


def download_model_weights_from_hf(model_path: str,
                                   weights_format: str) -> str:
    local_dir = _get_local_model_dir(model_path)
    with file_lock(_encode_names([model_path, local_dir, weights_format])):
        snapshot_download(
            model_path,
            allow_patterns=weights_format,
            local_dir=local_dir,
        )
    local_files = list_files(local_dir, weights_format)
    return local_files


def download_preprocessor_config_from_hf(model_path: str) -> str:
    local_dir = _get_local_model_dir(model_path)
    with file_lock(
            _encode_names([model_path, local_dir,
                           "preprocessor_config.json"])):
        snapshot_download(
            model_path,
            allow_patterns="preprocessor_config.json",
            local_dir=local_dir,
        )
    local_files = list_files(local_dir, "preprocessor_config.json")
    return local_files[0]
