# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import hashlib
import os
import shutil
import subprocess
from typing import List, Optional

import filelock
import huggingface_hub.constants
from huggingface_hub import HfFileSystem, snapshot_download
from tqdm.auto import tqdm

from tpu_inference.logger import init_logger

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


def download_model_weights_from_hf(model_path: str, cache_dir: Optional[str],
                                   weights_format: str) -> str:
    with get_lock(model_path):
        local_dir = snapshot_download(
            model_path,
            cache_dir=cache_dir,  # can be specified by HF_HOME or HF_HUB_CACHE
            allow_patterns=weights_format,
            tqdm_class=DisabledTqdm,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
    local_files = list_files(local_dir, weights_format)
    return local_files
