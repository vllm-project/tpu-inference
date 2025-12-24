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
import os

import requests

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/")
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}


def get_tpu_metadata(key: str = "") -> str:
    try:
        accelerator_type_request = requests.get(
            os.path.join(GCE_TPU_ACCELERATOR_ENDPOINT, key),
            headers=GCE_TPU_HEADERS,
        )
        if (accelerator_type_request.status_code == 200
                and accelerator_type_request.text):
            return accelerator_type_request.text
        else:
            logger.error(
                "Unable to poll TPU GCE Metadata. Got "
                f"status code: {accelerator_type_request.status_code} and "
                f"content: {accelerator_type_request.text}")
    except requests.RequestException as e:
        logger.error("Unable to poll the TPU GCE Metadata: %s", e)
    return None


def get_tpu_type() -> str:
    tpu_type = envs.TPU_ACCELERATOR_TYPE
    if tpu_type is None:
        tpu_type = get_tpu_metadata(key="accelerator-type")
    return tpu_type


def get_node_name() -> str:
    tpu_name = envs.TPU_NAME
    if not tpu_name:
        tpu_name = get_tpu_metadata(key="instance-id")
    return tpu_name


def get_node_worker_id() -> int:
    """For multi-host TPU VM, this returns the worker id for the current node."""
    worker_id = envs.TPU_WORKER_ID
    if worker_id is None:
        worker_id = get_tpu_metadata(key="agent-worker-number")
    if worker_id is None:
        return 0
    return int(worker_id)


def get_num_cores_per_chip() -> int:
    tpu_type = get_tpu_type()
    if tpu_type.startswith(("v5litepod", "v6e")):
        return 1
    return 2


def get_num_chips() -> int:
    accel_files = glob.glob("/dev/accel*")
    if accel_files:
        return len(accel_files)
    try:
        vfio_entries = os.listdir("/dev/vfio")
        numeric_entries = [
            int(entry) for entry in vfio_entries if entry.isdigit()
        ]
        return len(numeric_entries)
    except FileNotFoundError as e:
        logger.error("Failed to detect number of TPUs: %s", e)
        return 0
