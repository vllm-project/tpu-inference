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
import re

import requests

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

GOOGLE_PCI_VENDOR_ID = 0x1AE0
_PCI_BDF_PATTERN = re.compile(
    r"^(?:([0-9a-fA-F]+):)?([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)$")

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


def _parse_pci_bdf(pci_bdf: str) -> tuple[int, int, int, int]:
    match = _PCI_BDF_PATTERN.match(pci_bdf)
    if not match:
        return (0, 0, 0, 0)
    domain, bus, slot, func = match.groups()
    return (
        int(domain, 16) if domain else 0,
        int(bus, 16),
        int(slot, 16),
        int(func, 16),
    )


def get_ordered_chip_ids() -> list[int]:
    """Returns TPU_VISIBLE_CHIPS indices ordered by physical tray (NUMA, PCI BDF).

    On multi-tray TPU VMs such as ct6e-standard-8t (2x4), libtpu enumerates
    devices in /dev/vfio/[0-9]* order, whereas Linux guest PCI bus probing can
    interleave IOMMU group IDs across the two PCI root complexes / NUMA nodes.
    Sorting the libtpu scanner indices by (numa_node, pci_bdf) groups each 4-chip
    2x2 tray contiguously ([0:4] and [4:8]).
    """
    accel_files = glob.glob("/dev/accel*")
    if accel_files:
        return list(range(len(accel_files)))

    vfio_paths = sorted(glob.glob("/dev/vfio/[0-9]*"))
    if not vfio_paths:
        return []

    chip_entries: list[tuple[int, tuple[int, int, int, int], int]] = []
    scanner_id = 0
    for vfio_path in vfio_paths:
        group = os.path.basename(vfio_path)
        if not group.isdigit() or not os.access(vfio_path, os.R_OK):
            continue

        iommu_devices_dir = f"/sys/kernel/iommu_groups/{group}/devices"
        try:
            pci_devices = sorted(os.listdir(iommu_devices_dir))
        except OSError:
            return list(range(get_num_chips()))
        if not pci_devices:
            return list(range(get_num_chips()))

        pci_bdf = pci_devices[0]
        vendor_path = f"/sys/bus/pci/devices/{pci_bdf}/vendor"
        try:
            with open(vendor_path, encoding="utf-8") as vf:
                vendor_id = int(vf.read().strip(), 16)
            if vendor_id != GOOGLE_PCI_VENDOR_ID:
                continue
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            return list(range(get_num_chips()))

        numa_node = 0
        numa_path = f"/sys/bus/pci/devices/{pci_bdf}/numa_node"
        try:
            with open(numa_path, encoding="utf-8") as nf:
                numa_node = max(0, int(nf.read().strip()))
        except (OSError, ValueError):
            numa_node = 0

        chip_entries.append((numa_node, _parse_pci_bdf(pci_bdf), scanner_id))
        scanner_id += 1

    chip_entries.sort()
    return [entry_scanner_id for _, _, entry_scanner_id in chip_entries]


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


def get_tpu_vmem_size_bytes() -> int:
    """Returns the TPU Vector Memory (VMEM) size in Bytes per Tensor Core.

    Defaults to 0 if the capacity cannot be programmatically retrieved.
    """
    try:
        from jax.experimental.pallas import tpu as pltpu
        return pltpu.get_tpu_info().vmem_capacity_bytes
    except Exception as e:
        logger.warning("Failed to programmatically get TPU VMEM size: %s", e)
        return 0
