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

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from tpu_inference.tpu_info import (get_node_name, get_node_worker_id,
                                    get_num_chips, get_num_cores_per_chip,
                                    get_ordered_chip_ids, get_tpu_metadata,
                                    get_tpu_type)


# Mock requests.get for get_tpu_metadata tests
@patch("tpu_inference.tpu_info.requests.get")
def test_get_tpu_metadata_success(mock_get):
    """Test get_tpu_metadata when the request is successful."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "test_metadata_value"
    mock_get.return_value = mock_response
    assert get_tpu_metadata(key="test-key") == "test_metadata_value"


@patch("tpu_inference.tpu_info.requests.get")
def test_get_tpu_metadata_request_error(mock_get):
    """Test get_tpu_metadata when a RequestException is raised."""
    mock_get.side_effect = requests.RequestException("Test RequestException")
    assert get_tpu_metadata(key="test-key") is None


# Test get_tpu_type
@patch("tpu_inference.tpu_info.get_tpu_metadata")
@patch.dict(os.environ, {"TPU_ACCELERATOR_TYPE": "env_tpu_type"})
def test_get_tpu_type_from_env(mock_get_tpu_metadata):
    """Test get_tpu_type when TPU_ACCELERATOR_TYPE is set in environment."""
    # The function should return the env var value and not call get_tpu_metadata
    assert get_tpu_type() == "env_tpu_type"
    mock_get_tpu_metadata.assert_not_called()


@patch.dict(os.environ, {}, clear=True)
@patch("tpu_inference.tpu_info.get_tpu_metadata",
       return_value="metadata_tpu_type")
def test_get_tpu_type_from_metadata(mock_get_tpu_metadata):
    """Test get_tpu_type when environment variable is not set."""
    assert get_tpu_type() == "metadata_tpu_type"
    mock_get_tpu_metadata.assert_called_once_with(key="accelerator-type")


# Test get_node_name
@patch("tpu_inference.tpu_info.get_tpu_metadata")
@patch.dict(os.environ, {"TPU_NAME": "env_tpu_name"})
def test_get_node_name_from_env(mock_get_tpu_metadata):
    """Test get_node_name when TPU_NAME is set in environment."""
    assert get_node_name() == "env_tpu_name"
    mock_get_tpu_metadata.assert_not_called()


@patch.dict(os.environ, {}, clear=True)
@patch("tpu_inference.tpu_info.get_tpu_metadata",
       return_value="metadata_tpu_name")
def test_get_node_name_from_metadata(mock_get_tpu_metadata):
    """Test get_node_name when environment variable is not set."""
    assert get_node_name() == "metadata_tpu_name"
    mock_get_tpu_metadata.assert_called_once_with(key="instance-id")


# Test get_node_worker_id
@patch("tpu_inference.tpu_info.get_tpu_metadata")
@patch.dict(os.environ, {"TPU_WORKER_ID": "5"})
def test_get_node_worker_id_from_env(mock_get_tpu_metadata):
    """Test get_node_worker_id when TPU_WORKER_ID is set in environment."""
    assert get_node_worker_id() == 5
    mock_get_tpu_metadata.assert_not_called()


@patch.dict(os.environ, {}, clear=True)
@patch("tpu_inference.tpu_info.get_tpu_metadata", return_value="10")
def test_get_node_worker_id_from_metadata(mock_get_tpu_metadata):
    """Test get_node_worker_id when environment variable is not set."""
    assert get_node_worker_id() == 10
    mock_get_tpu_metadata.assert_called_once_with(key="agent-worker-number")


# Test get_num_cores_per_chip
@pytest.mark.parametrize(
    "tpu_type, expected",
    [
        ("v5litepod-4", 1),
        ("v6e-8", 1),
        ("v4-8", 2),
        ("v5p-16", 2),
        ("unknown-type", 2)  # Default case
    ])
@patch("tpu_inference.tpu_info.get_tpu_type")
def test_get_num_cores_per_chip(mock_get_tpu_type, tpu_type, expected):
    """Test get_num_cores_per_chip with different TPU types."""
    mock_get_tpu_type.return_value = tpu_type
    assert get_num_cores_per_chip() == expected


# Test get_num_chips
@patch("tpu_inference.tpu_info.glob.glob",
       return_value=["/dev/accel0", "/dev/accel1"])
def test_get_num_chips_from_accel(mock_glob):
    """Test get_num_chips when /dev/accel* files exist."""
    assert get_num_chips() == 2


@patch("tpu_inference.tpu_info.glob.glob", return_value=[])
@patch("tpu_inference.tpu_info.os.listdir", return_value=["0", "1", "2"])
def test_get_num_chips_from_vfio(mock_listdir, mock_glob):
    """Test get_num_chips when /dev/accel* files don't exist but /dev/vfio entries do."""
    assert get_num_chips() == 3


@patch("tpu_inference.tpu_info.glob.glob", return_value=[])
@patch("tpu_inference.tpu_info.os.listdir", side_effect=FileNotFoundError)
def test_get_num_chips_not_found(mock_listdir, mock_glob, caplog):
    """Test get_num_chips when neither files nor directory are found."""
    assert get_num_chips() == 0


def test_get_ordered_chip_ids_from_accel():
    """Test get_ordered_chip_ids when /dev/accel* devices exist."""
    with patch("tpu_inference.tpu_info.glob.glob",
               return_value=["/dev/accel0", "/dev/accel1", "/dev/accel2"]):
        assert get_ordered_chip_ids() == [0, 1, 2]


def test_get_ordered_chip_ids_interleaved_vfio_trays(tmp_path):
    """Test get_ordered_chip_ids groups two 4-chip trays separated by NUMA/PCI."""
    # Simulate an 8-chip VM where /dev/vfio/0..7 interleave two PCI root buses:
    # - Tray 0 (NUMA 0, root bus 00): vfio groups 1, 2, 3, 6
    # - Tray 1 (NUMA 1, root bus 80): vfio groups 0, 4, 5, 7
    vfio_layout = [
        ("0", "0000:80:05.0", "0x1ae0", "1"),
        ("1", "0000:00:04.0", "0x1ae0", "0"),
        ("2", "0000:00:05.0", "0x1ae0", "0"),
        ("3", "0000:00:07.0", "0x1ae0", "0"),
        ("4", "0000:80:04.0", "0x1ae0", "1"),
        ("5", "0000:80:07.0", "0x1ae0", "1"),
        ("6", "0000:00:06.0", "0x1ae0", "0"),
        ("7", "0000:80:06.0", "0x1ae0", "1"),
        # Non-TPU VFIO device that libtpu's scanner skips before assigning IDs:
        ("8", "0000:00:03.0", "0x1af4", "0"),
    ]

    vfio_dir = tmp_path / "dev" / "vfio"
    vfio_dir.mkdir(parents=True)
    vfio_paths = []
    iommu_map = {}
    pci_map = {}
    for group, bdf, vendor, numa in vfio_layout:
        vfio_file = vfio_dir / group
        vfio_file.write_text("", encoding="utf-8")
        vfio_paths.append(str(vfio_file))
        iommu_map[f"/sys/kernel/iommu_groups/{group}/devices"] = [bdf]
        pci_dir = tmp_path / "pci" / bdf
        pci_dir.mkdir(parents=True)
        (pci_dir / "vendor").write_text(vendor + "\n", encoding="utf-8")
        (pci_dir / "numa_node").write_text(numa + "\n", encoding="utf-8")
        pci_map[f"/sys/bus/pci/devices/{bdf}/vendor"] = str(pci_dir / "vendor")
        pci_map[f"/sys/bus/pci/devices/{bdf}/numa_node"] = str(pci_dir /
                                                               "numa_node")

    real_open = open

    def fake_glob(pattern):
        if pattern == "/dev/accel*":
            return []
        if pattern == "/dev/vfio/[0-9]*":
            return list(reversed(vfio_paths))
        return []

    def fake_listdir(path):
        if path in iommu_map:
            return iommu_map[path]
        raise FileNotFoundError(path)

    def fake_open(path, *args, **kwargs):
        return real_open(pci_map.get(path, path), *args, **kwargs)

    with (
            patch("tpu_inference.tpu_info.glob.glob", side_effect=fake_glob),
            patch("tpu_inference.tpu_info.os.listdir",
                  side_effect=fake_listdir),
            patch("builtins.open", side_effect=fake_open),
    ):
        ordered = get_ordered_chip_ids()

    # Tray 0 sorted by BDF (04.0, 05.0, 06.0, 07.0) -> scanner IDs [1, 2, 6, 3]
    # Tray 1 sorted by BDF (04.0, 05.0, 06.0, 07.0) -> scanner IDs [4, 0, 7, 5]
    assert ordered == [1, 2, 6, 3, 4, 0, 7, 5]


def test_get_ordered_chip_ids_missing_sysfs_fallback(tmp_path):
    """Test get_ordered_chip_ids falls back to sequential IDs if sysfs is absent."""
    vfio_dir = tmp_path / "dev" / "vfio"
    vfio_dir.mkdir(parents=True)
    vfio_paths = []
    for idx in range(4):
        vfio_file = vfio_dir / str(idx)
        vfio_file.write_text("", encoding="utf-8")
        vfio_paths.append(str(vfio_file))

    def fake_glob(pattern):
        if pattern == "/dev/accel*":
            return []
        if pattern == "/dev/vfio/[0-9]*":
            return vfio_paths
        return []

    with (
            patch("tpu_inference.tpu_info.glob.glob", side_effect=fake_glob),
            patch("tpu_inference.tpu_info.os.listdir",
                  side_effect=FileNotFoundError),
            patch("tpu_inference.tpu_info.get_num_chips", return_value=4),
    ):
        assert get_ordered_chip_ids() == [0, 1, 2, 3]
