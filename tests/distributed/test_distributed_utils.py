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

from collections import namedtuple

import pytest

from tpu_inference.distributed.utils import get_device_topology_order_id

# Mock TpuDevice object to simulate the real one.
TpuDevice = namedtuple('TpuDevice',
                       ['id', 'process_index', 'coords', 'core_on_chip'])


def test_get_device_topology_order_id():
    """
    Tests the get_device_topology_order_id function with a mock topology.
    """
    # V7x
    global_devices = [
        TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0),
        TpuDevice(id=1, process_index=0, coords=(0, 0, 0), core_on_chip=1),
        TpuDevice(id=2, process_index=0, coords=(1, 0, 0), core_on_chip=0),
        TpuDevice(id=3, process_index=0, coords=(1, 0, 0), core_on_chip=1),
        TpuDevice(id=4, process_index=0, coords=(0, 1, 0), core_on_chip=0),
        TpuDevice(id=5, process_index=0, coords=(0, 1, 0), core_on_chip=1),
        TpuDevice(id=6, process_index=0, coords=(1, 1, 0), core_on_chip=0),
        TpuDevice(id=7, process_index=0, coords=(1, 1, 0), core_on_chip=1),
        TpuDevice(id=8, process_index=1, coords=(0, 0, 1), core_on_chip=0),
        TpuDevice(id=9, process_index=1, coords=(0, 0, 1), core_on_chip=1),
        TpuDevice(id=10, process_index=1, coords=(1, 0, 1), core_on_chip=0),
        TpuDevice(id=11, process_index=1, coords=(1, 0, 1), core_on_chip=1),
        TpuDevice(id=12, process_index=1, coords=(0, 1, 1), core_on_chip=0),
        TpuDevice(id=13, process_index=1, coords=(0, 1, 1), core_on_chip=1),
        TpuDevice(id=14, process_index=1, coords=(1, 1, 1), core_on_chip=0),
        TpuDevice(id=15, process_index=1, coords=(1, 1, 1), core_on_chip=1),
    ]

    local_devices_1 = global_devices[:8]
    local_devices_2 = global_devices[8:]

    assert get_device_topology_order_id(local_devices_1, global_devices) == 0
    assert get_device_topology_order_id(local_devices_2, global_devices) == 1

    # Test with unsorted in global_devices
    shuffled_z_global_devices = [
        TpuDevice(id=8, process_index=1, coords=(0, 0, 1), core_on_chip=0),
        TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0),
    ]
    local_devices_z1 = [
        TpuDevice(id=8, process_index=1, coords=(0, 0, 1), core_on_chip=0)
    ]
    local_devices_z0 = [
        TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0)
    ]

    assert get_device_topology_order_id(local_devices_z0,
                                        shuffled_z_global_devices) == 0
    assert get_device_topology_order_id(local_devices_z1,
                                        shuffled_z_global_devices) == 1

    #v6e
    global_devices = [
        TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0),
        TpuDevice(id=1, process_index=1, coords=(1, 0, 0), core_on_chip=0),
        TpuDevice(id=2, process_index=2, coords=(0, 1, 0), core_on_chip=0),
        TpuDevice(id=3, process_index=3, coords=(1, 1, 0), core_on_chip=0)
    ]
    local_devices = [
        TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0)
    ]
    assert get_device_topology_order_id(local_devices, global_devices) == 0

    local_devices = [
        TpuDevice(id=1, process_index=1, coords=(1, 0, 0), core_on_chip=0)
    ]
    assert get_device_topology_order_id(local_devices, global_devices) == 2

    local_devices = [
        TpuDevice(id=2, process_index=2, coords=(0, 1, 0), core_on_chip=0)
    ]
    assert get_device_topology_order_id(local_devices, global_devices) == 1

    local_devices = [
        TpuDevice(id=3, process_index=3, coords=(1, 1, 0), core_on_chip=0)
    ]
    assert get_device_topology_order_id(local_devices, global_devices) == 3


def test_get_device_topology_order_id_empty_local():
    """
    Tests that a ValueError is raised for empty local_devices.
    """
    with pytest.raises(ValueError, match="local_devices cannot be empty"):
        get_device_topology_order_id([], [])


def test_get_device_topology_order_id_not_in_global():
    """
    Tests that a ValueError is raised if local z-coordinate is not in global list.
    """
    global_devices = [
        TpuDevice(id=0, process_index=0, coords=(0, 0, 0), core_on_chip=0),
    ]
    local_devices = [
        TpuDevice(id=1, process_index=1, coords=(0, 0, 1), core_on_chip=0),
    ]
    with pytest.raises(ValueError, match="do not exist in the global device:"):
        get_device_topology_order_id(local_devices, global_devices)
