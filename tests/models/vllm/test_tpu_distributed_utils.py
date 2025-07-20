# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import jax
import pytest
import torch
import torchax
from jax.sharding import Mesh, PartitionSpec

from tpu_commons.distributed.tpu_distributed_utils import (
    create_torchax_kv_cache, create_torchax_tensor_with_partition_spec)

torchax.enable_globally()


@pytest.fixture(autouse=True)
def set_tpu_backend_env():
    """Set TPU_BACKEND_TYPE=torchax for all tests in this module."""
    with patch.dict(os.environ, {"TPU_BACKEND_TYPE": "torchax"}):
        yield


@pytest.mark.parametrize("mesh,partition_spec", [
    (None, None),
    (Mesh(jax.devices(),
          axis_names=('x', )), PartitionSpec(None, None, 'x', None)),
])
def test_create_torchax_kv_cache(mesh, partition_spec):
    kv_cache_shape = (1024, 16, 16, 128)
    dtype = torch.bfloat16
    tensor = create_torchax_kv_cache(kv_cache_shape, dtype, mesh,
                                     partition_spec)

    # Check the properties of the created tensor
    assert isinstance(tensor, torchax.tensor.Tensor)
    assert tensor.shape == kv_cache_shape
    assert tensor.dtype == dtype


@pytest.mark.parametrize("mesh,partition_spec", [
    (None, None),
    (Mesh(jax.devices(), axis_names=('x', )), PartitionSpec('x', )),
])
def test_create_torchax_tensor_with_partition_spec(mesh, partition_spec):
    shape = (1024, 1024)
    dtype = torch.bfloat16
    torch_t = torch.empty(shape, dtype=dtype)

    # Create a Torchax tensor with the specified partition spec
    tensor = create_torchax_tensor_with_partition_spec(torch_t, mesh,
                                                       partition_spec)

    # Check the properties of the created tensor
    assert isinstance(tensor, torchax.tensor.Tensor)
    assert tensor.shape == shape
    assert tensor.dtype == dtype


def test_create_torchax_tensor_with_partition_spec_value_error():
    """Test that ValueError is raised when mesh is None but partition_spec is not None/empty."""
    shape = (512, 256)
    dtype = torch.float32
    torch_t = torch.randn(shape, dtype=dtype)

    # Test with mesh=None and partition_spec not None
    with pytest.raises(
            ValueError,
            match="If mesh is None, partition_spec must also be None or empty"
    ):
        create_torchax_tensor_with_partition_spec(torch_t,
                                                  mesh=None,
                                                  partition_spec=('x', ))
