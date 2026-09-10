# Copyright 2026 Google LLC
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
"""Static config objects for Reduce-Scatter."""

import dataclasses
import math

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Config:
    # yapf: disable
    """Dimensions and sharding sizes.

  ================================================================================================
              FIGURE 1: PHASE 1 TENSOR PARTITIONING (PIPELINE MICRO-BATCHES)
  ================================================================================================
              |<--------------------------------- hidden_dim_size -------------------------------->|
              |                                         |                                          |
  Iteration:  |<---------------- mb_size -------------->|<---------------- mb_size --------------->| (x num_micro_batches)
              +-----------------------------------------+------------------------------------------+ ---
  chunk_    ^ |                                         |                                          |  ^
  size      | |         Device 0, Microbatch 0          |          Device 0, Microbatch 1          |  |
            v +-----------------------------------------+------------------------------------------+  | num_
  chunk_    ^ |                                         |                                          |  | tokens
  size      | |         Device 1, Microbatch 0          |          Device 1, Microbatch 1          |  |
            v +-----------------------------------------+------------------------------------------+  |
              |                  ...                    |                    ...                   |  |
              +-----------------------------------------+------------------------------------------+  v
                                                                                                     ---
  ================================================================================================

  ================================================================================================
            FIGURE 2: MACRO PHASE 2 TENSOR PARTITIONING (ICI CROSS-CHIP SCATTER)
  ================================================================================================
  Zooming into a SINGLE Micro-batch column (`mb_size`) to show how Phase 2 slices the chunks further
  for Reduce-Scatter across the i-th hypercube network dimensions.

              |<-------------------------------- mb_size ---------------------------------------->|
              |                                         |                                         |
  ICI Phase 2:|<------------ hc_chunk_size ------------>|<------------ hc_chunk_size ------------>| (x num_hcube_dims)
              +-----------------------------------------+-----------------------------------------+ ---
  chunk_    ^ |                                         |                                         |  ^
  size      | |    Device 0, RS through 0th axis        |    Device 0, RS through 1st axis        |  |
            v +-----------------------------------------+-----------------------------------------+  | num_
  chunk_    ^ |                                         |                                         |  | tokens
  size      | |    Device 1, RS through 0th axis        |    Device 1, RS through 1st axis        |  |
            v +-----------------------------------------+-----------------------------------------+  |
              |                  ...                    |                    ...                  |  |
              +-----------------------------------------+-----------------------------------------+  v
  ================================================================================================

  ================================================================================================
                  FIGURE 3: CORE AND SUBCORE PARTITIONING
  ================================================================================================
  Zooming into a SINGLE grid cell (one Device's Chunk) to show how it physically maps
  onto the cores and subcores for accumulation.

              |<-------------------------- mb_size or hc_chunk_size -------------------------->|
              |                                                                                |
  Subcore Col:|<-- {p1,p2}_col_cs ->|<-- {p1,p2}_col_cs ->|<-- {p1,p2}_col_cs ->|     ...      |
              +=====================+=====================+=====================+==============+ ---
  subcore_  ^ |                     |                     |                     |              |  ^
  chunk_    | |  Core 0, Subcore 0  |  Core 0, Subcore 0  |  Core 0, Subcore 0  |              |  |
  size      v +---------------------+---------------------+---------------------+--------------+  |
  subcore_  ^ |                     |                     |                     |              |  | core_
  chunk_    | |  Core 0, Subcore 1  |  Core 0, Subcore 1  |  Core 0, Subcore 1  |              |  | chunk_
  size      v +---------------------+---------------------+---------------------+--------------+  | size
              |         ...         |         ...         |         ...         |              |  |
              +=====================+=====================+=====================+==============+  v
  subcore_  ^ |                     |                     |                     |              |  ^
  chunk_    | |  Core 1, Subcore 0  |  Core 1, Subcore 0  |  Core 1, Subcore 0  |              |  |
  size      v +---------------------+---------------------+---------------------+--------------+  |
  subcore_  ^ |                     |                     |                     |              |  | core_
  chunk_    | |  Core 1, Subcore 1  |  Core 1, Subcore 1  |  Core 1, Subcore 1  |              |  | chunk_
  size      v +---------------------+---------------------+---------------------+--------------+  | size
              |         ...         |         ...         |         ...         |              |  |
              +=====================+=====================+=====================+==============+  v
  ================================================================================================
  LEGEND:
  - mb_size            = hidden_dim_size // num_micro_batches
  - hc_chunk_size      = mb_size // num_hcube_dims
  - core_chunk_size    = chunk_size // num_cores
  - subcore_chunk_size = core_chunk_size // num_subcores_row
  - p1_col_cs          = mb_size // num_subcores_col
  - p2_col_cs          = hc_chunk_size // num_subcores_col
  ================================================================================================
  """
    # yapf: enable

    # Total number of devices executing this kernel
    num_devices: int
    # Total hidden size dimension (e.g., 4096)
    hidden_dim_size: int
    # Local sequence slice per device (= num_tokens // num_devices)
    chunk_size: int
    # Number of tokens
    num_tokens: int
    # Input data type (e.g., bfloat16)
    dtype: jnp.dtype
    # Pipelining unrolling factor for overlapping ALU/DMA. If None, determined by
    # heuristic.
    _num_micro_batches: int | None = None

    def __post_init__(self):
        assert self.cores_per_chip == 2, (
            "This kernel architecture strictly supports 2 cores per chip, but"
            f" found {self.cores_per_chip}.")
        assert (self.num_chips & (self.num_chips - 1)
                ) == 0, f"num_chips {self.num_chips} must be a power of 2"
        assert (self.num_hcube_dims
                >= 1), f"num_hcube_dims {self.num_hcube_dims} must be >= 1"
        assert self.hidden_dim_size % self.num_micro_batches == 0, (
            f"hidden_dim_size {self.hidden_dim_size} must be divisible by "
            f"num_micro_batches {self.num_micro_batches}")
        assert self.sc_info is not None, "Cannot find sc_info"

    @property
    def cores_per_chip(self) -> int:
        """Number of physical tensor cores per chip on the current TPU architecture."""
        return pltpu.get_tpu_info(
        ).chip_version.num_physical_tensor_cores_per_chip

    @property
    def num_chips(self) -> int:
        """Number of physical TPU chips in the mesh (num_devices // cores_per_chip)"""
        return self.num_devices // self.cores_per_chip

    @property
    def packing_factor(self) -> int:
        """Returns the number of array elements packed into a single 32-bit (4-byte) word."""
        return 4 // self.dtype.itemsize

    @property
    def num_hcube_dims(self) -> int:
        """ICI hypercube logical network dimensions (log2(num_chips))"""
        return int(math.log2(self.num_chips))

    @property
    def num_micro_batches(self) -> int:
        """Pipelining unrolling factor for overlapping ALU/DMA.

    If not set, use the best value based on the empirical results.
    """
        if self._num_micro_batches is not None:
            return self._num_micro_batches
        if self.num_tokens >= 4096:
            return 8
        elif self.num_tokens >= 2048:
            return 4
        elif self.num_tokens >= 256:
            return 2
        else:
            return 1

    @property
    def mb_size(self) -> int:
        """Micro batch slice size"""
        return self.hidden_dim_size // self.num_micro_batches

    @property
    def hc_chunk_size(self) -> int:
        """Phase 2 (C2C) hypercube chunk slice size"""
        return self.mb_size // self.num_hcube_dims

    @property
    def sc_info(self):
        return pltpu.get_tpu_info().sparse_core

    @property
    def num_cores(self) -> int:
        # Use all cores to maximize aggregate HBM memory bandwidth.
        return self.sc_info.num_cores

    @property
    def num_subcores_col(self) -> int:
        """Number of subcore columns used for column-wise hidden size partitioning"""
        # Shard column as much as possible as long as the chunk's width >= 128, which is for dma alignment.
        return min(self.sc_info.num_lanes, self.hc_chunk_size // 128)

    @property
    def num_subcores_row(self) -> int:
        """Number of subcore rows used for row-wise sequence partitioning"""
        # Rows are sharded both core and remaining subcores.
        return min(
            self.sc_info.num_lanes // self.num_subcores_col,
            self.chunk_size // (self.num_cores * self.packing_factor),
        )

    @property
    def num_subcores(self) -> int:
        return self.num_subcores_row * self.num_subcores_col

    @property
    def core_chunk_size(self) -> int:
        """Sequence slice size assigned to each physical core on a device"""
        return self.chunk_size // self.num_cores

    @property
    def subcore_chunk_size(self) -> int:
        """Sequence slice size assigned to each subcore row"""
        return self.core_chunk_size // self.num_subcores_row

    @property
    def subcore_col_chunk_size_p1(self) -> int:
        """Column slice size for Phase 1 DMA"""
        return self.mb_size // self.num_subcores_col

    @property
    def subcore_col_chunk_size_p2(self) -> int:
        """Column slice size for Phase 2 DMA"""
        return self.hc_chunk_size // self.num_subcores_col
