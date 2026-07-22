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
"""Topology abstraction and layout chunk computations, abstracted from the pipeline."""

import jax
from jax.experimental import pallas as pl

from tpu_inference.kernels.collectives.hierrs_sc.config import Config


class Topology:
    """Abstracts device indexing logic to find neighbors/partners."""

    def __init__(self, axis_name: str):
        self.cur_id = jax.lax.axis_index(axis_name)
        self.cur_chip_id = self.cur_id // 2
        self.cur_chiplet_bit = self.cur_id % 2
        self.partner_id = jax.lax.select(self.cur_chiplet_bit == 0,
                                         self.cur_id + 1, self.cur_id - 1)

    def get_device_id(self, chip_id, chiplet_bit):
        """Returns the global device ID from physical chip `chip_id` and chiplet coordinate `chiplet_bit` (0 or 1)."""
        return chip_id * 2 + chiplet_bit

    def get_neighbor_chip_id(self, dim):
        """Returns the physical chip ID of the logical neighbor in hypercube dimension `dim`.

    For example, on a 2D hypercube of 4 physical chips (IDs: 0, 1, 2, 3):
    - If current chip is 0 (binary 00):
      - Neighbor along dimension 0 is: 0 ^ (1 << 0) = 1 (binary 01).
      - Neighbor along dimension 1 is: 0 ^ (1 << 1) = 2 (binary 10).
    """
        return self.cur_chip_id ^ (1 << dim)

    def get_neighbor_device_id(self, dim):
        """Returns the ID of the neighbor device along hypercube dimension `dim` sharing the same chiplet position.

    For example, on a 2D hypercube of 4 chips (IDs 0-3) containing 8 logical
    devices (IDs 0-7):
    - If current device is 0 (physical chip 0, chiplet bit 0):
      - Neighbor along dimension 0 is: get_device_id(neighbor_chip=1, chiplet=0)
      = 2.
      - Neighbor along dimension 1 is: get_device_id(neighbor_chip=2, chiplet=0)
      = 4.
    """
        return self.get_device_id(self.get_neighbor_chip_id(dim),
                                  self.cur_chiplet_bit)


class ChunkLocator:
    """Encapsulates sequence and HBM indexing math for SparseCore Reduce-Scatter."""

    def __init__(
            self,
            config: Config,
            topo: Topology,
            core_idx: jax.Array,  # integer scalar.
            subcore_idx: jax.Array | None,  # integer scalar. None for SCS.
    ):
        self.config = config
        self.topo = topo
        self.core_idx = core_idx
        if subcore_idx is not None:
            self.subcore_row_idx = subcore_idx // config.num_subcores_col
            self.subcore_col_idx = subcore_idx % config.num_subcores_col
        else:
            self.subcore_row_idx = None
            self.subcore_col_idx = None
        self.mb_stride = config.num_hcube_dims * config.hc_chunk_size

    def _get_row_slice(self, chunk_idx, for_tec):
        """Returns a row slice for `chunk_idx` of core-level size,

    or subcore-level if `for_tec` is True.
    """
        row_offset = (chunk_idx * self.config.chunk_size +
                      self.core_idx * self.config.core_chunk_size)
        if for_tec:
            row_offset += self.subcore_row_idx * self.config.subcore_chunk_size
            row_size = self.config.subcore_chunk_size
        else:
            row_size = self.config.core_chunk_size
        return pl.ds(pl.multiple_of(row_offset, 8), row_size)

    def _get_col_slice(self, base_col_offset, col_size, col_chunk_size,
                       for_tec):
        """Returns a column slice from `base_col_offset` of width `col_size`,

    or sharded to `col_chunk_size` if `for_tec` is True.
    """
        if for_tec:
            col_offset = base_col_offset + self.subcore_col_idx * col_chunk_size
            col_width = col_chunk_size
        else:
            col_offset = base_col_offset
            col_width = col_size
        return pl.ds(col_offset, col_width)

    def get_phase1_slice(self, chunk_idx, mb_idx, *, for_tec=False):
        """Returns a 2D HBM slice for Phase 1 (D2D) for `chunk_idx` and `mb_idx`,

    mapped to subcore if `for_tec` is True.
    """
        return (
            self._get_row_slice(chunk_idx, for_tec),
            self._get_col_slice(
                mb_idx * self.config.mb_size,
                self.config.mb_size,
                self.config.subcore_col_chunk_size_p1,
                for_tec,
            ),
        )

    def get_phase2_slice(self,
                         chunk_idx,
                         mb_idx,
                         hcube_dim_idx,
                         *,
                         for_tec=False):
        """Returns a 2D HBM slice for Phase 2 (C2C) for `chunk_idx`, `mb_idx`

    and `hcube_dim_idx`, mapped to subcore if `for_tec` is True.
    """
        return (
            self._get_row_slice(chunk_idx, for_tec),
            self._get_col_slice(
                mb_idx * self.config.mb_size +
                hcube_dim_idx * self.config.hc_chunk_size,
                self.config.hc_chunk_size,
                self.config.subcore_col_chunk_size_p2,
                for_tec,
            ),
        )

    def get_phase1_chunk_idx(self, device_id, chip_idx):
        """Calculates the chunk index processed by `device_id` for `chip_idx`.

    In Phase 1, global token chunks are sharded across the topology. A device
    processes token chunks corresponding to all physical chips `chip_idx` in
    the mesh, filtered by its own chiplet position (even/odd device ID).
    """
        chiplet_bit = device_id % 2
        return chip_idx * 2 + chiplet_bit

    def get_phase1_chunk_idxes(self, device_id):
        """Returns all global chunk indices processed by the chiplet group of device `device_id`."""
        chiplet_bit = device_id % 2
        return [
            chip_idx * 2 + chiplet_bit
            for chip_idx in range(self.config.num_chips)
        ]

    def get_phase2_chunk_idx(self, device_id, step_idx, chunk_group_idx,
                             hcube_dim_idx):
        """Calculates the chunk index owned by a device `device_id` for chunk group `chunk_group_idx` during Phase 2 (C2C RS).

    During Phase 2, devices perform a hypercube reduction. At step `step_idx` of
    the hypercube reduction, the topology is partitioned into independent
    parallel sub-cubes/groups of devices exchanging along hypercube dimension
    `hcube_dim_idx`.
    """
        dim = (hcube_dim_idx + step_idx) % self.config.num_hcube_dims
        chip_id = device_id // 2
        my_dim_bit = (chip_id >> dim) & 1

        prev_dims = [(hcube_dim_idx + j) % self.config.num_hcube_dims
                     for j in range(step_idx)]
        future_dims = [(hcube_dim_idx + j) % self.config.num_hcube_dims
                       for j in range(step_idx + 1, self.config.num_hcube_dims)
                       ]

        my_base_chunk_idx = self.get_hcube_chunk_idx(device_id,
                                                     chunk_group_idx,
                                                     future_dims, prev_dims,
                                                     dim, my_dim_bit)
        chiplet_bit = device_id % 2
        return my_base_chunk_idx * 2 + chiplet_bit

    def get_hcube_chunk_idx(
        self,
        device_id,
        chunk_group_idx,
        future_dims,
        prev_dims,
        target_dim,
        dim_val,
    ):
        """Calculates the mapped HBM chunk index for the hypercube communication ring of device `device_id` at iteration `chunk_group_idx` along active dimension `target_dim` with bit value `dim_val`, given the processed dimensions `prev_dims` and unprocessed dimensions `future_dims`."""
        chip_id = device_id // 2
        base = 0
        for d in prev_dims:
            bit = (chip_id >> d) & 1
            base |= bit << d
        for bit_pos, d in enumerate(future_dims):
            bit = (chunk_group_idx >> bit_pos) & 1
            base |= bit << d
        base |= dim_val << target_dim
        return base
