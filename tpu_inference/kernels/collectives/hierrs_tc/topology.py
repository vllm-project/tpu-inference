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

from tpu_inference.kernels.collectives.hierrs_tc.config import Config


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
    """Encapsulates sequence and HBM indexing math for TensorCore Reduce-Scatter."""

    def __init__(self, config: Config, topo: Topology):
        self.config = config
        self.topo = topo
        self.mb_stride = config.num_hcube_dims * config.hc_chunk_size

    def get_slice(self, chunk_idx, start, size):
        """Returns a 2D HBM slice for a given chunk index and hidden dimension range."""
        return (
            pl.ds(chunk_idx * self.config.seq_chunk_size,
                  self.config.seq_chunk_size),
            pl.ds(start, size),
        )

    def get_phase1_slice(self, chunk_idx, mb_idx):
        """Returns the 2D HBM slice for Phase 1 (D2D) at `chunk_idx`, `mb_idx`."""
        return self.get_slice(chunk_idx, mb_idx * self.config.mb_size,
                              self.config.mb_size)

    def get_phase2_slice(self, chunk_idx, mb_idx, hcube_dim_idx):
        """Returns the 2D HBM slice for Phase 2 (C2C) at `chunk_idx`, `mb_idx`, `hcube_dim_idx`."""
        return self.get_slice(
            chunk_idx,
            mb_idx * self.mb_stride +
            hcube_dim_idx * self.config.hc_chunk_size,
            self.config.hc_chunk_size,
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
        """Calculates the chunk index owned by `device_id` for chunk group `chunk_group_idx` during Phase 2 (C2C RS).

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

    def make_phase1_index_fn(self, mb_idx):
        """Grid index fn for the Phase 1 emit_pipeline (chip_idx -> ref index)."""

        def phase1_index_fn(chip_idx):
            c_me = self.get_phase1_chunk_idx(self.topo.cur_id, chip_idx)
            return (c_me, mb_idx)

        return phase1_index_fn

    def make_phase1_in_index_fn_with_recv_sem(self, mb_idx):
        """Phase 1 input index fn that also reports the slice width for the recv semaphore."""

        def phase1_in_index_fn_with_recv_sem(grid_indices, ref):
            (chip_idx, ) = grid_indices
            c_me = self.get_phase1_chunk_idx(self.topo.cur_id, chip_idx)
            return self.get_phase1_slice(c_me, mb_idx), self.config.mb_size

        return phase1_in_index_fn_with_recv_sem

    def make_phase2_in_index_fn_with_recv_sem(self, step_idx, mb_idx):
        """Phase 2 input index fn that also reports the slice width for the recv semaphore."""

        def phase2_in_index_fn_with_recv_sem(grid_indices, ref):
            chunk_group_idx, hcube_dim_idx = grid_indices
            my_chunk_idx = self.get_phase2_chunk_idx(self.topo.cur_id,
                                                     step_idx, chunk_group_idx,
                                                     hcube_dim_idx)
            chunk_slice = self.get_phase2_slice(my_chunk_idx, mb_idx,
                                                hcube_dim_idx)
            return chunk_slice, self.config.hc_chunk_size

        return phase2_in_index_fn_with_recv_sem

    def make_phase2_index_fn(self, step_idx, mb_idx):
        """Grid index fn for the Phase 2 emit_pipeline."""

        def phase2_index_fn(chunk_group_idx, hcube_dim_idx):
            my_chunk_idx = self.get_phase2_chunk_idx(self.topo.cur_id,
                                                     step_idx, chunk_group_idx,
                                                     hcube_dim_idx)
            mb_start_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (my_chunk_idx, mb_start_idx)

        return phase2_index_fn

    def make_phase2_out_index_fn(self, step_idx, mb_idx):
        """Grid index fn for the Phase 2 output ref on the final step."""

        def phase2_out_index_fn(chunk_group_idx, hcube_dim_idx):
            mb_start_idx = mb_idx * self.config.num_hcube_dims + hcube_dim_idx
            return (0, mb_start_idx)

        return phase2_out_index_fn
