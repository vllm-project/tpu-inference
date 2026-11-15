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

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.deepseek_v4.compress_and_store import \
    config


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _BufferedRef(pltpu.BufferedRef):
    cfgs: config.Configs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.Array,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: config.Configs,
    ):
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfgs=cfgs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(_BufferedRef):

    def copy_in(self, src_ref, grid_indices):
        cos_sin_cache_ref, positions_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        pid = grid_indices[0]

        dest_ref = self.window_ref.at[slot]  # (tile_n, rope_head_dim)

        tile_n = self.cfgs.tile_sizes.tile_n
        compress_ratio = self.cfgs.dims.compress_ratio

        @pl.loop(0, tile_n, unroll=True)
        def loop_body(i):
            global_idx = pid * tile_n + i
            position = positions_ref[global_idx]
            compressed_pos = (position // compress_ratio) * compress_ratio

            src = cos_sin_cache_ref.at[compressed_pos,
                                       pl.ds(0, self.window_ref.shape[-1])]
            dest = dest_ref.at[i, :]
            pltpu.make_async_copy(src, dest, sem).start()

    def wait_in(self, src_ref, grid_indices):
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class PageBufferRef(_BufferedRef):
    # VMEM block: cfgs.page_buffer_shape()
    #   (tile_n, pages_to_buffer_per_token, page_size, hbm_pack, LANE) uint8

    def copy_in(self, src_ref, grid_indices):
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        pid = grid_indices[0]

        (
            state_cache_ref,
            positions_ref,
            block_table_ref,
            block_table_stride,
            token_to_req_indices_ref,
        ) = src_ref

        tile_n = self.cfgs.tile_sizes.tile_n
        pages_to_buffer_per_token = self.cfgs.pages_to_buffer_per_token
        block_size = self.cfgs.state_block_size

        page_buffer_ref = self.window_ref.at[slot]
        global_idx = pid * tile_n

        @pl.loop(0, tile_n * pages_to_buffer_per_token, unroll=True)
        def loop_body(idx):
            n = idx // pages_to_buffer_per_token
            p = idx % pages_to_buffer_per_token
            idx_n = global_idx + n

            position = positions_ref[idx_n]
            req_idx = token_to_req_indices_ref[idx_n]

            block_idx = position // block_size - (pages_to_buffer_per_token -
                                                  1) + p
            safe_block_idx = jnp.maximum(block_idx, 0)
            page = block_table_ref[req_idx * block_table_stride +
                                   safe_block_idx]

            src = state_cache_ref.at[page, :, :, :]
            dest = page_buffer_ref.at[n, p, :, :, :]
            pltpu.make_async_copy(src, dest, sem).start()

    def wait_in(self, src_ref, grid_indices):
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutputRef(_BufferedRef):
    # VMEM block: cfgs.output_shape() -> (tile_n, record_rows, hbm_pack, last_dim_size)

    def copy_in(self, src_ref, grid_indices):
        # Only called in CSA_INDEXER mode (INPUT_OUTPUT)
        cache_ref, kv_slot_mapping_ref, _ = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        dest_ref = self.window_ref.at[slot]
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        # Cache is laid out like this:
        # [num_page, kv_block_size / 4, 4, 256]
        page_size = self.cfgs.kv_block_size

        @pl.loop(0, tile_n)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid)
            def _read_nope():
                p = kv_slot // page_size
                # we fetch the entire tile (4, 128), with 4 pages per tile
                s_row = (kv_slot %
                         page_size) // (self.cfgs.tokens_in_second_minor)
                src = cache_ref.at[p, s_row, :, :]
                dest = dest_ref.at[n, 0, :, :]
                pltpu.make_async_copy(src, dest, sem).start()

    def wait_in(self, src_ref, grid_indices):
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n
        _, kv_slot_mapping_ref, _ = src_ref

        @pl.loop(0, tile_n)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid)
            def _wait_nope():
                pltpu.make_async_copy(
                    vmem_ref.at[n, 0, :, :],
                    vmem_ref.at[n, 0, :, :],
                    sem,
                ).wait()

    def copy_out(self, dest_ref, grid_indices):
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        src_ref = self.window_ref.at[slot]
        pid = grid_indices[0]
        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        if self.cfgs.dims.mode == config.Mode.CSA_INDEXER:
            cache_ref, kv_slot_mapping_ref, is_first_mask_ref = dest_ref

            @pl.loop(0, tile_n)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]
                is_first = is_first_mask_ref[idx_n]
                valid = (kv_slot >= 0) & is_first

                @pl.when(valid)
                def _write_nope():
                    p = kv_slot // self.cfgs.kv_block_size
                    s_row = (kv_slot % self.cfgs.kv_block_size) // (
                        self.cfgs.tokens_in_second_minor)
                    src = src_ref.at[n, 0, :, :]
                    dest = cache_ref.at[p, s_row, :, :]
                    pltpu.make_async_copy(src, dest, sem).start()
        else:
            cache_ref, kv_slot_mapping_ref = dest_ref
            # HCA: [num_pages, kv_block_size * 2, 4, 128] uint8
            # CSA: [num_pages, kv_block_size, 4, 128] uint8
            @pl.loop(0, tile_n)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]

                @pl.when(kv_slot >= 0)
                def _write_nope():
                    p = kv_slot // self.cfgs.physical_page_size
                    s = kv_slot % self.cfgs.physical_page_size

                    @pl.loop(0, self.cfgs.record_rows, unroll=True)
                    def write_loop(o):
                        src_nope = src_ref.at[n, o, ...]
                        dest_nope = cache_ref.at[p, s + o, ...]
                        pltpu.make_async_copy(src_nope, dest_nope, sem).start()

    def wait_out(self, dest_ref, grid_indices):
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pid = grid_indices[0]
        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        is_indexer = self.cfgs.dims.mode == config.Mode.CSA_INDEXER

        if is_indexer:
            _, kv_slot_mapping_ref, is_first_mask_ref = dest_ref

            @pl.loop(0, tile_n)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]
                is_first = is_first_mask_ref[idx_n]
                valid = (kv_slot >= 0) & is_first

                @pl.when(valid)
                def _wait_nope():
                    pltpu.make_async_copy(
                        vmem_ref.at[n, 0, :, :],
                        vmem_ref.at[n, 0, :, :],
                        sem,
                    ).wait()
        else:
            _, kv_slot_mapping_ref = dest_ref

            @pl.loop(0, tile_n)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]

                @pl.when(kv_slot >= 0)
                def _wait_nope():

                    @pl.loop(0, self.cfgs.record_rows, unroll=True)
                    def wait_loop(o):
                        pltpu.make_async_copy(
                            vmem_ref.at[n, o, :, :],
                            vmem_ref.at[n, o, :, :],
                            sem,
                        ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoPEOutputRef(_BufferedRef):
    # VMEM block: cfgs.rope_output_shape() -> (tile_n, 1, LANE) uint8
    # Only used for CSA (has_rope_cache=True).

    def copy_in(self, src_ref, grid_indices):
        rope_cache_ref, kv_slot_mapping_ref, is_first_mask_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        dest_ref = self.window_ref.at[slot]
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        @pl.loop(0, tile_n)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            is_first = is_first_mask_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid & is_first)
            def _read_rope():
                # divide by 4 because we pack 4 pages into 1 tile
                # layout: [num_page, kv_block_size / 4, 4, 256]
                rope_slot = kv_slot // 4
                rope_p = rope_slot // self.cfgs.rope_page_size
                s_row = rope_slot % self.cfgs.rope_page_size
                src = rope_cache_ref.at[rope_p, s_row, :, :]
                dest = dest_ref.at[n, :, :]
                pltpu.make_async_copy(src, dest, sem).start()

    def wait_in(self, src_ref, grid_indices):
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n
        _, kv_slot_mapping_ref, is_first_mask_ref = src_ref

        @pl.loop(0, tile_n)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            is_first = is_first_mask_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid & is_first)
            def _wait_rope():
                pltpu.make_async_copy(
                    vmem_ref.at[n, :, :],
                    vmem_ref.at[n, :, :],
                    sem,
                ).wait()

    def copy_out(self, dest_ref, grid_indices):
        rope_cache_ref, kv_slot_mapping_ref, is_first_mask_ref = dest_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]

        src_ref = self.window_ref.at[slot]
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        @pl.loop(0, tile_n)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            is_first = is_first_mask_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid & is_first)
            def _write_rope():
                rope_slot = kv_slot // 4
                rope_p = rope_slot // self.cfgs.rope_page_size
                s_row = rope_slot % self.cfgs.rope_page_size
                src_rope = src_ref.at[n, :, :]
                dest_rope = rope_cache_ref.at[rope_p, s_row, :, :]
                pltpu.make_async_copy(src_rope, dest_rope, sem).start()

    def wait_out(self, dest_ref, grid_indices):
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        _, kv_slot_mapping_ref, is_first_mask_ref = dest_ref

        @pl.loop(0, tile_n)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            is_first = is_first_mask_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid & is_first)
            def _wait_rope():
                pltpu.make_async_copy(
                    vmem_ref.at[n, :, :],
                    vmem_ref.at[n, :, :],
                    sem,
                ).wait()


def create_allocs_and_specs(
    cfgs: config.Configs,
    *,
    cache_ref,
    rope_cache_ref,
    state_cache_ref,
    cos_sin_cache_ref,
    positions_ref,
    block_table_ref,
    block_table_stride,
    token_to_req_indices_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
) -> tuple[
        tuple[CosSinRef | None, OutputRef, PageBufferRef, RoPEOutputRef
              | None],
        tuple[pl.BlockSpec | None, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec
              | None],
        tuple[
            tuple[Any, ...] | None,
            tuple[Any, ...],
            tuple[Any, ...],
            tuple[Any, ...] | None,
        ],
]:
    # cos_sin
    if cfgs.dims.has_rope:
        tile_n = cfgs.tile_sizes.tile_n
        cos_sin_spec = pl.BlockSpec(
            block_shape=cfgs.cos_sin_shape(),
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i * tile_n, 0),
        )
        cos_sin_alloc = CosSinRef.create(
            spec=cos_sin_spec,
            dtype_or_type=cfgs.dims.cos_sin_dtype,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfgs,
        )
        cos_sin_args = (cos_sin_cache_ref, positions_ref)
    else:
        cos_sin_alloc = cos_sin_spec = cos_sin_args = None

    # output
    output_spec = pl.BlockSpec(
        block_shape=cfgs.output_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0, 0),
    )
    is_indexer = cfgs.dims.mode == config.Mode.CSA_INDEXER
    buffer_type = (pltpu.BufferType.INPUT_OUTPUT
                   if is_indexer else pltpu.BufferType.OUTPUT)
    output_alloc = OutputRef.create(
        spec=output_spec,
        dtype_or_type=jnp.uint8,
        buffer_type=buffer_type,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    if is_indexer:
        output_args = (cache_ref, kv_slot_mapping_ref, is_first_mask_ref)
    else:
        output_args = (cache_ref, kv_slot_mapping_ref)

    # page_buffer
    page_buffer_spec = pl.BlockSpec(
        block_shape=cfgs.page_buffer_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0, 0),
    )
    page_buffer_alloc = PageBufferRef.create(
        spec=page_buffer_spec,
        dtype_or_type=jnp.uint8,
        buffer_type=pltpu.BufferType.INPUT,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    page_buffer_args = (
        cache_ref if state_cache_ref is None else state_cache_ref,
        positions_ref,
        block_table_ref,
        block_table_stride,
        token_to_req_indices_ref,
    )

    # rope
    if cfgs.dims.has_rope_cache:
        rope_spec = pl.BlockSpec(
            block_shape=cfgs.rope_output_shape(),
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, 0, 0),
        )
        rope_alloc = RoPEOutputRef.create(
            spec=rope_spec,
            dtype_or_type=jnp.uint8,
            buffer_type=pltpu.BufferType.INPUT_OUTPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfgs,
        )
        rope_args = (rope_cache_ref, kv_slot_mapping_ref,
                     is_first_mask_rope_ref)
    else:
        rope_alloc = rope_spec = rope_args = None

    return (
        (cos_sin_alloc, output_alloc, page_buffer_alloc, rope_alloc),
        (cos_sin_spec, output_spec, page_buffer_spec, rope_spec),
        (cos_sin_args, output_args, page_buffer_args, rope_args),
    )
