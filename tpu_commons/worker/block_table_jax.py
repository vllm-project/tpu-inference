# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
from vllm.utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheConfig


class BlockTable:

    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: jax.Device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        self.block_table = jnp.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=jnp.int32,
        )
        self.block_table_cpu = np.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=jnp.int32,
        )
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping_cpu = np.zeros(self.max_num_batched_tokens,
                                         dtype=jnp.int64)
        self.slot_mapping = jnp.zeros(self.max_num_batched_tokens,
                                      dtype=jnp.int64)

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table_cpu[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_cpu[tgt, :num_blocks] = self.block_table_cpu[
            src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        num_blocks_src = self.num_blocks_per_row[src]
        num_blocks_tgt = self.num_blocks_per_row[tgt]
        self.num_blocks_per_row[src] = num_blocks_tgt
        self.num_blocks_per_row[tgt] = num_blocks_src

        self.block_table_cpu[[src, tgt]] = self.block_table_cpu[[tgt, src]]

    def commit(self, num_reqs: int) -> None:
        self.block_table[:num_reqs].copy_(self.block_table_cpu[:num_reqs],
                                          non_blocking=True)

    def clear(self) -> None:
        self.block_table.fill_(0)
        self.block_table_cpu.fill_(0)

    def get_device_tensor(self) -> jax.Array:
        """Ruturns the device tensor of the block table."""
        return self.block_table

    def get_cpu_tensor(self) -> jax.Array:
        """Returns the CPU tensor of the block table."""
        return self.block_table_cpu


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_batched_tokens: int, pin_memory: bool,
                 device: jax.Device, kv_cache_config: KVCacheConfig) -> None:
        max_num_blocks_per_req = [
            cdiv(max_model_len, g.kv_cache_spec.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        self.block_tables = [
            BlockTable(max_num_reqs, max_num_blocks_per_req[i],
                       max_num_batched_tokens, pin_memory, device)
            for i in range(len(kv_cache_config.kv_cache_groups))
        ]

    def append_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: list[list[int]], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def commit(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit(num_reqs)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
