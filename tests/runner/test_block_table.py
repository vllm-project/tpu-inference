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

# test_block_table_jax.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def cdiv(a: int, b: int) -> int:
    """Ceiling division: (a + b - 1) // b."""
    return (a + b - 1) // b


class BlockTable:
    """A JAX-compatible BlockTable for managing memory blocks."""

    def __init__(
            self,
            max_num_reqs: int,
            max_num_blocks_per_req: int,
            max_num_batched_tokens: int,
            pin_memory: bool,  # Note: pin_memory is not used in JAX
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.block_table = jnp.zeros((max_num_reqs, max_num_blocks_per_req),
                                     dtype=jnp.int32)
        self.block_table_cpu = np.zeros((max_num_reqs, max_num_blocks_per_req),
                                        dtype=np.int32)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        if not block_ids:
            return
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table_cpu[row_idx, start:start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        # Clear the row for a clean overwrite
        self.block_table_cpu[row_idx].fill(0)
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table_cpu[tgt, :num_blocks] = self.block_table_cpu[
            src, :num_blocks]
        # Clear the rest of the target row to avoid stale data
        self.block_table_cpu[tgt, num_blocks:].fill(0)
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        self.num_blocks_per_row[[src,
                                 tgt]] = self.num_blocks_per_row[[tgt, src]]
        self.block_table_cpu[[src, tgt]] = self.block_table_cpu[[tgt, src]]

    def commit(self, num_reqs: int) -> None:
        """Corrected commit for JAX immutability."""
        self.block_table = self.block_table.at[:num_reqs].set(
            self.block_table_cpu[:num_reqs])

    def clear(self) -> None:
        """Corrected clear for JAX immutability and completeness."""
        self.block_table = jnp.zeros_like(self.block_table)
        self.block_table_cpu.fill(0)
        self.num_blocks_per_row.fill(0)

    def get_device_tensor(self) -> jax.Array:
        return self.block_table

    def get_cpu_tensor(self) -> np.ndarray:
        return self.block_table_cpu


class MultiGroupBlockTable:
    """Manages BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        block_sizes: list[int],
    ) -> None:
        self.block_tables = [
            BlockTable(
                max_num_reqs,
                cdiv(max_model_len, block_size),
                max_num_batched_tokens,
                pin_memory,
            ) for block_size in block_sizes
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
        return self.block_tables[idx]


# --- Pytest Fixtures ---


@pytest.fixture
def block_table_params():
    """Provides common parameters for creating a BlockTable."""
    return {
        "max_num_reqs": 8,
        "max_num_blocks_per_req": 16,
        "max_num_batched_tokens": 8 * 16,
        "pin_memory": False,
    }


@pytest.fixture
def block_table(block_table_params):
    """Provides a fresh BlockTable instance for each test."""
    return BlockTable(**block_table_params)


# --- Test Cases ---

##
## BlockTable Tests
##


class TestBlockTable:
    """Tests for the single BlockTable class."""

    def test_init(self, block_table, block_table_params):
        """Test constructor and initial state."""
        bt = block_table
        params = block_table_params

        assert bt.max_num_reqs == params["max_num_reqs"]
        assert bt.max_num_blocks_per_req == params["max_num_blocks_per_req"]

        # Check CPU table
        assert bt.block_table_cpu.shape == (
            params["max_num_reqs"],
            params["max_num_blocks_per_req"],
        )
        assert bt.block_table_cpu.dtype == np.int32
        np.testing.assert_array_equal(bt.block_table_cpu, 0)

        # Check device table
        assert bt.block_table.shape == (
            params["max_num_reqs"],
            params["max_num_blocks_per_req"],
        )
        assert bt.block_table.dtype == jnp.int32
        np.testing.assert_array_equal(np.array(bt.block_table), 0)

        # Check block counter per row
        assert bt.num_blocks_per_row.shape == (params["max_num_reqs"], )
        np.testing.assert_array_equal(bt.num_blocks_per_row, 0)

    def test_add_and_append_row(self, block_table):
        """Test adding and appending blocks to a row."""
        # Append to row 0
        block_table.append_row([1, 2, 3], row_idx=0)
        assert block_table.num_blocks_per_row[0] == 3
        np.testing.assert_array_equal(block_table.block_table_cpu[0, :3],
                                      [1, 2, 3])

        # Append more to row 0
        block_table.append_row([4, 5], row_idx=0)
        assert block_table.num_blocks_per_row[0] == 5
        np.testing.assert_array_equal(block_table.block_table_cpu[0, :5],
                                      [1, 2, 3, 4, 5])

        # Add (overwrite) row 1
        block_table.add_row([10, 11], row_idx=1)
        assert block_table.num_blocks_per_row[1] == 2
        np.testing.assert_array_equal(block_table.block_table_cpu[1, :2],
                                      [10, 11])

        # Add (overwrite) row 0
        block_table.add_row([6, 7, 8, 9], row_idx=0)
        assert block_table.num_blocks_per_row[0] == 4
        np.testing.assert_array_equal(block_table.block_table_cpu[0, :4],
                                      [6, 7, 8, 9])
        assert block_table.block_table_cpu[
            0, 4] == 0  # Ensure rest of row is clear

    def test_move_row(self, block_table):
        """Test moving a row's content."""
        block_table.add_row([10, 20, 30], row_idx=2)
        block_table.add_row([99], row_idx=5)  # Pre-existing data

        block_table.move_row(src=2, tgt=5)

        # Check target row
        assert block_table.num_blocks_per_row[5] == 3
        np.testing.assert_array_equal(block_table.get_cpu_tensor()[5, :3],
                                      [10, 20, 30])
        assert block_table.get_cpu_tensor()[
            5, 3] == 0  # Check old data is cleared

        # Check source row (should be unchanged)
        assert block_table.num_blocks_per_row[2] == 3
        np.testing.assert_array_equal(block_table.get_cpu_tensor()[2, :3],
                                      [10, 20, 30])

    def test_swap_row(self, block_table):
        """Test swapping two rows."""
        row_2_data = [10, 20, 30]
        row_5_data = [99, 88]
        block_table.add_row(row_2_data, row_idx=2)
        block_table.add_row(row_5_data, row_idx=5)

        block_table.swap_row(src=2, tgt=5)

        # Check that data and counts are swapped
        assert block_table.num_blocks_per_row[2] == 2
        assert block_table.num_blocks_per_row[5] == 3
        np.testing.assert_array_equal(block_table.block_table_cpu[2, :2],
                                      row_5_data)
        np.testing.assert_array_equal(block_table.block_table_cpu[5, :3],
                                      row_2_data)

    def test_commit(self, block_table):
        """Test committing the CPU table to the JAX device table."""
        block_table.add_row([1, 2, 3], row_idx=0)
        block_table.add_row([4, 5], row_idx=1)
        num_reqs_to_commit = 2

        # Before commit, device tensor is all zeros
        np.testing.assert_array_equal(
            np.array(block_table.get_device_tensor()), 0)

        block_table.commit(num_reqs_to_commit)
        device_table = np.array(block_table.get_device_tensor())

        # After commit, device tensor should match committed part of CPU tensor
        np.testing.assert_array_equal(
            device_table[:num_reqs_to_commit],
            block_table.get_cpu_tensor()[:num_reqs_to_commit],
        )
        # The rest of the device tensor should still be zero
        np.testing.assert_array_equal(device_table[num_reqs_to_commit:], 0)

    def test_clear(self, block_table):
        """Test clearing all table data."""
        block_table.add_row([1, 2, 3], row_idx=0)
        block_table.commit(num_reqs=1)

        # Pre-clear check
        assert np.any(block_table.get_cpu_tensor())
        assert jnp.any(block_table.get_device_tensor())
        assert np.any(block_table.num_blocks_per_row)

        block_table.clear()

        # Post-clear check
        np.testing.assert_array_equal(block_table.get_cpu_tensor(), 0)
        np.testing.assert_array_equal(
            np.array(block_table.get_device_tensor()), 0)
        np.testing.assert_array_equal(block_table.num_blocks_per_row, 0)


# ------------------------------------
# MultiGroupBlockTable Tests
# ------------------------------------


class TestMultiGroupBlockTable:
    """Tests for the MultiGroupBlockTable class."""

    @pytest.fixture
    def multi_table_params(self):
        return {
            "max_num_reqs": 4,
            "max_model_len": 32,
            "max_num_batched_tokens": 4 * 32,
            "pin_memory": False,
            "block_sizes": [16, 8],  # Two groups
        }

    @pytest.fixture
    def multi_table(self, multi_table_params):
        return MultiGroupBlockTable(**multi_table_params)

    def test_init(self, multi_table, multi_table_params):
        """Test constructor and initial state of multiple tables."""
        params = multi_table_params
        assert len(multi_table.block_tables) == len(params["block_sizes"])
        assert isinstance(multi_table[0], BlockTable)
        assert isinstance(multi_table[1], BlockTable)

        # Check that max_num_blocks_per_req is calculated correctly
        assert multi_table[0].max_num_blocks_per_req == cdiv(
            params["max_model_len"], params["block_sizes"][0])  # 32 / 16 = 2
        assert multi_table[1].max_num_blocks_per_req == cdiv(
            params["max_model_len"], params["block_sizes"][1])  # 32 / 8 = 4

    def test_add_row(self, multi_table):
        """Test add_row across multiple tables."""
        block_ids = [[101, 102], [201, 202, 203]]
        multi_table.add_row(block_ids, row_idx=0)

        # Check table 0
        assert multi_table[0].num_blocks_per_row[0] == 2
        np.testing.assert_array_equal(multi_table[0].get_cpu_tensor()[0, :2],
                                      block_ids[0])

        # Check table 1
        assert multi_table[1].num_blocks_per_row[0] == 3
        np.testing.assert_array_equal(multi_table[1].get_cpu_tensor()[0, :3],
                                      block_ids[1])

    def test_swap_row(self, multi_table):
        """Test swap_row across multiple tables."""
        row1_data = [[11], [11, 22]]
        row3_data = [[33], [33, 44, 55]]
        multi_table.add_row(row1_data, row_idx=1)
        multi_table.add_row(row3_data, row_idx=3)

        multi_table.swap_row(src=1, tgt=3)

        # Check row 1 now has row 3's data
        assert multi_table[0].num_blocks_per_row[1] == 1
        np.testing.assert_array_equal(multi_table[0].get_cpu_tensor()[1, :1],
                                      row3_data[0])
        assert multi_table[1].num_blocks_per_row[1] == 3
        np.testing.assert_array_equal(multi_table[1].get_cpu_tensor()[1, :3],
                                      row3_data[1])

        # Check row 3 now has row 1's data
        assert multi_table[0].num_blocks_per_row[3] == 1
        np.testing.assert_array_equal(multi_table[0].get_cpu_tensor()[3, :1],
                                      row1_data[0])
        assert multi_table[1].num_blocks_per_row[3] == 2
        np.testing.assert_array_equal(multi_table[1].get_cpu_tensor()[3, :2],
                                      row1_data[1])

    def test_commit_and_clear(self, multi_table):
        """Test commit and clear across multiple tables."""
        multi_table.add_row([[1], [1, 2]], row_idx=0)
        multi_table.commit(num_reqs=1)

        # Check commit worked for all tables
        for table in multi_table.block_tables:
            assert jnp.any(table.get_device_tensor())
            device_table = np.array(table.get_device_tensor())
            cpu_table = table.get_cpu_tensor()
            np.testing.assert_array_equal(device_table, cpu_table)

        multi_table.clear()

        # Check clear worked for all tables
        for table in multi_table.block_tables:
            np.testing.assert_array_equal(table.get_cpu_tensor(), 0)
            np.testing.assert_array_equal(np.array(table.get_device_tensor()),
                                          0)
            np.testing.assert_array_equal(table.num_blocks_per_row, 0)
