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

from tpu_inference.runner.continuous_block_pool import ContinuousFreeQueue


class MockKVCacheBlock:

    def __init__(self, block_id):
        self.block_id = block_id


def test_initialization():
    blocks = [MockKVCacheBlock(i) for i in range(10)]
    queue = ContinuousFreeQueue(blocks)
    assert queue.num_free_blocks == 10
    assert len(queue.intervals) == 1
    assert queue.intervals[0] == (0, 9)


def test_popleft_single():
    blocks = [MockKVCacheBlock(i) for i in range(10)]
    queue = ContinuousFreeQueue(blocks)

    # First popleft will grab 0 as an exception to be the null_block
    b = queue.popleft()
    assert b.block_id == 0
    assert queue.num_free_blocks == 9
    assert queue.intervals[0] == (1, 9)

    # Subsequent poplefts will pull from the top (decode mode)
    b = queue.popleft()
    assert b.block_id == 9
    assert queue.num_free_blocks == 8
    assert queue.intervals[0] == (1, 8)

    b = queue.popleft()
    assert b.block_id == 8
    assert queue.num_free_blocks == 7


def test_popleft_n_contiguous():
    blocks = [MockKVCacheBlock(i) for i in range(10)]
    queue = ContinuousFreeQueue(blocks)

    # Should get contiguous chunk from bottom, skipping 0 to leave it for null_block
    alloc = queue.popleft_n(3)
    assert len(alloc) == 3
    assert [b.block_id for b in alloc] == [1, 2, 3]

    assert queue.num_free_blocks == 7
    # 0 is still free, 4-9 are free
    assert queue.intervals == [(0, 0), (4, 9)]


def test_popleft_n_best_fit():
    blocks = [MockKVCacheBlock(i) for i in range(20)]
    queue = ContinuousFreeQueue(blocks)

    # Manually fragment the interval space
    # Free space: 0, 1-2, 4-6, 8-15, 17-19
    queue.free_blocks.clear()
    queue.intervals.clear()
    queue.append_n([
        blocks[0], blocks[1], blocks[2], blocks[4], blocks[5], blocks[6],
        blocks[8], blocks[9], blocks[10], blocks[11], blocks[12], blocks[13],
        blocks[14], blocks[15], blocks[17], blocks[18], blocks[19]
    ])

    assert queue.intervals == [(0, 2), (4, 6), (8, 15), (17, 19)]

    # Needs a size 3 contiguous chunk. It should pick (0, 2) over (4, 6) to Best-Fit and break ties (First-Fit).
    alloc = queue.popleft_n(3)
    assert len(alloc) == 3
    assert [b.block_id for b in alloc] == [0, 1, 2]
    assert queue.intervals == [(4, 6), (8, 15), (17, 19)]


def test_popleft_n_fallback():
    blocks = [MockKVCacheBlock(i) for i in range(10)]
    queue = ContinuousFreeQueue(blocks)

    # Leave scattered blocks
    queue.free_blocks.clear()
    queue.intervals.clear()
    queue.append_n([blocks[1], blocks[3], blocks[5], blocks[7], blocks[9]])

    # Ask for 3. No continuous chunk of size 3. Should fallback to grabbing from the top.
    alloc = queue.popleft_n(3)
    assert len(alloc) == 3
    # Top 3 are 9, 7, 5, after sorting it should be 5, 7, 9
    assert [b.block_id for b in alloc] == [5, 7, 9]


def test_remove_splits_intervals():
    blocks = [MockKVCacheBlock(i) for i in range(10)]
    queue = ContinuousFreeQueue(blocks)

    queue.remove(blocks[4])
    assert queue.num_free_blocks == 9
    assert queue.intervals == [(0, 3), (5, 9)]

    queue.remove(blocks[0])
    assert queue.intervals == [(1, 3), (5, 9)]

    queue.remove(blocks[9])
    assert queue.intervals == [(1, 3), (5, 8)]


def test_append_merges_intervals():
    blocks = [MockKVCacheBlock(i) for i in range(10)]
    queue = ContinuousFreeQueue(blocks)
    queue.free_blocks.clear()
    queue.intervals.clear()

    queue.append_n([blocks[2], blocks[4], blocks[5], blocks[7]])
    assert queue.intervals == [(2, 2), (4, 5), (7, 7)]

    # Merge left
    queue.append_n([blocks[3]])
    assert queue.intervals == [(2, 5), (7, 7)]

    # Merge right
    queue.append_n([blocks[6]])
    assert queue.intervals == [(2, 7)]
