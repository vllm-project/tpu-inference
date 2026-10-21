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

from bisect import bisect_left, bisect_right
from typing import List, Set, Tuple

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class ContinuousFreeQueue:
    """
    A custom FreeKVCacheBlockQueue that maintains free blocks as continuous intervals
    to optimize allocation for TPU Paged Attention (where `dynamic_update_slice_in_dim`
    is heavily preferred over scatters).

    - Prefill requests (N > 1 blocks) use a Best-Fit approach to find continuous segments.
    - Decode requests (1 block) take from the top-down to isolate fragmentation to the high IDs.
    """

    def __init__(self, blocks: List):
        self.free_blocks: Set[int] = set()
        self.blocks_ref = blocks
        self.intervals: List[Tuple[int, int]] = []
        self.append_n(blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _add_to_intervals(self, block_id: int):
        if not self.intervals:
            self.intervals.append((block_id, block_id))
            return

        idx = bisect_left(self.intervals, (block_id, block_id))
        merged = False

        if idx > 0:
            prev_start, prev_end = self.intervals[idx - 1]
            if prev_end == block_id - 1:
                self.intervals[idx - 1] = (prev_start, block_id)
                merged = True
                if idx < len(self.intervals):
                    next_start, next_end = self.intervals[idx]
                    if next_start == block_id + 1:
                        self.intervals[idx - 1] = (prev_start, next_end)
                        self.intervals.pop(idx)
                return

        if not merged and idx < len(self.intervals):
            next_start, next_end = self.intervals[idx]
            if next_start == block_id + 1:
                self.intervals[idx] = (block_id, next_end)
                merged = True
                return

        if not merged:
            self.intervals.insert(idx, (block_id, block_id))

    def _remove_from_intervals(self, block_id: int):
        idx = bisect_right(self.intervals, (block_id, float('inf'))) - 1
        start, end = self.intervals[idx]

        if start == end:
            self.intervals.pop(idx)
        elif start == block_id:
            self.intervals[idx] = (start + 1, end)
        elif end == block_id:
            self.intervals[idx] = (start, end - 1)
        else:
            self.intervals[idx] = (start, block_id - 1)
            self.intervals.insert(idx + 1, (block_id + 1, end))

    def append_n(self, blocks: List):
        for b in blocks:
            if b.block_id not in self.free_blocks:
                self.free_blocks.add(b.block_id)
                self._add_to_intervals(b.block_id)

    def remove(self, block):
        if block.block_id in self.free_blocks:
            self.free_blocks.remove(block.block_id)
            self._remove_from_intervals(block.block_id)

    def popleft(self):
        # Always reserve 0 for the null_block if it is untouched during initialization
        if 0 in self.free_blocks and len(self.free_blocks) == len(
                self.blocks_ref):
            block_id = 0
        else:
            block_id = self.intervals[-1][1]
            # Avoid popping 0 as a normal block if there are other choices
            if block_id == 0 and len(self.intervals) > 1:
                block_id = self.intervals[-2][1]

        self.free_blocks.remove(block_id)
        self._remove_from_intervals(block_id)

        logger.debug(
            f"ContinuousFreeQueue.popleft: Decode allocated 1 block at ID {block_id}"
        )
        return self.blocks_ref[block_id]

    def popleft_n(self, num_blocks: int) -> List:
        if num_blocks == 1:
            return [self.popleft()]

        best_idx = -1
        best_size = float('inf')

        # Best-Fit search for the smallest contiguous memory region that satisfies num_blocks
        for i, (start, end) in enumerate(self.intervals):
            actual_start = start
            if start == 0 and len(self.free_blocks) == len(self.blocks_ref):
                # Ensure we don't accidentally grab the null block initially
                actual_start = 1

            size = end - actual_start + 1
            if size >= num_blocks and size < best_size:
                best_size = size
                best_idx = i

        if best_idx != -1:
            start, end = self.intervals[best_idx]
            actual_start = max(1, start) if start == 0 and len(
                self.free_blocks) == len(self.blocks_ref) else start

            alloc_start = actual_start
            alloc_end = actual_start + num_blocks - 1

            allocated = [
                self.blocks_ref[i] for i in range(alloc_start, alloc_end + 1)
            ]
            for b in allocated:
                self.free_blocks.remove(b.block_id)
                self._remove_from_intervals(b.block_id)

            logger.debug(
                f"ContinuousFreeQueue.popleft_n: Prefill Best-Fit allocated {num_blocks} contiguous blocks "
                f"starting at ID {alloc_start} from free interval [{start}, {end}]"
            )
            return allocated
        else:
            # Fallback: Scattered allocation if no continuous range fits
            logger.debug(
                f"ContinuousFreeQueue.popleft_n: FALLBACK scattered allocation for {num_blocks} blocks"
            )
            allocated = []
            for _ in range(num_blocks):
                allocated.append(self.popleft())

            # Sort the fallback blocks by physical ID to naturally group any fragmented pieces
            # back into ascending continuous chunks for the insertion loop
            allocated.sort(key=lambda b: b.block_id)
            return allocated
