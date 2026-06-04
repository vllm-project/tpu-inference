# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import time
from typing import Callable, Optional


class ReleaseOutcome(enum.Enum):
    """Result of mamba slot release"""
    NOT_IN_USE = "not_in_use"
    NOT_IN_DEFERRED = "not_in_deferred"
    RELEASED = "released"
    DEFERRED = "deferred"


class MambaSlotPool:
    def __init__(self, total_slots: int, dp_size: int = 1) -> None:
        self.dp_size = dp_size
        self._local_slots = total_slots // dp_size
        self._free: list[list[int]] = self._build_free_lists()
        # req_id -> (rank, slot)
        self._in_use: dict[str, tuple[int, int]] = {}
        # req_id -> (rank, slot, expiration)
        self._free_deferred: dict[str, tuple[int, int, float]] = {}
        # Optional callback: (req_id, slot) -> Optional[float]
        # returning the expiration.
        self._on_release: Optional[Callable[[str, int],
                                             Optional[float]]] = None

    def _build_free_lists(self) -> list[list[int]]:
        """Construct the per-rank free lists from `_local_slots`."""
        free: list[list[int]] = []
        for k in range(self.dp_size):
            base = k * self._local_slots
            # Reverse order so `.pop()` yields the lowest slot first.
            free.append(
                list(range(base + self._local_slots - 1, base, -1)))
        return free

    def resize(self, total_slots: int) -> None:
        """Rebuild the per-rank free lists with the actual block count."""
        assert not self._in_use and not self._free_deferred, (
            "MambaSlotPool.resize called while pool has active entries.")
        assert total_slots % self.dp_size == 0, (
            f"total_slots ({total_slots}) must be divisible by dp_size ({self.dp_size}).")
        self._local_slots = total_slots // self.dp_size
        self._free = self._build_free_lists()

    def allocate(self, req_id: str, rank: int = 0) -> int:
        """Take a slot from the given DP rank's free list."""
        if self._free_deferred:
            self._expire_due(time.perf_counter())
        slot = self._free[rank].pop()
        self._in_use[req_id] = (rank, slot)
        return slot

    def release(self, req_id: str) -> ReleaseOutcome:
        """Release the in-use slot for `req_id`, idempotently."""
        entry = self._in_use.pop(req_id, None)
        if entry is None:
            return ReleaseOutcome.NOT_IN_USE
        rank, slot = entry
        defer_until = (self._on_release(req_id, slot)
                       if self._on_release is not None else None)
        if defer_until is None:
            self._free[rank].append(int(slot))
            return ReleaseOutcome.RELEASED
        self._free_deferred[req_id] = (rank, int(slot), float(defer_until))
        return ReleaseOutcome.DEFERRED

    def release_deferred(self, req_id: str) -> ReleaseOutcome:
        """Return a previously deferred slot to its rank's free list."""
        entry = self._free_deferred.pop(req_id, None)
        if entry is None:
            return ReleaseOutcome.NOT_IN_DEFERRED
        rank, slot, _ = entry
        self._free[rank].append(int(slot))
        return ReleaseOutcome.RELEASED

    def _expire_due(self, now: float) -> list[str]:
        """Auto-release any deferred slot whose deadline has passed."""
        expired: list[str] = []
        for req_id, (rank, slot,
                     expires_at) in list(self._free_deferred.items()):
            if expires_at <= now:
                self._free_deferred.pop(req_id, None)
                self._free[rank].append(int(slot))
                expired.append(req_id)
        return expired

    def register_release_hook(
        self,
        hook: Callable[[str, int], Optional[float]],
    ) -> None:
        """Register the deferred-release callback."""
        self._on_release = hook

    def unregister_release_hook(self) -> None:
        """Clear any previously-registered release hook."""
        self._on_release = None

    @property
    def local_slots(self) -> int:
        """Per-rank slot count. Equal to `total_slots // dp_size`."""
        return self._local_slots

    def get_slot(self, req_id: str) -> Optional[int]:
        """Return the slot for `req_id` regardless of state"""
        entry = self._in_use.get(req_id)
        if entry is not None:
            return int(entry[1])
        entry = self._free_deferred.get(req_id)
        return None if entry is None else int(entry[1])
