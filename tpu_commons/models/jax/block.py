"""Token blocks."""

from typing import List


class LogicalTokenBlock:
    """Represents the state of a logical block in the host."""

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size
        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, num_tokens: int) -> None:
        assert num_tokens <= self.get_num_empty_slots()
        self.num_tokens += num_tokens

    def __repr__(self) -> str:
        return f"({self.block_number}:{self.num_tokens}/{self.block_size})"


class PhysicalTokenBlock:
    """Represents the state of a physical block in the KV cache."""

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size
        self.ref_count = 0

    def __repr__(self) -> str:
        return f"(block_number={self.block_number}, " f"ref_count={self.ref_count})"


# Mapping: logical block number -> physical block.
BlockTable = List[PhysicalTokenBlock]

PrefixHash = int
