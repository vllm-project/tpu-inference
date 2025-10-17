# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LMCache project

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from tpu_inference.logger import init_logger

# Corresponds to the initial hash value
NONE_HASH = 0

logger = init_logger(__name__)


@dataclass(order=True)
class CacheKey:
    """
    A key for the cache engine.
    """
    model_name: str
    chunk_hash: int

    def __hash__(self):
        return hash((
            self.model_name,
            self.chunk_hash,
        ))

    def __eq__(self, other):
        if type(self) is type(other):
            return (self.model_name == other.model_name
                    and self.chunk_hash == other.chunk_hash)
        return False


class TokenProcessor:

    def __init__(self, model_name: str, chunk_size: int = 16):
        self.model_name = model_name
        self.chunk_size = chunk_size
        logger.info(f"TokenProcessor initialized with chunk_size={chunk_size}")

    def _hash_tokens(
        self,
        tokens: List[int],
        prefix_hash: Optional[int] = None,
    ) -> int:
        hasher = hashlib.sha256()
        hasher.update(str(prefix_hash).encode('utf-8'))
        hasher.update(str(tuple(tokens)).encode('utf-8'))
        return int(hasher.hexdigest(), 16)

    def process_tokens(
        self,
        tokens: Optional[List[int]] = None,
    ) -> Iterable[Tuple[int, int, CacheKey]]:
        """Process the tokens and return the corresponding cache keys."""
        if not tokens:
            return

        total_len = len(tokens)
        prefix_hash = NONE_HASH

        for i in range(0, total_len, self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            prefix_hash = self._hash_tokens(chunk, prefix_hash)
            start_idx = i
            end_idx = min(start_idx + self.chunk_size, total_len)
            logger.info(
                f"Processing chunk: start={start_idx}, end={end_idx}, hash={prefix_hash}"
            )
            yield (
                start_idx,
                end_idx,
                CacheKey(model_name=self.model_name, chunk_hash=prefix_hash),
            )
