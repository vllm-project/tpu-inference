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

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


class TransferStats:
    """Cumulative stats for kv transfer."""

    def __init__(self, log_prefix: str | None = None, log_interval: int = 32):
        """Initialization

        Args:
            log_prefix: Prefix to prepend to logs. Caller should bake in
                any identifying context (e.g. worker id, role).
            log_interval: Number of stats increment calls between periodic
                summary logging.
        """
        self.log_prefix = log_prefix
        self.log_interval = log_interval
        self.num_sends = 0
        self.bytes_sent = 0
        self.num_pulls = 0
        self.bytes_pulled = 0

    def _log_prefix(self) -> str:
        if self.log_prefix is None:
            return "--> stats |"
        return f"{self.log_prefix} --> stats |"

    def increment_send(self, num_bytes: int):
        """Record a kv cache transfer send.

        Args:
            num_bytes: Number of bytes sent.
        """
        self.num_sends += 1
        self.bytes_sent += num_bytes
        if self.num_sends % self.log_interval == 0:
            logger.info(f"{self._log_prefix()} "
                        f"cumulative_sends={self.num_sends} | "
                        f"cumulative_mb={_bytes_to_mb(self.bytes_sent):.2f}")

    def increment_pull(self, num_bytes: int):
        """Record a kv cache transfer pull

        Args:
            num_bytes: Number of bytes sent.
        """
        self.num_pulls += 1
        self.bytes_pulled += num_bytes
        if self.num_pulls % self.log_interval == 0:
            logger.info(f"{self._log_prefix()} "
                        f"cumulative_pulls={self.num_pulls} | "
                        f"cumulative_mb={_bytes_to_mb(self.bytes_pulled):.2f}")
