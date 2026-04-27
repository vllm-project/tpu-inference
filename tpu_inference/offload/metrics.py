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

import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional

from prometheus_client import Counter, Gauge, Histogram

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Custom buckets for latency histograms (in seconds)
LATENCY_BUCKETS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0,
                   2.0, 5.0, 10.0, 20, 30, float("inf"))

# Custom buckets defined in GBps
# Adjust these based on your hardware (e.g., PCIe 4.0/5.0 limits)
GBPS_BUCKETS = (1.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 500.0,
                float("inf"))

# Define buckets for bytes (e.g., 1KB to 100MB+)
BYTE_BUCKETS = (1024, 10240, 102400, 1048576, 10485760, 104857600,
                float("inf"))


@dataclass
class TPUKVCacheStats:
    """
    Data class to hold a snapshot of TPU KVCache statistics.
    """
    lookup_requests: int = 0
    lookup_hits: int = 0
    lookup_miss: int = 0
    d2h_operations: int = 0
    d2h_bytes: List[int] = field(default_factory=list)
    d2h_transfer_latencies: List[float] = field(default_factory=list)
    d2h_transfer_bw: List[float] = field(default_factory=list)
    h2d_operations: int = 0
    h2d_bytes: List[int] = field(default_factory=list)
    h2d_transfer_latencies: List[float] = field(default_factory=list)
    h2d_transfer_bw: List[float] = field(default_factory=list)
    host_memory_usage_bytes: int = 0
    staging_buffer_usage_blocks: int = 0
    staging_buffer_free_blocks: int = 0


class TPUKVCacheMetrics:
    """
    Singleton class for collecting and managing TPU KVCache metrics.

    This class provides thread-safe methods to record various metrics related to
    cache lookups, data transfers, and memory usage.
    """
    _instance: Optional["TPUKVCacheMetrics"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self):
        """Initializes the TPUKVCacheMetrics singleton instance."""
        if TPUKVCacheMetrics._instance is not None:
            raise RuntimeError("TPUKVCacheMetrics is a singleton")

        self._lookup_requests: int = 0
        self._lookup_hits: int = 0
        self._lookup_miss: int = 0
        self._d2h_operations: int = 0
        self._d2h_bytes: List[int] = []
        self._d2h_transfer_latencies: List[float] = []
        self._d2h_transfer_bw: List[float] = []
        self._h2d_operations: int = 0
        self._h2d_bytes: List[int] = []
        self._h2d_transfer_latencies: List[float] = []
        self._h2d_transfer_bw: List[float] = []
        self._host_memory_usage_bytes: int = 0
        self._staging_buffer_usage_blocks: int = 0
        self._staging_buffer_free_blocks: int = 0

        self._instance_lock = threading.Lock()
        self._reset_state()

    @classmethod
    def get_or_create(cls) -> "TPUKVCacheMetrics":
        if cls._instance is None:
            with cls._class_lock:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def destroy_instance(cls) -> None:
        with cls._class_lock:
            cls._instance = None

    def record_lookup_request(self):
        with self._instance_lock:
            self._lookup_requests += 1

    def record_cache_hit(self, tokens: int):
        with self._instance_lock:
            self._lookup_hits += tokens

    def record_cache_miss(self, tokens: int):
        with self._instance_lock:
            self._lookup_miss += tokens

    def record_d2h_operation(self):
        with self._instance_lock:
            self._d2h_operations += 1

    def record_d2h_transfer_latency(self, duration: float):
        with self._instance_lock:
            self._d2h_transfer_latencies.append(duration)

    def record_d2h_transfer_bw(self, bandwidth: float):
        with self._instance_lock:
            self._d2h_transfer_bw.append(bandwidth)

    def record_d2h_bytes(self, bytes: int):
        with self._instance_lock:
            self._d2h_bytes.append(bytes)

    def record_h2d_operation(self):
        with self._instance_lock:
            self._h2d_operations += 1

    def record_h2d_transfer_latency(self, duration: float):
        with self._instance_lock:
            self._h2d_transfer_latencies.append(duration)

    def record_h2d_transfer_bw(self, bandwidth: float):
        with self._instance_lock:
            self._h2d_transfer_bw.append(bandwidth)

    def record_h2d_bytes(self, bytes: int):
        with self._instance_lock:
            self._h2d_bytes.append(bytes)

    def record_host_memory_usage(self, bytes_used: int):
        with self._instance_lock:
            self._host_memory_usage_bytes = bytes_used

    def record_staging_buffer_usage(self, blocks_used: int):
        with self._instance_lock:
            self._staging_buffer_usage_blocks = blocks_used

    def record_staging_buffer_free(self, blocks_free: int):
        with self._instance_lock:
            self._staging_buffer_free_blocks = blocks_free

    def _reset_state(self):
        self._lookup_requests = 0
        self._lookup_hits = 0
        self._lookup_miss = 0
        self._d2h_operations = 0
        self._d2h_bytes.clear()
        self._d2h_transfer_latencies.clear()
        self._d2h_transfer_bw.clear()
        self._h2d_operations = 0
        self._h2d_bytes.clear()
        self._h2d_transfer_latencies.clear()
        self._h2d_transfer_bw.clear()
        self._host_memory_usage_bytes = 0
        self._staging_buffer_usage_blocks = 0
        self._staging_buffer_free_blocks = 0

    def get_stats_and_clear(self) -> TPUKVCacheStats:
        with self._instance_lock:
            stats = TPUKVCacheStats(
                lookup_requests=self._lookup_requests,
                lookup_hits=self._lookup_hits,
                lookup_miss=self._lookup_miss,
                d2h_operations=self._d2h_operations,
                d2h_bytes=list(self._d2h_bytes),
                d2h_transfer_latencies=list(self._d2h_transfer_latencies),
                d2h_transfer_bw=list(self._d2h_transfer_bw),
                h2d_operations=self._h2d_operations,
                h2d_bytes=list(self._h2d_bytes),
                h2d_transfer_latencies=list(self._h2d_transfer_latencies),
                h2d_transfer_bw=list(self._h2d_transfer_bw),
                host_memory_usage_bytes=self._host_memory_usage_bytes,
                staging_buffer_usage_blocks=self._staging_buffer_usage_blocks,
                staging_buffer_free_blocks=self._staging_buffer_free_blocks,
            )
            self._reset_state()
        return stats


class PrometheusLogger:
    _gauge_cls = Gauge
    _counter_cls = Counter
    _histogram_cls = Histogram

    _instance: Optional["PrometheusLogger"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self,
                 model_name: Optional[str] = None,
                 device_type: Optional[str] = None):
        if PrometheusLogger._instance is not None:
            raise RuntimeError("PrometheusLogger is a singleton")

        # Ensure PROMETHEUS_MULTIPROC_DIR is set before any metric registration
        pmd = os.environ.get("PROMETHEUS_MULTIPROC_DIR",
                             "/tmp/prometheus_multiproc")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = pmd
        os.makedirs(pmd, exist_ok=True)

        labels = {
            "model_name": model_name,
            "device_type": device_type,
        }
        labelnames = list(labels.keys())

        self.lookup_requests = self._counter_cls(
            "tpu_inference:prefix_cache_lookup_queries_total",
            "Total number of prefix cache lookup queries",
            labelnames=labelnames,
        ).labels(**labels)
        self.lookup_hit = self._counter_cls(
            "tpu_inference:prefix_cache_hits_total",
            "Total number of tokens prefix cache hits",
            labelnames=labelnames,
        ).labels(**labels)
        self.lookup_miss = self._counter_cls(
            "tpu_inference:prefix_cache_miss_total",
            "Total number of tokens prefix cache miss",
            labelnames=labelnames,
        ).labels(**labels)
        self.d2h_operations = self._counter_cls(
            "tpu_inference:prefix_cache_d2h_operations_total",
            "Total number of save data from device to host memory operations",
            labelnames=labelnames,
        ).labels(**labels)
        self.d2h_transfer_duration = self._histogram_cls(
            "tpu_inference:prefix_cache_d2h_transfer_duration_seconds",
            "Latency of transfer KV cache from device to host memory",
            labelnames=labelnames,
            buckets=LATENCY_BUCKETS,
        ).labels(**labels)
        self.d2h_transfer_bw = self._histogram_cls(
            "tpu_inference:prefix_cache_d2h_transfer_bw",
            "Bandwidth of transfer KV cache from device to host memory in GBps",
            labelnames=labelnames,
            unit="GBps",
            buckets=GBPS_BUCKETS,
        ).labels(**labels)
        self.d2h_bytes = self._histogram_cls(
            "tpu_inference:prefix_cache_d2h_request_bytes",
            "Distribution of save request sizes (D2H)",
            labelnames=labelnames,
            buckets=BYTE_BUCKETS).labels(**labels)
        self.h2d_operations = self._counter_cls(
            "tpu_inference:prefix_cache_h2d_operations_total",
            "Total number of load data from host memory to device operations",
            labelnames=labelnames,
        ).labels(**labels)
        self.h2d_transfer_duration = self._histogram_cls(
            "tpu_inference:prefix_cache_h2d_transfer_duration_seconds",
            "Latency of transfer KV cache from host memory to device",
            labelnames=labelnames,
            buckets=LATENCY_BUCKETS,
        ).labels(**labels)
        self.h2d_transfer_bw = self._histogram_cls(
            "tpu_inference:prefix_cache_h2d_transfer_bw",
            "Bandwidth of transfer KV cache from host memory to device in GBps",
            labelnames=labelnames,
            unit="GBps",
            buckets=GBPS_BUCKETS,
        ).labels(**labels)
        self.h2d_bytes = self._histogram_cls(
            "tpu_inference:prefix_cache_h2d_request_bytes",
            "Distribution of load request sizes (H2D)",
            labelnames=labelnames,
            buckets=BYTE_BUCKETS).labels(**labels)
        self.host_memory_usage = self._gauge_cls(
            "tpu_inference:prefix_cache_host_memory_usage",
            "Current host memory usage by the KV cache offload system",
            labelnames=labelnames,
            unit="GiB").labels(**labels)
        self.staging_buffer_usage = self._gauge_cls(
            "tpu_inference:prefix_cache_staging_buffer_usage",
            "Current staging buffer usage by the KV cache offload system",
            labelnames=labelnames,
            unit="blocks").labels(**labels)
        self.staging_buffer_free = self._gauge_cls(
            "tpu_inference:prefix_cache_staging_buffer_free",
            "Current staging buffer free for the KV cache offload system",
            labelnames=labelnames,
            unit="blocks").labels(**labels)

    def log_stats(self, stats: TPUKVCacheStats):
        """Updates Prometheus metrics from TPUKVCacheStats."""

        if stats.lookup_requests > 0:
            self.lookup_requests.inc(stats.lookup_requests)
        if stats.lookup_hits > 0:
            self.lookup_hit.inc(stats.lookup_hits)
        if stats.lookup_miss > 0:
            self.lookup_miss.inc(stats.lookup_miss)
        if stats.d2h_operations > 0:
            self.d2h_operations.inc(stats.d2h_operations)
        if stats.h2d_operations > 0:
            self.h2d_operations.inc(stats.h2d_operations)

        for latency in stats.d2h_transfer_latencies:
            self.d2h_transfer_duration.observe(latency)
        for bandwidth in stats.d2h_transfer_bw:
            self.d2h_transfer_bw.observe(bandwidth)
        for latency in stats.h2d_transfer_latencies:
            self.h2d_transfer_duration.observe(latency)
        for bandwidth in stats.h2d_transfer_bw:
            self.h2d_transfer_bw.observe(bandwidth)
        for size in stats.d2h_bytes:
            self.d2h_bytes.observe(size)
        for size in stats.h2d_bytes:
            self.h2d_bytes.observe(size)

        self.host_memory_usage.set(stats.host_memory_usage_bytes / (1024**3))
        self.staging_buffer_usage.set(stats.staging_buffer_usage_blocks)
        self.staging_buffer_free.set(stats.staging_buffer_free_blocks)

    @classmethod
    def get_or_create(cls,
                      model: Optional[str] = None,
                      device_type: Optional[str] = None) -> "PrometheusLogger":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls(model, device_type)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PrometheusLogger":
        assert cls._instance is not None, (
            "PrometheusLogger instance not created yet")
        return cls._instance

    @classmethod
    def get_instance_or_none(cls) -> Optional["PrometheusLogger"]:
        """
        Returns the singleton instance of PrometheusLogger if it exists,
        otherwise returns None.
        """
        return cls._instance


class TPUKVCacheStatsLogger:

    def __init__(self,
                 log_interval: int,
                 model_name: Optional[str] = None,
                 device_type: Optional[str] = None):
        logger.info(
            f"Initiating TPUKVCacheStatsLogger, log interval: {log_interval} seconds"
        )
        self.log_interval = log_interval
        self.metrics = TPUKVCacheMetrics.get_or_create()
        self.prometheus_logger = PrometheusLogger.get_or_create(
            model_name, device_type)
        self.is_running = True
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self.log_worker,
                                       daemon=True,
                                       name="tpukvcachestats-logger-thread")
        self.thread.start()

    def log_worker(self):
        while not self.shutdown_event.is_set():
            stats = self.metrics.get_stats_and_clear()
            self.prometheus_logger.log_stats(stats)
            # wait returns True if the flag is set, False if timeout
            if self.shutdown_event.wait(self.log_interval):
                break

    def shutdown(self):
        """Gracefully shuts down the stats logger thread, waking it immediately if sleeping."""
        logger.info("Initiating shutdown of TPUKVCacheStatsLogger...")

        # 1. Signal the worker thread to stop and wake it from any sleep state
        self.is_running = False
        self.shutdown_event.set()

        # 2. Early exit if the thread has already finished
        if not self.thread.is_alive():
            logger.info("Stats logger thread has already stopped.")
            logger.info("TPUKVCacheStatsLogger shutdown complete.")
            return

        # 3. Wait for the thread to finish its current loop
        timeout = 10.0
        logger.info(
            f"Waiting up to {timeout}s for the stats logger thread to finish..."
        )

        self.thread.join(timeout=timeout)

        # 4. Verify termination
        if self.thread.is_alive():
            logger.warning(
                f"Stats logger thread failed to terminate within the {timeout}s timeout. "
                "It may be blocked on an I/O operation. Proceeding with shutdown."
            )
        else:
            logger.info("Stats logger thread terminated successfully.")

        logger.info("TPUKVCacheStatsLogger shutdown complete.")
