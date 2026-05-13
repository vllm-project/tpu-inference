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
"""Stats and Prometheus metrics for the TPU KV connector."""

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics, KVConnectorStats, PromMetric, PromMetricT)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class TpuKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def record_d2h_transfer(self, d2h_slice_time, d2h_transfer_time):
        """Record a D2H transfer operation."""
        self.data["d2h_slice_time"].append(d2h_slice_time)
        self.data["d2h_transfer_time"].append(d2h_transfer_time)

    def record_successful_transfer(self, prepare_time, transfer_time,
                                   mb_transferred):
        """Record a successful TPU KV transfer operation."""
        self.data["prepare_time"].append(prepare_time)
        self.data["transfer_time"].append(transfer_time)
        self.data["mb_transferred"].append(mb_transferred)

    def record_failed_transfer(self):
        """Record a failed TPU KV transfer operation."""
        self.data["num_failed_transfers"].append(1)

    def reset(self):
        # Must be serializable
        self.data: dict[str, list[float | int]] = {
            "d2h_slice_time": [],
            "d2h_transfer_time": [],
            "prepare_time": [],
            "transfer_time": [],
            "mb_transferred": [],
            "num_failed_transfers": [],
        }

    def clone_and_reset(self) -> "TpuKVConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                assert isinstance(accumulator, list)
                accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        d2h_slice_time = np.asarray(self.data["d2h_slice_time"])
        d2h_transfer_time = np.asarray(self.data["d2h_transfer_time"])
        prepare_time = np.asarray(self.data["prepare_time"])
        transfer_time = np.asarray(self.data["transfer_time"])
        mb_transferred = np.asarray(self.data["mb_transferred"])

        total_mb = mb_transferred.sum()
        avg_mb = total_mb / self.num_successful_transfers if self.num_successful_transfers > 0 else 0

        total_time_seconds = transfer_time.sum() / 1e3
        throughput_mb_s = total_mb / total_time_seconds if total_time_seconds > 0 else 0

        return {
            "Avg D2H slice time (ms)":
            round(d2h_slice_time.mean(), 3)
            if d2h_slice_time.size > 0 else 0.0,
            "P90 D2H slice time (ms)":
            round(np.percentile(d2h_slice_time, 90).item(), 3)
            if d2h_slice_time.size > 0 else 0.0,
            "Avg D2H transfer time (ms)":
            round(d2h_transfer_time.mean(), 3)
            if d2h_transfer_time.size > 0 else 0.0,
            "P90 D2H transfer time (ms)":
            round(np.percentile(d2h_transfer_time, 90).item(), 3)
            if d2h_transfer_time.size > 0 else 0.0,
            "Num successful transfers":
            self.num_successful_transfers,
            "Num failed transfers":
            self.data['num_failed_transfers'],
            "Avg KV transfer prepare time (ms)":
            round(prepare_time.mean(), 3) if prepare_time.size > 0 else 0.0,
            "P90 KV transfer prepare time (ms)":
            round(np.percentile(prepare_time, 90).item(), 3)
            if prepare_time.size > 0 else 0.0,
            "Avg KV transfer time (ms)":
            round(transfer_time.mean(), 3) if transfer_time.size > 0 else 0.0,
            "P90 KV transfer time (ms)":
            round(np.percentile(transfer_time, 90).item(), 3)
            if transfer_time.size > 0 else 0.0,
            "Avg MB per transfer":
            round(avg_mb, 3),
            "Throughput (MB/s)":
            round(throughput_mb_s, 3),
        }

    def is_empty(self) -> bool:
        return (len(self.data["d2h_slice_time"]) == 0
                and len(self.data["d2h_transfer_time"]) == 0
                and self.num_successful_transfers == 0
                and len(self.data["num_failed_transfers"]) == 0)

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_time"])


class TpuKVConnectorPromMetrics(KVConnectorPromMetrics):

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames,
                         per_engine_labelvalues)

        # Buckets for time-based metrics in milliseconds
        buckets = [
            1.0,
            10.0,
            100.0,
            250.0,
            500.0,
            750.0,
            1000.0,
            2500.0,
            5000.0,
            7500.0,
            10000.0,
            25000.0,
            50000.0,
        ]
        tpu_histogram_d2h_slice_time = self._histogram_cls(
            name="vllm:tpu_d2h_slice_time_ms",
            documentation=
            "Histogram of D2H slice duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_d2h_slice_time = create_metric_per_engine(
            tpu_histogram_d2h_slice_time, self.per_engine_labelvalues)
        tpu_histogram_d2h_transfer_time = self._histogram_cls(
            name="vllm:tpu_d2h_transfer_time_ms",
            documentation=
            "Histogram of D2H transfer duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_d2h_transfer_time = create_metric_per_engine(
            tpu_histogram_d2h_transfer_time, self.per_engine_labelvalues)
        tpu_histogram_kv_prepare_time = self._histogram_cls(
            name="vllm:tpu_kv_prepare_time_ms",
            documentation=
            "Histogram of prepare duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_kv_prepare_time = create_metric_per_engine(
            tpu_histogram_kv_prepare_time, self.per_engine_labelvalues)
        tpu_histogram_kv_transfer_time = self._histogram_cls(
            name="vllm:tpu_kv_transfer_time_ms",
            documentation=
            "Histogram of transfer duration for TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_kv_transfer_time = create_metric_per_engine(
            tpu_histogram_kv_transfer_time, self.per_engine_labelvalues)
        # Buckets for data transferred
        buckets = [
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]
        tpu_histogram_kv_megabytes_transferred = self._histogram_cls(
            name="vllm:tpu_kv_megabytes_transferred",
            documentation=
            "Histogram of megabytes transferred per TPU KV Cache transfers.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.tpu_histogram_kv_megabytes_transferred = create_metric_per_engine(
            tpu_histogram_kv_megabytes_transferred,
            self.per_engine_labelvalues)
        counter_tpu_num_failed_transfers = self._counter_cls(
            name="vllm:tpu_num_failed_transfers",
            documentation="Number of failed TPU KV Cache transfers.",
            labelnames=labelnames,
        )
        self.counter_tpu_num_failed_transfers = create_metric_per_engine(
            counter_tpu_num_failed_transfers, self.per_engine_labelvalues)

    def observe(self,
                transfer_stats_data: dict[str, Any],
                engine_idx: int = 0):
        for prom_obj, list_item_key in zip(
            [
                self.tpu_histogram_d2h_slice_time,
                self.tpu_histogram_d2h_transfer_time,
                self.tpu_histogram_kv_prepare_time,
                self.tpu_histogram_kv_transfer_time,
                self.tpu_histogram_kv_megabytes_transferred,
            ],
            [
                "d2h_slice_time",
                "d2h_transfer_time",
                "prepare_time",
                "transfer_time",
                "mb_transferred",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)

        for counter_obj, counter_item_key in zip(
            [
                self.counter_tpu_num_failed_transfers,
            ],
            ["num_failed_transfers"],
        ):
            for list_item in transfer_stats_data[counter_item_key]:
                counter_obj[engine_idx].inc(list_item)
