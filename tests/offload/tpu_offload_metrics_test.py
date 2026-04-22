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
import time
import unittest
from unittest import mock

from prometheus_client import CollectorRegistry

from tpu_inference.offload.metrics import (PrometheusLogger, TPUKVCacheMetrics,
                                           TPUKVCacheStats,
                                           TPUKVCacheStatsLogger)


class TestTPUKVCacheMetrics(unittest.TestCase):

    def setUp(self):
        TPUKVCacheMetrics.destroy_instance()
        self.metrics = TPUKVCacheMetrics.get_or_create()

    def tearDown(self):
        TPUKVCacheMetrics.destroy_instance()

    def test_singleton(self):
        m1 = TPUKVCacheMetrics.get_or_create()
        m2 = TPUKVCacheMetrics.get_or_create()
        self.assertIs(m1, m2)
        with self.assertRaises(RuntimeError):
            TPUKVCacheMetrics()

    def test_record_metrics(self):
        self.metrics.record_lookup_request()
        self.metrics.record_cache_hit(10)
        self.metrics.record_cache_miss(5)
        self.metrics.record_d2h_operation()
        self.metrics.record_d2h_transfer_latency(0.1)
        self.metrics.record_d2h_transfer_bw(100.0)
        self.metrics.record_d2h_bytes(1024)
        self.metrics.record_h2d_operation()
        self.metrics.record_h2d_transfer_latency(0.2)
        self.metrics.record_h2d_transfer_bw(200.0)
        self.metrics.record_h2d_bytes(2048)
        self.metrics.record_host_memory_usage(10000)
        self.metrics.record_staging_buffer_usage(50)
        self.metrics.record_staging_buffer_free(150)

        stats = self.metrics.get_stats_and_clear()

        self.assertEqual(stats.lookup_requests, 1)
        self.assertEqual(stats.lookup_hits, 10)
        self.assertEqual(stats.lookup_miss, 5)
        self.assertEqual(stats.d2h_operations, 1)
        self.assertEqual(stats.d2h_transfer_latencies, [0.1])
        self.assertEqual(stats.d2h_transfer_bw, [100.0])
        self.assertEqual(stats.d2h_bytes, [1024])
        self.assertEqual(stats.h2d_operations, 1)
        self.assertEqual(stats.h2d_transfer_latencies, [0.2])
        self.assertEqual(stats.h2d_transfer_bw, [200.0])
        self.assertEqual(stats.h2d_bytes, [2048])
        self.assertEqual(stats.host_memory_usage_bytes, 10000)
        self.assertEqual(stats.staging_buffer_usage_blocks, 50)
        self.assertEqual(stats.staging_buffer_free_blocks, 150)

    def test_get_stats_and_clear(self):
        self.metrics.record_lookup_request()
        self.metrics.record_cache_hit(10)
        stats = self.metrics.get_stats_and_clear()
        self.assertEqual(stats.lookup_requests, 1)
        self.assertEqual(stats.lookup_hits, 10)

        # Check if state is reset
        stats = self.metrics.get_stats_and_clear()
        self.assertEqual(stats.lookup_requests, 0)
        self.assertEqual(stats.lookup_hits, 0)
        self.assertEqual(stats.lookup_miss, 0)
        self.assertEqual(stats.d2h_operations, 0)
        self.assertEqual(stats.d2h_transfer_latencies, [])
        self.assertEqual(stats.d2h_transfer_bw, [])
        self.assertEqual(stats.d2h_bytes, [])
        self.assertEqual(stats.h2d_operations, 0)
        self.assertEqual(stats.h2d_transfer_latencies, [])
        self.assertEqual(stats.h2d_transfer_bw, [])
        self.assertEqual(stats.h2d_bytes, [])
        self.assertEqual(stats.host_memory_usage_bytes, 0)
        self.assertEqual(stats.staging_buffer_usage_blocks, 0)
        self.assertEqual(stats.staging_buffer_free_blocks, 0)


class TestPrometheusLogger(unittest.TestCase):

    def setUp(self):
        PrometheusLogger._instance = None
        self.mock_metrics = {}

        # Patch the default registry to isolate tests
        self.registry_patch = mock.patch("prometheus_client.REGISTRY",
                                         new_callable=CollectorRegistry)
        self.mock_registry = self.registry_patch.start()
        self.addCleanup(self.registry_patch.stop)

        def mock_metric_factory(*args, **kwargs):
            metric_name = args[0]
            mock_metric_instance = mock.MagicMock(registry=self.mock_registry)
            self.mock_metrics[metric_name] = mock_metric_instance
            return mock_metric_instance

        # Patch the class attributes on PrometheusLogger directly
        mock_gauge = mock.patch.object(PrometheusLogger,
                                       "_gauge_cls",
                                       side_effect=mock_metric_factory)
        mock_counter = mock.patch.object(PrometheusLogger,
                                         "_counter_cls",
                                         side_effect=mock_metric_factory)
        mock_histogram = mock.patch.object(PrometheusLogger,
                                           "_histogram_cls",
                                           side_effect=mock_metric_factory)

        self.mock_gauge = mock_gauge.start()
        self.mock_counter = mock_counter.start()
        self.mock_histogram = mock_histogram.start()

        self.addCleanup(mock_gauge.stop)
        self.addCleanup(mock_counter.stop)
        self.addCleanup(mock_histogram.stop)

    def tearDown(self):
        PrometheusLogger._instance = None

    def test_singleton(self):
        p1 = PrometheusLogger.get_or_create("model1", "dev1")
        p2 = PrometheusLogger.get_or_create("model1", "dev1")
        self.assertIs(p1, p2)
        with self.assertRaises(RuntimeError):
            PrometheusLogger()

    def test_log_stats(self):
        prometheus_logger = PrometheusLogger.get_or_create(
            "test_model", "test_device")
        stats = TPUKVCacheStats(
            lookup_requests=1,
            lookup_hits=10,
            lookup_miss=5,
            d2h_operations=2,
            d2h_bytes=[1024, 2048],
            d2h_transfer_latencies=[0.1, 0.2],
            d2h_transfer_bw=[100.0, 150.0],
            h2d_operations=3,
            h2d_bytes=[4096],
            h2d_transfer_latencies=[0.3],
            h2d_transfer_bw=[200.0],
            host_memory_usage_bytes=1024**3 * 2,
            staging_buffer_usage_blocks=50,
            staging_buffer_free_blocks=150,
        )

        prometheus_logger.log_stats(stats)

        self.mock_metrics[
            "tpu_inference:prefix_cache_lookup_queries_total"].labels(
            ).inc.assert_called_once_with(1)
        self.mock_metrics["tpu_inference:prefix_cache_hits_total"].labels(
        ).inc.assert_called_once_with(10)
        self.mock_metrics["tpu_inference:prefix_cache_miss_total"].labels(
        ).inc.assert_called_once_with(5)
        self.mock_metrics[
            "tpu_inference:prefix_cache_d2h_operations_total"].labels(
            ).inc.assert_called_once_with(2)
        self.mock_metrics[
            "tpu_inference:prefix_cache_h2d_operations_total"].labels(
            ).inc.assert_called_once_with(3)

        d2h_duration_mock = self.mock_metrics[
            "tpu_inference:prefix_cache_d2h_transfer_duration_seconds"].labels(
            )
        self.assertEqual(d2h_duration_mock.observe.call_count, 2)
        d2h_duration_mock.observe.assert_any_call(0.1)
        d2h_duration_mock.observe.assert_any_call(0.2)

        d2h_bw_mock = self.mock_metrics[
            "tpu_inference:prefix_cache_d2h_transfer_bw"].labels()
        self.assertEqual(d2h_bw_mock.observe.call_count, 2)
        d2h_bw_mock.observe.assert_any_call(100.0)
        d2h_bw_mock.observe.assert_any_call(150.0)

        h2d_duration_mock = self.mock_metrics[
            "tpu_inference:prefix_cache_h2d_transfer_duration_seconds"].labels(
            )
        self.assertEqual(h2d_duration_mock.observe.call_count, 1)
        h2d_duration_mock.observe.assert_called_once_with(0.3)

        h2d_bw_mock = self.mock_metrics[
            "tpu_inference:prefix_cache_h2d_transfer_bw"].labels()
        self.assertEqual(h2d_bw_mock.observe.call_count, 1)
        h2d_bw_mock.observe.assert_called_once_with(200.0)

        d2h_bytes_mock = self.mock_metrics[
            "tpu_inference:prefix_cache_d2h_request_bytes"].labels()
        self.assertEqual(d2h_bytes_mock.observe.call_count, 2)
        d2h_bytes_mock.observe.assert_any_call(1024)
        d2h_bytes_mock.observe.assert_any_call(2048)

        h2d_bytes_mock = self.mock_metrics[
            "tpu_inference:prefix_cache_h2d_request_bytes"].labels()
        self.assertEqual(h2d_bytes_mock.observe.call_count, 1)
        h2d_bytes_mock.observe.assert_called_once_with(4096)

        self.mock_metrics[
            "tpu_inference:prefix_cache_host_memory_usage"].labels(
            ).set.assert_called_once_with(2.0)
        self.mock_metrics[
            "tpu_inference:prefix_cache_staging_buffer_usage"].labels(
            ).set.assert_called_once_with(50)
        self.mock_metrics[
            "tpu_inference:prefix_cache_staging_buffer_free"].labels(
            ).set.assert_called_once_with(150)

    def test_prometheus_multiproc_dir(self):
        tmp_dir = "/tmp/test_prometheus_multiproc"
        PrometheusLogger._instance = None

        # Safely patch the environment variable for the scope of the context manager
        with mock.patch.dict(os.environ,
                             {"PROMETHEUS_MULTIPROC_DIR": tmp_dir}):
            PrometheusLogger.get_or_create("test_model", "test_device")
            self.assertTrue(os.path.exists(tmp_dir))

        os.rmdir(tmp_dir)


class TestTPUKVCacheStatsLogger(unittest.TestCase):

    def setUp(self):
        TPUKVCacheMetrics.destroy_instance()
        PrometheusLogger._instance = None
        self.mock_metrics = mock.MagicMock(spec=TPUKVCacheMetrics)
        self.mock_prometheus_logger = mock.MagicMock(spec=PrometheusLogger)

        mock.patch.object(TPUKVCacheMetrics,
                          "get_or_create",
                          return_value=self.mock_metrics).start()
        mock.patch.object(PrometheusLogger,
                          "get_or_create",
                          return_value=self.mock_prometheus_logger).start()

        self.addCleanup(mock.patch.stopall)

    def test_logger_thread(self):
        stats = TPUKVCacheStats(lookup_requests=1)
        self.mock_metrics.get_stats_and_clear.return_value = stats

        logger = TPUKVCacheStatsLogger(log_interval=0.1)
        time.sleep(0.2)  # Allow thread to run at least once
        logger.shutdown()

        self.mock_metrics.get_stats_and_clear.assert_called()
        self.mock_prometheus_logger.log_stats.assert_called_with(stats)
        self.assertFalse(logger.thread.is_alive())

    def test_shutdown(self):
        logger = TPUKVCacheStatsLogger(log_interval=10)
        self.assertTrue(logger.thread.is_alive())
        logger.shutdown()
        self.assertFalse(logger.thread.is_alive())
        self.assertTrue(logger.shutdown_event.is_set())

    @mock.patch("tpu_inference.offload.metrics.logger")
    def test_shutdown_timeout(self, mock_logger):
        logger = TPUKVCacheStatsLogger(log_interval=0.1)
        original_join = logger.thread.join
        with mock.patch.object(logger.thread, "is_alive", return_value=True):
            with mock.patch.object(
                    logger.thread,
                    "join",
                    side_effect=lambda *args, **kwargs: original_join(0.01)):
                logger.shutdown()
        mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
