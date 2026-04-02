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

import unittest
from unittest.mock import MagicMock, patch

from tpu_inference.core.sched.tpu_scheduler import (DisaggTpuAsyncScheduler,
                                                     DisaggTpuScheduler)
from tpu_inference.runner.continuous_block_pool import ContinuousFreeQueue

print("=== Starting test file ===")


class TestTpuScheduler(unittest.TestCase):

    def test_tpu_scheduler_init(self):
        mock_pool = MagicMock()
        mock_pool.blocks = [MagicMock() for _ in range(10)]
        for i, b in enumerate(mock_pool.blocks):
            b.id = i

        mock_kv_mgr = MagicMock()
        mock_kv_mgr.block_pool = mock_pool

        with patch('vllm.v1.core.sched.scheduler.Scheduler.__init__'
                   ) as mock_base_init:

            def side_effect(*args, **kwargs):
                import inspect
                frame = inspect.currentframe().f_back
                self_obj = frame.f_locals.get('self')
                if self_obj:
                    self_obj.kv_cache_manager = mock_kv_mgr
                else:
                    print("DEBUG inspect failed to find self")

            mock_base_init.side_effect = side_effect

            scheduler = DisaggTpuScheduler()

            mock_base_init.assert_called_once()

            self.assertIsInstance(mock_pool.free_block_queue, ContinuousFreeQueue)
            self.assertIsNotNone(mock_pool.null_block)
            self.assertTrue(mock_pool.null_block.is_null)

    def test_tpu_async_scheduler_init(self):
        mock_pool = MagicMock()
        mock_pool.blocks = [MagicMock() for _ in range(10)]
        for i, b in enumerate(mock_pool.blocks):
            b.id = i

        mock_kv_mgr = MagicMock()
        mock_kv_mgr.block_pool = mock_pool

        with patch('vllm.v1.core.sched.async_scheduler.AsyncScheduler.__init__'
                   ) as mock_base_init:

            def side_effect(*args, **kwargs):
                self_obj = args[0]
                self_obj.kv_cache_manager = mock_kv_mgr

            mock_base_init.side_effect = side_effect

            scheduler = DisaggTpuAsyncScheduler()

            mock_base_init.assert_called_once()

            self.assertIsInstance(mock_pool.free_block_queue, ContinuousFreeQueue)
            self.assertIsNotNone(mock_pool.null_block)
            self.assertTrue(mock_pool.null_block.is_null)

if __name__ == '__main__':
    unittest.main()
