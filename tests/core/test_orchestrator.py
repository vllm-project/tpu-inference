# SPDX-License-Identifier: Apache-2.0

import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tpu_commons.core import orchestrator


class OrchestratorTest(unittest.TestCase):

    def setUp(self):
        # Stop the JetThread from killing the test process on exceptions
        # by patching it to be a regular thread.
        self.jet_thread_patcher = patch(
            "tpu_commons.core.orchestrator.JetThread", threading.Thread)
        self.jet_thread_patcher.start()
        self.addCleanup(self.jet_thread_patcher.stop)

    def _create_mock_engine(self, name="engine", max_concurrent_decodes=4):
        """Creates a mock JaxEngine."""
        engine = MagicMock(name=name)
        engine.is_prefill_idle.return_value = True
        engine.has_more_capacity.return_value = True
        engine.max_concurrent_decodes = max_concurrent_decodes
        engine.model_runner = MagicMock()

        engine.dump_stats.return_value = f"{name} stats"
        return engine

    def _create_mock_request(self,
                             request_id="test_req_1",
                             num_tokens=10,
                             client_index=0):
        """Creates a mock vLLM Request."""
        request = MagicMock(spec=Request)
        request.request_id = request_id
        request.num_tokens = num_tokens
        request.num_computed_tokens = 0
        request.num_cached_tokens = 0
        request.client_index = client_index
        request.status = RequestStatus.WAITING

        # is_finished will be controlled during the test
        request.is_finished.side_effect = [False, True]

        # Mock methods that are called on the request
        request.get_finished_reason.return_value = "length"
        request.take_events.return_value = []
        request.stop_reason = None

        return request

    def test_disaggregated_mode_lifecycle(self):
        """Tests a single request lifecycle in disaggregated mode."""
        mock_prefill_engine = self._create_mock_engine("prefill")
        mock_generate_engine = self._create_mock_engine("generate")
        driver = orchestrator.Driver(vllm_config=MagicMock(),
                                     prefill_engines=[mock_prefill_engine],
                                     generate_engines=[mock_generate_engine])

        mock_req = self._create_mock_request()

        # --- Mock setup for the request lifecycle ---
        prefill_kv_cache = {"data": "prefilled_kv"}
        prefill_runner_output = ModelRunnerOutput(
            req_ids=[mock_req.request_id],
            req_id_to_index={mock_req.request_id: 0},
            sampled_token_ids=[[101]],
            prompt_logprobs_dict={mock_req.request_id: [[(0, 0.0)]]},
            spec_token_ids=[],
            logprobs=None,
            pooler_output=[],
        )
        mock_prefill_engine.prefill.return_value = (
            {mock_req.request_id: prefill_kv_cache}, prefill_runner_output)

        mock_prefill_engine.is_prefill_idle.side_effect = [False, False, True, True]
        mock_generate_engine.is_generate_idle.side_effect = [False, False, False, False, True, True]

        mock_prefill_engine.model_runner.transfer_kv_cache.return_value = MagicMock()

        generate_runner_output_1 = ModelRunnerOutput(
            req_ids=[mock_req.request_id],
            req_id_to_index={mock_req.request_id: 0},
            sampled_token_ids=[[102]],
            logprobs=None,
            prompt_logprobs_dict={},
            spec_token_ids=[],
            pooler_output=[],
        )
        generate_runner_output_2 = ModelRunnerOutput(
            req_ids=[mock_req.request_id],
            req_id_to_index={mock_req.request_id: 0},
            sampled_token_ids=[[103]],
            spec_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        empty_runner_output = ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            spec_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

        mock_generate_engine.generate.side_effect = [
            generate_runner_output_1,
            generate_runner_output_2,
            # After finishing, generate should return empty output for this req
            empty_runner_output,
        ]

        try:
            # --- Start of test execution ---
            driver.place_request_on_prefill_queue(mock_req)

            # Get prefill output
            prefill_output = driver._vllm_output_backlogs.get(timeout=5)
            self.assertIn(0, prefill_output)
            self.assertEqual(prefill_output[0].outputs[0].request_id,
                             mock_req.request_id)
            self.assertEqual(prefill_output[0].outputs[0].new_token_ids, [101])

            # Get first generate output
            gen_output1 = driver._vllm_output_backlogs.get(timeout=5)
            self.assertIn(0, gen_output1)
            self.assertEqual(gen_output1[0].outputs[0].request_id,
                             mock_req.request_id)
            self.assertEqual(gen_output1[0].outputs[0].new_token_ids, [102])

            # Get second generate output
            gen_output2 = driver._vllm_output_backlogs.get(timeout=5)
            self.assertIn(0, gen_output2)
            self.assertEqual(gen_output2[0].outputs[0].request_id,
                             mock_req.request_id)
            self.assertEqual(gen_output2[0].outputs[0].new_token_ids, [103])
            
            mock_req.is_finished.return_value = True

            # Let threads run to free the request.
            # free_request is called in the generate thread.
            time.sleep(0.5)

            # --- Assertions on mock calls ---
            mock_prefill_engine.add_request.assert_called_once_with(
                mock_req, mock_req.num_tokens)
            mock_prefill_engine.prefill.assert_called_once()
            mock_generate_engine.model_runner.transfer_kv_cache.assert_called_once(
            )
            mock_generate_engine.model_runner.insert_request_with_kv_cache.assert_called_once(
            )
            mock_generate_engine.add_request.assert_called_once_with(
                mock_req, 1)
            # The generate loop might run more than twice.
            self.assertGreaterEqual(mock_generate_engine.generate.call_count,
                                    2)
            mock_prefill_engine.free_request.assert_called_once_with(mock_req)
            mock_generate_engine.free_request.assert_called_once_with(mock_req)

            # Check that request is removed from driver's tracking
            self.assertNotIn(mock_req.request_id, driver.requests)
        finally:
            driver.stop()

    def test_driver_stop(self):
        """Tests if the driver can be stopped correctly."""
        mock_prefill_engine = self._create_mock_engine("prefill")
        mock_generate_engine = self._create_mock_engine("generate")

        driver = orchestrator.Driver(vllm_config=MagicMock(),
                                     prefill_engines=[mock_prefill_engine],
                                     generate_engines=[mock_generate_engine])

        # Check all threads are alive
        self.assertTrue(all(t.is_alive() for t in driver._all_threads))

        # Stop driver
        driver.stop()

        # Check all threads are stopped
        for t in driver._all_threads:
            # Give some time for threads to join
            t.join(timeout=1)
        self.assertFalse(any(t.is_alive() for t in driver._all_threads))


if __name__ == '__main__':
    unittest.main()
