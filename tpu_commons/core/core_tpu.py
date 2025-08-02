# SPDX-License-Identifier: Apache-2.0
import functools
import itertools
import math
import math
import os
import queue
import signal
import signal
import threading
import time
import traceback
from typing import Any, Callable, Optional, TypeVar, Union

import jax
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.engine.core import EngineCoreProc as vLLMEngineCoreProc
from vllm.v1.request import Request, RequestStatus

from tpu_commons.core import disagg_executor, disagg_utils
from tpu_commons.interfaces.engine import IDisaggEngineCoreProc
from tpu_commons.runner.utils import LatencyTracker

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc


class JetThread(threading.Thread):
    """Thread that kills the program if it fails.

    If a driver thread goes down, we can't operate.
    """

    def run(self):
        try:
            super().run()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Thread {self.name} encountered an error: {e}")
            traceback.print_exc()
            os.kill(os.getpid(), signal.SIGKILL)


class _DisaggOrchestrator:
    """Internal orchestrator with clean dependencies.

    This class contains the actual disaggregation logic. It is instantiated
    by the DisaggEngineCoreProc adapter.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        log_stats: bool,
        engine_index: int = 0,
        **kwargs,
        **kwargs,
    ):
        if 'dp_rank' in kwargs or 'local_dp_rank' in kwargs:
            logger.debug(
                "Ignoring data parallelism arguments for non-DP disaggregated engine."
            )
        if 'dp_rank' in kwargs or 'local_dp_rank' in kwargs:
            logger.debug(
                "Ignoring data parallelism arguments for non-DP disaggregated engine."
            )
        # We don't invoke super class's ctor as we are not really the
        # engine core to be executed, instead we create other instance of
        # engine cores and let them do the work.
        self.vllm_config = vllm_config
        self._prefill_engines = prefill_engines
        self._decode_engines = decode_engines
        self.input_queue = input_queue
        self.output_queue = output_queue

        # Keep track of active requests.
        self._requests: dict[str, Request] = {}

        self._transfer_backlogs = [
            queue.Queue(4) for _ in range(len(self._prefill_engines))
        ]
        self._decode_backlogs = {
            idx: queue.Queue(self.vllm_config.scheduler_config.max_num_seqs)
            for idx in range(len(self._decode_engines))
        }

        self._prefill_threads = [
            JetThread(
                target=functools.partial(self._prefill, idx),
                name=f"prefill-{idx}",
                daemon=True,
            ) for idx in range(len(self._prefill_engines))
        ]
        self._transfer_threads = [
            JetThread(
                target=functools.partial(
                    self._transfer,
                    idx,
                ),
                name=f"transfer-{idx}",
                daemon=True,
            ) for idx in range(len(self._prefill_engines))
        ]
        self._decode_threads = [
            JetThread(
                target=functools.partial(
                    self._decode,
                    idx,
                ),
                name=f"decode-{idx}",
                daemon=True,
            ) for idx in range(len(self._decode_engines))
        ]
        self._all_threads = list(
            itertools.chain(
                self._prefill_threads,
                self._transfer_threads,
                self._decode_threads,
            ))
        self.live = True
        # Start all threads
        for t in self._all_threads:
            t.start()

        # We should be taking the input from the client, the code below is forked from
        # vllm.v1.engine.core.EngineCoreProc.
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
                handshake_address,
                identity,
                local_client,
                vllm_config,
                client_handshake_address=None) as addresses:
        with self._perform_handshakes(
                handshake_address,
                identity,
                local_client,
                vllm_config,
                client_handshake_address=None) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address)
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        ready_event = threading.Event()
        input_thread = threading.Thread(target=self.process_input_sockets,
                                        args=(addresses.inputs,
                                              addresses.coordinator_input,
                                              identity, ready_event),
                                        daemon=True)
        input_thread.start()

        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  self.engine_index),
            daemon=True)
        self.output_thread.start()

        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  self.engine_index),
            daemon=True)
        self.output_thread.start()

        # Don't complete handshake until DP coordinator ready message is
        # received.
        while not ready_event.wait(timeout=10):
            if not input_thread.is_alive():
                raise RuntimeError("Input socket thread died during startup")
            if addresses.coordinator_input is not None:
                logger.info("Waiting for READY message from DP Coordinator...")

    @staticmethod
    def _create_engine_cores(
        slice_sizes: tuple[int, ...],
        vllm_config: VllmConfig,
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
    ) -> list[vLLMEngineCore]:
        engine_cores = []
        for _ in slice_sizes:
            engine_core = vLLMEngineCore(
                vllm_config,
                disagg_executor.DisaggExecutor,
                log_stats,
                executor_fail_callback,
            )

            engine_cores.append(engine_core)
            logger.info("Disaggregated engine core created.")

        return engine_cores

    def _add_request(self, request: EngineCoreRequest) -> Request:
        if request.mm_hashes is not None:
            assert request.mm_inputs is not None
            request.mm_inputs = self._prefill_engines[
                0].mm_input_cache_server.get_and_update_p1(
                    request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)

        if req.use_structured_output:
            self._prefill_engines[0].structured_output_manager.grammar_init(
                req)

        return req

    def add_request(self, request: EngineCoreRequest):
        vllm_request = self._add_request(request)
        self._prefill_engines[0].scheduler.add_request(vllm_request)
        self._requests[request.request_id] = vllm_request

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            pass
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self._prefill_engines[0], method_name)
                output.result = method(
                    *self._convert_msgspec_args(method, args))
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (f"Call to {method_name} method"
                                          f" failed: {str(e)}")
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output)))
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error("Unrecognized input request type encountered: %s",
                         request_type)

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while True:
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)
            time.sleep(0.01)

    def _prefill(self, idx: int):
        prefill_engine = self._prefill_engines[idx]
        transfer_backlog = self._transfer_backlogs[idx]

        while self.live:
            if not prefill_engine.scheduler.has_requests():
                time.sleep(0.05)
                continue

            scheduler_output = prefill_engine.scheduler.schedule()
            with LatencyTracker(f"prefill-{idx}"):
                model_output = prefill_engine.execute_model_with_error_logging(
                    prefill_engine.model_executor.execute_model,
                    scheduler_output)

            if scheduler_output.total_num_scheduled_tokens > 0:
                kv_cache_map: dict[str, list[jax.Array]] = {}
                for req_id, idx_in_output in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx_in_output]) > 0:
                        request = self._requests[req_id]
                        block_ids = (prefill_engine.scheduler.kv_cache_manager.
                                     get_block_ids(req_id))
                        with LatencyTracker(
                                f"ExtractKVCache-{req_id}-{block_ids}"):
                            kv_cache_map[req_id] = (
                                prefill_engine.model_executor.driver_worker.
                                model_runner.get_kv_cache_for_block_ids(
                                    block_ids[0]))
                transfer_backlog.put(kv_cache_map, block=True)

                engine_core_outputs = prefill_engine.scheduler.update_from_output(
                    scheduler_output, model_output)

                for req_id, idx_in_output in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx_in_output]) > 0:
                        request = prefill_engine.scheduler.requests[req_id]
                        prefill_engine.scheduler.running.remove(request)
                        prefill_engine.scheduler.encoder_cache_manager.free(
                            request)
                        prefill_engine.scheduler.kv_cache_manager.free(request)
                        prefill_engine.scheduler.kv_cache_manager.free_block_hashes(
                            request)
                        prefill_engine.scheduler.requests.pop(req_id)

                for output in (engine_core_outputs.items()
                               if engine_core_outputs else ()):
                    self.output_queue.put_nowait(output)

    def _transfer(self, idx: int):
        transfer_backlog = self._transfer_backlogs[idx]
        while self.live:
            kv_cachce_map = transfer_backlog.get(block=True)
            if kv_cachce_map is None:
                break

            push_targets = []
            for req_id, kv_cache in kv_cachce_map.items():
                target_idx = -1
                cnt = 9999999
                for i, e in enumerate(self._decode_engines):
                    req_cnt = sum(e.scheduler.get_request_counts())
                    if req_cnt < cnt:
                        cnt = req_cnt
                        target_idx = i

                with LatencyTracker("KVCacheTransfer"):
                    kv_cache = self._decode_engines[
                        target_idx].model_executor.driver_worker.model_runner.transfer_kv_cache(
                            kv_cache)

                prefill_output = {
                    "cache": kv_cache,
                    "req_id": req_id,
                }
                push_targets.append((target_idx, prefill_output))

            for target_idx, prefill_output in push_targets:
                self._decode_backlogs[target_idx].put(prefill_output,
                                                      block=True)

    def _decode(self, idx: int):
        decode_engine = self._decode_engines[idx]
        decode_backlog = self._decode_backlogs[idx]

        while self.live:
            block = not decode_engine.scheduler.has_requests()
            while True:
                if (sum(decode_engine.scheduler.get_request_counts())
                        >= self.vllm_config.scheduler_config.max_num_seqs
                        or decode_engine.model_executor.driver_worker.
                        model_runner.input_batch.num_reqs
                        >= self.vllm_config.scheduler_config.max_num_seqs):
                    break

                try:
                    prefill_output = decode_backlog.get(block=block,
                                                        timeout=1.0)
                except queue.Empty:
                    if block:
                        continue
                    break

                if prefill_output is None:
                    break

                block = False

                req_id = prefill_output["req_id"]
                vllm_request = self._requests[req_id]
                # Caching num_computed_tokens. The tokens in kv manager allocate blocks
                # Caching num_computed_tokens. The tokens in kv manager allocate blocks
                # is computed as num_computed_tokens + num_new_tokens, so without caching
                # the token number would double.
                # the token number would double.
                prompt_tokens = vllm_request.num_computed_tokens
                vllm_request.num_computed_tokens = 0
                kv_cache = prefill_output["cache"]

                kv_cache_manager = decode_engine.scheduler.kv_cache_manager
                kv_cache_manager.allocate_slots(
                    vllm_request,
                    prompt_tokens,
                )
                vllm_request.num_computed_tokens = prompt_tokens
                new_block_ids = kv_cache_manager.get_block_ids(req_id)
                assert (len(new_block_ids[0]) == math.ceil(
                    prompt_tokens / self.vllm_config.cache_config.block_size))
                assert (len(new_block_ids[0]) == math.ceil(
                    prompt_tokens / self.vllm_config.cache_config.block_size))

                with LatencyTracker(f"KVCacheInsert-{len(new_block_ids[0])}"):
                    decode_engine.model_executor.driver_worker.model_runner.insert_request_with_kv_cache(
                        vllm_request, kv_cache, new_block_ids)

                vllm_request.status = RequestStatus.RUNNING
                decode_engine.scheduler.running.append(vllm_request)
                decode_engine.scheduler.requests[req_id] = vllm_request

                self._requests.pop(req_id)

            scheduler_output = decode_engine.scheduler.schedule()

            logger.info(f'''decode-{idx}: scheduler_output -
                {scheduler_output.scheduled_cached_reqs.num_computed_tokens},
                new block ids - {scheduler_output.scheduled_cached_reqs.new_block_ids}'''
                        )

            with LatencyTracker(f"decode-{idx}"):
                model_output = decode_engine.execute_model_with_error_logging(
                    decode_engine.model_executor.execute_model,
                    scheduler_output
                )
            if scheduler_output.total_num_scheduled_tokens > 0:
                engine_core_outputs = decode_engine.scheduler.update_from_output(
                    scheduler_output, model_output)
                for output in (engine_core_outputs.items()
                               if engine_core_outputs else ()):
                    self.output_queue.put_nowait(output)

    def shutdown(self):
        self.live = False
        for e in self._prefill_engines:
            e.shutdown()
        for e in self._decode_engines:
            e.shutdown()


class DisaggEngineCoreProc(vLLMEngineCoreProc, IDisaggEngineCoreProc):
    """Legacy Adapter for the Disaggregated Engine.

    This class maintains the old constructor for backward compatibility with vLLM.
    Internally, it instantiates and delegates to the new _DisaggOrchestrator class.
    This allows tpu_commons to be deployed safely before vllm is updated.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        log_stats: bool,
        engine_index: int = 0,
        **kwargs,
    ):
        # This adapter does not call super().__init__ because it is replacing
        # the parent's functionality entirely.

        # The adapter is responsible for setting up the ZMQ communication
        # that the internal orchestrator will use.
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()

        # Temporarily act as its own factory to create the engine cores.
        # This logic will eventually be "bubbled up" to a factory in vLLM.
        prefill_slice_sizes = disagg_utils.get_prefill_slices()
        decode_slice_sizes = disagg_utils.get_decode_slices()

        def executor_fail_callback():
            self.input_queue.put_nowait(
                (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        prefill_engines = self._create_engine_cores(
            prefill_slice_sizes,
            vllm_config,
            log_stats,
            executor_fail_callback,
        )
        decode_engines = self._create_engine_cores(
            decode_slice_sizes,
            vllm_config,
            log_stats,
            executor_fail_callback,
        )

        # Create and hold the real orchestrator
        self._orchestrator = _DisaggOrchestrator(
            vllm_config=vllm_config,
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )

        # The rest of the __init__ is the ZMQ/handshake logic lifted directly
        # from the original vLLMEngineCoreProc.
        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
                handshake_address,
                identity,
                local_client,
                vllm_config,
                client_handshake_address=None) as addresses:
            self.client_count = len(addresses.outputs)
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address)
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)

        ready_event = threading.Event()
        input_thread = threading.Thread(target=self.process_input_sockets,
                                        args=(addresses.inputs,
                                              addresses.coordinator_input,
                                              identity, ready_event),
                                        daemon=True)
        input_thread.start()

        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  self.engine_index),
            daemon=True)
        self.output_thread.start()

        while not ready_event.wait(timeout=10):
            if not input_thread.is_alive():
                raise RuntimeError("Input socket thread died during startup")
            if addresses.coordinator_input is not None:
                logger.info("Waiting for READY message from DP Coordinator...")

    @staticmethod
    def _create_engine_cores(
        slice_sizes: tuple[int, ...],
        vllm_config: VllmConfig,
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
    ) -> list[vLLMEngineCore]:
        engine_cores = []
        for _ in slice_sizes:
            engine_core = vLLMEngineCore(
                vllm_config,
                disagg_executor.DisaggExecutor,
                log_stats,
                executor_fail_callback,
            )
            engine_cores.append(engine_core)
        return engine_cores

    def run_busy_loop(self):
        """Delegates the call to the internal orchestrator."""
        self._orchestrator.run_busy_loop()

    def shutdown(self):
        """Delegates the call to the internal orchestrator."""
        self._orchestrator.shutdown()