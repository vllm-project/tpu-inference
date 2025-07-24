# SPDX-License-Identifier: Apache-2.0
import functools
import itertools
import os
import queue
import threading
import time
import traceback
import signal
from typing import Any, Callable, Optional, TypeVar, Union

import jax
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.engine.core import EngineCoreProc as vLLMEngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus

from tpu_commons.core import disagg_executor, disagg_utils
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


class DisaggEngineCoreProc(vLLMEngineCoreProc):
    """Wrapper for running vLLM EngineCore in background process."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        # We don't invoke super class's ctor as we are not really the
        # engine core to be executed, instead we create other instance of
        # engine cores and let them do the work.
        self.vllm_config = vllm_config

        devices = jax.devices()
        prefill_slice_sizes = disagg_utils.get_prefill_slices()
        decode_slice_sizes = disagg_utils.get_decode_slices()
        prefill_chip_cnt = sum(prefill_slice_sizes)
        decode_chip_cnt = sum(decode_slice_sizes)
        assert decode_chip_cnt + prefill_chip_cnt <= len(devices)
        assert prefill_chip_cnt > 0 and decode_chip_cnt > 0

        # Keep track of active requests.
        self._requests: dict[str, Request] = {}

        # Hack device config to pass in the subslice of TPUs.
        slice_sizes = list(prefill_slice_sizes)
        slice_sizes.extend(decode_slice_sizes)
        setattr(vllm_config.device_config, "slice", (0, slice_sizes))
        logger.info(f"Adding slice config to device config: {slice_sizes}")

        def executor_fail_callback():
            self.input_queue.put_nowait(
                (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        self._prefill_engines = self._create_engine_cores(
            prefill_slice_sizes,
            vllm_config,
            log_stats,
            executor_fail_callback,
        )
        logger.info(
            f"{len(self._prefill_engines)} Disaggregated prefill engines created."
        )

        self._transfer_backlogs = [
            queue.Queue(4) for i in range(len(self._prefill_engines))
        ]

        self._decode_engines = self._create_engine_cores(
            decode_slice_sizes,
            vllm_config,
            log_stats,
            executor_fail_callback,
        )
        logger.info(
            f"{len(self._decode_engines)} Disaggregated decode engines created."
        )
        self._decode_backlogs = {
            idx: queue.Queue(vllm_config.scheduler_config.max_num_seqs)
            for idx, engine in enumerate(self._decode_engines)
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

        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
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
        threading.Thread(target=self.process_input_sockets,
                         args=(addresses.inputs, addresses.coordinator_input,
                               identity),
                         daemon=True).start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  self.engine_index),
            daemon=True)
        self.output_thread.start()

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
            # Here, if hash exists for a multimodal input, then it will be
            # fetched from the cache, else it will be added to the cache.
            # Note that the cache here is mirrored with the client cache, so
            # anything that has a hash must have a HIT cache entry here
            # as well.
            assert request.mm_inputs is not None
            request.mm_inputs = self._prefill_engines[
                0].mm_input_cache_server.get_and_update_p1(
                    request.mm_inputs, request.mm_hashes)

        req = Request.from_engine_core_request(request)

        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self._prefill_engines[0].structured_output_manager.grammar_init(
                req)

        return req

    def add_request(self, request: EngineCoreRequest):
        vllm_request = self._add_request(request)

        # TODO(fhzhang): support multiple prefill engines.
        self._prefill_engines[0].scheduler.add_request(vllm_request)
        self._requests[request.request_id] = vllm_request

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            # TODO(fhzhang): we need to keep track of which engine is processing
            # the request and finish it there.
            # owner_engine.scheduler.finish_requests(request, RequestStatus.FINISHED_ABORTED)
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

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)
            # Yield control to other threads, as we are not doing any real work.
            # Without this sleep, we'd be hogging all the cpu cycles with our run_busy_loop.
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
                model_output = prefill_engine.execute_model(scheduler_output)

            if scheduler_output.total_num_scheduled_tokens > 0:
                logger.debug(f"Prefill result: {model_output}")

                kv_cache_map: dict[str, list[jax.Array]] = {}
                for req_id, idx in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx]) > 0:
                        request = self._requests[req_id]
                        block_ids = (prefill_engine.scheduler.kv_cache_manager.
                                     get_block_ids(req_id))
                        with LatencyTracker(
                                f"ExtractKVCache-{req_id}-{block_ids}"):
                            # Assume one KV cache group for now.
                            kv_cache_map[req_id] = (
                                prefill_engine.model_executor.driver_worker.
                                model_runner.get_kv_cache_for_block_ids(
                                    block_ids[0]))
                        logger.debug(f"prefill done: for {req_id}")
                transfer_backlog.put(kv_cache_map, block=True)

                # tweak model_output to let the scheduler know kv_transfer is done for requests, so they can be freed.
                engine_core_outputs = prefill_engine.scheduler.update_from_output(
                    scheduler_output, model_output)  # type: ignore

                for req_id, idx in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx]) > 0:
                        request = self._requests[req_id]
                        logger.debug(
                            f"request-{req_id}: tokens={request._all_token_ids} after prefill"
                        )
                        # Remove request from the prefill engine.

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
        """Transfers the kv cache on an active request to the least full
    decode backlog."""
        transfer_backlog = self._transfer_backlogs[idx]
        while self.live:
            # The transfer thread can just sleep until it has work to do.
            kv_cachce_map = transfer_backlog.get(block=True)
            if kv_cachce_map is None:
                break

            logger.debug(
                f"transfer-{idx}: KV Cache items received: {kv_cachce_map.keys()}"
            )

            push_targets = []
            for req_id, kv_cache in kv_cachce_map.items():
                target_idx = -1
                cnt = 9999999
                for i, e in enumerate(self._decode_engines):
                    req_cnt = sum(e.scheduler.get_request_counts())
                    if req_cnt < cnt:
                        cnt = req_cnt
                        target_idx = i

                # Only transfer the KVCache for the disaggregated serving.
                with LatencyTracker("KVCacheTransfer"):
                    kv_cache = self._decode_engines[
                        target_idx].model_executor.driver_worker.model_runner.transfer_kv_cache(
                            kv_cache)

                # TODO(fhzhang): Now how do we get the kv cache to the decode engine?
                prefill_output = {
                    "cache": kv_cache,
                    "req_id": req_id,
                }
                push_targets.append((target_idx, prefill_output))

            for target_idx, prefill_output in push_targets:
                self._decode_backlogs[target_idx].put(prefill_output,
                                                      block=True)
                logger.debug(
                    "Successfully transferred prefill request %s "
                    "from prefill engine %d to decode engine %d. decode backlog len %d",
                    prefill_output["req_id"],
                    idx,
                    target_idx,
                    self._decode_backlogs[target_idx].qsize(),
                )

    def _decode(self, idx: int):
        decode_engine = self._decode_engines[idx]
        decode_backlog = self._decode_backlogs[idx]

        while self.live:
            block = not decode_engine.scheduler.has_requests()
            while True:
                # We need to check input batch as well as the request completion is delayed
                # from scheduler to the runner.
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
                    logger.info(
                        f"decode-{idx} Empty output, and we are idle, exiting..."
                    )
                    break

                # We got a request, set block to False
                block = False

                # Insert the request to the decoder.
                req_id = prefill_output["req_id"]
                vllm_request = self._requests[req_id]
                kv_cache = prefill_output["cache"]

                kv_cache_manager = decode_engine.scheduler.kv_cache_manager
                kv_cache_manager.allocate_slots(
                    vllm_request,
                    vllm_request.num_computed_tokens,
                )
                new_block_ids = kv_cache_manager.get_block_ids(req_id)

                with LatencyTracker(f"KVCacheInsert-{len(new_block_ids[0])}"):
                    decode_engine.model_executor.driver_worker.model_runner.insert_request_with_kv_cache(
                        vllm_request, kv_cache, new_block_ids)

                vllm_request.status = RequestStatus.RUNNING
                decode_engine.scheduler.running.append(vllm_request)
                decode_engine.scheduler.requests[req_id] = vllm_request

                self._requests.pop(req_id)

            scheduler_output = decode_engine.scheduler.schedule()

            logger.debug(
                f"decode-{idx}: scheduler_output - {scheduler_output}")

            with LatencyTracker(f"decode-{idx}"):
                model_output = decode_engine.execute_model(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens > 0:
                logger.debug(f"Decode result: {model_output}")

                engine_core_outputs = decode_engine.scheduler.update_from_output(
                    scheduler_output, model_output)  # type: ignore
                for output in (engine_core_outputs.items()
                               if engine_core_outputs else ()):
                    self.output_queue.put_nowait(output)

    def shutdown(self):
        for e in self._prefill_engines:
            e.shutdown()
        for e in self._decode_engines:
            e.shutdown()
