# SPDX-License-Identifier: Apache-2.0
import functools
import itertools
import math
import os
import queue
import signal
import threading
import time
import traceback
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

import jax
# ======================================================================================
# Imports for DisaggEngineCoreProc (the vLLM adapter)
# ======================================================================================
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput,
                            UtilityResult)
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.engine.core import EngineCoreProc as vLLMEngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.request import Request, RequestStatus

from tpu_inference import utils as common_utils
from tpu_inference.core import disagg_executor, disagg_utils
from tpu_inference.runner.tpu_runner import AsyncTPUModelRunnerOutput
# ======================================================================================
# Imports for _DisaggOrchestrator (decoupled from vLLM)
# ======================================================================================
from tpu_inference.runner.utils import LatencyTracker

# This file contains two classes:
# 1. _DisaggOrchestrator: The clean, decoupled core orchestration logic.
# 2. DisaggEngineCoreProc: The vLLM-facing adapter that handles process management.

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc

# ======================================================================================
# Class 1: The Decoupled Orchestrator
# ======================================================================================


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
    """Contains the core orchestration logic, decoupled from vLLM."""

    def __init__(
        self,
        config: VllmConfig,
        output_queue: queue.Queue,
        prefill_engines: list[vLLMEngineCore],
        decode_engines: list[vLLMEngineCore],
        prefill_slice_sizes: tuple[int, ...],
        decode_slice_sizes: tuple[int, ...],
    ):
        self._config = config
        self._output_queue = output_queue
        self._prefill_engines = prefill_engines
        self._decode_engines = decode_engines

        # Keep track of active requests.
        self._requests: dict[str, Request] = {}

        # Hack device config to pass in the subslice of TPUs.
        slice_sizes = list(prefill_slice_sizes)
        slice_sizes.extend(decode_slice_sizes)

        self._transfer_backlogs = [
            queue.Queue(4) for i in range(len(self._prefill_engines))
        ]

        self._decode_backlogs = {}
        for idx, engine in enumerate(self._decode_engines):
            # Determine the decode backlog len by remaning hbm dividing max kv cache size of a single request
            runner = engine.model_executor.driver_worker.model_runner
            hbm_usage = common_utils.hbm_usage_bytes(
                engine.model_executor.driver_worker.devices)
            if not hbm_usage:
                self._decode_backlogs[idx] = queue.Queue(
                    self._config.scheduler_config.max_num_seqs)
                continue
            hbm_free = [limit - used for used, limit in hbm_usage]
            max_kv_bytes = len(runner.kv_caches) * (
                runner.max_model_len // runner.cache_config.block_size) * (
                    runner.kv_caches[0][0].nbytes) // len(hbm_free)
            max_queue_len = min(hbm_free[0] // max_kv_bytes,
                                self._config.scheduler_config.max_num_seqs)
            logger.debug(
                f"max kv bytes: {max_kv_bytes}, max_queue_len {max_queue_len}")
            self._decode_backlogs[idx] = queue.Queue(max_queue_len)

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

    def add_request(self, request: Request):
        """
        Adds a new request to the orchestrator.

        This is the main entry point for new work. It stores the request for
        internal state tracking and hands it off to the first stage of the
        processing pipeline (the prefill scheduler).
        """
        # Hand off the request to the prefill scheduler to be batched for execution.
        self._prefill_engines[0].scheduler.add_request(request)

        # Add to internal state for tracking by other threads.
        # The key is the request_id, the value is the request object.
        self._requests[request.request_id] = request

    def _prefill(self, idx: int):
        prefill_engine = self._prefill_engines[idx]
        transfer_backlog = self._transfer_backlogs[idx]

        while self.live:
            if not prefill_engine.scheduler.has_requests():
                time.sleep(0.05)
                continue

            scheduler_output = prefill_engine.scheduler.schedule()
            with LatencyTracker(f"prefill-{idx}"):
                future = prefill_engine.model_executor.execute_model(
                    scheduler_output, non_block=True)
                grammar_output = prefill_engine.scheduler.get_grammar_bitmask(
                    scheduler_output)
                with prefill_engine.log_error_detail(scheduler_output):
                    model_output = future.result()
                    if model_output is None:
                        model_output = prefill_engine.model_executor.sample_tokens(
                            grammar_output)
                    if isinstance(model_output, AsyncTPUModelRunnerOutput):
                        model_output = model_output.get_output()

            if scheduler_output.total_num_scheduled_tokens > 0:
                logger.debug(f"Prefill result: {model_output}")

                kv_cache_map: dict[str, Tuple(list[jax.Array], list[Any])] = {}
                for req_id, idx in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx]) > 0:
                        request = self._requests[req_id]
                        block_ids = (prefill_engine.scheduler.kv_cache_manager.
                                     get_block_ids(req_id))
                        # Assume one KV cache group for now.
                        kv_cache_map[req_id] = (
                            prefill_engine.model_executor.driver_worker.
                            model_runner.get_kv_cache_for_block_ids(
                                block_ids[0]), request.block_hashes)
                        logger.debug(f"prefill done: for {req_id}")
                transfer_backlog.put(kv_cache_map, block=True)

                # tweak model_output to let the scheduler know kv_transfer is done for requests, so they can be freed.
                engine_core_outputs = prefill_engine.scheduler.update_from_output(
                    scheduler_output, model_output)  # type: ignore

                for req_id, idx in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx]) > 0:
                        request = self._requests[req_id]
                        logger.debug(
                            f"request block_hashes at prefill: {request.block_hashes}"
                        )
                        logger.debug(
                            f"request-{req_id}: tokens={request.all_token_ids} after prefill"
                        )
                        # Remove request from the prefill engine.
                        if req_id in prefill_engine.scheduler.requests:
                            request = prefill_engine.scheduler.requests[req_id]
                            prefill_engine.scheduler.running.remove(request)
                            prefill_engine.scheduler.encoder_cache_manager.free(
                                request)

                            prefill_engine.scheduler.kv_cache_manager.free(
                                request)

                            prefill_engine.scheduler.requests.pop(req_id)

                for output in (engine_core_outputs.items()
                               if engine_core_outputs else ()):
                    self._output_queue.put_nowait(output)

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
            for req_id, (kv_cache, block_hashes) in kv_cachce_map.items():
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
                    "block_hashes": block_hashes,
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
                        >= self._config.scheduler_config.max_num_seqs
                        or decode_engine.model_executor.driver_worker.
                        model_runner.input_batch.num_reqs
                        >= self._config.scheduler_config.max_num_seqs):
                    break

                try:
                    prefill_output = decode_backlog.get(block=block,
                                                        timeout=1.0)
                except queue.Empty:
                    if block:
                        continue
                    break

                if prefill_output is None:
                    logger.debug(
                        f"decode-{idx} Empty output, and we are idle, exiting..."
                    )
                    break

                # We got a request, set block to False
                block = False

                # Insert the request to the decoder.
                req_id = prefill_output["req_id"]
                vllm_request = self._requests[req_id]
                # Caching num_computed_tokens. The tokens in kv manager allocate blocks
                # is computed as num_computed_tokens + num_new_tokens, so without caching
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
                logger.debug(
                    f"inserting {req_id} new_block_ids {new_block_ids}")
                if len(new_block_ids[0]) != math.ceil(
                        prompt_tokens / self._config.cache_config.block_size):
                    logger.warning("Running out of blocks in decode engine! ")
                    break

                decode_engine.model_executor.driver_worker.model_runner.insert_request_with_kv_cache(
                    vllm_request, kv_cache, new_block_ids)

                vllm_request.status = RequestStatus.RUNNING
                block_hashes = prefill_output["block_hashes"]
                vllm_request.block_hashes = block_hashes
                decode_engine.scheduler.running.append(vllm_request)
                decode_engine.scheduler.requests[req_id] = vllm_request

                self._requests.pop(req_id)

            scheduler_output = decode_engine.scheduler.schedule()

            logger.debug(f'''decode-{idx}: scheduler_output -
                {scheduler_output.scheduled_cached_reqs.num_computed_tokens},
                new block ids - {scheduler_output.scheduled_cached_reqs.new_block_ids}'''
                         )

            with LatencyTracker(f"decode-{idx}"):
                future = decode_engine.model_executor.execute_model(
                    scheduler_output, non_block=True)
                grammar_output = decode_engine.scheduler.get_grammar_bitmask(
                    scheduler_output)
                with decode_engine.log_error_detail(scheduler_output):
                    model_output = future.result()
                    if model_output is None:
                        model_output = decode_engine.model_executor.sample_tokens(
                            grammar_output)
                    if isinstance(model_output, AsyncTPUModelRunnerOutput):
                        model_output = model_output.get_output()

            if scheduler_output.total_num_scheduled_tokens > 0:
                logger.debug(f"Decode result: {model_output}")

                engine_core_outputs = decode_engine.scheduler.update_from_output(
                    scheduler_output, model_output)  # type: ignore
                for output in (engine_core_outputs.items()
                               if engine_core_outputs else ()):
                    self._output_queue.put_nowait(output)

    def shutdown(self):
        for e in self._prefill_engines:
            e.shutdown()
        for e in self._decode_engines:
            e.shutdown()


# ======================================================================================
# Class 2: The vLLM-Facing Adapter
# ======================================================================================


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
        logger.warning("Disaggregated engine core created.")

    return engine_cores


def _get_slice_sizes(devices):
    prefill_slice_sizes = disagg_utils.get_prefill_slices()
    decode_slice_sizes = disagg_utils.get_decode_slices()
    if isinstance(prefill_slice_sizes[0], int):
        prefill_chip_cnt = sum(prefill_slice_sizes)
    else:
        prefill_chip_cnt = sum([math.prod(t) for t in prefill_slice_sizes])
    if isinstance(decode_slice_sizes[0], int):
        decode_chip_cnt = sum(decode_slice_sizes)
    else:
        decode_chip_cnt = sum([math.prod(t) for t in decode_slice_sizes])
    assert decode_chip_cnt + prefill_chip_cnt <= len(devices)
    assert prefill_chip_cnt > 0 and decode_chip_cnt > 0

    slice_sizes = list(prefill_slice_sizes)
    slice_sizes.extend(decode_slice_sizes)
    return prefill_slice_sizes, decode_slice_sizes, slice_sizes


class DisaggEngineCore(vLLMEngineCore):
    """The vLLM-facing adapter that handles process management and I/O. Modifes vLLMEngineCore and is only used in in-process EngineCore client."""

    @staticmethod
    def is_supported() -> bool:
        """
        Returns True if this engine can run in the current environment.
        """
        return disagg_utils.is_disagg_enabled()

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Optional[Callable] = None,
    ):
        self.vllm_config = vllm_config

        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()

        self.devices = jax.devices()
        device_kind = self.devices[0].device_kind
        if device_kind != 'TPU7x':
            self.vllm_config.cache_config.gpu_memory_utilization = (
                self.vllm_config.cache_config.gpu_memory_utilization - 0.1)
        prefill_slice_sizes, decode_slice_sizes, slice_sizes = _get_slice_sizes(
            self.devices)

        if isinstance(slice_sizes[0], int):
            setattr(vllm_config.device_config, "slice",
                    (0, slice_sizes, self.devices))
        else:
            setattr(vllm_config.device_config, "slice",
                    ((0, 0), 0, slice_sizes, self.devices))
        logger.info(
            f"Creating DisaggEngineCore with slice_sizes {slice_sizes}...")

        self._prefill_engines = _create_engine_cores(
            prefill_slice_sizes,
            vllm_config,
            log_stats,
            executor_fail_callback,
        )
        logger.info(
            f"{len(self._prefill_engines)} Disaggregated prefill engines created."
        )

        self._decode_engines = _create_engine_cores(
            decode_slice_sizes,
            vllm_config,
            log_stats,
            executor_fail_callback,
        )
        logger.info(
            f"{len(self._decode_engines)} Disaggregated decode engines created."
        )

        self.batch_queue = None

        self.request_block_hasher = None
        if (self.vllm_config.cache_config.enable_prefix_caching
                or self._prefill_engines[0].scheduler.get_kv_connector()
                is not None):

            block_size = vllm_config.cache_config.block_size
            caching_hash_fn = common_utils.get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                block_size, caching_hash_fn)

        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)

        self.mm_receiver_cache = None
        self._orchestrator = _DisaggOrchestrator(
            config=vllm_config,
            output_queue=self.output_queue,
            prefill_engines=self._prefill_engines,
            decode_engines=self._decode_engines,
            prefill_slice_sizes=prefill_slice_sizes,
            decode_slice_sizes=decode_slice_sizes,
        )
        # for vllm compatibility
        self.model_executor = self._prefill_engines[0].model_executor

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self._prefill_engines[0].model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}")

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks()
                if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(f"Unsupported task: {pooling_params.task!r} "
                                 f"Supported tasks: {supported_pooling_tasks}")

        if request.kv_transfer_params is not None and (
                not self.scheduler.get_kv_connector()):
            logger.warning("Got kv_transfer_params, but no KVConnector found. "
                           "Disabling KVTransfer for this request.")

        self._orchestrator.add_request(request)

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        client_idx, output = self.output_queue.get()
        # logger.warning(f"step output: {output}")
        time.sleep(0.03)
        return {client_idx: output}, True

    def shutdown(self):
        self._orchestrator.shutdown()

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 processor, P0 mirror, P1 mirror)
        for engine in itertools.chain(self._prefill_engines,
                                      self._decode_engines):
            if engine.scheduler.has_unfinished_requests():
                logger.warning(
                    "Resetting the multi-modal cache when requests are "
                    "in progress may lead to desynced internal caches.")

            if engine.mm_receiver_cache is not None:
                engine.mm_receiver_cache.clear_cache()

    def reset_prefix_cache(self):
        for engine in itertools.chain(self._prefill_engines,
                                      self._decode_engines):
            engine.scheduler.reset_prefix_cache()


class DisaggEngineCoreProc(vLLMEngineCoreProc):
    """The vLLM-facing adapter that handles process management and I/O."""

    @staticmethod
    def is_supported() -> bool:
        """
        Returns True if this engine can run in the current environment.
        """
        return disagg_utils.is_disagg_enabled()

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        engine_index: int = 0,
        **kwargs,
    ):
        if 'dp_rank' in kwargs or 'local_dp_rank' in kwargs:
            logger.debug(
                "Ignoring data parallelism arguments for non-DP disaggregated engine."
            )
        # We don't invoke super class's ctor as we are not really the
        # engine core to be executed, instead we create other instance of
        # engine cores and let them do the work.
        self.vllm_config = vllm_config

        # We should be taking the input from the client, the code below is forked from
        # vllm.v1.engine.core.EngineCoreProc.
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        self.devices = jax.devices()
        device_kind = self.devices[0].device_kind
        if device_kind != 'TPU7x':
            self.vllm_config.cache_config.gpu_memory_utilization = (
                self.vllm_config.cache_config.gpu_memory_utilization - 0.1)
        prefill_slice_sizes, decode_slice_sizes, slice_sizes = _get_slice_sizes(
            self.devices)

        if isinstance(slice_sizes[0], int):
            setattr(vllm_config.device_config, "slice",
                    (0, slice_sizes, self.devices))
        else:
            setattr(vllm_config.device_config, "slice",
                    ((0, 0), 0, slice_sizes, self.devices))
        logger.info(
            f"Creating DisaggEngineCoreProc with slice_sizes {slice_sizes}...")

        def executor_fail_callback():
            self.input_queue.put_nowait(
                (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        # Don't complete handshake until DP coordinator ready message is
        # received.
        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
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

            self._prefill_engines = _create_engine_cores(
                prefill_slice_sizes,
                vllm_config,
                log_stats,
                executor_fail_callback,
            )
            logger.info(
                f"{len(self._prefill_engines)} Disaggregated prefill engines created."
            )

            self._decode_engines = _create_engine_cores(
                decode_slice_sizes,
                vllm_config,
                log_stats,
                executor_fail_callback,
            )
            logger.info(
                f"{len(self._decode_engines)} Disaggregated decode engines created."
            )

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
                    raise RuntimeError(
                        "Input socket thread died during startup")
                if addresses.coordinator_input is not None:
                    logger.info(
                        "Waiting for READY message from DP Coordinator...")
        self.request_block_hasher = None
        if (self.vllm_config.cache_config.enable_prefix_caching
                or self._prefill_engines[0].scheduler.get_kv_connector()
                is not None):

            block_size = vllm_config.cache_config.block_size
            caching_hash_fn = common_utils.get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                block_size, caching_hash_fn)

        self.mm_receiver_cache = None
        self._orchestrator = _DisaggOrchestrator(
            config=vllm_config,
            output_queue=self.output_queue,
            prefill_engines=self._prefill_engines,
            decode_engines=self._decode_engines,
            prefill_slice_sizes=prefill_slice_sizes,
            decode_slice_sizes=decode_slice_sizes,
        )

    def add_request(self, request: EngineCoreRequest, request_wave: int = 0):
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}")

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks()
                if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(f"Unsupported task: {pooling_params.task!r} "
                                 f"Supported tasks: {supported_pooling_tasks}")

        self._orchestrator.add_request(request)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""
        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req)
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
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
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

    def shutdown(self):
        self._orchestrator.shutdown()
