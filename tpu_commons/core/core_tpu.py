# SPDX-License-Identifier: Apache-2.0
import copy
import functools
import itertools
import queue
import signal
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack
from inspect import isclass, signature
from typing import Any, Callable, Optional, TypeVar, Union

import jax
import msgspec
import zmq
from tpu_commons.runner.utils import LatencyTracker
from vllm.config import ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.transformers_utils.config import \
    maybe_register_config_serialize_by_value
from vllm.utils import make_zmq_socket
from vllm.v1.core.kv_cache_utils import (get_kv_cache_config,
                                         unify_kv_cache_configs)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType, UtilityOutput)
from vllm.v1.engine.mm_input_cache import MirroredProcessingCache
from vllm.v1.engine.core import EngineCore as vLLMEngineCore, EngineCoreProc as vLLMEngineCoreProc
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import EngineHandshakeMetadata, EngineZmqAddresses
from vllm.sampling_params import SamplingParams
from vllm.version import __version__ as VLLM_VERSION

from tpu_commons.core import disagg_executor, disagg_utils, orchestrator
from tpu_commons.core.tpu_jax_engine import JaxEngine

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar('_R')  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: type[Executor],
                 log_stats: bool,
                 executor_fail_callback: Optional[Callable] = None):
        assert vllm_config.model_config.runner_type != "pooling"

        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins
        load_general_plugins()

        self.vllm_config = vllm_config
        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats

        # TODO(fhzhang): create config to setup disagg executors.
        devices = jax.devices()
        prefill_slice_sizes = disagg_utils.get_prefill_slices()
        decode_slice_sizes = disagg_utils.get_decode_slices()
        prefill_chip_cnt = sum(prefill_slice_sizes)
        decode_chip_cnt = sum(decode_slice_sizes)
        assert decode_chip_cnt + prefill_chip_cnt <= len(devices)
        assert prefill_chip_cnt > 0 and decode_chip_cnt > 0

        self.prefill_executors, self.prefill_engines = self._create_and_initialize_executors(
            prefill_slice_sizes, devices[:prefill_chip_cnt], vllm_config)
        logger.info(
            f"{len(self.prefill_executors)} Disaggregated prefill executor created."
        )

        self.decode_executors, self.decode_engines = self._create_and_initialize_executors(
            decode_slice_sizes,
            devices[prefill_chip_cnt:prefill_chip_cnt + decode_chip_cnt],
            vllm_config)
        logger.info(
            f"{len(self.decode_executors)} Disaggregated decode executor created."
        )

        self.structured_output_manager = StructuredOutputManager(vllm_config)
        logger.info("structure output manager created.")

        # Setup scheduler.

        logger.info("kv cache manager created.")
        # Setup MM Input Mapper.
        self.mm_input_cache_server = MirroredProcessingCache(
            vllm_config.model_config)
        logger.info("mirrored processing cache created.")

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.prefill_executors[
            0].max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)
        logger.warning("set up disaggregated driver")
        self.orchestrator = self._setup_driver(vllm_config)
        logger.warning("starting disaggregated orchestrator")

    def _create_and_initialize_executors(
        self,
        slice_sizes: tuple[int, ...],
        devices: list[Any],
        vllm_config: VllmConfig,
    ) -> tuple[list[disagg_executor.DisaggExecutor], list[JaxEngine]]:
        executors = []
        engines = []
        device_offset = 0
        for i, slice_size in enumerate(slice_sizes):
            logger.info(f"Creating executor {i} with slice size: {slice_size}")
            subslice = devices[device_offset:device_offset + slice_size]
            executor = disagg_executor.DisaggExecutor(vllm_config)
            executor.init_with_devices(subslice)
            num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
                self._initialize_kv_caches(vllm_config, executor)
            # TODO(fhzhang): we only need to set this once.
            vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
            vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

            engine = JaxEngine(vllm_config, kv_cache_config, executor)
            engines.append(engine)
            executors.append(executor)
            device_offset += slice_size
            logger.info("Disaggregated executor created.")

        return executors, engines

    def _initialize_kv_caches(self, vllm_config: VllmConfig,
                              executor) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = executor.get_kv_cache_specs()

        # Profiles the peak memory usage of the model to determine how much
        # memory can be allocated for kv cache.
        available_gpu_memory = executor.determine_available_memory()

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Get the kv cache tensor size
        # We leave 50% memory for processing and incoming batches. This is due to the
        # donation issues we are seeing:
        #   /mlir.py:1184: UserWarning: Some donated buffers were not usable: bfloat16[4,2737,32,128]...
        # TODO(fhzhang): fix the donation issues!
        kv_cache_configs = [
            get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                                available_gpu_memory_one_worker)
            for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
            zip(kv_cache_specs, available_gpu_memory)
        ]

        # Since we use a shared centralized controller, we need the
        # `kv_cache_config` to be consistent across all workers to make sure
        # all the memory operators can be applied to all workers.
        unify_kv_cache_configs(kv_cache_configs)

        # All workers have the same kv_cache_config except layer names, so use
        # an arbitrary one to initialize the scheduler.
        assert all([
            cfg.num_blocks == kv_cache_configs[0].num_blocks
            for cfg in kv_cache_configs
        ])
        num_gpu_blocks = kv_cache_configs[0].num_blocks
        num_cpu_blocks = 0
        scheduler_kv_cache_config = kv_cache_configs[0]

        # Initialize kv cache and warmup the execution
        executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info(("init engine (profile, create kv cache, "
                     "warmup model) took %.2f seconds"), elapsed)
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def _setup_driver(self, vllm_config):
        driver = orchestrator.Driver(
            vllm_config=vllm_config,
            prefill_engines=self.prefill_engines,
            generate_engines=self.decode_engines,
        )
        return driver

    def add_request(self, request: EngineCoreRequest):
        """Add request to the scheduler."""

        req = Request.from_engine_core_request(request)
        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self.structured_output_manager.grammar_init(req)

        self.orchestrator.place_request_on_prefill_queue(req)
        logger.debug("added req %s to disaggregated orchestrator.",
                     req.request_id)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""
        pass

    def shutdown(self):
        self.orchestrator.stop()
        for executor in self.prefill_executors:
            executor.shutdown()
        for executor in self.decode_executors:
            executor.shutdown()

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 processor, P0 mirror, P1 mirror)
        # if self.scheduler.has_unfinished_requests():
        #     logger.warning("Resetting the multi-modal cache when requests are "
        #                    "in progress may lead to desynced internal caches.")

        self.mm_input_cache_server.reset()

    def profile(self, is_start: bool = True):
        self.prefill_executors[0].profile(is_start)

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        self.prefill_executors[0].sleep(level)

    def wake_up(self, tags: Optional[list[str]] = None):
        self.prefill_executors[0].wake_up(tags=tags)

    def is_sleeping(self) -> bool:
        return self.prefill_executors[0].is_sleeping

    def execute_dummy_batch(self):
        self.prefill_executors[0].collective_rpc("execute_dummy_batch")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.prefill_executors[0].add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.prefill_executors[0].remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.prefill_executors[0].list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.prefill_executors[0].pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self.prefill_executors[0].save_sharded_state(path=path,
                                                     pattern=pattern,
                                                     max_size=max_size)

    def collective_rpc(self,
                       method: Union[str, Callable[..., _R]],
                       timeout: Optional[float] = None,
                       args: tuple = (),
                       kwargs: Optional[dict[str, Any]] = None) -> list[_R]:
        return self.prefill_executors[0].collective_rpc(
            method, timeout, args, kwargs)

    def save_tensorized_model(
        self,
        tensorizer_config,
    ) -> None:
        self.prefill_executors[0].save_tensorized_model(
            tensorizer_config=tensorizer_config, )


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        on_head_node: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        engine_index: int = 0,
    ):
        input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()

        def executor_fail_callback():
            input_queue.put_nowait(
                (EngineCoreRequestType.EXECUTOR_FAILED, b''))

        # Create input socket.
        input_ctx = zmq.Context()
        identity = engine_index.to_bytes(length=2, byteorder="little")
        with make_zmq_socket(input_ctx,
                             handshake_address,
                             zmq.DEALER,
                             identity=identity,
                             linger=5000,
                             bind=False) as handshake_socket:

            # Register engine with front-end.
            addresses = self.startup_handshake(handshake_socket, on_head_node,
                                               vllm_config.parallel_config)
            self.client_count = len(addresses.outputs)

            # Update config which may have changed from the handshake.
            vllm_config.__post_init__()

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self._init_data_parallel(vllm_config)

            # Initialize engine core and model.
            super().__init__(vllm_config, executor_class, log_stats,
                             executor_fail_callback)

            self.engine_index = engine_index
            # self.step_fn = (self.step if self.batch_queue is None else
            #                 self.step_with_batch_queue)
            self.engines_running = False
            self.last_counts = (0, 0)

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            handshake_socket.send(
                msgspec.msgpack.encode({
                    "status": "READY",
                    "local": on_head_node,
                    "num_gpu_blocks": num_gpu_blocks,
                }))

        # Background Threads and Queues for IO. These enable us to
        # overlap ZMQ socket IO with GPU since they release the GIL,
        # and to overlap some serialization/deserialization with the
        # model forward pass.
        # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
        self.input_queue = input_queue
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        threading.Thread(target=self.process_input_sockets,
                         args=(addresses.inputs, addresses.coordinator_input,
                               identity),
                         daemon=True).start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(addresses.outputs, addresses.coordinator_output,
                  engine_index),
            daemon=True)
        self.output_thread.start()

    @staticmethod
    def startup_handshake(
            handshake_socket: zmq.Socket, on_head_node: bool,
            parallel_config: ParallelConfig) -> EngineZmqAddresses:

        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode({
                "status": "HELLO",
                "local": on_head_node,
            }))

        # Receive initialization message.
        logger.info("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError("Did not receive response from front-end "
                               f"process within {HANDSHAKE_TIMEOUT_MINS} "
                               f"minutes")
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata)
        logger.debug("Received init message: %s", init_message)

        received_parallel_config = init_message.parallel_config
        for key, value in received_parallel_config.items():
            setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args,
                        dp_rank: int = 0,
                        local_dp_rank: int = 0,
                        **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: Optional[EngineCoreProc] = None
        try:
            engine_core = EngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) return the outputs.
            self._process_output_queue()
            time.sleep(0.05)

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_output_queue(self):
        """Called only when there are unfinished local requests."""
        my_vllm_output_backlog = self.orchestrator._vllm_output_backlogs
        while my_vllm_output_backlog.qsize() > 0:
            outputs = my_vllm_output_backlog.get(block=False)
            for output in (outputs.items() if outputs else ()):
                self.output_queue.put_nowait(output)

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: Any) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
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

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
         arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation) if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation) else v
            for v, p in zip(args, arg_types))

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal("vLLM shutdown signal from EngineCore failed "
                         "to send. Please report this issue.")

    def process_input_sockets(self, input_addresses: list[str],
                              coord_input_address: Optional[str],
                              identity: bytes):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx,
                                    input_address,
                                    zmq.DEALER,
                                    identity=identity,
                                    bind=False))
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(ctx,
                                    coord_input_address,
                                    zmq.XSUB,
                                    identity=identity,
                                    bind=False))
                # Send subscription message to coordinator.
                coord_socket.send(b'\x01')

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b'')
                poller.register(input_socket, zmq.POLLIN)
            if coord_socket is not None:
                poller.register(coord_socket, zmq.POLLIN)

            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(
                        copy=False)
                    request_type = EngineCoreRequestType(
                        bytes(type_frame.buffer))

                    # Deserialize the request data.
                    decoder = add_request_decoder if (
                        request_type
                        == EngineCoreRequestType.ADD) else generic_decoder
                    request = decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(self, output_paths: list[str],
                               coord_output_path: Optional[str],
                               engine_index: int):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000))
                for output_path in output_paths
            ]
            coord_socket = stack.enter_context(
                make_zmq_socket(
                    ctx, coord_output_path, zmq.PUSH, bind=False,
                    linger=4000)) if coord_output_path is not None else None
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index
                outputs.scheduler_stats = {}

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(buffers,
                                                               copy=False,
                                                               track=True)
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

class DisaggEngineCoreProc(vLLMEngineCoreProc):
    """Wrapper for running vLLM EngineCore in background process."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        on_head_node: bool,
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

        executor_fail_callback = lambda: self.input_queue.put_nowait(
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
            idx:
            queue.Queue(vllm_config.scheduler_config.max_num_seqs)
            for idx, engine in enumerate(self._decode_engines)
        }

        self._prefill_threads = [
            orchestrator.JetThread(
                target=functools.partial(self._prefill, idx),
                name=f"prefill-{idx}",
                daemon=True,
            ) for idx in range(len(self._prefill_engines))
        ]
        self._transfer_threads = [
            orchestrator.JetThread(
                target=functools.partial(
                    self._transfer,
                    idx,
                ),
                name=f"transfer-{idx}",
                daemon=True,
            ) for idx in range(len(self._prefill_engines))
        ]
        self._decode_threads = [
            orchestrator.JetThread(
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

        with self._perform_handshake(handshake_address, identity, on_head_node,
                                     vllm_config) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None

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
                0
            ].mm_input_cache_server.get_and_update_p1(
                request.mm_inputs, request.mm_hashes
            )

        req = Request.from_engine_core_request(request)

        if req.use_structured_output:
            # Start grammar compilation asynchronously
            self._prefill_engines[0].structured_output_manager.grammar_init(req)

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
                        block_ids = (
                            prefill_engine.scheduler.kv_cache_manager.get_block_ids(
                                req_id
                            )
                        )
                        with LatencyTracker(f"ExtractKVCache-{req_id}-{block_ids}"):
                            # Assume one KV cache group for now.
                            kv_cache_map[req_id] = (
                                prefill_engine.model_executor.driver_worker.model_runner.get_kv_cache_for_block_ids(
                                    block_ids[0]
                                )
                            )
                        logger.debug(f"prefill done: for {req_id}")
                transfer_backlog.put(kv_cache_map, block=True)

                # tweak model_output to let the scheduler know kv_transfer is done for requests, so they can be freed.
                engine_core_outputs = prefill_engine.scheduler.update_from_output(
                    scheduler_output, model_output)  # type: ignore

                for req_id, idx in model_output.req_id_to_index.items():
                    if len(model_output.sampled_token_ids[idx]) > 0:
                        request = self._requests[req_id]
                        logger.debug(f"request-{req_id}: tokens={request._all_token_ids} after prefill")
                        # Remove request from the prefill engine.

                        request = prefill_engine.scheduler.requests[req_id]
                        prefill_engine.scheduler.running.remove(request)
                        prefill_engine.scheduler.encoder_cache_manager.free(request)
                        prefill_engine.scheduler._cached_reqs_data.pop(req_id, None)

                        prefill_engine.scheduler.kv_cache_manager.free(request)
                        prefill_engine.scheduler.kv_cache_manager.free_block_hashes(request)

                        prefill_engine.scheduler.requests.pop(req_id)

                for output in (engine_core_outputs.items() if engine_core_outputs else ()):
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

            logger.debug(f"transfer-{idx}: KV Cache items received: {kv_cachce_map.keys()}")

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
                        target_idx
                    ].model_executor.driver_worker.model_runner.transfer_kv_cache(
                        kv_cache
                    )

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
                if (
                    sum(decode_engine.scheduler.get_request_counts())
                    >= self.vllm_config.scheduler_config.max_num_seqs or
                    decode_engine.model_executor.driver_worker.model_runner.input_batch.num_reqs
                    >= self.vllm_config.scheduler_config.max_num_seqs
                ):
                    break

                try:
                    prefill_output = decode_backlog.get(block=block, timeout=1.0)
                except queue.Empty:
                    if block:
                        continue
                    break

                if prefill_output is None:
                    logger.info(f"decode-{idx} Empty output, and we are idle, exiting...")
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

                with LatencyTracker(
                        f"KVCacheInsert-{len(new_block_ids[0])}"):
                    decode_engine.model_executor.driver_worker.model_runner.insert_request_with_kv_cache(
                        vllm_request, kv_cache, new_block_ids
                    )

                vllm_request.status = RequestStatus.RUNNING
                decode_engine.scheduler.running.append(vllm_request)
                decode_engine.scheduler.requests[req_id] = vllm_request

                self._requests.pop(req_id)

            scheduler_output = decode_engine.scheduler.schedule()

            logger.debug(f"decode-{idx}: scheduler_output - {scheduler_output}")

            with LatencyTracker(f"decode-{idx}"):
                model_output = decode_engine.execute_model(scheduler_output)
            if scheduler_output.total_num_scheduled_tokens > 0:
                logger.debug(f"Decode result: {model_output}")

                engine_core_outputs = decode_engine.scheduler.update_from_output(
                    scheduler_output, model_output)  # type: ignore
                for output in (engine_core_outputs.items() if engine_core_outputs else ()):
                    self.output_queue.put_nowait(output)

    def shutdown(self):
        for e in self._prefill_engines:
            e.shutdown()
        for e in self._decode_engines:
            e.shutdown()
