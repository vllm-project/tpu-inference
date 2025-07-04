# SPDX-License-Identifier: Apache-2.0
"""Orchestrates disaggregated prefill and decode engines for inference.

The Driver class manages the lifecycle of a request through a multi-stage
pipeline, using dedicated threads and queues for each stage to maximize
throughput and hardware utilization.

Request Lifecycle:
1.  A `vllm.v1.request.Request` object is submitted to the Driver and placed
    on the `_prefill_backlog` queue.

2.  A `_prefill_thread` picks up the request. It calls the `prefill()` method
    on a prefill engine, which processes the prompt and generates the initial
    Key-Value (KV) cache. The output, containing the KV cache and the updated
    request state, is placed on a `_transfer_backlog` queue.

3.  A `_transfer_thread` picks up the prefill output. It calls the decode
    engine's `transfer_kv_cache()` method to re-shard the KV cache for the
    decode engine's hardware topology. The request is then routed to the least
    busy `_generate_backlog` queue.

4.  A `_generate_thread` pulls the request from its backlog. It first calls
    `insert_request_with_kv_cache()` to load the request's state and KV cache
    into the decode engine. It then enters a loop, repeatedly calling the
    `generate()` method to produce one token at a time. The raw output
    (`ModelRunnerOutput`) is placed on an `_output_backlog`.

5.  An `_output_thread` processes the `ModelRunnerOutput`. It packages the raw
    token data into the final `EngineCoreOutput` format and places it on the
    `_vllm_output_backlog`. This final queue is consumed by the main `EngineCore`
    loop, which sends the results back to the client.

This architecture uses non-blocking or timed-blocking queues to manage the
flow of requests, preventing GIL contention and ensuring that compute resources
remain highly utilized.

## Testing
We currently mostly test this with manual test, namely on a v6e-8, you can run

  PREFILL_SLICES=2,2 DECODE_SLICES=2,2 \
  TPU_BACKEND_TYPE=jax \
  python tpu_commons/examples/offline_inference.py \
    --task=generate \
    --model=meta-llama/Meta-Llama-3-8B-Instruct \
    --max_model_len=1024 \
    --max_num_seqs=8
"""

import copy
import functools
import itertools
import logging
import os
import queue
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from typing import Any, Optional

import jax

from vllm.config import VllmConfig
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from tpu_commons.core.jetstream_commons.engine import engine_api

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


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


class Driver:
    """Drives the engines."""

    _prefill_engines: list[engine_api.Engine]
    _generate_engines: list[engine_api.Engine]
    # Stage 1
    _prefill_backlog: queue.Queue[Request | None]
    # Stage 2
    _transfer_backlogs: list[queue.Queue[Any]] = []
    # Stage 3
    # We keep this as a dict to avoid a possibly expensive object comparison
    # when logging the index of the generate engine we send a prefill result
    # to, it allows us to natively have the index from the min operation, rather
    # than have to call .index()
    _generate_backlogs: dict[int, queue.Queue[Any]] = {}
    # Stage 4
    # This can be a list because we can pass it as an arg to generate and
    # output threads. It is a list of tokens to be sent out.
    _output_backlogs: list[queue.Queue[Any]] = []
    _vllm_output_backlogs: list[queue.Queue[ModelRunnerOutput]] = []
    #   _active_requests: list[queue.Queue[tuple[int, ActiveRequest]]] = []

    # For interleaved_mode, only generate if all slots are full
    # or corresponding prefill queue is empty.
    _interleaved_mode: bool = False

    # All metrics we want to monitor should be collected with this
    #   _metrics_collector: JetstreamMetricsCollector | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefill_engines: Optional[list[engine_api.Engine]] = None,
        generate_engines: Optional[list[engine_api.Engine]] = None,
        interleaved_mode: bool = False,
    ):
        if prefill_engines is None:
            prefill_engines = []
        if generate_engines is None:
            generate_engines = []

        logging.info(
            "Initialising driver with %d prefill engines and %d generate engines.",
            len(prefill_engines),
            len(generate_engines),
        )
        self.vllm_config = vllm_config
        self._prefill_engines = prefill_engines
        self._generate_engines = generate_engines
        self._interleaved_mode = interleaved_mode
        self.requests = {}
        self.reqs_to_remove = []

        # Stages 1-4 represent the life cycle of a request.
        # Stage 1
        # At first, a request is placed here in order to get prefilled.
        self._prefill_backlog = queue.Queue()
        # if self._metrics_collector:
        #   self._metrics_collector.get_prefill_backlog_metric().set_function(
        #       lambda: float(self._prefill_backlog.qsize())
        #   )

        # Stage 2
        # After prefilling, it is placed here in order to get transferred to
        # one of the generate backlogs.
        # Interleaved Mode: Max size is 1 to increase the HBM utilization
        # during generate.
        # Disaggregated Mode: Max size is 4 to allow for 2 prefills to be enqueued
        # while 1 transfer is enqueued while 1 is being transferred.
        # TODO: Make queue size configurable.
        self._transfer_backlogs = [
            queue.Queue(1 if self._interleaved_mode else 4)
            for i in range(len(self._prefill_engines))
        ]
        # Stage 3
        # Each generate engine accesses its own generate backlog.
        # Interleaved Mode: Max size is 1 to increase the HBM utilization
        # during generate.
        # Disaggregated Mode: Set as 1/3 the number of concurrent decodes.
        # TODO: Calculate the backlog to saturate the generate engine while
        # minimizing the memory usage for disaggregated mode.
        # TODO: Make queue size configurable.
        self._generate_backlogs = {
            idx:
            queue.Queue(1 if self.
                        _interleaved_mode else engine.max_concurrent_decodes //
                        3)
            for idx, engine in enumerate(self._generate_engines)
        }
        # Stage 4
        # After generation, ActiveRequests are placed on the output backlog
        # for tokens to be sent into each ActiveRequest's return channel.
        # We have one of these per generate engine to simplify the logic keeping
        # track of which generation engine to replace slots on.
        # This is a queue of either - tuple[int, ActiveRequest] which represents our
        # active requests, or tuple[int, sample_tokens].
        self._output_backlogs = [
            # We don't let output accumulate more than 8 steps to avoid
            # synchronization issues.
            queue.Queue(8) for _ in self._generate_engines
        ]
        self._vllm_output_backlogs = [
            queue.Queue(8) for _ in self._generate_engines
        ]

        # Create all threads
        self._prefill_threads = [
            JetThread(
                target=functools.partial(self._prefill_thread, idx),
                name=f"prefill-{idx}",
                daemon=True,
            ) for idx in range(len(self._prefill_engines))
        ]
        self._transfer_threads = [
            JetThread(
                target=functools.partial(
                    self._transfer_thread,
                    idx,
                ),
                name=f"transfer-{idx}",
                daemon=True,
            ) for idx in range(len(self._prefill_engines))
        ]
        self._generate_threads = [
            JetThread(
                target=functools.partial(
                    self._generate_thread,
                    idx,
                ),
                name=f"generate-{idx}",
                daemon=True,
            ) for idx in range(len(self._generate_engines))
        ]
        self.output_threads = [
            JetThread(
                target=functools.partial(
                    self._output_thread,
                    idx,
                ),
                name=f"output-{idx}",
            ) for idx in range(len(self._generate_engines))
        ]
        self._all_threads = list(
            itertools.chain(
                self._prefill_threads,
                self._transfer_threads,
                self._generate_threads,
                self.output_threads,
            ))
        self.live = True
        # Start all threads
        for t in self._all_threads:
            t.start()

    def stop(self):
        """Stops the driver and all background threads."""
        # Signal to all threads that they should stop.
        self.live = False

        all_backlogs = list(
            itertools.chain(
                [self._prefill_backlog],
                self._transfer_backlogs,
                self._generate_backlogs.values(),
                self._output_backlogs,
                self._vllm_output_backlogs,
            ))

        while any(t.is_alive() for t in self._all_threads):
            # Empty all backlogs and mark any remaining requests as cancelled.
            for q in all_backlogs:
                while True:
                    try:
                        r = q.get_nowait()
                        if r is None:
                            continue
                    except queue.Empty:
                        break

            # Put sentinels to unblock threads.
            for q in all_backlogs:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass

        # Wait for all threads to stop.
        for t in self._all_threads:
            t.join()

    def get_total_concurrent_requests(self) -> int:
        """Gets the total number of concurrent requests the driver can handle."""
        # We don't support filling all backlogs at once because it can cause GIL
        # contention.
        total_max_concurrent_decodes = sum(
            [e.max_concurrent_decodes for e in self._generate_engines])
        return total_max_concurrent_decodes

    def place_request_on_prefill_queue(self, request: Any):
        """Used to place new requests for prefilling and generation."""
        # Don't block so we can fail and shed load when the queue is full.
        self._prefill_backlog.put(request, block=False)

    def _prefill_thread(self, idx: int):
        """Thread which runs in the background performing prefills."""
        logging.info("---------Spinning up prefill thread %d.---------", idx)
        prefill_engine = self._prefill_engines[idx]
        logging.info("---------Prefill params %d loaded.---------", idx)

        # TODO(fhzhang): add better dispatch algorithm.
        target_idx = idx % len(self._generate_backlogs)
        my_output_backlog = self._output_backlogs[target_idx]
        my_transfer_backlog = self._transfer_backlogs[idx]

        while self.live:
            input_batch = prefill_engine.model_runner.input_batch
            if len(input_batch.req_id_to_index) == input_batch.max_num_reqs:
                if self.reqs_to_remove != []:
                    for req in self.reqs_to_remove:
                        input_batch.remove_request(req)
                    self.reqs_to_remove = []
                else:
                    continue
            # The prefill thread can just sleep until it has work to do.
            vllm_request = self._prefill_backlog.get(block=True)
            logging.info(
                "get request %s from prefill backlog", vllm_request.request_id
                if vllm_request is not None else "None")

            if vllm_request is None:
                break

            # Compute new kv cache for the prefill_content.
            prefill_output, vllm_model_runner_output = prefill_engine.prefill(
                vllm_request=vllm_request
            )
            vllm_model_runner_output.req_id_to_index = copy.deepcopy(
                vllm_model_runner_output.req_id_to_index
            )
            req_id = vllm_request.request_id
            self.requests[req_id] = vllm_request
            logging.info("Put request %s in req dict. request.num_tokens: %s",
                         req_id, vllm_request.num_tokens)
            # logging.warning("finished prefill for request %s output %s \n", vllm_request.request_id, vllm_model_runner_output.__dict__)
            # logging.warning("added %s to requests dictionary \n", self.requests[req_id].__dict__)
            # request.prefill_result = prefill_result
            # Once prefill is complete, place it on the generation queue and block if
            # full.
            my_transfer_backlog.put(prefill_output, block=True)
            logging.info(
                f"Prefill worker {idx}: Finished prefill req {req_id}, Placed request on transfer queue {target_idx}"
            )

            vllm_model_runner_output.req_ids = copy.deepcopy(
                vllm_model_runner_output.req_ids
            )
            vllm_model_runner_output.req_id_to_index = copy.deepcopy(
                vllm_model_runner_output.req_id_to_index
            )
            logging.info(f"prefill -> output: {vllm_model_runner_output}")
            my_output_backlog.put(vllm_model_runner_output, block=True)

            del prefill_output
            del vllm_model_runner_output
            del vllm_request

    def _transfer_thread(self, idx: int):
        """Transfers the kv cache on an active request to the least full
    generate backlog."""
        transfer_backlog = self._transfer_backlogs[idx]
        while self.live:
            # The transfer thread can just sleep until it has work to do.
            prefill_output = transfer_backlog.get(block=True)
            if prefill_output is None:
                break
            kv_cache = prefill_output["cache"]
            request = prefill_output["request"]
            target_idx = min(self._generate_backlogs.items(),
                             key=lambda q: q[1].qsize())[0]
            # Only transfer the KVCache for the disaggregated serving.
            # TODO: Remove the conditional after fixing the compatibility.
            if not self._interleaved_mode:
                kv_cache = self._generate_engines[target_idx].model_runner.transfer_kv_cache(kv_cache)
                prefill_output["cache"] = kv_cache

            # Place the request on the correct generate backlog and block if full.
            self._generate_backlogs[target_idx].put(prefill_output, block=True)
            logging.info(
                "Successfully transferred prefill request %s"
                "from prefill engine %d to generate engine %d. generate backlog len %d",
                request.request_id,
                idx,
                target_idx,
                self._generate_backlogs[target_idx].qsize(),
            )

    def _generate_thread(self, idx: int):
        """Step token generation and insert prefills from backlog."""
        logging.info("---------Spinning up generate thread %d.---------", idx)
        generate_engine = self._generate_engines[idx]
        my_generate_backlog = self._generate_backlogs[idx]
        my_output_backlog = self._output_backlogs[idx]

        active_reqs: dict[str, Request] = {}
        while self.live:
            # Try to fill the batch with new requests from the generate backlog
            # before running a generation step. We don't want to block here
            # since we can still generate tokens for existing requests.
            while True:
                logging.info(f"generate({idx}): looping on backlog...")
                if not self._prefill_backlog.empty() and len(
                        self.reqs_to_remove) != 0:
                    time.sleep(1.0)
                    continue

                block = len(active_reqs) == 0
                try:
                    prefill_output = my_generate_backlog.get(block=block, timeout=1.0)
                    # Got free slot and new request, use them.
                except queue.Empty:
                    # No new requests, we can't insert, so put back slot.
                    # If we were blocking and hit the timeout, then retry the loop.
                    # Otherwise, we can exit and proceed to generation.
                    if block:
                        continue
                    else:
                        break
                if prefill_output is None:
                    return
                else:
                    pass
                logging.info(
                    f"generate backlog len: {my_generate_backlog.qsize()}")
                request = prefill_output["request"]
                kv_cache = prefill_output["cache"]
                new_block_ids = generate_engine.get_new_block_ids(request)
                generate_engine.model_runner.insert_request_with_kv_cache(
                    request, kv_cache, new_block_ids
                )
                active_reqs[request.request_id] = request

            logging.info(f"executing generation... #active_reqs={len(active_reqs)}")
            # At this point, we know that we have at least some slots filled.
            # Now we actually take a generate step on requests in the slots.
            vllm_model_runner_output, reqs_to_remove = generate_engine.generate(
                active_reqs)
            for req_id in reqs_to_remove:
                active_reqs.pop(req_id)

            vllm_model_runner_output.req_ids = copy.deepcopy(
                vllm_model_runner_output.req_ids
            )
            vllm_model_runner_output.req_id_to_index = copy.deepcopy(
                vllm_model_runner_output.req_id_to_index
            )
            if len(reqs_to_remove) != 0:
                self.reqs_to_remove = reqs_to_remove

            logging.info(f"generate enqueue: {vllm_model_runner_output}")
            my_output_backlog.put(vllm_model_runner_output, block=True)

            logging.info(
                "Finished generate step, req_ids %s, output tokens %s \n",
                vllm_model_runner_output.req_ids,
                vllm_model_runner_output.sampled_token_ids)

    def _output_thread(self, idx: int):
        """Processes raw model output and packages it for the client."""
        my_output_backlog = self._output_backlogs[idx]
        my_vllm_output_backlog = self._vllm_output_backlogs[idx]
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        while self.live:
            # pass
            model_runner_output = my_output_backlog.get(block=True)
            # for request in model_runner_output.req_id_to_index:
            if not self.live:
                break
            logging.info(f"output: {model_runner_output}")
            #   sampled_token_id = model_runner_output.sampled_token_ids[]
            # logging.info("Detokenize model runner output: %s", model_runner_output)
            req_ids = list(model_runner_output.prompt_logprobs_dict.keys())
            for req_id in req_ids:
                request = self.requests[req_id]
                sampled_token_index = model_runner_output.req_id_to_index[req_id]
                sampled_token_id = model_runner_output.sampled_token_ids[
                    sampled_token_index]
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=sampled_token_id,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=None,
                        new_prompt_logprobs_tensors=None,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        # kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }
            outputs = defaultdict(list)
            logging.info(
                "Put engine core outputs %s to vllm output backlog, queue len %s",
                req_ids, my_vllm_output_backlog.qsize())
            my_vllm_output_backlog.put(engine_core_outputs, block=True)
