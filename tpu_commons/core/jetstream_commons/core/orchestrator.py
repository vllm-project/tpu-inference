# Copyright 2024 Google LLC
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
"""Orchestrates the engines with performance optimization for inference.

1. A client sends a DecodeRequest via gRPC to the server, an 'LLMOrchestrator'.
2. This gets wrapped as an 'ActiveRequest' inside the orchestrator, with a
    'return_channel' queue as a place that output tokens can be placed.
    - The ActiveRequest is placed on the 'prefill_queue'.
    - A while loop runs continuously, yielding any tokens placed on the return
      channel until an end condition is met (EOS token or max tokens).
3. There is a prefill_thread per prefill_engine, each of which runs on a
    distinct prefill_slice.
4. There is a generate_thread per generate_engine, each of which runs on a
    distinct generate_slice.
5. Within a prefill thread:
    - It attempts to pop ActiveRequests off the prefill_queue.
    - It tokenizes the request.
    - When successful, it performs a prefill operation, transfers the kv cache
      to the generation slice and pops this information (still wrapped in the
      same ActiveRequest) onto the generation queue.
6. Within a generation thread:
   - There is a queue of integers representing 'available slots'.
   - It checks if there is something on both the slots_queue and generation_
     queue.
   - If so, the kv_cache associated with that request into the decoding state
    of the generation loop at the relevant slot.
   - Regardless, it performs a step.
  - It takes the sampled tokens, and places them on a 'detokenizing_queue'.
7. Within the detokenizing thread:
  - Tokens are detokenized for every 'slot' in a given set of sampled tokens.
  - When an end condition is met, the 'slot' integer is returned to the
    respective generation queue.
  - This does mean that a single generation step may run after detokenizing
    indicates that row is no longer valid (if the detokenizing is running behind
    generation steps), this is fine as it avoids detokenizing being blocking of
    the generate thread.

If you haven't worked with concurrency in python before - queues are thread-safe
by default, so we can happily use them to transfer pointers to data between
different processes. The structure of this server is simple as a result - a
thread for each thing we might want to do (prefill, transfer, generate,
detokenize), and corresponding queues that an active request is passed between.
The same goes for the 'return_channel' of the request itself, where we can just
pop tokens once they are done and try to pop them back to transmit them over
grpc.
It is literally queues all the way down! :)
The primary concern is GIL contention between threads, which is why we block
on queues that don't have an ongoing activity (i.e. everything but the
generation queue) because we don't control to go back to those queues until
necessary. Blocking means that the GIL doesn't switch back to that thread,
wheras continual queue get operations 'chop' control and mean that we do not
achieve good throughput. This is okay on the prefill/transfer/detokenization
threads because we don't need to do anything other than react to the presence
of items on these queues, wheras the generation thread needs to also run a
step - so it cannot block until it has new things to insert.

## Testing
This server is intended to be easy to locally test.

Either use :orchestrator test, which tests the multi-threading components,
:server_test, which extends this to test grpc_components, or run it locally
to debug hangs due to bugs in threads (it is easier to debug with live logs).
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


def delete_pytree(p):

    def delete_leaf(leaf):
        if isinstance(leaf, jax.Array):
            leaf.delete()
        del leaf

    jax.tree_map(delete_leaf, p)


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
    # Allows us to pre-load the params, primarily so that we can iterate quickly
    # on the driver in colab without reloading weights.
    _prefill_params: list[Any]
    _generate_params: list[Any]
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
    # detokenize threads. It is a list of tokens to be detokenized.
    _detokenize_backlogs: list[queue.Queue[Any]] = []
    _vllm_output_backlogs: list[queue.Queue[ModelRunnerOutput]] = []
    _generate_slots: list[queue.Queue[int]] = []
    #   _active_requests: list[queue.Queue[tuple[int, ActiveRequest]]] = []

    # For interleaved_mode, only generate if all slots are full
    # or corresponding prefill queue is empty.
    _interleaved_mode: bool = False

    # todo: remove jax_padding after all then engine migrate to np padding
    _jax_padding = True

    # All metrics we want to monitor should be collected with this
    #   _metrics_collector: JetstreamMetricsCollector | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefill_engines: Optional[list[engine_api.Engine]] = None,
        generate_engines: Optional[list[engine_api.Engine]] = None,
        prefill_params: Optional[list[Any]] = None,
        generate_params: Optional[list[Any]] = None,
        interleaved_mode: bool = False,
        jax_padding: bool = True,
        #   metrics_collector: JetstreamMetricsCollector | None = None,
        is_ray_backend: bool = False,
    ):
        if prefill_engines is None:
            prefill_engines = []
        if generate_engines is None:
            generate_engines = []
        if prefill_params is None:
            prefill_params = []
        if generate_params is None:
            generate_params = []

        logging.info(
            "Initialising driver with %d prefill engines and %d generate engines.",
            len(prefill_engines),
            len(generate_engines),
        )
        self.vllm_config = vllm_config
        self._prefill_engines = prefill_engines
        self._generate_engines = generate_engines
        self._prefill_params = prefill_params
        self._generate_params = generate_params
        self._interleaved_mode = interleaved_mode
        self.requests = {}
        self.reqs_to_remove = []
        # self._metrics_collector = metrics_collector

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
        # After generation, ActiveRequests are placed on the detokenization backlog
        # for tokens to be sent into each ActiveRequest's return channel.
        # We have one of these per generate engine to simplify the logic keeping
        # track of which generation engine to replace slots on.
        # This is a queue of either - tuple[int, ActiveRequest] which represents our
        # active requests, or tuple[int, sample_tokens]. We combine these into one
        # queue because it allows us to be somewhat clever with how we do
        # detokenization.
        # If the detokenization receives an (int, ActiveRequest) this signifies
        # that slot int should from now be placing tokens in the return channel of
        # the ActiveRequest.
        # If it receives (int, sample_tokens) then it actually
        # does a detokenization for any slots which have previously been set active
        # via the previous kind of object, and the int is used to log which step
        # the tokens were created at. By having them in one queue we prevent
        # the possibility of race conditions where a slot is made live before the
        # tokens are ready and it receives tokens from a different sequence,
        # or tokens detokenized before the relevant slot is live.
        self._detokenize_backlogs = [
            # We don't let detokenization accumulate more than 8 steps to avoid
            # synchronization issues.
            queue.Queue(8) for _ in self._generate_engines
        ]
        self._vllm_output_backlogs = [
            queue.Queue(8) for _ in self._generate_engines
        ]

        # A queue of integers representing available 'slots' in the decode
        # operation. I.e. potentially available rows in the batch and/or microbatch.
        # When we want to insert a prefill result, we pop an integer to insert at.
        # When this is empty, it means all slots are full.
        self._generate_slots = [
            queue.Queue(engine.max_concurrent_decodes)
            for engine in self._generate_engines
        ]
        _ = [[
            self._generate_slots[idx].put(i)
            for i in range(engine.max_concurrent_decodes)
        ] for idx, engine in enumerate(self._generate_engines)]

        self._jax_padding = jax_padding

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
        self.detokenize_threads = [
            JetThread(
                target=functools.partial(
                    self._detokenize_thread,
                    idx,
                ),
                name=f"detokenize-{idx}",
            ) for idx in range(len(self._generate_engines))
        ]
        self._all_threads = list(
            itertools.chain(
                self._prefill_threads,
                self._transfer_threads,
                self._generate_threads,
                self.detokenize_threads,
            ))
        self.live = True
        self._is_ray_backend = is_ray_backend
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
                self._detokenize_backlogs,
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
        # prefill_params = self._prefill_params[idx]
        logging.info("---------Prefill params %d loaded.---------", idx)

        while self.live:
            input_batch = prefill_engine.model_runner.input_batch
            if len(input_batch.req_id_to_index) == input_batch.max_num_reqs:
                if self.reqs_to_remove != []:
                    for req in self.reqs_to_remove:
                        input_batch.remove_request(req)
                    self.reqs_to_remove = []
                else:
                    continue
            my_transfer_backlog = self._transfer_backlogs[idx]
            # The prefill thread can just sleep until it has work to do.
            vllm_request = self._prefill_backlog.get(block=True)
            logging.info(
                "get request %s from prefill backlog", vllm_request.request_id
                if vllm_request is not None else "None")
            my_detokenize_backlog = self._detokenize_backlogs[idx]

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
                "Finished prefill req %s, Placed request on transfer queue %s",
                req_id,
                idx,
            )

            vllm_model_runner_output.req_ids = copy.deepcopy(
                vllm_model_runner_output.req_ids
            )
            vllm_model_runner_output.req_id_to_index = copy.deepcopy(
                vllm_model_runner_output.req_id_to_index
            )
            logging.info(f"prefill -> detoken: {vllm_model_runner_output}")
            my_detokenize_backlog.put(vllm_model_runner_output, block=True)

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
        my_detokenize_backlog = self._detokenize_backlogs[idx]

        # Keep track of what step tokens were generated at.
        generate_timestep = 0
        # State to store things like running kv cache in.

        # generate_params = self._generate_params[idx]
        # logging.info("---------Generate params %d loaded.---------", idx)
        # time_of_last_generate = time.time()
        # time_of_last_print = time.time() - 1
        active_reqs: dict[str, Request] = {}
        while self.live:
            # Check if there are any free my_slots. We don't want to block here since
            # we can still generate if we can't insert. We do this in a while loop to
            # insert as many sequences as possible.
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
            my_detokenize_backlog.put(vllm_model_runner_output, block=True)

            #   sampled_tokens.copy_to_host_async()
            #   # Respond to detokenization backpressure.
            #   my_detokenize_backlog.put((generate_timestep, sampled_tokens), block=True)
            generate_timestep += 1
            logging.info(
                "Finished generate step %s, req_ids %s, output tokens %s \n",
                generate_timestep, vllm_model_runner_output.req_ids,
                vllm_model_runner_output.sampled_token_ids)
            # logging.info(
            #     "Generate engine %d step %d - slots free : %d / %d, took %.2fms",
            #     idx,
            #     generate_timestep,
            #     my_slots_size,
            #     max_concurrent_decodes,
            #     (time.time() - time_of_last_generate) * 10**3,
            # )
            # time_of_last_generate = time.time()

    def _detokenize_thread(self, idx: int):
        """Detokenize sampled tokens and returns them to the user."""
        # One of these per generate engine.
        # For all filled my_slots, pop the sampled token onto the relevant
        # requests return channel. If it done, place it back onto free slots.
        my_detokenize_backlog = self._detokenize_backlogs[idx]
        # my_generate_engine = self._generate_engines[idx]
        # my_slots = self._generate_slots[idx]
        my_vllm_output_backlog = self._vllm_output_backlogs[idx]
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        while self.live:
            # pass
            model_runner_output = my_detokenize_backlog.get(block=True)
            # for request in model_runner_output.req_id_to_index:
            if not self.live:
                break
            logging.info(f"detoken: {model_runner_output}")
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
