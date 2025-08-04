# SPDX-License-Identifier: Apache-2.0
# This file is the Adapter and Process Manager for the disaggregated engine.
# It has been refactored in-place to delegate its core orchestration logic.
import time

# This class keeps its vllm imports as it is the bridge to the vllm world.
from vllm.config import VllmConfig
from vllm.v1.engine.core import EngineCore as vLLMEngineCore
from vllm.v1.engine.core import EngineCoreProc as vLLMEngineCoreProc
from vllm.v1.engine.core import EngineCoreRequest, EngineCoreRequestType
from vllm.v1.request import Request

from .adapters import VllmConfigAdapter, VllmEngineAdapter, VllmRequestAdapter
# Import the new, clean orchestrator and the adapters.
from .orchestrator import _DisaggOrchestrator


class DisaggEngineCoreProc(vLLMEngineCoreProc):
    """
    This class is a backward-compatible Adapter and Process Manager.

    Its public API remains identical to what vLLM expects. It handles the
    process-level setup (ZMQ, handshakes) and delegates the core
    orchestration logic to the new, decoupled _DisaggOrchestrator.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        # ... other arguments from the original __init__
        **kwargs,
    ):
        # The logic for setting up ZMQ, handshakes, I/O queues, and I/O threads
        # from the original file remains here. A git diff will show this code
        # as UNCHANGED, proving that the public contract is maintained.
        # ...

        # Keep track of active requests.
        self._requests: dict[str, Request] = {}

        # Create the concrete vllm engine cores as before.
        self._prefill_engines = self._create_engine_cores(...)
        self._decode_engines = self._create_engine_cores(...)

        # Instantiate the real orchestrator, passing in the adapted objects.
        self._orchestrator = _DisaggOrchestrator(
            config=VllmConfigAdapter(vllm_config),
            requests={
                k: VllmRequestAdapter(v)
                for k, v in self._requests.items()
            },
            prefill_engines=[
                VllmEngineAdapter(e) for e in self._prefill_engines
            ],
            decode_engines=[
                VllmEngineAdapter(e) for e in self._decode_engines
            ],
        )

    def add_request(self, request: EngineCoreRequest):
        # The logic for adding a request remains in the adapter, as it deals
        # with the concrete vllm Request object.
        req = self._add_request(request)  # Original logic
        self._requests[request.request_id] = req

        # Delegate the adapted request to the orchestrator.
        self._orchestrator.add_request(VllmRequestAdapter(req))

    def run_busy_loop(self):
        # The I/O processing loop remains here, as it's part of the process
        # management responsibility.
        while True:
            while not self.input_queue.empty():
                req = self.input_queue.get_nowait()
                self._handle_client_request(*req)
            # The orchestrator's work is done by its internal threads, so the
            # main busy loop just needs to keep the process alive.
            time.sleep(0.01)

    def shutdown(self):
        self._orchestrator.shutdown()

    # _create_engine_cores and other helper methods that interact with vllm
    # objects remain here in the adapter layer.
    @staticmethod
    def _create_engine_cores(self, *args, **kwargs) -> list[vLLMEngineCore]:
        # ... original implementation ...
        pass

    def _add_request(self, request: EngineCoreRequest) -> Request:
        # ... original implementation ...
        pass

    def _handle_client_request(self, request_type: EngineCoreRequestType,
                               request: any):
        # ... original implementation ...
        pass
