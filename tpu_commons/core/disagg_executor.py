# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import worker_receiver_cache_from_config
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        run_method)
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.utils import get_and_update_mm_cache
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class DisaggExecutor(Executor):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        slice_config = getattr(self.vllm_config.device_config, "slice")
        idx = slice_config[0]
        sizes = slice_config[1]

        start = sum(sizes[0:idx])
        end = start + sizes[idx]

        devices = jax.devices()[start:end]
        setattr(self.vllm_config.device_config, "slice", (idx + 1, sizes))
        logger.info(
            f"Creating DisaggExecutor with {devices}, index: {start} -> {end}")

        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        local_rank = 0
        rank = 0
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            devices=devices,
        )
        self.mm_receiver_cache = worker_receiver_cache_from_config(
            self.vllm_config, MULTIMODAL_REGISTRY, Lock())
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None,
                       non_block: bool = False) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        if self.mm_receiver_cache is not None and method == "execute_model":
            get_and_update_mm_cache(self.mm_receiver_cache, args)

        if not non_block:
            return [run_method(self.driver_worker, method, args, kwargs)]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    return [async_thread.submit(result.get_output)]
                result = result.get_output()
            future = Future[Any]()
            future.set_result(result)
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return [future]

    def check_health(self) -> None:
        # DisaggExecutor will always be healthy as long as
        # it's running.
        return
