# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        run_method)
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase


class DisaggExecutor(Executor):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        pass

    def init_with_devices(self, devices):
        """Initialize the worker with specified devices and load the model."""
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
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
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        answer = run_method(self.driver_worker, method, args, kwargs)
        return [answer]

    def check_health(self) -> None:
        # DisaggExecutor will always be healthy as long as
        # it's running.
        return
