import multiprocessing
import multiprocessing.queues
import os
import signal
import socket
import subprocess
import uuid
from typing import Generic, Hashable, List, Optional, OrderedDict, TypeVar

import ray
from ray._private.accelerators import TPUAcceleratorManager

from tpu_commons.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


def get_ip_address() -> str:
    return socket.gethostbyname(socket.gethostname())


def run_cmd(cmd: str, *args, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd.split(), *args, **kwargs)


def kill_process(pid: int = os.getpid()) -> None:
    os.kill(pid, signal.SIGTERM)


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


_megacore = False


def enable_megacore() -> None:
    global _megacore
    _megacore = True


def get_megacore() -> bool:
    return _megacore


def has_megacore_support() -> bool:
    return is_tpu_v4() or is_tpu_v5p()


def is_tpu_v6e() -> bool:
    tpu_type = TPUAcceleratorManager.get_current_node_accelerator_type()
    return "V6E" in tpu_type.upper()


def is_tpu_v5p() -> bool:
    tpu_type = TPUAcceleratorManager.get_current_node_accelerator_type()
    return "V5P" in tpu_type.upper()


def is_tpu_v5e() -> bool:
    tpu_type = TPUAcceleratorManager.get_current_node_accelerator_type()
    return "V5LITE" in tpu_type.upper()


def is_tpu_v4() -> bool:
    tpu_type = TPUAcceleratorManager.get_current_node_accelerator_type()
    return "V4" in tpu_type.upper()


def get_num_hosts() -> int:
    if ray.is_initialized():
        ray_resources = ray.available_resources()
        for key in ray_resources:
            if "accelerator_type" in key or "resource-pool" in key:
                return int(ray_resources[key])
        raise ValueError(f"No accelerator_type found in {ray_resources}")
    else:
        # Only single host is supported if not using Ray.
        return 1


def get_num_available_tpus() -> int:
    if ray.is_initialized():
        ray_resources = ray.available_resources()
        return int(ray_resources["TPU"])
    else:
        return TPUAcceleratorManager.get_current_node_num_accelerators()


def get_num_available_tpus_per_host() -> int:
    return get_num_available_tpus() // get_num_hosts()


def set_visible_tpu_ids(tpu_ids: List[int]) -> None:
    validate = TPUAcceleratorManager.validate_resource_request_quantity(
        len(tpu_ids))
    if not validate[0]:
        raise ValueError(validate[1])
    tpu_ids = [str(tpu_id) for tpu_id in tpu_ids]
    TPUAcceleratorManager.set_current_process_visible_accelerator_ids(tpu_ids)


def pad_to_multiple(x: int,
                    multiple: int = 8,
                    max_limit: Optional[int] = None,
                    keep_one: bool = False) -> int:
    assert x > 0
    if keep_one and x == 1:
        return x
    x = x + (-x % multiple)
    if max_limit is not None:
        x = min(x, max_limit)
    return x


# Reference from vLLM: https://source.corp.google.com/h/vertex-model-garden/vllm/+/vertex:vllm/utils.py
class LRUCache(Generic[T]):

    def __init__(self, capacity: int):
        self.cache: OrderedDict[Hashable, T] = OrderedDict()
        self.capacity = capacity

    def __contains__(self, key: Hashable) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: Hashable) -> Optional[T]:
        return self.get(key)

    def __setitem__(self, key: Hashable, value: T) -> None:
        self.put(key, value)

    def __delitem__(self, key: Hashable) -> None:
        self.pop(key)

    def touch(self, key: Hashable) -> None:
        self.cache.move_to_end(key)

    def get(self,
            key: Hashable,
            default_value: Optional[T] = None) -> Optional[T]:
        if key in self.cache:
            value: Optional[T] = self.cache[key]
            self.cache.move_to_end(key)
        else:
            value = default_value
        return value

    def put(self, key: Hashable, value: T) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        self._remove_old_if_needed()

    def _on_remove(self, key: Hashable, value: Optional[T]):
        pass

    def remove_oldest(self):
        if not self.cache:
            return
        key, value = self.cache.popitem(last=False)
        self._on_remove(key, value)

    def _remove_old_if_needed(self) -> None:
        while len(self.cache) > self.capacity:
            self.remove_oldest()

    def pop(self,
            key: Hashable,
            default_value: Optional[T] = None) -> Optional[T]:
        run_on_remove = key in self.cache
        value: Optional[T] = self.cache.pop(key, default_value)
        if run_on_remove:
            self._on_remove(key, value)
        return value

    def clear(self):
        while len(self.cache) > 0:
            self.remove_oldest()
        self.cache.clear()


class MpQueue(multiprocessing.queues.Queue):
    """A wrapper around multiprocessing.Queue that provides more methods."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize, ctx=multiprocessing.get_context())

    def put_nowait_batch(self, items: List[T]) -> None:
        for item in items:
            self.put_nowait(item)

    def get_nowait_batch(self, num_items: int) -> List[T]:
        # NOTE: This method calls self.get() instead of self.get_nowait(),
        # because self.get_nowait() will raise Empty exception if the queue is empty,
        # (even if check self.qsize() > 0, which is not reliable.)
        # Use blocking call to make the behavior consistent with ray.Queue.
        items = []
        for _ in range(num_items):
            items.append(self.get())
        return items
