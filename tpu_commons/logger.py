# SPDX-License-Identifier: Apache-2.0

from vllm.logger import _VllmLogger
from vllm.logger import init_logger as init_vllm_logger


def init_logger(name: str) -> _VllmLogger:
    # Prepend the root "vllm" to the module path to use vllm's configured logger.
    patched_name = "vllm." + name
    return init_vllm_logger(patched_name)
