# SPDX-License-Identifier: Apache-2.0

from abc import ABC


class AbstractModelRunnerOutput(ABC):
    """Abstract base class for model runner output."""
    pass


class AbstractSchedulerOutput(ABC):
    """Abstract base class for scheduler output."""
    pass


class AbstractLoRARequest(ABC):
    """Abstract base class for LoRA request."""
    pass


class AbstractKVCacheConfig(ABC):
    """Abstract base class for KV cache config."""
    pass


class AbstractKVCacheSpec(ABC):
    """Abstract base class for KV cache spec."""
    pass
