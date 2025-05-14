# SPDX-License-Identifier: Apache-2.0
import os

# Always export the base class
__all__ = ["get_tpu_communicator_cls"]


def get_tpu_communicator_cls(worker_type=None):
    """Get the appropriate TPU worker implementation."""

    # Use environment variable if no explicit type is provided
    if worker_type is None:
        worker_type = os.environ.get("TPU_BACKEND_TYPE", "pytorch_xla").lower()

    if worker_type == "pytorch_xla":
        from tpu_commons.distributed.device_communicators.tpu_communicator_torch_xla import \
            TpuCommunicator
        return TpuCommunicator
    else:
        raise ValueError(f"Unknown TPU worker type: {worker_type}")


# For convenience, also export the default worker
TPU_BACKEND_TYPE = os.environ.get("TPU_BACKEND_TYPE", "pytorch_xla").lower()
try:
    TpuCommunicator = get_tpu_communicator_cls(TPU_BACKEND_TYPE)
    __all__.append("TpuCommunicator")
except (ImportError, ValueError) as e:
    import warnings
    warnings.warn(f"Default TPU worker not available: {str(e)}")
