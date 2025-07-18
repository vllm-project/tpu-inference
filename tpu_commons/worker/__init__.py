# SPDX-License-Identifier: Apache-2.0
import os

# Always export the base class
__all__ = ["get_tpu_worker_cls"]


def get_tpu_worker_cls(worker_type=None):
    """Get the appropriate TPU worker implementation."""

    # Use environment variable if no explicit type is provided
    if worker_type is None:
        worker_type = os.environ.get("TPU_BACKEND_TYPE", "jax").lower()

    if worker_type == "torchax":
        from tpu_commons.worker.tpu_worker_torchax import TPUWorker
        return TPUWorker
    elif worker_type == "jax":
        from tpu_commons.worker.tpu_worker_jax import TPUWorker
        return TPUWorker
    else:
        raise ValueError(f"Unknown TPU worker type: {worker_type}")


# For convenience, also export the default worker
TPU_BACKEND_TYPE = os.environ.get("TPU_BACKEND_TYPE", "jax").lower()
try:
    TPUWorker = get_tpu_worker_cls(TPU_BACKEND_TYPE)
    __all__.append("TPUWorker")
except (ImportError, ValueError) as e:
    import warnings
    warnings.warn(f"Default TPU worker not available: {str(e)}")
