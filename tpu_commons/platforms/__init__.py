# SPDX-License-Identifier: Apache-2.0
import os

# Always export the base class
__all__ = ["get_tpu_platform_cls"]


def get_tpu_platform_cls(backend_type="pytorch_xla"):
    """Get the appropriate TPU worker implementation."""
    if backend_type == "pytorch_xla" or backend_type == "torchax":
        from tpu_commons.platforms.tpu_torch_xla import TpuPlatform
        return TpuPlatform
    elif backend_type == "jax":
        from tpu_commons.platforms.tpu_jax import TpuPlatform
        return TpuPlatform
    else:
        raise ValueError(f"Unknown TPU backend type: {backend_type}")


# For convenience, also export the default worker
TPU_BACKEND_TYPE = os.environ.get("TPU_BACKEND_TYPE", "pytorch_xla").lower()
try:
    TpuPlatform = get_tpu_platform_cls(TPU_BACKEND_TYPE)
    __all__.append("TpuPlatform")
except (ImportError, ValueError) as e:
    import warnings
    warnings.warn(f"Default TPU platform not available: {str(e)}")
