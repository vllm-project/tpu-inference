# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os
import sys
import types

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
# AOT compile is currently a Torch-only feature and thus we should not enable it
# for TPU
os.environ["VLLM_USE_AOT_COMPILE"] = "0"

# Handle XLA CPU compilation warning.
os.environ["XLA_FLAGS"] = "--xla_cpu_max_isa=AVX2 " + os.environ.get(
    "XLA_FLAGS", "")

# TODO: Remove this when SMEM capacity optimization for batched rpa lands.
os.environ[
    "LIBTPU_INIT_ARGS"] = "--xla_tpu_use_dynamic_smem_negotiation=true " + os.environ.get(
        "LIBTPU_INIT_ARGS", "")

# Monkeypatch vLLM to avoid ImportError: cannot import name 'SamplingParams' from 'vllm'
# in vllm/v1/... submodules due to circular imports or lazy loading failures.
try:
    import vllm
    import vllm.sampling_params
    if not hasattr(vllm, "SamplingParams"):
        vllm.SamplingParams = vllm.sampling_params.SamplingParams
    if not hasattr(vllm, "SamplingType"):
        vllm.SamplingType = vllm.sampling_params.SamplingType
    if not hasattr(vllm, "SamplingStatus"):
        from vllm.sampling_params import RequestOutputKind
        vllm.RequestOutputKind = RequestOutputKind
except ImportError:
    pass

# Bypass cutlass installation requirement. It is unconditionally imported by
# upstream vLLM (e.g. DeepSeek V4 ops), but only actually invoked on NVIDIA GPUs.
if "cutlass" not in sys.modules:
    sys.modules["cutlass"] = types.ModuleType("cutlass")

# Provide PyTorch/TorchAX implementation for per_token_group_quant_fp8 on TPU
# to avoid GPU Triton kernel invocation from vLLM's fp8_utils.
import torch


def _tpu_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype = torch.float8_e4m3fn,
    column_major_scales: bool = False,
    scale_ub: torch.Tensor | None = None,
    use_ue8m0: bool = False,
):
    fp8_max = torch.finfo(dtype).max
    x_reshaped = x.view(-1, group_size)
    amax = torch.amax(torch.abs(x_reshaped), dim=-1,
                      keepdim=True).clamp(min=eps)
    x_s = amax / fp8_max
    x_q = (x_reshaped / x_s).to(dtype).view_as(x)
    if not column_major_scales:
        x_s = x_s.view(x.shape[:-1] + (x.shape[-1] // group_size, ))
    else:
        x_s = x_s.view(x.shape[0], x.shape[1] // group_size)
    return x_q, x_s


try:
    import vllm.model_executor.layers.quantization.utils.fp8_utils as _vllm_fp8_utils
    _vllm_fp8_utils.per_token_group_quant_fp8 = _tpu_per_token_group_quant_fp8
except (ImportError, AttributeError):
    pass
