# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"
# yapf: disable
# NOTE: if the user specifies a custom --xla_tpu_use_tc_device_shape_on_sc value, the second value
# will be used by default, effectively overriding this value
# TODO (jrplatin/clee1994): remove this after https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/gather/gather_reduce.py
# is moved to Pallas
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_use_tc_device_shape_on_sc=true " + os.environ.get("LIBTPU_INIT_ARGS", "")
# AOT compile is currently a Torch-only feature and thus we should not enable it
# for TPU
os.environ["VLLM_USE_AOT_COMPILE"] = "0"

# Handle XLA CPU compilation warning.
os.environ["XLA_FLAGS"] = "--xla_cpu_max_isa=AVX2 " + os.environ.get("XLA_FLAGS", "")

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
