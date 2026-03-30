# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"

# Fetch the existing args, default to an empty string if it doesn't exist
current_args = os.environ.get("LIBTPU_INIT_ARGS", "")

# Keep a list of missing arguments we need to add
args_to_append = []

# Check for the first flag
if "xla_tpu_use_tc_device_shape_on_sc" not in current_args:
    args_to_append.append("--xla_tpu_use_tc_device_shape_on_sc=true")

# Check for the second flag
if "xla_tpu_scheduler_percent_shared_memory_limit" not in current_args:
    args_to_append.append(
        "--xla_tpu_scheduler_percent_shared_memory_limit=1000")

# If we found any missing flags, append them to the environment variable
if args_to_append:
    # Join our new args with a space
    new_args_str = " ".join(args_to_append)

    # Combine the old and new args
    os.environ["LIBTPU_INIT_ARGS"] = f"{current_args} {new_args_str}"

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
