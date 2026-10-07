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

# Configure JAX compilation cache environment variables if GCS or local cache is specified.
_gcs_cache_uri = (
    os.getenv("JAX_CACHE_GCS_DIR")
    or os.getenv("ROLLOUT_JAX_CACHE_GCS_DIR")
    or os.getenv("VLLM_JAX_CACHE_GCS_DIR")
)
if _gcs_cache_uri or os.getenv("LOCAL_JAX_CACHE_DIR") or os.getenv("JAX_CACHE_DIR"):
    try:
        from tpu_inference import gcs_cache
        gcs_cache.ensure_jax_cache_env()
    except Exception:
        pass

# Optional CLI arg hook for `vllm serve --jax-cache-gcs-dir gs://...`
try:
    import argparse
    import vllm.entrypoints.openai.cli_args as vllm_cli_args
    _orig_make_arg_parser = vllm_cli_args.make_arg_parser

    class _SetEnvAction(argparse.Action):
        def __init__(self, option_strings, dest, env_var, **kwargs):
            self.env_var = env_var
            super().__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            if values is not None:
                os.environ[self.env_var] = str(values)

    def _patched_make_arg_parser(parser):
        parser = _orig_make_arg_parser(parser)
        parser.add_argument(
            "--jax-cache-gcs-dir",
            type=str,
            default=None,
            action=_SetEnvAction,
            env_var="JAX_CACHE_GCS_DIR",
            help="GCS bucket URI to restore JAX compilation cache from and save to (e.g. gs://bucket/path).",
        )
        return parser

    vllm_cli_args.make_arg_parser = _patched_make_arg_parser
except (ImportError, AttributeError):
    pass

