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

def _patch_jax_pallas_fori_lowering() -> None:
    """Elide the dead remainder ``scf.for`` from Pallas ``fori_loop`` lowering.

    JAX Pallas' Mosaic lowering unconditionally emits a remainder loop for
    dynamic-bound ``jax.lax.fori_loop`` even when ``unroll=1`` (the default),
    where the main loop already covers the full range and the remainder is a
    no-op. libtpu still lowers the empty remainder to device code, which
    materially grows the emitted Mosaic body and slows decode-heavy MoE
    workloads that rely on ``fori_loop`` in Pallas scheduler kernels.

    Rewrites ``_lower_jaxpr_to_for_loop`` in place to gate the remainder
    emission on ``unroll != 1``. Applied at import so it takes effect before
    any Pallas kernel is traced. No-op if the source no longer contains the
    exact needle (upstream fix landed, or the file was refactored), so it
    remains safe across JAX bumps.
    """
    try:
        from jax._src.pallas.mosaic import lowering as _lowering
    except Exception:
        return
    if getattr(_lowering, "_torchtpu_fori_patch_applied", False):
        return
    orig = getattr(_lowering, "_lower_jaxpr_to_for_loop", None)
    if orig is None:
        return

    import inspect
    import textwrap
    try:
        src = textwrap.dedent(inspect.getsource(orig))
    except Exception:
        return
    needle = "elif has_dynamic_remainder:"
    fix = "elif has_dynamic_remainder and unroll != 1:"
    if needle not in src or fix in src:
        return
    patched_src = src.replace(needle, fix, 1)
    ns: dict = {"__name__": _lowering.__name__}
    try:
        exec(compile(patched_src, _lowering.__file__, "exec"),
             _lowering.__dict__, ns)
    except Exception:
        return
    new_fn = ns.get("_lower_jaxpr_to_for_loop")
    if new_fn is None:
        return
    _lowering._lower_jaxpr_to_for_loop = new_fn
    _lowering._torchtpu_fori_patch_applied = True


_patch_jax_pallas_fori_lowering()
