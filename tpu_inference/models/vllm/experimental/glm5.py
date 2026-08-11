# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU-compatible GLM-5.2 (``glm_moe_dsa``) model.

Unlike DeepSeek-V4 (``deepseek_v4.py`` in this package), GLM-5.2's upstream
vLLM model (``GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)``,
``vllm/model_executor/models/deepseek_v2.py``) is *not* independently
CUDA-bound: it reuses the shared ``DeepseekV2Model``/``DeepseekV2DecoderLayer``
/``DeepseekV2MoE``/``DeepseekV2MLP`` skeleton, whose Linear/Norm/MoE/Embedding
sub-layers are already TPU-compatible via this repo's existing generic OOT
registrations (``layers/vllm/custom_ops/{linear,embedding,fused_moe}.py``),
and whose MLA attention (``DeepseekV2MLAAttention``) delegates to
``MultiHeadLatentAttentionWrapper`` -- a vLLM ``PluggableLayer`` already
OOT-registered generically for *all* MLA models by
``layers/vllm/custom_ops/mla_attention.py::VllmMultiHeadLatentAttentionWrapper``
(dense MLA attention "just works" without any GLM-5.2-specific model class).

The one piece that is not independently OOT-swappable is ``Indexer``
(``deepseek_v2.py``, the DSA lightning indexer): it is instantiated by a
hardcoded module-level class reference inside
``DeepseekV2MLAAttention.__init__``, not via any ``CustomOp``/
``PluggableLayer`` registry hook, and its ``forward()`` calls CUDA/Triton-only
ops (``per_token_group_quant_fp8``). This thin subclass exists solely to
temporarily rebind that one symbol -- the same "monkeypatch a module-level
class reference for the duration of ``super().__init__()``" idiom
``deepseek_v4_attention.py``/``deepseek_v4_indexer.py`` use for DeepSeek-V4's
CUDA-bound pieces -- to
``tpu_inference.layers.vllm.custom_ops.experimental.glm5.glm5_indexer.VllmGlm5Indexer``,
which replaces the indexer's Q/K RoPE+quant+top-k pipeline with the Phase 2
GLM-5.2 Pallas kernels. Everything else in the model tree is constructed,
unmodified, from vLLM's own ``GlmMoeDsaForCausalLM``/``DeepseekV2*`` classes.

Two additional, genuinely-generic (not GLM-5.2-specific) gaps surfaced by
live smoke tests on the TPU VM, both in ``DeepseekV2Model.__init__``'s
``topk_indices_buffer`` allocation:

1. ``torch.empty(..., device=current_platform.device_type)`` -- literally
   ``device="tpu"`` -- crashes (``RuntimeError: ... at start of device
   string: tpu``): plain ``torch.empty`` doesn't recognize "tpu" as a valid
   device string in this environment (the ambient ``torch.utils._device``
   default-device context active during vLLM model construction passes the
   explicit device straight through to real PyTorch, bypassing torchax's
   own factory-function interception). Worked around, like
   ``deepseek_v4_attention.py::VllmDeepseekV4MLAAttention.__init__`` does
   for DeepSeek-V4's CUDA-event allocations, by forcing
   ``current_platform.device_type`` to ``"cpu"`` for construction.
2. **That fix alone is not sufficient and was confirmed wrong by a second
   live crash**: forcing ``device="cpu"`` makes ``torch.empty(...)``
   allocate a genuine *plain* (non-torchax) CPU ``torch.Tensor`` -- fine for
   shape/dtype bookkeeping, but incompatible with actually reading/writing
   it via JAX (``AttributeError: 'Tensor' object has no attribute '_elem'``
   from ``torchax``'s ``_aten_copy``/``type_as``, which assume a proper
   ``torchax.tensor.Tensor`` on both sides of any op touching it). Unlike
   DeepSeek-V4 (whose own analogous buffer is allocated the same way but
   never actually read/written via torch ops -- its indexer threads
   ``topk_indices`` through a return value instead, sidestepping the
   buffer entirely), GLM-5.2 relies on vLLM's *generic*
   ``MultiHeadLatentAttentionWrapper.forward()``, which calls
   ``self.indexer(...)`` purely for its side effect on
   ``topk_indices_buffer`` (discarding the return value) -- so this buffer
   genuinely needs to be a real, on-device, torchax-managed tensor.
   Fixed by replacing the plain-CPU buffer, right after construction, with
   one properly built via ``torch_view(jnp.zeros(...))`` (this repo's
   established JAX-array-to-torchax-tensor construction, used throughout
   the OOT custom-op layers), and propagating that replacement to every
   module that captured a reference to the original (wrong) object at
   construction time -- the ``DeepseekV2Model`` instance itself, every
   ``Indexer``, and every ``MLAAttentionImpl`` (not an ``nn.Module``, so not
   reachable via plain ``nn.Module.modules()`` -- reached instead via each
   ``MLAAttention`` module's ``.impl`` attribute).
"""
from unittest.mock import patch

import jax.numpy as jnp
import torch
from torchax.interop import torch_view
from torchax.ops.mappings import t2j_dtype
from vllm.config import VllmConfig
from vllm.model_executor.models import deepseek_v2
from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
from vllm.platforms import current_platform

from tpu_inference.layers.vllm.custom_ops.experimental.glm5.glm5_indexer import \
    VllmGlm5Indexer
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _fix_topk_indices_buffer(model: torch.nn.Module) -> None:
    """Replaces every module's ``topk_indices_buffer`` reference (a plain
    CPU tensor after construction, see module docstring gap (2)) with one
    properly constructed via ``torch_view(jnp.zeros(...))``, propagated to
    every holder captured at construction time.

    NOTE: ``DeepseekV2Model.__init__`` allocates ``topk_indices_buffer`` as
    a *local variable*, threaded down to each layer's constructor -- it is
    never stored as ``self.topk_indices_buffer`` on the ``DeepseekV2Model``
    instance itself (confirmed by reading the source after an initial
    version of this function silently no-op'd looking for it there: its own
    diagnostic log line never fired on a live run). The only holders are
    each ``Indexer``/``VllmGlm5Indexer`` instance and each
    ``MLAAttentionImpl`` (via ``MLAAttention.impl``) -- so the original
    buffer (for its shape/dtype) must be found by scanning those, not by
    reading it off the ``Model``.
    """
    old_buffer = None
    for module in model.modules():
        candidate = getattr(module, "topk_indices_buffer", None)
        if candidate is not None:
            old_buffer = candidate
            break
        impl = getattr(module, "impl", None)
        if impl is not None:
            candidate = getattr(impl, "topk_indices_buffer", None)
            if candidate is not None:
                old_buffer = candidate
                break
    if old_buffer is None:
        # Not a DSA model (`config.index_topk` absent) -- nothing to fix.
        return

    new_buffer = torch_view(
        jnp.zeros(tuple(old_buffer.shape),
                  dtype=t2j_dtype(old_buffer.dtype)))

    num_indexers = 0
    num_impls = 0
    for module in model.modules():
        if getattr(module, "topk_indices_buffer", None) is not None:
            module.topk_indices_buffer = new_buffer
            num_indexers += 1
        impl = getattr(module, "impl", None)
        if impl is not None and getattr(impl, "topk_indices_buffer",
                                        None) is not None:
            impl.topk_indices_buffer = new_buffer
            num_impls += 1
    if num_indexers == 0 and num_impls == 0:
        raise RuntimeError(
            "[glm5] Found an original topk_indices_buffer but replaced it "
            "on zero holders -- the holder-scan logic no longer matches "
            "how vLLM threads this buffer through the model tree. Fix "
            "_fix_topk_indices_buffer before proceeding (this must not "
            "silently no-op again).")
    logger.info(
        "[glm5] Replaced topk_indices_buffer (plain CPU tensor -> "
        "torchax tensor, shape=%s) on %d Indexer holder(s) and %d "
        "MLAAttentionImpl holder(s).", tuple(new_buffer.shape),
        num_indexers, num_impls)


class Glm5ForCausalLM(GlmMoeDsaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        orig_indexer = deepseek_v2.Indexer
        deepseek_v2.Indexer = VllmGlm5Indexer
        try:
            # See module docstring gap (1): `DeepseekV2Model.__init__`'s
            # `topk_indices_buffer = torch.empty(..., device=current_platform
            # .device_type)` fails on TPU outside a torchax device context
            # (`device="tpu"` is not a torch-recognized device string).
            # Force "cpu" for construction only, matching
            # `VllmDeepseekV4MLAAttention.__init__`'s identical workaround
            # (this avoids the crash; gap (2) below then fixes the
            # resulting plain-CPU-tensor's actual usability).
            with patch.object(current_platform, "device_type", "cpu"):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            deepseek_v2.Indexer = orig_indexer

        # See module docstring gap (2).
        _fix_topk_indices_buffer(self)
