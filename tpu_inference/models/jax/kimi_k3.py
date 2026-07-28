# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-Linear / Kimi-K3 JAX model (scaffolding).

Covers the Kimi-Linear model family (HF `moonshotai/Kimi-Linear-48B-A3B-
Instruct`, `moonshotai/Kimi-K3`): a hybrid of KDA linear attention (Kimi
Delta Attention, arXiv:2510.26692) and gated NoPE MLA full-attention
layers, with (for K3) latent-space MoE, SiTU activation, and attention
residuals. Reference implementations: the HF repos' modeling code and
vLLM PR #50000.

The registry entries are wired up first so the architecture names resolve;
the model implementation lands in follow-up changes.
"""


class KimiLinearForCausalLM:
    """Kimi-Linear-48B / Kimi-K3 text stack (KDA + NoPE MLA hybrid)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "[kimi-k3] KimiLinearForCausalLM JAX implementation is under "
            "development (KDA attention + hybrid KV cache pending).")


class KimiK3ForConditionalGeneration:
    """Kimi-K3 (text-only serving; vision tower weights are skipped)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "[kimi-k3] KimiK3ForConditionalGeneration JAX implementation is "
            "under development (text stack pending; vision unsupported).")
