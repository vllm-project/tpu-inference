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
"""LLM-driven mutation: client backends, prompts, diff applier, critic."""

from tools.kernel.evolve.mutator.bon import BestOfNMutator, BonCandidate
from tools.kernel.evolve.mutator.critic import Critique, critique_diff
from tools.kernel.evolve.mutator.diff_applier import (DiffResult, apply_diff,
                                                      extract_diff,
                                                      validate_python)
from tools.kernel.evolve.mutator.example_pool import (
    ExamplePool, PositiveExample, render_examples_for_prompt)
from tools.kernel.evolve.mutator.llm_client import (AnthropicClient, LLMClient,
                                                    StubClient)
from tools.kernel.evolve.mutator.local_llm import LocalLlmClient
from tools.kernel.evolve.mutator.programmatic import (LineRewriteRule,
                                                      LiteralRewriteRule,
                                                      ProgrammaticMutator)
from tools.kernel.evolve.mutator.prompts import (build_critic_prompts,
                                                 build_mutation_prompts)
from tools.kernel.evolve.mutator.vertex_anthropic import VertexAnthropicClient

__all__ = [
    "AnthropicClient",
    "BestOfNMutator",
    "BonCandidate",
    "Critique",
    "DiffResult",
    "ExamplePool",
    "LineRewriteRule",
    "LiteralRewriteRule",
    "LLMClient",
    "LocalLlmClient",
    "PositiveExample",
    "ProgrammaticMutator",
    "StubClient",
    "VertexAnthropicClient",
    "apply_diff",
    "build_critic_prompts",
    "build_mutation_prompts",
    "critique_diff",
    "extract_diff",
    "render_examples_for_prompt",
    "validate_python",
]
