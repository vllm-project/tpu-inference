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
"""Full fused EP MoE kernel.

Exposes ``fused_moe_func_rs``: EP MoE with ICI reduce-scatter fused into a
single ``gmm_fused_rs`` Pallas kernel (gather -> GMM1 -> act -> GMM2 -> RS).
"""

from .fused_moe_rs import (expert_parallel_gmm_rs, fused_moe_func_rs,
                           moe_gmm_local_rs_nodedup)

__all__ = [
    "fused_moe_func_rs",
    "expert_parallel_gmm_rs",
    "moe_gmm_local_rs_nodedup",
]
