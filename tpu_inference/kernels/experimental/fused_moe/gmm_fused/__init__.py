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
"""Per-round-push AG+GMM1 EP MoE kernel (example).

Provides a push-based all-gather fused with GMM1 + activation
(``gmm_v2_ag_gmm1``), driven by a precomputed per-round send schedule
(``per_round_schedule``), plus the paired GMM2 + ICI reduce-scatter kernels.
"""

from .gmm_v2_ag_rs import (gmm_v2_ag_gmm1, gmm_v2_scatter_ici_dedup,
                           gmm_v2_scatter_ici_nodedup)

__all__ = [
    "gmm_v2_ag_gmm1",
    "gmm_v2_scatter_ici_dedup",
    "gmm_v2_scatter_ici_nodedup",
]
