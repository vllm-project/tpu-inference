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
"""Experimental small-batch FP8 MoE that streams hit experts' weight units."""

from tpu_inference.kernels.experimental.moe_lowtok.kernel import (
    HOT_SLOTS, MOE_LOWTOK_MAX_TOKENS, can_use_moe_lowtok, hot_onehot,
    hot_tables, moe_lowtok, pack_tables, routing_table, slot_table,
    table_offsets)

__all__ = [
    "HOT_SLOTS",
    "MOE_LOWTOK_MAX_TOKENS",
    "can_use_moe_lowtok",
    "hot_onehot",
    "hot_tables",
    "moe_lowtok",
    "pack_tables",
    "routing_table",
    "slot_table",
    "table_offsets",
]
