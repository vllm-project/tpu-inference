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
"""Fused expert-parallel MoE kernel: the fused_ep_moe_v2 entry point, the
layout constants a caller must match, and the VMEM estimator and limit.
"""

from tpu_inference.kernels.fused_moe.v2 import host, kernel
from tpu_inference.kernels.fused_moe.v2.layer import fused_ep_moe_v2

# The layout constants a caller has to match, and the VMEM accounting,
# named here so a caller binds against the package rather than reaching
# into the module each one lives in.
ACT_FNS = kernel.ACT_FNS
ALIGNMENT_SLOT_FIELD = host.ALIGNMENT_SLOT_FIELD
AXIS = kernel.AXIS
PACK4 = host.PACK4
HIDDEN_LANE_BLOCK = host.HIDDEN_LANE_BLOCK
HIDDEN_MAX_BLOCKS = host.HIDDEN_MAX_BLOCKS
MIN_GENERATION = host.MIN_GENERATION
NBUF = host.NBUF
ROWBLK = host.ROWBLK
U32_SUBLANE_TILE = host.U32_SUBLANE_TILE
WIDEN_KCHUNK = host.WIDEN_KCHUNK
chip_generation = host.chip_generation
# The weight formats the kernel takes, the accepted (weight dtype, scale
# layout) pairs they stand for, and the two lookups the serving adapter
# answers a caller's weights with.
WeightFormat = host.WeightFormat
WEIGHT_FORMS = host.WEIGHT_FORMS
WEIGHT_FORMAT_NAMES = host.WEIGHT_FORMAT_NAMES
weight_form = host.weight_form
weight_format_of_dtype = host.weight_format_of_dtype
routing_block = host.routing_block
ragged_stride_bound = host.ragged_stride_bound
vmem_estimate_bytes = host.vmem_estimate_bytes
vmem_limit = host.vmem_limit

# The declared surface: the entry point, what the kernel takes, and the
# accounting a caller needs to predict whether a layer fits. The layout
# constants above stay reachable as module attributes for the in-tree
# serving adapter, which has to predict this kernel's geometry exactly, but
# they are the part most likely to change and they are not API: a caller
# outside this repository binds to the five names below.
__all__ = [
    "WEIGHT_FORMS", "WeightFormat", "fused_ep_moe_v2", "vmem_estimate_bytes",
    "vmem_limit"
]
