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
"""GDN v1 kernels.

The v1 kernel implementation uses recurrent Pallas kernels for both prefill and decode:

- **Decoding Kernel (`fused_gdn_decode_kernel.py`)**:
  Implements `fused_decoding_gdn` for single-step decode. It processes `bt` decode
  tokens per pipeline step using Pallas's `emit_pipeline` for q/k/v/g/b tiling.
  Uses double-buffered VMEM scratch space and asynchronous DMA copies to
  pipeline load/store states from HBM in bulk using `state_indices`.

- **Recurrent Kernel (`fused_gdn_recurrent_kernel.py`)**:
  Implements `fused_recurrent_gdn` for prefill. It Uses a pre-computed sequence block
  metadata array to map block IDs to sequence indices and token offsets,
  handling sequence boundaries dynamically. Then it executes a per-token recurrent loop 
  for state and output computation
"""
