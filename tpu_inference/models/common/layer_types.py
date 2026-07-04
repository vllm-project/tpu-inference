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
"""Shared string constants for HF `text_config.layer_types` entries.

These values come from the HF model configs and are matched in several
places (kv cache spec derivation, jit out-sharding selection, and the model
decoder layer). Keeping them in one module avoids a rename or typo silently
desyncing those sites, which would build a MambaSpec in one place and a
plain attention sharding in another for the same layer."""

FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
SLIDING_ATTENTION = "sliding_attention"
