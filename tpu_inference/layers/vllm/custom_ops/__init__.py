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

from tpu_inference.layers.vllm.custom_ops import embedding as embedding
from tpu_inference.layers.vllm.custom_ops import fused_moe as fused_moe
from tpu_inference.layers.vllm.custom_ops import linear as linear
from tpu_inference.layers.vllm.custom_ops import mla_attention as mla_attention

# Register custom op to vLLM so that vLLM model implementation will instantiante
# classes with definitions in tpu-inference.
