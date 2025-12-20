# Copyright 2025 Google LLC
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

UNQUANTIZED = "unquantized"
MXFP4 = "mxfp4"
AWQ = "awq"
COMPRESSED_TENSORS = "compressed-tensors"
FP8 = "fp8"


def get_tpu_quant_method(quant_method: str) -> str:
    return "tpu-" + quant_method
