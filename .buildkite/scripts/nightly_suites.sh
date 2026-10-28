#!/bin/bash
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

# The suites a nightly build runs for its MODEL_IMPL_TYPE, as folders under
# .buildkite/. upload_models_and_features.sh walks them on bare metal and
# bootstrap.sh uploads them on kube, so both fleets run the same suites.
# kernel_microbenchmarks stands for each of its subfolders.
nightly_suites() {
  case "${MODEL_IMPL_TYPE:-auto}" in
    auto) echo "parallelism models features rl kernel_microbenchmarks" ;;
    flax_nnx) echo "quantization parallelism features" ;;
    vllm) echo "quantization parallelism models features" ;;
  esac
}
