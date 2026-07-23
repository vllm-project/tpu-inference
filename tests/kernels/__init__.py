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
import os

# MLA-kernel-only vmem budget. Two constraints (see #3011 / #3159):
# 1. Must run before libtpu/TPU backend initialization (module top,
#    before any jax import inside tests/kernels/).
# 2. Must NOT move into tests/conftest.py: a global setting changes XLA
#    codegen for every e2e test in the suite (that is exactly how #3011
#    caused a deterministic multimodal output drift and e2e slowdowns).
if "--xla_tpu_scoped_vmem_limit_kib" not in os.environ.get(
        "LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") +
                                      " --xla_tpu_scoped_vmem_limit_kib=65536")
