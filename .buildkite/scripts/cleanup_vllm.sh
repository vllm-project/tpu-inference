#!/bin/bash
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


# shellcheck disable=all
set -e

echo "Cleaning up any running vLLM instances..."
pkill -f "vllm" || true
pkill -f "toy_proxy_server" || true
sleep 5
pkill -9 -f "vllm" || true
pkill -9 -f "toy_proxy_server" || true
fuser -k -9 /dev/vfio/* || true
fuser -k -9 /dev/accel* || true
