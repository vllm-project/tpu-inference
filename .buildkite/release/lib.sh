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

# Shared by the tpu-inference-release steps. Sourced, not run.

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# The version every step of the build agrees on, set once by preflight.sh.
release_version() {
  buildkite-agent meta-data get release-version
}

# a, b, rc and dev versions are pre-releases.
is_prerelease() {
  [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}
