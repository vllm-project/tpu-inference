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

import os


def stacked_rpa_bkv() -> int:
    return int(os.environ.get("STACKED_RPA_BKV", "0"))


def stacked_rpa_sw_bound() -> bool:
    return os.environ.get("STACKED_RPA_SW_BOUND", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
