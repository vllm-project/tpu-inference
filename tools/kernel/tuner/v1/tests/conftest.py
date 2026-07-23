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
"""
Test configuration for kernel tuner runner tests.
Prevents absl.flags.DuplicateFlagError when multiple kernel tuner modules
define the same command-line flags (e.g., gcp_project_id) and are collected
within the same pytest process.
"""

from absl.flags import _defines, _flagvalues

_orig_define_flag = _defines.DEFINE_flag


def _safe_define_flag(flag, flag_values, *args, **kwargs):
    if flag.name in flag_values:
        existing_flag = flag_values[flag.name]
        try:
            return _flagvalues.FlagHolder(flag_values, existing_flag,
                                          flag_values._check_writable)
        except (TypeError, AttributeError):
            try:
                return _flagvalues.FlagHolder(flag_values, existing_flag)
            except Exception:
                return existing_flag
    return _orig_define_flag(flag, flag_values, *args, **kwargs)


_defines.DEFINE_flag = _safe_define_flag
