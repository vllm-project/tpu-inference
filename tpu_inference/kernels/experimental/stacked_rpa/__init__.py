# Copyright (c) Meta Platforms, Inc. and affiliates.
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
"""Stacked RPA kernel implementation.

The public entry points live in ``wrapper``. Decode's sliding-window and global
packages, plus the prefill package, each own their schedule, buffered refs, and
kernel implementation. Only FlashAttention math and low-level layout utilities
are shared beneath those concrete paths.
"""
