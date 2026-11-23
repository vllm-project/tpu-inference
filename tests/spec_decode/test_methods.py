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

from types import SimpleNamespace

import pytest

from tpu_inference.spec_decode.methods import uses_eagle, uses_gemma4_mtp


def _spec_config(method, draft_model_type=None):
    draft_model_config = None
    if draft_model_type is not None:
        draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(
            model_type=draft_model_type))
    return SimpleNamespace(method=method,
                           draft_model_config=draft_model_config)


@pytest.mark.parametrize("method, expected", [
    ("eagle", True),
    ("eagle3", True),
    ("mtp", True),
    ("dflash", True),
    ("dspark", True),
    ("ngram", False),
    ("draft_model", False),
    ("suffix", False),
])
def test_uses_eagle(method, expected):
    assert uses_eagle(_spec_config(method)) is expected


@pytest.mark.parametrize("method, draft_model_type, expected", [
    ("mtp", "gemma4_mtp", True),
    ("mtp", "deepseek_mtp", False),
    ("mtp", None, False),
    ("eagle3", "gemma4_mtp", False),
])
def test_uses_gemma4_mtp(method, draft_model_type, expected):
    assert uses_gemma4_mtp(_spec_config(method, draft_model_type)) is expected
