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

import numpy as np
import pytest

from tpu_inference.runner.tpu_runner import _extract_valid_lengths_host


def _trim_row(tokens: np.ndarray, eos) -> list[int]:
    """Reference per-row trim: up to and including the first EOS token."""
    eos_indices = np.where(np.isin(tokens, eos))[0]
    if len(eos_indices) > 0:
        return tokens[:eos_indices[0] + 1].tolist()
    return tokens.tolist()


@pytest.mark.parametrize("eos", [7, [7, 9], np.array([7, 9])])
def test_vectorized_trim_matches_per_row_reference(eos):
    rng = np.random.default_rng(0)
    window = rng.integers(0, 12, size=(256, 65), dtype=np.int32)
    lengths, has_eos = _extract_valid_lengths_host(window, eos)
    assert lengths.shape == (256, ) and has_eos.shape == (256, )
    assert lengths.dtype == np.int64 and has_eos.dtype == bool
    for i in range(window.shape[0]):
        expected = _trim_row(window[i], eos)
        assert window[i, :lengths[i]].tolist() == expected
        assert bool(has_eos[i]) == bool(
            np.isin(window[i], np.atleast_1d(eos)).any())


def test_rows_without_eos_keep_the_full_window():
    window = np.full((3, 5), 1, dtype=np.int32)
    lengths, has_eos = _extract_valid_lengths_host(window, 7)
    assert lengths.tolist() == [5, 5, 5]
    assert not has_eos.any()


def test_eos_in_first_and_last_column():
    window = np.array([[1, 2, 7], [7, 2, 3]], dtype=np.int32)
    lengths, has_eos = _extract_valid_lengths_host(window, 7)
    assert lengths.tolist() == [3, 1]
    assert has_eos.tolist() == [True, True]


def test_only_the_first_eos_counts():
    window = np.array([[1, 7, 3, 7, 5]], dtype=np.int32)
    lengths, has_eos = _extract_valid_lengths_host(window, 7)
    assert lengths.tolist() == [2]
    assert has_eos.tolist() == [True]


def test_no_rows():
    lengths, has_eos = _extract_valid_lengths_host(
        np.zeros((0, 8), dtype=np.int32), 7)
    assert lengths.shape == (0, ) and has_eos.shape == (0, )
    assert lengths.dtype == np.int64 and has_eos.dtype == bool


def test_zero_width_window():
    # A window with rows but no steps must not run argmax over an empty axis.
    lengths, has_eos = _extract_valid_lengths_host(
        np.zeros((4, 0), dtype=np.int32), 7)
    assert lengths.tolist() == [0, 0, 0, 0]
    assert not has_eos.any()
