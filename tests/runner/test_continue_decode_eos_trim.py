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
    window = rng.integers(0, 12, size=(64, 33), dtype=np.int32)
    lengths, has_eos = _extract_valid_lengths_host(window, eos)
    assert lengths.shape == (64, ) and has_eos.shape == (64, )
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


def test_eos_in_last_column_counts_as_a_hit():
    window = np.array([[1, 2, 7], [7, 2, 3]], dtype=np.int32)
    lengths, has_eos = _extract_valid_lengths_host(window, 7)
    assert lengths.tolist() == [3, 1]
    assert has_eos.tolist() == [True, True]


def test_vectorized_trim_empty_window():
    lengths, has_eos = _extract_valid_lengths_host(
        np.zeros((0, 8), dtype=np.int32), 7)
    assert lengths.shape == (0, ) and has_eos.shape == (0, )
