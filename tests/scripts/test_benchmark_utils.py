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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BENCHMARKING_DIR = (Path(__file__).parents[2] / "scripts" / "vllm" /
                     "benchmarking")
# benchmark_utils imports its siblings (backend_request_func,
# benchmark_dataset) by bare name, the way benchmark_serving.py runs it.
sys.path.insert(0, str(_BENCHMARKING_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "benchmark_utils", _BENCHMARKING_DIR / "benchmark_utils.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

strip_reasoning = _MODULE.strip_reasoning

# postprocess_text_mmlu encodes "no answer could be read" as the index of None.
_UNPARSED = _MODULE._MMLU_CHOICES.index(None)


def _output(generated_text: str, completion: str = "D", success: bool = True):
    """A stand-in for RequestFuncOutput with only the fields the evals read."""
    return types.SimpleNamespace(
        success=success,
        generated_text=generated_text,
        input_request=types.SimpleNamespace(completion=completion),
    )


# --------------------------------------------------------------------------
# strip_reasoning
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning"])
def test_strip_reasoning_removes_closed_block(tag):
    text = f"<{tag}>I lean towards (B)</{tag}>Answer: (D)"
    assert strip_reasoning(text) == "Answer: (D)"


@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning"])
def test_strip_reasoning_is_case_insensitive(tag):
    text = f"<{tag.upper()}>maybe (B)</{tag.upper()}>Answer: (D)"
    assert strip_reasoning(text) == "Answer: (D)"


def test_strip_reasoning_spans_newlines():
    text = "<think>line one\nline two mentions (B)\n</think>\nAnswer: (D)"
    assert strip_reasoning(text) == "Answer: (D)"


def test_strip_reasoning_keeps_text_without_tags():
    text = "The answer is (D) because of the second law."
    assert strip_reasoning(text) == text


def test_strip_reasoning_handles_missing_open_tag():
    # Some serving stacks pre-fill the opening tag in the chat template, so it
    # never appears in the completion -- only the close tag does.
    assert strip_reasoning("weighed (B)</think>Answer: (D)") == "Answer: (D)"


def test_strip_reasoning_strips_through_last_close_tag():
    text = ("<think>first (A)</think>interlude (B)"
            "<think>second (C)</think>Answer: (D)")
    assert strip_reasoning(text) == "Answer: (D)"


def test_strip_reasoning_greedy_past_quoted_close_tag():
    # Documented trade-off: a trace that quotes a close tag must not end the
    # block early, so the last one in the text wins.
    text = "<think>the model writes </think> in its trace</think>Answer: (D)"
    assert strip_reasoning(text) == "Answer: (D)"


def test_strip_reasoning_drops_leading_whitespace_after_block():
    assert strip_reasoning(
        "<think>x</think>  \n\t Answer: (D)") == "Answer: (D)"


def test_strip_reasoning_preserves_trailing_whitespace():
    assert strip_reasoning("<think>x</think>Answer: (D)\n") == "Answer: (D)\n"


def test_strip_reasoning_truncated_generation_returns_empty():
    # An open tag with no close means the generation ran out of budget
    # mid-thought. There is no answer in it to grade.
    assert strip_reasoning("<think>Let me consider (B), but wait") == ""


def test_strip_reasoning_empty_block_returns_empty():
    assert strip_reasoning("<think>reasoned but never answered</think>") == ""


def test_strip_reasoning_passes_through_empty_string():
    assert strip_reasoning("") == ""


# --------------------------------------------------------------------------
# extract_abcd_gpqa
# --------------------------------------------------------------------------


def test_gpqa_scores_answer_not_reasoning():
    # The reasoning states an answer in the same format the extractor looks
    # for; only the one after the close tag should count.
    text = ("<think>Answer: (B)? No, that ignores the catalyst.</think>"
            "Answer: (D)")
    assert _MODULE.extract_abcd_gpqa(text) == "D"


def test_gpqa_truncated_reasoning_is_unparsed():
    assert _MODULE.extract_abcd_gpqa("<think>I think (B) because") is None


def test_gpqa_empty_response_is_unparsed():
    # Regression: `"" in "ABCD"` is True, so the last-resort branch used to
    # report an empty response as the answer `""`.
    assert _MODULE.extract_abcd_gpqa("") is None
    assert _MODULE.extract_abcd_gpqa("   \n ") is None


def test_gpqa_unchanged_without_reasoning_tags():
    assert _MODULE.extract_abcd_gpqa("Answer: C") == "C"
    assert _MODULE.extract_abcd_gpqa("The answer is (A)") == "A"


# --------------------------------------------------------------------------
# postprocess_text_mmlu
# --------------------------------------------------------------------------


def test_mmlu_scores_answer_not_reasoning():
    preds, targets = _MODULE.postprocess_text_mmlu(
        ["<think>maybe (B)</think>Answer: (D)"], ["D"])
    assert preds == targets


def test_mmlu_truncated_reasoning_is_unparsed():
    preds, _ = _MODULE.postprocess_text_mmlu(["<think>weighing (B) against"],
                                             ["D"])
    assert preds == [_UNPARSED]


def test_mmlu_unchanged_without_reasoning_tags():
    preds, targets = _MODULE.postprocess_text_mmlu(["Thus answer: C"], ["C"])
    assert preds == targets


# --------------------------------------------------------------------------
# unparsed reporting
# --------------------------------------------------------------------------


def test_eval_accuracy_gpqa_reports_unparsed():
    outputs = [
        _output("<think>(B)?</think>Answer: (D)", completion="D"),  # correct
        _output("Answer: (A)", completion="D"),  # wrong
        _output("<think>truncated", completion="D"),  # unparsed
        _output("", completion="D"),  # unparsed
    ]
    result = _MODULE.eval_accuracy_gpqa(outputs)

    assert result["correct"] == 1
    assert result["total"] == 4
    assert result["unparsed"] == 2
    assert result["unparsed_rate"] == 0.5
    assert result["accuracy"] == 0.25


def test_eval_accuracy_gpqa_skips_failed_requests():
    outputs = [
        _output("Answer: (D)", completion="D"),
        _output("<think>truncated", completion="D", success=False),
    ]
    result = _MODULE.eval_accuracy_gpqa(outputs)

    assert result["total"] == 1
    assert result["unparsed"] == 0
    assert result["gen_num"] == 2


def test_eval_accuracy_gpqa_no_successful_outputs():
    result = _MODULE.eval_accuracy_gpqa(
        [_output("Answer: (D)", success=False)])

    assert result["total"] == 0
    assert result["unparsed_rate"] == 0.0
    assert result["accuracy"] == 0.0


def test_eval_accuracy_mmlu_reports_unparsed(monkeypatch):
    # eval_accuracy_mmlu pulls the accuracy metric over the network and
    # downloads nltk corpora; neither is what is under test here.
    monkeypatch.setattr(
        _MODULE, "evaluate",
        types.SimpleNamespace(load=lambda name: types.SimpleNamespace(
            compute=lambda predictions, references: {
                "accuracy":
                sum(p == r for p, r in zip(predictions, references)) / len(
                    predictions)
            })))
    monkeypatch.setattr(_MODULE, "nltk",
                        types.SimpleNamespace(download=lambda name: None))

    outputs = [
        _output("<think>(B)?</think>Answer: (D)", completion="D"),  # correct
        _output("Answer: (A)", completion="D"),  # wrong
        _output("<think>truncated", completion="D"),  # unparsed
    ]
    result = _MODULE.eval_accuracy_mmlu(outputs)

    assert result["unparsed"] == 1
    assert result["unparsed_rate"] == 0.3333
    assert result["gen_num"] == 3
    assert result["accuracy"] == 0.3333
