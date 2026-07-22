# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import pathlib
import sys


def _load_batch_module():
    path = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
            "runner" / "diffusion" / "batch.py")
    spec = importlib.util.spec_from_file_location("diffusion_batch_under_test",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


batch = _load_batch_module()


def test_aligned_prompt_is_split_into_complete_blocks():
    plan = batch.plan_seeded_prompt(list(range(8)), 4, 99)

    assert plan.full_blocks == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert plan.partial_canvas is None
    assert plan.remainder_size == 0


def test_prompt_remainder_is_preserved_in_first_canvas():
    plan = batch.plan_seeded_prompt(list(range(6)), 4, 99)

    assert plan.full_blocks == ((0, 1, 2, 3), )
    assert plan.partial_canvas == (4, 5, 99, 99)
    assert plan.partial_mask == (False, False, True, True)
    assert plan.remainder_size == 2


def test_partial_block_is_flushed_at_the_next_scheduler_boundary():
    first, pending = batch.start_partial_block_output([4, 5, 6, 7], 2, 8)

    assert first == [6]
    assert batch.flush_partial_block_output(pending) == [7, 8]


def test_seeded_decode_emits_new_canvas_tokens_and_uncached_anchor():
    emitted = batch.complete_seeded_decode_block([10, 11, 12, 13], 14)

    assert emitted == [11, 12, 13, 14]


def test_cache_capacity_accounts_for_the_uncached_final_anchor():
    assert batch.required_cache_end(32, 32, 32) == 64
    assert batch.required_cache_end(33, 32, 32) == 64
    assert batch.required_cache_end(33, 33, 32) == 96
