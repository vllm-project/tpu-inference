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

import jax.numpy as jnp
import numpy as np

from tpu_inference.layers.jax.sample.sampling import diffusion_commit


class TestDiffusionCommit:
    """Unit tests for the block-diffusion threshold-commit sampler.

    All cases run on CPU (pure JAX, no TPU kernels involved).
    """

    def test_threshold_and_progress_and_untouched(self):
        # Two rows, 3 positions, vocab size 4.
        # Row 0:
        #   pos0: peaked on token 2 -> top_prob ~0.9999 (> threshold), masked.
        #   pos1: mild on token 1  -> top_prob ~0.475 (< threshold), masked.
        #   pos2: peaked on token 0 -> top_prob ~0.980 but ALREADY committed
        #         (mask False) -> must stay committed and never be re-selected.
        # Row 1 (progress guarantee): nothing exceeds the threshold, so only
        #   the single highest-confidence masked position (pos1) may commit.
        logits = jnp.array(
            [
                [[0.0, 0.0, 10.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                 [5.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0],
                 [0.5, 0.0, 0.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        mask = jnp.array([[True, True, False], [True, True, True]])

        tokens, new_mask = diffusion_commit(logits,
                                            mask,
                                            threshold=0.9,
                                            temperature=0.0)
        tokens = np.array(tokens)
        new_mask = np.array(new_mask)

        # ---- Threshold commit correctness (row 0) ----
        # pos0 exceeds threshold and is masked -> commits (mask flips to False).
        assert new_mask[0, 0] == False  # noqa: E712
        # pos1 is below threshold and is not the forced position -> stays masked.
        assert new_mask[0, 1] == True  # noqa: E712
        # Committed token at pos0 is the argmax (token 2).
        assert tokens[0, 0] == 2

        # ---- Already-committed positions are untouched (row 0, pos2) ----
        # pos2 started committed (mask False); it stays committed and, despite
        # having a high top_prob, is never re-selected as the forced position.
        assert new_mask[0, 2] == False  # noqa: E712
        # Exactly one position newly commits in row 0 (only pos0).
        newly_committed_row0 = mask[0] & ~jnp.asarray(new_mask[0])
        assert int(np.array(newly_committed_row0).sum()) == 1

        # ---- Progress guarantee (row 1) ----
        # No position exceeds the threshold, yet exactly one commits: the
        # highest-confidence masked position, which is pos1 (top_prob ~0.711).
        assert new_mask[1, 0] == True  # noqa: E712
        assert new_mask[1, 1] == False  # noqa: E712
        assert new_mask[1, 2] == True  # noqa: E712
        assert tokens[1, 1] == 2
        newly_committed_row1 = mask[1] & ~jnp.asarray(new_mask[1])
        assert int(np.array(newly_committed_row1).sum()) == 1

    def test_new_mask_is_subset_of_old_mask(self):
        # The returned mask must only ever clear bits, never set them.
        logits = jnp.array(
            [[[0.0, 0.0, 3.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]
             ],
            dtype=jnp.float32,
        )
        mask = jnp.array([[True, True, True]])
        _, new_mask = diffusion_commit(logits, mask, threshold=0.5)
        new_mask = np.array(new_mask)
        old_mask = np.array(mask)
        # new_mask implies old_mask everywhere (strict subset of True entries).
        assert np.all(~new_mask | old_mask)
        # At least one position committed (progress guarantee).
        assert int(new_mask.sum()) < int(old_mask.sum())

    def test_fully_committed_row_forces_nothing(self):
        # A row with no masked positions must be left entirely untouched: the
        # progress guarantee must NOT force a commit when nothing is masked.
        logits = jnp.array([[[0.0, 0.0, 10.0, 0.0], [10.0, 0.0, 0.0, 0.0]]],
                           dtype=jnp.float32)
        mask = jnp.array([[False, False]])
        _, new_mask = diffusion_commit(logits, mask, threshold=0.9)
        new_mask = np.array(new_mask)
        assert new_mask[0, 0] == False  # noqa: E712
        assert new_mask[0, 1] == False  # noqa: E712

    def test_temperature_affects_threshold_commits(self):
        # Single row, two masked positions, threshold 0.9.
        #   posA: logits favor token 2, plain top_prob ~0.711.
        #   posB: logits favor token 1, plain top_prob ~0.870.
        logits = jnp.array([[[0.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]]],
                           dtype=jnp.float32)
        mask = jnp.array([[True, True]])

        # temperature=0.0 (raw logits): neither exceeds 0.9, so only the forced
        # (highest-confidence) position, posB, commits.
        _, nm_plain = diffusion_commit(logits,
                                       mask,
                                       threshold=0.9,
                                       temperature=0.0)
        nm_plain = np.array(nm_plain)
        assert nm_plain[0, 0] == True  # posA still masked  # noqa: E712
        assert nm_plain[0, 1] == False  # posB forced-committed  # noqa: E712
        committed_plain = int((np.array(mask)[0] & ~nm_plain[0]).sum())
        assert committed_plain == 1

        # temperature=0.5 sharpens (divide by 0.5 == multiply logits by 2), so
        # BOTH positions now exceed 0.9 and commit via the threshold path.
        _, nm_sharp = diffusion_commit(logits,
                                       mask,
                                       threshold=0.9,
                                       temperature=0.5)
        nm_sharp = np.array(nm_sharp)
        committed_sharp = int((np.array(mask)[0] & ~nm_sharp[0]).sum())
        assert committed_sharp == 2

    def test_accepts_2d_single_row(self):
        # A bare (L, V) input (single implicit row) is supported.
        logits = jnp.array([[0.0, 0.0, 5.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                           dtype=jnp.float32)
        mask = jnp.array([True, True])
        tokens, new_mask = diffusion_commit(logits, mask, threshold=0.9)
        tokens = np.array(tokens)
        new_mask = np.array(new_mask)
        # pos0 is peaked -> commits; pos1 mild and not forced -> stays masked.
        assert new_mask[0] == False  # noqa: E712
        assert new_mask[1] == True  # noqa: E712
        assert tokens[0] == 2
