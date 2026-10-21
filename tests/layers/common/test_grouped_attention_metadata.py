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
"""GroupedAttentionMetadata: dict semantics, and one jit parameter per group."""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, GroupedAttentionMetadata)

GROUP_0 = ("layers.0.attn", "layers.1.attn", "layers.2.attn")
GROUP_1 = ("layers.0.attn.compressor", )


def _metadata(seed: int) -> AttentionMetadata:
    return AttentionMetadata(
        input_positions=jnp.arange(8, dtype=jnp.int32),
        block_tables=jnp.arange(16, dtype=jnp.int32) + seed,
        seq_lens=jnp.ones(4, jnp.int32),
        query_start_loc=jnp.arange(5, dtype=jnp.int32),
        request_distribution=jnp.array([1, 1, 4], jnp.int32),
    )


def _build() -> GroupedAttentionMetadata:
    return GroupedAttentionMetadata(
        groups=(_metadata(0), _metadata(100)),
        layer_names_per_group=(GROUP_0, GROUP_1),
    )


class GroupedAttentionMetadataTest(absltest.TestCase):

    def test_behaves_like_the_per_layer_dict_it_replaces(self):
        md = _build()
        # Consumers do `isinstance(attn_metadata, dict)` and index by layer.
        self.assertIsInstance(md, dict)
        self.assertEqual(set(md), set(GROUP_0) | set(GROUP_1))
        self.assertLen(md, len(GROUP_0) + len(GROUP_1))
        self.assertIsNotNone(md[GROUP_0[0]].block_tables)
        self.assertIsNotNone(next(iter(md.values())).block_tables)

    def test_layers_of_a_group_share_one_object(self):
        md = _build()
        for name in GROUP_0[1:]:
            self.assertIs(md[name], md[GROUP_0[0]])
        self.assertIsNot(md[GROUP_1[0]], md[GROUP_0[0]])

    def test_flattens_once_per_group_not_once_per_layer(self):
        md = _build()
        per_group = len(jax.tree_util.tree_leaves(_metadata(0)))
        self.assertLen(jax.tree_util.tree_leaves(md), 2 * per_group)

    def test_work_derived_per_layer_is_emitted_once_per_group(self):
        """The point of the class: one HLO parameter per group, so XLA shares."""

        def work(md):
            acc = jnp.zeros((), jnp.int32)
            for i, name in enumerate(GROUP_0):
                bt = md[name].block_tables
                # A gather off the block table, as the attention/compressor
                # paths do, with a different consumer per layer.
                acc += jnp.sum(bt[md[name].input_positions] * (i + 1))
            return acc

        grouped = _build()
        plain = {name: value for name, value in grouped.items()}

        def gathers(arg):
            text = jax.jit(work).lower(arg).compile().as_text()
            return sum(1 for line in text.splitlines() if " gather(" in line)

        self.assertEqual(jax.jit(work)(grouped), jax.jit(work)(plain))
        # The plain dict re-gathers per layer; the grouped one gathers once.
        self.assertLess(gathers(grouped), gathers(plain))
        self.assertEqual(gathers(grouped), 1)


if __name__ == "__main__":
    absltest.main()
