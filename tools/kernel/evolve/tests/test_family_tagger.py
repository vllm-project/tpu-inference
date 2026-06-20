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
"""Tests for ``family_tagger``."""

from tools.kernel.evolve.family_tagger import describe_tags, tag_diff


def test_tag_dtype_cast_is_family_h():
    """The +4.7% RPA v3 win — `astype(out_dtype)` on a reduction."""
    diff = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n"
        "-p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=out_dtype)\n"
        "+p_rowsum = jnp.sum(p, axis=1, keepdims=True).astype(out_dtype)\n")
    tags = tag_diff(diff)
    assert "H" in tags


def test_tag_block_size_is_family_e():
    diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n"
            "-BLOCK_M = 128\n"
            "+BLOCK_M = 64\n")
    tags = tag_diff(diff)
    assert "E" in tags


def test_tag_collective_is_family_f():
    diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
            "-out = psum(x, axis_name='model')\n"
            "+out = psum_scatter(x, axis_name='model', scatter_dimension=0)\n")
    tags = tag_diff(diff)
    assert "F" in tags


def test_tag_donate_is_family_a():
    diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
            "-@jax.jit\n"
            "+@jax.jit(donate_argnames=('state',))\n")
    tags = tag_diff(diff)
    assert "A" in tags


def test_tag_pretree_flatten_is_family_i():
    diff = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n"
        "-state = nnx.merge(graphdef, state)\n"
        "+state = nnx.merge(graphdef, tree_unflatten(_state_treedef, leaves))\n"
    )
    tags = tag_diff(diff)
    assert "I" in tags


def test_tag_multiple_families():
    """A diff that touches block sizes AND dtype should tag both E and H."""
    diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
            "-BLOCK_M = 128\n+BLOCK_M = 64\n"
            "-out = x.astype(jnp.bfloat16)\n+out = x.astype(out_dtype)\n")
    tags = tag_diff(diff)
    assert "E" in tags and "H" in tags


def test_tag_returns_empty_on_no_signal():
    diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n"
            "-return 1\n+return 2\n")
    assert tag_diff(diff) == []


def test_tag_returns_sorted():
    diff = ("--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n"
            "-out = psum(x)\n+out = psum_scatter(x)\n"
            "-BLOCK_M = 128\n+BLOCK_M = 64\n")
    assert tag_diff(diff) == sorted(tag_diff(diff))


def test_describe_tags_renders_families():
    assert "H=quantization" in describe_tags(["H"])
    assert "E=tiling" in describe_tags(["E"])


def test_describe_empty_tags():
    assert "no family" in describe_tags([])
