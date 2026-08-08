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
import types

import jax
import jax.numpy as jnp


def _cdiv(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def _load_attention_metadata():
    module_names = ("vllm", "vllm.utils", "vllm.utils.math_utils")
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    math_utils = types.ModuleType("vllm.utils.math_utils")
    math_utils.cdiv = _cdiv
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")
    sys.modules["vllm.utils.math_utils"] = math_utils
    path = (pathlib.Path(__file__).resolve().parents[3] / "tpu_inference" /
            "layers" / "common" / "attention_metadata.py")
    spec = importlib.util.spec_from_file_location(
        "attention_metadata_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


attention_metadata = _load_attention_metadata()
AttentionMaskKind = attention_metadata.AttentionMaskKind
AttentionMaskSpec = attention_metadata.AttentionMaskSpec
AttentionMetadata = attention_metadata.AttentionMetadata
resolve_use_causal_mask = attention_metadata.resolve_use_causal_mask


def _metadata(mask_spec=None):
    kwargs = {}
    if mask_spec is not None:
        kwargs["attention_mask_spec"] = mask_spec
    return AttentionMetadata(
        input_positions=jnp.arange(4, dtype=jnp.int32),
        block_tables=jnp.zeros((4, ), dtype=jnp.int32),
        seq_lens=jnp.array([4], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 4], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        **kwargs,
    )


def test_attention_mask_defaults_to_causal():
    assert resolve_use_causal_mask(_metadata()) is True


def test_bidirectional_mask_is_explicit_static_metadata():
    causal = _metadata()
    bidirectional = _metadata(
        AttentionMaskSpec(AttentionMaskKind.BIDIRECTIONAL))

    assert resolve_use_causal_mask(bidirectional) is False
    assert jax.tree_util.tree_structure(
        causal) != jax.tree_util.tree_structure(bidirectional)


def test_explicit_kernel_override_takes_precedence():
    bidirectional = _metadata(
        AttentionMaskSpec(AttentionMaskKind.BIDIRECTIONAL))

    assert resolve_use_causal_mask(bidirectional, override=True) is True
