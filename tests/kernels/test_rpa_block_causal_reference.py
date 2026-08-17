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

import jax.numpy as jnp
import numpy as np
import pytest


def _load_kernel_module():
    root = pathlib.Path(__file__).resolve().parents[2]
    package_paths = {
        "tpu_inference":
        root / "tpu_inference",
        "tpu_inference.kernels":
        root / "tpu_inference" / "kernels",
        "tpu_inference.kernels.ragged_paged_attention":
        root / "tpu_inference" / "kernels" / "ragged_paged_attention",
        "tpu_inference.kernels.ragged_paged_attention.v3":
        root / "tpu_inference" / "kernels" / "ragged_paged_attention" / "v3",
    }
    previous_modules = {name: sys.modules.get(name) for name in package_paths}
    for name, path in package_paths.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    module_name = "rpa_block_causal_kernel_under_test"
    path = package_paths["tpu_inference.kernels.ragged_paged_attention.v3"] / \
        "kernel.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


kernel = _load_kernel_module()


def _reference_inputs():
    head_dim = 128
    queries = jnp.zeros((4, 1, head_dim), dtype=jnp.float32)
    keys = jnp.zeros((4, 1, head_dim), dtype=jnp.float32)
    values = jnp.zeros((4, 1, head_dim), dtype=jnp.float32)
    values = values.at[:, 0, 0].set(jnp.array([1.0, 2.0, 4.0, 8.0]))
    kv_cache = jnp.zeros((1, 4, 2, 1, head_dim), dtype=jnp.float32)
    return (
        queries,
        keys,
        values,
        kv_cache,
        jnp.array([4], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0, 4], dtype=jnp.int32),
        jnp.array([0, 0, 1], dtype=jnp.int32),
    )


def test_reference_block_causal_attention_is_bidirectional_within_blocks():
    output, _ = kernel.ref_ragged_paged_attention(
        *_reference_inputs(),
        use_causal_mask=False,
        block_causal_size=2,
    )

    np.testing.assert_allclose(output[:, 0, 0], [1.5, 1.5, 3.75, 3.75])


def test_block_causal_and_token_causal_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        kernel.ref_ragged_paged_attention(
            *_reference_inputs(),
            use_causal_mask=True,
            block_causal_size=2,
        )
