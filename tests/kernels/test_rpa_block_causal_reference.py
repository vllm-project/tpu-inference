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
import inspect
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


def _bfloat16_reference_inputs():
    inputs = list(_reference_inputs())
    inputs[:3] = [value.astype(jnp.bfloat16) for value in inputs[:3]]
    inputs[3] = jnp.zeros(
        kernel.get_kv_cache_shape(1, 4, 1, 128, jnp.bfloat16),
        dtype=jnp.bfloat16,
    )
    return tuple(inputs)


def test_reference_block_causal_attention_is_bidirectional_within_blocks():
    output, _ = kernel.ref_ragged_paged_attention(
        *_reference_inputs(),
        use_causal_mask=False,
        block_causal_size=2,
    )

    np.testing.assert_allclose(output[:, 0, 0], [1.5, 1.5, 3.75, 3.75])


def test_reference_fp32_accumulator_preserves_bfloat16_output_storage():
    output, _ = kernel.ref_ragged_paged_attention(
        *_bfloat16_reference_inputs(),
        use_causal_mask=False,
        out_dtype=jnp.float32,
    )

    assert output.dtype == jnp.bfloat16


def test_block_causal_and_token_causal_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        kernel.ref_ragged_paged_attention(
            *_reference_inputs(),
            use_causal_mask=True,
            block_causal_size=2,
        )


def test_block_causal_size_must_be_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        kernel.ref_ragged_paged_attention(
            *_reference_inputs(),
            use_causal_mask=False,
            block_causal_size=3,
        )


def test_pallas_block_causal_mask_avoids_vector_integer_division():
    source = inspect.getsource(kernel._ragged_paged_attention_kernel_loop)

    assert "// block_causal_size" not in source
    assert "bitwise_and" in source


def test_pallas_fp32_accumulator_casts_to_output_storage_before_packed_store():
    source = inspect.getsource(kernel._ragged_paged_attention_kernel_loop)

    assert "accumulator_dtype = acc_ref.dtype" in source
    assert "output_dtype = o_hbm_ref.dtype" in source
    assert ").astype(output_dtype)" in source
    assert source.index(").astype(output_dtype)") < source.index(
        "pltpu.bitcast(out, out_ref.dtype)")
