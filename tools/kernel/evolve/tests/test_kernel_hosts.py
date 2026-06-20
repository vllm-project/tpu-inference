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
"""Tests for ``GenericHost`` (no TPU needed)."""

import jax.numpy as jnp

from tools.kernel.evolve.examples.kernel_hosts import (GenericHost,
                                                       _SimpleOracle)


def _trivial_reference(inputs):
    return inputs["x"] * 2.0


def _trivial_call(kernel, inputs):
    return kernel(inputs["x"])


def test_simple_oracle_compute_and_tol():
    o = _SimpleOracle(reference_fn=_trivial_reference)
    x = jnp.ones((4, ))
    out = o.compute({"x": x})
    assert jnp.allclose(out, jnp.full((4, ), 2.0))
    assert o.dtype_tolerance(jnp.bfloat16) == (0.2, 0.2)
    assert o.dtype_tolerance(jnp.float32) == (0.15, 0.15)
    assert o.dtype_tolerance(jnp.float8_e4m3fn) == (0.5, 0.5)


def test_simple_oracle_unpacks_tuple_output():

    def ref_returns_tuple(inputs):
        return inputs["x"] * 2.0, jnp.zeros_like(inputs["x"])

    o = _SimpleOracle(reference_fn=ref_returns_tuple)
    out = o.compute({"x": jnp.ones((2, ))})
    assert out.shape == (2, )


def test_generic_host_attributes(tmp_path):
    # Fake baseline file
    p = tmp_path / "ker.py"
    p.write_text("def trivial(x):\n    return x * 2.0\n")
    import tools.kernel.evolve.examples.kernel_hosts as kh

    # Patch _REPO_ROOT for this test
    orig_root = kh._REPO_ROOT
    kh._REPO_ROOT = tmp_path
    try:
        host = GenericHost(
            kernel_name="trivial",
            kernel_symbol="trivial",
            baseline_path_rel="ker.py",
            build_inputs=lambda: {"x": jnp.array([1.0, 2.0, 3.0])},
            reference_fn=_trivial_reference,
            call_kernel=_trivial_call,
            anti_cheat_skip=("noskip", ),
        )
        assert host.kernel_name == "trivial"
        assert host.kernel_symbol == "trivial"
        assert host.baseline_path == "ker.py"
        assert "trivial" in host.read_baseline_source()
        assert host.anti_cheat_skip_keys() == ("noskip", )
        assert "x" in host.inputs
    finally:
        kh._REPO_ROOT = orig_root


def test_generic_host_build_kernel_fn_invokes_call_kernel():
    """The closure returned by build_kernel_fn calls the user fn with the
    kernel symbol from the module and the host's input dict."""

    class _FakeModule:

        @staticmethod
        def trivial(x):
            return x * 2.0

    host = GenericHost(
        kernel_name="t",
        kernel_symbol="trivial",
        baseline_path_rel="ignored",
        build_inputs=lambda: {"x": jnp.array([1.0, 2.0])},
        reference_fn=_trivial_reference,
        call_kernel=_trivial_call,
    )
    fn = host.build_kernel_fn(_FakeModule)
    out = fn()
    assert jnp.allclose(out, jnp.array([2.0, 4.0]))


def test_generic_host_unpacks_tuple_kernel_output():
    """build_kernel_fn should return only [0] when the kernel returns a
    tuple — matches the RPA v3 / MLA convention."""

    class _FakeModule:

        @staticmethod
        def t(x):
            return x * 3.0, jnp.zeros_like(x)  # tuple

    host = GenericHost(
        kernel_name="t",
        kernel_symbol="t",
        baseline_path_rel="ignored",
        build_inputs=lambda: {"x": jnp.array([1.0])},
        reference_fn=_trivial_reference,
        call_kernel=lambda k, inp: k(inp["x"]),
    )
    fn = host.build_kernel_fn(_FakeModule)
    out = fn()
    assert out.shape == (1, )
    assert float(out[0]) == 3.0


def test_quantized_matmul_host_inputs_match_kernel_signature():
    from tools.kernel.evolve.examples.kernel_hosts import \
        make_quantized_matmul_host
    host = make_quantized_matmul_host(n_batch=16, n_in=128, n_out=64)
    assert host.kernel_name == "quantized_matmul"
    assert host.kernel_symbol == "quantized_matmul_kernel"
    assert set(host.inputs.keys()) == {"x", "w_q", "w_scale"}
    assert host.inputs["x"].shape == (16, 128)
    assert host.inputs["w_q"].shape == (64, 128)
    assert host.inputs["w_scale"].shape == (64, )
    assert host.inputs["w_q"].dtype == jnp.int8


def test_quantized_matmul_reference_is_dequant_matmul():
    """Smoke test: the reference is x @ (w_q * w_scale).T in bf16."""
    from tools.kernel.evolve.examples.kernel_hosts import \
        make_quantized_matmul_host
    host = make_quantized_matmul_host(n_batch=4, n_in=8, n_out=4)
    ref = host.get_oracle().compute(host.inputs)
    assert ref.shape == (4, 4)
    assert ref.dtype == jnp.bfloat16


def test_quantized_matmul_tolerance_relaxed_for_int8():
    """int8 path needs wider tolerance than the default bf16."""
    from tools.kernel.evolve.examples.kernel_hosts import \
        make_quantized_matmul_host
    host = make_quantized_matmul_host(n_batch=4, n_in=8, n_out=4)
    assert host.get_oracle().dtype_tolerance(jnp.int8) == (0.6, 0.6)
    assert host.get_oracle().dtype_tolerance(jnp.bfloat16) == (0.3, 0.3)
