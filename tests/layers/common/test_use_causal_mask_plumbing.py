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
"""Verifies that ``use_causal_mask`` is threaded through the attention stack.

Actually executing the ragged-paged-attention (RPA) kernel requires a TPU
(the Pallas kernel does not run on CPU), so these tests do NOT invoke the
kernel. Instead they assert the flag is correctly threaded through every layer
of the attention stack:

  * declared as a parameter on the public entry points, and
  * forwarded into the next call in the chain.

The core checks parse the source with ``ast`` and therefore run on CPU without
importing the heavy (TPU-only) dependency graph. An additional
``inspect.signature`` check imports the real modules when possible and is
skipped otherwise (e.g. when the TPU/vLLM deps are unavailable on CPU).
"""

import ast
import inspect
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_ATTENTION_INTERFACE = (_REPO_ROOT / "tpu_inference" / "layers" / "common" /
                        "attention_interface.py")
_JAX_ATTENTION = (_REPO_ROOT / "tpu_inference" / "layers" / "jax" /
                  "attention" / "attention.py")
_ATTENTION_METADATA = (_REPO_ROOT / "tpu_inference" / "layers" / "common" /
                       "attention_metadata.py")
_QWEN3 = (_REPO_ROOT / "tpu_inference" / "models" / "jax" / "qwen3.py")
_TPU_RUNNER = (_REPO_ROOT / "tpu_inference" / "runner" / "tpu_runner.py")


def _parse(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text())


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    """Find a top-level function by name."""
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _find_method(module: ast.Module, cls_name: str,
                 method_name: str) -> ast.FunctionDef:
    """Find a method by name inside a named class."""
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if (isinstance(sub, ast.FunctionDef)
                        and sub.name == method_name):
                    return sub
    raise AssertionError(f"method {cls_name}.{method_name!r} not found")


def _param_names(func: ast.FunctionDef) -> set[str]:
    a = func.args
    names = set()
    for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
        names.add(arg.arg)
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def _has_default_true(func: ast.FunctionDef, param: str) -> bool:
    """Return True if ``param`` has a literal default of True."""
    a = func.args
    # kwonlyargs pair 1:1 with kw_defaults (None => no default).
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        if arg.arg == param and default is not None:
            return isinstance(default, ast.Constant) and default.value is True
    # positional/pos-only args: defaults align with the tail of the arg list.
    pos = [*a.posonlyargs, *a.args]
    defaults = a.defaults
    if defaults:
        tail = pos[len(pos) - len(defaults):]
        for arg, default in zip(tail, defaults):
            if arg.arg == param:
                return (isinstance(default, ast.Constant)
                        and default.value is True)
    return False


class TestUseCausalMaskPlumbing:
    """AST-level checks (CPU-safe, no imports of the TPU dependency graph)."""

    def test_sharded_rpa_declares_and_forwards_flag(self):
        module = _parse(_ATTENTION_INTERFACE)
        func = _find_function(module, "sharded_ragged_paged_attention")

        # Declared as a parameter defaulting to True.
        assert "use_causal_mask" in _param_names(func)
        assert _has_default_true(func, "use_causal_mask")

        # Forwarded into the kernel-call kwargs (v3 default / batched path).
        src = ast.get_source_segment(_ATTENTION_INTERFACE.read_text(), func)
        assert 'kwargs["use_causal_mask"] = use_causal_mask' in src

    def test_attention_declares_and_forwards_flag(self):
        module = _parse(_ATTENTION_INTERFACE)
        func = _find_function(module, "attention")

        assert "use_causal_mask" in _param_names(func)
        assert _has_default_true(func, "use_causal_mask")

        # Forwarded into the sharded_ragged_paged_attention call.
        src = ast.get_source_segment(_ATTENTION_INTERFACE.read_text(), func)
        assert "use_causal_mask=use_causal_mask" in src

    def test_jax_attention_method_declares_and_forwards_flag(self):
        module = _parse(_JAX_ATTENTION)
        method = _find_method(module, "Attention", "attention")

        assert "use_causal_mask" in _param_names(method)
        assert _has_default_true(method, "use_causal_mask")

        # Forwarded into the ragged_paged_attention kernel call.
        src = ast.get_source_segment(_JAX_ATTENTION.read_text(), method)
        assert "use_causal_mask=use_causal_mask" in src

    def test_attention_metadata_declares_static_meta_field(self):
        # The flag rides on AttentionMetadata as a STATIC meta_field (not a data
        # leaf), so it threads through model.__call__ unchanged and True/False
        # compile as separate programs.
        text = _ATTENTION_METADATA.read_text()
        module = ast.parse(text)
        fields = set()
        for node in module.body:
            if isinstance(node,
                          ast.ClassDef) and node.name == "AttentionMetadata":
                for sub in node.body:
                    if isinstance(sub, ast.AnnAssign) and isinstance(
                            sub.target, ast.Name):
                        fields.add(sub.target.id)
        assert "use_causal_mask" in fields, "must be a dataclass field"
        meta = text.split("meta_fields=")[1].split("]")[0]
        assert '"use_causal_mask"' in meta, "must be a register_dataclass meta_field"

    def test_qwen3_attention_forwards_flag_from_metadata(self):
        module = _parse(_QWEN3)
        method = _find_method(module, "Qwen3Attention", "__call__")
        src = ast.get_source_segment(_QWEN3.read_text(), method)
        assert "use_causal_mask=attention_metadata.use_causal_mask" in src, (
            "Qwen3Attention.__call__ must forward the metadata flag into attention()"
        )

    def test_runner_requests_bidirectional_for_diffusion_canvas(self):
        # The block-diffusion canvas forward must request bidirectional attention.
        assert "use_causal_mask=False" in _TPU_RUNNER.read_text(), (
            "the diffusion forward_fn must set use_causal_mask=False on the canvas"
        )


class TestUseCausalMaskMetaFieldRuntime:
    """Runtime pytree behaviour of the meta_field (jax-only, CPU-safe).

    Loads ``attention_metadata.py`` by file path so it does not import the
    (TPU/vLLM-only) ``tpu_inference`` package ``__init__``. Skipped if jax or the
    standalone module load is unavailable on this host.
    """

    def test_meta_field_is_static_and_defaults_true(self):
        try:
            import importlib.util
            from dataclasses import replace

            import jax
            import jax.numpy as jnp
        except Exception as exc:  # pragma: no cover - depends on host deps
            pytest.skip(f"jax unavailable on this host: {exc}")

        spec = importlib.util.spec_from_file_location("_am_ucm",
                                                      str(_ATTENTION_METADATA))
        am = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(am)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"attention_metadata not loadable standalone: {exc}")

        def mk(use_causal_mask=True):
            return am.AttentionMetadata(
                input_positions=jnp.arange(4, dtype=jnp.int32),
                block_tables=jnp.zeros((4, ), jnp.int32),
                seq_lens=jnp.ones((2, ), jnp.int32),
                query_start_loc=jnp.array([0, 4], jnp.int32),
                request_distribution=jnp.zeros((3, ), jnp.int32),
                padded_num_reqs=2,
                use_causal_mask=use_causal_mask,
            )

        # AR default is unchanged.
        assert mk().use_causal_mask is True
        # Static -> not a data leaf.
        leaves, td_true = jax.tree_util.tree_flatten(mk(True))
        assert all(not isinstance(leaf, bool) for leaf in leaves)
        # True vs False -> distinct treedefs -> separate compiled programs.
        _, td_false = jax.tree_util.tree_flatten(mk(False))
        assert td_true != td_false
        # The runner's replace() path flips it.
        assert replace(mk(), use_causal_mask=False).use_causal_mask is False


class TestUseCausalMaskSignatureImport:
    """Import-based signature check.

    This imports the real modules and inspects live signatures. It is skipped
    when the modules cannot be imported on this host (the attention stack pulls
    in TPU/Pallas and vLLM dependencies that are not always present on CPU).
    """

    def test_live_signatures_expose_flag(self):
        try:
            from tpu_inference.layers.common import attention_interface
            from tpu_inference.layers.jax.attention import \
                attention as jax_attention
        except Exception as exc:  # pragma: no cover - depends on host deps
            pytest.skip(
                f"attention modules not importable on this host: {exc}")

        for fn in (attention_interface.attention,
                   attention_interface.sharded_ragged_paged_attention):
            params = inspect.signature(fn).parameters
            assert "use_causal_mask" in params
            assert params["use_causal_mask"].default is True

        method_params = inspect.signature(
            jax_attention.Attention.attention).parameters
        assert "use_causal_mask" in method_params
        assert method_params["use_causal_mask"].default is True
