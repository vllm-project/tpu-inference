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

import unittest

from flax import nnx
from jax import numpy as jnx

from tpu_inference.layers.jax import JaxModule


class MyModule(JaxModule):

    def __init__(self):
        super().__init__()
        self.w1 = nnx.Param(jnx.ones((2, 2)))


class NestedModule(JaxModule):

    def __init__(self):
        super().__init__()
        self.weight = nnx.Param(jnx.ones((4, 4)))
        self.inner = JaxModule()
        self.inner.quantization_scale = nnx.Param(jnx.ones((4, 4)))


class TestJaxModule(unittest.TestCase):

    def test_named_parameters(self):
        """Tests the named_parameters method of JaxModule."""

        module = MyModule()

        params = nnx.state(module, nnx.Param)

        self.assertEqual(len(params.items()), 1)

        params = nnx.state(module, nnx.Param)
        self.assertCountEqual(["w1"],
                              [k for k, _ in module.named_parameters()])

    def test_named_parameters_in_nested_module(self):
        """Tests named_parameters method in nested modules."""

        module = NestedModule()

        self.assertCountEqual(["inner.quantization_scale", "weight"],
                              [k for k, _ in module.named_parameters()])

    def test_named_children_in_nested_module(self):
        """Tests named_children method in nested modules."""

        class Depth3Module(JaxModule):

            def __init__(self):
                super().__init__()
                self.my_module_1 = MyModule()
                self.my_module_2 = MyModule()
                self.nested_module_a = NestedModule()
                self.nested_module_b = NestedModule()
                self.layers = nnx.List([MyModule() for _ in range(3)])

        module = Depth3Module()

        children = set(name for name, _ in module.named_children())
        # Should only return immediate children modules, without "inner" etc.
        self.assertCountEqual(
            children, {
                "my_module_1", "my_module_2", "nested_module_a",
                "nested_module_b", "layers"
            })

    def test_named_modules_yields_self_first(self):
        """Top-level module is yielded with empty prefix before any children."""

        module = NestedModule()
        names = [name for name, _ in module.named_modules()]
        self.assertEqual(names[0], "")
        self.assertIn("inner", names)

    def test_named_modules_in_nested_module(self):
        """Walks self + descendants, including JaxModuleList and its items."""

        class Depth3Module(JaxModule):

            def __init__(self):
                super().__init__()
                self.my_module = MyModule()
                self.nested_module = NestedModule()
                self.layers = nnx.List([MyModule() for _ in range(3)])

        module = Depth3Module()
        names = set(name for name, _ in module.named_modules())
        self.assertEqual(
            names,
            {
                "",  # self
                "my_module",
                "nested_module",
                "nested_module.inner",
                "layers",  # JaxModuleList itself
                "layers.0",
                "layers.1",
                "layers.2",
            })

    def test_named_modules_dedups_shared_submodule(self):
        """Shared submodules are yielded only once (matches torch semantics)."""

        shared = MyModule()

        class SharedRefModule(JaxModule):

            def __init__(self):
                super().__init__()
                self.a = shared
                self.b = shared

        module = SharedRefModule()
        names = [name for name, _ in module.named_modules()]
        # 'a' and 'b' point to the same module; only the first is yielded.
        self.assertEqual(names.count("a") + names.count("b"), 1)


if __name__ == "__main__":
    unittest.main()
