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


class TestJaxModule(unittest.TestCase):

    def test_register_parameter(self):
        """Tests the register_parameter method of JaxModule."""

        class MyModule(JaxModule):

            def __init__(self):
                super().__init__()
                self.w1 = nnx.Param(jnx.ones((2, 2)))

        module = MyModule()

        params = nnx.state(module, nnx.Param)

        self.assertEqual(len(params.items()), 1)

        # Test registering a valid parameter
        module.register_parameter("w2", nnx.Param(jnx.ones((3, 3))))
        params = nnx.state(module, nnx.Param)
        self.assertCountEqual(["w2", "w1"],
                              [k for k, _ in module.named_parameters()])

        # Test registering None parameter
        module.register_parameter("none_param", None)
        params = nnx.state(module, nnx.Param)
        self.assertCountEqual(["w2", "w1"],
                              [k for k, _ in module.named_parameters()])

    def test_register_parameter_in_nested_module(self):
        """Tests registering parameters in nested modules."""

        class NestedModule(JaxModule):

            def __init__(self):
                super().__init__()
                self.weight = nnx.Param(jnx.ones((4, 4)))
                self.inner = JaxModule()
                self.inner.register_parameter("quantization_scale",
                                              nnx.Param(jnx.ones((4, 4))))

        module = NestedModule()

        self.assertCountEqual(["inner.quantization_scale", "weight"],
                              [k for k, _ in module.named_parameters()])


if __name__ == "__main__":
    unittest.main()
