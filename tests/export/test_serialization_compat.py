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

import unittest
import jax
import jax.numpy as jnp
from vllm.v1.outputs import LogprobsTensors
from tpu_inference.export.serialization_compat import register_external_serialization_compat

class TestSerializationCompat(unittest.TestCase):
    def test_logprobs_tensors_serialization(self):
        # 1. Create dummy object with JAX arrays to verify type serialization
        x = LogprobsTensors(
            logprob_token_ids=jnp.zeros((2, 2), dtype=jnp.int32),
            logprobs=jnp.zeros((2, 2), dtype=jnp.float32),
            selected_token_ranks=jnp.zeros((2,), dtype=jnp.int32),
            cu_num_generated_tokens=None
        )
        
        def f(dummy):
            return x
            
        # 2. Try to export WITHOUT registration
        # This assumes the test runs in a clean environment where registration hasn't happened yet.
        try:
            exp = jax.export.export(jax.jit(f))(jnp.zeros(1))
            exp.serialize()
            self.fail("Expected serialization error before registration")
        except Exception as e:
            self.assertIn("unregistered type", str(e))
            print(f"Verified expected failure without registration: {e}")
            
        # 3. Now register
        register_external_serialization_compat()
        
        # 4. Try to export WITH registration
        try:
            exported = jax.export.export(jax.jit(f))(jnp.zeros(1))
            serialized = exported.serialize()
            self.assertIsNotNone(serialized)
            print("Successfully exported and serialized after registration.")
        except Exception as e:
            self.fail(f"Failed to export after registration: {e}")

    def test_param_serialization(self):
        try:
            from flax.nnx import Param
        except ImportError:
            self.skipTest("Flax NNX not available")
            
        x = Param(jnp.zeros((2, 2), dtype=jnp.float32))
        
        def f(dummy):
            return x
            
        try:
            exported = jax.export.export(jax.jit(f))(jnp.zeros(1))
            serialized = exported.serialize()
            self.assertIsNotNone(serialized)
            print("Successfully exported and serialized Param after registration.")
        except Exception as e:
            self.fail(f"Failed to export Param after registration: {e}")

if __name__ == '__main__':
    unittest.main()
