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

from jax import export as jax_export
from vllm.v1.outputs import LogprobsTensors


def register_external_serialization_compat():
    """Register serialization compatibility for external types (e.g., vLLM)."""
    jax_export.register_namedtuple_serialization(
        LogprobsTensors, serialized_name="vllm.v1.outputs.LogprobsTensors")

    try:
        from flax.nnx import Param

        def _serialize_param(aux_data):
            return b""

        def _deserialize_param(b):
            return ()

        jax_export.register_pytree_node_serialization(
            Param,
            serialized_name="flax.nnx.variablelib.Param",
            serialize_auxdata=_serialize_param,
            deserialize_auxdata=_deserialize_param,
        )
    except ImportError:
        pass
