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

from flax import nnx

from tpu_inference.layers.jax import JaxModule


class JaxConv(nnx.Conv, JaxModule):
    """Convolution layer that inherits from JaxModule and maps kernel to weight for compatibility."""

    def __init__(self, *args, **kwargs):
        # nnx.Conv uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        nnx.Conv.__init__(self, *args, **kwargs)

        # For compatibility, alias kernel to weight
        self.weight = self.kernel
        delattr(self, 'kernel')

    def __getattr__(self, name: str):
        if name == "kernel":
            return self.weight
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
