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

from typing import Iterator

from flax import nnx


class JaxModule(nnx.Module):
    """Base module for JAX layers, extending flax.nnx.Module.
    """

    def named_parameters(self,
                         prefix: str = "") -> Iterator[tuple[str, nnx.Param]]:
        """Yields the named parameters of the module."""
        params = nnx.state(self, nnx.Param)

        def _traverse_params(params, path=()):
            if hasattr(params, 'items'):
                for name, value in params.items():
                    yield from _traverse_params(value, path + (str(name), ))
            else:
                yield ".".join(path), params

        yield from _traverse_params(params, path=(prefix, ) if prefix else ())
