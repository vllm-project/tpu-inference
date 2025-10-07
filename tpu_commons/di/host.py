"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Any, Callable, Dict, Type


class DIHost:
    """
    A simple dependency injection host.

    This host manages a graph of functions, where each function is a provider
    for a specific data type and declares its own dependencies.
    """

    def __init__(self):
        self._providers: Dict[Type, Callable[..., Any]] = {}
        self._dependencies: Dict[Callable[..., Any], Dict[str, Type]] = {}

    def register(self,
                 provider: Callable[..., Any],
                 output_type: Type,
                 dependencies: Dict[str, Type] = None):
        """
        Registers a provider function with the host.

        Args:
            provider: The function that produces the output.
            output_type: The data type that the function produces.
            dependencies: A dictionary mapping argument names of the provider
                          to the data types they require.
        """
        self._providers[output_type] = provider
        if dependencies:
            self._dependencies[provider] = dependencies

    def resolve(self, target_type: Type) -> Any:
        """
        Resolves a dependency by creating an instance of the target type.

        This method will recursively resolve all dependencies required to call
        the provider for the target type.

        Args:
            target_type: The data type to be resolved.

        Returns:
            An instance of the target type.
        """
        if target_type not in self._providers:
            raise ValueError(
                f"No provider registered for type {target_type.__name__}")

        provider = self._providers[target_type]

        if provider not in self._dependencies:
            # Provider has no dependencies, so just call it.
            return provider()

        # Resolve dependencies for the provider.
        kwargs = {}
        for arg_name, dep_type in self._dependencies[provider].items():
            kwargs[arg_name] = self.resolve(dep_type)

        return provider(**kwargs)
