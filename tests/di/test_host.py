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
import pytest

from tpu_commons.di.host import DIHost


def test_register_and_resolve_no_dependencies():
    host = DIHost()

    class ServiceA:
        pass

    def provide_service_a():
        return ServiceA()

    host.register(provide_service_a, ServiceA)
    instance = host.resolve(ServiceA)
    assert isinstance(instance, ServiceA)


def test_register_and_resolve_with_one_dependency():
    host = DIHost()

    class ServiceA:
        pass

    class ServiceB:

        def __init__(self, service_a: ServiceA):
            self.service_a = service_a

    def provide_service_a():
        return ServiceA()

    def provide_service_b(service_a: ServiceA):
        return ServiceB(service_a)

    host.register(provide_service_a, ServiceA)
    host.register(provide_service_b,
                  ServiceB,
                  dependencies={"service_a": ServiceA})

    instance_b = host.resolve(ServiceB)
    assert isinstance(instance_b, ServiceB)
    assert isinstance(instance_b.service_a, ServiceA)


def test_resolve_multi_level_dependencies():
    host = DIHost()

    class ServiceA:
        pass

    class ServiceB:

        def __init__(self, service_a: ServiceA):
            self.service_a = service_a

    class ServiceC:

        def __init__(self, service_b: ServiceB):
            self.service_b = service_b

    def provide_service_a():
        return ServiceA()

    def provide_service_b(service_a: ServiceA):
        return ServiceB(service_a)

    def provide_service_c(service_b: ServiceB):
        return ServiceC(service_b)

    host.register(provide_service_a, ServiceA)
    host.register(provide_service_b,
                  ServiceB,
                  dependencies={"service_a": ServiceA})
    host.register(provide_service_c,
                  ServiceC,
                  dependencies={"service_b": ServiceB})

    instance_c = host.resolve(ServiceC)
    assert isinstance(instance_c, ServiceC)
    assert isinstance(instance_c.service_b, ServiceB)
    assert isinstance(instance_c.service_b.service_a, ServiceA)


def test_resolve_unregistered_type():
    host = DIHost()

    class UnregisteredService:
        pass

    with pytest.raises(
            ValueError,
            match="No provider registered for type UnregisteredService"):
        host.resolve(UnregisteredService)
