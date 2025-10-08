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
import abc


class HostInterface(abc.ABC):
    """
    An interface that the host system (e.g., SGLang, vLLM) must implement.
    This defines the contract for how the backend can call back into the host.
    """

    @abc.abstractmethod
    def get_next_batch_to_run(self):
        """
        The backend calls this to get the next batch of requests to process.
        """
        pass

    @abc.abstractmethod
    def process_batch_result(self, batch_result):
        """
        The backend calls this to return the results of a processed batch.
        """
        pass


class BackendInterface(abc.ABC):
    """
    An interface that the backend system (e.g., tpu_inference) must implement.
    This defines the contract for how the host can call into the backend.
    """

    @abc.abstractmethod
    def launch_tpu_batch(self, batch_to_launch):
        """
        The host calls this to launch a batch of requests on the backend.
        """
        pass
