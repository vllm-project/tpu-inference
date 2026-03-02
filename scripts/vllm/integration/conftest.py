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


def pytest_addoption(parser):
    """Adds custom command-line options to pytest."""
    parser.addoption("--tensor-parallel-size",
                     type=int,
                     default=1,
                     help="The tensor parallel size to use for the test.")
    parser.addoption(
        "--expected-value",
        type=float,
        default=None,
        help=
        "This value will be used to compare the measure value and determine if the test passes or fails."
    )
    parser.addoption("--model-name",
                     type=str,
                     default=None,
                     help="Model name to test (e.g., 'model1')")
    parser.addoption("--fp8-kv-model-name",
                     type=str,
                     default=None,
                     help="Model name to test fp8-kv (e.g., 'model1')")
    parser.addoption(
        "--dataset-path",
        type=str,
        default=None,
        help=
        "Path to the dataset file used for accuracy evaluation (CSV or PKL).")
    parser.addoption("--image-dir",
                     action="store",
                     default=None,
                     help="Path to the directory containing test images.")
    parser.addoption("--num-test-cases",
                     type=int,
                     default=None,
                     help="Number of test cases to run from the dataset.")