import pytest
import json

def pytest_addoption(parser):
    """Adds custom command-line options to pytest."""
    parser.addoption(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="The tensor parallel size to use for the test."
    )
    parser.addoption(
        "--expected-values-file",
        type=str,
        default=None,
        help="This is used to specify the JSON file that stores the expected values. " +
            "The results from running test_accuracy on a GPU will be saved to this file, " +
            "and when running on a TPU, the results will be read from this file for comparison.")
    parser.addoption(
        "--model-names",
        action="store",
        # default="meta-llama/Llama-3.1-8B-Instruct",
        default=None,
        help="Comma-separated list of model names to test (e.g., 'model1,model2')"
    )
    parser.addoption(
        "--fp8-kv-model-names",
        action="store",
        default=None,
        help="Comma-separated list of model names to test fp8-kv (e.g., 'model1,model2')"
    )