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
        help="Path to a JSON file with expected accuracy values."
    )
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