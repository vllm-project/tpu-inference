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
