#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import List

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    # Use a dictionary to store requirements, using the package name as the key.
    # This automatically handles duplicates and allows for easy overriding.
    req_map = {}

    def _get_pkg_key(line: str) -> str:
        """
        Extracts the normalized package name to use as a dictionary key.
        Example: 'jax[tpu]==0.8.0' -> 'jax'
        """
        # Strip version specifiers and comparison operators: 'jax[tpu]==0.8.0' -> 'jax[tpu]'
        # The regex splits by <, =, >, !, or space
        name_part = re.split(r'[<=>! ]', line.strip())[0]

        # Remove extras (square brackets): 'jax[tpu]' -> 'jax'
        # Normalize: lowercase and replace '-' with '_' to ensure consistency (PEP 503)
        return name_part.split('[')[0].lower().replace('-', '_')

    def _read_requirements(filename: str):
        with open(get_path(filename)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("-r "):
                    # Recursive call for included requirement files
                    _read_requirements(line.split()[1])
                elif line.startswith(("-", "--")):
                    # Skip pip flags like --index-url or --pre
                    continue
                else:
                    # The override happens here: later entries with the same key
                    # will replace earlier ones in the dictionary.
                    req_map[_get_pkg_key(line)] = line

    try:
        _read_requirements("requirements.txt")

        # For TPU v7x build
        if os.getenv("IS_FOR_V7X", "true").lower() == "true":
            print("Overriding and adding packages from requirements_v7x.txt")
            _read_requirements("requirements_v7x.txt")
    except (FileNotFoundError, IOError):
        print("Failed to read requirements.txt in vllm_tpu.")

    # For Debugging
    print(list(req_map.values()))

    # Convert the dictionary values back into a list for install_requires
    return list(req_map.values())


def get_version():
    version = os.getenv("VLLM_VERSION_OVERRIDE", "0.0.0").strip()
    if os.getenv("IS_FOR_V7X", "true").lower() == "false":
        version = f"{version}.post6"

    return version


setup(
    name="tpu_inference",
    version=get_version(),
    description="",
    long_description=open("README.md").read() if hasattr(
        open("README.md"), "read") else "",
    long_description_content_type="text/markdown",
    author="tpu_inference Contributors",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=get_requirements(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
