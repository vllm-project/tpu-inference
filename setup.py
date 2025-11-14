#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

from setuptools import find_packages, setup

ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_tpu.")
    return requirements


def get_version():
    if env_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        return env_version
    return "0.0.0"


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
