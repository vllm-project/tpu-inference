#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="tpu_commons",
    version="0.1.0",
    description="",
    long_description=open("README.md").read() if hasattr(
        open("README.md"), "read") else "",
    long_description_content_type="text/markdown",
    author="tpu_commons Contributors",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchxla>=2.0.0",
        "torchvision>=0.15.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Artificial Intelligence",
    ],
)
