#!/usr/bin/env python3
# Copyright 2026 Google LLC
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
"""Transitional empty meta-package forwarding ``tpu-inference`` to ``vllm-jax-core``."""

import os

from setuptools import setup

version = os.environ["VLLM_VERSION_OVERRIDE"].strip()
if not version:
    raise RuntimeError("VLLM_VERSION_OVERRIDE must be set and non-empty.")

setup(
    name="tpu-inference",
    version=version,
    description="Transitional meta-package installing vllm-jax-core",
    long_description=
    ("`tpu-inference` has been renamed to `vllm-jax-core` "
     "(or `pip install vllm-jax` for the combined vLLM + TPU distribution).\n"
     "This transitional meta-package installs the matching `vllm-jax-core` "
     "release without shipping duplicate files or entry points.\n"),
    long_description_content_type="text/markdown",
    author="vLLM JAX Contributors",
    url="https://github.com/vllm-project/vllm-jax",
    project_urls={
        "Documentation": "https://docs.vllm.ai/projects/tpu",
        "Source": "https://github.com/vllm-project/vllm-jax",
    },
    license="Apache-2.0",
    packages=[],
    python_requires=">=3.10",
    install_requires=[f"vllm-jax-core=={version}"],
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
