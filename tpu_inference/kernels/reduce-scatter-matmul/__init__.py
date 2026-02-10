# SPDX-License-Identifier: Apache-2.0
"""Reduce-scatter matmul kernel module."""

from .kernel import get_kernel_name, reduce_scatter_matmul

__all__ = [
    "reduce_scatter_matmul",
    "get_kernel_name",
]
