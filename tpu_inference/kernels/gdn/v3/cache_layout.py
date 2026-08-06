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
"""The allocation layout of the convolution state cache this kernel reads.

The requirement belongs to the kernel, so it is described here and not
where the cache happens to be allocated. Keeping it here also keeps it
importable on its own: nothing in this module needs the model runner or
the parts of vLLM the runner pulls in.
"""

import jax.numpy as jnp
from jax.experimental.layout import Layout

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _stored_channels_first() -> bool | None:
    """Whether vLLM stores a convolution state channels first.

    vLLM calls the two orders SD, `[time, channels]`, and DS,
    `[channels, time]`, and picks between them with
    `VLLM_SSM_CONV_STATE_LAYOUT`. Returns None where that answer cannot be
    obtained, which is any vLLM predating the setting; the caller then
    reads the order off the shape instead.
    """
    try:
        from vllm.model_executor.layers.mamba.mamba_utils import \
            is_conv_state_dim_first
    except ImportError:
        return None
    return is_conv_state_dim_first()


def conv_state_layout(cache_shape, dtype) -> "Layout | None":
    """The layout the GDN kernel reads its convolution state operand in.

    A TPU array is tiled over its two minor dimensions and sub-word
    elements are packed along the second-minor one, so both numbers are
    derived from the element size. Returns None, after a warning, for a
    cache this is not defined for; it is then allocated in the default
    layout rather than failing the boot. The warnings are once per
    message, because this runs once per mamba layer and every layer of a
    model carries the same cache shape and dtype.

    Every message names the first state of the mamba cache rather than
    calling it a convolution state: this is asked about every mamba
    layer, and a model whose first state is something else is exactly
    the case where the answer is no.
    """
    if len(cache_shape) != 3:
        logger.warning_once(
            "the first state of the mamba cache has shape %s, which is not "
            "the rank-3 [blocks, conv_kernel_size, channels] convolution "
            "state the GDN kernel takes; its operand layout cannot be "
            "derived, so the cache is allocated in the default layout",
            cache_shape)
        return None
    channels_first = _stored_channels_first()
    if channels_first is None:
        # Read the order off the shape where vLLM cannot be asked. A
        # channels-last cache is [blocks, conv_kernel_size - 1, channels]
        # and so has its wide dimension last; a channels-first one is
        # [blocks, channels, conv_kernel_size - 1] and has it second.
        # Channel counts are hundreds and a convolution keeps a handful of
        # rows, so the two are never close.
        channels_first = cache_shape[1] > cache_shape[2]
    if channels_first:
        logger.warning_once(
            "the first state of the mamba cache has shape %s, which stores "
            "the channels ahead of the convolution window (vLLM's DS "
            "layout, VLLM_SSM_CONV_STATE_LAYOUT=DS). The layout derived "
            "here describes the channels-last form, and pinning it onto this "
            "one would pad a minor dimension of %d out to a full 128-wide "
            "tile, so the cache is allocated in the default layout",
            cache_shape, cache_shape[2])
        return None
    itemsize = jnp.dtype(dtype).itemsize
    if itemsize not in (1, 2, 4):
        logger.warning_once(
            "the first state of the mamba cache has dtype %s at %d bytes "
            "per element; the operand layout is only defined for 1, 2 and "
            "4, so the cache is allocated in the default layout", dtype,
            itemsize)
        return None
    packing = 4 // itemsize
    tiling = ((8 // packing, 128), ) if packing == 1 else ((8 // packing, 128),
                                                           (packing, 1))
    return Layout(major_to_minor=(0, 1, 2), tiling=tiling)
