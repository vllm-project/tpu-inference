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

"""Fused, dense-DFT 2D FFT for TPU.

This kernel evaluates the separable forward DFT as ``W_col @ X @ W_row``.
It is not a Cooley-Tukey implementation: it is intended for workloads where
dense DFT matmuls are a good fit for the TPU MXU. A Pallas program produces one
full-height output-column stripe, retains the row transform in VMEM, and then
immediately consumes it for the column transform. This avoids an HBM write and
read of the intermediate.
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_PRECISION = jax.lax.Precision.HIGHEST


def _dft_matrices(
    height: int, width: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns real and imaginary forward-DFT matrices for both axes."""

    def dft_matrix(size: int) -> tuple[jax.Array, jax.Array]:
        index = jnp.arange(size, dtype=jnp.int32)
        phase_index = (index[:, None] * index[None, :]) % size
        phase = -2.0 * jnp.pi * phase_index.astype(jnp.float32) / size
        return jnp.cos(phase), jnp.sin(phase)

    row_real, row_imag = dft_matrix(width)
    col_real, col_imag = dft_matrix(height)
    return row_real, row_imag, col_real, col_imag


def _make_fused_stripe_kernel(
    height: int, width: int, row_k_tile: int, col_k_tile: int, col_tile: int
):
    """Builds a kernel for one ``(batch, column-stripe, height-block)`` program.

    The grid carries three axes: ``batch`` and the output ``column-stripe`` are
    parallel, while ``height-block`` is the reduction axis of the column
    transform and accumulates into a persistent VMEM scratch. Each program is
    handed only the matrix tiles it needs -- the column-stripe of ``W_row`` and
    one ``col_k_tile``-wide slice of ``W_col`` -- instead of whole DFT matrices,
    so VMEM use no longer grows with the full ``height``/``width``.
    """

    n_hblocks = height // col_k_tile

    def kernel(
        x_real_ref,
        x_imag_ref,
        row_real_ref,
        row_imag_ref,
        row_sum_ref,
        col_real_ref,
        col_imag_ref,
        col_sum_ref,
        z_real_ref,
        z_imag_ref,
        acc_real_ref,
        acc_imag_ref,
    ):
        hblock = pl.program_id(2)

        def dot(lhs, rhs):
            return jnp.dot(lhs, rhs, precision=_PRECISION)

        @pl.when(hblock == 0)
        def _init_accumulator():
            acc_real_ref[:, :] = jnp.zeros_like(acc_real_ref)
            acc_imag_ref[:, :] = jnp.zeros_like(acc_imag_ref)

        # Row transform for this block of `col_k_tile` rows:
        #   tmp[:, :] = X[block, :] @ W_row[:, stripe].
        # The `(width, col_tile)` row-matrix stripe is resident (invariant across
        # the height-block axis); tile only the `width` contraction so the dot
        # granularity stays bounded. `tmp` is a local value that is produced and
        # consumed within this program and never touches HBM.
        tmp_real = jnp.zeros((col_k_tile, col_tile), jnp.float32)
        tmp_imag = jnp.zeros((col_k_tile, col_tile), jnp.float32)
        for q_start in range(0, width, row_k_tile):
            x_real = x_real_ref[pl.ds(0, col_k_tile), pl.ds(q_start, row_k_tile)]
            x_imag = x_imag_ref[pl.ds(0, col_k_tile), pl.ds(q_start, row_k_tile)]
            row_real = row_real_ref[pl.ds(q_start, row_k_tile), :]
            row_imag = row_imag_ref[pl.ds(q_start, row_k_tile), :]
            row_sum = row_sum_ref[pl.ds(q_start, row_k_tile), :]
            k1 = dot(x_real, row_real)
            k2 = dot(x_imag, row_imag)
            k3 = dot(x_real + x_imag, row_sum)
            tmp_real += k1 - k2
            tmp_imag += k3 - k1 - k2

        # Column transform: accumulate W_col[:, block] @ tmp into the output
        # stripe. The `(height, col_k_tile)` column-matrix slice is streamed in
        # per height-block rather than held whole in VMEM.
        col_real = col_real_ref[:, :]
        col_imag = col_imag_ref[:, :]
        col_sum = col_sum_ref[:, :]
        k1 = dot(col_real, tmp_real)
        k2 = dot(col_imag, tmp_imag)
        k3 = dot(col_sum, tmp_real + tmp_imag)
        acc_real_ref[:, :] += k1 - k2
        acc_imag_ref[:, :] += k3 - k1 - k2

        @pl.when(hblock == n_hblocks - 1)
        def _write_output():
            z_real_ref[:, :] = acc_real_ref[:, :]
            z_imag_ref[:, :] = acc_imag_ref[:, :]

    return kernel


def _validate_tile(name: str, tile: int, dimension: int) -> None:
    if tile <= 0 or tile > dimension:
        raise ValueError(f"{name}={tile} must be in [1, {dimension}]")
    if dimension % tile:
        raise ValueError(f"{dimension=} must be divisible by {name}={tile}")
    if tile != dimension and tile % 128:
        raise ValueError(
            f"TPU Pallas requires {name}={tile} to be a multiple of 128 "
            f"unless it covers the full dimension {dimension}"
        )


@functools.partial(
    jax.jit,
    static_argnames=("col_tile", "row_k_tile", "col_k_tile"),
)
def _fft2d_impl(
    x_real: jax.Array,
    x_imag: jax.Array,
    *,
    col_tile: int,
    row_k_tile: int,
    col_k_tile: int,
) -> tuple[jax.Array, jax.Array]:
    if x_real.ndim != 3 or x_imag.ndim != 3:
        raise ValueError(
            "Expected real and imaginary inputs with shape [batch, height, width], "
            f"got {x_real.shape} and {x_imag.shape}"
        )
    if x_real.shape != x_imag.shape:
        raise ValueError(
            "Real and imaginary inputs must have the same shape, got "
            f"{x_real.shape} and {x_imag.shape}"
        )
    if x_real.dtype != jnp.float32 or x_imag.dtype != jnp.float32:
        raise TypeError(
            "Expected float32 real and imaginary inputs, got "
            f"{x_real.dtype} and {x_imag.dtype}"
        )

    batch, height, width = x_real.shape
    if not batch or not height or not width:
        raise ValueError("batch, height, and width must all be nonzero")

    _validate_tile("col_tile", col_tile, width)
    _validate_tile("row_k_tile", row_k_tile, width)
    _validate_tile("col_k_tile", col_k_tile, height)

    x_real = x_real.reshape(batch * height, width)
    x_imag = x_imag.reshape(batch * height, width)
    row_real, row_imag, col_real, col_imag = _dft_matrices(height, width)

    n_hblocks = height // col_k_tile
    # The output stripe and its accumulator both live in VMEM across the
    # height-block reduction axis.
    scratch_shapes = (
        pltpu.VMEM((height, col_tile), jnp.float32),
        pltpu.VMEM((height, col_tile), jnp.float32),
    )
    z_real, z_imag = pl.pallas_call(
        _make_fused_stripe_kernel(height, width, row_k_tile, col_k_tile, col_tile),
        out_shape=(
            jax.ShapeDtypeStruct((batch * height, width), jnp.float32),
            jax.ShapeDtypeStruct((batch * height, width), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=(
                # x: one `col_k_tile`-row block of the active batch.
                pl.BlockSpec(
                    (col_k_tile, width),
                    lambda b, j, hb: (b * n_hblocks + hb, 0),
                ),
                pl.BlockSpec(
                    (col_k_tile, width),
                    lambda b, j, hb: (b * n_hblocks + hb, 0),
                ),
                # W_row: the `(width, col_tile)` output column stripe.
                pl.BlockSpec((width, col_tile), lambda b, j, hb: (0, j)),
                pl.BlockSpec((width, col_tile), lambda b, j, hb: (0, j)),
                pl.BlockSpec((width, col_tile), lambda b, j, hb: (0, j)),
                # W_col: the `(height, col_k_tile)` reduction slice.
                pl.BlockSpec((height, col_k_tile), lambda b, j, hb: (0, hb)),
                pl.BlockSpec((height, col_k_tile), lambda b, j, hb: (0, hb)),
                pl.BlockSpec((height, col_k_tile), lambda b, j, hb: (0, hb)),
            ),
            out_specs=(
                pl.BlockSpec((height, col_tile), lambda b, j, hb: (b, j)),
                pl.BlockSpec((height, col_tile), lambda b, j, hb: (b, j)),
            ),
            grid=(batch, width // col_tile, n_hblocks),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            # The process must also be started with a matching (or larger)
            # --xla_tpu_scoped_vmem_limit_kib flag in LIBTPU_INIT_ARGS.
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name="fused_dense_dft2d",
    )(
        x_real,
        x_imag,
        row_real,
        row_imag,
        row_real + row_imag,
        col_real,
        col_imag,
        col_real + col_imag,
    )
    return (
        z_real.reshape(batch, height, width),
        z_imag.reshape(batch, height, width),
    )


def fft2d(
    x: jax.Array,
    *,
    col_tile: int,
    row_k_tile: int,
    col_k_tile: int,
) -> jax.Array:
    """Computes an unnormalised forward 2D DFT of a complex64 input on TPU.

    Args:
      x: Complex64 tensor with shape ``[batch, height, width]``.
      col_tile: Number of output columns assigned to one Pallas program. This
        must be selected by the caller to fit the target TPU's scoped VMEM.
      row_k_tile: Row-transform reduction tile width.
      col_k_tile: Column-transform reduction tile height.

    Returns:
      A complex64 array with the same shape as ``x``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [batch, height, width], got {x.shape}")
    if x.dtype != jnp.complex64:
        raise TypeError(f"Expected complex64 input, got {x.dtype}")
    z_real, z_imag = fft2d_split(
        jnp.real(x),
        jnp.imag(x),
        col_tile=col_tile,
        row_k_tile=row_k_tile,
        col_k_tile=col_k_tile,
    )
    return z_real.astype(jnp.complex64) + jnp.asarray(
        1j, dtype=jnp.complex64
    ) * z_imag.astype(jnp.complex64)


def fft2d_split(
    x_real: jax.Array,
    x_imag: jax.Array,
    *,
    col_tile: int,
    row_k_tile: int,
    col_k_tile: int,
) -> tuple[jax.Array, jax.Array]:
    """Computes a forward 2D DFT from already-split complex64 components.

    This is the zero-conversion path: it accepts and returns separate float32
    real and imaginary tensors with shape ``[batch, height, width]``.
    """
    if x_real.ndim != 3 or x_imag.ndim != 3:
        raise ValueError(
            "Expected real and imaginary inputs with shape [batch, height, width], "
            f"got {x_real.shape} and {x_imag.shape}"
        )
    if x_real.shape != x_imag.shape:
        raise ValueError(
            "Real and imaginary inputs must have the same shape, got "
            f"{x_real.shape} and {x_imag.shape}"
        )
    if x_real.dtype != jnp.float32 or x_imag.dtype != jnp.float32:
        raise TypeError(
            "Expected float32 real and imaginary inputs, got "
            f"{x_real.dtype} and {x_imag.dtype}"
        )
    return _fft2d_impl(
        x_real,
        x_imag,
        col_tile=col_tile,
        row_k_tile=row_k_tile,
        col_k_tile=col_k_tile,
    )


def _benchmark(
    name: str,
    fn: Callable[..., Any],
    args: tuple[jax.Array, ...],
    repeats: int,
) -> tuple[Any, float]:
    """Compiles ``fn``, then returns its output and mean execution time."""
    output = fn(*args)
    jax.block_until_ready(output)
    start = time.perf_counter()
    for _ in range(repeats):
        output = fn(*args)
        jax.block_until_ready(output)
    elapsed_ms = (time.perf_counter() - start) * 1_000 / repeats
    print(f"  {name:<18} {elapsed_ms:9.2f} ms")
    return output, elapsed_ms


def main(argv: Sequence[str] | None = None) -> None:
    """Checks and benchmarks dense DFTs at caller-selected square sizes."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=(1024,),
        help="Square spatial sizes to benchmark (default: 1024).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of square inputs per size (default: 100).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed iterations after compilation (default: 3).",
    )
    parser.add_argument(
        "--col-tile",
        type=int,
        required=True,
        help="Output-column stripe width; choose this for the target's VMEM.",
    )
    parser.add_argument(
        "--row-k-tile",
        type=int,
        required=True,
        help="Row-transform reduction tile width.",
    )
    parser.add_argument(
        "--col-k-tile",
        type=int,
        required=True,
        help="Column-transform reduction tile height.",
    )
    args = parser.parse_args(argv)
    if (
        args.batch_size <= 0
        or args.repeats <= 0
        or any(size <= 0 for size in args.sizes)
    ):
        parser.error("sizes, --batch-size, and --repeats must all be positive")

    devices = jax.devices()
    memory_stats = devices[0].memory_stats()
    hbm_gib = memory_stats["bytes_limit"] / 2**30
    vmem_bytes = pltpu.get_tpu_info().vmem_capacity_bytes
    required_vmem_flag = f"--xla_tpu_scoped_vmem_limit_kib={vmem_bytes // 1024}"
    print(
        f"JAX devices: {len(devices)} x {devices[0].device_kind}; "
        f"HBM per device: {hbm_gib:.2f} GiB"
    )
    print(f"batch size: {args.batch_size}; repeats: {args.repeats}")
    print(
        f"tiles: col={args.col_tile}; row-k={args.row_k_tile}; col-k={args.col_k_tile}"
    )
    print(f"kernel scoped VMEM request: {vmem_bytes / 2**20:.0f} MiB")
    if required_vmem_flag not in os.environ.get("LIBTPU_INIT_ARGS", ""):
        print(
            "warning: launch with "
            f"LIBTPU_INIT_ARGS='{required_vmem_flag}' to permit this request"
        )
    for size in args.sizes:
        key = jax.random.key(size)
        x_real = jax.random.normal(key, (args.batch_size, size, size), jnp.float32)
        x_imag = jax.random.normal(
            jax.random.fold_in(key, 1), (args.batch_size, size, size), jnp.float32
        )
        x = x_real + jnp.asarray(1j, jnp.complex64) * x_imag
        x.block_until_ready()

        print(f"\n{args.batch_size} x {size} x {size} complex64 input")
        jax_fft2 = jax.jit(jnp.fft.fft2)
        complex_fft2d = jax.jit(
            functools.partial(
                fft2d,
                col_tile=args.col_tile,
                row_k_tile=args.row_k_tile,
                col_k_tile=args.col_k_tile,
            )
        )
        split_fft2d = jax.jit(
            functools.partial(
                fft2d_split,
                col_tile=args.col_tile,
                row_k_tile=args.row_k_tile,
                col_k_tile=args.col_k_tile,
            )
        )
        reference, _ = _benchmark("jax.numpy.fft2", jax_fft2, (x,), args.repeats)
        complex_output, complex_ms = _benchmark(
            "fft2d(complex)", complex_fft2d, (x,), args.repeats
        )
        (split_real, split_imag), split_ms = _benchmark(
            "fft2d_split", split_fft2d, (x_real, x_imag), args.repeats
        )

        for name, output in (
            ("fft2d(complex)", complex_output),
            (
                "fft2d_split",
                split_real.astype(jnp.complex64)
                + jnp.asarray(1j, jnp.complex64) * split_imag.astype(jnp.complex64),
            ),
        ):
            max_abs_error = jnp.max(jnp.abs(output - reference))
            relative_error = max_abs_error / jnp.maximum(jnp.max(jnp.abs(reference)), 1)
            max_abs_error, relative_error = jax.device_get(
                jax.block_until_ready((max_abs_error, relative_error))
            )
            print(
                f"  {name:<18} max abs error {max_abs_error:.4e}; "
                f"relative error {relative_error:.4e}"
            )
        print(f"  split-input speedup: {complex_ms / split_ms:.2f}x")


if __name__ == "__main__":
    main()
