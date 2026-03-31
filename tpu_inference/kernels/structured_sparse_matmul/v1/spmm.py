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
"""A TPU Kernel for N:M structured sparse matmul (SPMM).

Note: this kernel provides software emulation of N:M SPMM on TPU. You will 
only get performance benefits if the workload is memory bound.

The N:M sparsity means at most N nonzeros in every set of M values where each
value are separated in a given stride. For example, 1:4 with stride 2 sparse
matrix with shape (8,) could be [1, 0, 0, 1, 0, 0, 0, 0]. There is only one
nonzero among indices 0, 2, 4, 6 and one nonzero among indices 1, 3, 5, 7.

This SPMM kernel recovers nonzeros and metadata (compressed nonzeros indices)
to sparse matrix in VMEM and matmul with the dense matrix.

We expect to support
  - any N:M sparsity as long as M <= 16;
  - data type: f32, bf16, int8;
  - either LHS or RHS is sparse matrix but not both;
  - either contracting or non-contracting dimension is sparse;
  - either RHS is transposed or not.
  - pass a default value for the masked out data.

Example of activation @ sparse_weight:
```
key = random.PRNGKey(1234)
mask = gen_sparse_mask(
    key, weight.shape, sparsity, stride=stride, sparse_dim=sparse_dim)

sparse_weight = Sparsifier(
    weight, mask, sparsity=sparsity, sparse_dim=sparse_dim, stride=stride
)

result = structured_spmm(
    sparsity,
    sparse_weight.nonzeros,
    sparse_weight.metadata,
    activation,
    sparse_dim,
    rhs_sparse,
    rhs_transpose,
    stride=stride,
    block_m,
    block_k,
    block_n,
)
```

Further improvements:
- TODO(jevinjiang): Consider pass Sparsifier object as input.
- TODO(jevinjiang): Consider use one decompress function
- TODO(jevinjiang): Support int4 dtype in Mosaic.
- TODO(jevinjiang): Support subelement mask in Mosaic.
- TODO(jevinjiang): Emulate pack/unpack for unsupported HW.
- TODO(jevinjiang): Add int2 dtype in JAX, so let astype do pack/unpack on HW.
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class Sparsifier:
    """The sparsifier for structured sparse matrix.

  The sparsifier encodes sparse matrix to nonzeros and metadata.

  Attributes:
    shape: the shape of original data. Only used for decode.
    dtype: the dtype of original data. Only used for decode
    sparsity: the N:M structured sparsity.
    stride: each value in N:M structured are separated in stride.
    sparse_dim: the sparse dimension.
    nonzeros: the nonzeros gathered from original data.
    metadata: the compressed nonzeros indices.
  """

    def __init__(
        self,
        data: jax.Array,
        mask: jax.Array,
        *,
        sparsity: tuple[int, int],
        sparse_dim: int,
        stride: int,
    ):
        self.shape = data.shape
        self.dtype = data.dtype
        self.sparsity = sparsity
        self.stride = stride
        self.sparse_dim = sparse_dim
        # TODO(jevinjiang) validate inputs.
        self.nonzeros, self.metadata = self.encode(data, mask)

    def encode(self, data, mask):
        """Encode data to nonzeros and metadata based on the mask."""
        if self.sparse_dim == 0:
            data = data.transpose()
            mask = mask.transpose()
        x, y = self.sparsity
        h, w = data.shape
        data = data.reshape(-1, y, self.stride).transpose(0, 2, 1)
        mask = mask.reshape(-1, y, self.stride).transpose(0, 2, 1)
        _, nz_idx = jax.lax.top_k(mask, x)
        # Let CPU do the gather work.
        data = jax.device_get(data)
        nz = data[
            np.arange(nz_idx.shape[0])[:, None, None],
            np.arange(nz_idx.shape[1])[:, None],
            nz_idx,
        ]
        nz = jax.device_put(nz)
        nz_idx = nz_idx.transpose(2, 0, 1).reshape(x, h, w // y)
        nz = nz.transpose(2, 0, 1).reshape(x, h, w // y)
        if self.sparse_dim == 0:
            nz = nz.transpose(0, 2, 1)
            nz_idx = nz_idx.transpose(0, 2, 1)
        metadata = self.compress(nz_idx, next_log2(y))
        return nz, metadata

    def decode(self):
        """Decode nonzeros and metadata to sparse matrix and mask."""
        _, y = self.sparsity
        nz = self.nonzeros
        nz_idx = self.decompress(self.metadata, next_log2(y))
        h, w = self.shape
        if self.sparse_dim == 0:
            h, w = w, h
            nz = nz.transpose(0, 2, 1)
            nz_idx = nz_idx.transpose(0, 2, 1)
        nz = nz.transpose(1, 2, 0)
        nz_idx = nz_idx.transpose(1, 2, 0)
        # Let CPU do the scatter work.
        data = np.zeros((h, w // y, y), dtype=self.dtype)
        mask = np.zeros((h, w // y, y), dtype=jnp.int32)
        data[
            np.arange(nz_idx.shape[0])[:, None, None],
            np.arange(nz_idx.shape[1])[:, None],
            nz_idx,
        ] = nz
        mask[
            np.arange(nz_idx.shape[0])[:, None, None],
            np.arange(nz_idx.shape[1])[:, None],
            nz_idx,
        ] = 1
        data = data.reshape(-1, self.stride, y).transpose(0, 2,
                                                          1).reshape(h, w)
        mask = mask.reshape(-1, self.stride, y).transpose(0, 2,
                                                          1).reshape(h, w)
        data = jax.device_put(data)
        mask = jax.device_put(mask)
        if self.sparse_dim == 0:
            data = data.transpose()
            mask = mask.transpose()
        return data, mask

    def compress(self, data, bitwidth):
        """Pack the rows of data based on the element's bitwidth."""
        if bitwidth >= 32:
            return data
        d0, d1, d2 = data.shape
        bitwidth = next_pow2(bitwidth)
        packing = 32 // bitwidth
        assert d1 % packing == 0
        comp = np.zeros((d0, d1 // packing, d2), dtype=jnp.int32)
        for i in range(d1):
            shift = i % packing * bitwidth
            comp_i = i // packing
            comp[:, comp_i, :] |= data[:, i, :] << shift
        comp = jax.device_put(comp)
        return comp

    def decompress(self, data, bitwidth):
        """Unpack the rows of data based on the element's bitwidth."""
        if bitwidth >= 32:
            return data
        bitwidth = next_pow2(bitwidth)
        packing = 32 // bitwidth
        d0, d1, d2 = data.shape
        d1 *= packing
        result = np.zeros((d0, d1, d2), dtype=jnp.int32)
        for i in range(d1):
            shift = i % packing * bitwidth
            comp_i = i // packing
            result[:, i, :] = (data[:, comp_i, :] >> shift) & (2**bitwidth - 1)
        result = jax.device_put(result)
        return result


def next_log2(x):
    return math.ceil(math.log2(x))


def next_pow2(x):
    return 2**next_log2(x)


def gen_sparse_mask(
    key: Any,
    shape: jax.Array | tuple[int, int],
    sparsity: tuple[int, int],
    *,
    sparse_dim: int,
    stride: int = 1,
) -> jax.Array:
    """Generates a mask with N:M sparsity on a given dim.

  Args:
    key: random key.
    shape: the shape of the mask.
    sparsity: the N:M structured sparsity.
    sparse_dim: the sparse dimension.
    stride: each value in N:M structured are separated in stride.

  Returns:
    A mask with given shape and bool data.

  Raises:
    ValueError: An error occurred passing invalid input.
  """
    if sparse_dim not in (0, 1):
        raise ValueError(f"sparse_dim must be 0 or 1, got {sparse_dim}")
    if len(shape) != 2:
        raise ValueError(f"shape must be 2D, got {shape}")
    if len(sparsity) != 2:
        raise ValueError(
            f"expected 2 values in sparsity, but got {len(sparsity)}")
    h, w = shape
    x, y = sparsity
    if sparse_dim == 0:
        return gen_sparse_mask(key, (w, h),
                               sparsity,
                               sparse_dim=1,
                               stride=stride).transpose()
    if not (y > x > 0 and h > 0 and w > 0 and stride > 0):
        raise ValueError("invalid inputs")
    if w % (y * stride) != 0:
        raise ValueError(f"expected width {w} is a multiple of {y * stride}")
    nd_idx = jax.lax.broadcasted_iota(jnp.int32, (h * w // y, y), 1)
    mask = random.permutation(key, nd_idx, -1, independent=True) < x
    return mask.reshape(-1, stride, y).transpose(0, 2, 1).reshape(h, w)


def _get_dot_general_dim_nums(transposed: bool):
    return (((1, ), (1, )), ((), ())) if transposed else (((1, ), (0, )), ((),
                                                                           ()))


def _decompress_metadata(md_tile_ref, packing):
    """Decompress metadata. Expected to use in a Pallas kernel."""
    x, h, w = md_tile_ref.shape
    bitwidth = 32 // packing
    decompressed_md = []
    for i in range(h):
        unpacked_md = jnp.broadcast_to(md_tile_ref[:, pl.ds(i, 1), :],
                                       (x, packing, w))
        shift = jax.lax.broadcasted_iota(jnp.int32, unpacked_md.shape,
                                         1) * bitwidth
        unpacked_md = jax.lax.bitwise_and(
            jax.lax.shift_right_logical(unpacked_md, shift),
            jnp.broadcast_to(2**bitwidth - 1, unpacked_md.shape),
        )
        decompressed_md.append(unpacked_md)
    return jnp.concatenate(decompressed_md, axis=1)


def _decompress_nonzeros(
    sparsity,
    nonzeros,
    nonzeros_idx,
    sparse_dim,
    stride,
    default_value,
):
    """Decompress nonzeros. Expected to use in a Pallas kernel."""
    x, y = sparsity
    target_size = nonzeros.shape[1 + sparse_dim] * y
    tiles: Any = [None] * (target_size // stride)
    for i in range(y):
        lhs_tile_part = default_value
        for xi in range(x):
            lhs_tile_part = jnp.where(nonzeros_idx[xi] == i, nonzeros[xi],
                                      lhs_tile_part)
        for j in range(target_size // (y * stride)):
            if sparse_dim == 1:
                tiles[j * y + i] = lhs_tile_part[:,
                                                 j * stride:(j + 1) * stride]
            else:
                tiles[j * y + i] = lhs_tile_part[j * stride:(j + 1) *
                                                 stride, :]
    return jnp.concatenate(tiles, axis=sparse_dim).astype(nonzeros.dtype)


def _get_metadata_packing(sparsity_base):
    md_bitwidth = next_pow2(next_log2(sparsity_base))
    return 32 // md_bitwidth


def _get_dim_mapping(rhs_sparse: bool, rhs_transpose: bool):
    """The mapping to (m, n, k) dimensions."""
    m, n, k = (0, 1, 2)
    if rhs_sparse:
        return ((n, k), (m, k)) if rhs_transpose else ((k, n), (m, k))
    else:
        return ((m, k), (n, k)) if rhs_transpose else ((m, k), (k, n))


def _get_in_sepcs(
    sparsity: tuple[int, int],
    sparse_dim: int,
    rhs_sparse: bool,
    rhs_transpose: bool,
    block_m: int,
    block_k: int,
    block_n: int,
):
    """Get in_specs of structured SPMM."""
    x, y = sparsity
    md_packing = _get_metadata_packing(y)
    s_dims, d_dims = _get_dim_mapping(rhs_sparse, rhs_transpose)
    block_sz = (block_m, block_n, block_k)
    nz_shape = [x, block_sz[s_dims[0]], block_sz[s_dims[1]]]
    nz_shape[sparse_dim + 1] //= y
    md_shape = nz_shape.copy()
    md_shape[-2] //= md_packing
    nz_shape = tuple(nz_shape)
    md_shape = tuple(md_shape)
    mat_shape = (block_sz[d_dims[0]], block_sz[d_dims[1]])

    def get_sparse_index_map(i, j, k):
        indices = (i, j, k)
        return (0, indices[s_dims[0]], indices[s_dims[1]])

    def get_dense_index_map(i, j, k):
        indices = (i, j, k)
        return (indices[d_dims[0]], indices[d_dims[1]])

    return [
        pl.BlockSpec(nz_shape, get_sparse_index_map),
        pl.BlockSpec(md_shape, get_sparse_index_map),
        pl.BlockSpec(mat_shape, get_dense_index_map)
    ]


def _infer_out_dtype(ty1: jnp.dtype, ty2: jnp.dtype):
    """Infer matmul output dtype."""
    if ty1 == ty2:
        return ty1
    if jnp.issubdtype(ty1, jnp.integer) == jnp.issubdtype(ty2, jnp.integer):
        return ty1 if jnp.dtype(ty1).itemsize > jnp.dtype(
            ty2).itemsize else ty2
    return jnp.float32


def _verify(
    sparsity: tuple[int, int],
    nz: jax.Array,
    md: jax.Array,
    mat: jax.Array,
    *,
    sparse_dim: int,
    rhs_sparse: bool,
    rhs_transpose: bool,
    stride: int,
    block_m: int,
    block_k: int,
    block_n: int,
):
    """Validate inputs of structured SPMM and return original dim size m, n, k."""
    ##########################################################################
    # Verify sparsity.
    ##########################################################################
    x, y = sparsity
    if not (y > x > 0):
        raise ValueError(f"Invalid sparsity: {sparsity}")
    if y > 16:
        # TODO(jevinjiang): this requires padding because one compressed row is
        # decompressed to < 8 sublanes.
        raise ValueError("Not implemented: when sparsity base is > 16")
    if sparse_dim not in (0, 1):
        raise ValueError(f"Expected sparse_dim is 0 or 1, got {sparse_dim}")
    ##########################################################################
    # Verify shape and dtype.
    ##########################################################################
    if len(nz.shape) != 3:
        raise ValueError(f"Expected nonzeros' rank is 3, got {len(nz.shape)}")
    if len(md.shape) != 3:
        raise ValueError(f"Expected metadata's rank is 3, got {len(md.shape)}")
    if len(mat.shape) != 2:
        raise ValueError(f"Expected matrix's rank is 2, got {len(mat.shape)}")
    if md.dtype != jnp.int32:
        raise ValueError(f"Expected metadata's dtype is int32, got {md.dtype}")
    ##########################################################################
    # Verify stride.
    ##########################################################################
    nz_bitwidth = nz.dtype.itemsize * 8
    nz_packing = 32 // nz_bitwidth
    if sparse_dim:
        stride_multiple = 128
    else:
        stride_multiple = 8 * nz_packing
    if stride % stride_multiple != 0:
        raise ValueError(
            f"Expected {stride=} is a multiple of {stride_multiple}")
    ##########################################################################
    # Verify metadata compression produces expected shape.
    ##########################################################################
    name = "rhs" if rhs_sparse else "lhs"
    nz_x, nz_h, nz_w = nz.shape
    md_x, md_h, md_w = md.shape
    # For convenience (to avoid dealing with paddings), we try to select power of
    # 2 as nonzeross idx bitwidth. For example, if sparsity is 2:5, we will choose
    # 4 bits instead of 3.
    md_packing = _get_metadata_packing(y)
    if x != nz_x:
        raise ValueError(
            f"Expected {name} nonzeros shape[0] is {x}, got {nz_x}")
    if x != md_x:
        raise ValueError(
            f"Expected {name} metadata shape[0] is {x}, got {md_x}")
    if nz_w != md_w:
        raise ValueError(
            f"Expected nonzeros and metadata of {name} have same shape[2], got"
            f" {nz_w} vs {md_w}")
    if nz_h % md_packing != 0:
        raise ValueError(
            f"Expected {name} rows is a multiple of {md_packing=}, got {nz_h}")
    if nz_h // md_packing != md_h:
        raise ValueError(
            f"Expected {name} metadata rows can be perfectly packed by {md_packing}"
        )
    ##########################################################################
    # Verify nonzeros shape and matrix shape are expected.
    ##########################################################################
    decompressed_shape = list(nz.shape[1:])
    decompressed_shape[sparse_dim] *= y
    s_dims, d_dims = _get_dim_mapping(rhs_sparse, rhs_transpose)
    mnk = [None, None, None]
    for i in range(2):
        mnk[s_dims[i]] = decompressed_shape[i]
    for i in range(2):
        if mnk[d_dims[i]] and mnk[d_dims[i]] != mat.shape[i]:
            lhs_k = mnk[d_dims[i]]
            rhs_k = mat.shape[i]
            if rhs_transpose:
                lhs_k, rhs_k = rhs_k, lhs_k
            raise ValueError(
                f"Expected lhs contracting size {lhs_k} is equal to rhs contracting"
                f" size {rhs_k}")
        mnk[d_dims[i]] = mat.shape[i]
    m, n, k = mnk
    ##########################################################################
    # Verify block dimensions.
    ##########################################################################
    dim_names = ["m", "n", "k"]
    block_sizes = [block_m, block_n, block_k]
    block_multiples = [[], [], []]
    block_multiples[s_dims[sparse_dim]].append(stride * y)
    block_multiples[s_dims[0]].append(md_packing)

    for name, sz, bsz, multiples in zip(dim_names, mnk, block_sizes,
                                        block_multiples):
        for factor in multiples:
            if bsz % factor != 0:
                raise ValueError(
                    f"Expected block_{name}={bsz} is a multiple of {factor}")
        if sz % bsz != 0:
            raise ValueError(
                f"Expected block_{name}={bsz} can fully divide output rows {sz}"
            )
    return m, n, k


@functools.partial(
    jax.jit,
    static_argnames=[
        "sparsity",
        "sparse_dim",
        "rhs_sparse",
        "rhs_transpose",
        "stride",
        "block_m",
        "block_k",
        "block_n",
        "default_value",
        "out_dtype",
        "debug",
    ],
)
def _structured_spmm(
    sparsity: tuple[int, int],  # specify the x:y structured sparsity.
    nonzeros: jax.Array,
    metadata: jax.Array,
    matrix: jax.Array,
    *,
    sparse_dim: int,
    rhs_sparse: bool,
    rhs_transpose: bool = False,
    stride: int,
    block_m: int,
    block_k: int,
    block_n: int,
    default_value: Any = 0,
    out_dtype: Any = None,
    debug: bool = False,
) -> jax.Array:
    """General structured sparse matmul.

  Args:
    sparsity: the N:M structured sparsity.
    nonzeros: the nonzeros matrix.
    metadata: packed nonzeros idices.
    matrix: the dense matrix to matmul with.
    sparse_dim: sparsity is on each row when 0 or each column when 1.
    rhs_sparse: if true, nonzeros and metadata are from RHS and `matrix` is LHS.
    rhs_transpose: if true, RHS is transposed.
    stride: each value in N:M structured are separated in stride.
    block_m: the block size of LHS non-contracting dim.
    block_k: the block size of contracting dim.
    block_n: the block size of RHS non-contracting dim.
    default_value: the default value for the masked out data.
    out_dtype: the dtype of the output.
    debug: if true, enable debug mode in pallas kernel.

  Returns:
    The result of sparse matmul.

  Raises:
    ValueError: An error occurred passing invalid input.
  """
    m, n, k = _verify(
        sparsity,
        nonzeros,
        metadata,
        matrix,
        sparse_dim=sparse_dim,
        rhs_sparse=rhs_sparse,
        rhs_transpose=rhs_transpose,
        stride=stride,
        block_m=block_m,
        block_k=block_k,
        block_n=block_n,
    )

    _, y = sparsity
    md_packing = _get_metadata_packing(y)

    if not out_dtype:
        out_dtype = _infer_out_dtype(matrix.dtype, nonzeros.dtype)

    acc_dtype = jnp.float32
    if jnp.issubdtype(out_dtype, jnp.integer):
        acc_dtype = jnp.int32

    def _kernel(
        nz_tile_ref,
        md_tile_ref,
        mat_tile_ref,
        out_tile_ref,
        acc_tile_ref,
    ):

        @pl.when(pl.program_id(2) == 0)
        def _():
            acc_tile_ref[...] = jnp.zeros_like(acc_tile_ref, acc_dtype)

        # Decompress the metadata which contains nonzeros' indices.
        nonzeros_idx = _decompress_metadata(md_tile_ref, md_packing)
        # TODO(jevinjiang): try with large tiling.
        nonzeros = nz_tile_ref[...].astype(acc_dtype)
        decompressed_tile = _decompress_nonzeros(
            sparsity,
            nonzeros,
            nonzeros_idx,
            sparse_dim,
            stride,
            default_value,
        )
        mat_tile = mat_tile_ref[...]
        if rhs_sparse:
            lhs_tile, rhs_tile = mat_tile, decompressed_tile
        else:
            lhs_tile, rhs_tile = decompressed_tile, mat_tile
        # TODO(b/332970531): support arbitrary shape matmul for packed type.
        packing = 4 // lhs_tile.dtype.itemsize
        if packing > 1 and lhs_tile.shape[0] % (8 * packing) != 0:
            lhs_tile = lhs_tile.astype(acc_dtype)
        if packing > 1 and rhs_tile.shape[0] % (8 * packing) != 0:
            rhs_tile = rhs_tile.astype(acc_dtype)

        dim_nums = _get_dot_general_dim_nums(rhs_transpose)
        acc_tile_ref[...] += jax.lax.dot_general(
            lhs_tile, rhs_tile, dim_nums, preferred_element_type=acc_dtype)

        @pl.when(pl.program_id(2) == k // block_k - 1)
        def _():
            out_tile_ref[...] = acc_tile_ref[...].astype(out_dtype)

    in_specs = _get_in_sepcs(
        sparsity,
        sparse_dim,
        rhs_sparse,
        rhs_transpose,
        block_m,
        block_k,
        block_n,
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(m // block_m, n // block_n, k // block_k),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((block_m, block_n), acc_dtype)],
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        debug=debug,
    )(nonzeros, metadata, matrix)


# TODO(jevinjiang): Deprecate this API.
@functools.partial(
    jax.jit,
    static_argnames=[
        "sparsity",
        "rhs_sparse",
        "contract_sparse",
        "rhs_transpose",
        "stride",
        "block_m",
        "block_k",
        "block_n",
        "default_value",
        "out_dtype",
        "debug",
    ],
)
def structured_spmm(
    sparsity: tuple[int, int],  # specify the x:y structured sparsity.
    nonzeros: jax.Array,
    metadata: jax.Array,
    matrix: jax.Array,
    *,
    rhs_sparse: bool,
    contract_sparse: bool,
    rhs_transpose: bool = False,
    stride: int,
    block_m: int,
    block_k: int,
    block_n: int,
    default_value: Any = 0,
    out_dtype: Any = None,
    debug: bool = False,
) -> jax.Array:
    """General structured sparse matmul.

  Args:
    sparsity: the N:M structured sparsity.
    nonzeros: the nonzeros matrix.
    metadata: packed nonzeros idices.
    matrix: the dense matrix to matmul with.
    rhs_sparse: if true, nonzeros and metadata are from RHS and `matrix` is LHS.
    contract_sparse: if true, the values are sparse on the contracting dimension
    rhs_transpose: if true, RHS is transposed.
    stride: each value in N:M structured are separated in stride.
    block_m: the block size of LHS non-contracting dim.
    block_k: the block size of contracting dim.
    block_n: the block size of RHS non-contracting dim.
    default_value: the default value for the masked out data.
    out_dtype: the dtype of the output.
    debug: if true, enable debug mode in pallas kernel.

  Returns:
    The result of sparse matmul.

  Raises:
    ValueError: An error occurred passing invalid input.
  """
    # TODO(jevinjiang): Deprecate contract_sparse to use sparse_dim directly.
    if rhs_sparse:
        if contract_sparse:
            sparse_dim = 1 if rhs_transpose else 0
        else:
            sparse_dim = 0 if rhs_transpose else 1
    else:
        if contract_sparse:
            sparse_dim = 1
        else:
            sparse_dim = 0
    return _structured_spmm(
        sparsity,
        nonzeros,
        metadata,
        matrix,
        sparse_dim=sparse_dim,
        rhs_sparse=rhs_sparse,
        rhs_transpose=rhs_transpose,
        stride=stride,
        block_m=block_m,
        block_k=block_k,
        block_n=block_n,
        default_value=default_value,
        out_dtype=out_dtype,
        debug=debug,
    )
