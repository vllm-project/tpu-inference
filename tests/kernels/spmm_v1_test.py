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

import math

import hypothesis as hp
import hypothesis.strategies as hps
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import random

import tpu_inference.kernels.structured_sparse_matmul.v1.spmm as structured_spmm


class StructuredSpmmTest(parameterized.TestCase):

    @parameterized.product(
        dtype=(jnp.float32, jnp.bfloat16, jnp.int8),
        stride=(1, 2, 3, 4),
        sparse_dim=(0, 1),
    )
    @hp.given(
        sparsity=hps.tuples(hps.integers(1, 3), hps.integers(4, 10)),
        shape_multiplier=hps.tuples(hps.integers(4, 10), hps.integers(4, 10)),
        seed=hps.integers(0, 4),
    )
    @hp.settings(deadline=None,
                 max_examples=20,
                 verbosity=hp.Verbosity.verbose)
    def test_sparse_matrix_encode(self, sparsity, stride, sparse_dim,
                                  shape_multiplier, dtype, seed):
        key = random.PRNGKey(seed)
        _, y = sparsity
        bitwidth = structured_spmm.next_pow2(structured_spmm.next_log2(y))
        packing = 32 // bitwidth
        n = shape_multiplier[0] * packing
        m = shape_multiplier[1]
        if sparse_dim == 0:
            n *= stride * y
        else:
            m *= stride * y
        shape = (n, m)
        mask = structured_spmm.gen_sparse_mask(key,
                                               shape,
                                               sparsity,
                                               sparse_dim=sparse_dim,
                                               stride=stride)
        data = random.normal(key, shape, dtype=jnp.float32).astype(dtype)
        sm = structured_spmm.Sparsifier(data,
                                        mask,
                                        sparsity=sparsity,
                                        sparse_dim=sparse_dim,
                                        stride=stride)
        decoded_data, decoded_mask = sm.decode()
        expected_data = jnp.where(mask, data, 0)
        np.testing.assert_array_equal(decoded_data, expected_data)
        np.testing.assert_array_equal(decoded_mask, mask)

    @parameterized.product(
        sparsity=((1, 4), (2, 5)),
        shape_spec=(
            (128, 128, 128),
            (128, 256, 512),
            (256, 128, 512),
            (512, 128, 256),
        ),
        rhs_sparse=(True, False),
        contract_sparse=(True, False),
        rhs_transpose=(True, False),
        default_val=(-1.9, 1.9),
        lhs_dtype=(jnp.float32, jnp.bfloat16, jnp.int8),
        rhs_dtype=(jnp.float32, jnp.bfloat16, jnp.int8),
    )
    def test_structured_spmm(
        self,
        sparsity,
        shape_spec,
        rhs_sparse,
        contract_sparse,
        rhs_transpose,
        default_val,
        lhs_dtype,
        rhs_dtype,
    ):
        # TODO(jevinjiang): support mix int/int matmtul in Mosaic.
        if lhs_dtype == rhs_dtype == jnp.int8:
            return
        _, y = sparsity
        m, k, n = shape_spec
        block_m = 128
        block_n = 128
        stride = 128
        if rhs_sparse:
            if contract_sparse:
                sparse_dim = int(rhs_transpose)
                k = math.lcm(stride * y, k)
            else:
                sparse_dim = int(not rhs_transpose)
                n = math.lcm(stride * y, n)
                block_n = math.lcm(stride * y, block_n)
        else:
            if contract_sparse:
                sparse_dim = 1
                k = math.lcm(stride * y, k)
            else:
                sparse_dim = 0
                m = math.lcm(stride * y, m)
                block_m = math.lcm(stride * y, block_m)
        lhs_shape = (m, k)
        rhs_shape = (n, k) if rhs_transpose else (k, n)

        key = random.PRNGKey(123)
        lhs = random.normal(key, lhs_shape,
                            dtype=jnp.float32).astype(lhs_dtype)
        rhs = random.normal(key, rhs_shape,
                            dtype=jnp.float32).astype(rhs_dtype)
        sparse_dtype = rhs_dtype if rhs_sparse else lhs_dtype
        default_val = (int(default_val) if jnp.issubdtype(
            sparse_dtype, jnp.integer) else default_val)
        out_dtype = structured_spmm._infer_out_dtype(lhs_dtype, rhs_dtype)

        if rhs_sparse:
            mask = structured_spmm.gen_sparse_mask(key,
                                                   rhs.shape,
                                                   sparsity,
                                                   sparse_dim=sparse_dim,
                                                   stride=stride)
            sm = structured_spmm.Sparsifier(rhs,
                                            mask,
                                            sparsity=sparsity,
                                            sparse_dim=sparse_dim,
                                            stride=stride)
            mat = lhs
            rhs = jnp.where(mask, rhs, default_val)
        else:
            mask = structured_spmm.gen_sparse_mask(key,
                                                   lhs.shape,
                                                   sparsity,
                                                   sparse_dim=sparse_dim,
                                                   stride=stride)
            sm = structured_spmm.Sparsifier(lhs,
                                            mask,
                                            sparsity=sparsity,
                                            sparse_dim=sparse_dim,
                                            stride=stride)
            mat = rhs
            lhs = jnp.where(mask, lhs, default_val)

        actual = structured_spmm.structured_spmm(
            sparsity,
            sm.nonzeros,
            sm.metadata,
            mat,
            rhs_sparse=rhs_sparse,
            contract_sparse=contract_sparse,
            rhs_transpose=rhs_transpose,
            stride=stride,
            block_m=block_m,
            block_k=k,
            block_n=block_n,
            default_value=default_val,
            out_dtype=out_dtype,
        )
        if rhs_transpose:
            rhs = rhs.T
        expected = jnp.dot(lhs, rhs, preferred_element_type=out_dtype)
        if out_dtype == jnp.bfloat16:
            actual = actual.astype(jnp.float32)
            expected = expected.astype(jnp.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)


if __name__ == "__main__":
    absltest.main()
