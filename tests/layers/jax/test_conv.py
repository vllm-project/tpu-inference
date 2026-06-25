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

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.jax.conv import JaxConv


class TestJaxConv:
    """Test suite for JaxConv layer."""

    @pytest.mark.parametrize("rng_key", [jax.random.PRNGKey(0)])
    @pytest.mark.parametrize("in_channels", [3])
    @pytest.mark.parametrize("out_channels", [16])
    @pytest.mark.parametrize("kernel_size", [(2, 4, 4)])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_numerical_correctness_against_flax(self, rng_key, in_channels,
                                                out_channels, kernel_size,
                                                dtype):
        """Verifies JaxConv output is identical to flax.nnx.Conv given identical inputs."""
        rngs = nnx.Rngs(0)

        # Instantiate JaxConv (tests 'dtype' fallback forwarding to 'param_dtype')
        jax_conv = JaxConv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            dtype=dtype,
            rngs=rngs,
        )

        # Reset RNGs so standard flax.nnx.Conv initializes identically
        rngs = nnx.Rngs(0)
        nnx_conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs,
        )

        # Confirm identical initialization (using weight[...] instead of weight.value)
        assert jnp.allclose(jax_conv.weight[...], nnx_conv.kernel[...])
        assert jnp.allclose(jax_conv.bias[...], nnx_conv.bias[...])

        # Generate a mock 3D image/video input (Batch, Temporal, Height, Width, Channels)
        # matching Qwen3-VL 3D patch embedding layout
        x = jax.random.uniform(rng_key, (1, 2, 16, 16, in_channels),
                               dtype=dtype)

        # Compare forward outputs
        jax_output = jax_conv(x)
        nnx_output = nnx_conv(x)

        assert jnp.allclose(jax_output, nnx_output, rtol=1e-5, atol=1e-5)

    def test_parameter_aliasing(self):
        """Verifies weight aliasing, kernel deletion, and named_parameters matching."""
        rngs = nnx.Rngs(0)
        jax_conv = JaxConv(
            in_features=3,
            out_features=16,
            kernel_size=(2, 4, 4),
            rngs=rngs,
        )

        assert "kernel" not in jax_conv.__dict__
        assert "weight" in jax_conv.__dict__
        assert isinstance(jax_conv.weight, nnx.Param)

        assert jax_conv.kernel is jax_conv.weight

        named_params = dict(jax_conv.named_parameters())
        assert "weight" in named_params
        assert "kernel" not in named_params

    def test_sharding_assignment(self):
        """Verifies that custom init-level partitioning and sharding works correctly on JaxConv."""
        mesh = Mesh(jax.devices('cpu')[:1], ("model", ))

        with jax.set_mesh(mesh):
            rngs = nnx.Rngs(0)
            jax_conv = JaxConv(
                in_features=3,
                out_features=16,
                kernel_size=(2, 4, 4),
                kernel_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                  sharding=(None, None, None,
                                                            None, "model")),
                bias_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                                sharding=("model", )),
                rngs=rngs,
            )

            # Confirm sharding on weight and bias is preserved by inspecting PartitionSpec (.spec)
            assert jax_conv.weight.sharding.spec == P(None, None, None, None,
                                                      "model")
            assert jax_conv.bias.sharding.spec == P("model", )
