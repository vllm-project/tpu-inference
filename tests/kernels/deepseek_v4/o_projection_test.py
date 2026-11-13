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

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from tests.kernels.deepseek_v4.rope_test import (make_cos_sin_cache,
                                                 rope_ref_impl)
from tpu_inference.kernels.experimental.deepseek_v4.o_projection import (
    LANE, gather_cos_sin, wo_a_projection)

# DeepSeek-V4: head_dim 512, qk_rope_head_dim 64, 8 heads per group.
HEAD_DIM = 512
ROTARY_DIM = 64
MAX_POSITION = 4096
HEADS_PER_GROUP = 8


def o_projection_ref_impl(activations, wo_a, wo_a_scale, *, num_groups):
    """``DeepseekV4Attention._o_proj``'s wo_a einsum, up to (not incl.) ``wo_b``."""
    num_tokens = activations.shape[0]
    o_f = activations.reshape(num_tokens, num_groups, -1)  # [t, g, d]
    reduction = o_f.shape[-1]
    w = wo_a.reshape(reduction, num_groups, -1)
    s = wo_a_scale.reshape(num_groups, -1)
    z = (jnp.einsum(
        "tgd,dgr->tgr",
        o_f,
        w.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ) * s.astype(jnp.bfloat16)[None, ...])
    return z.astype(jnp.bfloat16).reshape(num_tokens, -1)


def make_inputs(rng,
                *,
                num_tokens,
                num_groups,
                lora_rank,
                heads_per_group=HEADS_PER_GROUP):
    """Activations in the ``[T, G * H, head_dim]`` view the kernel takes."""
    reduction = heads_per_group * HEAD_DIM
    num_heads = num_groups * heads_per_group
    activations = jnp.asarray(
        rng.standard_normal((num_tokens, num_heads, HEAD_DIM)),
        dtype=jnp.bfloat16,
    )
    wo_a = jnp.asarray(
        rng.standard_normal((reduction, num_groups * lora_rank)),
        dtype=jnp.float8_e4m3fn,
    )
    wo_a_scale = jnp.asarray(rng.uniform(0.5, 1.5,
                                         size=num_groups * lora_rank),
                             dtype=jnp.float32)
    return activations, wo_a, wo_a_scale


class OProjectionTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(  # DeepSeek-V4-Flash, unsharded.
            testcase_name="dsv4_flash",
            num_tokens=256,
            num_groups=8,
            lora_rank=1024,
        ),
        dict(  # The benchmarked shape: activations [1024, 128, 512].
            testcase_name="dsv4_pro",
            num_tokens=1024,
            num_groups=16,
            lora_rank=1024,
        ),
    )
    def test_matches_reference(
        self,
        *,
        num_tokens,
        num_groups,
        lora_rank,
        tile_t=None,
        tile_r=None,
        sub_t=None,
    ):
        """The fused inverse-RoPE path against ``rope`` + the wo_a einsum."""
        rng = np.random.default_rng(0)
        activations, wo_a, wo_a_scale = make_inputs(
            rng,
            num_tokens=num_tokens,
            num_groups=num_groups,
            lora_rank=lora_rank,
        )
        positions = jnp.asarray(rng.integers(0,
                                             MAX_POSITION,
                                             size=(num_tokens, )),
                                dtype=jnp.int32)
        cos_sin_cache = jnp.asarray(make_cos_sin_cache(MAX_POSITION,
                                                       ROTARY_DIM),
                                    dtype=jnp.float32)
        cos_sin = gather_cos_sin(positions, cos_sin_cache, inverse=True)
        self.assertEqual(cos_sin.shape, (num_tokens, 2 * LANE))
        out = wo_a_projection(
            activations,
            wo_a,
            wo_a_scale,
            cos_sin,
            tile_t=tile_t,
            tile_r=tile_r,
            sub_t=sub_t,
            quantize_activations=False,
        )

        roped = rope_ref_impl(activations,
                              positions,
                              cos_sin_cache,
                              inverse=True)
        expected = o_projection_ref_impl(roped,
                                         wo_a,
                                         wo_a_scale,
                                         num_groups=num_groups)

        np.testing.assert_allclose(
            np.asarray(out, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    absltest.main()
