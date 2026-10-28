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
"""gdn_attention_v3_test's GDNAttentionTest on the flywheel_tpu GDN kernel."""

from unittest import mock

import jax
import jax.numpy as jnp
from absl.testing import absltest
from flywheel_tpu.linear_attention.gdn import wrapper as flywheel_gdn

from tests.kernels import gdn_attention_v3_test


def fused_conv1d_gdn(qkv: jax.Array, *args, **kwargs):
    # Note: the flywheel kernel takes bf16 activations, as the model
    # layers pass them.
    return flywheel_gdn.fused_conv1d_gdn(qkv.astype(jnp.bfloat16), *args,
                                         **kwargs)


class FlywheelGDNAttentionTest(gdn_attention_v3_test.GDNAttentionTest):
    """Every GDN v3 case, reference and tolerance, with the kernel swapped."""

    def setUp(self):
        super().setUp()
        self.enter_context(
            mock.patch.object(gdn_attention_v3_test.wrapper,
                              "fused_conv1d_gdn", fused_conv1d_gdn))


if __name__ == "__main__":
    absltest.main()
