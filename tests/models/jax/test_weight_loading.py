# SPDX-License-Identifier: Apache-2.0
# Test for LoRA weight loading API

import jax
import jax.numpy as jnp
from flax import nnx
from jax._src import test_util as jtu

from tpu_commons.models.jax.utils.weight_utils import \
    transfer_state_with_mappings

# ----- nnx.Module Wrappers -----


class SourceLayer(nnx.Module):

    def __init__(self, rngs):
        self.kernel = nnx.Param(jax.random.normal(rngs(), (4, 4)))
        self.bias = nnx.Param(jax.random.normal(rngs(), (4, )))


class SourceModel(nnx.Module):

    def __init__(self, rngs):
        self.src_lm_head = nnx.Param(jax.random.normal(rngs(), (2, 4)))
        self.layers = {0: SourceLayer(rngs)}


class TargetLinear(nnx.Module):

    def __init__(self, rngs):
        self.kernel = nnx.Param(jnp.zeros((4, 4)))
        self.bias = nnx.Param(jnp.zeros((4, )))


class TargetBlock(nnx.Module):

    def __init__(self, rngs):
        self.mlp = {"up_proj": TargetLinear(rngs)}


class TargetModel(nnx.Module):

    def __init__(self, rngs):
        self.tgt_lm_head = nnx.Param(jnp.zeros((2, 4)))
        self.model = {"layers": {0: TargetBlock(rngs)}}


# ----- Test -----
class WeightTransfer(jtu.JaxTestCase):

    def test_transfer_state(self):
        rng = nnx.Rngs(0)
        src_model = SourceModel(rng)
        tgt_model = TargetModel(rng)

        # Get split states
        _, src_state = nnx.split(src_model)
        _, tgt_state = nnx.split(tgt_model)

        # Overwrite known values
        src_state["layers"][0]["kernel"].value = jnp.ones((4, 4)) * 42.0
        src_state["layers"][0]["bias"].value = jnp.ones((4, )) * 7.0
        src_state["src_lm_head"].value = jnp.ones((2, 4)) * 6.0
        # Mapping for both kernel and bias
        mappings = {
            "layers.*.kernel": ("model.layers.*.mlp.up_proj.kernel", (None, )),
            "layers.*.bias": ("model.layers.*.mlp.up_proj.bias", (None, )),
            "src_lm_head": ("tgt_lm_head", (None, None)),
        }

        # Transfer
        new_tgt_state = transfer_state_with_mappings(src_state, tgt_state,
                                                     mappings)

        # Assert correctness
        assert jnp.allclose(
            new_tgt_state["model"]["layers"][0]["mlp"]["up_proj"]
            ["kernel"].value, 42.0)
        assert jnp.allclose(
            new_tgt_state["model"]["layers"][0]["mlp"]["up_proj"]
            ["bias"].value, 7.0)
        assert jnp.allclose(new_tgt_state["tgt_lm_head"].value, 6.0)
