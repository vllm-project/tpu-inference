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

import functools
import time
from typing import Callable, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.sharding import SingleDeviceSharding

from tpu_inference import envs
from tpu_inference.layers.common.moe import (FusedMoEMethodBase, MoEBackend,
                                             moe_apply)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, UnfusedMoEWeights, process_moe_weights,
    quantize_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quantization import (
    MXFP4_BLOCK_SIZE, MXFP4_REQUANTIZED_BLOCK_SIZE,
    dequantize_tensor_from_mxfp4_packed, e8m0_to_fp32, u8_unpack_e2m1)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import cpu_mesh, cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    jax_array_from_reshaped_torch, shard_put)
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

# TODO(#2952): remove once MXFP4 supports all fused MoE backends.
MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS = [
    MoEBackend.GMM_EP, MoEBackend.GMM_TP
]

_GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS = {
    "gate_up_proj_blocks": "w13_blocks",
    "gate_up_proj_scales": "w13_scales",
    "gate_up_proj_bias": "w13_bias",
    "down_proj_blocks": "w2_blocks",
    "down_proj_scales": "w2_scales",
    "down_proj_bias": "w2_bias",
}


class Mxfp4FusedMoEMethod(QuantizeMethodBase):
    """MXFP4 method for GPT-OSS routed experts."""

    def __init__(self):
        self.extra_backend_kwargs = {}

    def load_weights(self, *, layer: JaxRoutedExperts,
                     original_load_weights_fn,
                     weights: Iterable[tuple[str, torch.Tensor]]) -> set:
        """Stage the six GPT-OSS MXFP4 expert tensors on the layer."""
        # FP8 delegates to `original_load_weights_fn`, but here that hook is
        # just a stub, so we load all six expert tensors ourselves.
        loaded_names = set()
        unexpected = []
        for torch_name, torch_weight in weights:
            suffix = torch_name.rsplit(".", maxsplit=1)[-1]
            attr_name = _GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS.get(suffix)
            if attr_name is None:
                unexpected.append(torch_name)
                continue
            jax_param = getattr(layer, attr_name, None)
            assert isinstance(jax_param, nnx.Param)
            staging_shape = tuple(jax_param.shape)
            # Checkpoint blocks come in 4-D (E, rows, groups, 16); flatten the
            # group axes back down to the 3-D staging shape. The identity
            # permute is just to stop the helper's default 2-D transpose.
            reshape_dims = (staging_shape
                            if suffix.endswith("_blocks") else None)
            jax_weight = jax_array_from_reshaped_torch(
                torch_weight,
                reshape_dims=reshape_dims,
                permute_dims=tuple(range(len(staging_shape))))
            if tuple(jax_weight.shape) != staging_shape:
                raise ValueError(
                    f"Converted MXFP4 tensor {torch_name} has shape "
                    f"{tuple(jax_weight.shape)}, expected staging shape "
                    f"{staging_shape} for {attr_name}.")
            jax_param.set_raw_value(jax_weight)
            jax_param.set_metadata("_is_loaded", True)
            loaded_names.add(attr_name)

        if unexpected:
            raise ValueError(
                "Mxfp4FusedMoEMethod only handles GPT-OSS MXFP4 expert "
                f"tensors, got unexpected checkpoint tensors: {unexpected}")

        logger.debug(
            f"Loaded {len(loaded_names)} MXFP4 tensors for {layer.prefix} MoE layer."
        )

        return loaded_names

    def create_weights_jax(self, layer: JaxRoutedExperts, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """
        Create the quant method-specific weights.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to create weights for.
        """
        if layer.moe_backend in MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            E = layer.num_local_experts
            D = layer.hidden_size
            F = layer.intermediate_size_moe

            for param_name in [
                    "kernel_gating_EDF", "kernel_up_proj_EDF",
                    "kernel_down_proj_EFD"
            ]:
                param = getattr(layer, param_name, None)
                assert isinstance(
                    param, nnx.Param
                ), f"Expected nnx.Param for {param_name}, got {type(param)}"
                delattr(layer, param_name)

            for param_name, shape, dtype in [
                ("w13_blocks", (E, 2 * F, D // 2), jnp.uint8),
                ("w13_scales", (E, 2 * F, D // 32), jnp.uint8),
                ("w13_bias", (E, 2 * F), layer.dtype),
                ("w2_blocks", (E, D, F // 2), jnp.uint8),
                ("w2_scales", (E, D, F // 32), jnp.uint8),
                ("w2_bias", (E, D), layer.dtype),
            ]:
                param = nnx.Param(jnp.zeros(shape, dtype=dtype),
                                  eager_sharding=False)
                param.set_metadata('mesh', cpu_mesh())
                setattr(layer, param_name, param)
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def process_weights_after_loading(self, layer: JaxRoutedExperts) -> bool:
        """
        Process the staged GPT-OSS MXFP4 expert tensors.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        """
        if layer.moe_backend in MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            if not all(
                    getattr(layer, name).get_metadata("_is_loaded", False)
                    for name in _GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS.values()):
                # A module's weights can be split across multiple files, so
                # this can get called more than once. Wait until all of them
                # are loaded before processing.
                return False

            w13_interleave = layer.activation == "swigluoai"
            w13_reorder_size = get_mesh_shape_product(
                layer.mesh, ShardingAxisName.MLP_TENSOR)

            # Keep the large dequantized expert tensors off device until
            # sharding.
            with cpu_mesh_context():
                weights = FusedMoEWeights(
                    w13_weight=dequantize_tensor_from_mxfp4_packed(
                        layer.w13_blocks[...], layer.w13_scales[...], 2,
                        jnp.float32),
                    w13_weight_scale=None,
                    w13_bias=layer.w13_bias[...],
                    w2_weight=dequantize_tensor_from_mxfp4_packed(
                        layer.w2_blocks[...], layer.w2_scales[...], 2,
                        jnp.float32),
                    w2_weight_scale=None,
                    w2_bias=layer.w2_bias[...],
                )
                weights = quantize_moe_weights(weights,
                                               jnp.float4_e2m1fn,
                                               MXFP4_REQUANTIZED_BLOCK_SIZE,
                                               w13_interleave=w13_interleave)
                weights = process_moe_weights(
                    weights,
                    moe_backend=layer.moe_backend,
                    w13_reorder_size=w13_reorder_size,
                    w13_interleave=w13_interleave)

                for name in _GPT_OSS_MXFP4_EXPERT_TENSOR_ATTRS.values():
                    delattr(layer, name)

            weights = shard_moe_weights(weights,
                                        moe_backend=layer.moe_backend,
                                        mesh=layer.mesh)

            layer.kernel_gating_upproj_EDF = nnx.Param(weights.w13_weight)
            layer.kernel_gating_upproj_EDF_weight_scale = nnx.Param(
                weights.w13_weight_scale)
            layer.kernel_gating_upproj_EDF_bias = nnx.Param(weights.w13_bias)
            layer.kernel_down_proj_EFD = nnx.Param(weights.w2_weight)
            layer.kernel_down_proj_EFD_weight_scale = nnx.Param(
                weights.w2_weight_scale)
            layer.kernel_down_proj_EFD_bias = nnx.Param(weights.w2_bias)
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """
        Run the forward pass of the MoE layer.

        Args:
            layer: The layer to apply the quantization method to.
            x: The input to the layer.

        Returns:
            The MoE output.
        """
        assert isinstance(layer, (JaxMoE, JaxRoutedExperts))

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))

        if layer.moe_backend in MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            weights = FusedMoEWeights(
                w13_weight=layer.kernel_gating_upproj_EDF[...],
                w13_weight_scale=layer.kernel_gating_upproj_EDF_weight_scale[
                    ...],
                w13_bias=layer.kernel_gating_upproj_EDF_bias[...],
                w2_weight=layer.kernel_down_proj_EFD[...],
                w2_weight_scale=layer.kernel_down_proj_EFD_weight_scale[...],
                w2_bias=layer.kernel_down_proj_EFD_bias[...],
            )
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {MXFP4_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


MXFP4_PACK_QUANTIZED_FORMAT = "mxfp4-pack-quantized"

# When the previous MoE layer's decode finished, so each layer can report how
# long it waited for its own weights to stream in. Loading is single-threaded
# per process, so one module-level slot is enough.
_LAST_LAYER_FINISHED_AT = 0.0

# Compiled per-shard decodes, keyed by (decode fn, shard shape, device); see
# `_decode_executable`.
_DECODE_EXECUTABLES: dict = {}

# compressed-tensors names each expert projection separately, and checkpoints
# use either the HF `wN` spelling or the expanded one. `w1`/`gate_proj` is the
# gate branch, `w3`/`up_proj` the up branch, `w2`/`down_proj` the down branch.
_CT_EXPERT_PROJECTIONS = {
    "w1": "gating",
    "gate_proj": "gating",
    "w3": "up_proj",
    "up_proj": "up_proj",
    "w2": "down_proj",
    "down_proj": "down_proj",
}

# Staging attribute names, keyed by the projection above.
_CT_STAGED_ATTRS = {
    "gating": ("gate_packed", "gate_scale"),
    "up_proj": ("up_packed", "up_scale"),
    "down_proj": ("down_packed", "down_scale"),
}

# Final parameter names, keyed by the projection above.
_CT_FINAL_ATTRS = {
    "gating": "kernel_gating_EDF",
    "up_proj": "kernel_up_proj_EDF",
    "down_proj": "kernel_down_proj_EFD",
}


def _named_sharding(sharding, mesh: Mesh) -> NamedSharding:
    """Normalize the spellings `shard_put` accepts into a `NamedSharding`."""
    if isinstance(sharding, NamedSharding):
        return sharding
    if isinstance(sharding, P):
        return NamedSharding(mesh, sharding)
    return NamedSharding(mesh, P(*(sharding or ())))


def _full_slice(s: slice | None, size: int) -> slice:
    """A concrete `[start, stop)` for one axis of a device's index tuple."""
    if s is None:
        return slice(0, size)
    assert s.step in (None, 1), f"unexpected strided shard index {s}"
    return slice(0 if s.start is None else s.start,
                 size if s.stop is None else s.stop)


@functools.lru_cache(maxsize=None)
def _one_device_mesh(device: jax.Device) -> Mesh:
    """A mesh holding just `device`.

    Each shard is decoded by itself, but the loader runs inside a
    `jax.set_mesh` of the *whole* model mesh, and jitting a single-device
    computation under that context is rejected ("Received incompatible
    devices ... jit's context mesh with device ids [0..N]"). Entering this
    mesh for the decode says which single device the computation belongs to.
    """
    return Mesh(np.array([device]).reshape(1), ("mxfp4_shard", ))


def _decode_executable(decode: Callable, shape: tuple[int, ...],
                       device: jax.Device):
    """A compiled `[e, out, in_] -> [e, in, out]` decode for one device.

    Held as an already-compiled executable rather than a `jax.jit` wrapper
    because the loader calls `jax.clear_caches()` after every module, which
    would otherwise re-trace and re-compile these for each of the model's
    layers -- the shard shapes repeat, so one compile per shape is enough.
    """
    key = (decode, shape, device.id)
    executable = _DECODE_EXECUTABLES.get(key)
    if executable is None:
        aval = jax.ShapeDtypeStruct(shape,
                                    jnp.uint8,
                                    sharding=SingleDeviceSharding(device))
        with jax.set_mesh(_one_device_mesh(device)):
            executable = jax.jit(lambda source: jnp.swapaxes(
                decode(source), 1, 2)).lower(aval).compile()
        _DECODE_EXECUTABLES[key] = executable
    return executable


def _decode_sharded(staged: list[jax.Array], decode: Callable, expansion: int,
                    sharding, mesh: Mesh) -> jax.Array:
    """Build one decoded, sharded expert tensor a device shard at a time.

    `staged` holds this layer's per-expert checkpoint tensors, each
    `[1, out, in_]` uint8, where `decode` expands the last axis by
    `expansion` (2 for the fp4 nibble unpack, 1 for the E8M0 scales). The
    result is `[E, in_ * expansion, out]` laid out over `mesh` as `sharding`
    -- the same array the host decode produces, assembled shard by shard:
    each device's slice is cut out of the *packed* bytes, sent to that
    device, and decoded there.

    Doing it in this order is what makes it cheap. The decode is elementwise
    along the packed axis and every shard boundary below falls on a whole
    uint8 word, so a shard decoded on its own is bit-for-bit the shard the
    whole-layer decode would have produced -- while the host never
    materializes the ~4.5x-larger decoded layer, only packed bytes cross the
    host/device boundary, and each process touches just the experts its own
    devices keep instead of all of them.
    """
    num_experts = len(staged)
    _, out, packed_in = staged[0].shape
    shape = (num_experts, packed_in * expansion, out)
    sharding = _named_sharding(sharding, mesh)

    # numpy views of the staged tensors: the per-shard gather below is pure
    # slicing and concatenation, which numpy does without a jax dispatch per
    # expert (there are hundreds of experts per shard).
    source = [np.asarray(w) for w in staged]

    shards = []
    for device, index in sharding.addressable_devices_indices_map(
            shape).items():
        expert_slice = _full_slice(index[0], num_experts)
        in_slice = _full_slice(index[1], packed_in * expansion)
        out_slice = _full_slice(index[2], out)
        if in_slice.start % expansion or in_slice.stop % expansion:
            raise ValueError(
                f"[mxfp4-ct] shard {in_slice} of the {shape} decoded tensor "
                f"splits a packed uint8 word ({expansion} values each); "
                f"cannot decode this shard on its own.")
        packed_slice = slice(in_slice.start // expansion,
                             in_slice.stop // expansion)
        # `[e, out, in_]` for the experts this device keeps, still packed.
        local = np.concatenate([
            w[:, out_slice, packed_slice]
            for w in source[expert_slice.start:expert_slice.stop]
        ],
                               axis=0)
        executable = _decode_executable(decode, local.shape, device)
        with jax.set_mesh(_one_device_mesh(device)):
            shards.append(executable(jax.device_put(local, device)))
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)


class CompressedTensorsMxfp4MoEMethod(QuantizeMethodBase, FusedMoEMethodBase):
    """MXFP4 routed experts stored in compressed-tensors `mxfp4-pack-quantized`.

    Unlike the GPT-OSS MXFP4 path above, this decode is **lossless**: the
    checkpoint's fp4 codes and E8M0 group scales are carried through unchanged
    (nibbles bitcast to `float4_e2m1fn`, exponents expanded to the exact
    powers of two they denote) instead of being dequantized to fp32 and
    re-quantized at a larger block size. Kimi-K3's experts are
    quantization-aware-trained, so recomputing the scales would move the
    weights away from the values training converged on.

    Layout differences from GPT-OSS handled here:
      * one tensor per expert per projection (`experts.<i>.w1.weight_packed`)
        rather than one stacked `gate_up_proj_blocks` per layer, so the expert
        axis is built by concatenation while loading;
      * the checkpoint stores `[out, in]` with the *input* axis packed, while
        the expert kernels are `(E, in, out)`, so the decoded values and their
        scales are transposed once at the end of loading.
    """

    def __init__(self, layer: JaxMoE, weight_quant=None):
        FusedMoEMethodBase.__init__(self, layer.moe_backend, "model")
        self.group_size = getattr(weight_quant, "group_size",
                                  None) or MXFP4_BLOCK_SIZE
        num_bits = getattr(weight_quant, "num_bits", 4)
        strategy = getattr(weight_quant, "strategy", "group")
        if num_bits != 4 or str(strategy).endswith("tensor"):
            raise NotImplementedError(
                f"[mxfp4-ct] {layer.prefix}: only 4-bit group-wise MXFP4 is "
                f"supported, got num_bits={num_bits} strategy={strategy}.")
        if layer.moe_backend in MoEBackend.fused_moe_backends():
            # The fused GMM backends pick experts from the raw router logits
            # themselves, which drops the router's expert-score correction
            # bias, and they have no `situ` activation branch -- both of which
            # the Kimi models need. Fail loudly rather than serve wrong tokens.
            raise NotImplementedError(
                f"[mxfp4-ct] {layer.prefix}: compressed-tensors MXFP4 experts "
                f"are wired for the unfused MoE backends (DENSE_MAT, "
                f"MEGABLX_GMM), got {layer.moe_backend}.")

    def create_weights_jax(self, layer: JaxMoE, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """Replace the bf16 expert kernels with packed MXFP4 staging tensors."""
        E = layer.num_local_experts
        D = layer.hidden_size
        F = layer.intermediate_size_moe
        gs = self.group_size
        for dim, name in ((D, "hidden_size"), (F, "intermediate_size_moe")):
            if dim % gs:
                raise ValueError(
                    f"[mxfp4-ct] {layer.prefix}: {name}={dim} is not a "
                    f"multiple of the MXFP4 group size {gs}.")

        for param_name in _CT_FINAL_ATTRS.values():
            delattr(layer, param_name)

        # Checkpoint orientation, expert axis prepended: [E, out, in].
        for projection, (out, in_) in (("gating", (F, D)), ("up_proj", (F, D)),
                                       ("down_proj", (D, F))):
            packed_attr, scale_attr = _CT_STAGED_ATTRS[projection]
            for attr, shape in ((packed_attr, (E, out, in_ // 2)),
                                (scale_attr, (E, out, in_ // gs))):
                param = nnx.Param(jnp.zeros(shape, dtype=jnp.uint8),
                                  eager_sharding=False)
                param.set_metadata('mesh', cpu_mesh())
                param.set_metadata('_weights_to_load', [None] * E)
                setattr(layer, attr, param)

    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,
                     weights: Iterable[tuple[str, torch.Tensor]]) -> set:
        """Stage per-expert `weight_packed` / `weight_scale` tensors."""
        loaded_names = set()
        for torch_name, torch_weight in weights:
            parts = torch_name.split(".")
            if len(parts) < 3:
                raise ValueError(
                    f"[mxfp4-ct] {layer.prefix}: cannot parse expert tensor "
                    f"name '{torch_name}'; expected "
                    f"<...>.<expert_id>.<projection>.<weight_packed|weight_scale>."
                )
            kind, projection_name, expert_id = parts[-1], parts[-2], parts[-3]
            projection = _CT_EXPERT_PROJECTIONS.get(projection_name)
            if projection is None or kind not in ("weight_packed",
                                                  "weight_scale"):
                raise ValueError(
                    f"[mxfp4-ct] {layer.prefix}: unexpected checkpoint tensor "
                    f"'{torch_name}'. Expected one of "
                    f"{sorted(_CT_EXPERT_PROJECTIONS)} carrying weight_packed "
                    f"or weight_scale.")
            packed_attr, scale_attr = _CT_STAGED_ATTRS[projection]
            attr = packed_attr if kind == "weight_packed" else scale_attr
            jax_param = getattr(layer, attr)
            slot = int(expert_id)
            # `[None] + shape` so the expert axis exists for concatenation.
            jax_param._weights_to_load[slot] = jax_array_from_reshaped_torch(
                torch_weight, reshape_dims=(1, ) + tuple(torch_weight.shape))
            loaded_names.add(attr)

        logger.debug(f"Staged {len(loaded_names)} MXFP4 tensor groups for "
                     f"{layer.prefix} MoE layer.")
        return loaded_names

    def process_weights_after_loading(self, layer: JaxMoE) -> bool:
        """Decode the staged tensors into device-resident fp4 + fp32 scales."""
        staged_attrs = [a for pair in _CT_STAGED_ATTRS.values() for a in pair]
        if not all(
                all(w is not None
                    for w in getattr(layer, attr)._weights_to_load)
                for attr in staged_attrs):
            # Expert weights can be split across safetensors files, so this is
            # called more than once; wait until every expert has arrived.
            return False

        started = time.perf_counter()
        global _LAST_LAYER_FINISHED_AT
        # How long this layer's weights took to stream in, for everything
        # after the first: loading is a single pass, so the gap since the
        # previous layer's decode is time spent reading the checkpoint.
        streamed_in = (f"{started - _LAST_LAYER_FINISHED_AT:.1f}s"
                       if _LAST_LAYER_FINISHED_AT else "the first layer")

        mesh = layer.mesh
        shard_then_decode = envs.MXFP4_SHARD_THEN_DECODE
        for projection, (packed_attr, scale_attr) in _CT_STAGED_ATTRS.items():
            final_attr = _CT_FINAL_ATTRS[projection]
            sharding = (layer.efd_sharding
                        if projection == "down_proj" else layer.edf_sharding)
            staged_packed = getattr(layer, packed_attr)._weights_to_load
            staged_scale = getattr(layer, scale_attr)._weights_to_load
            if shard_then_decode:
                # [E, out, in//2] uint8 -> [E, in, out] fp4, and
                # [E, out, in//group] E8M0 -> [E, in//group, out] fp32, one
                # device shard at a time.
                values = _decode_sharded(staged_packed, u8_unpack_e2m1, 2,
                                         sharding, mesh)
                scale = _decode_sharded(staged_scale, e8m0_to_fp32, 1,
                                        sharding, mesh)
            else:
                values, scale = self._decode_on_host(staged_packed,
                                                     staged_scale, sharding,
                                                     mesh)
            setattr(layer, final_attr, nnx.Param(values))
            setattr(layer, f"{final_attr}_weight_scale", nnx.Param(scale))
            delattr(layer, packed_attr)
            delattr(layer, scale_attr)

        _LAST_LAYER_FINISHED_AT = time.perf_counter()
        logger.info(
            "[mxfp4-ct] %s: decoded %d MXFP4 expert projections losslessly "
            "(group size %d, checkpoint scales passed through) in %.1fs "
            "[shard_then_decode=%s; waited %s for these weights to stream "
            "in].", layer.prefix, len(_CT_STAGED_ATTRS), self.group_size,
            _LAST_LAYER_FINISHED_AT - started, shard_then_decode, streamed_in)
        return True

    @staticmethod
    def _decode_on_host(staged_packed: list[jax.Array],
                        staged_scale: list[jax.Array], sharding,
                        mesh) -> tuple[jax.Array, jax.Array]:
        """Decode a whole layer on the host, then cut it into shards.

        Materializes ~4.5x the packed bytes per projection (fp4 values are a
        byte each on the host and the E8M0 scales expand to fp32) and decodes
        every expert on every process, so `_decode_sharded` above is the
        default. Kept as a fallback for the same numbers by a second route.
        """
        # Keep the decode off device until the sharding is applied.
        with cpu_mesh_context():
            packed = jnp.concatenate(staged_packed, axis=0)
            scale_u8 = jnp.concatenate(staged_scale, axis=0)
            # [E, out, in//2] uint8 -> [E, out, in] fp4 -> [E, in, out].
            values = jnp.swapaxes(u8_unpack_e2m1(packed), 1, 2)
            # [E, out, in//group] E8M0 -> exact fp32 powers of two, then
            # [E, in//group, out] to match the kernel's (E, K, N) weights.
            scale = jnp.swapaxes(e8m0_to_fp32(scale_u8), 1, 2)
        # `shard_put`, not `jax.device_put(x, NamedSharding(mesh, ...))`.
        # Under the Ray multi-host backend a process addresses only its own
        # devices, so naming a sharding over the whole mesh is rejected ("must
        # be a Device or a Sharding which represents addressable devices").
        # `shard_put` routes to `general_device_put`, which assembles the
        # global array out of each process's addressable shards instead. The
        # decode above runs under `cpu_mesh_context`, so `values`/`scale` are
        # process-local and fully addressable -- exactly the case that needs
        # it. Same helper the unquantized expert path already uses.
        return (shard_put(values, sharding,
                          mesh=mesh), shard_put(scale, sharding, mesh=mesh))

    def apply_jax(self, layer: JaxMoE, x: jax.Array, *,
                  router_logits) -> jax.Array:
        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))
        weights = UnfusedMoEWeights(
            w1_weight=layer.kernel_gating_EDF.value,
            w1_weight_scale=layer.kernel_gating_EDF_weight_scale.value,
            w1_bias=None,
            w2_weight=layer.kernel_up_proj_EDF.value,
            w2_weight_scale=layer.kernel_up_proj_EDF_weight_scale.value,
            w2_bias=None,
            w3_weight=layer.kernel_down_proj_EFD.value,
            w3_weight_scale=layer.kernel_down_proj_EFD_weight_scale.value,
            w3_bias=None,
        )
        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


class Mxfp4Config(QuantizationConfig):
    """Quantization config for GPT-OSS MXFP4 routed experts.

    GPT-OSS MXFP4 has no configurable JAX-side options yet.
    """

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxRoutedExperts):
            return Mxfp4FusedMoEMethod()
        return None
