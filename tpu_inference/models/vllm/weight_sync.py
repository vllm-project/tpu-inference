# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""In-place weight updates for the vLLM (torchax) model path.

The torchax model runner keeps its weights in a flat ``{name: jax.Array}``
state dict whose layout is an internal detail of tpu-inference: linear
kernels are stored transposed (``[in, out]``) with fused projections
reordered per tensor-parallel shard (`process_linear_weights`), fused MoE
kernels go through the backend-specific `process_moe_weights` (transposes,
per-shard padding), and KV heads are replicated when ``tp_size`` exceeds the
number of KV heads.  None of that should leak to a trainer that wants to push
new weights (RL weight sync).

This module defines the trainer-facing contract as vLLM's *canonical*
parameter layout -- what ``model.named_parameters()`` holds on a single GPU
at TP=1 (``[out, in]`` linears, ``qkv_proj`` = ``[q | k | v]`` with
unreplicated KV heads, ``w13_weight`` = ``[E, 2F, D]`` gate-first,
``w2_weight`` = ``[E, D, F]``) -- and re-runs tpu-inference's own weight
processing to turn canonical arrays into the internal layout:

    specs = wrapper.canonical_weight_specs()      # name -> ShapeDtypeStruct
    wrapper.load_canonical_weights(new_weights, state)

Names are vLLM module paths (``language_model.model.layers.0.self_attn.
qkv_proj.weight``); the ``vllm_model.`` prefix of the runner state is
accepted but optional.
"""
from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import torch
from jax.sharding import NamedSharding, PartitionSpec
from torchax.interop import jax_view, torch_view
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, QKVParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding

from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights)
from tpu_inference.layers.common.process_weights.moe_weights import (
    process_unquantized_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.common.utils import general_device_put
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedEmbeddingMethod, VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

STATE_PREFIX = "vllm_model."

P = PartitionSpec


def _strip_prefix(name: str) -> str:
    return name[len(STATE_PREFIX):] if name.startswith(STATE_PREFIX) else name


def _state_entry(state: Mapping[str, Any], name: str) -> jax.Array:
    key = STATE_PREFIX + _strip_prefix(name)
    if key not in state:
        raise KeyError(f"{key} is not a parameter of the torchax model state")
    return jax_view(state[key])


def _linear_canonical_out(layer: LinearBase) -> int:
    if isinstance(layer, QKVParallelLinear):
        # `output_sizes` already contain the tp-replicated KV heads; the
        # canonical layout carries each KV head once.
        return (layer.total_num_heads +
                2 * layer.total_num_kv_heads) * layer.head_size
    return layer.output_size


def _replicate_kv_heads(layer: QKVParallelLinear,
                        weight: jax.Array) -> jax.Array:
    """[q | k | v] with unique KV heads -> [q | k*r | v*r] as vLLM lays out
    ``qkv_proj`` when tp_size > num_kv_heads (each KV head repeated
    `num_kv_head_replicas` times consecutively, one copy per TP rank)."""
    r = layer.num_kv_head_replicas
    if r == 1:
        return weight
    hs = layer.head_size
    q_size = layer.total_num_heads * hs
    kv_size = layer.total_num_kv_heads * hs
    q = weight[:q_size]
    k = weight[q_size:q_size + kv_size]
    v = weight[q_size + kv_size:q_size + 2 * kv_size]

    def rep(x):
        x = x.reshape(layer.total_num_kv_heads, hs, *x.shape[1:])
        x = jnp.repeat(x, r, axis=0)
        return x.reshape(layer.total_num_kv_heads * r * hs, *x.shape[2:])

    return jnp.concatenate([q, rep(k), rep(v)], axis=0)


def _on_runner_devices(arr: jax.Array, target: jax.sharding.Sharding,
                       name: str) -> jax.Array:
    """Make sure `arr` can be resharded onto the runner mesh with device_put.

    Single-host `jax.device_put` reshards between two meshes only when they
    list the same devices in the same order. The runner builds its mesh from
    the physical topology, so a trainer mesh over the same chips may well
    order them differently. In that case: replicate on the source mesh
    (same-mesh reshard), re-express the per-device buffers on the runner mesh
    (no copy), and let the caller shard from there. Transient cost: one
    replicated copy of the tensor.
    """
    src = getattr(arr, "sharding", None)
    src_mesh = getattr(src, "mesh", None)
    tgt_mesh = getattr(target, "mesh", None)
    if src_mesh is None or tgt_mesh is None:
        return arr
    src_ids = [d.id for d in src_mesh.devices.flat]
    tgt_ids = [d.id for d in tgt_mesh.devices.flat]
    if src_ids == tgt_ids or len(src_ids) == 1:
        return arr
    if sorted(src_ids) != sorted(tgt_ids):
        raise ValueError(
            f"{name}: source array lives on devices {src_ids}, the runner "
            f"mesh uses {tgt_ids}; cross-device-set weight sync needs a "
            "transport (Pathways reshard / Raiden) before load_canonical_weights.")
    if not arr.is_fully_addressable:
        raise ValueError(
            f"{name}: source mesh order {src_ids} differs from the runner "
            f"mesh {tgt_ids} and the array is not fully addressable; reshard "
            "it onto the runner mesh before load_canonical_weights.")
    logger.info_once(
        f"Source mesh device order {src_ids} differs from the runner mesh "
        f"{tgt_ids}; re-expressing arrays on the runner mesh via replication.")
    # A jitted identity keeps the all-gather on device (plain device_put may
    # take the host slow path for sharded -> replicated).
    replicated = jax.jit(lambda x: x,
                         out_shardings=NamedSharding(src_mesh, P()))(arr)
    replicated.block_until_ready()
    by_device = {s.device.id: s.data for s in replicated.addressable_shards}
    return jax.make_array_from_single_device_arrays(
        replicated.shape, NamedSharding(tgt_mesh, P()),
        [by_device[d.id] for d in tgt_mesh.devices.flat])


def canonical_weight_specs(
        vllm_model: torch.nn.Module,
        state: Mapping[str, Any]) -> dict[str, jax.ShapeDtypeStruct]:
    """Canonical (vLLM TP=1) shape/dtype for every parameter in `state`."""
    specs: dict[str, jax.ShapeDtypeStruct] = {}
    modules = dict(vllm_model.named_modules())
    for key in state:
        name = _strip_prefix(key)
        mod_path, _, pname = name.rpartition(".")
        layer = modules.get(mod_path)
        arr = jax_view(state[key])
        shape = tuple(arr.shape)
        qm = getattr(layer, "quant_method", None)
        if isinstance(layer, LinearBase) and isinstance(
                qm, VllmUnquantizedLinearMethod):
            out = _linear_canonical_out(layer)
            if pname == "weight":
                shape = (out, layer.input_size)
            elif pname == "bias":
                shape = (out, )
            else:
                continue
        elif isinstance(layer, RoutedExperts) and isinstance(
                qm, VllmUnquantizedFusedMoEMethod):
            e = layer.global_num_experts
            d = layer.hidden_size
            f = layer.moe_config.intermediate_size
            if pname == "w13_weight":
                shape = (e, 2 * f, d)
            elif pname == "w2_weight":
                shape = (e, d, f)
            elif pname == "w13_bias":
                shape = (e, 2 * f)
            elif pname == "w2_bias":
                shape = (e, d)
            else:
                continue
        # Embeddings, replicated linears (vLLM's default method), norms,
        # conv1d, A_log, ...: the internal layout is the canonical one.
        specs[name] = jax.ShapeDtypeStruct(shape, arr.dtype)
    return specs


def _process_linear(layer: LinearBase, qm: VllmUnquantizedLinearMethod,
                    weight: jax.Array | None,
                    bias: jax.Array | None) -> LinearWeights:
    cfg = qm.linear_config
    if not cfg.fuse_matmuls:
        raise NotImplementedError(
            f"{layer}: weight sync into unfused (split) linear weights is not "
            "supported yet; the layer keeps one processed tensor per "
            "projection.")
    if isinstance(layer, QKVParallelLinear):
        if weight is not None:
            weight = _replicate_kv_heads(layer, weight)
        if bias is not None:
            bias = _replicate_kv_heads(layer, bias)
    if weight is not None:
        weight = jnp.transpose(weight)  # [out, in] -> [in, out]
    weights = process_linear_weights(
        LinearWeights(weight=weight,
                      weight_scale=None,
                      zero_point=None,
                      bias=bias),
        fused=True,
        output_sizes=cfg.output_sizes,
        reorder_size=cfg.n_shards,
    )
    return shard_linear_weights(weights,
                                mesh=cfg.mesh,
                                weight_p_spec=cfg.weight_sharding,
                                bias_p_spec=cfg.bias_sharding)


def _process_moe(layer: RoutedExperts, qm: VllmUnquantizedFusedMoEMethod,
                 w13: jax.Array, w2: jax.Array, w13_bias: jax.Array | None,
                 w2_bias: jax.Array | None):
    weights = process_unquantized_moe_weights(mesh=qm.mesh,
                                              moe_backend=qm.moe_backend,
                                              activation=layer.activation,
                                              w13_weight=w13,
                                              w13_bias=w13_bias,
                                              w2_weight=w2,
                                              w2_bias=w2_bias)
    return shard_moe_weights(weights, qm.moe_backend, qm.mesh)


def _flush_moe(vllm_model: torch.nn.Module,
               modules: dict[str, torch.nn.Module], state: dict[str, Any],
               mod_path: str, layer: RoutedExperts,
               qm: VllmUnquantizedFusedMoEMethod,
               parts: dict[str, jax.Array]) -> list[str]:
    """Re-expresses, processes and installs one MoE layer's expert weights."""
    ready = {}
    for pname, arr in parts.items():
        name = f"{mod_path}.{pname}"
        old = jax_view(state[STATE_PREFIX + name])
        ready[pname] = _on_runner_devices(jnp.asarray(arr), old.sharding,
                                          name).astype(old.dtype)
    processed = _process_moe(layer, qm, ready["w13_weight"],
                             ready["w2_weight"], ready.get("w13_bias"),
                             ready.get("w2_bias"))
    del ready
    updated = []
    for pname in ("w13_weight", "w2_weight", "w13_bias", "w2_bias"):
        if pname in parts:
            name = f"{mod_path}.{pname}"
            _install(vllm_model, modules, state, name,
                     getattr(processed, pname))
            updated.append(STATE_PREFIX + name)
    jax.effects_barrier()
    return updated


def _install(vllm_model: torch.nn.Module, modules: dict[str, torch.nn.Module],
             state: dict[str, Any], name: str, new: jax.Array) -> None:
    key = STATE_PREFIX + name
    old = jax_view(state[key])
    if tuple(new.shape) != tuple(old.shape) or new.dtype != old.dtype:
        raise ValueError(
            f"{name}: processed weight has shape {new.shape}/{new.dtype}, "
            f"model state expects {old.shape}/{old.dtype}")
    if getattr(new, "sharding", None) != old.sharding:
        # Same spec on the same mesh is what the compiled step expects; a
        # different placement would trigger a recompile (or fail).
        new = general_device_put(new, old.sharding)
    # Bound the transient HBM: without this, async dispatch keeps enqueueing
    # per-tensor intermediates (replicated / pre-shard copies) for the whole
    # model before any of them is freed.
    new.block_until_ready()
    # The runner state holds plain jax arrays (`jax_view(params_and_buffers)`).
    state[key] = new
    # Re-point the module parameter too, so the old buffer is released and
    # anything reading `layer.<param>` directly (the GDN op reads conv1d /
    # A_log / dt_bias off the module) sees the new array.
    mod_path, _, pname = name.rpartition(".")
    layer = modules.get(mod_path)
    if layer is not None and pname in layer._parameters:
        # `.data = ...` is rejected across the torchax tensor type; replace the
        # Parameter object like the loading path does.
        layer._parameters[pname] = torch.nn.Parameter(torch_view(new),
                                                      requires_grad=False)


def load_canonical_weights(vllm_model: torch.nn.Module,
                           weights: Mapping[str, jax.Array],
                           state: dict[str, Any],
                           strict: bool = False) -> list[str]:
    """Convert canonical-layout arrays to the internal layout and write them
    into `state` (the runner's flat params dict) in place.

    Args:
        vllm_model: the wrapped ``torch.nn.Module`` (``wrapper.model.vllm_model``).
        weights: canonical name -> jax.Array (any sharding / mesh; cast to the
            state dtype here).  Names may carry the ``vllm_model.`` prefix.
        state: the runner's ``{name: array}`` state dict, updated in place.
        strict: raise if a parameter of the model is not covered by `weights`
            (attention scale scalars are always exempt).

    Returns:
        The list of state names that were updated.
    """
    modules = dict(vllm_model.named_modules())
    pending_moe: dict[str, dict[str, jax.Array]] = {}
    updated: list[str] = []

    def _hbm_gib() -> str:
        try:
            return " ".join(
                f"{d.memory_stats()['bytes_in_use'] / 2**30:.1f}"
                for d in jax.local_devices())
        except Exception:  # pragma: no cover - stats are best effort
            return "n/a"

    logger.info("load_canonical_weights: %d incoming tensors; HBM in use "
                "per device (GiB): %s", len(weights), _hbm_gib())

    for raw_name, arr in weights.items():
        name = _strip_prefix(raw_name)
        key = STATE_PREFIX + name
        if key not in state:
            msg = f"{name} is not a parameter of the torchax model state"
            if strict:
                raise KeyError(msg)
            logger.warning(msg)
            continue
        mod_path, _, pname = name.rpartition(".")
        layer = modules.get(mod_path)
        qm = getattr(layer, "quant_method", None)
        old = jax_view(state[key])

        if isinstance(layer, RoutedExperts) and isinstance(
                qm, VllmUnquantizedFusedMoEMethod):
            # Keep the caller's array as is until the layer's w13/w2 pair is
            # complete, then process the layer right away: the per-tensor
            # re-expression below may hold a replicated copy, and parking
            # those for every MoE layer does not fit in HBM.
            parts = pending_moe.setdefault(mod_path, {})
            parts[pname] = arr
            needed = {"w13_weight", "w2_weight"}
            if qm.moe.has_bias:
                needed |= {"w13_bias", "w2_bias"}
            if needed <= set(parts):
                updated.extend(
                    _flush_moe(vllm_model, modules, state, mod_path, layer,
                               qm, pending_moe.pop(mod_path)))
            continue

        arr = _on_runner_devices(jnp.asarray(arr), old.sharding, name)
        arr = arr.astype(old.dtype)

        if isinstance(layer, LinearBase) and isinstance(
                qm, VllmUnquantizedLinearMethod) and pname in ("weight",
                                                               "bias"):
            processed = _process_linear(layer, qm,
                                        arr if pname == "weight" else None,
                                        arr if pname == "bias" else None)
            new = processed.weight if pname == "weight" else processed.bias
            _install(vllm_model, modules, state, name, new)
            updated.append(key)
        elif isinstance(layer, VocabParallelEmbedding) and isinstance(
                qm, VllmUnquantizedEmbeddingMethod):
            sharding = NamedSharding(qm.mesh, P(ShardingAxisName.MLP_TENSOR,
                                                None))
            _install(vllm_model, modules, state, name,
                     general_device_put(arr, sharding))
            updated.append(key)
        else:
            # Layout-preserving parameters: keep the state's own placement.
            old = jax_view(state[key])
            _install(vllm_model, modules, state, name,
                     general_device_put(arr, old.sharding))
            updated.append(key)

    if pending_moe:
        incomplete = {k: sorted(v) for k, v in pending_moe.items()}
        raise ValueError(
            "fused MoE layers need w13_weight and w2_weight (and the biases "
            f"when present) in the same update; incomplete: {incomplete}")

    updated_set = set(updated)
    missing = [
        k for k in state if k not in updated_set
        and not k.rsplit(".", 1)[-1].startswith("_")  # attn _k_scale etc.
    ]
    if missing:
        msg = (f"{len(missing)} model parameters were not updated, e.g. "
               f"{missing[:5]}")
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    logger.info(
        "Updated %d/%d torchax model parameters in place; HBM in use per "
        "device (GiB): %s", len(updated), len(state), _hbm_gib())
    return updated
