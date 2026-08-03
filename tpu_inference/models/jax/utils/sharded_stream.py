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
"""Per-host sharded checkpoint reads for compressed-tensors MXFP4 experts.

The default runai-streamer load path makes every host read every byte of the
checkpoint and slice to its devices' shards only after the bytes arrived. For
a large MoE model the routed-expert tensors are the overwhelming majority of
those bytes (Kimi-K3: ~1.4 TB of a 1.45 TB checkpoint), and with the expert
axis sharded E-ways across hosts each host ultimately keeps only 1/E of them
-- so with 4 expert groups, ~3/4 of a ~110-minute cold-start read is spent on
bytes the host immediately throws away, and the aggregate object-store egress
is hosts x checkpoint size.

This module removes that waste at the request level, opt-in via
``K3_SHARDED_EXPERT_STREAMING=1``:

1. Derive the set of expert ids any of this process's devices keep from the
   expert kernels' sharding -- the same
   ``sharding.addressable_devices_indices_map`` primitive the shard-by-shard
   decode (``mxfp4._decode_sharded``) already trusts, so misalignment between
   the expert axis and hosts can only shrink the saving, never break
   correctness.
2. Stream the safetensors files through a filtered request list: the
   per-file header reads stay as they are (small ranged GETs), but every
   ``.experts.<id>.`` tensor whose id is not in the needed set is dropped
   from the byte ranges before they are requested. Non-expert tensors always
   stream. Surviving tensors are coalesced into contiguous runs, one
   ``FileChunks`` request per run -- the layout the streamer's own
   distributed partitioner emits, so no streamer-library change is needed.
3. Mark each MXFP4 MoE layer with its needed set so the decode gate in
   ``mxfp4.process_weights_after_loading`` waits on the local set instead of
   all experts (the rest will never arrive), and verify after the load that
   every layer actually decoded -- a needed expert the filter dropped fails
   loudly with its ids rather than as a half-loaded model.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import jax
from jax.sharding import Mesh

from tpu_inference import envs
from tpu_inference.layers.jax.quantization.mxfp4 import (
    _CT_STAGED_ATTRS, _SHARDED_STREAM_NEEDED_ATTR,
    CompressedTensorsMxfp4MoEMethod, _full_slice, _named_sharding)
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# `<...>.experts.<id>.<projection>.<weight_packed|weight_scale>` -- the name
# shape `mxfp4.load_weights` asserts. Anchored on the `.experts.` segment so
# router / shared-expert / non-MoE tensors never match.
_EXPERT_TENSOR_RE = re.compile(r"(?:^|\.)experts\.(\d+)\.")

_GiB = 1024**3


def expert_id_from_tensor_name(name: str) -> Optional[int]:
    """The routed-expert id in a checkpoint tensor name, or None."""
    match = _EXPERT_TENSOR_RE.search(name)
    return int(match.group(1)) if match else None


def needed_expert_ids(
        sharding,
        mesh: Mesh,
        num_experts: int,
        local_devices: Optional[Iterable[jax.Device]] = None) -> frozenset:
    """Expert ids some device of this host keeps a shard of.

    The union of the expert-axis (axis 0) slices in the sharding's
    device->index map, restricted to `local_devices` -- by default this
    process's addressable devices, exactly the devices
    `mxfp4._decode_sharded` will later cut shards for. Passing an explicit
    subset makes the derivation testable on a single host: a simulated
    host's device group yields the set that host would stream.
    """
    named = _named_sharding(sharding, mesh)
    # A stand-in shape for the index map: the expert axis is exact, and each
    # trailing axis is sized to the product of the mesh axes sharding it so
    # every slice divides evenly. Axis 0's slices depend only on
    # `num_experts` and the expert-axis device count, so they match what the
    # real staged shape would produce.
    dims = [num_experts]
    for axis_spec in tuple(named.spec)[1:]:
        if axis_spec is None:
            dims.append(1)
        else:
            names = axis_spec if isinstance(axis_spec,
                                            tuple) else (axis_spec, )
            size = 1
            for axis_name in names:
                size *= mesh.shape[axis_name]
            dims.append(size)
    shape = tuple(dims)

    if local_devices is None:
        index_map = named.addressable_devices_indices_map(shape)
    else:
        wanted = set(local_devices)
        index_map = {
            device: index
            for device, index in named.devices_indices_map(shape).items()
            if device in wanted
        }

    needed: set = set()
    for index in index_map.values():
        expert_slice = _full_slice(index[0], num_experts)
        needed.update(range(expert_slice.start, expert_slice.stop))
    return frozenset(needed)


@dataclass
class FilterStats:
    """What the checkpoint filter kept vs dropped, in tensors and bytes."""
    kept_tensors: int = 0
    total_tensors: int = 0
    kept_bytes: int = 0
    skipped_bytes: int = 0
    # Every dropped tensor name, so the post-load validation can prove each
    # one belonged to a layer whose decode gate knows about the filtering.
    dropped_names: list = field(default_factory=list)


def build_filtered_requests(paths: list[str], safetensors_metadatas: list,
                            keep: Callable[[str], bool]):
    """Filtered, coalesced streaming requests for the given files.

    `safetensors_metadatas` is `prepare_request`'s output: per file,
    `(data_start_offset, [SafetensorMetadata sorted by offset], [sizes])`.
    Dropped tensors split a file's single all-covering request into one
    request per contiguous run of kept tensors (the safetensors data region
    has no gaps, so a run of kept tensors is one contiguous byte range).
    Request ids are fresh and unique across files; the returned metadata map
    is keyed by them in chunk order, which is exactly what
    `SafetensorsStreamer.get_tensors` walks.

    Returns `(requests, id_to_tensors_metadata, stats)`.
    """
    from runai_model_streamer.file_streamer import FileChunks

    requests: list = []
    id_to_meta: dict = {}
    stats = FilterStats()

    for path, (file_offset, tensors_metadata,
               tensor_sizes) in zip(paths, safetensors_metadatas):
        run_meta: list = []
        run_sizes: list = []
        run_start = 0

        def flush():
            if not run_meta:
                return
            if sum(run_sizes) == 0:
                # The streamer's request iterator treats an all-empty request
                # as end-of-queue, so a run consisting solely of zero-byte
                # tensors cannot be submitted -- and silently dropping it
                # would starve `get_tensors` of those names. No real
                # checkpoint stores zero-byte expert neighbours; refuse
                # loudly rather than mis-stream.
                raise ValueError(
                    "[sharded-stream] a contiguous run of kept tensors is "
                    "entirely zero-byte, which the filtered streamer cannot "
                    "submit: "
                    f"{[m.name for m in run_meta][:8]}. Unset "
                    "K3_SHARDED_EXPERT_STREAMING to load this checkpoint "
                    "with the stock full read.")
            request_id = len(requests)
            requests.append(
                FileChunks(request_id, path, file_offset + run_start,
                           list(run_sizes)))
            id_to_meta[request_id] = list(run_meta)

        for meta, size in zip(tensors_metadata, tensor_sizes):
            stats.total_tensors += 1
            if keep(meta.name):
                if not run_meta:
                    run_start = meta.offsets.start
                run_meta.append(meta)
                run_sizes.append(size)
                stats.kept_tensors += 1
                stats.kept_bytes += size
            else:
                flush()
                run_meta, run_sizes = [], []
                stats.skipped_bytes += size
                stats.dropped_names.append(meta.name)
        flush()

    return requests, id_to_meta, stats


def _filtered_stream(streamer,
                     paths: list[str],
                     keep: Callable[[str], bool],
                     device: str = "cpu") -> FilterStats:
    """`SafetensorsStreamer.stream_files`, minus the filtered-out tensors.

    Runs the same per-file header reads (`prepare_request`), then submits
    only the kept tensors' byte ranges. `streamer.get_tensors` then works
    unchanged: it looks tensors up by `(request id, chunk index)`, which the
    filtered request map preserves.
    """
    from runai_model_streamer.safetensors_streamer import safetensors_pytorch

    streamer.files_to_tensors_metadata = {}
    metadatas = safetensors_pytorch.prepare_request(streamer.file_streamer,
                                                    paths, None)
    requests, id_to_meta, stats = build_filtered_requests(
        paths, metadatas, keep)
    streamer.files_to_tensors_metadata = id_to_meta
    streamer.file_streamer.stream_files(requests,
                                        credentials=None,
                                        device=device,
                                        is_distributed=False)
    return stats


def sharded_expert_weights_iterator(hf_weights_files: list[str],
                                    needed_experts: frozenset,
                                    num_experts: int,
                                    use_tqdm_on_load: bool = False,
                                    dropped_names_sink: Optional[list] = None):
    """Yield `(name, torch.Tensor)` like the stock runai iterator, but only
    reading the expert tensors this host's devices keep (plus every
    non-expert tensor).

    `dropped_names_sink`, when given, receives every dropped tensor name so
    the caller can validate after the load that each one belonged to a layer
    whose decode gate expects the filtering.
    """
    from runai_model_streamer import SafetensorsStreamer
    from tqdm.auto import tqdm

    def keep(name: str) -> bool:
        expert_id = expert_id_from_tensor_name(name)
        return expert_id is None or expert_id in needed_experts

    with SafetensorsStreamer() as streamer:
        stats = _filtered_stream(streamer, hf_weights_files, keep)
        if dropped_names_sink is not None:
            dropped_names_sink.extend(stats.dropped_names)
        logger.info(
            "[sharded-stream] process %d streams %d/%d routed experts: "
            "keeping %d/%d checkpoint tensors (%.1f GiB), skipping %.1f GiB "
            "of non-local expert bytes before they are read.",
            jax.process_index(), len(needed_experts), num_experts,
            stats.kept_tensors, stats.total_tensors, stats.kept_bytes / _GiB,
            stats.skipped_bytes / _GiB)
        tensor_iter = tqdm(streamer.get_tensors(),
                           total=stats.kept_tensors,
                           desc="Loading safetensors (sharded expert read)",
                           disable=not use_tqdm_on_load,
                           mininterval=2)
        yielded = 0
        for name, tensor in tensor_iter:
            yielded += 1
            # The streamer reuses its CPU buffer across requests; detach the
            # tensor from it before handing it out, as the stock iterator
            # does.
            yield name, tensor.clone()
        if yielded != stats.kept_tensors:
            # A silent shortfall here would surface much later as a
            # partially-initialized model (or not at all); convert any
            # request/metadata misalignment into a load failure.
            raise ValueError(
                f"[sharded-stream] the filtered stream yielded {yielded} "
                f"tensors but the request filter kept {stats.kept_tensors}; "
                f"the streamer dropped or duplicated part of the "
                f"checkpoint. Unset K3_SHARDED_EXPERT_STREAMING and reload.")


def _mxfp4_moe_layers(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(getattr(module, "quant_method", None),
                          CompressedTensorsMxfp4MoEMethod)]


def plan_sharded_expert_streaming(model) -> Optional[tuple[frozenset, int]]:
    """Mark every MXFP4 MoE layer with this host's needed expert set.

    Returns `(needed_expert_ids, num_experts)`, or None when the model has
    no compressed-tensors MXFP4 MoE layers (nothing worth filtering; other
    quant paths stage and require all experts). The union over the gate/up
    (EDF) and down (EFD) shardings is used for both the checkpoint filter
    and the per-layer decode gate, so anything the decode can touch is
    guaranteed to have streamed.
    """
    layers = _mxfp4_moe_layers(model)
    if not layers:
        return None

    # Two passes: compute every layer's set first, mark only if they all
    # agree. Marking any layer relaxes its decode gate, which is only sound
    # when the filter (driven by the shared set) actually runs -- so on
    # disagreement nothing may be marked and the caller falls back to the
    # stock full read, whose unrelaxed gates need no filter.
    needed: Optional[frozenset] = None
    num_experts: Optional[int] = None
    for name, layer in layers:
        layer_needed = needed_expert_ids(
            layer.edf_sharding, layer.mesh,
            layer.num_local_experts) | needed_expert_ids(
                layer.efd_sharding, layer.mesh, layer.num_local_experts)
        if needed is None:
            needed, num_experts = layer_needed, layer.num_local_experts
        elif (layer_needed, layer.num_local_experts) != (needed, num_experts):
            logger.warning(
                "[sharded-stream] MoE layers disagree on the local expert "
                "set (%s needs %d of %d experts, an earlier layer %d of %d); "
                "one checkpoint filter cannot serve both, falling back to "
                "the full checkpoint read.", name, len(layer_needed),
                layer.num_local_experts, len(needed), num_experts)
            return None
    for _, layer in layers:
        setattr(layer, _SHARDED_STREAM_NEEDED_ATTR, frozenset(needed))
    return needed, num_experts


def _marked_layer_prefixes(layers) -> list[str]:
    """Name paths under which dropped expert tensors are accounted for."""
    prefixes = []
    for module_name, layer in layers:
        for prefix in (getattr(layer, "prefix", ""), module_name):
            if prefix:
                prefixes.append(prefix)
    return prefixes


def _covered_by_marked_layer(dropped_name: str, prefixes: list[str]) -> bool:
    """Whether a dropped checkpoint tensor belongs to a marked MoE layer.

    The checkpoint namespace may carry an extra wrapper prefix the module
    tree does not (`language_model.` on Kimi-K3), so the tensor's path up to
    its expert id is suffix-matched against each marked layer's prefix.
    """
    match = _EXPERT_TENSOR_RE.search(dropped_name)
    if not match:
        return False
    # `...layers.3.block_sparse_moe.experts` -- the path up to the id.
    stem = dropped_name[:match.start(1) - 1]
    for prefix in prefixes:
        # The marked prefix may or may not spell the trailing `.experts`
        # segment itself, depending on how the module names it.
        candidates = [prefix]
        if not prefix.endswith(".experts"):
            candidates.append(prefix + ".experts")
        for candidate in candidates:
            if stem == candidate or stem.endswith("." + candidate):
                return True
    return False


def validate_sharded_streaming_complete(model, dropped_names=()) -> None:
    """Fail loudly if the filtered load left anything unaccounted for.

    Two checks, both after the weights iterator is exhausted:

    * every marked layer decoded its local experts -- the decode gate
      returns False while streaming is in flight, so a needed expert the
      filter wrongly dropped surfaces here as staging attributes that are
      still present (decode deletes them) with `None` in needed slots;
    * every DROPPED tensor name belongs to a marked layer. The name filter
      drops `.experts.<id>.` tensors model-wide, but only the marked
      (compressed-tensors MXFP4) layers have the relaxed gate that knows
      slots may stay empty -- an expert tensor of any other module would be
      silently missing (the auto-loader has no missing-weight check) and
      that module would serve its initialization values.
    """
    layers = _mxfp4_moe_layers(model)
    marked = [(name, layer) for name, layer in layers
              if getattr(layer, _SHARDED_STREAM_NEEDED_ATTR, None) is not None]
    prefixes = _marked_layer_prefixes(marked)
    unaccounted = [
        name for name in dropped_names
        if not _covered_by_marked_layer(name, prefixes)
    ]
    if unaccounted:
        raise ValueError(
            f"[sharded-stream] {len(unaccounted)} dropped checkpoint "
            f"tensors do not belong to any marked MXFP4 MoE layer (e.g. "
            f"{unaccounted[:8]}); their module would silently keep its "
            f"initialization weights. Marked layer prefixes: "
            f"{sorted(set(prefixes))[:8]}. Unset K3_SHARDED_EXPERT_STREAMING "
            f"for this model.")

    staged_attrs = [a for pair in _CT_STAGED_ATTRS.values() for a in pair]
    problems = []
    for name, layer in layers:
        needed = getattr(layer, _SHARDED_STREAM_NEEDED_ATTR, None)
        if needed is None:
            continue
        for attr in staged_attrs:
            param = getattr(layer, attr, None)
            if param is None:
                # Decode ran and deleted the staging tensor -- healthy.
                continue
            missing = sorted(i for i in needed
                             if param._weights_to_load[i] is None)
            if missing:
                problems.append(
                    f"{name}.{attr}: missing local expert ids "
                    f"{missing[:16]}{'...' if len(missing) > 16 else ''} "
                    f"({len(missing)} of {len(needed)} needed)")
            else:
                problems.append(
                    f"{name}.{attr}: all {len(needed)} local experts are "
                    f"staged but the decode never ran")
    if problems:
        raise ValueError(
            "[mxfp4] sharded expert streaming finished but these MoE layers "
            "never decoded -- experts local to this host are missing from "
            "the stream (filter bug or truncated checkpoint): " +
            "; ".join(problems))


def maybe_load_with_sharded_expert_streaming(loader, model,
                                             model_config) -> bool:
    """Load via the per-host filtered read when it is enabled and applicable.

    Returns True when the model was loaded here; False means the caller
    should fall back to the stock full-read path. Every bail-out is logged:
    an operator who set the flag should be able to see from the log alone
    why a host still read the whole checkpoint.
    """
    if not envs.K3_SHARDED_EXPERT_STREAMING:
        return False

    from vllm.model_executor.model_loader.runai_streamer_loader import \
        RunaiModelStreamerLoader

    if not isinstance(loader, RunaiModelStreamerLoader):
        logger.warning(
            "[sharded-stream] K3_SHARDED_EXPERT_STREAMING=1 but the model "
            "loader is %s, not RunaiModelStreamerLoader; falling back to the "
            "full checkpoint read.",
            type(loader).__name__)
        return False
    if getattr(loader, "_is_distributed", False):
        logger.warning(
            "[sharded-stream] K3_SHARDED_EXPERT_STREAMING=1 but the runai "
            "loader is in distributed mode, which re-broadcasts every chunk "
            "to all ranks and would defeat (and fight) the per-host filter; "
            "falling back to the full checkpoint read.")
        return False
    if not envs.MXFP4_SHARD_THEN_DECODE:
        logger.warning(
            "[sharded-stream] K3_SHARDED_EXPERT_STREAMING=1 requires "
            "MXFP4_SHARD_THEN_DECODE=1 (the host decode concatenates all "
            "experts, so filtering the read would break it); falling back "
            "to the full checkpoint read.")
        return False

    plan = plan_sharded_expert_streaming(model)
    if plan is None:
        logger.warning(
            "[sharded-stream] K3_SHARDED_EXPERT_STREAMING=1 but the model "
            "has no compressed-tensors MXFP4 MoE layers; falling back to "
            "the full checkpoint read.")
        return False
    needed, num_experts = plan

    model_weights = model_config.model
    if getattr(model_config, "model_weights", None):
        model_weights = model_config.model_weights
    hf_weights_files = loader._prepare_weights(model_weights,
                                               model_config.revision)
    dropped_names: list = []
    model.load_weights(
        sharded_expert_weights_iterator(hf_weights_files,
                                        needed,
                                        num_experts,
                                        loader.load_config.use_tqdm_on_load,
                                        dropped_names_sink=dropped_names))
    validate_sharded_streaming_complete(model, dropped_names=dropped_names)
    return True
