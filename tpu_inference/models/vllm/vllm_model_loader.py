# Copyright 2025 Google LLC
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

import os
import re as stdlib_re

import regex as re
import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import \
    compute_local_expert_ids
from vllm.model_executor.model_loader.runai_streamer_loader import \
    RunaiModelStreamerLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from tpu_inference import envs
from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = stdlib_re.compile(r'model\.layers\.(\d+)\.')


def _count_groups(sorted_ids: list[int]) -> int:
    if not sorted_ids:
        return 0
    groups = 1
    for i in range(1, len(sorted_ids)):
        if sorted_ids[i] != sorted_ids[i - 1] + 1:
            groups += 1
    return groups


def _compute_mesh_aware_local_expert_ids(
    num_experts: int,
    ep_size: int,
) -> list[int] | None:
    """Return the GLOBAL expert ids this process's local devices will own,
    ORDERED by mesh flat-iteration of local devices.

    Contract: this order must match how ``make_array_from_process_local_data``
    slices ``local_data`` onto this process's devices. JAX iterates devices in
    row-major order of the mesh and filters by process_index; the i-th local
    device gets ``local_data[i*experts_per_slot:(i+1)*experts_per_slot]``.

    Runtime kernel (``fused_moe_gmm.py``) assumes mesh EP flat slot ``k`` owns
    global experts ``[k*experts_per_slot : (k+1)*experts_per_slot]``, so each
    chunk in the returned list must equal ``[slot*eps, slot*eps+eps)`` where
    ``slot`` is the local device's EP flat position.

    Returns ``None`` if the mesh cannot be reconstructed; caller should
    fall back to the linear assignment (which is only correct when each
    process's devices occupy a contiguous EP flat slot range).
    """
    try:
        import jax
        import numpy as np
        from jax.experimental import mesh_utils
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        sharding = getattr(vllm_config, "sharding_config", None)
        if sharding is None:
            return None

        mesh_shape = (
            sharding.model_dp_size,
            sharding.attn_dp_size,
            sharding.attn_dp_expert_size,
            sharding.expert_size,
            sharding.tp_size,
        )

        devices = jax.devices()
        try:
            device_mesh = mesh_utils.create_device_mesh(
                mesh_shape, devices, allow_split_physical_axes=True)
        except (AssertionError, ValueError, RuntimeError):
            device_mesh = np.array(devices).reshape(mesh_shape)

        local_proc = jax.process_index()
        # EXPERT sharding axes = ('attn_dp', 'attn_dp_expert', 'expert', 'model')
        # = mesh positions 1,2,3,4. EP flat position within those axes is
        # row-major flat index over (attn_dp, attn_dp_expert, expert, model).
        ep_total = mesh_shape[1] * mesh_shape[2] * mesh_shape[3] * mesh_shape[4]
        if ep_total <= 0 or num_experts % ep_total != 0:
            return None
        experts_per_slot = num_experts // ep_total

        # Row-major flat iteration over entire mesh. Within a model_dp group,
        # flat index inside that group == EP flat slot.
        flat_mesh = device_mesh.reshape(
            mesh_shape[0],
            ep_total,
        )
        ordered: list[int] = []
        for d_idx in range(flat_mesh.shape[0]):
            for slot in range(flat_mesh.shape[1]):
                dev = flat_mesh[d_idx, slot]
                if dev.process_index != local_proc:
                    continue
                start = slot * experts_per_slot
                ordered.extend(range(start, start + experts_per_slot))
        return ordered if ordered else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Mesh-aware EP filter init failed, falling back to linear: %s",
            exc)
        return None


def attach_incremental_weight_loader(model: torch.nn.Module) -> None:
    """
    Traverses the model and overrides the weight_loader of each parameter to support incremental loading.
    This allows processing and sharding of weights after all weights for a module have been loaded.
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            # Loading the weight
            res = original_loader(param, loaded_weight, *args, **kwargs)

            # Processing and sharding
            # For now, only handle unquantized linear and moe layers.
            quant_method = getattr(layer, "quant_method", None)
            if isinstance(quant_method, VllmQuantizationMethod):
                quant_method.maybe_process_weights(layer, param_name, args,
                                                   kwargs)

            return res

        return weight_loader_wrapper

    for name, module in model.named_modules():
        # Weight loader will be invoked multiple times for module. In order to determine when all the weights are loaded,
        # we need to keep track of the loaded weights for each module.
        module._loaded_weights = set()
        for param_name, param in module.named_parameters(recurse=False):
            # Omit parameters that do not have a weight_loader
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            setattr(
                param, "weight_loader",
                create_weight_loader(module, original_loader, name,
                                     param_name))


@register_model_loader("tpu_streaming_loader")
class IncrementalModelLoader(DefaultModelLoader):
    """
    Model loader that supports incremental weight loading and sharding.

    This loader is needed to inject the `attach_incremental_weight_loader` logic
    before the actual weight loading begins. This allows us to wrap the
    parameter weight loaders so that weights are sharded to TPU and freed from
    CPU memory as soon as a layer is fully loaded, rather than waiting for the
    entire model to be loaded into CPU memory first.
    """

    def __init__(self, load_config: LoadConfig):
        load_config.load_format = "auto"
        super().__init__(load_config)
        self.local_expert_ids = None
        self._pp_layer_range = None  # (start, end) for PP filtering
        self._tp_plan = None  # Set by _init_tp_weight_filter
        self._tp_size = 1
        self._tp_rank = 0

    def _init_tpu_ep_weight_filter(self, model_config: ModelConfig) -> None:
        """Initialize EP weight filter using the JAX process index as EP rank.

        For large FP8 MoE models (e.g., GLM-5.1-FP8 744B) that cannot fit
        entirely in CPU RAM, this filter ensures each TPU host only loads the
        expert weights assigned to it by tpu-inference's expert-parallelism
        sharding.  Without this, every host would attempt to load all ~756 GB
        of expert weights, causing an OOM on the head node.

        The filter is keyed on ``sharding_config.expert_size`` (set via
        ``--additional-config '{"sharding": {...}}'``) and JAX's
        ``process_index()`` which equals the host rank on multi-host setups.
        """
        try:
            import jax
            ep_size = int(os.environ.get("TPU_EP_SIZE", "0"))
            if ep_size <= 1:
                from vllm.config import get_current_vllm_config

                vllm_config = get_current_vllm_config()
                sharding_config = getattr(vllm_config, "sharding_config",
                                          None)
                if sharding_config is None:
                    return

                # When enable_dp_attention=true on MLA models (e.g. GLM-5.1),
                # expert parallelism is carried on the `attn_dp_expert` axis
                # instead of the `expert` axis, so expert_size alone reads as
                # 1 and the filter would bail out — causing every host to
                # load the full 744 GB of FP8 MoE weights and OOM.
                # Use the full EP axes product so both configs route through
                # the mesh-aware per-process filter.
                ep_size = (sharding_config.expert_size *
                           sharding_config.attn_dp_expert_size)
            if ep_size <= 1:
                return

            num_experts = model_config.get_num_experts()
            if num_experts <= 0:
                hf_config = getattr(model_config, "hf_config", None)
                if hf_config is not None:
                    num_experts = int(
                        getattr(hf_config, "n_routed_experts", 0) or
                        getattr(hf_config, "num_experts", 0) or 0)
            if num_experts <= 0:
                return

            # Single-host EP: one process handles ALL ep_size shards via a
            # local mesh, so it must load ALL experts (not just its rank's
            # slice). The EP sharding happens later via NamedSharding(mesh,
            # P(EXPERT)) which distributes the full weight across local devices.
            # Skip the filter when there's only one JAX process in the cluster.
            if jax.process_count() <= 1:
                logger.info(
                    "TPU EP weight filter: single-host mode (process_count=1), "
                    "loading ALL %d experts (EP sharding applied later via mesh)",
                    num_experts,
                )
                return

            # Try mesh-aware ordered assignment first. Required when physical
            # mesh scatters a process's devices across non-contiguous EP flat
            # slots (e.g. v4-64 TP=4 EP=8: proc N → slots [2N,2N+1,2N+16,2N+17]).
            # The returned list is ordered by mesh flat-iteration so
            # make_array_from_process_local_data places each 8-expert chunk on
            # the right local device, matching the kernel's shard_idx*8 rule.
            ordered = _compute_mesh_aware_local_expert_ids(num_experts, ep_size)
            if ordered is not None:
                # vLLM's DefaultModelLoader uses a set for should_skip_weight
                # membership; keep the ordered list separately for the
                # module annotation where order matters.
                self.local_expert_ids = set(ordered)
                self._ordered_local_expert_ids = ordered
                sample_chunks = [ordered[i:i + 8] for i in range(0, min(32, len(ordered)), 8)]
                logger.info(
                    "[EP_DIAG loader proc=%d MESH_AWARE] ep_size=%d "
                    "loading %d/%d experts (ordered); chunks_sample=%s",
                    jax.process_index(), ep_size,
                    len(ordered), num_experts, sample_chunks,
                )
            else:
                ep_rank = int(os.environ.get("TPU_EP_RANK",
                                             str(jax.process_index())))
                self.local_expert_ids = compute_local_expert_ids(
                    num_experts, ep_size, ep_rank)
                self._ordered_local_expert_ids = None
                _ids_sorted = sorted(self.local_expert_ids)
                logger.info(
                    "[EP_DIAG loader proc=%d LINEAR_FALLBACK] ep_size=%d "
                    "ep_rank=%d loading %d/%d experts; first8=%s last4=%s",
                    jax.process_index(), ep_size, ep_rank,
                    len(self.local_expert_ids), num_experts,
                    _ids_sorted[:8], _ids_sorted[-4:],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize TPU EP weight filter: %s", exc)

    def _init_pp_layer_filter(self, vllm_config: VllmConfig) -> None:
        """Set up PP layer range filter to skip weights for other stages."""
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        truncate = int(os.environ.get("TPU_TRUNCATE_LAYERS", "0"))
        if pp_size <= 1 and truncate <= 0:
            return
        try:
            if truncate > 0 and pp_size <= 1:
                pp_rank = 0
                num_layers = truncate
                start, end = 0, truncate
                self._pp_is_first = True
                self._pp_is_last = True
            else:
                from vllm.distributed.parallel_state import get_pp_group
                pp_rank = get_pp_group().rank_in_group
                num_layers = vllm_config.model_config.get_total_num_hidden_layers()
                start, end = get_pp_indices(num_layers, pp_rank, pp_size)
                self._pp_is_first = get_pp_group().is_first_rank
                self._pp_is_last = get_pp_group().is_last_rank
            self._pp_layer_range = (start, end)
            logger.info(
                "PP layer filter: pp_rank=%d, layers %d-%d (%d/%d), "
                "first=%s, last=%s",
                pp_rank, start, end - 1, end - start, num_layers,
                self._pp_is_first, self._pp_is_last,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize PP layer filter: %s", exc)

    def _should_skip_for_pp(self, name: str) -> bool:
        """Return True if this weight name belongs to a different PP stage."""
        if self._pp_layer_range is None:
            return False
        m = _LAYER_RE.match(name)
        if m:
            layer_idx = int(m.group(1))
            return layer_idx < self._pp_layer_range[0] or layer_idx >= self._pp_layer_range[1]
        # Non-layer weights: embed_tokens on first rank, lm_head/norm on last
        if 'embed_tokens' in name:
            return not self._pp_is_first
        if 'lm_head' in name or name.endswith('.norm.weight'):
            return not self._pp_is_last
        return False

    def _init_tp_weight_filter(self, model: torch.nn.Module,
                               vllm_config: VllmConfig) -> None:
        """Build per-host TP slice plan.

        Only activates for **pure TP multi-host** runs (PP=1, EP=1,
        process_count > 1) where each host would otherwise load the full
        multi-TB checkpoint into CPU RAM and OOM. For PP or EP modes, the
        existing PP-layer and EP-expert filters already own the per-host
        slicing, and this filter must stay a no-op.

        Gated by env ``TPU_TP_SELECTIVE_LOAD=1``.
        """
        from tpu_inference.models.vllm import tp_selective_loader as tpsl
        if not tpsl.is_enabled():
            logger.info("[TP-selective] DIAG: disabled (env TPU_TP_SELECTIVE_LOAD != '1')")
            return
        logger.info("[TP-selective] DIAG: entering _init_tp_weight_filter")
        try:
            import jax
            sharding_config = getattr(vllm_config, "sharding_config", None)
            if sharding_config is None:
                # vllm_config.sharding_config attr is set on APIServer via
                # tpu_platform._initialize_sharding_config but that ad-hoc attr
                # does not survive pickle across Ray actor boundary.  Rebuild
                # from the underlying additional_config dict (which does survive).
                from tpu_inference.layers.common.sharding import \
                    ShardingConfigManager
                try:
                    sharding_config = ShardingConfigManager.from_vllm_config(vllm_config)
                    logger.info("[TP-selective] DIAG: rebuilt sharding_config from additional_config")
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "[TP-selective] rebuild sharding_config failed: %s", exc)
                    return
            pp_size = int(vllm_config.parallel_config.pipeline_parallel_size)
            # EP axes: expert and attn_dp_expert both carry expert parallelism
            # on MLA models; either being >1 means EP filter owns MoE shards.
            ep_total = (int(sharding_config.expert_size) *
                        int(sharding_config.attn_dp_expert_size))
            process_count = int(jax.process_count())
            tp_global = int(sharding_config.tp_size)
            logger.info(
                "[TP-selective] DIAG: pp=%d ep_total=%d proc_cnt=%d tp_global=%d",
                pp_size, ep_total, process_count, tp_global)
            if pp_size != 1 or ep_total != 1 or process_count <= 1:
                logger.info(
                    "[TP-selective] no-op (pp=%d ep_total=%d proc_cnt=%d "
                    "tp_global=%d) — PP/EP filters already slice per-host",
                    pp_size, ep_total, process_count, tp_global)
                return
            if tp_global <= process_count:
                logger.info(
                    "[TP-selective] no-op: tp_global=%d <= proc_cnt=%d (single-host TP)",
                    tp_global, process_count)
                return
            # Pure-TP multi-host. Coarse TP axis = process_count (cross-host).
            # Each host then JAX-shards its slice across its local chips
            # (tp_global / process_count) via the runtime mesh.
            coarse_tp = process_count
            coarse_rank = int(jax.process_index())
            tpsl._apply_vllm_patches()
            plan = tpsl.build_tp_plan(model, coarse_tp, coarse_rank)
            if not plan:
                logger.info(
                    "[TP-selective] empty plan (no TP-aware params found); "
                    "falling back to full-tensor load")
                return
            self._tp_plan = plan
            self._tp_size = coarse_tp
            self._tp_rank = coarse_rank
            logger.info(
                "[TP-selective] plan built: %d disk-name entries, "
                "coarse_tp=%d rank=%d tp_global=%d",
                len(plan), coarse_tp, coarse_rank, tp_global)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TP-selective init failed: %s", exc)
            self._tp_plan = None

    def _get_weights_iterator(self, source):
        """Override to add TP-slice + PP layer filtering before tensor read."""
        # TP-selective path: bypass DefaultLoader's iterator entirely and
        # stream slices directly from safetensors. Honors PP + EP filters.
        if self._tp_plan:
            from tpu_inference.models.vllm import tp_selective_loader as tpsl
            _hf_folder, hf_weights_files, use_safetensors = (
                self._prepare_weights(
                    source.model_or_path, source.subfolder, source.revision,
                    source.fall_back_to_pt, source.allow_patterns_overrides))
            if not use_safetensors:
                # Fall back to base iterator for non-safetensors checkpoints.
                logger.warning(
                    "[TP-selective] non-safetensors source; falling back")
                return super()._get_weights_iterator(source)
            pp_skip = (self._should_skip_for_pp
                       if self._pp_layer_range is not None else None)
            raw_iter = tpsl.tp_sliced_iterator(
                hf_weights_files=hf_weights_files,
                use_tqdm_on_load=self.load_config.use_tqdm_on_load,
                plan=self._tp_plan,
                tp_size=self._tp_size,
                tp_rank=self._tp_rank,
                local_expert_ids=self.local_expert_ids,
                pp_skip_fn=pp_skip,
                safetensors_load_strategy=(
                    self.load_config.safetensors_load_strategy),
            )
            prefix = source.prefix
            return ((prefix + n, t) for (n, t) in raw_iter)

        if self._pp_layer_range is None:
            return super()._get_weights_iterator(source)

        # Wrap the parent iterator. We can't filter before get_tensor()
        # without reimplementing the full iterator, but we CAN free skipped
        # tensors immediately and avoid the heavy weight_loader path.
        base_iter = super()._get_weights_iterator(source)

        def _filtered():
            skipped = loaded = 0
            for name, tensor in base_iter:
                if self._should_skip_for_pp(name):
                    skipped += 1
                    del tensor
                    continue
                loaded += 1
                yield name, tensor
            if skipped:
                logger.info(
                    "PP filter: loaded %d, skipped %d weight tensors",
                    loaded, skipped)

        return _filtered()

    def load_model(self,
                   vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   prefix: str = "") -> torch.nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Override weight loader logic of each parameter to support incremental loading.
            attach_incremental_weight_loader(model)
            self.local_expert_ids = None
            # Set up EP weight filter so each host only loads its expert shard.
            # This is critical for large FP8 MoE models to avoid CPU RAM OOM.
            self._init_tpu_ep_weight_filter(model_config)
            # Set up PP layer filter so each stage skips weights for
            # layers on other stages (avoids reading ~7/8 of the model).
            self._init_pp_layer_filter(vllm_config)
            # Set up TP slice filter so each host reads only its per-rank
            # share of each TP-sharded tensor. Gated by TPU_TP_SELECTIVE_LOAD=1.
            # Required for pure-TP multi-host runs of large FP8 checkpoints
            # (without PP/EP filtering) to avoid CPU RAM OOM.
            self._init_tp_weight_filter(model, vllm_config)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            # Annotate FusedMoE layers with the number of locally-loaded experts.
            # process_weights_after_loading uses this to slice the full
            # (num_experts_global, ...) tensor down to (num_local_experts, ...)
            # before sharding, avoiding HBM OOM on large MoE models.
            if self.local_expert_ids:
                from vllm.model_executor.layers.fused_moe import FusedMoE
                # Mesh-aware path preserves iteration order required by
                # make_array_from_process_local_data. Linear fallback has
                # contiguous shards, so any deterministic order works.
                ordered = getattr(self, "_ordered_local_expert_ids", None)
                if ordered is not None:
                    local_expert_ids = tuple(ordered)
                else:
                    local_expert_ids = tuple(sorted(self.local_expert_ids))
                for _, module in model.named_modules():
                    if isinstance(module, FusedMoE):
                        module._tpu_ep_local_count = len(local_expert_ids)
                        module._tpu_ep_local_expert_ids = local_expert_ids
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


@register_model_loader("runai_streamer")
class RunaiIncrementalModelLoader(RunaiModelStreamerLoader):
    """Model loader that supports both RunAI streaming and incremental weight sharding."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def _prepare_weights(self, model_name_or_path: str,
                         revision: str | None) -> list[str]:
        hf_weights_files = super()._prepare_weights(model_name_or_path,
                                                    revision)
        hf_weights_files.sort(key=lambda f: [
            int(s) if s.isdigit() else s
            for s in re.split(r"(\d+)", os.path.basename(f))
        ])
        return hf_weights_files

    def load_model(self,
                   vllm_config: VllmConfig,
                   model_config: ModelConfig,
                   prefix: str = "") -> torch.nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)
            # Override weight loader logic of each parameter to support incremental loading.
            attach_incremental_weight_loader(model)
            # Quantization does not happen in `load_weights` but after it
            self.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
