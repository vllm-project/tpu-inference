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
from collections.abc import Generator
from typing import Optional

import regex as re
import torch
from safetensors import safe_open
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.runai_streamer_loader import \
    RunaiModelStreamerLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Weight-file metadata tracking
# ---------------------------------------------------------------------------

# Global mapping from safetensors tensor name → file path.
# Populated during weight iteration so that ``process_weights_after_loading``
# can tell each host which file to read directly from shared storage.
_WEIGHT_TENSOR_TO_FILE: dict[str, str] = {}


def get_weight_file_for_tensor(tensor_name: str) -> Optional[str]:
    """Return the safetensors file path that contains *tensor_name*, or None."""
    return _WEIGHT_TENSOR_TO_FILE.get(tensor_name)


def _build_tensor_to_file_map(
    hf_weights_files: list[str],
    local_model_dir: Optional[str] = None,
) -> dict[str, str]:
    """Build a mapping from tensor name → safetensors file path.

    For local files, reads safetensors file headers directly (lightweight).
    For GCS paths (``gs://``), reads the ``model.safetensors.index.json``
    from *local_model_dir* (where RunAI model streamer caches config/JSON
    files).

    The returned paths are the **original** file paths (e.g. ``gs://`` URIs)
    so that colocated python functions on TPU hosts can convert them to
    GCSFuse paths themselves.

    Args:
        hf_weights_files: List of safetensors file paths (local or ``gs://``).
        local_model_dir: Local directory containing ``model.safetensors.index.json``.
            Required when *hf_weights_files* are remote (``gs://``).
    """
    if not hf_weights_files:
        return {}

    sample = hf_weights_files[0]
    is_gcs = sample.lower().startswith("gs://")

    if not is_gcs:
        # Local path — read headers directly with safetensors.
        tensor_to_file: dict[str, str] = {}
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    tensor_to_file[name] = st_file
        return tensor_to_file

    # GCS path — read the index JSON from the local cache instead.
    import json as _json
    gs_folder = os.path.dirname(sample)
    basename_to_gs_path = {os.path.basename(f): f for f in hf_weights_files}

    # Look for model.safetensors.index.json in local_model_dir.
    index_path = None
    if local_model_dir and os.path.isdir(local_model_dir):
        candidate = os.path.join(local_model_dir,
                                 "model.safetensors.index.json")
        if os.path.isfile(candidate):
            index_path = candidate

    if index_path is None:
        logger.warning(
            "Could not find model.safetensors.index.json in '%s'. "
            "Distributed file loading will be disabled.",
            local_model_dir)
        return {}

    with open(index_path, "r") as fobj:
        index_data = _json.load(fobj)

    weight_map = index_data.get("weight_map", {})
    tensor_to_file: dict[str, str] = {}
    for tname, filename in weight_map.items():
        gs_path = basename_to_gs_path.get(
            filename, os.path.join(gs_folder, filename))
        tensor_to_file[tname] = gs_path

    logger.info(
        "Built tensor→file map from '%s' (%d tensors across %d files)",
        index_path, len(tensor_to_file), len(hf_weights_files))
    return tensor_to_file


def attach_incremental_weight_loader(
    model: torch.nn.Module,
) -> None:
    """
    Traverses the model and overrides the weight_loader of each parameter to support incremental loading.
    This allows processing and sharding of weights after all weights for a module have been loaded.
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            # Track which safetensors file(s) contributed to this layer.
            # The _hf_tensor_name is set by the annotating iterator
            # in RunaiIncrementalModelLoader under Pathways.
            hf_name = getattr(loaded_weight, '_hf_tensor_name', None)
            if hf_name and hasattr(layer, '_weight_file_refs'):
                file_path = get_weight_file_for_tensor(hf_name)
                if file_path:
                    layer._weight_file_refs.append(
                        (hf_name, file_path, param_name, args, kwargs))

            # Loading the weight
            res = original_loader(param, loaded_weight, *args, **kwargs)

            # Processing and sharding
            # For now, only handle unquantized linear and moe layers.
            quant_method = layer.quant_method
            if isinstance(quant_method, VllmQuantizationMethod):
                quant_method.maybe_process_weights(layer, param_name, args,
                                                   kwargs)

            return res

        return weight_loader_wrapper

    for name, module in model.named_modules():
        # Weight loader will be invoked multiple times for module. In order to determine when all the weights are loaded,
        # we need to keep track of the loaded weights for each module.
        module._loaded_weights = set()
        # Track file references for distributed loading.
        module._weight_file_refs = []
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


def _annotating_weights_iterator(
    weights_iter: Generator[tuple[str, torch.Tensor], None, None],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Wrap a weights iterator to annotate each tensor with its HF name.

    Attaches ``_hf_tensor_name`` to each yielded tensor so that the
    incremental weight loader can record which safetensors file each weight
    originally came from.
    """
    for name, tensor in weights_iter:
        tensor._hf_tensor_name = name
        yield name, tensor


@register_model_loader("runai_streamer")
class RunaiIncrementalModelLoader(RunaiModelStreamerLoader):
    """Model loader that supports RunAI streaming and incremental weight sharding.

    When running under Pathways (``VLLM_TPU_USING_PATHWAYS`` is set), this
    loader also enables **distributed file loading**: it builds a mapping
    from safetensors tensor name → file path and annotates each weight
    tensor with its HF name.  During ``process_weights_after_loading``,
    non-merged layers can then be read in parallel by every TPU host
    directly from shared storage (e.g. GCS via GCSFuse), avoiding the
    Pathways head as the I/O bottleneck.

    Usage:
        Set ``--load-format runai_streamer`` (the default for GCS models).
        Under Pathways the distributed path activates automatically.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self._local_model_dir: Optional[str] = None

    def _prepare_weights(self, model_name_or_path: str,
                         revision: str | None) -> list[str]:
        hf_weights_files = super()._prepare_weights(model_name_or_path,
                                                    revision)
        hf_weights_files.sort(key=lambda f: [
            int(s) if s.isdigit() else s
            for s in re.split(r"(\d+)", os.path.basename(f))
        ])
        return hf_weights_files

    def _get_weights_iterator(
        self, model_or_path: str, revision: str
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Return a weights iterator, optionally with distributed-loading metadata.

        Under Pathways, builds a tensor→file map from safetensors metadata
        and annotates each yielded tensor with ``_hf_tensor_name`` so that
        ``process_weights_after_loading`` can dispatch non-merged layers to
        file-based parallel loading.

        Outside Pathways this is equivalent to the parent's iterator.
        """
        import vllm.envs as vllm_envs

        if not vllm_envs.VLLM_TPU_USING_PATHWAYS:
            yield from super()._get_weights_iterator(model_or_path, revision)
            return

        global _WEIGHT_TENSOR_TO_FILE

        hf_weights_files = self._prepare_weights(model_or_path, revision)

        # Build the tensor→file map.  For GCS paths, uses the index.json
        # from the local cache directory (set by load_model).
        _WEIGHT_TENSOR_TO_FILE = _build_tensor_to_file_map(
            hf_weights_files, local_model_dir=self._local_model_dir)
        logger.info(
            "Built tensor→file map with %d tensors across %d files",
            len(_WEIGHT_TENSOR_TO_FILE), len(hf_weights_files))

        # Delegate to the parent iterator (RunAI streamer reads the files)
        # and annotate each tensor with its HF name.
        yield from _annotating_weights_iterator(
            super()._get_weights_iterator(model_or_path, revision))

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

        # Store the local model directory so _get_weights_iterator can use
        # it to find model.safetensors.index.json for GCS models.
        self._local_model_dir = model_config.model

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