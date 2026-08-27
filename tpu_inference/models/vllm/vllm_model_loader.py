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
import time

import regex as re
import torch
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

# Set TPU_WEIGHT_LOAD_PROFILE=1 to log where weight-load wall clock goes.
#
# For a large MoE checkpoint the tqdm bar is the only signal today, and it
# cannot say whether time is spent waiting on the streamer, copying tensors
# into the module params, or in the quantization method's per-layer sharding
# barrier. Those have completely different fixes. Every one of the three is
# observable from this wrapper alone: the wrapper brackets the load and the
# process calls, and whatever elapses between one wrapper returning and the
# next being entered is the consumer waiting on the streamer.
#
# Off by default; when off the only cost is one module-level bool test.
_PROFILE = os.environ.get("TPU_WEIGHT_LOAD_PROFILE", "0") == "1"
_PROFILE_EVERY = int(os.environ.get("TPU_WEIGHT_LOAD_PROFILE_EVERY", "10000"))


class _LoadProfile:
    """Running split of weight-load wall clock into stream / load / process."""

    def __init__(self):
        self.n = 0
        # Inside next() on the streamer generator: blocked on the streamer.
        self.iter_s = 0.0
        # Between one weight_loader returning and the next being entered, minus
        # iter_s: vLLM's per-tensor dispatch.
        self.gap_s = 0.0
        self.load_s = 0.0
        self.process_s = 0.0
        self.t_start = None
        self.t_last_exit = None
        # Slowest single process call, which is the per-layer barrier.
        self.max_process_s = 0.0
        self.max_process_at = ""

    def start(self) -> None:
        if self.t_start is None:
            self.t_start = self.t_last_exit = time.perf_counter()

    def report(self, tag: str) -> None:
        now = time.perf_counter()
        total = now - self.t_start if self.t_start else 0.0
        dispatch = self.gap_s - self.iter_s
        other = total - self.gap_s - self.load_s - self.process_s

        def pct(x):
            return 100 * x / total if total else 0.0

        print(f"[weight-load-profile] {tag} n={self.n} total={total:.0f}s"
              f" | stream {self.iter_s:.0f}s ({pct(self.iter_s):.0f}%)"
              f" | dispatch {dispatch:.0f}s ({pct(dispatch):.0f}%)"
              f" | load {self.load_s:.0f}s ({pct(self.load_s):.0f}%)"
              f" | process {self.process_s:.0f}s ({pct(self.process_s):.0f}%)"
              f" | other {other:.0f}s ({pct(other):.0f}%)"
              f" | slowest process {self.max_process_s:.1f}s"
              f" @ {self.max_process_at}",
              flush=True)


_profile = _LoadProfile()


def attach_incremental_weight_loader(model: torch.nn.Module) -> None:
    """
    Traverses the model and overrides the weight_loader of each parameter to support incremental loading.
    This allows processing and sharding of weights after all weights for a module have been loaded.
    """

    def create_weight_loader(layer, original_loader, layer_name, param_name):

        def weight_loader_wrapper(param: torch.nn.Parameter,
                                  loaded_weight: torch.Tensor, *args,
                                  **kwargs):
            if not _PROFILE:
                # Loading the weight
                res = original_loader(param, loaded_weight, *args, **kwargs)

                # Processing and sharding
                # Incremental processing and sharding for supported layers.
                # Currently only unquantized and fp8 linear and moe layers
                # supported.
                quant_method = getattr(layer, "quant_method", None)
                if isinstance(quant_method, VllmQuantizationMethod):
                    quant_method.maybe_process_weights(layer, param_name,
                                                       args, kwargs)

                return res

            p = _profile
            p.start()  # no-op unless the iterator was not the timed one
            t0 = time.perf_counter()
            # Everything between the previous call returning and this one:
            # pulling the next tensor off the streamer plus vLLM's dispatch.
            # report() splits it using iter_s.
            p.gap_s += t0 - p.t_last_exit

            res = original_loader(param, loaded_weight, *args, **kwargs)
            t1 = time.perf_counter()
            p.load_s += t1 - t0

            quant_method = getattr(layer, "quant_method", None)
            if isinstance(quant_method, VllmQuantizationMethod):
                quant_method.maybe_process_weights(layer, param_name, args,
                                                   kwargs)
            t2 = time.perf_counter()
            p.process_s += t2 - t1
            if t2 - t1 > p.max_process_s:
                p.max_process_s = t2 - t1
                p.max_process_at = layer_name

            p.n += 1
            if p.n % _PROFILE_EVERY == 0:
                p.report("progress")
            p.t_last_exit = time.perf_counter()
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
            if _PROFILE:
                _profile.report("final")
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()


@register_model_loader("runai_streamer")
class RunaiIncrementalModelLoader(RunaiModelStreamerLoader):
    """Model loader that supports both RunAI streaming and incremental weight sharding."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def _get_weights_iterator(self, *args, **kwargs):
        it = super()._get_weights_iterator(*args, **kwargs)
        if not _PROFILE:
            return it

        # Without this, time spent in vLLM's own per-tensor dispatch (name
        # matching against the expert mapping, which is 6 * num_experts entries
        # wide for this model) is indistinguishable from time blocked on the
        # streamer, since both fall between two weight_loader calls. Timing the
        # generator separates them: what elapses inside next() is the streamer,
        # and the remainder of the gap is dispatch.
        def timed():
            p = _profile
            # Start the clock here rather than at the first weight_loader call,
            # so the first tensor's dispatch is inside the accounting too.
            p.start()
            while True:
                t0 = time.perf_counter()
                try:
                    item = next(it)
                except StopIteration:
                    return
                p.iter_s += time.perf_counter() - t0
                yield item

        return timed()

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
            if _PROFILE:
                _profile.report("final")
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
