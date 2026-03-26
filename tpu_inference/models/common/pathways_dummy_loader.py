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
"""Pathways dummy model loader.

Generates small random weights **directly on the TPU mesh** without any
CPU allocation, colocated-CPU transfer, or file I/O.  This is useful for
Pathways environments where the main process has no local CPU device and
sending full unsharded tensors to a single device would OOM.

Works for both the ``flax_nnx`` (JAX) and ``vllm`` (torchax) backends.
"""

import time

import jax
import jax.numpy as jnp
import torch.nn
from jax.sharding import Mesh, NamedSharding
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (initialize_model,
                                                     process_weights_after_loading)
from vllm.utils.torch_utils import set_default_torch_dtype

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# ---- helpers ----------------------------------------------------------------

_LOW = -1e-3
_HIGH = 1e-3
_SEED = 1234


def create_dummy_weights_on_tpu(
    tpu_sharding: NamedSharding,
    weight_shape: tuple[int, ...],
    weight_dtype: jnp.dtype,
) -> jax.Array:
    """Create small random dummy weights directly on the TPU mesh.

    The values are drawn uniformly from ``[_LOW, _HIGH]`` so that they do
    not cause NaN / overflow during a forward pass while still being
    non-trivial for performance profiling.
    """
    t0 = time.perf_counter()
    key = jax.random.PRNGKey(_SEED)

    @jax.jit(out_shardings=tpu_sharding)
    def _generate(key):
        return jax.random.uniform(
            key,
            shape=weight_shape,
            dtype=weight_dtype,
            minval=_LOW,
            maxval=_HIGH,
        )

    tpu_array = _generate(key)
    t1 = time.perf_counter()
    logger.info(
        "create_dummy_weights_on_tpu shape=%s dtype=%s: %.3fs",
        weight_shape,
        weight_dtype,
        t1 - t0,
    )
    return tpu_array


# ---- JAX / flax_nnx backend ------------------------------------------------


def load_dummy_weights_jax(model, mesh: Mesh) -> None:
    """Fill every ``nnx.Param`` in *model* with random TPU-resident data.

    This is the ``flax_nnx`` counterpart: it iterates over the model's
    named parameters, reads only shape / dtype / sharding metadata, and
    calls :func:`create_dummy_weights_on_tpu` to materialise the weight
    directly on TPU.
    """
    from jax.sharding import PartitionSpec, SingleDeviceSharding

    from tpu_inference.models.jax.utils.weight_utils import (
        assign_and_shard_param)

    t0 = time.perf_counter()

    for param_name, param in model.named_parameters():
        spec = param.get_metadata().get("sharding", ())
        if isinstance(spec, NamedSharding):
            spec = spec.spec
        elif isinstance(spec, SingleDeviceSharding):
            spec = ()

        param_mesh = param.get_metadata().get("mesh", None) or mesh
        sharding = NamedSharding(param_mesh, spec if isinstance(spec, PartitionSpec) else PartitionSpec())

        is_moe = hasattr(param, "_weights_to_load")
        param_shape = param.value.shape

        if is_moe:
            # MoE: downstream post-loading fusion expects transposed shape
            # (E, F, D) instead of (E, D, F).
            num_experts, input_dim, intermediate_dim = param_shape
            param_shape = (num_experts, intermediate_dim, input_dim)

        dummy = create_dummy_weights_on_tpu(
            tpu_sharding=sharding,
            weight_shape=param_shape,
            weight_dtype=param.value.dtype,
        )

        if is_moe:
            param._weights_to_load[:] = jnp.vsplit(
                dummy, indices_or_sections=num_experts)

        assign_and_shard_param(param, dummy, param_name)

    # Post-process (quantisation etc.) per-module.
    _process_weights_after_loading_jax(model)

    logger.info(
        "Pathways dummy weight loading (jax) took %.2fs",
        time.perf_counter() - t0,
    )


def _process_weights_after_loading_jax(module) -> None:
    """Recursively call ``process_weights_after_loading`` if available."""
    from tpu_inference.layers.jax import JaxModuleList
    from tpu_inference.layers.jax.quantization import QuantizeMethodBase

    if (quant_method := getattr(module, "quant_method", None)) is not None:
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(module)
            return
    if isinstance(module, JaxModuleList):
        for sub_module in module:
            _process_weights_after_loading_jax(sub_module)
    else:
        for _name, sub_module in module.named_children():
            _process_weights_after_loading_jax(sub_module)


# ---- registered model loader -----------------------------------------------


@register_model_loader("pathways_dummy")
class PathwaysDummyModelLoader(BaseModelLoader):
    """Model loader that creates dummy weights directly on the TPU mesh.

    Intended for use under Pathways where the main process has no local
    CPU device.  Works for both ``flax_nnx`` (JAX) and ``vllm`` (torchax)
    backends.

    **JAX (flax_nnx) backend:**
        Used as a model loader via ``get_model_loader``.  Call
        ``load_weights(model, model_config)`` after ``nnx.eval_shape``
        to fill abstract parameters with TPU-resident dummy data.

    **vLLM (torchax) backend:**
        Used via ``vllm_get_model`` → ``load_model()``.  The model
        structure is initialised on CPU with ``initialize_model``, but
        **no** CPU weight data is loaded (``load_weights`` is a no-op).
        The subsequent ``process_weights_after_loading`` triggers the
        per-layer quant method's ``process_weights_after_loading`` which
        calls ``_load_weight_for_layer`` — under Pathways + dummy mode
        this creates random weights directly on TPU via
        :func:`create_dummy_weights_on_tpu`.  Remaining unsharded tensors
        (embeddings, buffers) are handled by ``shard_model_to_tpu``.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download.

    def load_weights(self, model, model_config: ModelConfig) -> None:
        """Load dummy weights into *model*.

        Dispatches to the jax helper for flax_nnx models.  For vllm
        (torchax) models this is a no-op — weight creation is handled
        by ``process_weights_after_loading`` via the quant methods.
        """
        from tpu_inference.layers.jax import JaxModule

        if isinstance(model, JaxModule):
            # Jax (flax_nnx) backend — need mesh from model params.
            mesh = jax.sharding.get_mesh()
            if mesh is None:
                raise RuntimeError(
                    "PathwaysDummyModelLoader requires an active JAX mesh "
                    "context for flax_nnx models.  Wrap the call in "
                    "`jax.set_mesh(mesh)` or `with mesh:`.")
            load_dummy_weights_jax(model, mesh)
        else:
            # vllm/torchax: no-op.  Weights are created during
            # process_weights_after_loading by the quant methods.
            pass

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> torch.nn.Module:
        """Initialise a vLLM (torchax) model without loading real weights.

        The model structure is created via ``initialize_model`` and then
        ``process_weights_after_loading`` is called so that quant methods
        can create dummy weights directly on TPU.
        """
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (device_config.device
                       if load_config.device is None else load_config.device)
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config,
                                         prefix=prefix)
            # process_weights_after_loading triggers the quant methods'
            # process_weights_after_loading which, under Pathways + dummy,
            # creates random weights directly on TPU.
            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
