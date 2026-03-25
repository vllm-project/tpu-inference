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

from typing import Any, Callable, Optional

import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
import vllm.envs as vllm_envs
from jax.experimental import colocated_python
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEConfig,
                                                  UnquantizedFusedMoEMethod)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod, VocabParallelEmbedding)

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_moe_weights, shard_moe_weights)
from tpu_inference.layers.common.quant_methods import UNQUANTIZED
from tpu_inference.layers.common.quantization import \
    unquantized as common_unquantized
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.vllm.interface.moe import (
    select_moe_backend_from_fused_moe_config, vllm_moe_apply)
from tpu_inference.layers.vllm.process_weights.cleanup_sharding import \
    _tensor_is_in_cpu
from tpu_inference.layers.vllm.quantization.base import VllmQuantizationMethod
from tpu_inference.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product, to_jax_dtype

P = PartitionSpec

logger = init_logger(__name__)


def _use_dummy_weights() -> bool:
    """Return True when dummy (random) weights should be used under Pathways.

    Controlled by the ``PATHWAYS_DUMMY_WEIGHTS`` environment variable.
    When set to a truthy value (``1``, ``true``, ``yes``), every weight is
    generated as small random values (uniform in [-1e-3, 1e-3]) directly on
    the colocated CPU workers, bypassing all file I/O and head-based data
    transfer.

    This flag is independent of ``VLLM_TPU_USING_PATHWAYS`` — callers should
    check both.
    """
    return os.getenv("PATHWAYS_DUMMY_WEIGHTS", "0").lower() in ("1", "true", "yes")


def _colocated_cpu_mesh(tpu_mesh: Mesh) -> Mesh:
    """Return a CPU mesh with colocated CPU devices matching the TPU mesh."""
    return colocated_python.colocated_cpu_devices(tpu_mesh)


def _make_load_weight_on_cpu(
    weight_np: np.ndarray,
    weight_dtype: jnp.dtype,
    weight_global_shape: tuple[int, ...],
    weight_pspec: PartitionSpec,
):
    """Create a colocated python function that loads a weight tensor onto CPU devices.

    Non-array arguments (numpy data, dtype, shape, pspec) are captured in the
    closure and pickled to the remote workers automatically.  The returned
    function accepts and returns only ``jax.Array``, as required by
    ``colocated_python``.

    Args:
        weight_np: The weight as a numpy array (float32).
        weight_dtype: The target ``jax.numpy.dtype``.
        weight_global_shape: The global shape tuple.
        weight_pspec: The ``PartitionSpec`` describing how the weight is
            sharded across the mesh.
    """

    @colocated_python.colocated_python
    def _load_weight_on_cpu(dummy_cpu_array: jax.Array) -> jax.Array:
        # The dummy carries a replicated sharding; reconstruct the real
        # weight sharding from the dummy's mesh and the captured pspec.
        mesh = dummy_cpu_array.sharding.mesh
        sharding = NamedSharding(mesh, weight_pspec)
        devices = list(sharding.addressable_devices)

        local_array = jnp.asarray(weight_np)

        # Use sharding to compute the expected per-device shape.  This
        # correctly handles both partitioned (P("x")) and replicated
        # (P(None)) axes — np.split by device count only works when every
        # axis is partitioned across all devices.
        #
        # Cast to the target dtype *per-shard* before assembling the global
        # array.  Calling .astype() on the assembled global array would
        # trigger a JAX operation referencing all devices in the mesh,
        # including those on other hosts — which are unknown inside
        # colocated_python and cause "Unknown PjRt global device ID" errors.
        device_arrays = []
        for device in devices:
            indices = sharding.addressable_devices_indices_map(weight_global_shape)[device]
            shard = jnp.asarray(local_array[indices], dtype=weight_dtype)
            device_arrays.append(jax.device_put(shard, device))

        return jax.make_array_from_single_device_arrays(
            shape=weight_global_shape,
            sharding=sharding,
            arrays=device_arrays,
        )

    return _load_weight_on_cpu


def _make_load_weight_from_file_on_cpu(
    file_path: str,
    tensor_name: str,
    weight_dtype: jnp.dtype,
    weight_pspec: PartitionSpec,
    weight_shape: tuple[int, ...],
    use_dummy: bool = False,
):
    """Create a colocated python function that loads a weight from file or generates dummy random values.

    When *use_dummy* is True, generates small random shards (uniform in
    [-1e-3, 1e-3]) directly on the colocated CPU workers — no file I/O, no
    data sent from the head.

    When *use_dummy* is False, each colocated CPU worker reads the safetensors
    file from shared storage (e.g. GCSFuse) in parallel.

    Args:
        file_path: Path to the safetensors file (unused when *use_dummy*).
        tensor_name: Key of the tensor inside the safetensors file.
        weight_dtype: The target ``jax.numpy.dtype``.
        weight_pspec: The ``PartitionSpec`` describing how the weight is
            sharded across the mesh.
        weight_shape: The global shape of the weight tensor.
        use_dummy: If True, generate zeros instead of reading the file.
    """

    @colocated_python.colocated_python
    def _load_weight_from_file_on_cpu(dummy_cpu_array: jax.Array) -> jax.Array:
        import time

        t_start = time.perf_counter()

        sharding = NamedSharding(dummy_cpu_array.sharding.mesh, weight_pspec)
        devices = list(sharding.addressable_devices)

        t_setup = time.perf_counter()

        if use_dummy:
            # ---- Dummy path: generate small random weights ----
            # Mirrors vllm's initialize_single_dummy_weight: uniform in
            # [-1e-3, 1e-3] with a fixed seed so results are reproducible.
            low, high = -1e-3, 1e-3
            seed = 1234
            device_arrays = []
            for device in devices:
                indices = sharding.addressable_devices_indices_map(weight_shape)[device]
                shard_shape = tuple(
                    weight_shape[i] if (isinstance(idx, slice) and idx.start is None and idx.stop is None)
                    else (idx.stop - idx.start if isinstance(idx, slice) else 1)
                    for i, idx in enumerate(indices)
                )
                key = jax.random.PRNGKey(seed)
                shard = jax.random.uniform(
                    key, shape=shard_shape, dtype=weight_dtype,
                    minval=low, maxval=high,
                )
                device_arrays.append(jax.device_put(shard, device))

            t_read = t_setup  # no read step
        else:
            # ---- Real path: read from safetensors file ----
            # Ensure safetensors is installed on the colocated workers
            # (it may not be present in the base container image).
            import subprocess, sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "safetensors"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            from safetensors import safe_open
            with safe_open(file_path, framework="numpy") as f:
                weight_np = f.get_tensor(tensor_name)

            t_read = time.perf_counter()

            local_array = jnp.asarray(weight_np)
            device_arrays = []
            for device in devices:
                indices = sharding.addressable_devices_indices_map(weight_shape)[device]
                shard = jnp.asarray(local_array[indices], dtype=weight_dtype)
                device_arrays.append(jax.device_put(shard, device))

        t_shard = time.perf_counter()

        result = jax.make_array_from_single_device_arrays(
            shape=weight_shape,
            sharding=sharding,
            arrays=device_arrays,
        )

        t_assemble = time.perf_counter()

        mode = "RANDOM" if use_dummy else "FILE"
        print(
            f"[host {jax.process_index()}] _load_weight_from_file_on_cpu "
            f"'{tensor_name}' shape={weight_shape} ({mode}): "
            f"setup={t_setup - t_start:.3f}s, "
            f"read={t_read - t_setup:.3f}s, "
            f"shard={t_shard - t_read:.3f}s, "
            f"assemble={t_assemble - t_shard:.3f}s, "
            f"total={t_assemble - t_start:.3f}s",
            flush=True,
        )

        return result

    return _load_weight_from_file_on_cpu


def _torch_to_jax_via_colocated(
    tensor: torch.Tensor,
    tpu_sharding: NamedSharding,
) -> jax.Array:
    """Convert a torch tensor to a JAX array using colocated python.

    Under Pathways there is no local CPU device accessible from the main
    process, so we use ``colocated_python`` to run weight conversion on
    colocated CPU devices (one per host) and then transfer the result to
    the TPU mesh.

    Steps:
        1. Build a colocated CPU mesh that mirrors the TPU mesh topology.
        2. Create a dummy array sharded on the CPU mesh to establish the
           execution context for colocated python.
        3. Build a colocated function that captures the weight data in its
           closure (pickled to remote workers).
        4. Call the colocated python function which creates a sharded CPU
           array and returns it.
        5. Transfer the CPU array to TPU via ``jax.device_put``.

    Args:
        tensor: The PyTorch tensor to convert (in host memory).
        tpu_sharding: Target ``NamedSharding`` for the TPU mesh.
    """
    tpu_mesh = tpu_sharding.mesh
    cpu_mesh = _colocated_cpu_mesh(tpu_mesh)

    dtype = to_jax_dtype(tensor.dtype)
    np_tensor = tensor.detach().cpu().to(torch.float32).numpy()
    global_shape = tuple(tensor.shape)

    import time as _time
    t0 = _time.perf_counter()

    # Create a dummy CPU-sharded array to establish the execution context.
    # The dummy just carries device/mesh info to the colocated function;
    # we use a simple replicated sharding so the dummy shape doesn't need to
    # be divisible by the weight's partition spec.  The actual weight pspec
    # is captured in the closure instead.
    dummy_sharding = NamedSharding(cpu_mesh, P())
    dummy_cpu = jax.device_put(jnp.zeros(len(cpu_mesh.devices.flat)),
                               dummy_sharding)

    t1 = _time.perf_counter()

    # Build the colocated function with weight data captured in the closure.
    load_fn = _make_load_weight_on_cpu(np_tensor, dtype, global_shape,
                                       tpu_sharding.spec)

    t2 = _time.perf_counter()

    # Run on colocated CPUs – returns a sharded CPU array.
    cpu_array = load_fn(dummy_cpu)

    t3 = _time.perf_counter()

    # Transfer from colocated CPU devices to TPU devices directly,
    # without going through the Pathways client (host).
    with (
        jax.transfer_guard_device_to_host("disallow_explicit"),
        jax.transfer_guard_host_to_device("disallow_explicit"),
    ):
        tpu_array = jax.device_put(cpu_array, tpu_sharding)

    t4 = _time.perf_counter()

    logger.info(
        "_torch_to_jax_via_colocated shape=%s: "
        "dummy=%.3fs, build_fn=%.3fs, colocated_load=%.3fs, "
        "cpu_to_tpu=%.3fs, total=%.3fs",
        global_shape,
        t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0,
    )

    return tpu_array


def _torch_to_jax_via_colocated_file(
    file_path: str,
    tensor_name: str,
    tensor_dtype: torch.dtype,
    tpu_sharding: NamedSharding,
    tensor_shape: tuple[int, ...],
    use_dummy: bool = False,
) -> jax.Array:
    """Load a weight tensor via colocated python (file or dummy).

    When *use_dummy* is True, every colocated CPU worker generates small
    random values with the correct shape/dtype — no file I/O, no data from
    the head.

    When *use_dummy* is False, every colocated CPU worker reads the
    safetensors file from shared storage in parallel.

    Args:
        file_path: Path to the safetensors file (unused when *use_dummy*).
        tensor_name: Key of the tensor inside the safetensors file.
        tensor_dtype: The original PyTorch dtype (used to derive JAX dtype).
        tpu_sharding: Target ``NamedSharding`` for the TPU mesh.
        tensor_shape: The global shape of the weight tensor.
        use_dummy: If True, generate zeros instead of reading the file.
    """
    tpu_mesh = tpu_sharding.mesh
    cpu_mesh = _colocated_cpu_mesh(tpu_mesh)

    dtype = to_jax_dtype(tensor_dtype)

    import time as _time
    t0 = _time.perf_counter()

    # Create a dummy CPU-sharded array to establish the execution context.
    # The dummy just carries device/mesh info to the colocated function;
    # we use a simple replicated sharding so the dummy shape doesn't need to
    # be divisible by the weight's partition spec.  The actual weight pspec
    # is captured in the closure instead.
    dummy_sharding = NamedSharding(cpu_mesh, P())
    dummy_cpu = jax.device_put(jnp.zeros(len(cpu_mesh.devices.flat)),
                               dummy_sharding)

    t1 = _time.perf_counter()

    # Build the colocated function with file metadata captured in the closure.
    load_fn = _make_load_weight_from_file_on_cpu(file_path, tensor_name, dtype,
                                                  tpu_sharding.spec,
                                                  tensor_shape,
                                                  use_dummy=use_dummy)

    t2 = _time.perf_counter()

    # Each host runs the colocated function in parallel on colocated CPUs.
    cpu_array = load_fn(dummy_cpu)

    t3 = _time.perf_counter()

    # Transfer from colocated CPU devices to TPU devices directly,
    # without going through the Pathways client (host).
    with (
        jax.transfer_guard_device_to_host("disallow_explicit"),
        jax.transfer_guard_host_to_device("disallow_explicit"),
    ):
        tpu_array = jax.device_put(cpu_array, tpu_sharding)

    t4 = _time.perf_counter()

    logger.info(
        "_torch_to_jax_via_colocated_file '%s' from '%s': "
        "dummy=%.3fs, build_fn=%.3fs, colocated_load=%.3fs, "
        "cpu_to_tpu=%.3fs, total=%.3fs",
        tensor_name, file_path,
        t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0,
    )

    return tpu_array


def _torch_to_jax(tensor: torch.Tensor,
                  sharding: NamedSharding | None = None) -> jax.Array:
    """Convert a torch tensor to a JAX array.

    Under Pathways we use *colocated python* to load weights on colocated
    CPU devices and then transfer to the TPU mesh.  This avoids the
    absence of a local CPU device on the main Pathways process and
    prevents placing a full unsharded copy on a single device (OOM).

    Outside Pathways, ``t2j`` works fine because the caller sets
    ``jax.default_device(cpu)``.

    Args:
        tensor: The PyTorch tensor to convert.
        sharding: Target sharding for direct placement.  Required under
            Pathways; ignored otherwise.
    """
    if vllm_envs.VLLM_TPU_USING_PATHWAYS:
        assert sharding is not None, (
            "_torch_to_jax: sharding must be provided under Pathways")
        return _torch_to_jax_via_colocated(tensor, sharding)
    return t2j(tensor, use_dlpack=False)


def _load_weight_for_layer(
    layer: torch.nn.Module,
    param_name: str,
    tpu_sharding: NamedSharding,
) -> jax.Array:
    """Load a layer's weight parameter onto the TPU mesh.

    Behaviour depends on the environment:

    * **Outside Pathways** — delegates to ``_torch_to_jax``.
    * **Pathways + PATHWAYS_DUMMY_WEIGHTS** — ALL parameters (merged and
      non-merged) are generated as small random values directly on colocated
      CPU workers via ``_torch_to_jax_via_colocated_file``.  No real data
      leaves the head; the CPU tensor storage is freed eagerly.
    * **Pathways (real weights)** — non-merged parameters (single
      ``_weight_file_refs`` entry) are loaded in parallel from shared
      storage on every host via ``_torch_to_jax_via_colocated_file``.
      Merged parameters (QKV, MergedColumn) fall back to
      ``_torch_to_jax`` which sends the pre-merged tensor through the
      head via ``_torch_to_jax_via_colocated``.

    Args:
        layer: The ``torch.nn.Module`` owning the weight.
        param_name: Name of the parameter (e.g. ``"weight"`` or ``"bias"``).
        tpu_sharding: Target ``NamedSharding`` for the TPU mesh.

    Returns:
        A ``jax.Array`` sharded on the TPU mesh.
    """
    tensor = getattr(layer, param_name)

    if not vllm_envs.VLLM_TPU_USING_PATHWAYS:
        return _torch_to_jax(tensor, sharding=tpu_sharding)

    use_dummy = _use_dummy_weights()

    if use_dummy:
        # ---- Dummy path ----
        # All parameters (merged and non-merged) become small random values
        # on colocated CPUs.  Only shape/dtype are read from the tensor
        # metadata.
        tensor_shape = tuple(tensor.shape)
        tensor_dtype = tensor.dtype
        # Free CPU storage eagerly — we only needed shape/dtype.
        tensor.untyped_storage().resize_(0)
        logger.info(
            "Dummy colocated loading for '%s' shape=%s dtype=%s",
            param_name, tensor_shape, tensor_dtype)
        return _torch_to_jax_via_colocated_file(
            file_path="<dummy>",
            tensor_name=param_name,
            tensor_dtype=tensor_dtype,
            tpu_sharding=tpu_sharding,
            tensor_shape=tensor_shape,
            use_dummy=True,
        )

    # ---- Real weights path ----
    file_refs = getattr(layer, '_weight_file_refs', [])

    if not file_refs:
        # No file refs (e.g. standard IncrementalModelLoader) — send
        # tensor data through the head.
        return _torch_to_jax(tensor, sharding=tpu_sharding)

    # Collect file refs that apply to *this* parameter.
    param_refs = [(hf_name, file_path)
                  for hf_name, file_path, pname, _args, _kwargs in file_refs
                  if pname == param_name]

    if len(param_refs) == 1:
        # Non-merged: one safetensors tensor → one layer parameter.
        # Every host reads the file from shared storage in parallel.
        hf_name, file_path = param_refs[0]
        logger.info(
            "Distributed file loading for '%s' tensor '%s' from '%s'",
            param_name, hf_name, file_path)
        return _torch_to_jax_via_colocated_file(
            file_path=file_path,
            tensor_name=hf_name,
            tensor_dtype=tensor.dtype,
            tpu_sharding=tpu_sharding,
            tensor_shape=tuple(tensor.shape),
            use_dummy=False,
        )

    # Merged: multiple safetensors tensors combined into one layer parameter.
    # Fall back to head-based transfer.
    if param_refs:
        logger.info(
            "Merged layer '%s' has %d contributing tensors; "
            "falling back to head-based transfer",
            param_name, len(param_refs))
    return _torch_to_jax(tensor, sharding=tpu_sharding)


@register_quantization_config(UNQUANTIZED)
class VllmUnquantizedConfig(QuantizationConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return UNQUANTIZED

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # Always supported

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # No extra configs required.

    @classmethod
    def from_config(cls, _: dict[str, Any]) -> "VllmUnquantizedConfig":
        return cls()

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        match layer:
            case vllm_linear.LinearBase():
                linear_config = self.get_linear_config(layer)
                return VllmUnquantizedLinearMethod(linear_config)
            case FusedMoE():
                moe_config = self.get_moe_config(layer)
                return VllmUnquantizedFusedMoEMethod(moe_config, self.mesh)
            case Attention():
                return None
            case VocabParallelEmbedding():
                return VllmUnquantizedEmbeddingMethod(self.mesh)
            case _:
                return None


class VllmUnquantizedEmbeddingMethod(UnquantizedEmbeddingMethod):

    def __init__(self, mesh):
        self.mesh = mesh

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = t2j(layer.weight, use_dlpack=False)
        weight = jax.device_put(
            weight,
            NamedSharding(self.mesh, P(ShardingAxisName.MLP_TENSOR, None)))
        layer.weight = Parameter(torch_view(weight), requires_grad=False)

        if isinstance(layer, ParallelLMHead) and layer.bias is not None:
            bias = t2j(layer.bias, use_dlpack=False)
            bias = jax.device_put(
                bias, NamedSharding(self.mesh, P(ShardingAxisName.MLP_TENSOR)))
            layer.bias = Parameter(torch_view(bias), requires_grad=False)


class VllmUnquantizedLinearMethod(vllm_linear.UnquantizedLinearMethod,
                                  common_unquantized.UnquantizedLinearMethod,
                                  VllmQuantizationMethod):

    def __init__(self, linear_config: VllmQuantLinearConfig):
        super().__init__(linear_config)

    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        """Check if all weights are loaded for the layer. If so, process and shard the weights."""
        if isinstance(layer, vllm_linear.QKVParallelLinear):
            assert len(args) == 1, "Expecting shard_id as the only argument"
            shard_id = args[0]
            # Keep track of loaded weights for QKVLinear, e.g. (('weight', 'q'), ('bias', 'q'), ('weight', 'k'), ('bias', 'k'), ...)
            layer._loaded_weights.add((param_name, shard_id))
        elif isinstance(layer, vllm_linear.MergedColumnParallelLinear):
            assert len(args) == 1, "Expecting shard_id as the only argument"
            shard_id = args[0]
            layer._loaded_weights.add((param_name, shard_id))
        else:
            # Keep track of loaded weights for other linear layers, e.g. ('weight', 'bias')
            layer._loaded_weights.add(param_name)

        if len(layer._loaded_weights) == self.linear_config.num_proj * len(
                dict(layer.named_parameters(recurse=False))):
            logger.debug(f"Start sharding weights for layer {type(layer)}")
            self.process_weights_after_loading(layer)
            logger.debug(f"Complete sharding weights for layer {type(layer)}")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.weight):
            # Already processed and sharded.
            return
        # Under Pathways, shard weights directly onto the TPU mesh to avoid
        # placing a full unsharded copy on a single device (OOM).
        weight_sharding = NamedSharding(self.linear_config.mesh,
                                        self.linear_config.weight_sharding)
        weight = _load_weight_for_layer(layer, "weight", weight_sharding)

        # Free CPU memory immediately
        layer.weight.untyped_storage().resize_(0)
        delattr(layer, 'weight')
        if layer.bias is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias_sharding = NamedSharding(self.linear_config.mesh,
                                          self.linear_config.bias_sharding)
            bias = _load_weight_for_layer(layer, "bias", bias_sharding)
            layer.bias.untyped_storage().resize_(0)
            delattr(layer, 'bias')
        else:
            bias = None

        @jax.jit
        def process_unquantized_linear_weights(
            weight: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=None,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
            )

        weights = process_unquantized_linear_weights(weight, bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))
        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert isinstance(layer, vllm_linear.LinearBase)

        with jax.named_scope(layer._get_name()):
            if in_sharding := self.linear_config.get_input_sharding(x):
                x.shard_(NamedSharding(self.linear_config.mesh, in_sharding))

            x_jax = jax_view(x)
            bias_jax = jax_view(
                bias) if bias is not None and not layer.skip_bias_add else None
            if self.linear_config.fuse_matmuls:
                weight_jax = jax_view(layer.weight)
                out_jax = self._apply_fused(x_jax, weight_jax, bias_jax)
                out: torch.Tensor = torch_view(out_jax)
            else:
                assert isinstance(layer.weight, torch.nn.ParameterList)
                # jax_view cannot handle ParameterList directly, so explicitly
                # convert to list.
                weight_jax = [jax_view(w) for w in layer.weight]
                if bias_jax is not None:
                    assert isinstance(layer.bias, torch.nn.ParameterList)
                    bias_jax = [jax_view(b) for b in layer.bias]
                out_jax = self._apply_split(x_jax, weight_jax, bias_jax)
                out: torch.Tensor = torch_view(out_jax)

            if out_sharding := self.linear_config.get_output_sharding(out):
                out.shard_(NamedSharding(self.linear_config.mesh,
                                         out_sharding))

        return out


class VllmUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod,
                                    VllmQuantizationMethod):

    def __init__(
        self,
        moe: FusedMoEConfig,
        mesh: Mesh,
        ep_axis_name: str = "model",
    ):
        super().__init__(moe)
        self.mesh = mesh
        self.moe_backend = select_moe_backend_from_fused_moe_config(self.moe)

        self.extra_backend_kwargs = {}
        if self.moe_backend == MoEBackend.FUSED_MOE:
            # When fused moe kernle is used, we pass extra arguments like
            # tuned block sizes to the kernel.
            self.extra_backend_kwargs = dict(ep_axis_name=ep_axis_name, )

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self) -> Callable:
        return self.apply_monolithic

    def maybe_process_weights(self, layer: torch.nn.Module, param_name: str,
                              args, kwargs):
        """Check if all weights are loaded for the layer. If so, process and shard the weights."""
        expert_id = kwargs.get('expert_id')
        shard_id = kwargs.get('shard_id')
        assert expert_id is not None, "Expecting expert_id argument"
        assert shard_id is not None, "Expecting shard_id argument"
        # Keep track of loaded weights for MoE layers, e.g. (('0', 'w1'), ('0', 'w2'), ('0', 'w3'), ('1', 'w1'), ...)
        layer._loaded_weights.add((expert_id, shard_id))
        if len(layer._loaded_weights) == layer.global_num_experts * len(
            ('w1', 'w2', 'w3')):
            logger.debug(f"Start sharding weights for layer {type(layer)}")
            self.process_weights_after_loading(layer)
            logger.debug(f"Complete sharding weights for layer {type(layer)}")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _tensor_is_in_cpu(layer.w13_weight):
            # Already processed and sharded.
            return
        assert isinstance(layer, FusedMoE)

        # Under Pathways, shard weights directly onto the TPU mesh to avoid
        # placing a full unsharded copy on a single device (OOM for large MoE).
        ep_sharding = NamedSharding(self.mesh, P(ShardingAxisName.EXPERT))
        w13_weight = _load_weight_for_layer(layer, "w13_weight", ep_sharding)
        w2_weight = _load_weight_for_layer(layer, "w2_weight", ep_sharding)
        # Free CPU memory immediately
        layer.w13_weight.untyped_storage().resize_(0)
        layer.w2_weight.untyped_storage().resize_(0)
        delattr(layer, 'w13_weight')
        delattr(layer, 'w2_weight')

        if self.moe.has_bias:
            w13_bias = _load_weight_for_layer(layer, "w13_bias", ep_sharding)
            w2_bias = _load_weight_for_layer(layer, "w2_bias", ep_sharding)
            layer.w13_bias.untyped_storage().resize_(0)
            layer.w2_bias.untyped_storage().resize_(0)
            delattr(layer, 'w13_bias')
            delattr(layer, 'w2_bias')
        else:
            w13_bias = w2_bias = None

        @jax.jit
        def process_unquantized_moe_weights(
            w13_weight: jax.Array,
            w13_bias: jax.Array | None,
            w2_weight: jax.Array,
            w2_bias: jax.Array | None,
        ) -> FusedMoEWeights:
            w13_interleave = layer.activation == MoEActivation.SWIGLUOAI
            w13_reorder_size = get_mesh_shape_product(
                self.mesh, ShardingAxisName.MLP_TENSOR)

            return process_moe_weights(
                FusedMoEWeights(
                    w13_weight=w13_weight,
                    w13_weight_scale=None,
                    w13_bias=w13_bias,
                    w2_weight=w2_weight,
                    w2_weight_scale=None,
                    w2_bias=w2_bias,
                ),
                moe_backend=self.moe_backend,
                w13_reorder_size=w13_reorder_size,
                w13_interleave=w13_interleave,
            )

        weights = process_unquantized_moe_weights(
            w13_weight,
            w13_bias,
            w2_weight,
            w2_bias,
        )
        weights = torch_view(
            shard_moe_weights(weights, self.moe_backend, self.mesh))
        layer.w13_weight = Parameter(weights.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(weights.w2_weight, requires_grad=False)

        if self.moe.has_bias:
            layer.w13_bias = Parameter(weights.w13_bias, requires_grad=False)
            layer.w2_bias = Parameter(weights.w2_bias, requires_grad=False)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:

        weights = FusedMoEWeights(
            w13_weight=jax_view(layer.w13_weight),
            w13_weight_scale=None,
            w13_bias=jax_view(layer.w13_bias) if self.moe.has_bias else None,
            w2_weight=jax_view(layer.w2_weight),
            w2_weight_scale=None,
            w2_bias=jax_view(layer.w2_bias) if self.moe.has_bias else None,
        )

        return vllm_moe_apply(layer=layer,
                              weights=weights,
                              quant_method_instance=self,
                              x=x,
                              router_logits=router_logits)
