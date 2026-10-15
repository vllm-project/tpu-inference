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
"""The TPU attention backends, selected with --attention-backend <name>, which
picks the attention and the linear attention backend of that name.

To add a backend, use one lowercase name as the key in:
- ATTENTION_BACKENDS (required): its AttentionBackend, whose hooks the
  runner, the KV cache allocator and attention() call.
- LINEAR_ATTN_BACKENDS (optional): its LinearAttentionBackend, the GDN
  kernel of hybrid models. Without one, the name runs GDN v3.
- VLLM_ATTENTION_BACKENDS (required): the class path of its vLLM adapter,
  the class vLLM's Attention layer builds: a PallasAttentionBackend
  subclass whose get_impl_cls returns an impl with attention_backend set
  to the name.

vLLM's --attention-backend picks the framework adapter, not the TPU kernel
behind it. register_vllm_attention_backends adds the names of
VLLM_ATTENTION_BACKENDS to vLLM's AttentionBackendEnum, so that the flag
accepts them, and TpuPlatform.get_attn_backend_cls maps each name to its
adapter: PallasAttentionBackend with an impl that passes the name on to
attention_interface.attention. The runner, the KV cache allocator and the GDN
op read the same name from vllm_config (attention_backend_name). JAX
(flax_nnx) models call attention() without a backend and run RPA.

AttentionBackend itself is the default "rpa" backend: each hook returns
nothing to add, and the RPA code inline in the platform, the runner, the
compilation manager and attention_interface runs as it is. Another backend
overrides the hooks it needs, the way a vLLM backend implements
validate_configuration / get_kv_cache_shape / AttentionMetadataBuilder.build /
build_for_cudagraph_capture / AttentionImpl.forward.

The linear attention layers of a hybrid model have kernels of their own, in
LINEAR_ATTN_BACKENDS; for now one name selects both families. A name without
a linear attention backend keeps LinearAttentionBackend, the "default"
backend, the GDN v3 kernel.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    from tpu_inference.layers.common.attention_metadata import (
        AttentionMetadata, SharedAttentionMetadata)
    from tpu_inference.utils import DeviceBuffer

AttentionFields = Callable[[Mapping[str, "jax.Array"], int],
                           dict[str, "str | int | jax.Array | None"]]


def no_attention_fields(metadata: Mapping[str, "jax.Array"],
                        gid: int) -> dict[str, "str | int | jax.Array | None"]:
    del metadata, gid
    return {}


@dataclass(frozen=True)
class HostStepInputs:
    """The host arrays TPUModelRunner._prepare_inputs has filled for one step.

    They follow its per-DP-rank layout: rank r's requests start at
    r * max_num_reqs_per_dp_rank in seq_lens and the block tables, and at
    r * (max_num_reqs_per_dp_rank + 1) in query_start_loc. request_distribution
    is the flat (dp_size * 3,) [num_decode, num_decode, num_reqs] per rank, and
    block_tables maps a KV cache group id to its (max_num_seqs,
    max_num_blocks_per_req) view.
    """
    kv_cache_groups: "Sequence[KVCacheGroupSpec]"
    block_tables: Mapping[int, np.ndarray]
    query_start_loc: np.ndarray
    seq_lens: np.ndarray
    request_distribution: np.ndarray
    scheduled_tokens_per_dp_rank: Mapping[int, Sequence[int]]
    dp_size: int
    max_num_reqs_per_dp_rank: int
    padded_num_scheduled_tokens_per_dp_rank: int


class AttentionBackend:
    """The default "rpa" backend; see the module docstring."""

    name = "rpa"

    zero_warmup_lengths = False

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Raise for an engine configuration this backend cannot serve; may
        adjust it (e.g. the block size). Runs in
        TpuPlatform.check_and_update_config, before any worker starts."""

    def get_kv_cache_shape(self, total_num_pages: int, page_size: int,
                           num_kv_heads: int, head_dim: int,
                           kv_dtype: "jnp.dtype") -> tuple[int, ...] | None:
        """One shard's KV cache shape for num_kv_heads heads, or None for the
        RPA shape."""
        return None

    def attention(
        self,
        kv_cache: "jax.Array",
        q: "jax.Array",
        k: "jax.Array",
        v: "jax.Array",
        md: "AttentionMetadata",
        shared_md: "AttentionMetadata | SharedAttentionMetadata",
        mesh: "Mesh",
        *,
        sm_scale: float,
        attention_chunk_size: int | None,
        q_scale: float | None,
        k_scale: float | None,
        v_scale: float | None,
        sinks: "jax.Array | None",
        update_kv_cache: bool,
        use_causal_mask: bool,
        attn_logits_soft_cap: float | None,
        decode_query_size: int,
    ) -> tuple["jax.Array", "jax.Array"] | None:
        """(kv_cache, output) of attention_interface.attention, or None for
        its RPA path."""
        return None

    def prepare_step(self, step: HostStepInputs,
                     device_buffer: "DeviceBuffer") -> AttentionFields:
        """Host-side metadata of one step. Arrays the device needs go into
        device_buffer, which reaches the device in the step's one transfer;
        the returned fields read them back from the unpacked buffer, per KV
        cache group."""
        return no_attention_fields

    def warmup_variants(self, kv_cache_groups: "Sequence[KVCacheGroupSpec]", *,
                        num_tokens: int, num_tokens_paddings: Sequence[int],
                        max_num_reqs: int,
                        dp_size: int) -> list[AttentionFields]:
        """The fields of each backbone executable a step at this token bucket
        can ask for; the warmup compiles each one."""
        return [no_attention_fields]

    def warmup_arrays(self, kv_cache_groups: "Sequence[KVCacheGroupSpec]", *,
                      num_tokens: int, max_num_reqs: int, dp_size: int,
                      mesh: "Mesh",
                      sharding: "NamedSharding") -> Mapping[str, "jax.Array"]:
        """Dummy arrays for the warmup_variants fields to read, shaped like
        prepare_step's so serving reuses the executable."""
        return {}


ATTENTION_BACKENDS: dict[str, Callable[[], AttentionBackend]] = {
    "rpa": AttentionBackend,
}

GDNKernel = Callable[..., tuple[tuple["jax.Array", "jax.Array"], "jax.Array"]]


class LinearAttentionBackend:
    """The default linear attention backend; see the module docstring."""

    name = "default"

    def check_config(self, vllm_config: "VllmConfig") -> None:
        """Same contract as AttentionBackend.check_config."""

    def gdn_kernel(self) -> GDNKernel:
        """The fused conv1d + GDN kernel run_jax_gdn_attention shard-maps,
        with the signature of kernels.gdn.v3.wrapper.fused_conv1d_gdn."""
        # Note (david): imported here so that the platform, which imports this
        # module, does not load the Pallas kernels.
        from tpu_inference.kernels.gdn.v3 import wrapper
        return wrapper.fused_conv1d_gdn


LINEAR_ATTN_BACKENDS: dict[str, Callable[[], LinearAttentionBackend]] = {
    "default": LinearAttentionBackend,
}

# The vLLM backend class that TpuPlatform.get_attn_backend_cls returns for each
# name besides "rpa" (vLLM's FLASH_ATTN, PallasAttentionBackend).
VLLM_ATTENTION_BACKENDS: dict[str, str] = {}


def register_vllm_attention_backends() -> None:
    """Add the names of VLLM_ATTENTION_BACKENDS to vLLM's AttentionBackendEnum,
    so that --attention-backend accepts them. Runs from the
    vllm.general_plugins hook, which vLLM calls in every process, and before
    it parses the flag."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    if not hasattr(AttentionBackendEnum, "register"):
        # vLLM before vllm#41091 cannot add a name.
        return
    for name, class_path in VLLM_ATTENTION_BACKENDS.items():
        if name.upper() not in AttentionBackendEnum.__members__:
            AttentionBackendEnum.register(name.upper(), class_path)


def attention_backend_name(vllm_config: "VllmConfig") -> str:
    """The backend --attention-backend selects: a name of
    VLLM_ATTENTION_BACKENDS, else "rpa" (FLASH_ATTN, no flag, or a backend vLLM
    picked itself, such as TRITON_ATTN for Gemma-4)."""
    backend = vllm_config.attention_config.backend
    if backend is not None and backend.name.lower() in VLLM_ATTENTION_BACKENDS:
        return backend.name.lower()
    return AttentionBackend.name


def create_attention_backend(name: str) -> AttentionBackend:
    """The attention backend of that name."""
    return ATTENTION_BACKENDS[name]()


def create_linear_attention_backend(name: str) -> LinearAttentionBackend:
    """The linear attention backend of that name, or the default one when the
    name only selects an attention backend."""
    return LINEAR_ATTN_BACKENDS.get(name, LinearAttentionBackend)()


def check_attention_backend_config(vllm_config: "VllmConfig") -> None:
    """Raise for an engine configuration the selected backends cannot serve.
    Runs in TpuPlatform.check_and_update_config, before any worker starts."""
    name = attention_backend_name(vllm_config)
    if name != AttentionBackend.name:
        check_forked_single_host(name)
    create_attention_backend(name).check_config(vllm_config)
    create_linear_attention_backend(name).check_config(vllm_config)


def check_forked_single_host(name: str) -> None:
    """Raise unless vllm_config only reaches forked processes.

    register_vllm_attention_backends adds the member for name at runtime.
    Enum members pickle by value, and a process that unpickles vllm_config
    before it loads the plugins, a spawned EngineCore or a Ray worker, cannot
    rebuild the member. A forked process inherits it."""
    from vllm import envs as vllm_envs
    from vllm.ray.lazy_utils import is_in_ray_actor

    from tpu_inference import envs
    reasons = []
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        reasons.append("TPU_MULTIHOST_BACKEND=ray")
    if vllm_envs.VLLM_WORKER_MULTIPROC_METHOD == "spawn":
        reasons.append("VLLM_WORKER_MULTIPROC_METHOD=spawn")
    if is_in_ray_actor():
        reasons.append("the engine runs in a Ray actor")
    if reasons:
        raise ValueError(
            f"--attention-backend {name} supports a single host with forked "
            f"processes only, not: {', '.join(reasons)}.")
