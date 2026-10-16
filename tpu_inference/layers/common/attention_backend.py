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
"""The TPU attention backends that --attention-backend <name> selects.

An AttentionBackend runs the kernel of the attention layers and lays out their
KV cache. A LinearAttentionBackend runs the kernel of the GDN linear attention
layers of hybrid models. By default the layers run RPAAttentionBackend (RPA v3)
and GDNLinearAttentionBackend (GDN v3); register_attention_backend adds
another pair under a name.
"""

import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm import envs as vllm_envs
from vllm.ray.lazy_utils import is_in_ray_actor
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tpu_inference import envs

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from vllm.config import VllmConfig

    from tpu_inference.layers.common.attention_metadata import (
        AttentionMetadata, SharedAttentionMetadata)


class AttentionBackend(abc.ABC):
    """The kernel of the attention layers and the layout of their KV cache."""

    @abc.abstractmethod
    def get_kv_cache_shape(self, total_num_pages: int, page_size: int,
                           num_kv_heads: int, head_dim: int,
                           kv_dtype: "jnp.dtype") -> tuple[int, ...]:
        """One shard's KV cache shape for num_kv_heads heads, which
        kv_cache.get_kv_cache_shape_with_mesh scales to the mesh."""

    @abc.abstractmethod
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
        head_dim_original: int,
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
    ) -> tuple["jax.Array", "jax.Array"]:
        """(kv_cache, output) of attention_interface.attention."""


class RPAAttentionBackend(AttentionBackend):
    """The default attention backend: ragged paged attention v3."""

    name = "rpa"

    def get_kv_cache_shape(self, total_num_pages: int, page_size: int,
                           num_kv_heads: int, head_dim: int,
                           kv_dtype: "jnp.dtype") -> tuple[int, ...]:
        # Note: the kernels are imported here so that the platform, which
        # imports this module, does not load them.
        # NOTE(chengjiyao): Currently, the attention kernel is tailored to the
        # specific model, rather than being determined by the head_dim. If new
        # models are introduced with a head_dim of 64, this will require
        # additional model-specific adjustments.
        if head_dim == 64:
            import tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 as rpa
        elif envs.USE_BATCHED_RPA_KERNEL:
            import tpu_inference.kernels.experimental.batched_rpa.wrapper as rpa
        else:
            import tpu_inference.kernels.ragged_paged_attention.v3.kernel as rpa
        return rpa.get_kv_cache_shape(total_num_pages, page_size, num_kv_heads,
                                      head_dim, kv_dtype)

    def attention(self, *args, **kwargs) -> tuple["jax.Array", "jax.Array"]:
        # Note: imported here so that the platform, which imports this module,
        # does not load the kernels.
        from tpu_inference.layers.common.attention_interface import \
            rpa_attention
        return rpa_attention(*args, **kwargs)


# A fused conv1d + GDN kernel, with the signature of
# kernels.gdn.v3.wrapper.fused_conv1d_gdn. It returns
# ((new_conv_state, new_recurrent_state), output): the updated conv state and
# recurrent state caches, and the output of the GDN layer.
Conv1dGDNKernel = Callable[..., tuple[tuple["jax.Array", "jax.Array"],
                                      "jax.Array"]]


class LinearAttentionBackend(abc.ABC):
    """The kernel of the GDN linear attention layers of hybrid models."""

    @abc.abstractmethod
    def conv1d_gdn_kernel(self) -> Conv1dGDNKernel:
        """The fused conv1d + GDN kernel run_jax_gdn_attention shard-maps."""


class GDNLinearAttentionBackend(LinearAttentionBackend):
    """The default linear attention backend: the GDN v3 kernel."""

    def conv1d_gdn_kernel(self) -> Conv1dGDNKernel:
        # Note: imported here so that the platform, which imports this module,
        # does not load the Pallas kernels.
        from tpu_inference.kernels.gdn.v3 import wrapper
        return wrapper.fused_conv1d_gdn


@dataclass(frozen=True)
class _RegisteredBackend:
    attention: Callable[[], AttentionBackend]
    linear_attention: Callable[[], LinearAttentionBackend]


_REGISTERED_BACKENDS: dict[str, _RegisteredBackend] = {}


def register_attention_backend(
    name: str,
    *,
    attention: Callable[[], AttentionBackend],
    vllm_backend: str,
    linear_attention: Callable[
        [], LinearAttentionBackend] = GDNLinearAttentionBackend
) -> None:
    """Make --attention-backend <name> run these backends.

    vllm_backend is the class path of the vLLM attention backend that
    TpuPlatform.get_attn_backend_cls gives the attention layers: a
    PallasAttentionBackend subclass whose get_impl_cls returns a
    PallasAttentionBackendImpl subclass with attention_backend = name.

    Call it from a vllm.general_plugins entry point, which vLLM runs in every
    process before it parses --attention-backend, and may run more than once
    in one process."""
    name = name.lower()
    member = AttentionBackendEnum.__members__.get(name.upper())
    if member is None:
        AttentionBackendEnum.register(name.upper(), vllm_backend)
    elif member.get_path() != vllm_backend:
        raise ValueError(f"vLLM already has an attention backend named "
                         f"{name.upper()}, at {member.get_path()}.")
    _REGISTERED_BACKENDS[name] = _RegisteredBackend(attention,
                                                    linear_attention)


def is_registered_attention_backend(
        backend: AttentionBackendEnum | None) -> bool:
    """Whether vLLM's backend is one of register_attention_backend."""
    return backend is not None and backend.name.lower() in _REGISTERED_BACKENDS


def attention_backend_name(vllm_config: "VllmConfig") -> str:
    """The backend --attention-backend selects: a registered name, else "rpa"
    (FLASH_ATTN, no flag, or a backend vLLM picked itself, such as TRITON_ATTN
    for Gemma-4)."""
    backend = vllm_config.attention_config.backend
    if is_registered_attention_backend(backend):
        return backend.name.lower()
    return RPAAttentionBackend.name


def create_attention_backend(name: str) -> AttentionBackend:
    """The attention backend of that name."""
    if name == RPAAttentionBackend.name:
        return RPAAttentionBackend()
    return _REGISTERED_BACKENDS[name].attention()


def create_linear_attention_backend(name: str) -> LinearAttentionBackend:
    """The linear attention backend of that name."""
    if name == RPAAttentionBackend.name:
        return GDNLinearAttentionBackend()
    return _REGISTERED_BACKENDS[name].linear_attention()


def check_attention_backend_config(vllm_config: "VllmConfig") -> None:
    """Raise if a registered backend is selected for an engine whose processes
    cannot rebuild its name. Runs in TpuPlatform.check_and_update_config,
    before any worker starts.

    register_attention_backend adds the name to AttentionBackendEnum at
    runtime. Enum members pickle by value, and a process that unpickles
    vllm_config before it loads the plugins, a spawned EngineCore or a Ray
    worker, cannot rebuild the member. A forked process inherits it.

    TODO: support spawn and Ray. vLLM must rebuild a registered
    AttentionBackendEnum member when it unpickles it, e.g. by loading the
    general plugins first."""
    name = attention_backend_name(vllm_config)
    if name == RPAAttentionBackend.name:
        return
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
