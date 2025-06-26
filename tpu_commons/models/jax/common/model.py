import abc
from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import jax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.kv_cache import KVCacheType
from tpu_commons.models.jax.common.layers import EmbedderConfig
from tpu_commons.models.jax.common.transformer_block import \
    TransformerBlockConfig


@dataclass
class ModelConfig():
    emb: EmbedderConfig
    layers: TransformerBlockConfig
    num_layers: int

    # TODO: If user passes --additional-config in vllm, then update the configs with it.
    def apply_vllm_overrides(self, overrides: Mapping[str, Any]):
        if overrides is not None:
            pass


class Model(nnx.Module, abc.ABC):

    def __init__(self, vllm_config: VllmConfig, rng: PRNGKey, mesh: Mesh):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self,
                 is_prefill: bool,
                 do_sampling: bool,
                 kv_caches: List[KVCacheType],
                 input_ids: jax.Array,
                 attention_metadata: AttentionMetadata,
                 temperatures: jax.Array = None,
                 top_ps: jax.Array = None,
                 top_ks: jax.Array = None,
                 *args,
                 **kwargs) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:
        raise NotImplementedError
