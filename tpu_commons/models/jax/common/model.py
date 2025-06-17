import abc
from typing import Any, Mapping

from flax.typing import PRNGKey
import jax
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.models.jax.common.attention import AttentionMetadata
from tpu_commons.models.jax.common.kv_cache import KVCache_type
from tpu_commons.models.jax.common.layers import (EmbedderConfig, TransformerBlockConfig)


@dataclass
class ModelConfig():
    # embed: EmbedderConfig
    # attn: AttentionConfig
    # ffw: FFWConfig
    emb: EmbedderConfig
    layers: TransformerBlockConfig
    num_layers: int

    # TODO: If user passes --additional-config in vllm, then update the configs with it.
    def apply_vllm_overrides(self, overrides: Mapping[str, Any]):
        if overrides is not None:
            pass


class Model(nnx.Module, abc.ABC):
    vllm_config: VllmConfig
    rng: PRNGKey
    mesh: Mesh

    @abc.abstractmethod
    def __call__(self,
                 is_prefill: bool,
                 do_sampling: bool,
                 kv_caches: List[KVCache_type],
                 input_ids: jax.Array,
                 attention_metadata: AttentionMetadata,
                 temperatures: jax.Array = None,
                 top_ps: jax.Array = None,
                 top_ks: jax.Array = None,
                 *args,
                 **kwargs) -> Tuple[List[KVCache_type], jax.Array, jax.Array]:
        raise NotImplementedError