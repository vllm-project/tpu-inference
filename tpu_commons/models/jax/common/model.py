import abc
from dataclasses import dataclass
from typing import List, Tuple

import jax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.base import Config
from tpu_commons.models.jax.common.kv_cache import KVCacheType
from tpu_commons.models.jax.common.layers import EmbedderConfig
from tpu_commons.models.jax.common.transformer_block import \
    TransformerBlockConfig

logger = init_logger(__name__)


@dataclass
class ModelConfig(Config):
    emb: EmbedderConfig
    layers: TransformerBlockConfig
    num_layers: int
    vllm_config: VllmConfig


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

    def load_weights(self, rng_key: jax.Array):
        try:
            use_random_weights = self.vllm_config.additional_config[
                "random_weights"]
            logger.warning(
                "Using randomly initialized weights instead of loading parameter weights."
            )
            return
        except KeyError:
            use_random_weights = False
        self.weight_loader.load_weights(self)
