import abc
from typing import List, Optional, Tuple

import jax
from flax import nnx
from flax.typing import PRNGKey
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.common.attention.attention import AttentionMetadata
from tpu_commons.models.jax.common.constants import KVCacheType

logger = init_logger(__name__)


class Model(nnx.Module, abc.ABC):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: PRNGKey,
                 mesh: Mesh,
                 force_random_weights: bool = False):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, kv_caches: List[KVCacheType], input_ids: jax.Array,
                 attention_metadata: AttentionMetadata,
                 *args) -> Tuple[List[KVCacheType], jax.Array, jax.Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_weights(self, rng: jax.Array, cache_dir: Optional[str] = None):
        raise NotImplementedError
