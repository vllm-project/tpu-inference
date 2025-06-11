# from typing import Tuple

# import jax
# from flax import nnx
# from jax.sharding import Mesh
# from vllm.config import VllmConfig

# from tpu_commons.logger import init_logger
# from tpu_commons.models.jax.layers.misc import Embedder

# logger = init_logger(__name__)

# init_fn = nnx.initializers.uniform()

# class Qwen2Model(nnx.Module):
#     vllm_config: VllmConfig
#     mesh: Mesh

# class Qwen2ForCausalLM(nnx.Module):
#     config: VllmConfig
#     rng: jax.Array
#     mesh: Mesh

#     @nnx.compact
#     def setup(self) -> None:
#         model_config = self.vllm_config.model_config

#         self.embed_tokens = Embedder(
#             vocab_size=model_config.get_vocab_size(),
#             hidden_size=model_config.get_hidden_size(),
#             dtype=model_config.dtype,
#             mesh=self.mesh,
#         )
#         self.model = Qwen2Model(
#             vllm_config=self.vllm_config,
#             mesh=self.mesh,
#         )
#         # try:
#         #     self.lm_head = self.param(
#         #         "lm_head",
#         #         sharding_init(
#         #             (None, "model"),
#         #             self.mesh,
#         #         ),
#         #         (model_config.get_hidden_size(),
#         #          model_config.get_vocab_size()),
#         #         model_config.dtype,
#         #     )
#         # except Exception:
#         #     self.lm_head = None
#         self.lm_head = None
