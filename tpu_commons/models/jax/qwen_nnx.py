from flax import nnx

from tpu_commons.logger import init_logger

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()

# class Qwen2Model(nnx.Module):

#     def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs,
#                  mesh: Mesh) -> None:
#         model_config = vllm_config.model_config
#         hf_config = model_config.hf_config

# class Qwen2ForCausalLM(nnx.Module):

#     def __init__(self, vllm_config: VllmConfig, rng: jax.Array,
#                  mesh: Mesh) -> None:
#         model_config = vllm_config.model_config
#         vocab_size = model_config.get_vocab_size()
#         hidden_size = model_config.get_hidden_size()
#         dtype = model_config.dtype

#         self.vllm_config = vllm_config
#         self.rng = nnx.Rngs(rng)
#         self.mesh = mesh

#         self.embed = nnx.Embed(
#             num_embeddings=vocab_size,
#             features=hidden_size,
#             param_dtype=dtype,
#             embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
#             rngs=self.rng,
#         )
#         self.model = Qwen2Model(
#             vllm_config=vllm_config,
#             rng=self.rng,
#             mesh=mesh,
#         )
#         self.lm_head = nnx.Param(
#             init_fn(self.rng.params(), (hidden_size, vocab_size), dtype),
#             sharding=(None, "model"),
#         )
