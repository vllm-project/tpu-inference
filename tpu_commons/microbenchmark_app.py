
from vllm_config_utils import VllmConfig, ModelConfig
import json
from dataclasses import dataclass, field
from typing import Any, List, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
from tpu_commons.kernels.ragged_paged_attention.v3.util import cdiv
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from microbenchmark_utils import get_model
import tpu_commons.kernels.ragged_paged_attention.v3.kernel as rpa
from microbenchmark_utils import Sampler, init_mesh, device_array, get_padded_num_kv_cache_update_slices, nearest_power_of_two
from absl import app, flags
import os

logger = init_logger(__name__)
power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens
_MAX_SEQ_LEN = flags.DEFINE_integer(
    "max_seq_len", 2048, "Maximum sequence length."
)
_MAX_PREFILL_LEN = flags.DEFINE_integer(
    "max_prefill_len", 1024, "Maximum prefill length."
)
_PREFILL_BATCH_SIZE = flags.DEFINE_integer(
    "prefill_batch_size", 1, "Prefill batch size."
)
_PREFILL_STEPS = flags.DEFINE_integer(
    "prefill_steps", 5, "Number of prefill steps."
)
_DECODE_BATCH_SIZE = flags.DEFINE_integer(
    "decode_batch_size", 1, "Decode batch size."
)
_DECODE_STEPS = flags.DEFINE_integer(
    "decode_steps", 5, "Number of decode steps."
)
_BLOCK_SIZE = flags.DEFINE_integer("block_size", 16, "Block size.")
_SAMPLER_TYPE = flags.DEFINE_string("sampler_type", "fixed", "Sampler type.")
_SAMPLER_STD = flags.DEFINE_float(
    "sampler_std", 1.0, "Sampler standard deviation."
)
_ADDITIONAL_CONFIG = flags.DEFINE_string(
    "additional_config",
    "",
    "Additional configuration for the model.",
)
_MODEL_CONFIG = flags.DEFINE_string(
    "model_config",
    "",
    "Model configuration for the model.",
)
NEW_MODEL_DESIGN = flags.DEFINE_string(
    "NEW_MODEL_DESIGN",
    "True",
    "Model design to use. If True, uses the new model design.",
)

@dataclass
class InputArgs:
    block_size: int
    # TODO: split into prefill & decode batch size to allow mixing
    batch_size: int
    max_prefill_len: int
    max_seq_len: int
    sampler: Sampler
    num_kv_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int

@dataclass
class ModelInputs:
    is_prefill: bool
    do_sampling: bool
    kv_caches: List[jax.Array]
    input_ids: jnp.array
    attention_metadata: AttentionMetadata
    temperatures: jnp.array
    top_ps: jnp.array
    top_ks: jnp.array

class InputCreator:
    def __init__(self,
                 input_args: InputArgs,
                 sharding: NamedSharding | None,
                 rng: nnx.Rngs,
                 mesh: jax.sharding.Mesh,
                 permute_block_table: bool = False,
                 kv_cache_dtype: jnp.dtype = jnp.bfloat16):
        self.rng = rng
        self.input_args = input_args
        self.sharding = sharding
        self.mesh = mesh
        self.permute_block_table = permute_block_table
        self.kv_cache_dtype = kv_cache_dtype
        self.setup()
    def setup(self):
        self.max_blocks_per_req = cdiv(self.input_args.max_seq_len,
                                       self.input_args.block_size)
        padded_max_seq_len = self.max_blocks_per_req * self.input_args.block_size
        if padded_max_seq_len != self.input_args.max_seq_len:
            logger.warning(
                f"Padding max_seq_len from {self.input_args.max_seq_len} to {padded_max_seq_len}."
            )
        self.num_blocks = cdiv(self.input_args.batch_size * padded_max_seq_len,
                               self.input_args.block_size)
        self.block_table = self._create_mock_block_table(
            self.permute_block_table)
        self.kv_caches = self.create_kv_caches()

    def create_kv_caches(self):
        # TODO (singhvijaya): update this for quantized KV cache
        cache_dtype = jnp.bfloat16
        shard_cnt = self.mesh.shape["model"]
        # TODO(xiang): fix this together with get_kv_cache_spec
        # cache_dtype = kv_cache_spec.dtype

        # NOTE(jevinjiang): Instead of sharding automatically, we manually calculate
        # the kv cache for each shard because the padding logic for RPA's KV cache
        # needs to know the exact head number on each shard. In other words, we can
        # not determine the padding logics for kv cache globally.
        # shard_cnt = self.mesh.shape["model"]
        # assert num_kv_heads % shard_cnt == 0
        cache_shape_per_shard = rpa.get_kv_cache_shape(self.num_blocks, self.input_args.block_size,
                                                    self.input_args.num_kv_heads // shard_cnt,
                                                    self.input_args.head_dim, cache_dtype)
        # Intended to be replicated.
        sharding = NamedSharding(self.mesh, PartitionSpec())
        key = self.rng.params()
        rng_keys = jax.random.split(key, num=self.input_args.num_layers)

        def _allocate(rng) -> jax.Array:
            return jax.random.normal(rng,
                                     shape=cache_shape_per_shard,
                                     dtype=cache_dtype)
        
        sharded_allocate = jax.jit(_allocate, out_shardings=sharding)
        kv_caches = []
        for i in range(self.input_args.num_layers):
            kv_caches.append(sharded_allocate(rng_keys[i]))
        return kv_caches
    
    def _init_sharded_kv_cache(self):
        kv_cache_shape = (self.num_blocks, self.input_args.block_size,
                          2 * self.input_args.num_kv_heads,
                          self.input_args.head_dim)
        self.kv_caches = []
        # TODO: use constants instead of hardcoding axis names.
        sharding = NamedSharding(self.mesh, PartitionSpec(None, None, "model"))
        key = self.rng.params()
        rng_keys = jax.random.split(key, num=self.input_args.num_layers)
        @nnx.jit(out_shardings=sharding)
        def _allocate(rng: jax.Array):
            return jax.random.normal(rng,
                                     shape=kv_cache_shape,
                                     dtype=self.kv_cache_dtype)
        kv_caches = []
        print("rng type = ", type(self.rng))
        for i in range(self.input_args.num_layers):
            kv_caches.append(_allocate(rng_keys[i]))
        return kv_caches
    
    
    def _create_mock_block_table(self, random_permute: bool = False):
        block_table = np.arange(self.num_blocks,
                          dtype=jnp.int32)\
                             .reshape(self.input_args.batch_size, -1)
        if random_permute:
            block_table = np.random.permutation(block_table)
        return block_table
    # TODO: lets try to split this logic out in TPURUnner or add CI/CD so that if any
    # changes are made to tpu_comons, it doesn't break our code.
    def _mock_kv_write_indices(self, seq_lens: List[int],
                               phase_types: List[str]):
        total_block_len = 0
        block_lens = []
        slice_starts = []
        slice_ends = []
        batch_size = self.input_args.batch_size
        for (seq_len, phase_type) in zip(seq_lens, phase_types):
            # slice_start is the start index of the current request
            # slice_end is the last index of the currnet request
            if phase_type == "prefill":
                slice_start = 0
                slice_end = seq_len
            elif phase_type == "decode":
                slice_start = seq_len
                slice_end = seq_len + 1
            slice_starts.append(slice_start)
            slice_ends.append(slice_end)
            block_start = slice_start // self.input_args.block_size
            block_end = (slice_end - 1) // self.input_args.block_size
            block_len = block_end - block_start
            block_lens.append(block_len)
        total_block_len = sum(block_lens)
        flattened_block_indices = self.block_table.flatten()[:total_block_len]
        # Create placeholder buffer for start and end of each write slice
        slot_mapping_slices = np.repeat(np.array(
            [[0, self.input_args.block_size]], dtype=np.int32),
                                        total_block_len,
                                        axis=0)
        block_lens_cumsum = np.zeros(len(block_lens) + 1, dtype=np.int32)
        np.cumsum(block_lens, out=block_lens_cumsum[1:])
        # Slots correspond to amount of blocks being reserved for the requests.
        # We assume requests occupy k-2 full blocks + 1 possibly partially full, starting block + 1 partiall full, ending block
        for req_idx in range(batch_size):
            # Update the start and end of the slices if they are not evenly divisible by block size.
            # Start block sizes only need to be updated at the start of each request.
            slot_mapping_slices[block_lens_cumsum[req_idx]][0] = \
               slice_starts[req_idx] % self.input_args.block_size
            # End block sizes only need to be updated at the end of each request.
            slot_mapping_slices[
               block_lens_cumsum[req_idx + 1] - 1][1] = \
                  (slice_ends[req_idx] - 1) % self.input_args.block_size + 1
            # Number of tokens corresponding to write for the request.
            slice_lens = slot_mapping_slices[:, 1] - slot_mapping_slices[:, 0]
            slices_lens_cumsum = np.zeros(len(slice_lens) + 1, dtype=np.int32)
            np.cumsum(slice_lens, out=slices_lens_cumsum[1:])
            # Map the block indices back to KV cache token indices
            kv_cache_start_indices = slot_mapping_slices[:, 0] + \
               (flattened_block_indices * self.input_args.block_size)
            # The new KV cache indices to write to.
            # (essentially the indices corresponding to request_size + previously_computed)
            new_kv_start_indices = slices_lens_cumsum[:-1]
            slot_mapping_metadata = np.stack(
                [kv_cache_start_indices, new_kv_start_indices, slice_lens],
                axis=1)
            # TODO: Perform padding from jax/tpu_jax_runner.py:403
            return slot_mapping_metadata
    def create_prefill_input(self,
                             previous_input: AttentionMetadata = None
                             ) -> ModelInputs:
        # NOTE(singhvijaya) seq_lens is padded shape (max_seq_len) in tpu_jax_runner: https://github.com/vllm-project/tpu_commons/blob/38df9a7cbdc490bae5a8f63938b518e0e636d829/tpu_commons/runner/jax/tpu_jax_runner.py#L713
        # But I don't think this should affect things much.
        batch_size = self.input_args.batch_size
        seq_lens = self.input_args.sampler.generate_samples(
            shape=(batch_size, ), fill_val=self.input_args.max_prefill_len)
        if not self.input_args.sampler:
            seq_lens = batch_size * [self.input_args.max_prefill_len]
        total_tokens = sum(seq_lens)
        padded_total_tokens = nearest_power_of_two(total_tokens)
        ## Generate random input_ids
        input_ids = np.random.randint(
            low=0,
            high=self.input_args.vocab_size - 1,
            size=(padded_total_tokens, ),  # Pad to nearest power of two
            dtype=np.int32)
        num_decode_seqs = batch_size
        padded_input_positions = np.zeros(padded_total_tokens, dtype=np.int32)
        # Calculate token positions within each sequence
        input_positions = np.concatenate(
            [np.arange(seq_len, dtype=np.int32) for seq_len in seq_lens],
            out=padded_input_positions)  # Pad to nearest power of 2
        kv_cache_write_indices = self._mock_kv_write_indices(
            seq_lens, ["prefill"] * len(seq_lens))
        # Padd the kv_cache_write_indices (used to avoid recompilation)
        padded_num_slices = get_padded_num_kv_cache_update_slices(
            padded_total_tokens, batch_size, self.input_args.block_size)
        kv_cache_write_indices = np.pad(
            kv_cache_write_indices,
            [[0, padded_num_slices - len(kv_cache_write_indices)], [0, 0]],
            constant_values=0)
        kv_cache_write_indices = np.transpose(kv_cache_write_indices)
        num_prefill_seqs = np.array([kv_cache_write_indices.shape[0]])
        prefill_query_start_offsets = np.zeros(batch_size + 1, dtype=np.int32)
        prefill_query_start_offsets[0] = 0
        np.cumsum(seq_lens, out=prefill_query_start_offsets[1:batch_size + 1])
        prefill_query_start_offsets[batch_size + 1:] = 1  # no-op?
        sharding = NamedSharding(
            self.mesh,
            PartitionSpec())  ## TODO: Should this actually be using DP??
        request_distribution = np.array([1, 1, 1], dtype=np.int32)
        (input_ids, input_positions, kv_cache_write_indices, num_prefill_seqs,
         self.block_table, prefill_query_start_offsets, seq_lens,
         num_decode_seqs, request_distribution) = device_array(
             (input_ids, input_positions, kv_cache_write_indices,
              num_prefill_seqs, self.block_table, prefill_query_start_offsets,
              seq_lens, num_decode_seqs, request_distribution),
             mesh=self.mesh,
             sharding=sharding)
        breakpoint()
        ########### TODO: May need to add sampling component (which would incur extra latency)
        return ModelInputs(
            is_prefill=True,
            do_sampling=False,
            kv_caches=self.kv_caches,
            input_ids=input_ids,
            attention_metadata=AttentionMetadata(
                input_positions=input_positions,
                block_tables=self.block_table.reshape(-1),
                seq_lens=seq_lens,
                query_start_loc=prefill_query_start_offsets,
                request_distribution=request_distribution,
                ),
            temperatures=None,
            top_ps=None,
            top_ks=None)
class Benchmarker:
    def __init__(self, vllm_config: VllmConfig, model: Any,
                 mesh: jax.sharding.Mesh, sampler: Sampler, rng: nnx.Rngs, model_hf_config, state):
        self.vllm_config = vllm_config
        self.model = model
        self.mesh = mesh
        self.sampler = sampler
        self.rng = rng
        self.model_hf_config = model_hf_config
        self.state = state
    def benchmark(self, phase: str):
        if phase == "prefill":
            # TODO: Should you update the sharding to use DP?
            additional_config = self.vllm_config.additional_config
            input_args = InputArgs(
                block_size=_BLOCK_SIZE.value,
                batch_size=_PREFILL_BATCH_SIZE.value,
                max_prefill_len=_MAX_PREFILL_LEN.value,
                max_seq_len=_MAX_SEQ_LEN.value,
                sampler=self.sampler,
                num_kv_heads=self.model_hf_config.num_key_value_heads,
                head_dim=self.model_hf_config.head_dim,
                num_layers=self.model_hf_config.num_hidden_layers,
                vocab_size=self.model_hf_config.vocab_size,)
            
            input_creator = InputCreator(input_args=input_args,
                                         sharding=None,
                                         mesh=self.mesh,
                                         rng=self.rng)
            model_input = input_creator.create_prefill_input()
            # TODO: add tracing
            inputs = (
                model_input.kv_caches,
                model_input.input_ids,
                model_input.attention_metadata,
            )
            jax.profiler.start_trace("/tmp/profile-data")
            kv_caches, act = self.model(self.state, model_input.kv_caches, model_input.input_ids, model_input.attention_metadata,)
            # kv_caches, act = self.model(self.state, *inputs[:3])
            act.block_until_ready()
            jax.profiler.stop_trace()
def main(argv: Sequence[str]):
    sampler = Sampler(type=_SAMPLER_TYPE.value, std=_SAMPLER_STD.value)
    rng = nnx.Rngs(params=0)
    vllm_config = VllmConfig(
      additional_config=json.loads(_ADDITIONAL_CONFIG.value),
      model_config=ModelConfig(**json.loads(_MODEL_CONFIG.value)),
    )
    
    # Deepseek v3 HF Config
    # vllm_config.model_config.hf_config.attribute_map = {
    #             "architectures": [
    #                 "DeepseekV3ForCausalLM"
    #             ],
    #             "attention_bias": False,
    #             "attention_dropout": 0.0,
    #             "auto_map": {
    #                 "AutoConfig": "configuration_deepseek.DeepseekV3Config",
    #                 "AutoModel": "modeling_deepseek.DeepseekV3Model",
    #                 "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM"
    #             },
    #             "bos_token_id": 0,
    #             "eos_token_id": 1,
    #             "ep_size": 1,
    #             "first_k_dense_replace": 3,
    #             "hidden_act": "silu",
    #             "hidden_size": 7168,
    #             "initializer_range": 0.02,
    #             "intermediate_size": 18432,
    #             "kv_lora_rank": 512,
    #             "max_position_embeddings": 163840,
    #             "model_type": "deepseek_v3",
    #             "moe_intermediate_size": 2048,
    #             "moe_layer_freq": 1,
    #             "n_group": 8,
    #             "n_routed_experts": 256,
    #             "n_shared_experts": 1,
    #             "norm_topk_prob": True,
    #             "num_attention_heads": 128,
    #             "num_experts_per_tok": 8,
    #             "num_hidden_layers": 61,
    #             "num_key_value_heads": 128,
    #             "num_nextn_predict_layers": 1,
    #             "q_lora_rank": 1536,
    #             "qk_nope_head_dim": 128,
    #             "qk_rope_head_dim": 64,
    #             "quantization_config": {
    #                 "activation_scheme": "dynamic",
    #                 "fmt": "e4m3",
    #                 "quant_method": "fp8",
    #                 "weight_block_size": [
    #                 128,
    #                 128
    #                 ]
    #             },
    #             "rms_norm_eps": 1e-06,
    #             "rope_scaling": {
    #                 "beta_fast": 32,
    #                 "beta_slow": 1,
    #                 "factor": 40,
    #                 "mscale": 1.0,
    #                 "mscale_all_dim": 1.0,
    #                 "original_max_position_embeddings": 4096,
    #                 "type": "yarn"
    #             },
    #             "rope_theta": 10000,
    #             "routed_scaling_factor": 2.5,
    #             "scoring_func": "sigmoid",
    #             "tie_word_embeddings": False,
    #             "topk_group": 4,
    #             "topk_method": "noaux_tc",
    #             "torch_dtype": "bfloat16",
    #             "transformers_version": "4.33.1",
    #             "use_cache": True,
    #             "v_head_dim": 128,
    #             "vocab_size": 129280
    #             }

    
    
    # Qwen3 32B config
    vllm_config.model_config.hf_config.attribute_map = {
                        "architectures": [
                            "Qwen3ForCausalLM"
                        ],
                        "attention_bias": False,
                        "attention_dropout": 0.0,
                        "bos_token_id": 151643,
                        "eos_token_id": 151645,
                        "head_dim": 128,
                        "hidden_act": "silu",
                        "hidden_size": 5120,
                        "initializer_range": 0.02,
                        "intermediate_size": 25600,
                        "max_position_embeddings": 40960,
                        "max_window_layers": 64,
                        "model_type": "qwen3",
                        "num_attention_heads": 64,
                        "num_hidden_layers": 64,
                        "num_key_value_heads": 8,
                        "rms_norm_eps": 1e-06,
                        "rope_scaling": None,
                        "rope_theta": 1000000,
                        "sliding_window": None,
                        "tie_word_embeddings": False,
                        "torch_dtype": "bfloat16",
                        "transformers_version": "4.51.0",
                        "use_cache": True,
                        "use_sliding_window": False,
                        "vocab_size": 151936
                        }
    
    mesh = init_mesh(vllm_config, jax.devices())
    model_fn, compute_logits_fn, get_multimodal_embeddings_fn, get_input_embeddings_fn, state, model_hf_config = get_model(
            vllm_config,
            rng.params(),
            mesh,
        )
    benchmarker = Benchmarker(vllm_config, model_fn, mesh, sampler, rng, model_hf_config, state)
    for _ in range(_PREFILL_STEPS.value):
        benchmarker.benchmark("prefill")
    # TODO:
    for _ in range(_DECODE_STEPS.value):
        pass
if __name__ == "__main__":
    # os.environ['NEW_MODEL_DESIGN'] = 'True'
    os.environ['JAX_RANDOM_WEIGHTS'] = 'True'
    os.environ['TPU_BACKEND_TYPE'] = 'JAX'
    app.run(main)