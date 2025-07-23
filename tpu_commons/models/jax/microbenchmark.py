import bisect
import json
import logging
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np

from tpu_commons.kernels.ragged_paged_attention.kernel import cdiv
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.common.sharding import Sharding
from tpu_commons.models.jax.recipes.llama3 import LlamaForCausalLM
from tpu_commons.runner.jax.tpu_jax_runner import \
    NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK

vllm_logger = logging.getLogger("vllm")
original_level = vllm_logger.level
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Set the vLLM logger to ERROR to suppress its messages
    vllm_logger.setLevel(logging.ERROR)

    # Import the class; all warnings will be suppressed
    from vllm.config import ModelConfig

vllm_logger.setLevel(logging.WARNING)

logger = init_logger(__name__)

power_of_two = np.pow(2, np.arange(18))  # up to 128k seq lens


@dataclass
class VllmConfig():
    additional_config: Mapping[str, Any] = field(default_factory=dict)
    # Set default max_model_len to turn off warnings.
    model_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(max_model_len=1024))


@dataclass
class Sampler:
    type: str
    std: float = None

    def generate_samples(self, shape: Tuple[int], fill_val: Any) -> np.array:
        if self.type.lower() == "fixed":
            return np.full(shape, fill_val)
        elif self.type.lower() == "normal":
            return np.random.normal(loc=0.0, scale=self.std, size=shape)


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


def _nearest_power_of_two(val: int) -> int:
    index = bisect.bisect_left(power_of_two, val)
    assert index < len(power_of_two)
    return power_of_two[index]


def _get_padded_num_kv_cache_update_slices(num_tokens: int, max_num_reqs: int,
                                           page_size: int) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    padded_num_slices = (
        padded_num_slices + NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK - 1
    ) // NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK * \
        NUM_SLICES_PER_KV_CACHE_UPDATE_BLOCK
    return padded_num_slices


def _get_config_arg_or_default(config, query_keys: List[str] | str,
                               default: Any):
    try:
        if isinstance(query_keys, str):
            return config[query_keys]
        elif isinstance(query_keys, list):
            results = config[query_keys[0]]
            for key in query_keys[1:]:
                results = results[key]
            return results
    except KeyError:
        return default


def _init_mesh(vllm_config: VllmConfig):
    devices = jax.devices()
    try:
        # TODO: Update override steps.
        sharding_strategy = \
           vllm_config.additional_config["sharding"]["sharding_strategy"]
    except KeyError:
        logger.warning(
            f"No sharding strategy passed! Using default of full model parallelism={len(devices)}"
        )
        sharding_strategy = {"tensor_parallelism": len(devices)}
    sharding = Sharding(strategy_dict=sharding_strategy,
                        vllm_config=vllm_config)
    return sharding.mesh


def device_array(*args, mesh, sharding=None, **kwargs) -> jax.Array:
    if sharding is None:
        sharding = NamedSharding(mesh, PartitionSpec(None))
    return jax.device_put(*args, device=sharding, **kwargs)


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
        self.kv_caches = self._init_sharded_kv_cache()

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
        # NOTE(gpolovets) seq_lens is padded shape (max_seq_len) in tpu_jax_runner: https://github.com/vllm-project/tpu_commons/blob/38df9a7cbdc490bae5a8f63938b518e0e636d829/tpu_commons/runner/jax/tpu_jax_runner.py#L713
        # But I don't think this should affect things much.
        batch_size = self.input_args.batch_size
        seq_lens = self.input_args.sampler.generate_samples(
            shape=(batch_size, ), fill_val=self.input_args.max_prefill_len)
        if not self.input_args.sampler:
            seq_lens = batch_size * [self.input_args.max_prefill_len]
        total_tokens = sum(seq_lens)
        padded_total_tokens = _nearest_power_of_two(total_tokens)

        ## Generate random input_ids
        input_ids = np.random.randint(
            low=0,
            high=self.input_args.vocab_size - 1,
            size=(padded_total_tokens, ),  # Pad to nearest power of two
            dtype=np.int32)
        num_decode_seqs = batch_size
        chunked_prefill_enabled = False
        padded_input_positions = np.zeros(padded_total_tokens, dtype=np.int32)

        # Calculate token positions within each sequence
        input_positions = np.concatenate(
            [np.arange(seq_len, dtype=np.int32) for seq_len in seq_lens],
            out=padded_input_positions)  # Pad to nearest power of 2
        breakpoint()
        kv_cache_write_indices = self._mock_kv_write_indices(
            seq_lens, ["prefill"] * len(seq_lens))
        # Padd the kv_cache_write_indices (used to avoid recompilation)
        padded_num_slices = _get_padded_num_kv_cache_update_slices(
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

        (input_ids, input_positions, kv_cache_write_indices, num_prefill_seqs,
         self.block_table, prefill_query_start_offsets, seq_lens,
         num_decode_seqs) = device_array(
             (input_ids, input_positions, kv_cache_write_indices,
              num_prefill_seqs, self.block_table, prefill_query_start_offsets,
              seq_lens, num_decode_seqs),
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
                slot_mapping=kv_cache_write_indices,
                block_tables=self.block_table,
                #seq_lens=np.ones((seq_lens, ), dtype=np.int32),
                seq_lens=seq_lens,
                query_start_loc=prefill_query_start_offsets,
                num_seqs=np.array(num_prefill_seqs, dtype=jnp.int32),
                num_slices=np.array([1], dtype=np.int32)),
                # kv_cache_write_indices=kv_cache_write_indices,
                # num_prefill_seqs=num_prefill_seqs,
                # prefill_query_start_offsets=prefill_query_start_offsets,
                # num_decode_seqs=num_decode_seqs,
                # chunked_prefill_enabled=chunked_prefill_enabled),
            temperatures=None,
            top_ps=None,
            top_ks=None)

    # TODO:
    def decode_input(self,
                     previous_input: AttentionMetadata) -> AttentionMetadata:
        if not self.input_args.sampler:
            seq_lens = self.input_args.batch_size * [
                self.input_args.max_prefill_len
            ]
        else:
            raise NotImplementedError
        total_tokens = sum(seq_lens)
        input_ids = jax.random.randint(self.rng.params(),
                                       shape=(self.input_args.batch_size),
                                       minval=0,
                                       maxval=self.input_args.vocab_size - 1)
        num_prefill_seqs = 0
        num_decode_seqs = len(self.input_args.batch_size)
        input_positions = jnp.concatenate(
            [jnp.array([seq_len], dtype=jnp.int32) for seq_len in seq_lens])


class Benchmarker:

    def __init__(self, vllm_config: VllmConfig, model: Any,
                 mesh: jax.sharding.Mesh, sampler: Sampler, rng: nnx.Rngs):
        self.vllm_config = vllm_config
        self.model = model
        self.mesh = mesh
        self.sampler = sampler
        self.rng = rng

    def benchmark(self, phase: str):
        if phase == "prefill":
            # TODO: Should you update the sharding to use DP?
            additional_config = self.vllm_config.additional_config
            input_args = InputArgs(
                block_size=additional_config["block_size"],
                batch_size=additional_config["prefill_batch_size"],
                max_prefill_len=additional_config["max_prefill_len"],
                max_seq_len=additional_config["max_seq_len"],
                sampler=self.sampler,
                num_kv_heads=self.model.cfg.model.layers.attention.
                num_key_value_heads,
                head_dim=self.model.cfg.model.layers.attention.head_dim,
                num_layers=self.model.cfg.model.num_layers,
                vocab_size=self.model.cfg.model.emb.vocab_size)
            input_creator = InputCreator(input_args=input_args,
                                         sharding=None,
                                         mesh=self.mesh,
                                         rng=self.rng)
            model_input = input_creator.create_prefill_input()
            # TODO: add tracing
            self.model(kv_caches=model_input.kv_caches, 
                       input_ids=model_input.input_ids, 
                       attention_metadata=model_input.attention_metadata)


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--max_seq_len", type=int, default=2048)
    argparser.add_argument("--max_prefill_len", type=int, default=1024)
    argparser.add_argument("--prefill_batch_size", type=int, default=1)
    argparser.add_argument("--prefill_steps", type=int, default=5)
    argparser.add_argument("--decode_batch_size", type=int, default=1)
    argparser.add_argument("--decode_steps", type=int, default=5)
    argparser.add_argument("--block_size", type=int, default=16)
    argparser.add_argument("--sampler_type", type=str, default="fixed")
    argparser.add_argument("--sampler_std", type=float, default=1.0)
    argparser.add_argument("--additional_config", type=json.loads, default={})
    argparser.add_argument("--model_config", type=json.loads, default={})
    args = argparser.parse_args()
    sampler = Sampler(type=args.sampler_type, std=args.sampler_std)
    rng = nnx.Rngs(params=0)
    arg_dict = {
        key: val
        for (key, val) in vars(args).items() if key != "additional_config"
    }
    vllm_config = VllmConfig(additional_config=args.additional_config, model_config=ModelConfig(**args.model_config))
    vllm_config.additional_config.update(
        arg_dict)  # add all of the cmd-line args to additional_config
    mesh = _init_mesh(vllm_config)
    model = LlamaForCausalLM(vllm_config, rng.params(), mesh)
    model.load_weights(jax.random.PRNGKey(42))# Load the model weights
    benchmarker = Benchmarker(vllm_config, model, mesh, sampler, rng)

    for _ in range(args.prefill_steps):
        benchmarker.benchmark("prefill")

    # TODO:
    for _ in range(args.decode_steps):
        pass


if __name__ == "__main__":
    #   app.run(main)
    main()
