from dataclasses import dataclass
from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from microbenchmark_utils import Sampler

from tpu_commons.kernels.ragged_paged_attention.v3.util import cdiv
from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.runner.kv_cache import create_kv_caches
from tpu_commons.utils import device_array


@dataclass
class InputArgs:
    max_num_seq: int
    max_prefill_len: int
    max_model_len: int
    sampler: Sampler
    model_hf_config: Any
    min_prefill_len: int
    max_prefill_len: int
    decode_offset_from_prefill: int
    phase: str
    block_size: int
    num_blocks_override: int


@dataclass
class ModelInputs:
    """
    Dataclass to hold model inputs. This should be extended as needed and should have everything needed to call model function"""
    kv_caches: List[jax.Array]
    input_ids: jax.Array
    attention_metadata: AttentionMetadata


class InputCreator:

    def __init__(self,
                 input_args: InputArgs,
                 sharding: NamedSharding | None,
                 rng: nnx.Rngs,
                 mesh: jax.sharding.Mesh,
                 permute_block_table: bool = False,
                 kv_cache_dtype: jnp.dtype = jnp.bfloat16,
                 logger=None):
        self.rng = rng
        self.input_args = input_args
        self.sharding = sharding
        self.mesh = mesh
        self.permute_block_table = permute_block_table
        self.kv_cache_dtype = kv_cache_dtype
        self.logger = logger if logger is not None else init_logger(__name__)

        self.setup()

    def setup(self):
        self.max_blocks_per_req = cdiv(self.input_args.max_model_len,
                                       self.input_args.block_size)
        padded_max_seq_len = self.max_blocks_per_req * self.input_args.block_size
        if padded_max_seq_len != self.input_args.max_model_len:
            self.logger.warning(
                f"Padding max_seq_len from {self.input_args.max_model_len} to {padded_max_seq_len}."
            )
        self.num_blocks = self.input_args.num_blocks_override
        self.block_table = self._create_mock_block_table(
            self.permute_block_table)
        self.kv_caches = create_kv_caches(
            self.num_blocks, self.input_args.block_size,
            self.input_args.model_hf_config.num_key_value_heads,
            self.input_args.model_hf_config.head_dim, self.mesh, [
                "layer_%d" % i for i in range(
                    self.input_args.model_hf_config.num_hidden_layers)
            ])

    def _create_mock_block_table(self, random_permute: bool = False):
        block_table = np.arange(self.input_args.num_blocks_override,
                          dtype=jnp.int32)\
                             .reshape(self.input_args.max_num_seq, -1)
        if random_permute:
            block_table = np.random.permutation(block_table)
        return block_table

    def create_input(self, phase) -> ModelInputs:
        """
        Creates model input for the given phase.
        phase: 'decode' or 'prefill'
        Returns: ModelInputs dataclass 
        which has kv_caches, input_ids, attention_metadata
        """
        seq_lens = create_sequence_lengths(
            self.input_args.max_num_seq,
            self.input_args.min_prefill_len,
            self.input_args.max_prefill_len,
            self.input_args.max_model_len,
            self.input_args.decode_offset_from_prefill,
            phase=phase)

        input_ids = create_input_ids(
            self.input_args.max_num_seq,
            self.input_args.model_hf_config.vocab_size,
            self.input_args.max_prefill_len,
            phase=phase)

        input_positions = create_input_positions(
            self.input_args.max_model_len,
            self.input_args.max_prefill_len,
            self.input_args.max_num_seq,
            self.input_args.decode_offset_from_prefill,
            phase=phase)

        query_start_locations = create_query_start_locations(
            self.input_args.max_num_seq, seq_lens, phase)

        request_distribution = create_request_distribution(
            self.input_args.max_num_seq, phase=phase)

        self.block_table = self.block_table.reshape(-1)
        (input_ids, input_positions, self.block_table, query_start_locations,
         seq_lens, request_distribution) = device_array(
             self.mesh,
             (input_ids, input_positions, self.block_table,
              query_start_locations, seq_lens, request_distribution))

        ########### TODO: May need to add sampling component (which would incur extra latency)
        return ModelInputs(kv_caches=self.kv_caches,
                           input_ids=input_ids,
                           attention_metadata=AttentionMetadata(
                               input_positions=input_positions,
                               block_tables=self.block_table,
                               seq_lens=seq_lens,
                               query_start_loc=query_start_locations,
                               request_distribution=request_distribution,
                           ))


# shape is 3, (i,j,k)
def create_request_distribution(max_num_sequence: int,
                                num_sequence_prefill=None,
                                num_sequence_decode=None,
                                phase='decode') -> jnp.ndarray:
    """
    Creates request distribution based on phase.
    request_distribution is of shape (3,) where
    request_distribution[0] = number of sequences in decode phase
    request_distribution[1] = number of sequences in prefill phase
    request_distribution[2] = total number of sequences"""

    if phase == 'decode':
        # this means all sequences are in decode phase
        return np.array([max_num_sequence, max_num_sequence, max_num_sequence],
                        dtype=np.int32)
    elif phase == 'prefill':
        # this means all sequences are in prefill phase
        return np.array([0, 0, max_num_sequence], dtype=np.int32)
    else:
        # TODO: implement mixed phase
        raise NotImplementedError(f"Phase {phase} not implemented")


# padded input positions power of 2 >> Same as input positions shape
def create_input_ids(max_num_sequence: int,
                     vocab_size: int,
                     max_prefill_len: int,
                     phase='decode') -> jnp.ndarray:
    
    """
    Creates input ids based on phase.
    In decode phase, all sequences are of length 1
    In prefill phase, all sequences are of length max_prefill_len"""
    if phase == 'decode':
        # this means all sequences are in decode phase
        return np.random.randint(1, vocab_size, size=max_num_sequence)
    elif phase == 'prefill':
        # this means all sequences are in prefill phase of length max_prefill_len
        return np.random.randint(1,
                                 vocab_size,
                                 size=(max_num_sequence * max_prefill_len))
    else:
        raise NotImplementedError(f"Phase {phase} not implemented")


# shape is max_num_seqs
def create_sequence_lengths(max_num_seqs: int,
                            min_prefill_len: int,
                            max_prefill_len: int,
                            max_model_len: int,
                            offset_from_prefill: int = 1,
                            phase='decode') -> jnp.ndarray:
    """
    Creates sequence lengths based on phase.
    In decode phase, all sequences are of length offset_from_prefill (usually 1) + max_prefill_len -1
    In prefill phase, all sequences are of length max_prefill_len."""

    if phase == 'decode':
        return np.full((max_num_seqs, ),
                       max_prefill_len + offset_from_prefill - 1,
                       dtype=np.int32)
    elif phase == 'prefill':
        # this means all sequences are in prefill phase with same prefill length which is max_prefill_len
        return np.random.randint(max_prefill_len,
                                 max_prefill_len + 1,
                                 size=max_num_seqs)
    else:
        raise NotImplementedError(f"Phase {phase} not implemented")


# padded input positions power of 2
def create_input_positions(max_model_len: int,
                           max_prefill_len: int,
                           max_num_seqs: int,
                           offset_from_prefill: int = 1,
                           phase='decode') -> jnp.ndarray:
    """
    Creates input positions based on phase.
    In decode phase, all sequences are of length 1 and the position is immediately after the prefill length
    In prefill phase, all sequences are of length max_prefill_len and positions are [0, 1, 2, ..., max_prefill_len-1] repeated for max_num_seqs
    """
    if phase == 'decode':
        # in decode phase, all sequences are of length 1 and the position is immediately after the prefill length
        return np.full((max_num_seqs, ),
                       max_prefill_len + offset_from_prefill - 1,
                       dtype=np.int32)
    elif phase == 'prefill':
        # in prefill phase, all sequences are of length max_prefill_len and positions are [0, 1, 2, ..., max_prefill_len-1] repeated for max_num_seqs
        single_prefill = jnp.arange(0, max_prefill_len, dtype=jnp.int32)
        return np.repeat(single_prefill, max_num_seqs)
    else:
        raise NotImplementedError(f"Phase {phase} not implemented")


# shape is 1 + max_num_seqs
def create_query_start_locations(max_num_seq, seq_lens, phase) -> jnp.ndarray:
    """
    Creates query start locations based on phase.
    In prefill phase, it is cumulative sum of sequence lengths.
    which means if 0th seq is len 3, 1st seq is len 5, 2nd seq is len 2
    then query start locations is [0, 3, 8, 10]

    In decode phase, it is simply [0, 1, 2, ..., max_num_seq]
    which means each sequence is of length 1.
    """
    if phase == 'prefill':
        query_start_offsets = np.zeros(max_num_seq + 1, dtype=np.int32)
        query_start_offsets[0] = 0
        np.cumsum(seq_lens, out=query_start_offsets[1:max_num_seq + 1])
        return query_start_offsets
    elif phase == 'decode':
        # in decode phase, all sequences are of length 1
        return np.arange(0, max_num_seq + 1, dtype=np.int32)
    else:
        raise NotImplementedError(f"Phase {phase} not implemented")


def create_num_blocks(max_model_len: int,
                      block_size: int,
                      num_block_override=0) -> int:
    """
    Ideally creates number of blocks based on max_model_len and block_size.
    We dont support dynamic number of blocks yet, so we just return num_block_override

    """
    if num_block_override > 0:
        return num_block_override
    else:
        raise NotImplementedError(
            f"num_block_override must be > 0, got {num_block_override}")
