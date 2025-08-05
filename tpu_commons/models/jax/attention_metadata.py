import functools
from dataclasses import dataclass

import jax


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "slot_mapping",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "num_seqs",
        "num_slices",
        "request_distribution",
    ],
    meta_fields=[],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (3, padded_num_slices)
    slot_mapping: jax.Array
    # (max_num_seqs, max_num_blocks_per_req)
    block_tables: jax.Array
    # (max_num_seqs,)
    seq_lens: jax.Array
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (1,)
    num_seqs: jax.Array = None
    # (1,)
    num_slices: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (padded_total_num_scheduled_tokens,)
    input_mrope_positions: jax.Array = None
