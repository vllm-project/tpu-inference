import functools
from dataclasses import dataclass, field
from typing import Any

import jax


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "k_scale_cache",
        "v_scale_cache",
    ],
    meta_fields=[],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None

    k_scale_cache: jax.Array = None
    v_scale_cache: jax.Array = None

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)
