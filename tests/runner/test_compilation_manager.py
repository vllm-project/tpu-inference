# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from tpu_inference.layers.common.attention_metadata import \
    SharedAttentionMetadata
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)
from tpu_inference.layers.jax.sample.sampling import \
    logprobs_use_processed_logits
from tpu_inference.runner.compilation_manager import (CompilationManager,
                                                      _describe_signature)


class TestDescribeSignature:
    """`Precompile ...` lines name one padding combination each.

    `_run_compilation` uses its `**kwargs` only to build that line (the
    lowering goes through `*args` / `call_kwargs`), so the line should carry
    the scalars that identify the variant and nothing else.
    """

    def test_keeps_the_scalars_that_name_the_variant(self):
        assert _describe_signature({
            "num_tokens": 64,
            "num_reqs": 64,
        }) == {
            "num_tokens": 64,
            "num_reqs": 64,
        }

    def test_keeps_every_scalar_kind(self):
        kwargs = {
            "num_tokens": 64,
            "ratio": 0.5,
            "do_sampling": True,
            "kind": "decode",
            "spec_step_idx": None,
        }
        assert _describe_signature(kwargs) == kwargs

    def test_drops_array_payloads(self):
        # `_precompile_backbone` passes the whole metadata object; its repr
        # prints every element of every array, once per padding combination.
        num_reqs = 64
        ones = jnp.ones((num_reqs, ), dtype=jnp.int32)
        kwargs = {
            "num_tokens":
            num_reqs,
            "num_reqs":
            num_reqs,
            "shared_attention_metadata":
            SharedAttentionMetadata(
                input_positions=jnp.ones((2, num_reqs), dtype=jnp.int32),
                seq_lens=ones,
                query_start_loc=ones,
                request_distribution=jnp.zeros((3, ), dtype=jnp.int32),
                mamba_state_indices=None,
                padded_num_reqs=num_reqs,
            ),
        }

        described = _describe_signature(kwargs)

        assert described == {"num_tokens": num_reqs, "num_reqs": num_reqs}
        # The point of the change: the line stays one line.
        assert "\n" not in f"{described}"
        assert len(f"{described}") < len(f"{kwargs}") / 10

    def test_drops_bare_arrays(self):
        kwargs = {"num_tokens": 8, "positions": jnp.arange(64)}
        assert _describe_signature(kwargs) == {"num_tokens": 8}


def _precompiled_logits_specs(logprobs_mode):
    """PartitionSpecs `_precompile_gather_logprobs` warms up for a mode."""
    # Only the PartitionSpec that gets selected matters here, not the mesh
    # extent, so size the mesh off one device to stay portable across lanes.
    # Name every axis: ShardingAxisName resolves to ShardingAxisNameBase under
    # NEW_MODEL_DESIGN/USE_2D_TP, whose specs reference pcp/dcp/expert/... and
    # NamedSharding rejects a spec naming an axis the mesh does not have.
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    runner = SimpleNamespace(
        mesh=mesh,
        rank=0,
        vocab_size=256,
        num_reqs_paddings=[8, 16],
        num_tokens_paddings=[],
        speculative_config=None,
        model_config=SimpleNamespace(max_logprobs=5,
                                     logprobs_mode=logprobs_mode),
    )
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = runner

    specs = []

    def record(name, fn, logits, token_ids, *args, **kwargs):
        if name.endswith("gather_logprobs"):
            specs.append(logits.sharding.spec)

    manager._run_compilation = record
    CompilationManager._precompile_gather_logprobs(manager)
    return specs


class TestPrecompileGatherLogprobsSharding:
    """Precompiled logits sharding has to match what the runner passes in.

    `compute_and_gather_logprobs` is a `jax.jit`, so a mismatched input
    sharding is a silent cache miss: serving pays a full compile of the
    vocab-wide log_softmax/gather for every `num_reqs` bucket.
    """

    @pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "raw_logits"])
    def test_raw_modes_use_the_compute_logits_sharding(self, logprobs_mode):
        # Raw modes hand compute_logits' output straight to the logprobs jit.
        expected = PartitionSpec(ShardingAxisName.MLP_DATA,
                                 ShardingAxisName.MLP_TENSOR)
        assert _precompiled_logits_specs(logprobs_mode) == [expected] * 2

    @pytest.mark.parametrize("logprobs_mode",
                             ["processed_logprobs", "processed_logits"])
    def test_processed_modes_use_the_sample_output_sharding(
            self, logprobs_mode):
        # Processed modes hand sample()'s output in, which sample_full_vocab
        # constrains to P(ATTN_DATA, None).
        expected = PartitionSpec(ShardingAxisName.ATTN_DATA, None)
        assert _precompiled_logits_specs(logprobs_mode) == [expected] * 2

    @pytest.mark.parametrize("logprobs_mode", [
        "raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"
    ])
    def test_matches_the_branch_the_runner_takes(self, logprobs_mode):
        # Both sides must read the mode the same way, or the warmup misses.
        uses_processed = logprobs_use_processed_logits(logprobs_mode)
        spec = _precompiled_logits_specs(logprobs_mode)[0]
        assert (spec == PartitionSpec(ShardingAxisName.ATTN_DATA,
                                      None)) is uses_processed


def _decode_only_shapes(*,
                        dp_size=1,
                        token_paddings_per_dp,
                        attn_req_paddings_per_dp,
                        max_decode_tokens=1):
    runner = SimpleNamespace(
        dp_size=dp_size,
        num_tokens_paddings_per_dp=token_paddings_per_dp,
        attn_num_reqs_paddings_per_dp=attn_req_paddings_per_dp,
        input_batch=SimpleNamespace(max_decode_tokens=max_decode_tokens),
    )
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = runner
    return CompilationManager._decode_only_backbone_shapes(manager)


class TestDecodeOnlyBackboneShapes:
    """The DCP decode-only backbone (`is_decode=True`) is a separate compiled
    variant per (num_tokens, num_reqs) padding pair, so the warm-up has to
    enumerate exactly the pairs `_prepare_inputs` can produce.
    """

    TOKENS = [16, 32, 64, 128, 256, 512, 1024, 2048]

    def test_smallest_request_bucket_pairs_with_smallest_token_bucket(self):
        # Regression: with ATTN_CUSTOM_NUM_REQS_BUCKETS=4 and max_num_seqs=16
        # a decode-only step with <= 4 live requests pads to 16 tokens x 4
        # reqs. The old `num_tokens == num_reqs` rule only warmed up (16, 16),
        # and the first such step paid a ~30 s JIT compile mid-benchmark.
        shapes = _decode_only_shapes(token_paddings_per_dp=self.TOKENS,
                                     attn_req_paddings_per_dp=[4, 16])
        assert shapes == {(16, 4), (16, 16)}

    def test_matches_runtime_padding_for_power_of_two_buckets(self):
        shapes = _decode_only_shapes(token_paddings_per_dp=self.TOKENS,
                                     attn_req_paddings_per_dp=[8, 16, 32, 64])
        # 1..8 reqs -> 16 tokens; 9..16 -> 16; 17..32 -> 32; 33..64 -> 64.
        assert shapes == {(16, 8), (16, 16), (32, 32), (64, 64)}

    def test_scales_by_dp_size(self):
        shapes = _decode_only_shapes(dp_size=4,
                                     token_paddings_per_dp=self.TOKENS,
                                     attn_req_paddings_per_dp=[16, 64])
        # Per rank: 1..16 reqs -> 16 tokens; 17..32 -> 32; 33..64 -> 64.
        assert shapes == {(64, 64), (128, 256), (256, 256)}

    def test_spec_decode_covers_up_to_max_decode_tokens_per_request(self):
        # 16 reqs x 3 tokens = 48 -> buckets 16, 32 and 64 are all reachable.
        shapes = _decode_only_shapes(token_paddings_per_dp=self.TOKENS,
                                     attn_req_paddings_per_dp=[16],
                                     max_decode_tokens=3)
        assert shapes == {(16, 16), (32, 16), (64, 16)}

    def test_never_exceeds_the_step_token_budget(self):
        # max_num_batched_tokens=32: a decode-only step can never hold more
        # than 32 live requests, so the 64-request bucket is unreachable and
        # nothing pads past the largest token bucket.
        shapes = _decode_only_shapes(token_paddings_per_dp=[16, 32],
                                     attn_req_paddings_per_dp=[16, 64],
                                     max_decode_tokens=4)
        assert shapes == {(16, 16), (32, 16), (32, 64)}
