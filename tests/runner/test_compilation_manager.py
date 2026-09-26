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
from tpu_inference.runner.utils import ForbidCompile
from tpu_inference.utils import DeviceBuffer, DeviceBufferMetadata


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


def _two_segment_layout(num_tokens, num_logits):
    return DeviceBufferMetadata(keys=("input_ids", "logits_indices"),
                                sizes=(num_tokens, num_logits))


def _precompiled_unpacks(**runner_overrides):
    """The `_run_compilation` calls `_precompile_unpack_arrays` makes."""
    # One device, every axis named: see _precompiled_logits_specs.
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    runner = SimpleNamespace(
        mesh=Mesh(devices, MESH_AXIS_NAMES),
        dp_size=1,
        num_tokens_paddings_per_dp=[8, 16, 32],
        num_reqs_paddings_per_dp=[8, 16],
        num_logits_paddings=None,
        enable_continue_decode=False,
        speculative_config=None,
        _metadata_blob_layout=_two_segment_layout,
    )
    vars(runner).update(runner_overrides)
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = runner

    calls = []

    def record(name, fn, blob, layout, **kwargs):
        calls.append(
            SimpleNamespace(fn=fn,
                            blob=blob,
                            layout=layout,
                            sizes=(kwargs["num_tokens"],
                                   kwargs["num_logits"])))

    manager._run_compilation = record
    CompilationManager._precompile_unpack_arrays(manager)
    return calls


class TestPrecompileUnpackArrays:
    """`_prepare_inputs` splits its metadata blob with an un-jitted
    `jnp.split`, which compiles once per layout. Every layout a step can pack
    has to be split here, or the first step to use it compiles (or reads the
    persistent cache) mid-request, and raises under VLLM_XLA_CHECK_RECOMPILATION.
    """

    def test_every_token_and_logits_padding_pair(self):
        # (8, 16) is skipped: a step never has more logits than tokens.
        sizes = [c.sizes for c in _precompiled_unpacks()]
        assert sizes == [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16)]

    def test_per_rank_paddings_are_scaled_by_dp(self):
        # _prepare_inputs pads per rank and packs all ranks into one blob.
        sizes = [c.sizes for c in _precompiled_unpacks(dp_size=2)]
        assert sizes == [(16, 16), (32, 16), (32, 32), (64, 16), (64, 32)]

    def test_continue_decode_pads_tokens_to_request_paddings(self):
        sizes = [
            c.sizes
            for c in _precompiled_unpacks(num_tokens_paddings_per_dp=[16, 32],
                                          enable_continue_decode=True)
        ]
        assert (8, 8) in sizes
        assert (8, 8) not in [
            c.sizes
            for c in _precompiled_unpacks(num_tokens_paddings_per_dp=[16, 32])
        ]

    def test_spec_decode_pads_logits_to_logits_paddings(self):
        sizes = [
            c.sizes
            for c in _precompiled_unpacks(speculative_config=object(),
                                          num_logits_paddings=[8, 16, 32])
        ]
        assert sizes == [(8, 8), (16, 8), (16, 16), (32, 8), (32, 16),
                         (32, 32)]

    def test_blob_is_the_one_prepare_inputs_transfers(self):
        # The eager split's executable is keyed on the input's shape, dtype
        # and sharding, so all three must match the runtime blob.
        for call in _precompiled_unpacks():
            assert call.fn is DeviceBuffer.unpack_arrays
            assert call.layout == _two_segment_layout(*call.sizes)
            assert call.blob.shape == (sum(call.layout.sizes), )
            assert call.blob.dtype == jnp.int32
            assert call.blob.sharding.spec == PartitionSpec(
                ShardingAxisName.BATCH)

    def test_a_warmed_layout_does_not_compile_when_serving(self):
        calls = _precompiled_unpacks(num_tokens_paddings_per_dp=[40],
                                     num_reqs_paddings_per_dp=[24])
        for call in calls:
            call.fn(call.blob, call.layout)
        (call, ) = calls

        fresh_blob = jax.device_put(np.arange(64, dtype=np.int32),
                                    call.blob.sharding)
        with ForbidCompile():
            parts = DeviceBuffer.unpack_arrays(fresh_blob, call.layout)
        assert parts["logits_indices"].tolist() == list(range(40, 64))

        # The guard in _prepare_inputs only means something if an unwarmed
        # layout trips it.
        with pytest.raises(RuntimeError, match="forbidden"):
            with ForbidCompile():
                DeviceBuffer.unpack_arrays(fresh_blob,
                                           _two_segment_layout(41, 23))
