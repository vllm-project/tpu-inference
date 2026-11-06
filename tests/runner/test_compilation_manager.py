# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

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


class TestMultiHostLockstep:
    """On multi-host, every host must finish compiling before any host
    starts the warmup collectives, or the PJRT stream can deadlock."""

    def _manager(self, tasks):
        manager = CompilationManager.__new__(CompilationManager)
        manager.runner = SimpleNamespace(
            mesh=Mesh(np.array(jax.devices()[:1]), ("x", )))
        manager._compile_futures = []
        manager._warmup_tasks = tasks
        return manager

    def test_barrier_runs_before_warmup_on_multi_host(self):
        events = []
        tasks = [("t", lambda: events.append("warmup") or jnp.zeros(
            ()), (), {}, None)]
        manager = self._manager(tasks)
        with mock.patch("jax.process_count", return_value=2), mock.patch(
                "tpu_inference.runner.compilation_manager.multihost_utils."
                "sync_global_devices",
                side_effect=lambda name: events.append(name)):
            manager._flush_compilations()
        assert events == ["flush_pre_warmup", "warmup"]

    def test_no_barrier_on_single_host(self):
        manager = self._manager([])
        with mock.patch("jax.process_count", return_value=1), mock.patch(
                "tpu_inference.runner.compilation_manager.multihost_utils."
                "sync_global_devices") as mock_sync:
            manager._flush_compilations()
        mock_sync.assert_not_called()


class TestPrecompileSpecDecodeDPSharding:
    """Under DP attention (dp_size > 1), precompiled speculative decoding
    helpers and rejection sampler must use P(ATTN_DATA, ...) sharding and
    the target model's actual logits dtype."""

    def _make_manager(self):
        devices = np.array(jax.devices()[:1]).reshape(
            (1, ) * len(MESH_AXIS_NAMES))
        mesh = Mesh(devices, MESH_AXIS_NAMES)
        runner = SimpleNamespace(
            mesh=mesh,
            rank=0,
            dp_size=2,
            vocab_size=256,
            max_num_reqs=8,
            num_reqs_paddings=[8],
            num_logits_paddings=[32],
            num_tokens_paddings=[32],
            attn_num_reqs_paddings=[8],
            rng_params_for_sampling=jnp.zeros((2, ), dtype=jnp.uint32),
            rejection_sampler=mock.MagicMock(),
            vllm_config=SimpleNamespace(parallel_config=None),
            scheduler_config=SimpleNamespace(async_scheduling=True),
            speculative_config=SimpleNamespace(
                num_speculative_tokens=3,
                method="mtp",
                draft_model_config=SimpleNamespace(get_hidden_size=lambda: 64),
            ),
            model_config=SimpleNamespace(
                max_logprobs=5,
                logprobs_mode="processed_logprobs",
                get_hidden_size=lambda: 64,
                dtype=jnp.bfloat16,
            ),
        )
        manager = CompilationManager.__new__(CompilationManager)
        manager.runner = runner
        manager._get_logits_dtype = lambda: jnp.float32
        manager._should_skip_padding_combination = lambda *args, **kwargs: False
        return manager

    def test_precompile_rejection_sampler_uses_attn_data_sharding_and_logits_dtype(
            self):
        manager = self._make_manager()
        calls = []

        def record(name, fn, *args, **kwargs):
            calls.append((name, args))

        manager._run_compilation = record
        CompilationManager._precompile_rejection_sampler(manager)

        assert len(calls) == 2
        for _name, args in calls:
            draft_token_ids, num_draft_tokens, _draft_probs, target_probs, _bonus_ids, sampling_meta, _rng = args
            assert draft_token_ids.sharding.spec == PartitionSpec(
                ShardingAxisName.ATTN_DATA)
            assert num_draft_tokens.sharding.spec == PartitionSpec(
                ShardingAxisName.ATTN_DATA)
            assert target_probs.sharding.spec == PartitionSpec(
                ShardingAxisName.ATTN_DATA, None)
            assert target_probs.dtype == jnp.float32
            if sampling_meta.do_sampling:
                assert sampling_meta.temperature.sharding.spec == PartitionSpec(
                    ShardingAxisName.ATTN_DATA)

    def test_precompile_extract_last_sampled_tokens_uses_attn_data_sharding(
            self):
        manager = self._make_manager()
        calls = {}

        def record(name, fn, *args, **kwargs):
            calls[name] = args

        manager._run_compilation = record
        CompilationManager._precompile_extract_last_sampled_tokens(manager)

        extract_args = calls["worker0 extract_last_sampled_tokens"]
        spec_meta, sampled_token_ids = extract_args[0], extract_args[1]
        assert spec_meta.draft_lengths.sharding.spec == PartitionSpec(
            ShardingAxisName.ATTN_DATA)
        assert sampled_token_ids.sharding.spec == PartitionSpec(
            ShardingAxisName.ATTN_DATA)

    def test_precompile_extend_logits_simple_and_gather_logprobs_use_attn_data_sharding(
            self):
        manager = self._make_manager()
        calls = {}

        def record(name, fn, *args, **kwargs):
            calls[name] = args

        manager._run_compilation = record
        CompilationManager._precompile_extend_logits_simple(manager)
        CompilationManager._precompile_gather_logprobs(manager)

        extend_args = calls["worker0 _extend_logits_simple"]
        target_logits, bonus_logits = extend_args[0], extend_args[1]
        assert target_logits.sharding.spec == PartitionSpec(
            ShardingAxisName.ATTN_DATA, None)
        assert bonus_logits.sharding.spec == PartitionSpec(
            ShardingAxisName.ATTN_DATA, None)

        gather_spec_args = calls["worker0 gather_logprobs_spec"]
        spec_logits, spec_token_ids = gather_spec_args[0], gather_spec_args[1]
        assert spec_logits.sharding.spec == PartitionSpec(
            ShardingAxisName.ATTN_DATA, None)
        assert spec_token_ids.sharding.spec == PartitionSpec(
            ShardingAxisName.ATTN_DATA)
