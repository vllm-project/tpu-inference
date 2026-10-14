# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, GroupedAttentionMetadata, SharedAttentionMetadata)
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName,
                                                  ShardingAxisNameBase)
from tpu_inference.layers.jax.sample.sampling import \
    logprobs_use_processed_logits
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.runner.compilation_manager import (CompilationManager,
                                                      _describe_signature)
from tpu_inference.runner.decode_loop import (TpuSamplingState,
                                              _get_decode_core,
                                              continue_decode)
from tpu_inference.utils import device_array


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


def _continue_decode_shapes(*,
                            dp_size=1,
                            req_paddings_per_dp=None,
                            attn_req_paddings_per_dp=None,
                            num_reqs_paddings=None):
    runner = SimpleNamespace(
        dp_size=dp_size,
        num_reqs_paddings_per_dp=req_paddings_per_dp,
        attn_num_reqs_paddings_per_dp=attn_req_paddings_per_dp,
        num_reqs_paddings=num_reqs_paddings or [],
    )
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = runner
    return CompilationManager._continue_decode_shapes(manager)


class TestContinueDecodeShapes:
    """`_precompile_continue_decode` must warm up every `(num_reqs, attn_num_reqs)`
    pair that `_prepare_input_metadata` can produce when `use_continue_decode`
    is True.
    """

    def test_default_single_attn_bucket_pairs_with_each_req_bucket(self):
        # Default ATTN_BUCKETIZED_NUM_REQS=False: attn_num_reqs_paddings is
        # [max_num_seqs] while num_reqs_paddings is power-of-two buckets.
        shapes = _continue_decode_shapes(req_paddings_per_dp=[8, 16, 32, 64],
                                         attn_req_paddings_per_dp=[64])
        assert shapes == [(8, 64), (16, 64), (32, 64), (64, 64)]

    def test_bucketized_attn_reqs_pairs_diagonally(self):
        shapes = _continue_decode_shapes(
            req_paddings_per_dp=[8, 16, 32, 64],
            attn_req_paddings_per_dp=[8, 16, 32, 64])
        assert shapes == [(8, 8), (16, 16), (32, 32), (64, 64)]

    def test_custom_attn_req_buckets_enumerates_all_reachable_pairs(self):
        # 1..4 live reqs -> (8, 4); 5..8 -> (8, 16); 9..16 -> (16, 16).
        shapes = _continue_decode_shapes(req_paddings_per_dp=[8, 16],
                                         attn_req_paddings_per_dp=[4, 16])
        assert shapes == [(8, 4), (8, 16), (16, 16)]

    def test_scales_by_dp_size(self):
        shapes = _continue_decode_shapes(dp_size=4,
                                         req_paddings_per_dp=[8, 16],
                                         attn_req_paddings_per_dp=[16])
        assert shapes == [(32, 64), (64, 64)]

    def test_fallback_when_per_dp_paddings_missing(self):
        shapes = _continue_decode_shapes(num_reqs_paddings=[8, 16])
        assert shapes == [(8, 8), (16, 16)]


def _make_continue_decode_runner(
    *,
    dp_size: int = 1,
    max_num_reqs: int = 16,
    req_paddings_per_dp: list[int] | None = None,
    attn_req_paddings_per_dp: list[int] | None = None,
    kv_cache_groups: list | None = None,
    block_table_max_blocks: list[int] | None = None,
    has_mamba_layers: bool = True,
    mamba_cache_mode: str = "none",
    uses_mrope: bool = False,
):
    devices = np.array(jax.devices()[:1]).reshape((1, ) * len(MESH_AXIS_NAMES))
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    if req_paddings_per_dp is None:
        req_paddings_per_dp = [8, 16]
    if attn_req_paddings_per_dp is None:
        attn_req_paddings_per_dp = [16]
    if kv_cache_groups is None:
        kv_cache_groups = [
            SimpleNamespace(layer_names=["layers.0.attn", "layers.2.attn"]),
            SimpleNamespace(
                layer_names=["layers.1.linear_attn", "layers.3.linear_attn"]),
        ]
    if block_table_max_blocks is None:
        block_table_max_blocks = [16, 4]

    block_tables = [
        SimpleNamespace(max_num_blocks_per_req=b)
        for b in block_table_max_blocks
    ]
    runner = SimpleNamespace(
        mesh=mesh,
        rank=0,
        dp_size=dp_size,
        max_num_reqs=max_num_reqs,
        attn_max_num_seqs=max_num_reqs,
        num_reqs_paddings=[r * dp_size for r in req_paddings_per_dp],
        num_reqs_paddings_per_dp=req_paddings_per_dp,
        attn_num_reqs_paddings=[r * dp_size for r in attn_req_paddings_per_dp],
        attn_num_reqs_paddings_per_dp=attn_req_paddings_per_dp,
        uses_mrope=uses_mrope,
        zero_array=device_array(mesh,
                                jnp.array(0, dtype=jnp.int32),
                                sharding=NamedSharding(mesh, PartitionSpec())),
        state_leaves=device_array(mesh,
                                  jnp.zeros((1, ), dtype=jnp.float32),
                                  sharding=NamedSharding(
                                      mesh, PartitionSpec())),
        kv_caches=[
            device_array(mesh,
                         jnp.zeros((8, 4), dtype=jnp.float32),
                         sharding=NamedSharding(mesh, PartitionSpec())),
            (device_array(mesh,
                          jnp.zeros((8, 4), dtype=jnp.float32),
                          sharding=NamedSharding(mesh, PartitionSpec())),
             device_array(mesh,
                          jnp.zeros((8, 4), dtype=jnp.float32),
                          sharding=NamedSharding(mesh, PartitionSpec()))),
        ],
        eos_token_id=(2, ),
        pad_token_id=0,
        rng_params_for_sampling=device_array(mesh,
                                             jax.random.key(0),
                                             sharding=NamedSharding(
                                                 mesh, PartitionSpec())),
        layer_name_to_kvcache_index={
            "layers.0.attn": 0,
            "layers.1.linear_attn": 1,
        },
        lora_utils=SimpleNamespace(extract_lora_metadata=lambda: None),
        is_first_rank=True,
        is_last_rank=True,
        continue_decode_eos_check_interval=1,
        model=SimpleNamespace(step_fn_no_options=lambda *a, **kw: None),
        compute_logits_fn=lambda *a, **kw: None,
        model_config=SimpleNamespace(max_logprobs=5,
                                     logprobs_mode="raw_logprobs"),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
        kv_cache_config=SimpleNamespace(
            has_mamba_layers=has_mamba_layers,
            kv_cache_groups=kv_cache_groups,
        ),
        input_batch=SimpleNamespace(block_table=block_tables),
        vllm_config=SimpleNamespace(
            parallel_config=None,
            sharding_config=SimpleNamespace(total_dp_size=dp_size),
            additional_config={"max_decode_steps": 4},
            model_config=SimpleNamespace(enable_return_routed_experts=False),
        ),
    )
    manager = CompilationManager.__new__(CompilationManager)
    manager.runner = runner
    return manager


class TestPrecompileContinueDecode:
    """Verify `_precompile_continue_decode` builds initial state and metadata
    matching runtime `_execute_continue_decode` / `_prepare_inputs` for GDN
    and hybrid models.
    """

    def test_gdn_hybrid_grouped_metadata_mamba_none_mode(self):
        manager = _make_continue_decode_runner(
            max_num_reqs=16,
            req_paddings_per_dp=[8, 16],
            attn_req_paddings_per_dp=[16],
            has_mamba_layers=True,
            mamba_cache_mode="none",
        )
        recorded = []

        def record(name, fn, *args, **kwargs):
            recorded.append((name, fn, args, kwargs))

        manager._run_compilation = record
        CompilationManager._precompile_continue_decode(manager)

        assert len(recorded) == 2
        for (expected_reqs, expected_attn_reqs), (_, _, args,
                                                  kwargs) in zip([(8, 16),
                                                                  (16, 16)],
                                                                 recorded):
            assert kwargs["num_reqs"] == expected_reqs
            assert kwargs["attn_num_reqs"] == expected_attn_reqs
            init_state: TpuSamplingState = args[6]
            assert init_state.current_tokens.shape == (expected_reqs, )
            assert init_state.active_mask.shape == (expected_reqs, )
            assert init_state.active_mask.dtype == jnp.bool_

            attn = init_state.attn_metadata
            assert isinstance(attn, GroupedAttentionMetadata)
            assert len(attn.groups) == 2
            assert attn.layer_names_per_group == (
                ("layers.0.attn", "layers.2.attn"),
                ("layers.1.linear_attn", "layers.3.linear_attn"),
            )
            for g in attn.groups:
                assert g.is_decode is True
                assert g.padded_num_reqs == expected_attn_reqs
                assert g.input_positions.shape == (expected_reqs, )
                assert g.seq_lens.shape == (16, )
                assert g.query_start_loc.shape == (17, )
                assert g.mamba_state_indices is not None
                assert g.mamba_state_indices.shape == (16, )
            # Distinct block table sizes per group (attention vs GDN/mamba).
            assert attn.groups[0].block_tables.shape == (16 * 16, )
            assert attn.groups[1].block_tables.shape == (16 * 4, )

    def test_gdn_hybrid_mamba_align_mode_omits_mamba_state_indices(self):
        manager = _make_continue_decode_runner(
            max_num_reqs=16,
            req_paddings_per_dp=[8, 16],
            attn_req_paddings_per_dp=[16],
            has_mamba_layers=True,
            mamba_cache_mode="align",
        )
        recorded = []
        manager._run_compilation = lambda name, fn, *args, **kw: recorded.append(
            args[6])
        CompilationManager._precompile_continue_decode(manager)

        assert len(recorded) == 2
        for init_state in recorded:
            attn = init_state.attn_metadata
            assert isinstance(attn, GroupedAttentionMetadata)
            for g in attn.groups:
                assert g.is_decode is True
                assert g.padded_num_reqs == 16
                assert g.mamba_state_indices is None
            assert attn.groups[0].block_tables.shape == (16 * 16, )
            assert attn.groups[1].block_tables.shape == (16 * 4, )

    @pytest.mark.parametrize("uses_mrope", [False, True])
    def test_input_positions_respects_uses_mrope(self, uses_mrope):
        manager = _make_continue_decode_runner(
            max_num_reqs=16,
            req_paddings_per_dp=[8],
            attn_req_paddings_per_dp=[16],
            uses_mrope=uses_mrope,
        )
        recorded = []
        manager._run_compilation = lambda name, fn, *args, **kw: recorded.append(
            args[6])
        CompilationManager._precompile_continue_decode(manager)

        init_state = recorded[0]
        pos = init_state.attn_metadata.input_positions
        if uses_mrope:
            assert pos.shape == (3, 8)
            assert pos.sharding.spec == PartitionSpec(
                None, ShardingAxisName.ATTN_DATA)
        else:
            assert pos.shape == (8, )
            assert pos.sharding.spec == PartitionSpec(
                ShardingAxisName.ATTN_DATA)

    def test_sharding_matches_runtime_under_base_sharding_with_zero_recompile(
            self, monkeypatch):
        # Under ShardingAxisNameBase (NEW_MODEL_DESIGN=1 / USE_2D_TP=1),
        # ShardingAxisName.BATCH != ShardingAxisName.ATTN_DATA. Verify that
        # precompiling continue_decode warms up the exact _get_decode_core()
        # JIT cache entry hit at runtime by _execute_continue_decode.
        if jax.default_backend() != "tpu":
            monkeypatch.setattr(
                "tpu_inference.runner.decode_loop.get_step_fn_compiler_options",
                lambda: {},
            )
            _get_decode_core.cache_clear()
        ShardingAxisName.override(
            ATTN_DATA=ShardingAxisNameBase.ATTN_DATA,
            BATCH=ShardingAxisNameBase.BATCH,
        )
        try:
            manager = _make_continue_decode_runner(
                max_num_reqs=16,
                req_paddings_per_dp=[8],
                attn_req_paddings_per_dp=[16],
                has_mamba_layers=True,
                mamba_cache_mode="align",
                uses_mrope=True,
            )
            runner = manager.runner

            def step_fn(state,
                        kv_caches,
                        input_ids,
                        attn_metadata,
                        inputs_embeds,
                        input_positions,
                        layer_name_to_kvcache_index,
                        lora_metadata,
                        intermediate_tensors=None,
                        is_first_rank=True,
                        is_last_rank=True,
                        shared_attention_metadata=None):
                assert isinstance(attn_metadata, GroupedAttentionMetadata)
                assert attn_metadata.is_decode is True
                assert attn_metadata.padded_num_reqs == 16
                assert shared_attention_metadata.is_decode is True
                assert shared_attention_metadata.padded_num_reqs == 16
                attn_kv = kv_caches[0] + 1.0
                conv_s, ssm_s = kv_caches[1]
                new_kvc = [attn_kv, (conv_s + 1.0, ssm_s + 1.0)]
                hidden = jnp.ones((input_ids.shape[0], 4), dtype=jnp.float32)
                return new_kvc, hidden, [], None

            def compute_logits_fn(state, hidden_states, lora_metadata):
                return jnp.zeros((hidden_states.shape[0], 8),
                                 dtype=jnp.float32)

            def sample_fn(rng,
                          mesh,
                          logits,
                          sampling_metadata,
                          allow_distributed_sampling=False):
                next_tokens = jnp.ones((logits.shape[0], ), dtype=jnp.int32)
                return next_tokens, logits

            runner.model.step_fn_no_options = step_fn
            runner.compute_logits_fn = compute_logits_fn

            def run_warmup(name, fn, *args, **kwargs):
                warmup_handler = kwargs.get("warmup_handler")
                init_state = args[6]
                # Check shardings match _prepare_inputs (BATCH vs ATTN_DATA).
                batch_spec = PartitionSpec(ShardingAxisName.BATCH)
                assert init_state.current_tokens.sharding.spec == batch_spec
                assert init_state.active_mask.sharding.spec == batch_spec
                attn = init_state.attn_metadata
                assert attn.seq_lens.sharding.spec == batch_spec
                assert attn.query_start_loc.sharding.spec == batch_spec
                assert attn.request_distribution.sharding.spec == batch_spec
                for g in attn.groups:
                    assert g.block_tables.sharding.spec == batch_spec
                assert attn.input_positions.sharding.spec == PartitionSpec(
                    None, ShardingAxisName.ATTN_DATA)
                # Replace sample with our lightweight sample_fn for CPU execution.
                new_args = list(args)
                new_args[3] = sample_fn
                warmup_handler(fn, new_args, {})

            manager._run_compilation = run_warmup
            CompilationManager._precompile_continue_decode(manager)

            decode_core = _get_decode_core()
            cache_size_after_precompile = decode_core._cache_size()

            # Now construct runtime inputs the exact way _prepare_inputs +
            # _execute_continue_decode do for 3 live requests (pads to 8 reqs,
            # attn_padded_num_reqs=16).
            metadata_attn_sharding = NamedSharding(
                runner.mesh, PartitionSpec(ShardingAxisName.BATCH))
            mrope_sharding = NamedSharding(
                runner.mesh, PartitionSpec(None, ShardingAxisName.ATTN_DATA))
            replicated_empty = NamedSharding(runner.mesh, PartitionSpec())

            rt_tokens = device_array(runner.mesh,
                                     np.ones((8, ), dtype=np.int32),
                                     sharding=metadata_attn_sharding)
            rt_positions = device_array(runner.mesh,
                                        np.ones((3, 8), dtype=np.int32),
                                        sharding=mrope_sharding)
            rt_active_mask = device_array(runner.mesh,
                                          np.array([True, True, True] +
                                                   [False] * 5,
                                                   dtype=np.bool_),
                                          sharding=metadata_attn_sharding)
            rt_seq_lens = device_array(runner.mesh,
                                       np.zeros((16, ), dtype=np.int32),
                                       sharding=metadata_attn_sharding)
            rt_qsl = device_array(runner.mesh,
                                  np.zeros((17, ), dtype=np.int32),
                                  sharding=metadata_attn_sharding)
            rt_dist = device_array(runner.mesh,
                                   np.array([3, 3, 3], dtype=np.int32),
                                   sharding=metadata_attn_sharding)
            rt_bt0 = device_array(runner.mesh,
                                  np.zeros((16 * 16, ), dtype=np.int32),
                                  sharding=metadata_attn_sharding)
            rt_bt1 = device_array(runner.mesh,
                                  np.zeros((16 * 4, ), dtype=np.int32),
                                  sharding=metadata_attn_sharding)

            rt_attn = GroupedAttentionMetadata(
                groups=(
                    AttentionMetadata(
                        input_positions=rt_positions,
                        block_tables=rt_bt0,
                        seq_lens=rt_seq_lens,
                        query_start_loc=rt_qsl,
                        request_distribution=rt_dist,
                        mamba_state_indices=None,
                        padded_num_reqs=16,
                        is_decode=True,
                    ),
                    AttentionMetadata(
                        input_positions=rt_positions,
                        block_tables=rt_bt1,
                        seq_lens=rt_seq_lens,
                        query_start_loc=rt_qsl,
                        request_distribution=rt_dist,
                        mamba_state_indices=None,
                        padded_num_reqs=16,
                        is_decode=True,
                    ),
                ),
                layer_names_per_group=(
                    ("layers.0.attn", "layers.2.attn"),
                    ("layers.1.linear_attn", "layers.3.linear_attn"),
                ),
            )
            rt_init_state = TpuSamplingState(
                current_tokens=rt_tokens,
                active_mask=rt_active_mask,
                attn_metadata=rt_attn,
                step_counter=runner.zero_array,
            )
            rt_sampling_metadata = TPUSupportedSamplingMetadata(
                temperature=None,
                top_k=None,
                top_p=None,
                _cache_collision_dummy=device_array(runner.mesh,
                                                    jnp.zeros((2, ),
                                                              dtype=jnp.int32),
                                                    sharding=replicated_empty),
                do_sampling=False,
                logprobs=False,
            )

            continue_decode(
                state=runner.state_leaves,
                model_fn=step_fn,
                compute_logits_fn=compute_logits_fn,
                sample_fn=sample_fn,
                mesh=runner.mesh,
                sampling_metadata=rt_sampling_metadata,
                init_state=rt_init_state,
                kv_caches=runner.kv_caches,
                max_decode_steps=jnp.array(2, dtype=jnp.int32),
                static_max_decode_steps=4,
                eos_token_id=runner.eos_token_id,
                padding_token_id=runner.pad_token_id,
                rng=runner.rng_params_for_sampling,
                layer_name_to_kvcache_index=tuple(
                    runner.layer_name_to_kvcache_index.items()),
                lora_metadata=None,
                is_first_rank=True,
                is_last_rank=True,
                dp_size=1,
                collect_expert_indices=False,
                max_logprobs=5,
                logprobs_mode="raw_logprobs",
                continue_decode_eos_check_interval=1,
            )

            assert decode_core._cache_size() == cache_size_after_precompile
        finally:
            ShardingAxisName.reset()
