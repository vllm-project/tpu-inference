# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import vllm.envs as vllm_envs
from jax.sharding import NamedSharding, PartitionSpec

import tpu_inference.envs as envs
from tpu_inference.core.disagg_utils import is_disagg_enabled
from tpu_inference.core.sched.utils import DEFAULT_MAX_DECODE_STEPS
from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling import (
    compute_and_gather_logprobs, compute_and_gather_prompt_logprobs, sample)
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.runner.decode_loop import TpuSamplingState, continue_decode
from tpu_inference.runner.utils import SpecDecodeMetadata
from tpu_inference.spec_decode.jax.utils import (
    concat_last_sampled_tokens_and_draft_tokens, extend_logits_simple,
    extract_last_sampled_tokens, process_and_extend_logits)
from tpu_inference.utils import (device_array, get_mesh_shape_product,
                                 time_function, to_jax_dtype)

if TYPE_CHECKING:
    from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)

# Constants for block bucketing in disaggregated utilities
BLOCK_BUCKETS = [1, 2, 4, 8, 16, 32, 64]


class CompilationManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        self._sampling_precompiled = False
        self._gather_logprobs_precompiled = False

        if not vllm_envs.VLLM_DISABLE_COMPILE_CACHE:
            logger.info("Enabling JAX compile cache.")
            jax.config.update("jax_compilation_cache_dir",
                              vllm_envs.VLLM_XLA_CACHE_PATH)
            if vllm_envs.VLLM_XLA_CHECK_RECOMPILATION:
                # Ensure small compiled function are cached as well.
                jax.config.update("jax_persistent_cache_min_entry_size_bytes",
                                  -1)
                jax.config.update("jax_persistent_cache_min_compile_time_secs",
                                  -1)
        # Thread pool for parallel XLA compilation. NUM_PRECOMPILE_WORKERS=1
        # disables the pool and runs compilations sequentially in the main
        # thread.
        num_workers = envs.NUM_PRECOMPILE_WORKERS
        self._prev_stack_size: Optional[int] = None
        if num_workers == 1:
            self._compile_executor = None
        else:
            # Pool threads default to the system thread stack size (~8MB on
            # Linux), much smaller than the main thread. XLA lowering overflows
            # that stack on large graphs. Bump the default stack size.
            try:
                self._prev_stack_size = threading.stack_size()
                threading.stack_size(64 * 1024 * 1024)
            except (RuntimeError, ValueError):
                self._prev_stack_size = None
            logger.info(
                "Parallel AOT compilation enabled (NUM_PRECOMPILE_WORKERS=%d).",
                num_workers)
            self._compile_executor = ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="aot_compilation")
        self._compile_futures: list[Future] = []
        self._warmup_tasks: list = []

    def _create_dummy_tensor(self,
                             shape: Tuple[int, ...],
                             dtype: Any,
                             sharding: Optional[NamedSharding] = None) -> Any:
        """Helper to create dummy tensors for precompilation."""
        if len(shape) > 1:
            # Use parallel_config as the primary source of truth during initialization
            # to avoid AttributeError when self.runner.mesh is not yet assigned (common in unit tests).
            tp_size = 1
            if self.runner.vllm_config.parallel_config is not None:
                tp_size = self.runner.vllm_config.parallel_config.tensor_parallel_size
            elif hasattr(self.runner, 'mesh') and self.runner.mesh is not None:
                tp_size = self.runner.mesh.shape.get(ShardingAxisName.MODEL, 1)
            assert shape[
                1] % tp_size == 0, f"Dimension size {shape[1]} is not divisible by TP size {tp_size} for shape {shape}"

        tensor = jnp.ones(shape, dtype=to_jax_dtype(dtype))
        if sharding:
            return device_array(self.runner.mesh, tensor, sharding=sharding)
        return device_array(self.runner.mesh, tensor)

    def _should_skip_padding_combination(self, outer_val: int, inner_val: int,
                                         only_equal: bool) -> bool:
        """Helper to determine if we should skip this padding combination."""
        if only_equal:
            return inner_val != outer_val
        return inner_val > outer_val

    def _run_compilation(self,
                         name: str,
                         fn: Callable,
                         *args,
                         call_kwargs=dict(),
                         warmup_handler: Optional[Callable] = None,
                         aot: bool = True,
                         compile_only: bool = False,
                         **kwargs) -> None:
        log_name = f"{name} --> {kwargs}"
        logger.info(f"Precompile {log_name}")
        # Unwrap functools.partial so the underlying jit's static_argnums are
        # respected.
        while isinstance(fn, functools.partial):
            args = fn.args + args
            call_kwargs = {**fn.keywords, **call_kwargs}
            fn = fn.func

        is_jit = hasattr(fn, 'lower')

        if compile_only:
            if not is_jit:
                raise ValueError(
                    f"compile_only=True requires a JITted function, but {name} is not a JIT."
                )
        else:
            self._warmup_tasks.append(
                (name, fn, args, call_kwargs, warmup_handler))

        if not compile_only and (not aot or not is_jit):
            # Skip AOT when the caller opts out, or when fn is unjitted.
            # The warmup pass will run fn() and populate the inner-jit caches.
            reason = "aot=False" if not aot else "not a jit"
            logger.info(
                "AOT lower skipped for %s (%s); will compile in warmup.", name,
                reason)
            return

        try:
            with jax.set_mesh(self.runner.mesh):
                lowered = fn.lower(*args, **call_kwargs)
        except Exception as e:
            if compile_only:
                logger.error(
                    f"Failed to lower {name} with compile_only=True: {e}")
                raise
            else:
                # AOT lower not supported here (e.g. a jit whose body contains a
                # nested jit with compiler_options). Fall back to warmup-only — the
                # warmup pass will trigger inline compile.
                logger.info(
                    "AOT lower skipped for %s (%r); will compile in warmup.",
                    name, e)
                return

        # Compilation is thread-safe
        def _compile(lowered, name, mesh):
            with jax.set_mesh(mesh):
                start = time.perf_counter()
                compiled = lowered.compile()
                elapsed = time.perf_counter() - start
                logger.info("Compilation of %s finished in %.2f [secs].", name,
                            elapsed)
                return compiled

        if self._compile_executor is None:
            _compile(lowered, log_name, self.runner.mesh)
        else:
            future = self._compile_executor.submit(_compile, lowered, log_name,
                                                   self.runner.mesh)
            self._compile_futures.append(future)

    def _flush_compilations(self) -> None:
        """Wait for all currently-pending background compilations and run their
        warmups.
        """
        futures, self._compile_futures = self._compile_futures, []
        tasks, self._warmup_tasks = self._warmup_tasks, []

        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                raise RuntimeError(
                    f"Compilation failed: {e}\n"
                    "Hint: if you are seeing memory errors or stack overflows "
                    "during parallel precompilation, try lowering "
                    "NUM_PRECOMPILE_WORKERS (e.g. NUM_PRECOMPILE_WORKERS=1 "
                    "runs all compilations sequentially in the main thread)."
                ) from e

        warmup_start = time.perf_counter()
        with jax.set_mesh(self.runner.mesh):
            for name, fn, args, call_kwargs, warmup_handler in tasks:
                if warmup_handler is not None:
                    out = warmup_handler(fn, args, call_kwargs)
                else:
                    out = fn(*args, **call_kwargs)
                jax.tree.map(lambda r: r.block_until_ready(), out)
        warmup_elapsed = time.perf_counter() - warmup_start
        if tasks:
            logger.info(
                "Warm-up call pass finished in %.2f [secs] over %d tasks.",
                warmup_elapsed, len(tasks))

    @time_function
    def capture_model(self) -> None:
        if envs.SKIP_JAX_PRECOMPILE or self.runner.model_config.enforce_eager:
            return
        logger.info("Precompile all the subgraphs with possible input shapes.")
        compilation_start_time = time.perf_counter()

        try:
            with self.runner.maybe_setup_dummy_loras(
                    self.runner.lora_config), jax.set_mesh(self.runner.mesh):
                self._precompile_backbone_text_only()
                self._flush_compilations()
                if self.runner.is_multimodal_model:
                    if self.runner.precompile_vision_encoder_fn is not None:
                        self.runner.precompile_vision_encoder_fn(
                            self._run_compilation, )
                    self._precompile_input_embeddings_merger()
                    self._flush_compilations()
                    self._precompile_backbone_with_inputs_embeds()
                    self._flush_compilations()
                if self.runner.scheduler_config.async_scheduling:
                    self._precompile_substitute_placeholder_token()
                    self._flush_compilations()
                    if self.runner.speculative_config:
                        self._precompile_subtract_num_rejected_tokens()
                        self._flush_compilations()
                        self._precompile_concat_last_sampled_tokens_and_draft_tokens(
                        )
                        self._flush_compilations()

                if not self.runner.is_last_rank:
                    return
                self._precompile_select_from_array()
                self._flush_compilations()
                if not self.runner.is_pooling_model:
                    self._precompile_compute_logits()
                else:
                    self._precompile_compute_pooling()
                self._flush_compilations()
                # Skip sampling if already precompiled before KV cache allocation
                if not self._sampling_precompiled:
                    self._precompile_sampling()
                    self._flush_compilations()
                self._precompile_disagg_utils()
                self._flush_compilations()
                # Skip gather_logprobs if already precompiled before KV cache allocation
                if not self._gather_logprobs_precompiled:
                    self._precompile_gather_logprobs()
                    self._flush_compilations()
                self._precompile_structured_decoding()
                self._flush_compilations()
                if self.runner.speculative_config:
                    self._precompile_speculative_decoding()
                    self._flush_compilations()
                if self.runner.enable_continue_decode:
                    self._precompile_continue_decode()
                    self._flush_compilations()
        finally:
            self._finalize_compilation()
        elapsed = time.perf_counter() - compilation_start_time
        self.runner.vllm_config.compilation_config.compilation_time += elapsed

    def _finalize_compilation(self) -> None:
        """Shut down the precompile pool and restore the thread stack default
        so the bumped stack size doesn't leak to threads spawned later by the
        engine."""
        if self._compile_executor is not None:
            self._compile_executor.shutdown(wait=True)
            self._compile_executor = None
        if self._prev_stack_size is not None:
            try:
                threading.stack_size(self._prev_stack_size)
            except (RuntimeError, ValueError):
                pass
            self._prev_stack_size = None

    def _precompile_input_embeddings_merger(self) -> None:
        for num_tokens in self.runner.num_tokens_paddings:
            hidden_size = self.runner.vllm_config.model_config.get_hidden_size(
            )
            hf_conf = self.runner.vllm_config.model_config.hf_config

            # Identify multimodal embedding size
            mm_hidden_size = hidden_size
            vision_config = getattr(hf_conf, "vision_config", None)

            if vision_config:
                visual_dim = getattr(vision_config, "out_hidden_size", None)
                deepstack_indexes = getattr(vision_config,
                                            "deepstack_visual_indexes", None)

                # If both exist, we apply the deepstack concat logic
                if visual_dim is not None and deepstack_indexes is not None:
                    deepstack_levels = len(deepstack_indexes)
                    mm_hidden_size = visual_dim * (1 + deepstack_levels)

            sharding = NamedSharding(
                self.runner.mesh,
                PartitionSpec(ShardingAxisName.ATTN_DATA, None))
            input_sharding = NamedSharding(
                self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))

            dummy_multimodal_embeddings = self._create_dummy_tensor(
                (num_tokens, mm_hidden_size),
                self.runner.vllm_config.model_config.dtype,
                sharding=sharding)
            dummy_input_ids = self._create_dummy_tensor(
                (num_tokens, ), jnp.int32, sharding=input_sharding)
            dummy_is_multimodal = self._create_dummy_tensor(
                (num_tokens, ), jnp.bool_, sharding=input_sharding)

            self._run_compilation(
                "input_embeddings_merger",
                self.runner.embed_input_ids_fn,
                self.runner.state_leaves,
                dummy_input_ids,
                # Make _compute_deepstack_embeds happy.
                [dummy_multimodal_embeddings],
                call_kwargs={"is_multimodal": dummy_is_multimodal},
                num_tokens=num_tokens,
            )

            self._run_compilation(
                "input_embeddings_merger_text_only",
                self.runner.embed_input_ids_fn,
                self.runner.state_leaves,
                dummy_input_ids,
                None,
                call_kwargs={"is_multimodal": None},
                num_tokens=num_tokens,
            )

    def _precompile_backbone_helper(self,
                                    name,
                                    *,
                                    input_ids,
                                    positions,
                                    inputs_embeds,
                                    intermediate_tensors=None,
                                    is_first_rank=True,
                                    is_last_rank=True,
                                    num_reqs: int) -> None:
        num_tokens = None
        if input_ids is not None:
            num_tokens = input_ids.shape[0]
        elif inputs_embeds is not None:
            num_tokens = inputs_embeds.shape[0]
        assert num_tokens is not None

        dp_size = self.runner.vllm_config.sharding_config.total_dp_size
        dp_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, ))

        # Keep existing pattern for complex array operations
        seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                             jnp.int32, dp_sharding)
        query_start_loc = self._create_dummy_tensor(
            (self.runner.max_num_reqs + dp_size, ), jnp.int32, dp_sharding)

        # Keep existing pattern for specific value arrays
        request_distribution = np.array([0, 0, 0] * dp_size, dtype=np.int32)
        request_distribution = device_array(self.runner.mesh,
                                            request_distribution,
                                            sharding=dp_sharding)
        # Dummy mamba_state_indices for compile-cache pre-tracing. Only
        # populate for hybrid attn+mamba models — for pure-attention models we
        # pass None at runtime (see `_prepare_inputs`), and the precompile
        # primer must match that shape so the cached HLO is reused.
        if self.runner.kv_cache_config.has_mamba_layers:
            mamba_state_indices = device_array(self.runner.mesh,
                                               np.zeros(
                                                   self.runner.max_num_reqs,
                                                   dtype=np.int32),
                                               sharding=dp_sharding)
        else:
            mamba_state_indices = None

        def build_block_table(kv_cache_gid: int) -> jax.Array:
            block_table_obj = self.runner.input_batch.block_table[kv_cache_gid]
            shape = (self.runner.max_num_reqs,
                     block_table_obj.max_num_blocks_per_req)
            block_tables = np.zeros(shape, dtype=np.int32)
            block_tables = block_tables.reshape(-1)
            block_tables = device_array(self.runner.mesh,
                                        block_tables,
                                        sharding=dp_sharding)
            return block_tables

        def build_attn(block_tables: jax.Array | None) -> AttentionMetadata:
            attention_metadata_gid = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                mamba_state_indices=mamba_state_indices,
                padded_num_reqs=num_reqs,
            )

            return attention_metadata_gid

        def build_shared_attn() -> SharedAttentionMetadata:
            return SharedAttentionMetadata(
                input_positions=positions,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                mamba_state_indices=mamba_state_indices,
                padded_num_reqs=num_reqs,
            )

        attention_metadata: AttentionMetadata | dict[str, AttentionMetadata]
        shared_attention_metadata: SharedAttentionMetadata
        if len(self.runner.kv_cache_config.kv_cache_groups) <= 1:
            # Pooling model will not using kv cache
            no_kv_cache = len(self.runner.kv_cache_config.kv_cache_groups) == 0
            block_tables = build_block_table(0) if not no_kv_cache else None
            attention_metadata = build_attn(block_tables)
            shared_attention_metadata = build_shared_attn()
        else:
            attention_metadata = {
                name: build_attn(build_block_table(gid))
                for gid, kv_cache_group in enumerate(
                    self.runner.kv_cache_config.kv_cache_groups)
                for name in kv_cache_group.layer_names
            }
            shared_attention_metadata = build_shared_attn()

        def model_fn_warmup(_fn, _args, _call_kwargs):
            out = self.runner.model_fn(
                self.runner.state_leaves,
                self.runner.kv_caches,
                input_ids,
                attention_metadata,
                inputs_embeds,
                positions,
                tuple(self.runner.layer_name_to_kvcache_index.items()),
                lora_metadata,
                intermediate_tensors,
                is_first_rank,
                is_last_rank,
                shared_attention_metadata=shared_attention_metadata,
            )
            self.runner.kv_caches = out[0]
            return out

        with self.runner.maybe_select_dummy_loras(
                self.runner.lora_config, np.array([num_tokens],
                                                  dtype=np.int32)):
            lora_metadata = self.runner.lora_utils.extract_lora_metadata()
            self._run_compilation(
                name,
                self.runner.model_fn,
                self.runner.state_leaves,
                self.runner.kv_caches,
                input_ids,
                attention_metadata,
                inputs_embeds,
                positions,
                tuple(self.runner.layer_name_to_kvcache_index.items()),
                lora_metadata,
                intermediate_tensors,
                is_first_rank,
                is_last_rank,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                warmup_handler=model_fn_warmup,
                shared_attention_metadata=shared_attention_metadata,
            )

    def _precompile_substitute_placeholder_token(self) -> None:
        dp_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, ))
        replicated_sharding = NamedSharding(self.runner.mesh,
                                            PartitionSpec(None))
        indices_sharding = NamedSharding(self.runner.mesh, PartitionSpec(None))

        placeholder_num = self._create_dummy_tensor((1, ), jnp.int32)

        def _compile_one(input_padding: int, input_sharding: NamedSharding,
                         next_tokens_size: int,
                         next_tokens_sharding: NamedSharding) -> None:
            padded_token_in_tpu_cur_input_indices = self._create_dummy_tensor(
                (input_padding, ), jnp.int32, sharding=indices_sharding)
            padded_token_in_tpu_pre_next_tokens_indices = self._create_dummy_tensor(
                (input_padding, ), jnp.int32, sharding=indices_sharding)

            input_ids = self._create_dummy_tensor((input_padding, ),
                                                  jnp.int32,
                                                  sharding=input_sharding)
            next_tokens = self._create_dummy_tensor(
                (next_tokens_size, ), jnp.int32, sharding=next_tokens_sharding)

            self._run_compilation(
                "_substitute_placeholder_token_fn",
                self.runner._substitute_placeholder_token_fn,
                input_ids,
                padded_token_in_tpu_cur_input_indices,
                padded_token_in_tpu_pre_next_tokens_indices,
                next_tokens,
                placeholder_num,
                compile_only=False,
                num_tokens=input_padding,
                next_tokens_size=next_tokens_size,
            )

        all_token_sizes = sorted(
            list(
                set(self.runner.num_tokens_paddings +
                    self.runner.num_reqs_paddings)))

        if self.runner.speculative_config:
            num_spec_tokens = (
                self.runner.speculative_config.num_speculative_tokens)
            spec_next_tokens_size = self.runner.max_num_reqs * (
                num_spec_tokens + 1)
            for num_tokens in all_token_sizes:
                _compile_one(num_tokens, dp_sharding, spec_next_tokens_size,
                             dp_sharding)
            for num_logits in self.runner.num_logits_paddings:
                _compile_one(num_logits, replicated_sharding,
                             spec_next_tokens_size, dp_sharding)
        else:
            for num_tokens in all_token_sizes:
                for next_tokens_size in all_token_sizes:
                    _compile_one(num_tokens, dp_sharding, next_tokens_size,
                                 dp_sharding)
                for num_reqs in self.runner.num_reqs_paddings:
                    _compile_one(num_tokens, dp_sharding, num_reqs,
                                 replicated_sharding)

    def _precompile_subtract_num_rejected_tokens(self) -> None:
        from tpu_inference.runner.tpu_runner import \
            _subtract_num_rejected_tokens_fn

        dp_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, ))
        replicated_sharding = NamedSharding(self.runner.mesh,
                                            PartitionSpec(None))

        for num_tokens in self.runner.num_tokens_paddings:
            # `_subtract_num_rejected_tokens_fn` donates seq_lens and positions
            # (donate_argnums=(0, 1)) so a fresh pair must be created per call.
            seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                                 jnp.int32, dp_sharding)
            if self.runner.uses_mrope:
                mrope_sharding = NamedSharding(
                    self.runner.mesh,
                    PartitionSpec(None, ShardingAxisName.ATTN_DATA))
                positions = self._create_dummy_tensor(
                    (3, num_tokens), jnp.int32, mrope_sharding)
            else:
                positions = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)
            num_rejected_tokens = self._create_dummy_tensor(
                (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
            seq_lens_subtract_indices = self._create_dummy_tensor(
                (self.runner.max_num_reqs, ), jnp.int32, replicated_sharding)
            positions_subtract_indices = self._create_dummy_tensor(
                (num_tokens, ), jnp.int32, replicated_sharding)

            self._run_compilation(
                "_subtract_num_rejected_tokens_fn",
                _subtract_num_rejected_tokens_fn,
                seq_lens,
                positions,
                num_rejected_tokens,
                seq_lens_subtract_indices,
                positions_subtract_indices,
                num_tokens=num_tokens,
            )

    def _precompile_concat_last_sampled_tokens_and_draft_tokens(self) -> None:
        logger.info(
            "Compiling concat_last_sampled_tokens_and_draft_tokens with "
            "different input shapes.")
        num_spec_tokens = (
            self.runner.speculative_config.num_speculative_tokens)
        max_num_reqs = self.runner.max_num_reqs
        data_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))

        last_sampled_tokens = device_array(self.runner.mesh,
                                           jnp.ones((max_num_reqs, ),
                                                    dtype=jnp.int32),
                                           sharding=data_sharding)
        draft_tokens = device_array(self.runner.mesh,
                                    jnp.ones((max_num_reqs, num_spec_tokens),
                                             dtype=jnp.int32),
                                    sharding=data_sharding)
        self._run_compilation(
            f"worker{self.runner.rank} "
            "concat_last_sampled_tokens_and_draft_tokens",
            concat_last_sampled_tokens_and_draft_tokens,
            last_sampled_tokens,
            draft_tokens,
            max_num_reqs=max_num_reqs,
            num_spec_tokens=num_spec_tokens,
        )

    def _precompile_backbone_text_only(self) -> None:
        hidden_size = self.runner.model_config.get_hidden_size()
        for num_tokens in self.runner.num_tokens_paddings:
            for num_reqs in self.runner.attn_num_reqs_paddings:
                dp_sharding = NamedSharding(
                    self.runner.mesh,
                    PartitionSpec(ShardingAxisName.ATTN_DATA, ))
                input_ids = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)
                if self.runner.uses_mrope:
                    mrope_sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(None, ShardingAxisName.ATTN_DATA))
                    positions = self._create_dummy_tensor(
                        (3, num_tokens), jnp.int32, mrope_sharding)
                else:
                    positions = self._create_dummy_tensor(
                        (num_tokens, ), jnp.int32, dp_sharding)
                is_first_rank = self.runner.is_first_rank
                is_last_rank = self.runner.is_last_rank
                if is_first_rank:
                    intermediate_tensors = None
                else:
                    sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.ATTN_DATA, None))
                    hidden_states = self._create_dummy_tensor(
                        (num_tokens, hidden_size),
                        jnp.bfloat16,
                        sharding=sharding)
                    residual = self._create_dummy_tensor(
                        (num_tokens, hidden_size),
                        jnp.bfloat16,
                        sharding=sharding)
                    intermediate_tensors = JaxIntermediateTensors(
                        tensors={
                            "hidden_states": hidden_states,
                            "residual": residual
                        })
                self._precompile_backbone_helper(
                    f"worker{self.runner.rank} backbone",
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=None,
                    intermediate_tensors=intermediate_tensors,
                    is_first_rank=is_first_rank,
                    is_last_rank=is_last_rank,
                    num_reqs=num_reqs)

    def _precompile_backbone_with_inputs_embeds(self) -> None:
        hidden_size = self.runner.model_config.get_hidden_size()
        dtype = self.runner.model_config.dtype

        # Identify multimodal embedding size including Deepstack
        hf_conf = self.runner.vllm_config.model_config.hf_config
        vision_config = getattr(hf_conf, "vision_config", None)
        embeds_hidden_size = hidden_size

        if vision_config:
            visual_dim = getattr(vision_config, "out_hidden_size", None)
            deepstack_indexes = getattr(vision_config,
                                        "deepstack_visual_indexes", None)

            # If both exist, we apply the deepstack concat logic
            if visual_dim is not None and deepstack_indexes is not None:
                deepstack_levels = len(deepstack_indexes)
                embeds_hidden_size = visual_dim * (1 + deepstack_levels)

        # Compile for both standard (4k) and Deepstack (16k) dimensions if they differ
        hidden_sizes_to_compile = [hidden_size]
        if embeds_hidden_size != hidden_size:
            hidden_sizes_to_compile.append(embeds_hidden_size)

        for h_size in hidden_sizes_to_compile:
            for num_tokens in self.runner.num_tokens_paddings:
                for num_reqs in self.runner.attn_num_reqs_paddings:
                    sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.ATTN_DATA, None))
                    input_sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.ATTN_DATA))

                    inputs_embeds = self._create_dummy_tensor(
                        (num_tokens, h_size), dtype, sharding=sharding)
                if self.runner.uses_mrope:
                    mrope_sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(None, ShardingAxisName.ATTN_DATA))
                    positions = self._create_dummy_tensor(
                        (3, num_tokens), jnp.int32, sharding=mrope_sharding)
                else:
                    positions = self._create_dummy_tensor(
                        (num_tokens, ), jnp.int32, sharding=input_sharding)
                is_first_rank = self.runner.is_first_rank
                is_last_rank = self.runner.is_last_rank
                if not is_first_rank:
                    hidden_states = self._create_dummy_tensor(
                        (num_tokens, hidden_size),
                        jnp.bfloat16,
                        sharding=sharding)
                    residual = self._create_dummy_tensor(
                        (num_tokens, hidden_size),
                        jnp.bfloat16,
                        sharding=sharding)
                    intermediate_tensors = JaxIntermediateTensors(
                        tensors={
                            "hidden_states": hidden_states,
                            "residual": residual
                        })
                else:
                    intermediate_tensors = None
                self._precompile_backbone_helper(
                    f"worker{self.runner.rank} backbone with embeds",
                    input_ids=None,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                    intermediate_tensors=intermediate_tensors,
                    is_first_rank=is_first_rank,
                    is_last_rank=is_last_rank,
                    num_reqs=num_reqs)

    def _precompile_select_from_array_helper(
        self,
        name: str,
        source_paddings: List[int],
        indices_paddings: List[int],
        hidden_dim: int,
        input_sharding: Optional[NamedSharding] = None,
        indices_sharding: Optional[NamedSharding] = None,
        only_equal_paddings: bool = False,
        check_should_skip_padding: bool = True,
    ) -> None:
        """Precompile select_from_array operations with various input shape combinations.

        This helper method generates and precompiles the select_from_array function for different
        combinations of array sizes and index counts. The operation being precompiled is
        array[indices] where:
        - array has shape (array_size, hidden_dim)
        - indices has shape (indices_count,)
        - result has shape (indices_count, hidden_dim)

        This is essential for TPU compilation as JAX needs to precompile functions with all
        possible input shapes that will be encountered during runtime.

        Args:
            name: Descriptive name for logging purposes (e.g., "select all logits")
            source_paddings: List of possible sizes for the array being indexed (first dimension)
            indices_paddings: List of possible counts of indices to select
            hidden_dim: Second dimension size of the array (e.g., hidden_size or vocab_size)
            sharding: Optional sharding specification for distributed computation
            only_equal_paddings: If True, only compile when array size equals indices count
            check_should_skip_padding: If True, check whether to skip certain padding combinations to reduce compilation time
        """
        logger.info(f"Compiling select_from_array for {name}.")
        for array_size in source_paddings:
            for indices_count in indices_paddings:
                if check_should_skip_padding and self._should_skip_padding_combination(
                        array_size, indices_count, only_equal_paddings):
                    continue

                array = jax.ShapeDtypeStruct((array_size, hidden_dim),
                                             jnp.bfloat16,
                                             sharding=input_sharding)
                indices_to_select = jax.ShapeDtypeStruct(
                    (indices_count, ), jnp.int32, sharding=indices_sharding)

                self._run_compilation(
                    f"select_from_array [{name}]",
                    self.runner._select_from_array_fn,
                    array,
                    indices_to_select,
                    self.runner.mesh,
                    compile_only=True,
                    array_size=array_size,
                    index_size=indices_count,
                )

    def _skip_self_arg_warmup_handler(self, fn, args, call_kwargs):
        """Warmup handler for methods compiled with an explicit `self` as the
        first positional arg.  At warm-up time the object is already bound, so
        we drop args[0] and forward the rest.
        """
        return fn(*args[1:], **call_kwargs)

    def _precompile_select_from_array(self) -> None:
        logger.info("Compiling select_from_array with different input shapes.")
        hsize = self.runner.model_config.get_hidden_size()

        if self.runner.speculative_config:
            index_paddings = self.runner.num_logits_paddings
        else:
            index_paddings = self.runner.num_reqs_paddings
        dp_sharding = NamedSharding(self.runner.mesh,
                                    PartitionSpec(ShardingAxisName.ATTN_DATA))
        hidden_states_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
        self._precompile_select_from_array_helper(
            name=f"worker{self.runner.rank} select all logits",
            source_paddings=self.runner.num_tokens_paddings,
            indices_paddings=index_paddings,
            hidden_dim=hsize,
            input_sharding=hidden_states_sharding,
            indices_sharding=dp_sharding,
        )

        if self.runner.speculative_config:
            vocab_size = self.runner.vocab_size
            self._precompile_select_from_array_helper(
                name=
                f"worker{self.runner.rank} select bonus tokens for spec decoding",
                source_paddings=self.runner.num_logits_paddings,
                indices_paddings=self.runner.num_reqs_paddings,
                hidden_dim=vocab_size,
                input_sharding=NamedSharding(
                    self.runner.mesh,
                    PartitionSpec(ShardingAxisName.MLP_DATA,
                                  ShardingAxisName.MLP_TENSOR)),
            )
            self._precompile_select_from_array_helper(
                name=
                f"worker{self.runner.rank} select target tokens for spec decoding",
                source_paddings=self.runner.num_logits_paddings,
                indices_paddings=self.runner.num_logits_paddings,
                hidden_dim=vocab_size,
                input_sharding=NamedSharding(
                    self.runner.mesh,
                    PartitionSpec(ShardingAxisName.MLP_DATA,
                                  ShardingAxisName.MLP_TENSOR)),
                only_equal_paddings=True,
            )

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        hsize = self.runner.model_config.get_hidden_size()
        leading_shape = self.runner.num_reqs_paddings if not self.runner.speculative_config else self.runner.num_logits_paddings
        # Use PartitionSpec(ATTN_DATA, None) (2D explicit) to match the sharding
        # that _select_from_array_fn produces at inference time. shard_map with
        # out_specs=P('data') returns arrays with spec P('data', None); since
        # compute_logits_fn has no in_shardings, JAX uses the actual input
        # sharding as the jit cache key. P('data',) != P('data', None) as cache
        # keys, causing a cache miss inside ForbidCompile → RuntimeError for VLMs.
        hidden_states_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
        for num_reqs in leading_shape:
            hidden_states = jax.ShapeDtypeStruct(
                (num_reqs, hsize),
                jnp.bfloat16,
                sharding=hidden_states_sharding)
            with self.runner.maybe_select_dummy_loras(
                    self.runner.lora_config,
                    np.array([num_reqs], dtype=np.int32)):
                lora_metadata = self.runner.lora_utils.extract_lora_metadata()
                self._run_compilation(
                    f"worker{self.runner.rank} compute_logits",
                    self.runner.compute_logits_fn,
                    self.runner.state_leaves,
                    hidden_states,
                    lora_metadata,
                    compile_only=True,
                    num_reqs=num_reqs,
                )

    def _precompile_compute_pooling(self) -> None:
        logger.info("Compiling compute_pooling with different input shapes.")

        # vLLM pooling layer design has complex and dynamic logic. There are
        # interoperate between tensors from host and accelerator.
        # It's quite hard, if not impossible, to move all tensor to accelerator
        # and apply JIT on the entire computation.
        # See PoolingCursor and AllPool, MeanPool ... in vLLM repo for details.

    def _precompile_sampling(self) -> None:
        logger.info("Compiling sampling with different input shapes.")
        hsize = self.runner.vocab_size
        replicated_sharding = NamedSharding(self.runner.mesh, PartitionSpec())
        for num_reqs in self.runner.num_reqs_paddings:
            # `logits_sharding` need to be consistent with
            # compute_logits_fn's output sharding to avoid serving
            # time re-compilation.
            logits_sharding = NamedSharding(
                self.runner.mesh,
                PartitionSpec(ShardingAxisName.MLP_DATA,
                              ShardingAxisName.MLP_TENSOR))
            # Similarly, `sampling_metadata_sharding` need to consistent
            # with runtime sampling_metadata sharding to the sample
            # function.
            sampling_metadata_sharding = NamedSharding(
                self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))
            logits = self._create_dummy_tensor((num_reqs, hsize),
                                               jnp.float32,
                                               sharding=logits_sharding)
            for do_sampling in (True, False):
                for logprobs in (True, False):
                    if do_sampling:
                        temperature = self._create_dummy_tensor(
                            (num_reqs, ),
                            jnp.float32,
                            sharding=sampling_metadata_sharding)
                        top_k = self._create_dummy_tensor(
                            (num_reqs, ),
                            jnp.int32,
                            sharding=sampling_metadata_sharding)
                        top_p = self._create_dummy_tensor(
                            (num_reqs, ),
                            jnp.float32,
                            sharding=sampling_metadata_sharding)
                    else:
                        temperature = None
                        top_k = None
                        top_p = None

                    # Use a dummy tensor with a unique shape for each logprobs config.
                    # This avoids persistent cache collisions.
                    dummy_shape = (1 if logprobs else 2, )
                    _cache_collision_dummy = self._create_dummy_tensor(
                        dummy_shape, jnp.int32, sharding=replicated_sharding)

                    sampling_metadata = TPUSupportedSamplingMetadata(
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        _cache_collision_dummy=_cache_collision_dummy,
                        do_sampling=do_sampling,
                        logprobs=logprobs)
                    self._run_compilation(
                        f"worker{self.runner.rank} sample",
                        sample,
                        self.runner.rng_params_for_sampling,
                        self.runner.mesh,
                        logits,
                        sampling_metadata,
                        compile_only=False,
                        num_reqs=num_reqs,
                        do_sampling=do_sampling,
                        logprobs=logprobs,
                    )

        self._sampling_precompiled = True

    def _precompile_disagg_utils(self) -> None:
        if not is_disagg_enabled():
            return
        logger.info(
            "Compiling disaggregated util with different input shapes.")
        block_size = self.runner.block_size
        for num_blocks in range(1, self.runner.max_num_blocks_per_req // 2):
            logger.info(
                f"Precompile slice and insert for num_blocks {num_blocks}")
            block_numbers = list(range(1, num_blocks + 1))
            kv_cache_slices = self.runner.kv_cache_manager.get_kv_cache_for_block_ids(
                block_numbers)
            # Prevent the slices from getting freed by insert before finishing this operation
            for layer_cache in kv_cache_slices:
                layer_cache.block_until_ready()
            self.runner.kv_caches = self.runner.kv_cache_manager._jitted_insert_continuous_kv_cache_from_slice(
                block_size,
                num_blocks,
                self.runner.kv_caches,
                kv_cache_slices,
                0,
                block_numbers[0],
            )
            for layer_cache in self.runner.kv_caches:
                layer_cache.block_until_ready()

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        hsize = self.runner.vocab_size
        for num_reqs in self.runner.num_reqs_paddings:
            logits_sharding = NamedSharding(
                self.runner.mesh,
                PartitionSpec(ShardingAxisName.MLP_DATA,
                              ShardingAxisName.MLP_TENSOR))
            token_ids_sharding = NamedSharding(self.runner.mesh,
                                               PartitionSpec())
            logits = jax.ShapeDtypeStruct((num_reqs, hsize),
                                          jnp.float32,
                                          sharding=logits_sharding)
            token_ids = jax.ShapeDtypeStruct((num_reqs, ),
                                             jnp.int32,
                                             sharding=token_ids_sharding)
            self._run_compilation(
                f"worker{self.runner.rank} gather_logprobs",
                compute_and_gather_logprobs,
                logits,
                token_ids,
                self.runner.model_config.max_logprobs,
                compile_only=True,
                num_reqs=num_reqs,
            )

        if self.runner.speculative_config:
            logger.info(
                "Compiling gather_logprobs for speculative decoding shapes.")
            for num_logits in self.runner.num_logits_paddings:
                for num_reqs in self.runner.num_reqs_paddings:
                    if num_reqs > num_logits:
                        continue
                    combined_size = num_logits + num_reqs
                    logits_sharding = NamedSharding(self.runner.mesh,
                                                    PartitionSpec())
                    token_ids_sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.ATTN_DATA))
                    logits = jax.ShapeDtypeStruct((combined_size, hsize),
                                                  jnp.float32,
                                                  sharding=logits_sharding)
                    token_ids = jax.ShapeDtypeStruct(
                        (combined_size, ),
                        jnp.int32,
                        sharding=token_ids_sharding)
                    self._run_compilation(
                        f"worker{self.runner.rank} gather_logprobs_spec",
                        compute_and_gather_logprobs,
                        logits,
                        token_ids,
                        self.runner.model_config.max_logprobs,
                        compile_only=True,
                        num_logits=num_logits,
                        num_reqs=num_reqs,
                    )

        logger.info(
            "Compiling compute_and_gather_prompt_logprobs with different input shapes."
        )
        # Bypassed MAX_PRECOMPILE_PROMPT_TOKENS limit as ShapeDtypeStruct compilation allocates no HBM
        for num_tokens in self.runner.num_tokens_paddings:
            logits_sharding = NamedSharding(
                self.runner.mesh,
                PartitionSpec(ShardingAxisName.MLP_DATA,
                              ShardingAxisName.MLP_TENSOR))
            token_ids_sharding = NamedSharding(self.runner.mesh,
                                               PartitionSpec())
            logits = jax.ShapeDtypeStruct((num_tokens, hsize),
                                          jnp.float32,
                                          sharding=logits_sharding)
            token_ids = jax.ShapeDtypeStruct((num_tokens, ),
                                             jnp.int32,
                                             sharding=token_ids_sharding)
            self._run_compilation(
                f"worker{self.runner.rank} compute_and_gather_prompt_logprobs",
                compute_and_gather_prompt_logprobs,
                logits,
                token_ids,
                self.runner.model_config.max_logprobs,
                compile_only=True,
                num_tokens=num_tokens,
            )

        self._gather_logprobs_precompiled = True

    def _precompile_process_and_extend_logits(self) -> None:
        logger.info(
            "Compiling _process_and_extend_logits with different input shapes."
        )
        vocab_size = self.runner.vocab_size
        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                if num_reqs > num_logits:
                    continue

                logits_sharding = NamedSharding(self.runner.mesh,
                                                PartitionSpec())
                dp_sharding = NamedSharding(self.runner.mesh, PartitionSpec())

                target_logits = self._create_dummy_tensor(
                    (num_logits, vocab_size), jnp.float32, logits_sharding)

                processed_bonus_logits = self._create_dummy_tensor(
                    (num_reqs, vocab_size), jnp.float32, logits_sharding)

                draft_lengths = self._create_dummy_tensor(
                    (num_reqs, ), jnp.int32, dp_sharding)

                temperature = self._create_dummy_tensor(
                    (num_reqs, ), np.float32, dp_sharding)
                top_k = self._create_dummy_tensor((num_reqs, ), np.int32,
                                                  dp_sharding)
                top_p = self._create_dummy_tensor((num_reqs, ), np.float32,
                                                  dp_sharding)

                dummy_shape = (1, )  # logprobs=True
                _cache_collision_dummy = jnp.zeros(dummy_shape,
                                                   dtype=jnp.int32)
                _cache_collision_dummy = device_array(self.runner.mesh,
                                                      _cache_collision_dummy)

                sampling_metadata = TPUSupportedSamplingMetadata(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    _cache_collision_dummy=_cache_collision_dummy,
                    do_sampling=True,
                    logprobs=True,
                )

                spec_decode_metadata = SpecDecodeMetadata(
                    draft_lengths=draft_lengths,
                    target_logits_indices=self._create_dummy_tensor(
                        (num_logits, ), jnp.int32, dp_sharding),
                    bonus_logits_indices=self._create_dummy_tensor(
                        (num_reqs, ), jnp.int32, dp_sharding),
                    final_logits_indices=self._create_dummy_tensor(
                        (num_logits, ), jnp.int32, dp_sharding),
                )

                self._run_compilation(
                    f"worker{self.runner.rank} _process_and_extend_logits",
                    process_and_extend_logits,
                    self.runner.mesh,
                    target_logits,
                    processed_bonus_logits,
                    spec_decode_metadata,
                    sampling_metadata,
                    num_logits=num_logits,
                    num_reqs=num_reqs,
                )

    def _precompile_extend_logits_simple(self) -> None:
        logger.info(
            "Compiling _extend_logits_simple with different input shapes.")
        vocab_size = self.runner.vocab_size
        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                if num_reqs > num_logits:
                    continue

                attn_data_size = get_mesh_shape_product(
                    self.runner.mesh, ShardingAxisName.ATTN_DATA)
                if attn_data_size == 1:
                    logits_spec = PartitionSpec()
                else:
                    logits_spec = PartitionSpec(ShardingAxisName.ATTN_DATA,
                                                None)

                logits_sharding = NamedSharding(self.runner.mesh, logits_spec)

                target_logits = self._create_dummy_tensor(
                    (num_logits, vocab_size), jnp.bfloat16, logits_sharding)
                bonus_logits = self._create_dummy_tensor(
                    (num_reqs, vocab_size), jnp.bfloat16, logits_sharding)

                self._run_compilation(
                    f"worker{self.runner.rank} _extend_logits_simple",
                    extend_logits_simple,
                    target_logits,
                    bonus_logits,
                    self.runner.mesh,
                    num_logits=num_logits,
                    num_reqs=num_reqs,
                )

    def _precompile_speculative_decoding(self) -> None:
        logger.info(
            "Compiling speculative_decoding with different input shapes.")
        self._precompile_rejection_sampler()
        self._precompile_extract_last_sampled_tokens()
        self._precompile_extract_draft_token_ids()
        self._precompile_process_and_extend_logits()
        self._precompile_extend_logits_simple()
        self._precompile_select_from_array_spec_decode()
        if self.runner.speculative_config.method == "eagle3":
            self._precompile_eagle3_helpers()
        elif self.runner.speculative_config.method == "dflash":
            self._precompile_dflash_helpers()
        elif self.runner.speculative_config.method == "mtp":
            self._precompile_mtp_helpers()

    def _precompile_select_from_array_spec_decode(self) -> None:
        logger.info("Compiling select_from_array with different input shapes.")
        vocab_size = self.runner.vocab_size
        sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, None))
        indices_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))

        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                # Case 1: Select bonus logits (indices shape (num_reqs, ))
                array = self._create_dummy_tensor(
                    (num_logits, vocab_size), self.runner.model_config.dtype,
                    sharding)
                indices_bonus = self._create_dummy_tensor(
                    (num_reqs, ), jnp.int32, indices_sharding)
                self._run_compilation(
                    f"worker{self.runner.rank} select_bonus_logits",
                    self.runner._select_from_array_fn,
                    array,
                    indices_bonus,
                    self.runner.mesh,
                    num_logits=num_logits,
                    num_reqs=num_reqs,
                )

                # Case 2: Select target logits (indices shape (num_logits, ))
                indices_target = self._create_dummy_tensor(
                    (num_logits, ), jnp.int32, indices_sharding)
                self._run_compilation(
                    f"worker{self.runner.rank} select_target_logits",
                    self.runner._select_from_array_fn,
                    array,
                    indices_target,
                    self.runner.mesh,
                    num_logits=num_logits,
                )

    def _precompile_extract_draft_token_ids(self) -> None:
        logger.info(
            "Compiling extract_draft_token_ids with different input shapes.")
        data_parallel_attn_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))
        for num_tokens in self.runner.num_tokens_paddings:
            for num_logits in self.runner.num_logits_paddings:
                if self._should_skip_padding_combination(num_tokens,
                                                         num_logits,
                                                         only_equal=False):
                    continue
                input_ids = self._create_dummy_tensor(
                    (num_tokens, ),
                    jnp.int32,
                    sharding=data_parallel_attn_sharding)
                final_logits_indices = self._create_dummy_tensor(
                    (num_logits, ),
                    jnp.int32,
                    sharding=data_parallel_attn_sharding)
                target_logits_indices = self._create_dummy_tensor(
                    (num_logits, ),
                    jnp.int32,
                    sharding=data_parallel_attn_sharding)
                self._run_compilation(
                    f"worker{self.runner.rank} extract_draft_token_ids",
                    self.runner._extract_draft_token_ids,
                    self.runner,
                    input_ids,
                    final_logits_indices,
                    target_logits_indices,
                    num_tokens=num_tokens,
                    num_logits=num_logits,
                    warmup_handler=self._skip_self_arg_warmup_handler,
                )

    def _precompile_extract_last_sampled_tokens(self) -> None:
        logger.info(
            "Compiling extract_last_sampled_tokens with different input shapes."
        )
        vocab_size = self.runner.vocab_size
        num_speculative_tokens = (
            self.runner.speculative_config.num_speculative_tokens)
        max_num_reqs_per_dp_rank = (self.runner.max_num_reqs //
                                    self.runner.dp_size)

        data_parallel_attn_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA))

        # sampled_token_ids has shape (num_logits + num_reqs,) from the
        # rejection_sampler output — [main_tokens (num_logits),
        # bonus_tokens (num_reqs)].
        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                sampled_token_ids = self._create_dummy_tensor(
                    (num_logits + num_reqs, ),
                    jnp.int32,
                    sharding=data_parallel_attn_sharding)
                spec_decode_metadata = SpecDecodeMetadata(
                    draft_lengths=self._create_dummy_tensor(
                        (num_reqs, ),
                        jnp.int32,
                        sharding=data_parallel_attn_sharding),
                    target_logits_indices=self._create_dummy_tensor(
                        (num_logits, ),
                        jnp.int32,
                        sharding=data_parallel_attn_sharding),
                    bonus_logits_indices=self._create_dummy_tensor(
                        (num_reqs, ),
                        jnp.int32,
                        sharding=data_parallel_attn_sharding),
                    final_logits_indices=self._create_dummy_tensor(
                        (num_logits, ),
                        jnp.int32,
                        sharding=data_parallel_attn_sharding),
                )
                self._run_compilation(
                    f"worker{self.runner.rank} extract_last_sampled_tokens",
                    extract_last_sampled_tokens,
                    spec_decode_metadata,
                    sampled_token_ids,
                    num_speculative_tokens,
                    vocab_size,
                    max_num_reqs_per_dp_rank,
                    self.runner.mesh,
                    num_logits=num_logits,
                    num_reqs=num_reqs,
                )

    def _precompile_rejection_sampler(self) -> None:
        logger.info("Compiling rejection_sampler with different input shapes.")
        vocab_size = self.runner.vocab_size
        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                sharding = NamedSharding(
                    self.runner.mesh,
                    PartitionSpec(ShardingAxisName.MLP_DATA,
                                  ShardingAxisName.MLP_TENSOR))
                target_probs = self._create_dummy_tensor(
                    (num_logits, vocab_size), self.runner.model_config.dtype,
                    sharding)
                draft_token_ids = self._create_dummy_tensor((num_logits, ),
                                                            jnp.int32)
                num_draft_tokens = self._create_dummy_tensor((num_reqs, ),
                                                             jnp.int32)
                bonus_token_ids = self._create_dummy_tensor((num_reqs, ),
                                                            jnp.int32)

                for do_sampling in (False, True):
                    draft_probs = None
                    # Use a dummy tensor with a unique shape for each logprobs config.
                    # Currently logprobs=False for rejection_sampler.
                    logprobs_dummy = False
                    dummy_shape = (1 if logprobs_dummy else 2, )
                    _cache_collision_dummy = jnp.zeros(dummy_shape,
                                                       dtype=jnp.int32)
                    _cache_collision_dummy = device_array(
                        self.runner.mesh, _cache_collision_dummy)

                    if do_sampling:
                        compilation_name = "random_rejection_sampler"
                        temperature = self._create_dummy_tensor((num_reqs, ),
                                                                np.float32)
                        top_k = self._create_dummy_tensor((num_reqs, ),
                                                          np.int32)
                        top_p = self._create_dummy_tensor((num_reqs, ),
                                                          np.float32)
                        sampling_metadata = TPUSupportedSamplingMetadata(
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            _cache_collision_dummy=_cache_collision_dummy,
                            do_sampling=do_sampling)
                    else:
                        compilation_name = "greedy_rejection_sampler"
                        sampling_metadata = TPUSupportedSamplingMetadata(
                            _cache_collision_dummy=_cache_collision_dummy,
                            do_sampling=do_sampling)

                    self._run_compilation(
                        f"worker{self.runner.rank} {compilation_name}",
                        self.runner.rejection_sampler,
                        draft_token_ids,
                        num_draft_tokens,
                        draft_probs,
                        target_probs,
                        bonus_token_ids,
                        sampling_metadata,
                        self.runner.rng_params_for_sampling,
                        num_logits=num_logits,
                        num_reqs=num_reqs,
                        do_sampling=do_sampling,
                    )

    def _precompile_eagle3_helpers(self) -> None:
        logger.info(
            "Compiling eagle3 jitted helpers with different input shapes.")
        target_hidden_size = self.runner.model_config.get_hidden_size()
        draft_hidden_size = self.runner.speculative_config.draft_model_config.get_hidden_size(
        )
        dtype = self.runner.model_config.dtype
        dp_size = self.runner.dp_size

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = self.runner.input_batch.block_table[
            draft_kv_cache_group_id].get_cpu_tensor().reshape(-1)
        dp_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, ))
        block_tables = device_array(self.runner.mesh,
                                    block_tables,
                                    sharding=dp_sharding)

        seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                             jnp.int32, dp_sharding)
        # query_start_loc carries one start-of-loc entry per DP rank, matching
        # the runtime layout produced by `_prepare_inputs`.
        query_start_loc = self._create_dummy_tensor(
            (self.runner.max_num_reqs + dp_size, ), jnp.int32, dp_sharding)

        # request_distribution stores 3 counters per DP rank
        # (decode/decode/total), so the shape scales with dp_size.
        request_distribution = np.array([0, 0, 0] * dp_size, dtype=np.int32)
        request_distribution = device_array(self.runner.mesh,
                                            request_distribution,
                                            sharding=dp_sharding)
        # Dummy mamba_state_indices for spec-decode compile-cache pre-tracing.
        # Must match the ATTN_DATA sharding `_prepare_inputs_*` produces at
        # runtime — otherwise the draft model_fn cache misses and the
        # ForbidCompile guard inside `Eagle3Proposer.propose` raises. None for
        # pure-attention models (the common eagle3 case) so the field stays
        # absent end-to-end.
        if self.runner.kv_cache_config.has_mamba_layers:
            eagle3_mamba_state_indices = device_array(
                self.runner.mesh,
                np.zeros(self.runner.max_num_reqs, dtype=np.int32),
                sharding=dp_sharding)
        else:
            eagle3_mamba_state_indices = None

        num_reqs_dp = self._create_dummy_tensor((dp_size, ),
                                                jnp.int32,
                                                sharding=dp_sharding)
        last_token_indices = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
        for num_tokens in self.runner.num_tokens_paddings:
            for num_reqs in self.runner.attn_num_reqs_paddings:
                positions = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)
                attention_metadata = AttentionMetadata(
                    input_positions=positions,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=request_distribution,
                    mamba_state_indices=eagle3_mamba_state_indices,
                    padded_num_reqs=num_reqs,
                )

                def drafter_propose_warmup(_fn, _args, _call_kwargs):
                    new_args = (self.runner.kv_caches, ) + _args[1:]
                    kv_caches, draft_token_ids = self.runner.drafter.propose(
                        *new_args, **_call_kwargs)
                    self.runner.kv_caches = kv_caches
                    return draft_token_ids

                draft_hidden_states = self._create_dummy_tensor(
                    (num_tokens, draft_hidden_size), dtype,
                    NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.MLP_DATA,
                                      ShardingAxisName.MLP_TENSOR)))
                input_ids = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)
                self._run_compilation(
                    "drafter_propose",
                    self.runner.drafter.propose,
                    self.runner.kv_caches,
                    input_ids,
                    attention_metadata,
                    last_token_indices,
                    draft_hidden_states,
                    num_tokens=num_tokens,
                    warmup_handler=drafter_propose_warmup,
                )
                aux_hidden_states = [
                    self._create_dummy_tensor(
                        (num_tokens, target_hidden_size), jnp.bfloat16,
                        NamedSharding(
                            self.runner.mesh,
                            PartitionSpec(ShardingAxisName.ATTN_DATA, None))),
                    self._create_dummy_tensor(
                        (num_tokens, target_hidden_size), jnp.bfloat16,
                        NamedSharding(
                            self.runner.mesh,
                            PartitionSpec(ShardingAxisName.ATTN_DATA, None))),
                    self._create_dummy_tensor(
                        (num_tokens, target_hidden_size), jnp.bfloat16,
                        NamedSharding(
                            self.runner.mesh,
                            PartitionSpec(ShardingAxisName.ATTN_DATA, None))),
                ]
                last_sampled_token_id = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
                next_prompt_token_id = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ),
                    jnp.int32,
                    sharding=dp_sharding)
                is_in_prefill = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ),
                    jnp.int32,
                    sharding=dp_sharding)
                num_rejected_tokens = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)

                self._run_compilation(
                    "drafter_prepare_inputs",
                    self.runner.drafter.prepare_inputs,
                    attention_metadata,
                    input_ids,
                    aux_hidden_states,
                    last_sampled_token_id,
                    next_prompt_token_id,
                    is_in_prefill,
                    num_rejected_tokens,
                    num_reqs_dp,
                    num_tokens=num_tokens,
                )

    def _precompile_mtp_helpers(self) -> None:
        logger.info(
            "Compiling mtp jitted helpers with different input shapes.")
        target_hidden_size = self.runner.model_config.get_hidden_size()
        draft_hidden_size = self.runner.speculative_config.draft_model_config.get_hidden_size(
        )
        dtype = self.runner.model_config.dtype
        dp_size = self.runner.dp_size

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = self.runner.input_batch.block_table[
            draft_kv_cache_group_id].get_cpu_tensor().reshape(-1)
        dp_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.ATTN_DATA, ))
        block_tables = device_array(self.runner.mesh,
                                    block_tables,
                                    sharding=dp_sharding)

        seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                             jnp.int32, dp_sharding)
        # query_start_loc carries one start-of-loc entry per DP rank, matching
        # the runtime layout produced by `_prepare_inputs`.
        query_start_loc = self._create_dummy_tensor(
            (self.runner.max_num_reqs + dp_size, ), jnp.int32, dp_sharding)

        # request_distribution stores 3 counters per DP rank
        # (decode/decode/total), so the shape scales with dp_size.
        request_distribution = np.array([0, 0, 0] * dp_size, dtype=np.int32)
        request_distribution = device_array(self.runner.mesh,
                                            request_distribution,
                                            sharding=dp_sharding)
        # Dummy mamba_state_indices for spec-decode compile-cache pre-tracing.
        # Must match the ATTN_DATA sharding `_prepare_inputs_*` produces at
        # runtime — otherwise the draft model_fn cache misses and the
        # ForbidCompile guard inside `Eagle3Proposer.propose` raises. None for
        # pure-attention models (the common eagle3 case) so the field stays
        # absent end-to-end.
        if self.runner.kv_cache_config.has_mamba_layers:
            mamba_state_indices = device_array(self.runner.mesh,
                                               np.zeros(
                                                   self.runner.max_num_reqs,
                                                   dtype=np.int32),
                                               sharding=dp_sharding)
        else:
            mamba_state_indices = None

        num_reqs_dp = self._create_dummy_tensor((dp_size, ),
                                                jnp.int32,
                                                sharding=dp_sharding)
        last_token_indices = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
        for num_tokens in self.runner.num_tokens_paddings:
            for num_reqs in self.runner.attn_num_reqs_paddings:
                if self.runner.uses_mrope:
                    mrope_sharding = NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(None, ShardingAxisName.ATTN_DATA))
                    positions = self._create_dummy_tensor(
                        (3, num_tokens), jnp.int32, mrope_sharding)
                else:
                    positions = self._create_dummy_tensor(
                        (num_tokens, ), jnp.int32, dp_sharding)

                attention_metadata = AttentionMetadata(
                    input_positions=positions,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=request_distribution,
                    mamba_state_indices=mamba_state_indices,
                    padded_num_reqs=num_reqs,
                )

                def drafter_propose_warmup(_fn, _args, _call_kwargs):
                    new_args = (self.runner.kv_caches, ) + _args[1:]
                    kv_caches, draft_token_ids = self.runner.drafter.propose(
                        *new_args, **_call_kwargs)
                    self.runner.kv_caches = kv_caches
                    return draft_token_ids

                draft_hidden_states = self._create_dummy_tensor(
                    (num_tokens, draft_hidden_size), dtype,
                    NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.ATTN_DATA, None)))

                input_ids = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)
                self._run_compilation(
                    "drafter_propose",
                    self.runner.drafter.propose,
                    self.runner.kv_caches,
                    input_ids,
                    attention_metadata,
                    last_token_indices,
                    draft_hidden_states,
                    num_tokens=num_tokens,
                    warmup_handler=drafter_propose_warmup,
                )

                aux_hidden_states = (self._create_dummy_tensor(
                    (num_tokens, target_hidden_size), jnp.bfloat16,
                    NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(ShardingAxisName.ATTN_DATA, None))), )
                last_sampled_token_id = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
                next_prompt_token_id = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ),
                    jnp.int32,
                    sharding=dp_sharding)
                is_in_prefill = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ),
                    jnp.int32,
                    sharding=dp_sharding)
                num_rejected_tokens = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)

                self._run_compilation(
                    "drafter_prepare_inputs",
                    self.runner.drafter.prepare_inputs,
                    attention_metadata,
                    input_ids,
                    aux_hidden_states,
                    last_sampled_token_id,
                    next_prompt_token_id,
                    is_in_prefill,
                    num_rejected_tokens,
                    num_reqs_dp,
                    num_tokens=num_tokens,
                )

    def _precompile_dflash_helpers(self) -> None:
        logger.info("Compiling dflash jitted helpers.")
        target_hidden_size = self.runner.model_config.get_hidden_size()
        dtype = self.runner.model_config.dtype
        dp_size = self.runner.dp_size

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = self.runner.input_batch.block_table[
            draft_kv_cache_group_id].get_cpu_tensor().reshape(-1)
        dp_spec = PartitionSpec() if dp_size == 1 else PartitionSpec(
            ShardingAxisName.ATTN_DATA)
        dp_spec_2d = PartitionSpec(ShardingAxisName.ATTN_DATA, None)

        dp_sharding = NamedSharding(self.runner.mesh, dp_spec)
        block_tables = device_array(self.runner.mesh,
                                    block_tables,
                                    sharding=dp_sharding)

        seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                             jnp.int32, dp_sharding)
        query_start_loc = self._create_dummy_tensor(
            (self.runner.max_num_reqs + dp_size, ), jnp.int32, dp_sharding)

        request_distribution = np.array([0, 0, 0] * dp_size, dtype=np.int32)
        request_distribution = device_array(self.runner.mesh,
                                            request_distribution,
                                            sharding=dp_sharding)

        if self.runner.kv_cache_config.has_mamba_layers:
            dflash_mamba_state_indices = device_array(
                self.runner.mesh,
                np.zeros(self.runner.max_num_reqs, dtype=np.int32),
                sharding=dp_sharding)
        else:
            dflash_mamba_state_indices = None

        num_reqs_dp = self._create_dummy_tensor((dp_size, ),
                                                jnp.int32,
                                                sharding=dp_sharding)
        last_token_indices = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)

        # Get number of auxiliary layers configured for dflash target features
        hf_config = self.runner.speculative_config.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", {})
        target_layer_ids = dflash_config.get("target_layer_ids", None)
        if target_layer_ids is not None:
            num_aux_states = len(target_layer_ids)
        else:
            num_aux_states = getattr(hf_config, "num_target_layers",
                                     hf_config.num_hidden_layers)

        block_size = self.runner.drafter.block_size

        # -------------------------------------------------------------
        # Part 1: Compile drafter_propose (loops over batch sizes)
        # -------------------------------------------------------------
        for num_tokens in self.runner.num_tokens_paddings:
            for num_reqs in self.runner.attn_num_reqs_paddings:
                num_tokens_propose = self.runner.max_num_reqs * block_size
                input_ids_propose = self._create_dummy_tensor(
                    (num_tokens_propose, ), jnp.int32, dp_sharding)
                positions_propose = self._create_dummy_tensor(
                    (num_tokens_propose, ), jnp.int32, dp_sharding)
                attention_metadata_propose = AttentionMetadata(
                    input_positions=positions_propose,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=request_distribution,
                    mamba_state_indices=dflash_mamba_state_indices,
                    padded_num_reqs=num_reqs,
                )

                def drafter_propose_warmup(_fn,
                                           _args,
                                           _call_kwargs,
                                           num_reqs=num_reqs):
                    new_args = (self.runner.kv_caches, ) + _args[1:]
                    logger.info("Warmup drafter_propose: num_reqs=%d",
                                num_reqs)
                    kv_caches, draft_token_ids = self.runner.drafter.propose(
                        *new_args, **_call_kwargs)
                    self.runner.kv_caches = kv_caches
                    return draft_token_ids

                draft_hidden_size = hf_config.hidden_size
                target_hidden_propose = self._create_dummy_tensor(
                    (num_tokens, draft_hidden_size), dtype,
                    NamedSharding(
                        self.runner.mesh,
                        PartitionSpec(
                            None
                            if dp_size == 1 else ShardingAxisName.MLP_DATA,
                            ShardingAxisName.MLP_TENSOR)))
                new_query_start_loc = self._create_dummy_tensor(
                    (self.runner.max_num_reqs + dp_size, ), jnp.int32,
                    dp_sharding)
                new_input_positions = self._create_dummy_tensor(
                    (num_tokens, ), jnp.int32, dp_sharding)
                target_hidden_states_propose = (target_hidden_propose,
                                                new_query_start_loc,
                                                new_input_positions)

                self._run_compilation(
                    f"drafter_propose (num_tokens={num_tokens})",
                    self.runner.drafter.propose,
                    self.runner.kv_caches,
                    input_ids_propose,
                    attention_metadata_propose,
                    last_token_indices,
                    target_hidden_states_propose,
                    num_reqs=num_reqs,
                    warmup_handler=drafter_propose_warmup,
                )

        # -------------------------------------------------------------
        # Part 2: Compile drafter_prepare_inputs (loops over target shapes)
        # -------------------------------------------------------------
        for num_tokens in self.runner.num_tokens_paddings:
            for num_reqs in self.runner.attn_num_reqs_paddings:
                positions = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)
                attention_metadata = AttentionMetadata(
                    input_positions=positions,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=request_distribution,
                    mamba_state_indices=dflash_mamba_state_indices,
                    padded_num_reqs=num_reqs,
                )

                aux_hidden_states = [
                    self._create_dummy_tensor(
                        (num_tokens, target_hidden_size), jnp.bfloat16,
                        NamedSharding(self.runner.mesh, dp_spec_2d))
                    for _ in range(num_aux_states)
                ]
                last_sampled_token_id = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
                next_prompt_token_id = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ),
                    jnp.int32,
                    sharding=dp_sharding)
                is_in_prefill = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ),
                    jnp.int32,
                    sharding=dp_sharding)
                num_rejected_tokens = self._create_dummy_tensor(
                    (self.runner.max_num_reqs, ), jnp.int32, dp_sharding)
                input_ids = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32, dp_sharding)

                self._run_compilation(
                    "drafter_prepare_inputs",
                    self.runner.drafter.prepare_inputs,
                    attention_metadata,
                    input_ids,
                    aux_hidden_states,
                    last_sampled_token_id,
                    next_prompt_token_id,
                    is_in_prefill,
                    num_rejected_tokens,
                    num_reqs_dp,
                    num_tokens=num_tokens,
                )

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
        if self.runner.vllm_config.sharding_config.total_dp_size > 1:
            logger.warning(
                "Structured decoding precompilation skipped since structured decoding is not supported with DP."
            )
            return
        for num_reqs in self.runner.num_reqs_paddings:
            dummy_logits = self._create_dummy_tensor(
                (num_reqs, self.runner.vocab_size), jnp.bfloat16)
            dummy_require_struct_decoding = self.runner.require_structured_out_cpu[:
                                                                                   num_reqs]
            dummy_grammar_bitmask = self.runner.grammar_bitmask_cpu[:num_reqs]

            (dummy_logits, dummy_require_struct_decoding,
             dummy_grammar_bitmask, arange) = device_array(
                 self.runner.mesh,
                 (dummy_logits, dummy_require_struct_decoding,
                  dummy_grammar_bitmask, self.runner.structured_decode_arange))

            self._run_compilation(
                "structured_decode",
                self.runner.structured_decoding_manager.structured_decode_fn,
                self.runner.structured_decoding_manager,
                dummy_require_struct_decoding,
                dummy_grammar_bitmask,
                dummy_logits,
                arange,
                num_reqs=num_reqs,
                warmup_handler=self._skip_self_arg_warmup_handler,
            )

    def _precompile_continue_decode(self) -> None:
        logger.info("Precompiling continue_decode loop.")
        dp_size = self.runner.vllm_config.sharding_config.total_dp_size
        dp_spec = PartitionSpec(ShardingAxisName.ATTN_DATA, )
        dp_sharding = NamedSharding(self.runner.mesh, dp_spec)

        user_max_decode_steps = self.runner.vllm_config.additional_config.get(
            "max_decode_steps", DEFAULT_MAX_DECODE_STEPS)

        # We only need to compile once for user_max_decode_steps

        # We also need to construct TPUSupportedSamplingMetadata
        # For greedy decoding, we can use empty parameters
        _cache_collision_dummy = jnp.zeros((2, ), dtype=jnp.int32)
        _cache_collision_dummy = device_array(self.runner.mesh,
                                              _cache_collision_dummy)
        sampling_metadata = TPUSupportedSamplingMetadata(
            temperature=None,
            top_k=None,
            top_p=None,
            _cache_collision_dummy=_cache_collision_dummy,
            do_sampling=False,
            logprobs=False)

        for num_reqs in self.runner.num_reqs_paddings:
            init_tokens = self._create_dummy_tensor((num_reqs, ), jnp.int32,
                                                    dp_sharding)
            active_mask = self._create_dummy_tensor((num_reqs, ), jnp.bool_,
                                                    dp_sharding)

            seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                                 jnp.int32, dp_sharding)
            query_start_loc = self._create_dummy_tensor(
                (self.runner.max_num_reqs + dp_size, ), jnp.int32, dp_sharding)

            request_distribution = np.array([0, 0, 0] * dp_size,
                                            dtype=np.int32)
            request_distribution = device_array(self.runner.mesh,
                                                request_distribution,
                                                sharding=dp_sharding)

            if self.runner.kv_cache_config.has_mamba_layers:
                mamba_state_indices = device_array(
                    self.runner.mesh,
                    np.zeros(self.runner.max_num_reqs, dtype=np.int32),
                    sharding=dp_sharding)
            else:
                mamba_state_indices = None

            def build_block_table(kv_cache_gid: int) -> jax.Array:
                block_table_obj = self.runner.input_batch.block_table[
                    kv_cache_gid]
                shape = (self.runner.max_num_reqs,
                         block_table_obj.max_num_blocks_per_req)
                block_tables = np.zeros(shape, dtype=np.int32)
                block_tables = block_tables.reshape(-1)
                block_tables = device_array(self.runner.mesh,
                                            block_tables,
                                            sharding=dp_sharding)
                return block_tables

            if len(self.runner.kv_cache_config.kv_cache_groups) <= 1:
                no_kv_cache = len(
                    self.runner.kv_cache_config.kv_cache_groups) == 0
                block_tables = build_block_table(
                    0) if not no_kv_cache else None
                attn_metadata = AttentionMetadata(
                    input_positions=init_tokens,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=request_distribution,
                    mamba_state_indices=mamba_state_indices,
                    padded_num_reqs=num_reqs,
                )
            else:
                attn_metadata = {
                    name:
                    AttentionMetadata(
                        input_positions=init_tokens,
                        block_tables=build_block_table(gid),
                        seq_lens=seq_lens,
                        query_start_loc=query_start_loc,
                        request_distribution=request_distribution,
                        mamba_state_indices=mamba_state_indices,
                        padded_num_reqs=num_reqs,
                    )
                    for gid, kv_cache_group in enumerate(
                        self.runner.kv_cache_config.kv_cache_groups)
                    for name in kv_cache_group.layer_names
                }

            init_state = TpuSamplingState(
                current_tokens=init_tokens,
                active_mask=active_mask,
                attn_metadata=attn_metadata,
                step_counter=self.runner.zero_array,
            )

            lora_metadata = self.runner.lora_utils.extract_lora_metadata()

            # Compile once for the max steps using JAX array for dynamic bound
            max_decode_steps_arr = jnp.array(user_max_decode_steps,
                                             dtype=jnp.int32)

            def continue_decode_wrapper(
                state,
                model_fn,
                compute_logits_fn,
                sample_fn,
                mesh,
                sampling_metadata,
                init_state,
                kv_caches,
                max_decode_steps,
                static_max_decode_steps,
                eos_token_id,
                padding_token_id,
                rng,
                layer_name_to_kvcache_index,
                lora_metadata,
                is_first_rank,
                is_last_rank,
                dp_size,
                collect_expert_indices,
                continue_decode_eos_check_interval,
            ):
                (generated_tokens, final_kv_caches, final_state, final_rng,
                 all_expert_indices, logprobs_tensors) = continue_decode(
                     state=state,
                     model_fn=model_fn,
                     compute_logits_fn=compute_logits_fn,
                     sample_fn=sample_fn,
                     mesh=mesh,
                     sampling_metadata=sampling_metadata,
                     init_state=init_state,
                     kv_caches=kv_caches,
                     max_decode_steps=max_decode_steps,
                     static_max_decode_steps=static_max_decode_steps,
                     eos_token_id=eos_token_id,
                     padding_token_id=padding_token_id,
                     rng=rng,
                     layer_name_to_kvcache_index=layer_name_to_kvcache_index,
                     lora_metadata=lora_metadata,
                     is_first_rank=is_first_rank,
                     is_last_rank=is_last_rank,
                     dp_size=dp_size,
                     collect_expert_indices=collect_expert_indices,
                     max_logprobs=self.runner.model_config.max_logprobs,
                     logprobs_mode=self.runner.model_config.logprobs_mode,
                     continue_decode_eos_check_interval=
                     continue_decode_eos_check_interval,
                 )
                self.runner.kv_caches = final_kv_caches
                return generated_tokens

            def continue_decode_warmup(_fn, _args, _call_kwargs):
                new_args = list(_args)
                new_args[7] = self.runner.kv_caches
                return _fn(*new_args, **_call_kwargs)

            self._run_compilation(
                f"worker{self.runner.rank} continue_decode_steps_{user_max_decode_steps}_reqs_{num_reqs}",
                continue_decode_wrapper,
                self.runner.state_leaves,
                getattr(self.runner.model, "step_fn_no_options",
                        self.runner.model_fn),
                self.runner.compute_logits_fn,
                sample,
                self.runner.mesh,
                sampling_metadata,
                init_state,
                self.runner.kv_caches,
                max_decode_steps_arr,
                user_max_decode_steps,
                self.runner.eos_token_id,
                self.runner.pad_token_id,
                self.runner.rng_params_for_sampling,
                tuple(self.runner.layer_name_to_kvcache_index.items()),
                lora_metadata,
                self.runner.is_first_rank,
                self.runner.is_last_rank,
                self.runner.dp_size,
                getattr(self.runner.vllm_config.model_config,
                        "enable_return_routed_experts", False),
                self.runner.continue_decode_eos_check_interval,
                warmup_handler=continue_decode_warmup,
            )
