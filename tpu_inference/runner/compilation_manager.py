import os
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import vllm.envs as envs
from jax.sharding import NamedSharding, PartitionSpec
from vllm.utils import cdiv

from tpu_inference.core.disagg_utils import is_disagg_enabled
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.sample.sampling import sample
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_inference.logger import init_logger
from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from tpu_inference.runner.tpu_jax_runner import TPUModelRunner

logger = init_logger(__name__)

# Constants for block bucketing in disaggregated utilities
BLOCK_BUCKETS = [1, 2, 4, 8, 16, 32, 64]


class CompilationManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            logger.info("Enabling JAX compile cache.")
            jax.config.update("jax_compilation_cache_dir",
                              envs.VLLM_XLA_CACHE_PATH)

    def _create_dummy_tensor(self,
                             shape: Tuple[int, ...],
                             dtype: Any,
                             sharding: Optional[NamedSharding] = None) -> Any:
        """Helper to create dummy tensors for precompilation."""
        tensor = jnp.ones(shape, dtype=dtype)
        if sharding:
            return device_array(self.runner.mesh, tensor, sharding=sharding)
        return device_array(self.runner.mesh, tensor)

    def _should_skip_padding_combination(self, outer_val: int, inner_val: int,
                                         only_equal: bool) -> bool:
        """Helper to determine if we should skip this padding combination."""
        if only_equal:
            return inner_val != outer_val
        return inner_val > outer_val

    def _run_compilation(self, name: str, fn: Callable, *args,
                         **kwargs) -> None:
        logger.info(f"Precompile {name} --> {kwargs}")
        start = time.perf_counter()
        result = fn(*args)
        if result is not None:
            if isinstance(result, tuple):
                for r in result:
                    r.block_until_ready()
            else:
                result.block_until_ready()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)

    def capture_model(self) -> None:
        if os.getenv("SKIP_JAX_PRECOMPILE", False):
            return
        logger.info("Precompile all the subgraphs with possible input shapes.")

        with self.runner.maybe_setup_dummy_loras(self.runner.lora_config):
            self._precompile_backbone_text_only()
            if self.runner.is_multimodal_model:
                self._precompile_backbone_with_inputs_embeds()
            self._precompile_select_from_array()
            self._precompile_compute_logits()
            self._precompile_disagg_utils()
            self._precompile_sampling()
            self._precompile_gather_logprobs()
            self._precompile_structured_decoding()
            if self.runner.speculative_config:
                self._precompile_speculative_decoding()

    def _precompile_backbone_helper(self, name, *, input_ids, positions,
                                    inputs_embeds) -> None:
        num_tokens = None
        if input_ids is not None:
            num_tokens = input_ids.shape[0]
        elif inputs_embeds is not None:
            num_tokens = inputs_embeds.shape[0]
        assert num_tokens is not None

        # Keep existing pattern for complex array operations
        block_tables = self.runner.block_table_cpu[:self.runner.max_num_reqs]
        block_tables = block_tables.reshape(-1)
        block_tables = device_array(self.runner.mesh, block_tables)

        seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                             jnp.int32)
        query_start_loc = self._create_dummy_tensor(
            (self.runner.max_num_reqs + 1, ), jnp.int32)

        # Keep existing pattern for specific value arrays
        request_distribution = np.array([0, 0, 0], dtype=np.int32)
        request_distribution = device_array(self.runner.mesh,
                                            request_distribution)

        attention_metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        def model_fn_wrapper(
            state,
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kvcache_index,
            lora_metadata,
        ):
            kv_caches, hidden_states, aux_hidden_states = self.runner.model_fn(
                state, kv_caches, input_ids, attention_metadata, inputs_embeds,
                layer_name_to_kvcache_index, lora_metadata)
            self.runner.kv_caches = kv_caches
            return hidden_states

        with self.runner.maybe_select_dummy_loras(
                self.runner.lora_config, np.array([num_tokens],
                                                  dtype=np.int32)):
            lora_metadata = self.runner.lora_utils.extract_lora_metadata()
            self._run_compilation(
                name,
                model_fn_wrapper,
                self.runner.state,
                self.runner.kv_caches,
                input_ids,
                attention_metadata,
                inputs_embeds,
                tuple(self.runner.layer_name_to_kvcache_index.items()),
                lora_metadata,
                num_tokens=num_tokens,
            )

    def _precompile_backbone_text_only(self) -> None:
        for num_tokens in self.runner.num_tokens_paddings:
            input_ids = self._create_dummy_tensor((num_tokens, ), jnp.int32)
            positions = self._create_dummy_tensor((num_tokens, ), jnp.int32)
            self._precompile_backbone_helper("backbone",
                                             input_ids=input_ids,
                                             positions=positions,
                                             inputs_embeds=None)

    def _precompile_backbone_with_inputs_embeds(self) -> None:
        hidden_size = self.runner.model_config.get_hidden_size()
        dtype = self.runner.model_config.dtype
        for num_tokens in self.runner.num_tokens_paddings:
            inputs_embeds = self._create_dummy_tensor(
                (num_tokens, hidden_size), dtype)
            if self.runner.uses_mrope:
                positions = self._create_dummy_tensor((3, num_tokens),
                                                      jnp.int32)
            else:
                positions = self._create_dummy_tensor((num_tokens, ),
                                                      jnp.int32)
            self._precompile_backbone_helper("backbone with embeds",
                                             input_ids=None,
                                             positions=positions,
                                             inputs_embeds=inputs_embeds)

    def _precompile_select_from_array_helper(
        self,
        name: str,
        source_paddings: List[int],
        indices_paddings: List[int],
        hidden_dim: int,
        sharding: Optional[NamedSharding] = None,
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

                input_tensor = self._create_dummy_tensor(
                    (array_size, hidden_dim), jnp.bfloat16, sharding)
                indices_to_select = self._create_dummy_tensor(
                    (indices_count, ), jnp.int32)

                self._run_compilation(
                    f"select_from_array [{name}]",
                    self.runner._select_from_array_fn, input_tensor,
                    indices_to_select, **{
                        "array_size": array_size,
                        "index_size": indices_count
                    })

    def _precompile_select_from_array(self) -> None:
        logger.info("Compiling select_from_array with different input shapes.")
        hsize = self.runner.model_config.get_hidden_size()

        if self.runner.speculative_config:
            index_paddings = self.runner.num_logits_paddings
        else:
            index_paddings = self.runner.num_reqs_paddings

        self._precompile_select_from_array_helper(
            name="select all logits",
            source_paddings=self.runner.num_tokens_paddings,
            indices_paddings=index_paddings,
            hidden_dim=hsize,
            sharding=NamedSharding(self.runner.mesh, PartitionSpec(None,
                                                                   None)),
        )

        if self.runner.speculative_config:
            vocab_size = self.runner.model_config.get_vocab_size()
            self._precompile_select_from_array_helper(
                name="select bonus tokens for spec decoding",
                source_paddings=self.runner.num_logits_paddings,
                indices_paddings=self.runner.num_reqs_paddings,
                hidden_dim=vocab_size,
                sharding=NamedSharding(self.runner.mesh,
                                       PartitionSpec(None, "model")),
            )
            self._precompile_select_from_array_helper(
                name="select target tokens for spec decoding",
                source_paddings=self.runner.num_logits_paddings,
                indices_paddings=self.runner.num_logits_paddings,
                hidden_dim=vocab_size,
                sharding=NamedSharding(self.runner.mesh,
                                       PartitionSpec(None, "model")),
                only_equal_paddings=True,
            )

            self._precompile_select_from_array_helper(
                name="select hidden states for eagle-3",
                source_paddings=self.runner.num_tokens_paddings,
                indices_paddings=[self.runner.max_num_reqs],
                hidden_dim=hsize,
                sharding=NamedSharding(self.runner.mesh,
                                       PartitionSpec(None, None)),
                check_should_skip_padding=False,
            )

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        hsize = self.runner.model_config.get_hidden_size()
        leading_shape = self.runner.num_reqs_paddings if not self.runner.speculative_config else self.runner.num_logits_paddings
        for num_reqs in leading_shape:
            hidden_states = self._create_dummy_tensor((num_reqs, hsize),
                                                      jnp.bfloat16)
            with self.runner.maybe_select_dummy_loras(
                    self.runner.lora_config,
                    np.array([num_reqs], dtype=np.int32)):
                lora_metadata = self.runner.lora_utils.extract_lora_metadata()
                self._run_compilation(
                    "compute_logits",
                    self.runner.compute_logits_fn,
                    self.runner.state,
                    hidden_states,
                    lora_metadata,
                    num_reqs=num_reqs,
                )

    def _precompile_sampling(self) -> None:
        logger.info("Compiling sampling with different input shapes.")
        hsize = self.runner.model_config.get_vocab_size()
        for num_reqs in self.runner.num_reqs_paddings:
            sharding = NamedSharding(self.runner.mesh,
                                     PartitionSpec(None, "model"))
            logits = self._create_dummy_tensor((num_reqs, hsize), jnp.bfloat16,
                                               sharding)
            for do_sampling in (True, False):
                if do_sampling:
                    temperature = np.full((num_reqs, ), 0.7, dtype=np.float32)
                    top_k = np.full((num_reqs, ), 20, dtype=np.int32)
                    top_p = np.full((num_reqs, ), 0.8, dtype=np.float32)
                    (temperature, top_k,
                     top_p) = device_array(self.runner.mesh,
                                           (temperature, top_k, top_p))
                else:
                    temperature = None
                    top_k = None
                    top_p = None

                sampling_metadata = TPUSupportedSamplingMetadata(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sampling=do_sampling,
                )
                self._run_compilation(
                    "sample",
                    sample,
                    self.runner.rng_params_for_sampling,
                    self.runner.mesh,
                    logits,
                    sampling_metadata,
                    num_reqs=num_reqs,
                    do_sampling=do_sampling,
                )

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
            self.runner.kv_caches = self.runner.kv_cache_manager._jitted_insert_continuous_kv_cache(
                block_size,
                self.runner.kv_caches,
                kv_cache_slices,
                block_numbers[0],
            )
            for layer_cache in self.runner.kv_caches:
                layer_cache.block_until_ready()

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        hsize = self.runner.model_config.get_vocab_size()
        for num_reqs in self.runner.num_reqs_paddings:
            logits = self._create_dummy_tensor((num_reqs, hsize), jnp.bfloat16)
            token_ids = self._create_dummy_tensor((num_reqs, ), jnp.int32)
            self._run_compilation(
                "gather_logprobs",
                self.runner._compute_and_gather_logprobs,
                logits,
                token_ids,
                self.runner.model_config.max_logprobs,
                num_reqs=num_reqs,
            )

    def _precompile_speculative_decoding(self) -> None:
        logger.info(
            "Compiling speculative_decoding with different input shapes.")
        self._precompile_rejection_sampler()
        if self.runner.speculative_config.method == "eagle3":
            self._precompile_eagle3_helpers()

    def _precompile_rejection_sampler(self) -> None:
        logger.info("Compiling rejection_sampler with different input shapes.")
        vocab_size = self.runner.model_config.get_vocab_size()
        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                sharding = NamedSharding(self.runner.mesh,
                                         PartitionSpec(None, "model"))
                target_probs = self._create_dummy_tensor(
                    (num_logits, vocab_size), jnp.bfloat16, sharding)
                draft_token_ids = self._create_dummy_tensor((num_logits, ),
                                                            jnp.int32)
                num_draft_tokens = self._create_dummy_tensor((num_reqs, ),
                                                             jnp.int32)
                bonus_token_ids = self._create_dummy_tensor((num_reqs, ),
                                                            jnp.int32)

                for do_sampling in (False, True):
                    draft_probs = None
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
                            do_sampling=do_sampling)
                    else:
                        compilation_name = "greedy_rejection_sampler"
                        sampling_metadata = TPUSupportedSamplingMetadata(
                            do_sampling=do_sampling)

                    self._run_compilation(
                        compilation_name,
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
        hidden_size = self.runner.model_config.get_hidden_size()
        draft_hidden_size = self.runner.vllm_config.speculative_config.draft_model_config.hf_config.hidden_size * 3
        dtype = self.runner.model_config.dtype

        num_kv_cache_groups = len(self.runner.kv_cache_config.kv_cache_groups)
        draft_kv_cache_group_id = num_kv_cache_groups - 1
        block_tables = jnp.ones(
            (self.runner.max_num_reqs,
             cdiv(self.runner.max_model_len, self.runner.block_size)),
            jnp.int32)
        self._run_compilation(
            "eagle3_reshape_block",
            self.runner.drafter._reshape_block_tables,
            block_tables,
        )
        block_tables = self.runner.input_batch.block_table[
            draft_kv_cache_group_id].get_device_tensor().reshape(-1)
        block_tables_loop = jax.device_put(
            block_tables, NamedSharding(self.runner.mesh,
                                        PartitionSpec(None, )))

        selected_positions = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32)
        seq_lens = self._create_dummy_tensor((self.runner.max_num_reqs, ),
                                             jnp.int32)
        query_start_loc = self._create_dummy_tensor(
            (self.runner.max_num_reqs + 1, ), jnp.int32)
        self._run_compilation("_prepare_input_loop for the first loop",
                              self.runner.drafter._prepare_input_loop,
                              selected_positions, seq_lens, block_tables)
        self._run_compilation("_prepare_input_loop for the subsequent loops",
                              self.runner.drafter._prepare_input_loop,
                              selected_positions, seq_lens, block_tables_loop)

        request_distribution = np.array([0, 0, 0], dtype=np.int32)
        request_distribution = device_array(self.runner.mesh,
                                            request_distribution)

        for num_reqs_padding in self.runner.num_reqs_paddings:
            logits = self._create_dummy_tensor(
                (num_reqs_padding, self.runner.vocab_size), jnp.bfloat16,
                NamedSharding(self.runner.mesh, PartitionSpec(None, "model")))
            self._run_compilation(
                "_get_draft_token_ids",
                self.runner.drafter._get_draft_token_ids,
                logits,
                num_reqs=num_reqs_padding,
            )

            for i in range(1, self.runner.drafter.num_speculative_tokens + 1):
                draft_token_ids_list = [
                    self._create_dummy_tensor(
                        (num_reqs_padding, ), jnp.int32,
                        NamedSharding(self.runner.mesh, PartitionSpec()))
                    for _ in range(i)
                ]
                self._run_compilation(
                    "_stack_draft_token_ids",
                    self.runner.drafter._stack_draft_token_ids,
                    draft_token_ids_list,
                    num_reqs=num_reqs_padding,
                    draft_token_ids_list_length=len(draft_token_ids_list))

        for num_logits in self.runner.num_logits_paddings:
            hidden_states = self._create_dummy_tensor(
                (num_logits, hidden_size), jnp.bfloat16)
            self._run_compilation(
                "drafter_compute_logits",
                self.runner.drafter.compute_logits_fn,
                self.runner.drafter.state,
                hidden_states,
                None,
                num_logits=num_logits,
            )

        position_indices = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32)
        next_token_ids = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32)
        input_ids_loop = self._create_dummy_tensor(
            (self.runner.max_num_reqs, ), jnp.int32,
            NamedSharding(self.runner.mesh, PartitionSpec()))
        target_hidden_state_loop = self._create_dummy_tensor(
            (self.runner.max_num_reqs, hidden_size), dtype,
            NamedSharding(self.runner.mesh, PartitionSpec(None, None)))
        for num_tokens in self.runner.num_tokens_paddings:
            positions = self._create_dummy_tensor((num_tokens, ), jnp.int32)
            self._run_compilation(
                "select_from_array [select input positions for eagle3]",
                self.runner._select_from_array_fn,
                positions,
                position_indices,
                num_tokens=num_tokens)

            aux_hidden_states = [
                self._create_dummy_tensor((num_tokens, hidden_size), dtype),
                self._create_dummy_tensor((num_tokens, hidden_size), dtype),
                self._create_dummy_tensor((num_tokens, hidden_size), dtype),
            ]
            self._run_compilation(
                "eagle3_concate_hidden_states",
                self.runner.drafter._concate_hidden_states,
                aux_hidden_states,
                num_tokens=num_tokens,
            )

            input_ids = self._create_dummy_tensor((num_tokens, ), jnp.int32)
            aux_hidden_states = [
                self._create_dummy_tensor(
                    (num_tokens, hidden_size), jnp.bfloat16,
                    NamedSharding(self.runner.mesh, PartitionSpec(None,
                                                                  None))),
                self._create_dummy_tensor(
                    (num_tokens, hidden_size), jnp.bfloat16,
                    NamedSharding(self.runner.mesh, PartitionSpec(None,
                                                                  None))),
                self._create_dummy_tensor(
                    (num_tokens, hidden_size), jnp.bfloat16,
                    NamedSharding(self.runner.mesh, PartitionSpec(None,
                                                                  None))),
            ]
            for num_indices in self.runner.num_tokens_paddings:
                indices = jnp.ones((num_indices, ), dtype=jnp.int32)
                self._run_compilation(
                    "select_from_array [select input ids for eagle3]",
                    self.runner._select_from_array_fn,
                    input_ids,
                    indices,
                    num_tokens=num_tokens,
                    num_indices=num_indices)
                self._run_compilation(
                    "select_from_array [select hidden states for eagle3]",
                    self.runner.drafter._select_target_hidden_states,
                    aux_hidden_states, indices)

            attention_metadata = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )
            target_hidden_states = self._create_dummy_tensor(
                (num_tokens, hidden_size), dtype,
                NamedSharding(self.runner.mesh, PartitionSpec(None, "model")))

            def draft_model_fn_wrapper(
                state,
                kv_caches,
                input_ids,
                target_hidden_states,
                attention_metadata,
            ):
                kv_caches, hidden_states, _ = self.runner.drafter.model_fn(
                    state, kv_caches, input_ids, target_hidden_states,
                    attention_metadata)
                self.runner.kv_caches = kv_caches
                return hidden_states

            self._run_compilation(
                "draft_model_fn",
                draft_model_fn_wrapper,
                self.runner.drafter.state,
                self.runner.kv_caches,
                input_ids,
                target_hidden_states,
                attention_metadata,
                num_tokens=num_tokens,
            )

            attention_metadata.query_start_loc = jax.device_put(
                attention_metadata.query_start_loc,
                NamedSharding(self.runner.mesh, PartitionSpec()))
            attention_metadata.block_tables = block_tables_loop
            attention_metadata.input_positions = self._create_dummy_tensor(
                (self.runner.max_num_reqs, ), jnp.int32)
            self._run_compilation(
                "draft_model_fn in a loop",
                draft_model_fn_wrapper,
                self.runner.drafter.state,
                self.runner.kv_caches,
                input_ids_loop,
                target_hidden_state_loop,
                attention_metadata,
                num_tokens=num_tokens,
            )

            target_hidden_states = self._create_dummy_tensor(
                (num_tokens, draft_hidden_size), dtype)
            self._run_compilation(
                "draft_model_combine_hidden_states_fn",
                self.runner.drafter.combine_hidden_states_fn,
                self.runner.drafter.state,
                target_hidden_states,
                num_tokens=num_tokens,
            )

            target_token_ids = self._create_dummy_tensor((num_tokens, ),
                                                         jnp.int32)
            self._run_compilation(
                "_prepare_input_ids",
                self.runner.drafter._prepare_input_ids,
                query_start_loc,
                target_token_ids,
                next_token_ids,
                self.runner.input_batch.num_reqs,
            )

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
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
                dummy_require_struct_decoding,
                dummy_grammar_bitmask,
                dummy_logits,
                arange,
                num_reqs=num_reqs,
            )
