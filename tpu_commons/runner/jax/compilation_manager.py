import os
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from tpu_commons.logger import init_logger
from tpu_commons.models.jax.attention_metadata import AttentionMetadata
from tpu_commons.models.jax.layers.sample.sampling import sample
from tpu_commons.models.jax.layers.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tpu_commons.utils import device_array

if TYPE_CHECKING:
    from tpu_commons.runner.jax.tpu_jax_runner import TPUModelRunner

logger = init_logger(__name__)


class CompilationManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

    def capture_model(self) -> None:
        if os.getenv("SKIP_JAX_PRECOMPILE", False):
            return
        logger.info("Precompile all the subgraphs with possible input shapes.")

        if self.runner.is_multimodal_model:
            logger.info("[TEMP] skip precompiling for multi-modal models")
            return

        self._precompile_backbone()
        self._precompile_select_from_array()
        self._precompile_compute_logits()
        self._precompile_disagg_utils()
        self._precompile_sampling()
        self._precompile_gather_logprobs()
        self._precompile_structured_decoding()
        if self.runner.speculative_config:
            self._precompile_rejection_sampler()

    def _precompile_backbone(self) -> None:
        for num_tokens in self.runner.num_tokens_paddings:
            input_ids = np.ones((num_tokens, ), dtype=np.int32)
            positions = np.ones((num_tokens, ), dtype=np.int32)
            block_tables = self.runner.block_table_cpu[:self.runner.
                                                       max_num_reqs]
            seq_lens = np.ones((self.runner.max_num_reqs, ), dtype=np.int32)
            query_start_loc = np.ones((self.runner.max_num_reqs + 1, ),
                                      dtype=np.int32)
            request_distribution = np.array([0, 0, 0], dtype=np.int32)

            block_tables = block_tables.reshape(-1)

            (input_ids, positions, block_tables, query_start_loc,
             seq_lens, request_distribution) = device_array(
                 self.runner.mesh,
                 (input_ids, positions, block_tables, query_start_loc,
                  seq_lens, request_distribution))
            logger.info(f"Precompile backbone --> num_tokens={num_tokens}")

            attention_metadata = AttentionMetadata(
                input_positions=positions,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )

            inputs_embeds = None
            start = time.perf_counter()
            kv_caches, hidden_states = self.runner.model_fn(
                self.runner.state, self.runner.kv_caches, input_ids,
                attention_metadata, inputs_embeds)
            self.runner.kv_caches = kv_caches
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)
            hidden_states.block_until_ready()

    def _precompile_select_from_array_helper(
        self,
        name: str,
        outer_paddings: List[int],
        input_shape_fn: Callable[[int], Tuple[int, ...]],
        input_dtype: Any,
        sharding: Optional[NamedSharding] = None,
        inner_loop_var_name: str = "num_tokens",
    ) -> None:
        logger.info(f"Compiling select_from_array for {name}.")
        for outer_loop_val in outer_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                if num_reqs > outer_loop_val:
                    continue
                input_tensor = jnp.ones(input_shape_fn(outer_loop_val),
                                        dtype=input_dtype)
                indices_to_select = jnp.ones((num_reqs, ), dtype=jnp.int32)
                if sharding:
                    input_tensor = device_array(self.runner.mesh,
                                                input_tensor,
                                                sharding=sharding)
                else:
                    input_tensor = device_array(self.runner.mesh, input_tensor)
                indices_to_select = device_array(self.runner.mesh,
                                                 indices_to_select)
                start = time.perf_counter()
                logger.info(
                    f"Precompile select_from_array --> {inner_loop_var_name}={outer_loop_val} | "
                    f"num_reqs={num_reqs}")
                result = self.runner._select_from_array_fn(
                    input_tensor, indices_to_select)
                result.block_until_ready()
                end = time.perf_counter()
                logger.info("Compilation finished in %.2f [secs].",
                            end - start)

    def _precompile_select_from_array(self) -> None:
        logger.info("Compiling select_from_array with different input shapes.")
        hsize = self.runner.model_config.get_hidden_size()

        self._precompile_select_from_array_helper(
            name="regular hidden states",
            outer_paddings=self.runner.num_tokens_paddings,
            input_shape_fn=lambda num_tokens: (num_tokens, hsize),
            input_dtype=jnp.bfloat16,
            inner_loop_var_name="num_tokens",
        )

        if self.runner.speculative_config:
            vocab_size = self.runner.model_config.get_vocab_size()
            self._precompile_select_from_array_helper(
                name="speculative decoding",
                outer_paddings=self.runner.num_logits_paddings,
                input_shape_fn=lambda num_logits: (num_logits, vocab_size),
                input_dtype=jnp.bfloat16,
                sharding=NamedSharding(self.runner.mesh,
                                       PartitionSpec(None, "model")),
                inner_loop_var_name="num_logits",
            )

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        hsize = self.runner.model_config.get_hidden_size()
        for num_reqs in self.runner.num_reqs_paddings:
            hidden_states = jnp.ones((num_reqs, hsize), dtype=jnp.bfloat16)
            hidden_states = device_array(self.runner.mesh, hidden_states)
            logger.info(f"Precompile compute_logits --> num_reqs={num_reqs}")
            start = time.perf_counter()
            result = self.runner.compute_logits_fn(self.runner.state,
                                                   hidden_states)
            result.block_until_ready()
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_sampling(self) -> None:
        logger.info("Compiling sampling with different input shapes.")
        hsize = self.runner.model_config.get_vocab_size()
        for num_reqs in self.runner.num_reqs_paddings:
            logits = jnp.ones((num_reqs, hsize), dtype=jnp.bfloat16)
            sharding = NamedSharding(self.runner.mesh,
                                     PartitionSpec(None, "model"))
            logits = device_array(self.runner.mesh, logits, sharding=sharding)
            logger.info(f"Precompile sampling --> num_reqs={num_reqs}")
            start = time.perf_counter()
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
                result = sample(self.runner.rng_params_for_sampling,
                                self.runner.mesh, logits, sampling_metadata)
                result.block_until_ready()
                end = time.perf_counter()
                logger.info("Compilation finished in %.2f [secs].",
                            end - start)

    def _precompile_disagg_utils(self) -> None:
        logger.info(
            "Compiling disaggregated util with different input shapes.")
        block_size = self.runner.block_size
        for num_blocks in range(1, self.runner.max_num_blocks_per_req//2):
            logger.info(
                f"Precompile slice and insert for num_blocks {num_blocks}")
            start = time.perf_counter()
            block_numbers = list(range(1, num_blocks + 1))
            kv_cache_slices = self.runner.kv_cache_manager.get_kv_cache_for_block_ids(
                block_numbers)
            block_buckets = [
                1, 2, 4, 8, 16, 32, 64, self.runner.max_num_blocks_per_req
            ]
            import bisect
            bucket_index = bisect.bisect_left(block_buckets, num_blocks)
            padded_num_blocks = block_buckets[bucket_index]
            padding_size = padded_num_blocks - num_blocks
            block_numbers.extend([0] * padding_size)
            padded_block_numbers = jnp.array(block_numbers, dtype=jnp.int32)
            self.runner.kv_caches = self.runner.kv_cache_manager._jitted_insert_kv_cache(
                block_size,
                self.runner.kv_caches,
                kv_cache_slices,
                padded_block_numbers,
            )
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        hsize = self.runner.model_config.get_vocab_size()
        for num_reqs in self.runner.num_reqs_paddings:
            logits = jnp.ones((num_reqs, hsize), dtype=jnp.bfloat16)
            logits, = device_array(self.runner.mesh, (logits, ))
            logger.info(f"Precompile gather_logprobs --> num_reqs={num_reqs}")
            start = time.perf_counter()
            token_ids = jnp.ones((num_reqs, ), dtype=jnp.int32)
            token_ids, = device_array(self.runner.mesh, (token_ids, ))
            result = self.runner._compute_and_gather_logprobs(
                logits, token_ids, self.runner.model_config.max_logprobs)
            result.logprob_token_ids.block_until_ready()
            end = time.perf_counter()
            logger.info("Compilation finished in %.2f [secs].", end - start)

    def _precompile_rejection_sampler(self) -> None:
        logger.info(
            "Compiling greedy_rejection_sampler with different input shapes.")
        vocab_size = self.runner.model_config.get_vocab_size()
        for num_logits in self.runner.num_logits_paddings:
            for num_reqs in self.runner.num_reqs_paddings:
                target_probs = jnp.ones((num_logits, vocab_size),
                                        dtype=jnp.bfloat16)
                sharding = NamedSharding(self.runner.mesh,
                                         PartitionSpec(None, "model"))
                target_probs = device_array(self.runner.mesh,
                                            target_probs,
                                            sharding=sharding)
                draft_token_ids = jnp.ones((num_logits, ), dtype=jnp.int32)
                num_draft_tokens = jnp.ones((num_reqs, ), dtype=jnp.int32)
                bonus_token_ids = jnp.ones((num_reqs, ), dtype=jnp.int32)
                draft_token_ids, num_draft_tokens, bonus_token_ids = device_array(
                    self.runner.mesh,
                    (draft_token_ids, num_draft_tokens, bonus_token_ids))
                logger.info("Precompile greedy_rejection_sampler --> "
                            f"num_logits={num_logits} num_reqs={num_reqs}")
                start = time.perf_counter()
                result = self.runner.rejection_sampler(draft_token_ids,
                                                       num_draft_tokens, -1,
                                                       None, target_probs,
                                                       bonus_token_ids, None)
                result.block_until_ready()
                end = time.perf_counter()
                logger.info("Compilation finished in %.2f [secs].",
                            end - start)

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.runner.num_reqs_paddings:
            dummy_logits = jnp.zeros((num_reqs, self.runner.vocab_size),
                                     dtype=jnp.bfloat16)
            dummy_require_struct_decoding = self.runner.require_structured_out_cpu[:
                                                                                   num_reqs]
            dummy_grammar_bitmask = self.runner.grammar_bitmask_cpu[:num_reqs]

            (dummy_logits, dummy_require_struct_decoding,
             dummy_grammar_bitmask, arange) = device_array(
                 self.runner.mesh,
                 (dummy_logits, dummy_require_struct_decoding,
                  dummy_grammar_bitmask, self.runner.structured_decode_arange))

            self.runner.structured_decoding_manager.structured_decode_fn(
                dummy_require_struct_decoding, dummy_grammar_bitmask,
                dummy_logits, arange)
            logger.info("  -- num_seqs: %d", num_reqs)

        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
