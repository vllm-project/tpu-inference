import functools
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
from engine_api import Engine
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput

from tpu_commons.runner import TPUModelRunner


class TPUEngine(Engine):

    def __init__(self, config: VllmConfig, devices: List[Any],
                 random_key: jax.random.PRNGKey):
        self.config = config
        self.model_runner = TPUModelRunner(self.vllm_config, self.device,
                                           self.original_parallel_config)


    def prefill(
        self,
        scheduler_output: VllmSchedulerOutput,
    ) -> Tuple[List[Tuple[jax.Array, jax.Array]], ModelRunnerOutput]:
        """Public API for prefill that updates page state outside JIT."""

        runner_output = self.model_runner.execute_model(scheduler_output)
        return self.model_runner.kv_caches, runner_output

    # Public non-JIT generate method that updates page state
    def generate(
        self,
        scheduler_output: VllmSchedulerOutput,
        kv_cache: List[Tuple[jax.Array, jax.Array]] = []
    ) -> Tuple[List[Tuple[jax.Array, jax.Array]], ModelRunnerOutput]:
        """Public API for generate that updates page state outside JIT."""
        self.model_runner.kv_cache = kv_cache
        # TODO(gpolovets) confirm that KV cache is updated with each decode step.
        runner_output = self.model_runner.execute_model(scheduler_output)

        return self.model_runner.kv_caches, runner_output

    @functools.partial(
        jax.jit,
        static_argnums=(0, ),
        donate_argnums=(
            1,
            2,
        ),
    )
    def bulk_insert(
        self,
        prefix: Prefix,
        decode_state: DecodeState,
        slots: list[int],
    ) -> DecodeState:
        """Insert a single computed prefill cache into multiple slots in KV cache."""
        unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)

        unboxed_prefix["cache"] = self._maybe_unstack_prefill_result_cache(
            unboxed_prefix["cache"])

        def copy(path, partial_cache, full_cache, annotations):
            path_key = path[-1].key
            if path_key in [
                    "cache_ar_index",
                    "cached_ar_key",
                    "cached_ar_value",
                    "cached_ar_key_scale",
                    "cached_ar_value_scale",
            ]:
                return full_cache  # we don't even zero these out because we can mask them out.

            batch_idx = -1
            if "cache_batch" in annotations:
                batch_idx = annotations.index("cache_batch")
            elif "cache_scale_batch" in annotations:
                batch_idx = annotations.index("cache_scale_batch")

            if batch_idx < 0:
                raise ValueError(
                    f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}"
                )

            for slot in slots:
                if path_key == "cache_ar_segment_id":
                    ### goal: zero this out in case there is existing data
                    s = list(full_cache.shape)
                    s[batch_idx] = 1
                    zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
                    full_cache = jax.lax.dynamic_update_index_in_dim(
                        full_cache, zeros, slot, batch_idx)
                elif path_key == "cache_prefill_segment_id":
                    s = list(full_cache.shape)
                    s[batch_idx] = 1
                    zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
                    ## zero out in case prefill cache is too small to cover
                    full_cache = jax.lax.dynamic_update_index_in_dim(
                        full_cache, zeros, slot, batch_idx)
                    ## copy prefill cachce
                    full_cache = jax.lax.dynamic_update_index_in_dim(
                        full_cache, partial_cache, slot, batch_idx)
                elif path_key == "cached_ar_lengths":
                    full_cache = full_cache.at[slot].set(0)
                elif path_key in [
                        "cached_prefill_key",
                        "cached_prefill_value",
                        "cached_prefill_key_scale",
                        "cached_prefill_value_scale",
                ]:
                    full_cache = jax.lax.dynamic_update_index_in_dim(
                        full_cache, partial_cache, slot, batch_idx)
                else:
                    raise ValueError(
                        f"We don't have a strategy for inserting {path_key}")

            return full_cache

        inserted_cache = jax.tree_util.tree_map_with_path(
            copy,
            unboxed_prefix["cache"],
            decode_state["cache"],
            self.kv_cache_annotations_named,
        )

        for i, slot in enumerate(slots):
            decode_state["logits"] = jax.lax.dynamic_update_index_in_dim(
                decode_state["logits"], unboxed_prefix["logits"], slot, 0)
            decode_state["next_pos"] = jax.lax.dynamic_update_index_in_dim(
                decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0)
            decode_state[
                "generated_tokens"] = jax.lax.dynamic_update_index_in_dim(
                    decode_state["generated_tokens"],
                    jnp.expand_dims(unboxed_prefix["generated_tokens"][i],
                                    axis=0),
                    slot,
                    0,
                )
            decode_state["tokens"] = jax.lax.dynamic_update_index_in_dim(
                decode_state["tokens"],
                jnp.expand_dims(unboxed_prefix["tokens"][i], axis=0),
                slot,
                0,
            )

        inserted_logits = jax.lax.with_sharding_constraint(
            decode_state["logits"], self.replicated_sharding)
        inserted_generated_tokens = jax.lax.with_sharding_constraint(
            decode_state["generated_tokens"], self.replicated_sharding)
        inserted_next_pos = jax.lax.with_sharding_constraint(
            decode_state["next_pos"], self.replicated_sharding)
        inserted_tokens = jax.lax.with_sharding_constraint(
            decode_state["tokens"], self.replicated_sharding)
        inserted_cache = jax.lax.with_sharding_constraint(
            inserted_cache, self.kv_cache_shardings)

        return {
            "logits": inserted_logits,
            "cache": inserted_cache,
            "next_pos": inserted_next_pos,
            "generated_tokens": inserted_generated_tokens,
            "tokens": inserted_tokens,
        }

    @functools.partial(jax.jit,
                       static_argnums=(0, ),
                       donate_argnames=("prefix", "decode_state"))
    def _insert_jit(
        self,
        prefix: Prefix,
        decode_state: DecodeState,
        slot: int,
        request_id: Optional[uuid.UUID] = None,  # pylint: disable=unused-argument
        page_state_in: Optional[PageState] = None,
    ) -> DecodeState:
        """Insert a single computed prefill cache into KV cache."""
        unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)
        unboxed_prefix["cache"] = self._maybe_unstack_prefill_result_cache(
            unboxed_prefix["cache"])

        def copy(path, partial_cache, full_cache, annotations):
            path_key = path[-1].key
            if path_key in [
                    "cache_ar_index",
                    "cached_ar_key",
                    "cached_ar_value",
                    "cached_ar_key_scale",
                    "cached_ar_value_scale",
            ]:
                return full_cache

            batch_idx = -1
            if "cache_batch" in annotations:
                batch_idx = annotations.index("cache_batch")
            elif "cache_scale_batch" in annotations:
                batch_idx = annotations.index("cache_scale_batch")

            if batch_idx < 0:
                raise ValueError(
                    f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}"
                )

            if path_key == "cache_ar_segment_id":
                s = list(full_cache.shape)
                s[batch_idx] = 1
                zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
                return jax.lax.dynamic_update_index_in_dim(
                    full_cache, zeros, slot, batch_idx)
            elif path_key == "cache_prefill_segment_id":
                s = list(full_cache.shape)
                s[batch_idx] = 1
                zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
                # zero out in case prefill cache is too small to cover
                full_cache = jax.lax.dynamic_update_index_in_dim(
                    full_cache, zeros, slot, batch_idx)
                # copy prefill cache
                full_cache = jax.lax.dynamic_update_index_in_dim(
                    full_cache, partial_cache, slot, batch_idx)
                return full_cache
            elif path_key == "cached_ar_lengths":
                return full_cache.at[slot].set(0)
            elif path_key in [
                    "cached_prefill_key",
                    "cached_prefill_value",
                    "cached_prefill_key_scale",
                    "cached_prefill_value_scale",
            ]:
                return jax.lax.dynamic_update_index_in_dim(
                    full_cache, partial_cache, slot, batch_idx)
            else:
                raise ValueError(
                    f"We don't have a strategy for inserting {path_key}")

        if self.config.attention == "paged" and self.page_state is not None:

            def _copy_paged(path, prefix_cache, decode_state_cache):
                path_key = path[-1].key
                if path_key in ["key_pages", "value_pages"]:
                    page_map_for_slot = page_state_in.page_map[slot]  # pytype: disable=attribute-error
                    num_pages_to_copy = page_state_in.num_pages_used[slot]  # pytype: disable=attribute-error

                    def _update_pages(prefix_page_idx, state):
                        decode_state_pages, prefix_pages, current_page_map = state
                        prefix_page = jax.lax.dynamic_index_in_dim(
                            prefix_pages, prefix_page_idx, axis=1)
                        dest_page_idx = current_page_map[prefix_page_idx]
                        decode_state_pages = jax.lax.dynamic_update_slice_in_dim(
                            decode_state_pages,
                            prefix_page,
                            dest_page_idx,
                            axis=1)
                        return decode_state_pages, prefix_pages, current_page_map

                    decode_state_cache, _, _ = jax.lax.fori_loop(
                        0,
                        num_pages_to_copy,
                        _update_pages,
                        (decode_state_cache, prefix_cache, page_map_for_slot),
                    )
                    return decode_state_cache
                else:
                    raise ValueError(
                        f"We don't have a strategy for inserting {path_key} for paged attention."
                    )

            inserted_cache = jax.tree_util.tree_map_with_path(
                _copy_paged,
                unboxed_prefix["cache"],
                decode_state["cache"],
            )
        else:
            inserted_cache = jax.tree_util.tree_map_with_path(
                copy,
                unboxed_prefix["cache"],
                decode_state["cache"],
                self.kv_cache_annotations_named,
            )

        inserted_logits = jax.lax.dynamic_update_index_in_dim(
            decode_state["logits"], unboxed_prefix["logits"], slot, 0)
        inserted_next_pos = jax.lax.dynamic_update_index_in_dim(
            decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0)
        inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
            decode_state["generated_tokens"],
            unboxed_prefix["generated_tokens"],
            slot,
            0,
        )
        inserted_tokens = jax.lax.dynamic_update_index_in_dim(
            decode_state["tokens"], unboxed_prefix["tokens"], slot, 0)

        inserted_logits = jax.lax.with_sharding_constraint(
            inserted_logits, self.replicated_sharding)
        inserted_generated_tokens = jax.lax.with_sharding_constraint(
            inserted_generated_tokens, self.replicated_sharding)
        inserted_next_pos = jax.lax.with_sharding_constraint(
            inserted_next_pos, self.replicated_sharding)
        inserted_tokens = jax.lax.with_sharding_constraint(
            inserted_tokens, self.replicated_sharding)
        inserted_cache = jax.lax.with_sharding_constraint(
            inserted_cache, self.kv_cache_shardings)

        return {
            "logits": inserted_logits,
            "cache": inserted_cache,
            "next_pos": inserted_next_pos,
            "generated_tokens": inserted_generated_tokens,
            "tokens": inserted_tokens,
        }

    def insert(
        self,
        prefix: Prefix,
        decode_state: DecodeState,
        slot: int,
        request_id: Optional[uuid.UUID] = None,
    ) -> DecodeState:
        """Non-JIT wrapper for inserting prefill cache."""

        current_page_state = None
        if self.config.attention == "paged" and self.page_manager is not None:
            if self.page_state is None:
                self.page_state = self.page_manager.get_initial_page_state()
            current_page_state = self.page_state

        updated_decode_state = self._insert_jit(
            prefix=prefix,
            decode_state=decode_state,
            slot=slot,
            page_state_in=current_page_state,
        )

        # Update the PageState after the JIT call
        if self.config.attention == "paged" and self.page_manager is not None and self.page_state is not None:
            new_has_active_page = self.page_state.has_active_page.at[slot].set(
                True)
            self.page_state = self.page_state.replace(
                has_active_page=new_has_active_page)
        return updated_decode_state

    @functools.partial(
        jax.jit,
        static_argnums=(0, ),
        static_argnames=(
            "num_prompts",
            "seq_len",
        ),
        donate_argnums=(
            1,
            2,
        ),
    )
    def insert_partial(
        self,
        prefix: PackedPrefix,
        decode_state: DecodeState,
        cache: Any,
        slots: jax.Array,
        *,
        start_indices: jax.Array,
        num_prompts: int,
        seq_len: int,
    ) -> DecodeState:
        """Insert into KV cache"""
        unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)
        cache_unboxed = max_utils.unbox_logicallypartioned(cache)
        cache_unboxed = self._maybe_unstack_prefill_result_cache(cache_unboxed)
        start_idx = 0
        slot = slots[0]

        def copy(path, partial_cache, full_cache, annotations):
            path_key = path[-1].key
            if path_key in [
                    "cache_ar_index",
                    "cached_ar_key",
                    "cached_ar_value",
                    "cached_ar_key_scale",
                    "cached_ar_value_scale",
            ]:
                return full_cache  # we don't even zero these out because we can mask them out.

            batch_idx = -1
            if "cache_batch" in annotations:
                batch_idx = annotations.index("cache_batch")
            elif "cache_scale_batch" in annotations:
                batch_idx = annotations.index("cache_scale_batch")

            if batch_idx < 0:
                raise ValueError(
                    f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}"
                )

            if path_key == "cache_ar_segment_id":
                ### goal: zero this out in case there is existing data
                zeros = jnp.zeros((1, self.config.max_target_length -
                                   self.config.max_prefill_predict_length),
                                  dtype=jnp.int32)
                return jax.lax.dynamic_update_index_in_dim(
                    full_cache, zeros, slot, batch_idx)
            elif path_key == "cache_prefill_segment_id":
                zeros = jnp.zeros((1, self.config.max_prefill_predict_length),
                                  dtype=jnp.int32)
                ## zero out in case prefill cache is too small to cover
                full_cache = jax.lax.dynamic_update_index_in_dim(
                    full_cache, zeros, slot, batch_idx)
                # In case partial_cache is too small to slice at the given index, pad it with an extra seqlen
                if i == num_prompts - 1:
                    pad = jnp.zeros((1, seq_len), dtype=int)
                    partial_cache = jnp.concatenate([partial_cache, pad],
                                                    axis=1)
                ## copy prefill cache
                partial_cache = jax.lax.dynamic_slice(partial_cache,
                                                      (0, start_idx),
                                                      (1, seq_len))
                partial_cache = (partial_cache == partial_cache[0,
                                                                0]).astype(int)
                full_cache = jax.lax.dynamic_update_index_in_dim(
                    full_cache, partial_cache, slot, batch_idx)
                return full_cache
            elif path_key == "cached_ar_lengths":
                return full_cache.at[slot].set(0)
            elif path_key in [
                    "cached_prefill_key",
                    "cached_prefill_value",
                    "cached_prefill_key_scale",
                    "cached_prefill_value_scale",
            ]:
                seqlen_index = self.config.prefill_cache_axis_order.split(
                    ",").index("1")
                start_indices = [0, 0, 0, 0]
                start_indices[seqlen_index] = start_idx
                slice_size = list(partial_cache.shape)
                slice_size[seqlen_index] = seq_len

                slice_size = tuple(slice_size)
                # Same as in prefill_segment_id processing
                if i == num_prompts - 1:
                    pad = jnp.zeros(slice_size, dtype=partial_cache.dtype)
                    partial_cache = jnp.concatenate([partial_cache, pad],
                                                    axis=seqlen_index)
                partial_cache = jax.lax.dynamic_slice(partial_cache,
                                                      start_indices,
                                                      slice_size)

                return jax.lax.dynamic_update_index_in_dim(
                    full_cache, partial_cache, slot, batch_idx)
            else:
                raise ValueError(
                    f"We don't have a strategy for inserting {path_key}")

        inserted_cache = decode_state["cache"]
        inserted_logits = decode_state["logits"]
        inserted_next_pos = decode_state["next_pos"]
        inserted_generated_tokens = decode_state["generated_tokens"]
        inserted_tokens = decode_state["tokens"]

        for i in range(num_prompts):
            start_idx = start_indices[i]
            slot = slots[i]
            inserted_cache = jax.tree_util.tree_map_with_path(
                copy, cache_unboxed, inserted_cache,
                self.kv_cache_annotations_named)
            inserted_logits = jax.lax.dynamic_update_index_in_dim(
                inserted_logits, unboxed_prefix["logits"][i, ...], slot, 0)
            inserted_next_pos = jax.lax.dynamic_update_index_in_dim(
                inserted_next_pos, unboxed_prefix["next_pos"][i, ...], slot, 0)
            inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
                inserted_generated_tokens,
                unboxed_prefix["generated_tokens"][i, ...],
                slot,
                0,
            )
            inserted_tokens = jax.lax.dynamic_update_index_in_dim(
                inserted_tokens, unboxed_prefix["tokens"][i, ...], slot, 0)

        inserted_logits = jax.lax.with_sharding_constraint(
            inserted_logits, self.replicated_sharding)
        inserted_generated_tokens = jax.lax.with_sharding_constraint(
            inserted_generated_tokens, self.replicated_sharding)
        inserted_next_pos = jax.lax.with_sharding_constraint(
            inserted_next_pos, self.replicated_sharding)
        inserted_tokens = jax.lax.with_sharding_constraint(
            inserted_tokens, self.replicated_sharding)
        inserted_cache = jax.lax.with_sharding_constraint(
            inserted_cache, self.kv_cache_shardings)

        return {
            "logits": inserted_logits,
            "cache": inserted_cache,
            "next_pos": inserted_next_pos,
            "generated_tokens": inserted_generated_tokens,
            "tokens": inserted_tokens,
        }