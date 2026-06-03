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

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

from tpu_inference.models.jax.utils.multi_modal_utils import \
    sanity_check_mm_encoder_outputs

if TYPE_CHECKING:
    from tpu_inference.runner.tpu_runner import TPUModelRunner


class MultiModalManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner
        # Per-budget JIT manager for the vision encoder, enabled via
        # vLLM's compilation_config.cudagraph_mm_encoder (same flag the GPU
        # EncoderCudaGraphManager uses), when the model implements the
        # SupportsEncoderCudaGraph protocol. Lazy-init on first scheduled MM
        # batch (the runner's params_and_buffers aren't fully bound until
        # after weights load).
        self._mm_encoder_jit_manager = None
        self._mm_encoder_jit_init_attempted = False

    def _maybe_init_mm_encoder_jit_manager(self):
        """Lazy-init the per-budget JIT manager. No-op unless
        ``compilation_config.cudagraph_mm_encoder`` is set and the model
        implements the SupportsEncoderCudaGraph protocol. Runs at most once
        per process. Mirrors the GPU runner's setup gate."""
        if self._mm_encoder_jit_init_attempted:
            return
        self._mm_encoder_jit_init_attempted = True

        comp_config = self.runner.vllm_config.compilation_config
        if not comp_config.cudagraph_mm_encoder:
            return

        vllm_model = getattr(getattr(self.runner.model, "model", None),
                             "vllm_model", None)
        if vllm_model is None:
            return
        from vllm.model_executor.models.interfaces import \
            supports_encoder_cudagraph
        if not supports_encoder_cudagraph(vllm_model):
            return

        from tpu_inference.logger import init_logger
        from tpu_inference.runner.mm_encoder_jit_manager import \
            MMEncoderJITManager
        _log = init_logger(__name__)
        try:
            self._mm_encoder_jit_manager = MMEncoderJITManager(
                vllm_config=self.runner.vllm_config,
                vllm_runner=self.runner.model.model,  # _VllmRunner
                vllm_model=vllm_model,
                params_and_buffers=self.runner.state_leaves,
            )
            _log.warning(
                "[mm_encoder_jit] manager initialized — budgets=%s "
                "max_batch_size=%d",
                self._mm_encoder_jit_manager.token_budgets,
                self._mm_encoder_jit_manager.max_batch_size)
        except Exception as e:
            import traceback
            _log.error(
                "[mm_encoder_jit] manager init FAILED — falling back to "
                "embed_multimodal_fn: %s: %s\n%s",
                type(e).__name__, e, traceback.format_exc())
            self._mm_encoder_jit_manager = None

    def calc_mrope_positions(
        self,
        scheduler_output: "VllmSchedulerOutput",
        req_ids_dp: dict[int, list[str]],
        padded_num_scheduled_tokens_per_dp_rank: int,
    ):
        """Calculate and update the mrope_positions for the scheduled tokens in
        the runner's mrope_positions_cpu array.

        Args:
            scheduler_output: The VllmSchedulerOutput containing scheduling info for the current step.
            req_ids_dp: A dict mapping DP rank to the list of request IDs assigned to that rank for
                the current step, in the order they are packed.
            padded_num_scheduled_tokens_per_dp_rank: The number of tokens scheduled for each DP rank,
                including padding. This determines the width of each DP rank's slot in the global
                mrope_positions_cpu array, which should be sufficient to hold all scheduled tokens
                for that rank.
        """
        # Each DP rank owns the slice
        # mrope_positions_cpu[:, dp_rank*padded_per_rank : (dp_rank+1)*padded_per_rank].
        # Pack each rank's requests into that slot in order. Mirrors the
        # token_offset ramp used by _prepare_inputs for input_ids/positions.
        for dp_rank, req_ids in req_ids_dp.items():
            mrope_pos_ptr = padded_num_scheduled_tokens_per_dp_rank * dp_rank
            for req_id in req_ids:
                req = self.runner.requests[req_id]
                assert req.mrope_positions is not None

                index = self.runner.input_batch.req_id_to_index[req_id]
                num_computed_tokens = \
                    self.runner.input_batch.num_computed_tokens_cpu[index]
                num_scheduled_tokens = \
                    scheduler_output.num_scheduled_tokens[req_id]
                num_prompt_tokens = len(req.prompt_token_ids)

                if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                    prompt_part_len = max(
                        0, num_prompt_tokens - num_computed_tokens)
                    completion_part_len = max(
                        0, num_scheduled_tokens - prompt_part_len)
                else:
                    prompt_part_len = num_scheduled_tokens
                    completion_part_len = 0

                assert num_scheduled_tokens == prompt_part_len + completion_part_len

                if prompt_part_len > 0:
                    # prompt's mrope_positions are pre-computed
                    dst_start = mrope_pos_ptr
                    dst_end = mrope_pos_ptr + prompt_part_len
                    src_start = num_computed_tokens
                    src_end = num_computed_tokens + prompt_part_len

                    self.runner.mrope_positions_cpu[:, dst_start:dst_end] = \
                        req.mrope_positions[:, src_start:src_end]

                    mrope_pos_ptr += prompt_part_len

                if completion_part_len > 0:
                    # compute completion's mrope_positions on-the-fly
                    dst_start = mrope_pos_ptr
                    dst_end = mrope_pos_ptr + completion_part_len

                    MRotaryEmbedding.get_next_input_positions_tensor(
                        out=self.runner.mrope_positions_cpu,
                        out_offset=dst_start,
                        mrope_position_delta=req.mrope_position_delta,
                        context_len=num_computed_tokens + prompt_part_len,
                        num_new_tokens=completion_part_len,
                    )

                    mrope_pos_ptr += completion_part_len

    def execute_mm_encoder(self, scheduler_output: "VllmSchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.runner.requests[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        self._maybe_init_mm_encoder_jit_manager()
        encoder_outputs = []
        for modality, num_items, mm_kwargs_group in group_and_batch_mm_kwargs(
                mm_kwargs):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            if (self._mm_encoder_jit_manager is not None and
                    self._mm_encoder_jit_manager.supports_modality(modality)):
                curr_group_outputs = self._mm_encoder_jit_manager.execute(
                    mm_kwargs_group)
            else:
                curr_group_outputs = self.runner.embed_multimodal_fn(
                    self.runner.state_leaves, **mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # Cache the encoder outputs.
        for (mm_hash, _), output in zip(
                mm_hashes_pos,
                encoder_outputs,
        ):

            self.runner.encoder_cache[mm_hash] = output

    def gather_mm_embeddings(
        self,
        scheduler_output: "VllmSchedulerOutput",
        target_pad_len: int,
        req_ids_dp: dict[int, list[str]],
        padded_num_scheduled_tokens_per_dp_rank: int,
    ) -> tuple[list[jax.Array] | None, jax.Array | None]:
        """Gather multimodal_embeddings from the encoder cache with is_multimodal.

        Args:
            scheduler_output: The VllmSchedulerOutput.
            target_pad_len: The target length to pad the resulting boolean mask
                to. Must equal `padded_num_scheduled_tokens_per_dp_rank * dp_size`
                so each DP rank's slot is sized correctly.
            req_ids_dp: dp_rank -> list of req_ids assigned to that rank (in
                packing order).
            padded_num_scheduled_tokens_per_dp_rank: per-rank slot width in the
                global token buffer.

        Returns:
            A tuple containing:
                - mm_embeds: A list of JAX arrays containing the unpadded multimodal
                    embeddings, or None if there are no multimodal embeddings.
                - is_mm_embed: A boolean JAX array mask of length target_pad_len.
                    Within each DP rank's slot, True positions appear in the same
                    order as the corresponding embeddings in mm_embeds, so a
                    downstream cumsum-based gather aligns correctly.
        """

        mm_embeds: list[jax.Array] = []
        is_mm_embed_cpu = np.zeros((target_pad_len, ), dtype=np.bool_)

        # Pack per DP rank into its dedicated slot. Within a rank, advance
        # req_start_idx by num_scheduled_tokens as before; the global position
        # is (rank_token_offset + req_start_idx).
        for dp_rank, req_ids in req_ids_dp.items():
            rank_token_offset = (padded_num_scheduled_tokens_per_dp_rank *
                                 dp_rank)
            req_start_idx = 0
            for req_id in req_ids:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                    req_id]
                req_state = self.runner.requests[req_id]
                num_computed_tokens = req_state.num_computed_tokens
                mm_features = req_state.mm_features
                for _, mm_feature in enumerate(mm_features):
                    pos_info = mm_feature.mm_position
                    start_pos = pos_info.offset
                    num_encoder_tokens = pos_info.length

                    # The encoder output is needed if the two ranges overlap:
                    # [num_computed_tokens,
                    #  num_computed_tokens + num_scheduled_tokens) and
                    # [start_pos, start_pos + num_encoder_tokens)
                    if start_pos >= num_computed_tokens + num_scheduled_tokens:
                        # The encoder output is not needed in this step.
                        break
                    if start_pos + num_encoder_tokens <= num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        continue

                    start_idx = max(num_computed_tokens - start_pos, 0)
                    end_idx = min(
                        num_computed_tokens - start_pos + num_scheduled_tokens,
                        num_encoder_tokens)
                    assert start_idx < end_idx
                    curr_embeds_start, curr_embeds_end = (
                        pos_info.get_embeds_indices_in_range(
                            start_idx, end_idx))
                    if curr_embeds_start == curr_embeds_end:
                        continue

                    mm_hash = mm_feature.identifier
                    encoder_output = self.runner.encoder_cache.get(
                        mm_hash, None)
                    assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."
                    encoder_output = self.runner.encoder_cache[mm_hash]

                    if (is_embed := pos_info.is_embed) is not None:
                        is_embed = is_embed[start_idx:end_idx]
                        mm_embeds_item = encoder_output[
                            curr_embeds_start:curr_embeds_end]
                    else:
                        mm_embeds_item = encoder_output[start_idx:end_idx]

                    mm_embeds.append(mm_embeds_item)

                    req_start_pos = (rank_token_offset + req_start_idx +
                                     start_pos - num_computed_tokens)

                    # use cpu numpy array for inplace modification
                    if is_embed is None:
                        is_mm_embed_cpu[req_start_pos +
                                        start_idx:req_start_pos +
                                        end_idx] = True
                    else:
                        # is_embed is torch Tensor in cpu
                        is_mm_embed_cpu[req_start_pos +
                                        start_idx:req_start_pos +
                                        end_idx] |= is_embed.numpy()

                req_start_idx += num_scheduled_tokens

        if not mm_embeds:
            return None, None

        # OPT1: bucket-pad mm_embeds + is_mm_embed_cpu so the downstream
        # bool-indexed scatter inside vllm's `_merge_multimodal_embeddings`
        # sees a stable shape signature across batches. Without this, each
        # unique num_mm forces an XLA scatter recompile (~14s on TPU at
        # hidden=5376). Flips the last `pad_n` False positions to True;
        # those land in the tail padding region that the LM ignores.
        if os.environ.get("TPU_OPT1_PAD_MM"):
            BUCKET = int(os.environ.get("TPU_OPT1_BUCKET", "256"))
            num_mm = int(sum(int(x.shape[0]) for x in mm_embeds))
            K = ((num_mm + BUCKET - 1) // BUCKET) * BUCKET
            pad_n = K - num_mm
            if pad_n > 0:
                false_idx = np.where(~is_mm_embed_cpu)[0]
                n_to_flip = min(pad_n, len(false_idx))
                if n_to_flip > 0:
                    flip_positions = false_idx[-n_to_flip:]
                    is_mm_embed_cpu[flip_positions] = True
                    H = int(mm_embeds[0].shape[-1])
                    zero_pad = jnp.zeros((n_to_flip, H),
                                         dtype=mm_embeds[0].dtype)
                    mm_embeds.append(zero_pad)

        is_mm_embed = jnp.array(is_mm_embed_cpu, dtype=jnp.bool_)
        assert target_pad_len == is_mm_embed.shape[0]

        return mm_embeds, is_mm_embed
