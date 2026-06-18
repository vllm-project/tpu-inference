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

from typing import TYPE_CHECKING, Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.ir import enable_torch_wrap
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal.inputs import (MultiModalKwargsItem, NestedTensors,
                                    PlaceholderRange)
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

import tpu_inference.envs
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.multi_modal_utils import \
    sanity_check_mm_encoder_outputs
from tpu_inference.models.vllm.mm_encoder_manager import MMEncoderManager

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import SupportsEncoderCudaGraph

    from tpu_inference.models.common.interface import ModelInterface
    from tpu_inference.models.vllm.vllm_model_wrapper import _VllmRunner
    from tpu_inference.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class _EncoderGraphForward(Protocol):

    def __call__(
        self,
        params_and_buffers: dict[str, jax.Array],
        inputs: dict[str, jax.Array],
    ) -> jax.Array:
        """The forward function for encoder graph of multi modal data."""
        ...


def _to_tpu(v: torch.Tensor | None) -> torch.Tensor | None:
    """Move the tensor to TPU and replicate it across the mesh."""

    # None does occur as some field values of mm_kwargs in some model,
    # like Qwen 3 VL, during prepare_encoder_cudagraph_replay_buffers.
    if v is None:
        return None

    return torch_view(t2j(v, use_dlpack=False))


class MultiModalManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

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
        encoder_outputs = []
        for _, num_items, mm_kwargs_group in group_and_batch_mm_kwargs(
                mm_kwargs):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
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
        is_mm_embed = jnp.array(is_mm_embed_cpu, dtype=jnp.bool_)
        assert target_pad_len == is_mm_embed.shape[0]

        return mm_embeds, is_mm_embed

    def optional_encoder_graph_optimization(
        self,
        embed_multimodal_fn,
        precompile_vision_encoder_fn,
    ):
        original_functions = (
            embed_multimodal_fn,
            precompile_vision_encoder_fn,
        )

        # Only support using SupportsEncoderCudaGraph in torchax path.
        if tpu_inference.envs.MODEL_IMPL_TYPE != "vllm":
            return original_functions

        runner = self.runner

        from vllm.model_executor.models.interfaces import \
            supports_encoder_cudagraph

        # Try detect the support of SupportsEncoderCudaGraph
        # and initialized required fields.
        model_interface: "ModelInterface" = runner.model
        vllm_runner: "_VllmRunner" = model_interface.model
        use_graph_feature: bool = all([
            supports_encoder_cudagraph(vllm_runner.vllm_model),
            runner.vllm_config.compilation_config.cudagraph_mm_encoder,
        ])
        if not use_graph_feature:
            return original_functions

        logger.warning(
            "Detect the implementation of SupportsEncoderCudaGraph. "
            "Overriding embed_multimodal_fn and precompile_vision_encoder_fn.",
        )

        model: "SupportsEncoderCudaGraph" = vllm_runner.vllm_model
        manager = MMEncoderManager(
            runner.vllm_config,
            model,
        )
        graph_config = manager.config
        padding_logics = graph_config.padding_logics

        @jax.jit
        def graph_forward_wrapper(
            params_and_buffers: dict[str, jax.Array],
            inputs: dict[str, jax.Array],
        ) -> jax.Array:

            torch_inputs = torch_view(inputs)
            # Note that we didn't activate torchax environment here,
            # as we leaves the responsibility to the caller of this func.
            torch_results = torch.func.functional_call(
                vllm_runner,
                torch_view(params_and_buffers),
                kwargs={
                    "call_method": "encoder_cudagraph_forward",
                    "call_args": (torch_inputs, ),
                    "call_kwargs": {},
                },
                tie_weights=False,
            )
            results = jax_view(torch_results)

            return results

        encoder_graph_forward = cast(
            _EncoderGraphForward,
            graph_forward_wrapper,
        )

        params: dict[str, jax.Array] = runner.state

        def precompile_encoder_graph(
                run_compilation: Any,  # see CompilationManger._run_compilation
        ) -> None:

            def job(budget: int) -> None:
                inputs = manager.by_budget[budget]
                with (
                        torchax.default_env(),
                        enable_torch_wrap(False),
                ):
                    inputs = jax.tree.map(_to_tpu, inputs)
                    _ = encoder_graph_forward(params, jax_view(inputs))

            for budget in manager.token_budgets:
                run_compilation(
                    "multimodal_encoder_graph_forward",
                    job,
                    budget,
                    budget=budget,
                )

        def embed_multimodal_graph_forward_all(
            jax_params_and_buffers: dict[str, jax.Array],
            **mm_kwargs: dict[str, NestedTensors],
        ) -> list[jax.Array]:

            def get_fit_val(values: list[int], value: int) -> int | None:
                # assume the values are ascending sorted.
                for v in values:
                    if v >= value:
                        return v
                return None

            item_specs = model.get_encoder_cudagraph_item_specs(mm_kwargs)
            num_items = len(item_specs)
            out_tokens = [spec.output_tokens for spec in item_specs]

            # batches is a list of (expected budget, list of item ID)"""
            batches: list[tuple[int | None, list[int]]] = []
            sorted_indices = sorted(
                range(num_items),
                key=lambda i: out_tokens[i],
            )

            # Greedy packing into batches.
            # NOTE: conditions must be inline in the while predicate, not
            # pre-evaluated into a list. A frozen list like
            #   has_space = [count < max_batch_size, used + x < max_budget]
            # evaluates both booleans once and never updates them, so the inner
            # loop would consume all items regardless of budget, causing `used`
            # to exceed max_budget and get_fit_val to return None.
            max_budget = manager.token_budgets[-1]
            idx = 0
            while idx < num_items:
                indexes: list[int] = []
                count = 0
                used = 0
                while (idx < num_items and count < manager.max_batch_size and
                       used + out_tokens[sorted_indices[idx]] <= max_budget):
                    count += 1
                    used += out_tokens[sorted_indices[idx]]
                    indexes.append(sorted_indices[idx])
                    idx += 1

                if not indexes:
                    # Single item exceeds max_budget; force it into its own batch.
                    indexes.append(sorted_indices[idx])
                    used = out_tokens[sorted_indices[idx]]
                    idx += 1

                budget = get_fit_val(manager.token_budgets, used)
                batches.append((budget, indexes))

            def run_batch(
                budget: int | None,
                output_tokens: list[int],
                mm_kwargs: dict[str, NestedTensors],
            ) -> list[jax.Array]:
                if budget is None:  # Budget size not supported
                    return embed_multimodal_fn(
                        jax_params_and_buffers,
                        **mm_kwargs,
                    )
                # Now, for normal cases.
                local_indexes = list(range(len(output_tokens)))
                outputs = run_graph_forward(
                    jax_params_and_buffers,
                    mm_kwargs,
                    budget,
                    local_indexes,
                    output_tokens,
                )

                return jax_view(outputs)

            outputs: dict[int, jax.Array] = {}
            for budget, indexes in batches:
                batch_mm_kwargs = model.select_encoder_cudagraph_items(
                    mm_kwargs,
                    indexes,
                )
                batch_outputs = run_batch(
                    budget,
                    [out_tokens[i] for i in indexes],
                    batch_mm_kwargs,
                )
                outputs |= {
                    index: output
                    for index, output in zip(indexes, batch_outputs)
                }

            return [outputs[i] for i in range(len(item_specs))]

        def run_graph_forward(
            jax_params_and_buffers: dict[str, jax.Array],
            mm_kwargs: dict[str, NestedTensors],
            token_budget: int,
            indexes: list[int],
            output_tokens: list[int],
        ) -> list[torch.Tensor]:
            # The return tensor contains result from (possibly) multiple items.

            # Duplication of EncoderCudaGraphManager._copy_padded_buffer
            def copy_padded_buffer(
                dst: torch.Tensor,
                src: torch.Tensor,
            ) -> None:
                dst.zero_()
                dst[:src.shape[0]].copy_(src)

            values = model.prepare_encoder_cudagraph_replay_buffers(
                mm_kwargs,
                manager.max_batch_size,
                manager.max_frames_per_batch,
            ).values

            inputs = manager.by_budget[token_budget]
            # Move per-req states into fixed-sized tensor buffers.
            for key in graph_config.buffer_keys:
                src = values.get(key)
                if src is None:
                    continue
                buf = inputs[key]
                if src.ndim == 0:
                    buf.copy_(src)
                    continue
                else:
                    pad = padding_logics.get(key, copy_padded_buffer)
                    pad(buf, src)

            with (
                    torchax.default_env(),
                    enable_torch_wrap(False),
            ):

                inputs = jax.tree.map(_to_tpu, inputs)

                jax_outputs = encoder_graph_forward(
                    jax_params_and_buffers,
                    jax_view(inputs),
                )
                outputs = torch_view(jax_outputs)

                by_idx: dict[int, torch.Tensor] = {}
                model.postprocess_encoder_output(
                    outputs,
                    indexes,
                    output_tokens,
                    dest=by_idx,  # the output parameter
                    clone=True,
                    batch_mm_kwargs=mm_kwargs,
                )

            return [by_idx[i] for i in indexes]

        return (
            embed_multimodal_graph_forward_all,
            precompile_encoder_graph,
        )
