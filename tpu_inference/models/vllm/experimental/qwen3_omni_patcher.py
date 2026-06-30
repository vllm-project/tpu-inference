# Copyright 2026 Google LLC
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
"""Qwen3-Omni Patches for running vLLM Qwen3-Omni model via TorchAX.

This file provides patches to make the Qwen3-Omni model compatible
with JIT compilation on TPUs.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm.model_executor.models.utils as vllm_utils
from vllm.model_executor.layers.attention.mm_encoder_attention import \
    MMEncoderAttention
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.sequence import IntermediateTensors

from tpu_inference.distributed.jax_parallel_state import \
    get_pp_group as jax_get_pp_group
from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.experimental.qwen3_vl_patcher import (
    _patched_flatten_embeddings, _patched_get_deepstack,
    _patched_set_deepstack)

logger = init_logger(__name__)


def _patched_qwen3_omni_embed_input_ids(
    vllm_model,
    input_ids: torch.Tensor,
    multimodal_embeddings=None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    """A completely JIT-friendly implementation of embed_input_ids that avoids .cpu() and item() syncs."""
    # 1. Get base text embeddings
    inputs_embeds = vllm_model._embed_text_input_ids(
        input_ids,
        vllm_model.language_model.embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds

    # 2. Extract deepstack features if deepstack is enabled
    deepstack_input_embeds = None
    has_vision_embeddings = [
        embeddings.shape[-1] != vllm_model.config.text_config.hidden_size
        for embeddings in multimodal_embeddings
    ]
    if vllm_model.visual.deepstack_visual_indexes is not None and any(
            has_vision_embeddings):
        multiscale_len = len(vllm_model.visual.deepstack_visual_indexes)
        multimodal_embeddings_multiscale = []

        # We process each multimodal embedding
        for index, embeddings in enumerate(multimodal_embeddings):
            if embeddings.shape[
                    -1] != vllm_model.config.text_config.hidden_size:
                visual_dim = embeddings.shape[-1] // (multiscale_len + 1)
                multi_dim = visual_dim * multiscale_len
                # Split main features from multiscale features
                embeddings_main, embeddings_multiscale = torch.split(
                    embeddings, [visual_dim, multi_dim], dim=-1)
                multimodal_embeddings[index] = embeddings_main
                multimodal_embeddings_multiscale.append(embeddings_multiscale)

        # Merge multiscale features for Deepstack
        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0), multiscale_len * inputs_embeds.size(1))
        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )

        # Reshape to match deepstack buffer shape
        deepstack_input_embeds = (deepstack_input_embeds.view(
            inputs_embeds.shape[0], multiscale_len,
            vllm_model.visual_dim).permute(1, 0, 2).contiguous())
        vllm_model._set_deepstack_input_embeds(deepstack_input_embeds)

    # 3. Standard merge of remaining multimodal embeddings (images/audio)
    return _merge_multimodal_embeddings(
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


def _patched_qwen3_omni_forward(
    vllm_model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor | IntermediateTensors:
    """Pure JIT-friendly forward passing deepstack embeddings statelessly."""
    if intermediate_tensors is not None:
        inputs_embeds = None

    if inputs_embeds is not None and jax_get_pp_group().is_first_rank:
        deepstack_input_embeds = vllm_model._get_deepstack_input_embeds(
            inputs_embeds.size(0))
    else:
        deepstack_input_embeds = None

    hidden_states = vllm_model.language_model.model(
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=inputs_embeds,
        # args for deepstack
        deepstack_input_embeds=deepstack_input_embeds,
    )

    if inputs_embeds is not None and jax_get_pp_group().is_first_rank:
        vllm_model._clear_deepstack_input_embeds(inputs_embeds.size(0))

    return hidden_states


def _patched_rot_pos_emb(self, grid_thw) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT-friendly rot_pos_emb avoiding item() calls and dynamic tensor slicing."""
    if isinstance(grid_thw, torch.Tensor):
        grid_thw_list = grid_thw.tolist()
    else:
        grid_thw_list = grid_thw

    pos_ids = []
    for t, h, w in grid_thw_list:
        hpos_ids = torch.arange(h,
                                device=self.device).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w,
                                device=self.device).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = max(max(h, w) for _, h, w in grid_thw_list)

    # Use pre-computed cos_sin_cache from RotaryEmbedding
    cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

    pos_ids = pos_ids.to(cos.device, non_blocking=True)
    cos_combined = cos[pos_ids].flatten(1)
    sin_combined = sin[pos_ids].flatten(1)

    return cos_combined, sin_combined


def _patched_vision_transformer_forward(self, x: torch.Tensor,
                                        grid_thw) -> torch.Tensor:
    """JIT-friendly forward avoiding .cpu().numpy() and dynamic list slicing on device."""
    hidden_states = x.to(device=self.device, dtype=self.dtype)
    hidden_states = self.patch_embed(hidden_states)

    if self.apply_vit_abs_pos_embed:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
    rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw)

    if isinstance(grid_thw, torch.Tensor):
        grid_thw_list = grid_thw.tolist()
    else:
        grid_thw_list = grid_thw

    # Statically compute cu_seqlens
    patches_per_frame = [h * w for _, h, w in grid_thw_list]
    repeated_patches = []
    for (t, _, _), patches in zip(grid_thw_list, patches_per_frame):
        repeated_patches.extend([patches] * t)

    # Cumsum in python
    cu_seqlens_list = [0]
    current = 0
    for val in repeated_patches:
        current += val
        cu_seqlens_list.append(current)

    cu_seqlens = torch.tensor(cu_seqlens_list,
                              dtype=torch.int32,
                              device=self.device)

    hidden_states = hidden_states.unsqueeze(1)
    rotary_pos_emb_cos = rotary_pos_emb_cos.to(hidden_states.device)
    rotary_pos_emb_sin = rotary_pos_emb_sin.to(hidden_states.device)

    # Statically compute max_seqlen
    max_seqlen_val = max(repeated_patches) if repeated_patches else 0
    max_seqlen = torch.tensor(max_seqlen_val,
                              dtype=torch.int32,
                              device=self.device)

    # Recompute cu_seqlens in numpy statically to avoid GPU->CPU sync
    cu_seqlens_np = np.array(cu_seqlens_list, dtype=np.int32)
    sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
        self.attn_backend,
        cu_seqlens_np,
        self.device,
    )

    hidden_states_list = []
    deepstack_visual_indexes = self.deepstack_visual_indexes

    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        if (deepstack_visual_indexes is not None
                and layer_num in deepstack_visual_indexes):
            hidden_states_list.append(hidden_states)

    hidden_states = self.merger(hidden_states)

    # processing deepstack
    if deepstack_visual_indexes is not None:
        processed_hidden_states_list = [hidden_states]
        for idx, x in enumerate(hidden_states_list):
            x = self.merger_list[idx](x)
            processed_hidden_states_list.append(x)
        hidden_states = torch.cat(processed_hidden_states_list, dim=1)

    return hidden_states


def _get_feat_extract_output_lengths(
        input_lengths: tuple[int, ...]) -> tuple[int, ...]:
    output_lengths = []
    for length in input_lengths:
        input_lengths_leave = length % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        out_len = (
            (feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (length // 100) * 13
        output_lengths.append(out_len)
    return tuple(output_lengths)


def _patched_process_audio_input(vllm_model, audio_input):
    """JIT-friendly processing of audio inputs, keeping dynamic values static."""
    input_features = audio_input["input_features"]
    audio_feature_lengths = audio_input["audio_feature_lengths"]

    # Compute static output lengths using pure Python math on the static tuple
    audio_output_lengths = _get_feat_extract_output_lengths(
        audio_feature_lengths)

    audio_features = vllm_model.audio_tower(
        input_features.to(vllm_model.audio_tower.dtype),
        feature_lens=audio_feature_lengths,
        aftercnn_lens=audio_output_lengths,
    )
    return audio_features.split(audio_output_lengths)


def _patched_audio_tower_forward(
    self,
    input_features: torch.Tensor,
    feature_lens,
    aftercnn_lens,
):
    """A completely JIT-friendly, trace-safe implementation of AudioTower forward."""
    n_window = self.n_window

    # 1. Compute chunk lengths in Python (to keep them static in JAX trace)
    chunk_lengths = []
    for length in feature_lens:
        num_chunks = math.ceil(length / (n_window * 2))
        for i in range(num_chunks - 1):
            chunk_lengths.append(n_window * 2)
        last_chunk_len = length % (n_window * 2)
        if last_chunk_len == 0:
            last_chunk_len = n_window * 2
        chunk_lengths.append(last_chunk_len)

    # 2. Split and pad features
    chunk_list = input_features.T.split(chunk_lengths, dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list,
                                               batch_first=True).transpose(
                                                   1, 2)

    # 3. Compute CNN output lengths and mask in Python
    cnn_lengths = []
    for length in chunk_lengths:
        val = length
        for _ in range(3):
            val = (val - 1) // 2 + 1
        cnn_lengths.append(val)

    max_len_after_cnn = max(cnn_lengths)

    # 4. Apply CNN layers
    padded_feature = padded_feature.unsqueeze(1)
    if padded_feature.size(0) <= self.conv_chunksize:
        padded_embed = F.gelu(self.conv2d1(padded_feature))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
    else:
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

    b, c, f, t = padded_embed.size()
    padded_embed = self.conv_out(
        padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    # Add positional embedding
    positional_embedding = (
        self.positional_embedding.
        positional_embedding[:padded_embed.shape[1], :].unsqueeze(0).to(
            padded_embed.dtype))
    padded_embed = padded_embed + positional_embedding

    # Extract valid hidden states using static slicing instead of dynamic boolean indexing
    valid_embeds = []
    for i, cnn_len in enumerate(cnn_lengths):
        valid_embeds.append(padded_embed[i, :cnn_len, :])
    hidden_states = torch.cat(valid_embeds, dim=0)

    # 5. Compute cumulative sequence lengths in Python
    cu_chunk_lens = [0]
    window_aftercnn = max_len_after_cnn * (self.n_window_infer //
                                           (self.n_window * 2))
    for cnn_len in aftercnn_lens:
        num_full_chunks = cnn_len // window_aftercnn
        remainder = cnn_len % window_aftercnn
        cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
        if remainder:
            cu_chunk_lens.append(remainder)

    cu_seqlens = torch.tensor(cu_chunk_lens,
                              dtype=torch.int32,
                              device=padded_feature.device).cumsum(
                                  -1, dtype=torch.int32)

    max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

    # Apply transformer layers
    for encoder_layer in self.layers:
        hidden_states = encoder_layer(
            hidden_states,
            cu_seqlens,
            max_seqlen,
        )

    # Apply output layers
    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return hidden_states


def _apply_pad_sequence_patch():
    """Globally patch pad_sequence with a pure-PyTorch sliced implementation.

    This avoids the opaque torch._C._nn.pad_sequence C++ op that fails or
    corrupts values during TorchAX tracing on TPU.
    """
    if getattr(torch.nn.utils.rnn.pad_sequence, "_is_tpu_patched", False):
        return

    logger.info(
        "Applying global pure-PyTorch pad_sequence patch for torchax compatibility"
    )

    def patched_pad_sequence(sequences,
                             batch_first=False,
                             padding_value=0.0,
                             padding_side="right"):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)
        else:
            sequences = tuple(sequences)

        if len(sequences) == 0:
            raise RuntimeError(
                "pad_sequence: Expected a non-empty list of Tensors")

        # Find max length and trailing dimensions
        max_len = max([s.size(0) for s in sequences])
        num_seqs = len(sequences)
        trailing_dims = sequences[0].shape[1:]

        # Initialize the output tensor with the padding value
        if batch_first:
            out_dims = (num_seqs, max_len) + trailing_dims
        else:
            out_dims = (max_len, num_seqs) + trailing_dims

        out_tensor = torch.full(out_dims,
                                padding_value,
                                dtype=sequences[0].dtype,
                                device=sequences[0].device)

        # Fill the tensor using standard slicing (which torchax traces perfectly)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                if padding_side == "right":
                    out_tensor[i, :length, ...] = tensor
                else:  # left
                    out_tensor[i, max_len - length:, ...] = tensor
            else:
                if padding_side == "right":
                    out_tensor[:length, i, ...] = tensor
                else:  # left
                    out_tensor[max_len - length:, i, ...] = tensor

        return out_tensor

    patched_pad_sequence._is_tpu_patched = True
    torch.nn.utils.rnn.pad_sequence = patched_pad_sequence


def apply_qwen3_omni_patches(vllm_model):
    """Apply Qwen3-Omni specific patches for stateless Deepstack support and JIT vision tower."""
    # Apply pad_sequence patch for TorchAX compatibility
    _apply_pad_sequence_patch()
    # Patch the vision tower module
    if hasattr(vllm_model, "visual"):
        visual_module = type(vllm_model.visual)
        visual_module.rot_pos_emb = _patched_rot_pos_emb
        visual_module.forward = _patched_vision_transformer_forward
        logger.info(
            "Patched Qwen3-Omni Vision Transformer rot_pos_emb and forward for JIT."
        )

    # Patch the audio tower module
    if hasattr(vllm_model, "audio_tower"):
        audio_module = type(vllm_model.audio_tower)
        audio_module.forward = _patched_audio_tower_forward
        logger.info("Patched Qwen3-Omni Audio Tower forward for JIT.")

    # Override _process_audio_input on the model
    vllm_model._process_audio_input = lambda audio_input: _patched_process_audio_input(
        vllm_model, audio_input)
    logger.info("Patched Qwen3-Omni _process_audio_input for JIT.")

    if not getattr(vllm_model, "use_deepstack", False):
        return

    # 1. Override deepstack setter to avoid stateful updates of native vLLM variables
    orig_set_deepstack = getattr(vllm_model, "_set_deepstack_input_embeds",
                                 None)
    if orig_set_deepstack is not None:
        vllm_model._set_deepstack_input_embeds = lambda embeds: _patched_set_deepstack(
            vllm_model, embeds)

    # 2. Override deepstack getter to prefer JAX-compatible cached tensors
    orig_get_deepstack = getattr(vllm_model, "_get_deepstack_input_embeds",
                                 None)
    if orig_get_deepstack is not None:
        vllm_model._get_deepstack_input_embeds = lambda num_tokens: _patched_get_deepstack(
            vllm_model, orig_get_deepstack, num_tokens)

    # 3. Patch embed_input_ids to pack/merge features traceably
    orig_embed_input_ids = getattr(vllm_model, "embed_input_ids", None)
    if orig_embed_input_ids is not None:
        vllm_model.embed_input_ids = lambda *args, **kwargs: _patched_qwen3_omni_embed_input_ids(
            vllm_model, *args, **kwargs)

    # 4. Patch forward to pass deepstack embeddings statelessly
    vllm_model.forward = lambda *args, **kwargs: _patched_qwen3_omni_forward(
        vllm_model, *args, **kwargs)

    # 5. Patch _flatten_embeddings in vllm utils to handle negative indexes correctly in torchax
    vllm_utils._flatten_embeddings = _patched_flatten_embeddings


def is_qwen3_omni(vllm_model) -> bool:
    """Check if the given vLLM model is of architecture Qwen3OmniMoeThinkerForConditionalGeneration."""
    return type(
        vllm_model).__name__ == "Qwen3OmniMoeThinkerForConditionalGeneration"


def maybe_apply_qwen3_omni_patches(vllm_model: nn.Module) -> None:
    if is_qwen3_omni(vllm_model):
        apply_qwen3_omni_patches(vllm_model)

        if hasattr(vllm_model, "deepstack_input_embeds"):
            target_device = next(vllm_model.parameters()).device
            logger.info(
                f"Patching Qwen3-Omni deepstack buffers to device: {target_device}"
            )
            vllm_model.deepstack_input_embeds = [
                t.to(device=target_device)
                for t in vllm_model.deepstack_input_embeds
            ]
