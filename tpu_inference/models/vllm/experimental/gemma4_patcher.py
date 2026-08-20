# Copyright 2026 Google LLC
"""Gemma-4 Patches for running vLLM Gemma-4 model via TorchAX on TPU."""

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def maybe_apply_gemma4_patches(vllm_model) -> None:
  """Apply Gemma-4 specific patches for Torchax TPU execution."""
  if hasattr(vllm_model, "per_layer_embeddings"):
    logger.info("Disabled static CUDA PLE buffer for TPU execution.")
    vllm_model.per_layer_embeddings = None