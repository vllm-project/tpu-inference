# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Asynchronous vLLM Sampler engine implementation for tpu-inference.

This module provides a standalone vLLM serving and rollout sampler (`VllmSampler`)
tailored for Reinforcement Learning (RL) workloads on TPUs.
Satisfies the open-source Tunix `Sampler` Protocol defined in:
https://github.com/google/tunix/blob/main/tunix/experimental/sampler/sampler.py
without importing or depending on Tunix.
"""

import asyncio
from dataclasses import dataclass, field
import logging
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as VllmSamplingParams

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
  """Helper to extract an attribute or dict key seamlessly from any duck-typed request."""
  if obj is None:
    return default
  if isinstance(obj, dict):
    return obj.get(key, default)
  return getattr(obj, key, default)


# ==============================================================================
# Configuration for VllmSampler
# ==============================================================================


@dataclass
class VllmSamplerConfig:
  """Configuration parameters for VllmSampler in tpu-inference."""

  model_path: str = "Qwen/Qwen2.5-1.5B"
  tensor_parallel_size: int = 1
  data_parallel_size: int = 1
  expert_parallel_size: int = 1
  max_num_seqs: int = 256
  max_num_batched_tokens: int = 8192
  hbm_utilization: float = 0.80
  enable_prefix_caching: bool = True
  enable_continue_decode: bool = False
  weight_dtype: str = "bfloat16"
  additional_engine_args: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Concrete VllmSampler Implementation (Duck-Typed for any RL orchestrator)
# ==============================================================================


class VllmSampler:
  """Asynchronous vLLM sampler for RL inside `tpu-inference`.

  Satisfies the open-source Tunix `Sampler` Protocol without importing Tunix.
  Handles:

    - Direct binding to vLLM's `AsyncLLMEngine`.
    - Dynamic attribute extraction (`_get_val`) for arbitrary request inputs.
    - Non-blocking asynchronous batch generation (`sample`).
    - Sticky routing header support (`route_key`) for multi-turn prefix cache hits.
    - Native `TPUWorker` 3-phase weight synchronization (`pre_weight_sync`, `weight_sync`, `post_weight_sync`).
  """

  def __init__(self, config: VllmSamplerConfig | None = None):
    self.config = config or VllmSamplerConfig()
    self._engine: Any | None = None
    self._is_running = False
    self._is_paused = False
    self._kv_cache_valid = True
    self._mesh: Any | None = None
    self._dst_controller_id = f"raiden_dst_{id(self)}"
    self._dst_controller_ip = "127.0.0.1"
    self._src_controller_ip: str | None = None
    self._transfer_statuses: dict[str, str] = {}
    self._policy_version = 0

  def _get_tpu_workers(self) -> list[Any]:
    """Retrieves active TPUWorker instances from underlying model executor."""
    if self._engine is None:
      return []
    llm_engine = getattr(self._engine, "engine", None)
    model_executor = getattr(llm_engine, "model_executor", None) if llm_engine else None
    if model_executor is None:
      return []
    workers = getattr(model_executor, "workers", None)
    if workers:
      return list(workers)
    driver_worker = getattr(model_executor, "driver_worker", None)
    return [driver_worker] if driver_worker else []

  async def start(self, **kwargs: Any) -> None:
    """Initializes the vLLM engine and execution environment."""
    if self._is_running:
      logger.warning("VllmSampler is already running.")
      return

    logger.info("Initializing VllmSampler with model: %s", self.config.model_path)

    engine_args = AsyncEngineArgs(
        model=self.config.model_path,
        tensor_parallel_size=self.config.tensor_parallel_size,
        dtype=self.config.weight_dtype,
        gpu_memory_utilization=self.config.hbm_utilization,
        enable_prefix_caching=self.config.enable_prefix_caching,
        max_num_seqs=self.config.max_num_seqs,
        max_num_batched_tokens=self.config.max_num_batched_tokens,
        **self.config.additional_engine_args,
    )
    self._engine = AsyncLLMEngine.from_engine_args(engine_args)
    self._is_running = True

    init_info = {
        "model_path": self.config.model_path,
        "tp_size": self.config.tensor_parallel_size,
    }
    for w in self._get_tpu_workers():
      if hasattr(w, "init_weight_transfer_engine"):
        w.init_weight_transfer_engine(init_info)

    logger.info("VllmSampler started successfully.")

  async def stop(self, **kwargs: Any) -> None:
    """Stops the sampler and releases resources."""
    if not self._is_running:
      return
    logger.info("Stopping VllmSampler...")
    self._is_paused = True
    self._engine = None
    self._is_running = False
    logger.info("VllmSampler stopped.")

  async def pause(self, **kwargs: Any) -> None:
    """Pauses request intake and drains active batch iterations during weight updates."""
    if self._is_paused:
      return
    logger.info("Pausing VllmSampler inference intake for weight sync...")
    self._is_paused = True
    if self._engine and hasattr(self._engine, "pause_background_loop"):
      await self._engine.pause_background_loop()
    await asyncio.sleep(0.01)

  async def resume(self, **kwargs: Any) -> None:
    """Resumes inference processing after weight sync completion."""
    if not self._is_paused:
      return
    logger.info("Resuming VllmSampler inference serving...")
    if self._engine and hasattr(self._engine, "resume_background_loop"):
      await self._engine.resume_background_loop()
    self._is_paused = False

  async def get_mesh(self, **kwargs: Any) -> Any | None:
    """Returns the JAX device mesh."""
    return self._mesh

  # ----------------------------------------------------------------------------
  # Helper Methods for Sample Stream Processing
  # ----------------------------------------------------------------------------

  def _build_vllm_params(
      self,
      req: Any,
      defaults: dict[str, Any],
  ) -> VllmSamplingParams:
    """Builds a vLLM SamplingParams object from any duck-typed request."""
    sparams = _get_val(req, "sampling_params")
    return VllmSamplingParams(
        temperature=_get_val(sparams, "temperature", defaults["temperature"]),
        top_p=_get_val(sparams, "top_p", defaults["top_p"]),
        top_k=_get_val(sparams, "top_k", -1),
        max_tokens=_get_val(sparams, "max_tokens", defaults["max_tokens"]),
        stop=_get_val(sparams, "stop_sequences") or _get_val(sparams, "stop") or None,
        prompt_logprobs=1 if _get_val(sparams, "return_logprobs", defaults["return_logprobs"]) else None,
        logprobs=1 if _get_val(sparams, "return_logprobs", defaults["return_logprobs"]) else None,
    )

  async def _process_stream_output(
      self,
      req_id: str,
      route_key: str | None,
      task: Any,
  ) -> SimpleNamespace:
    """Consumes an AsyncLLMEngine output stream and formats output result."""
    try:
      final_output = None
      async for step_output in task:
        final_output = step_output

      if final_output and final_output.outputs:
        output_choice = final_output.outputs[0]
        text = output_choice.text
        token_ids_arr = np.array(output_choice.token_ids, dtype=np.int32)
        cum_logprob = float(getattr(output_choice, "cumulative_logprob", 0.0) or 0.0)

        logprobs_arr = None
        if getattr(output_choice, "logprobs", None):
          logprob_vals = [
              lp_dict[next(iter(lp_dict.keys()))].logprob if lp_dict else 0.0
              for lp_dict in output_choice.logprobs
          ]
          logprobs_arr = np.array(logprob_vals, dtype=np.float32)

        return SimpleNamespace(
            request_id=req_id,
            text=text,
            token_ids=token_ids_arr,
            logprobs=logprobs_arr,
            cumulative_logprob=cum_logprob,
            finish_reason=getattr(output_choice, "finish_reason", "stop") or "stop",
            route_key=route_key,
            error=None,
        )

      err_obj = SimpleNamespace(
          error_type="EmptyOutput",
          message="No output generated by vLLM",
          retryable=False,
      )
      return SimpleNamespace(
          request_id=req_id,
          text="",
          token_ids=np.zeros(0, dtype=np.int32),
          logprobs=None,
          cumulative_logprob=0.0,
          route_key=route_key,
          error=err_obj,
      )
    except Exception as e:
      logger.exception("Error generating sampling result for req_id=%s", req_id)
      err_obj = SimpleNamespace(
          error_type=type(e).__name__,
          message=str(e),
          retryable=True,
      )
      return SimpleNamespace(
          request_id=req_id,
          text="",
          token_ids=np.zeros(0, dtype=np.int32),
          logprobs=None,
          cumulative_logprob=0.0,
          route_key=route_key,
          error=err_obj,
      )

  # ----------------------------------------------------------------------------
  # Dynamic Batch Sampling
  # ----------------------------------------------------------------------------

  async def sample(
      self,
      sampling_requests: Any,
      **kwargs: Any,
  ) -> Any:
    """Generates completions for a batch of sampling requests or raw prompts.

    Accepts raw string lists, dictionaries, or duck-typed objects from callers
    (such as `tunix.experimental.common.datatypes.SamplingRequest`).
    """
    while self._is_paused:
      await asyncio.sleep(0.05)

    if not self._is_running or self._engine is None:
      await self.start()

    raw_input_mode = False
    if isinstance(sampling_requests, (str, list)) and (
        isinstance(sampling_requests, str)
        or not sampling_requests
        or isinstance(sampling_requests[0], str)
        or (isinstance(sampling_requests[0], list) and not _get_val(sampling_requests[0], "prompt"))
    ):
      raw_input_mode = True
      items = sampling_requests if isinstance(sampling_requests, list) else [sampling_requests]
      req_list = [
          {
              "prompt": item,
              "request_id": f"req_{idx}_{time.time_ns()}",
              "route_key": kwargs.get("route_key"),
          }
          for idx, item in enumerate(items)
      ]
    elif _get_val(sampling_requests, "prompt") is not None:
      req_list = [sampling_requests]
    else:
      req_list = list(sampling_requests)

    defaults = {
        "max_tokens": kwargs.get("max_tokens", 128),
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.95),
        "return_logprobs": kwargs.get("return_logprobs", True),
    }

    stream_tasks = []
    for idx, req in enumerate(req_list):
      vllm_params = self._build_vllm_params(req, defaults)
      route_key = _get_val(req, "route_key")
      req_id = _get_val(req, "request_id") or f"req_{route_key or 'default'}_{time.time_ns()}_{idx}"
      prompt_val = _get_val(req, "prompt")
      prompt_text = prompt_val if isinstance(prompt_val, str) else str(prompt_val)

      task_gen = self._engine.generate(prompt_text, vllm_params, request_id=req_id)
      stream_tasks.append((req_id, route_key, task_gen))

    results = await asyncio.gather(*[
        self._process_stream_output(req_id, route_key, task_gen)
        for req_id, route_key, task_gen in stream_tasks
    ])

    if raw_input_mode:
      return [r.text for r in results]
    return list(results)

  # ----------------------------------------------------------------------------
  # Cache Management
  # ----------------------------------------------------------------------------

  async def clear_cache(self) -> None:
    """Purges KV-cache and prefix caching blocks from HBM."""
    logger.info("Clearing vLLM PagedAttention KV cache and prefix cache...")
    self._kv_cache_valid = False
    if self._engine and hasattr(self._engine, "reset_prefix_cache"):
      await self._engine.reset_prefix_cache()
    self._kv_cache_valid = True

  # ----------------------------------------------------------------------------
  # Weight Synchronization (Integrated with TPUWorker)
  # ----------------------------------------------------------------------------

  async def get_weight_sync_metadata(self, **kwargs: Any) -> dict[str, Any]:
    """Returns PyTree of weight sharding rules, dtype, and shape across devices."""
    return {
        "sharding": f"{self.config.tensor_parallel_size}x{self.config.data_parallel_size}",
        "dtype": self.config.weight_dtype,
        "model_path": self.config.model_path,
        "policy_version": self._policy_version,
        "layers_valid": self._kv_cache_valid,
    }

  def get_weight_metadata(self) -> dict[str, Any]:
    """Synchronous alias for get_weight_sync_metadata."""
    return {
        "sharding": f"{self.config.tensor_parallel_size}x{self.config.data_parallel_size}",
        "dtype": self.config.weight_dtype,
        "model_path": self.config.model_path,
        "policy_version": self._policy_version,
        "layers_valid": self._kv_cache_valid,
    }

  async def pre_weight_sync(
      self,
      sync_request: Any = None,
      src_controller_ip: str | None = None,
      dst_controller_id: str | None = None,
      free_kv_cache: bool = True,
      **kwargs: Any,
  ) -> None:
    """Phase 1: Pauses intake, resets prefix cache, and calls start_weight_update()."""
    ip = src_controller_ip
    cid = dst_controller_id
    if sync_request is not None:
      ip = (
          ip
          or _get_val(sync_request, "src_controller_ip", "")
          or _get_val(sync_request, "controller_id", "")
      )
      cid = cid or _get_val(sync_request, "controller_id")
      self._policy_version = _get_val(sync_request, "policy_version", self._policy_version)

    ip = ip or "10.0.0.1"
    logger.info("Executing pre_weight_sync from Trainer IP: %s (policy_version=%d)", ip, self._policy_version)
    self._src_controller_ip = ip
    if cid:
      self._dst_controller_id = cid

    await self.pause()
    await self.clear_cache()

    for w in self._get_tpu_workers():
      if hasattr(w, "start_weight_update"):
        w.start_weight_update(free_kv_cache=free_kv_cache)

    self._kv_cache_valid = False

  async def weight_sync(
      self,
      sync_request: Any = None,
      update_info: dict[str, Any] | None = None,
      **kwargs: Any,
  ) -> None:
    """Phase 2: Calls TPUWorker.update_weights(update_info)."""
    logger.info("Executing weight_sync DMA from controller: %s", self._src_controller_ip)
    u_info = update_info or _get_val(sync_request, "extra_config") or {}
    for w in self._get_tpu_workers():
      if hasattr(w, "update_weights"):
        w.update_weights(u_info)
    await asyncio.sleep(0.01)

  async def post_weight_sync(
      self,
      sync_request: Any = None,
      req_id: str | None = None,
      **kwargs: Any,
  ) -> None:
    """Phase 3: Calls TPUWorker.finish_weight_update()."""
    rid = req_id
    if sync_request is not None:
      rid = rid or _get_val(sync_request, "req_id")
    logger.info("Executing post_weight_sync (req_id=%s)...", rid)

    for w in self._get_tpu_workers():
      if hasattr(w, "finish_weight_update"):
        w.finish_weight_update()

    self._kv_cache_valid = True
    if rid:
      self._transfer_statuses[rid] = "SUCCESS"
    await self.resume()

  async def get_transfer_status(self, req_id: Any, **kwargs: Any) -> str:
    """Returns status of weight transfer or KV-cache migration request."""
    return self._transfer_statuses.get(str(req_id), "SUCCESS")

  def get_transfer_controller_info(self) -> tuple[str, str]:
    """Returns (dst_controller_id, dst_controller_ip) for state transfer coordination."""
    return self._dst_controller_id, self._dst_controller_ip

  async def migrate_kv_cache(
      self,
      route_key: str,
      source_server_id: str,
      target_server_id: str,
      token_ids: list[int],
  ) -> bool:
    """Triggers Raiden P2P KV-cache block transfer from source to target sampler slice."""
    raise NotImplementedError(
        "P2P KV-cache migration is pending live TPU Raiden controller integration."
    )
