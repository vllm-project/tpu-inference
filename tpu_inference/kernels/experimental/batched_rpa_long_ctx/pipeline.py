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

"""Manual triple buffering pipeline scheduling loop for RPA."""

from typing import Any, Callable

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

from tpu_inference.kernels.experimental.batched_rpa_long_ctx import configs


def _rem(val: int | jax.Array, n: int) -> int | jax.Array:
  """Computes remainder modulo n for both Python ints and JAX arrays/tracers."""
  if isinstance(val, int):
    return val % n
  return jax.lax.rem(val.astype(jnp.int32), n)


def run_manual_triple_buffer(
    total_steps: int | jax.Array,
    compute_fn: Callable[..., Any],
    q_bref: Any,
    o_bref: Any,
    kv_bref: Any,
    lse_bref: Any | None = None,
) -> None:
  """Executes an explicit triple-buffered pipeline over `total_steps`.

  Inputs rotate over `configs.NUM_INPUT_SLOTS` physical slots:
  - comp_slot = step % NUM_INPUT_SLOTS       (slot being computed on)
  - comm_slot = (step + 2) % NUM_INPUT_SLOTS (slot fetched two steps ahead)

  Outputs rotate independently over `configs.NUM_OUTPUT_SLOTS` slots:
  - out_slot = step % NUM_OUTPUT_SLOTS            (slot being written)
  - wait_out_slot = (step - 2) % NUM_OUTPUT_SLOTS (slot whose writeback is
  awaited before it is reused)

  Args:
    total_steps: Total number of pipeline steps to execute.
    compute_fn: Callable of the main kernel function.
    q_bref: Fetch-only ref for the Q blocks.
    o_bref: Writeback-only ref for the attention output.
    kv_bref: Input/output ref for the KV cache.
    lse_bref: Optional writeback-only ref for the log-sum-exp values.
  """
  with jax.named_scope("ep_initialize"):
    q_bref.copy_in(0, 0, total_steps)
    kv_bref.copy_in(0, 0, total_steps)
    q_bref.copy_in(1, 1, total_steps)
    kv_bref.copy_in(1, 1, total_steps)

  @pl.loop(0, total_steps)
  def steady_state(step):
    in_slots = configs.NUM_INPUT_SLOTS
    out_slots = configs.NUM_OUTPUT_SLOTS
    comp_slot = _rem(step, in_slots)
    comm_slot = _rem(step + 2, in_slots)
    out_slot = _rem(step, out_slots)
    wait_out_slot = _rem(step - 2, out_slots)

    with jax.named_scope("wait_in_q_kv"):
      q_bref.wait_in(step, comp_slot, total_steps)
      kv_bref.wait_in(step, comp_slot, total_steps)

    with jax.named_scope("wait_out_o"):
      o_bref.wait_out(step - 2, wait_out_slot, total_steps)
      if lse_bref is not None:
        lse_bref.wait_out(step - 2, wait_out_slot, total_steps)

    with jax.named_scope("ep_run_kernel"):

      def start_next_fetch():
        q_bref.copy_in(step + 2, comm_slot, total_steps)
        kv_bref.copy_in(step + 2, comm_slot, total_steps)

      def start_kv_writeback():
        kv_bref.copy_out(step, comp_slot, total_steps)

      compute_fn(
          step,
          comp_slot,
          total_steps,
          out_slot=out_slot,
          start_next_fetch=start_next_fetch,
          start_kv_writeback=start_kv_writeback,
      )

      with jax.named_scope("wait_out_kv"):
        kv_bref.wait_out(step, comp_slot, total_steps)

      o_bref.copy_out(step, out_slot, total_steps)
      if lse_bref is not None:
        lse_bref.copy_out(step, out_slot, total_steps)

  with jax.named_scope("epilogue"):
    out_slots = configs.NUM_OUTPUT_SLOTS
    last_two = total_steps - 2
    last_one = total_steps - 1
    o_bref.wait_out(last_two, _rem(last_two, out_slots), total_steps)
    if lse_bref is not None:
      lse_bref.wait_out(last_two, _rem(last_two, out_slots), total_steps)
    o_bref.wait_out(last_one, _rem(last_one, out_slots), total_steps)
    if lse_bref is not None:
      lse_bref.wait_out(last_one, _rem(last_one, out_slots), total_steps)
