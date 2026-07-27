"""Utility functions for compressor benchmarking."""

import json
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

try:
  # google3-only. Outside google3 the Xprof path is unavailable and
  # run_benchmark_and_profile falls back to Python-only timing.
  from google3.perftools.accelerators.xprof.api.python import xprof_analysis_client
  from google3.perftools.accelerators.xprof.api.python import xprof_session
except ImportError:
  xprof_analysis_client = None
  xprof_session = None


def get_op_runtime(name: str, session_id: str) -> tuple[float, int]:
  """Gets the runtime of an op from xprof session."""
  client = xprof_analysis_client.XprofAnalysisClient()
  trace = client.get_profile_data("trace_viewer.json", session_id)
  jtrace = json.loads(trace[1])
  t = 0
  cnt = 0
  for e in jtrace["traceEvents"]:
    if name in e.get("name", ""):
      t += e["dur"]
      cnt += 1
  if cnt == 0:
    raise ValueError(f"Op {name} not found in xprof session.")
  return t, cnt


def get_device_name() -> str:
  """Returns the name of the TPU device."""
  kind = jax.devices()[0].device_kind
  if "TPU" not in kind:
    raise RuntimeError("Expected TPU devices")
  suffix = ""
  if kind.endswith(" lite"):
    kind = kind[: -len(" lite")]
    suffix = "e"
  if kind == "TPU7x":
    kind = "TPU v7"
  assert kind[:-1] == "TPU v", kind
  return kind + suffix


def run_benchmark_and_profile(
    run_fn: Callable[..., Any],
    fn_args: tuple[Any, ...],
    kernel_name: str,
    verbose: bool = False,
    num_iters: int = 20,
    profile: bool = False,
) -> tuple[str, float | None]:
  """Runs the benchmark function with warmup and profiling/timing."""
  def _copy_arg(x):
    if isinstance(x, (jax.Array, np.ndarray)):
      return jnp.copy(x)
    return x

  if verbose:
    print("Warming up...")
  for _ in range(3):
    jax.block_until_ready(run_fn(*[_copy_arg(x) for x in fn_args]))

  if profile and xprof_session is None:
    if verbose:
      print("Xprof unavailable outside google3; falling back to timing.")
    profile = False

  if not profile:
    if verbose:
      print("Timing (Python-only)...")
    start = time.perf_counter()
    for _ in range(num_iters):
      copied_args = [_copy_arg(x) for x in fn_args]
      jax.block_until_ready(run_fn(*copied_args))
    end = time.perf_counter()
    t_us = ((end - start) * 1e6) / num_iters
    if verbose:
      print(f"Latency: {t_us} us")
    return "", t_us

  if verbose:
    print("Profiling (Xprof)...")
  xprof_sess = xprof_session.XprofSession()
  xprof_sess.start_session(
      device_name="",
      enable_python_tracer=True,
      host_trace_level=2,
      host_cpu_profile=False,
      trace_mode="TRACE_ALL",
  )

  jax.block_until_ready(run_fn(*[_copy_arg(x) for x in fn_args]))

  session_id = xprof_sess.end_session_and_get_session_id(
      tag="percale", username=""
  )

  if verbose:
    print(
        "Percale:"
        f" https://percale.corp.google.com/llo?id={session_id}&name={kernel_name}.1"
    )
    print(f"Xprof: http://xprof/?session_id={session_id}")

  t_us = None
  try:
    t_sum, cnt = get_op_runtime(kernel_name + ".1", session_id)
    t_us = t_sum / cnt if cnt > 0 else 0.0
    if verbose:
      print(f"Latency: {t_us} us, Count: {cnt}")
  except Exception as e:  # pylint: disable=broad-exception-caught
    if verbose:
      print(f"Could not extract runtime: {e}")
  return session_id, t_us


def generate_kv_slot_mapping(
    num_boundary_tokens: int,
    num_pages: int,
    page_size: int,
    slots_per_part_out: int,
) -> jax.Array:
  """Generates kv_slot_mapping for boundary tokens."""
  kv_slot_mapping_np = np.full((num_boundary_tokens,), -1, dtype=np.int32)
  boundary_count = 0
  total_slots_needed = num_boundary_tokens * slots_per_part_out
  pages_needed = (total_slots_needed + page_size - 1) // page_size
  start_page = max(0, num_pages - pages_needed)

  for t in range(num_boundary_tokens):
    kv_slot_mapping_np[t] = start_page * page_size + boundary_count
    boundary_count += slots_per_part_out
  return jnp.array(kv_slot_mapping_np, dtype=jnp.int32)


def generate_caches(
    cfgs: Any,
    num_pages: int,
) -> tuple[jax.Array, jax.Array | None]:
  """Generates empty cache and rope_cache."""
  cache = jnp.zeros(
      (num_pages, cfgs.page_size, cfgs.hbm_pack, 128),
      dtype=jnp.uint8,
  )
  if cfgs.dims.has_rope_cache:
    rope_cache = jnp.zeros(
        (num_pages, cfgs.rope_page_size, cfgs.rope_pack, 128), dtype=jnp.uint8
    )
  else:
    rope_cache = None
  return cache, rope_cache


def print_table(header, rows, *, col_width_extra=None):
  """Prints a formatted ASCII/Unicode table to stdout."""
  if col_width_extra is None:
    col_width_extra = {}
  sz = len(header)
  col_width = [(len(str(h)) + 3 + col_width_extra.get(h, 0)) for h in header]
  start_separator = "╒" + "╤".join("═" * w for w in col_width) + "╕"
  middle_separator = "╞" + "╪".join("═" * w for w in col_width) + "╡"
  end_separator = "╘" + "╧".join("═" * w for w in col_width) + "╛"
  fmt = "│" + "│".join("{{:<{}}}".format(w) for w in col_width) + "│"
  print(start_separator)
  print(fmt.format(*header))
  for row in rows:
    assert len(row) == sz
    print(middle_separator)
    print(fmt.format(*row))
  print(end_separator)