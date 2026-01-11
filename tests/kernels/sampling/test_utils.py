import functools
import gzip
import json
import os
from glob import glob
import jax
import jax.numpy as jnp
import pandas as pd


@jax.jit
def exact_match(xs, ys):
  """Check if two pytrees match exactly (including NaN positions)."""
  return jnp.array(
    jax.tree.leaves(
      jax.tree.map(lambda x, y: jnp.array_equal(x, y, equal_nan=True), xs, ys)
    )
  ).all()


def uniquely_define_topk(logits, k):
  """Ensure topk is well-defined by handling ties at the k-th boundary.

  If more than k values are >= the k-th largest value, set extras at the boundary to -inf.
  """
  boundary_val = jax.lax.sort(logits)[-k]
  mask = logits == boundary_val
  # if more than k values gt k-th largest value, set them to -inf
  k_covered = (logits > boundary_val).sum()
  mask = mask & (mask.cumsum() > k - k_covered)
  logits = jnp.where(mask, float("-inf"), logits)
  # jax.debug.print('k>=threshold {} for k={}', (logits >= boundary_val).sum(), k)
  return logits


def verify_topk_output(x, outs, axis=1):
  """Validate top-k outputs for correctness.

  Args:
      x: Input array (must be 2D)
      outs: Tuple of (values, indices) from top-k (both must be 2D)
      axis: Axis along which top-k was computed (0 or 1, default 1)
 

  Returns:
      Boolean array indicating validity for each batch element
  """
  if x.ndim != 2:
    raise ValueError(
      f"verify_topk_output only supports 2D inputs, got {x.ndim}D"
    )

  out_vals, out_indexs = outs

  if out_vals.ndim != 2 or out_indexs.ndim != 2:
    raise ValueError(
      f"verify_topk_output requires 2D outputs, got values.ndim={out_vals.ndim}, indices.ndim={out_indexs.ndim}"
    )

  batch_axis = 1 - axis

  @functools.partial(jax.vmap, in_axes=batch_axis)
  def verify_slice(x_slice, vals_slice, idxs_slice):
    k = len(vals_slice)
    n = len(x_slice)

    true_topk_vals = jax.lax.top_k(x_slice, k)[0]

    indices_mapping_valid = (x_slice[idxs_slice] == vals_slice).all()
    i = jnp.unique(idxs_slice, size=k, fill_value=-1)
    indices_bounds_valid = ((i >= 0) & (i < n)).all()
    indices_valid = indices_mapping_valid & indices_bounds_valid

    vals_valid = (vals_slice == true_topk_vals).all()
    return vals_valid & indices_valid

  return verify_slice(x, out_vals, out_indexs)


def benchmark(_run):
  """Benchmark function and print timing from profiler trace."""

  def run():
    return jax.block_until_ready(_run())

  # Warmup
  run()

  tmpdir = "."
  with jax.profiler.trace(tmpdir):
    run()

  # Find trace file
  files = glob(f"{tmpdir}/plugins/profile/*/**.json.gz", recursive=True)
  if not files:
    print("No trace file generated.")
    return

  path = sorted(files, key=os.path.getmtime)[-1]
  try:
    with gzip.open(path, "rb") as f:
      trace = json.load(f)
  except Exception as e:
    print(f"Failed to load trace: {e}")
    return

  if "traceEvents" not in trace:
    print("No traceEvents in trace.")
    return

  df = pd.DataFrame(trace["traceEvents"])
  if df.empty or "name" not in df.columns:
    print("Trace dataframe empty or no name column.")
    return

  df = df[~df.name.isna()]
  df["name"] = df.name.apply(lambda s: s.split("(")[0])

  # Look for JIT compiled functions
  mask = df.name.str.startswith("jit_")
  res = df[mask][["name", "dur"]]

  if not res.empty:
    print(res.to_string(index=False))
  else:
    print("No jit functions found in trace.")
