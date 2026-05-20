"""Minimal repro: lax.while_loop forces an explicit relayout copy on a
weight that the same matmul would consume "for free" without the loop.

Run on TPU. Compares the optimized HLO of:
  no_loop:   x @ W.T                          (one matmul)
  with_loop: while_loop body doing x @ W.T    (matmul inside body, W closed over)

Look for: a `copy(...)` op on the [OUT, HIDDEN] weight that appears in
with_loop's HLO but not no_loop's. That copy is the in-graph relayout
inserted because the while-loop boundary pins the carried/loop-invariant
weight to one fixed layout, which doesn't match what the matmul fusion
inside the body prefers.
"""

import jax
import jax.numpy as jnp
from jax.experimental.layout import Format, Layout
from jax._src.layout import AutoLayout

BATCH, HIDDEN, OUT, STEPS = 16, 2048, 5120, 8
DTYPE = jnp.bfloat16

PROFILE_ROOT = "gs://wenxindong-vm/trace/debug_copy"
PROFILE_ITERS = 10  # number of timed iterations inside the trace


def no_loop(W, x):
    return x @ W.T  # [B,H] @ [H,O] -> [B,O]


def with_loop(W, x):
    def body(state):
        i, x = state
        y = x @ W.T        # [B, OUT]   <-- the matmul we care about
        x_next = y @ W      # back to [B, HIDDEN] so the carry shape is stable
        return i + 1, x_next

    def cond(state):
        i, _ = state
        return i < STEPS

    _, x_final = jax.lax.while_loop(cond, body, (jnp.int32(0), x))
    return x_final


def dump_hlo_from_compiled(label, compiled):
    text = compiled.as_text()
    n_copy = sum(1 for line in text.splitlines() if " copy(" in line)
    n_weight_copy = sum(
        1 for line in text.splitlines()
        if " copy(" in line and f"[{OUT},{HIDDEN}]" in line)
    print(f"\n===== {label} =====")
    print(f"total copy ops:           {n_copy}")
    print(f"copy ops on weight shape: {n_weight_copy}")
    for line in text.splitlines():
        if f"[{OUT},{HIDDEN}]" in line and (
                "copy(" in line or "parameter" in line or "while" in line
                or "fusion" in line or "ENTRY" in line):
            print(line.strip()[:260])


def dump_hlo(label, jitted, *args):
    compiled = jitted.lower(*args).compile()
    dump_hlo_from_compiled(label, compiled)
    return compiled


def warmup(jitted, *args):
    out = jitted(*args)
    jax.block_until_ready(out)


def run_n(label, jitted, *args):
    for i in range(PROFILE_ITERS):
        with jax.profiler.TraceAnnotation(f"{label}_call_{i}"):
            out = jitted(*args)
    jax.block_until_ready(out)


W = jnp.zeros((OUT, HIDDEN), dtype=DTYPE)
x = jnp.zeros((BATCH, HIDDEN), dtype=DTYPE)

# Plain jits (entry param layout is whatever the caller passes — T(8,128) here).
no_loop_jit = jax.jit(no_loop)
with_loop_jit = jax.jit(with_loop)

# Fix per XLA-team recommendation: pin the entry-param layout of W to the
# tile the in-loop matmul/carry wants. Two variants tried:
#   1. AUTO   — let XLA pick the layout for the entry param.
#   2. Explicit T(16,128) — manually set major_to_minor=(0,1), tiling=(16,128).
# We then `jax.device_put` W into that layout once and call the jit with the
# relaid array, so no ENTRY bridge copy is needed at runtime.
default_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

auto_W_fmt = Format(AutoLayout(), default_sharding)
with_loop_auto_jit = jax.jit(with_loop, in_shardings=(auto_W_fmt, None))

# Explicit T(16,128) with minor-to-major {1,0} — i.e. major_to_minor=(0,1).
explicit_W_layout = Layout(major_to_minor=(0, 1), tiling=((16, 128), (2, 1)))
explicit_W_fmt = Format(explicit_W_layout, default_sharding)
with_loop_explicit_jit = jax.jit(with_loop,
                                  in_shardings=(explicit_W_fmt, None))

# Discover layouts via ShapeDtypeStruct lowering (AUTO + concrete array is
# rejected, so we lower with abstract shapes).
W_sds = jax.ShapeDtypeStruct(W.shape, W.dtype)
x_sds = jax.ShapeDtypeStruct(x.shape, x.dtype)
auto_compiled = with_loop_auto_jit.lower(W_sds, x_sds).compile()
explicit_compiled = with_loop_explicit_jit.lower(W_sds, x_sds).compile()
auto_chosen_W_fmt = auto_compiled.input_formats[0][0]
explicit_chosen_W_fmt = explicit_compiled.input_formats[0][0]
print(f"\nAUTO-chosen Format for W:    {auto_chosen_W_fmt}")
print(f"Explicit-chosen Format for W: {explicit_chosen_W_fmt}")

W_auto = jax.device_put(W, auto_chosen_W_fmt)
W_explicit = jax.device_put(W, explicit_chosen_W_fmt)

no_loop_c = dump_hlo("no_loop", no_loop_jit, W, x)
with_loop_c = dump_hlo("with_loop", with_loop_jit, W, x)
dump_hlo_from_compiled("with_loop_auto", auto_compiled)
dump_hlo_from_compiled("with_loop_explicit_T16x128", explicit_compiled)

# Skip with_loop_auto from runtime profiling: AUTO declared at jit-level
# refuses concrete jax.Array inputs, and its HLO is identical to the plain
# with_loop baseline above (AUTO picked T(8,128), same as default). Profile
# only the three variants that actually differ.
warmup(no_loop_jit, W, x)
warmup(with_loop_jit, W, x)
warmup(with_loop_explicit_jit, W_explicit, x)

print(f"\n>>> profiling runs into {PROFILE_ROOT} ({PROFILE_ITERS} iters each)")
jax.profiler.start_trace(PROFILE_ROOT)
try:
    with jax.profiler.TraceAnnotation("no_loop"):
        run_n("no_loop", no_loop_jit, W, x)
    with jax.profiler.TraceAnnotation("with_loop"):
        run_n("with_loop", with_loop_jit, W, x)
    with jax.profiler.TraceAnnotation("with_loop_explicit"):
        run_n("with_loop_explicit", with_loop_explicit_jit, W_explicit, x)
finally:
    jax.profiler.stop_trace()

print(f"<<< saved combined trace under {PROFILE_ROOT}")
