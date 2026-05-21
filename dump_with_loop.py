"""Minimal with-loop (baseline, no layout addons) variant for all-pass HLO
dumping. This is the version that exhibits the ENTRY relayout copy in front
of the while op.

Run with:
  XLA_FLAGS="--xla_dump_to=hlo_dump_with_loop --xla_dump_hlo_pass_re=.* \
             --xla_dump_hlo_as_text" python dump_with_loop.py
"""
import jax
import jax.numpy as jnp

BATCH, HIDDEN, OUT, STEPS = 16, 2048, 5120, 8
DTYPE = jnp.bfloat16


@jax.jit
def with_loop(W, x):
    def body(state):
        i, x = state
        y = x @ W.T        # [B, OUT]  -- the matmul we care about
        x_next = y @ W     # back to [B, HIDDEN] so carry shape is stable
        return i + 1, x_next

    def cond(state):
        i, _ = state
        return i < STEPS

    _, x_final = jax.lax.while_loop(cond, body, (jnp.int32(0), x))
    return x_final


W = jnp.zeros((OUT, HIDDEN), dtype=DTYPE)
x = jnp.zeros((BATCH, HIDDEN), dtype=DTYPE)
out = with_loop(W, x)
jax.block_until_ready(out)
print("with_loop compiled and ran; HLO dumped via XLA_FLAGS.")
