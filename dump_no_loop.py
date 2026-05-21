"""Minimal no-loop variant for all-pass HLO dumping.

Run with:
  XLA_FLAGS="--xla_dump_to=hlo_dump_no_loop --xla_dump_hlo_pass_re=.* \
             --xla_dump_hlo_as_text" python dump_no_loop.py
"""
import jax
import jax.numpy as jnp

BATCH, HIDDEN, OUT = 16, 2048, 5120
DTYPE = jnp.bfloat16


@jax.jit
def no_loop(W, x):
    return x @ W.T  # [B,H] @ [H,O] -> [B,O]


W = jnp.zeros((OUT, HIDDEN), dtype=DTYPE)
x = jnp.zeros((BATCH, HIDDEN), dtype=DTYPE)
out = no_loop(W, x)
jax.block_until_ready(out)
print("no_loop compiled and ran; HLO dumped via XLA_FLAGS.")
