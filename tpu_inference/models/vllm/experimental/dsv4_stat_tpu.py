# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU-side binding for the shared DeepSeek-V4 intermediate-tensor probe.

The model runs under ``jax.jit``, so values cannot simply be printed. This
module reduces each tensor to ~13 scalars ON DEVICE and ships only those to a
``jax.debug.callback``. That matters: handing whole 512x4096 activations to the
host would move gigabytes during the ~44 warmup compilations before a single
real prompt is served.

The statistic set mirrors ``dsv4_stat.stats()`` (the numpy reference used on
the GPU side) key for key. Accumulation is float32 here versus float64 in
numpy; that is a ~1e-7 relative difference, which is irrelevant against the
~1e-2 divergences being hunted, but do not read the last digits as meaningful.

Enable with ``DSV4_STAT=1``; records are only written once the arm file named
by ``DSV4_STAT_ARM`` exists (see the shared module for why).
"""

import functools
import importlib.util
import os

import jax
import jax.numpy as jnp

_SHARED = os.environ.get(
    "DSV4_STAT_SHARED",
    "/home/gxd_google_com/work-dir/mmlu_probe/dsv4_stat.py")

_stat = None
_ENABLED = os.environ.get("DSV4_STAT", "") == "1"

if _ENABLED:
    try:
        _spec = importlib.util.spec_from_file_location("dsv4_stat", _SHARED)
        _stat = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_stat)
    except Exception as e:  # noqa: BLE001
        print(f"[DSV4STAT] could not load shared module {_SHARED}: {e!r}",
              flush=True)
        _ENABLED = False


def enabled() -> bool:
    return _ENABLED


def _to_jax(t):
    if t is None:
        return None
    if isinstance(t, jax.Array):
        return t
    try:
        from torchax.interop import jax_view
        return jax_view(t)
    except Exception:  # noqa: BLE001
        return None


def _last_row_mask(f):
    """Boolean mask selecting the LAST index along axis 0, broadcastable to f.

    Written as a mask rather than ``f[-1]`` on purpose. These tensors are
    sharded across the leading axis (DP attention), and slicing a single row
    out of a sharded dim raises
    ``ShardingTypeError: slicing on sharded dims where out dim (1) is not
    divisible by mesh axes (8)``. A mask + full reduction is legal under any
    sharding.
    """
    n0 = f.shape[0]
    shape1 = (n0, ) + (1, ) * (f.ndim - 1)
    return jnp.arange(n0).reshape(shape1) == (n0 - 1)


def _device_stats(x):
    """Reduce to scalars on device. Mirrors dsv4_stat.stats()."""
    f = x.astype(jnp.float32)
    if f.ndim == 1:
        f = f.reshape(1, -1)

    isfin = jnp.isfinite(f)
    n_nan = jnp.isnan(f).sum()
    n_inf = jnp.isinf(f).sum()
    cnt = jnp.maximum(isfin.sum(), 1)
    fin = jnp.where(isfin, f, jnp.float32(0))
    mean = fin.sum() / cnt
    rms = jnp.sqrt((fin * fin).sum() / cnt)
    absmax = jnp.abs(fin).max()

    # "Last row" = the last index along axis 0 (the final token), reduced over
    # all trailing dims. dsv4_stat.stats() uses the identical definition.
    rmask = _last_row_mask(f)
    lastv = jnp.where(rmask, fin, jnp.float32(0))
    lcnt = jnp.maximum(jnp.where(rmask, isfin, False).sum(), 1)
    last_rms = jnp.sqrt((lastv * lastv).sum() / lcnt)
    # Summing over axis 0 collapses the sharded axis, so the result is
    # replicated and safe to reshape/slice.
    last8 = lastv.sum(axis=0).reshape(-1)[:8]
    last8 = jnp.pad(last8, (0, max(0, 8 - last8.shape[0])))

    # PER-TOKEN-POSITION rms. This is the statistic that actually survives the
    # TPU's token padding: the runner pads a 110-token prefill out to
    # [1024, 4096], so every whole-tensor reduction above is dominated by ~914
    # rows that have no counterpart on GPU (and whose contents differ per
    # kernel -- zeros for mla_swa, large garbage for sparse_mla/mla). Reducing
    # only over the trailing dims keeps one number per token, so the analysis
    # can restrict itself to real positions. Only the leading axis is sharded,
    # so this reduction is sharding-legal; the host slices it afterwards.
    row_rms = jnp.sqrt((fin * fin).reshape(f.shape[0], -1).mean(axis=-1))
    return n_nan, n_inf, mean, rms, absmax, last_rms, last8, row_rms


_ROWCAP = int(os.environ.get("DSV4_STAT_ROWCAP", "192"))


def _callback(tag, layer, cr, shape, dtype, bump, extra, n_nan, n_inf, mean,
              rms, absmax, last_rms, last8, row_rms):
    if not _stat.armed():
        return
    rec = {
        "step": _stat.next_step(layer, tag),
        "layer": layer,
        "cr": cr,
        "tag": tag,
        "shape": list(shape),
        "dtype": dtype,
        "nan": int(n_nan),
        "inf": int(n_inf),
        "mean": float(mean),
        "rms": float(rms),
        "absmax": float(absmax),
        "last_rms": float(last_rms),
        "last8": [float(v) for v in last8],
        # Truncated host-side: the probe prompts are 91-117 tokens, so 192 rows
        # covers every real position with room to see where padding begins.
        "row_rms": [float(v) for v in row_rms[:_ROWCAP]],
    }
    if extra:
        rec.update(extra)
    _stat.write_record(rec)


def emit(tag: str,
         t,
         layer=None,
         cr=None,
         bump: bool = False,
         **extra) -> None:
    """Record float-tensor stats for ``t`` (torchax tensor or jax.Array)."""
    if not _ENABLED:
        return
    x = _to_jax(t)
    if x is None or x.size == 0:
        return
    try:
        if jnp.issubdtype(x.dtype, jnp.integer):
            emit_int(tag, x, layer=layer, cr=cr, **extra)
            return
        vals = _device_stats(x)
        # NOTE: do NOT pass ordered=True here. JAX rejects ordered effects on
        # more than one device ("ordered effects are not supported for more
        # than 1 device"), so on the 8-chip mesh it fails AOT lowering and then
        # kills engine startup outright. Ordering is instead made irrelevant by
        # deriving the step index per (layer, tag) -- see dsv4_stat.next_step.
        jax.debug.callback(
            functools.partial(_callback, tag, layer, cr, tuple(x.shape),
                              str(x.dtype), bump, extra or None), *vals)
    except Exception as e:  # noqa: BLE001
        print(f"[DSV4STAT] emit({tag}) failed: {e!r}", flush=True)


def _callback_region(tag, layer, cr, shape, dtype, lo, hi, extra, row_sum,
                     row_max, row_nz):
    if not _stat.armed():
        return
    rec = {
        "step": _stat.next_step(layer, tag),
        "layer": layer,
        "cr": cr,
        "tag": tag,
        "shape": list(shape),
        "dtype": dtype,
        "region": [lo, hi],
        # One value per CACHE BLOCK (leading axis). Truncated like row_rms --
        # a 110-token probe touches only the first block or two, and the rest
        # of a 16384-block cache is untouched memory whose contents differ
        # between runs for the same reason padded token rows do.
        "row_sum": [int(v) for v in row_sum[:_ROWCAP]],
        "row_max": [int(v) for v in row_max[:_ROWCAP]],
        "row_nz": [int(v) for v in row_nz[:_ROWCAP]],
    }
    if extra:
        rec.update(extra)
    _stat.write_record(rec)


def emit_cache_regions(tag: str,
                       t,
                       layer=None,
                       cr=None,
                       regions=(),
                       **extra) -> None:
    """Per-block byte statistics for named BYTE RANGES of a packed KV cache.

    The DSV4 CSA cache record is 640 uint8: 448 fp8 NoPE, 64 bf16 RoPE, 7 fp8
    scales, 7 e8m0 scales, then alignment padding. Comparing whole records
    across commits is useless here -- 09159b458 moved the RoPE channels out to
    a companion array, so that slot legitimately differs while the NoPE and
    scale bytes should not. Splitting by byte range is what makes the write
    side testable: if NoPE and the scales match but the attention output does
    not, the fault is entirely in the companion RoPE array.

    Masks rather than slices, for two independent reasons:
      * the cache is sharded on axis 0 (``P(ShardingAxisName.BATCH)``), and
      * a trailing-dim slice would materialise a multi-GB copy of a cache that
        is ~10 GB per layer; a masked reduction fuses and allocates nothing.
    """
    if not _ENABLED:
        return
    x = _to_jax(t)
    if x is None or x.size == 0 or x.ndim < 2:
        return
    try:
        # A "record" is one token slot, and it spans ALL dims after
        # [num_blocks, block_size] -- the caches are stored as
        # [num_blocks, page_size, 4, 128], so a record is 4*128 = 512 B, NOT
        # shape[-1] = 128. Using shape[-1] silently makes every byte range
        # wrong: offsets never exceed 127, so [0:448] selects the whole record
        # and [448:576] selects nothing at all (reads as a row of zeros that
        # looks like real data). rec_bytes also differs BETWEEN commits -- the
        # parent's CSA record is 640 B (rope inline) and the new one 512 B
        # (rope moved out) -- which is exactly why it must be derived from the
        # array rather than assumed.
        rec_bytes = 1
        for d in x.shape[2:]:
            rec_bytes *= int(d)
        if x.ndim == 2:
            rec_bytes = int(x.shape[-1])
        f = x.reshape(x.shape[0], -1).astype(jnp.int32)
        # Byte offset WITHIN a record for every flattened column, so a range
        # selects the same field of every record in the block.
        col = jnp.arange(f.shape[1], dtype=jnp.int32) % rec_bytes
        for (name, lo, hi) in regions:
            m = (col >= lo) & (col < hi)
            sel = jnp.where(m, f, 0)
            row_sum = sel.sum(axis=-1)
            row_max = jnp.where(m, f, -1).max(axis=-1)
            # Non-zero byte count separates "written" from "never touched",
            # which is what proves the companion array is actually populated.
            row_nz = (sel != 0).sum(axis=-1)
            ex = dict(extra or {})
            # Record the size actually used, so a wrong record size can never
            # again be mistaken for a measurement.
            ex["rec_bytes"] = rec_bytes
            jax.debug.callback(
                functools.partial(_callback_region, f"{tag}:{name}", layer, cr,
                                  tuple(x.shape), str(x.dtype), lo, hi, ex),
                row_sum, row_max, row_nz)
    except Exception as e:  # noqa: BLE001
        print(f"[DSV4STAT] emit_cache_regions({tag}) failed: {e!r}",
              flush=True)


def _callback_int(tag, layer, cr, shape, dtype, extra, mn, mx, n_neg1, mean,
                  last8, row_valid, row_max, row_sum):
    if not _stat.armed():
        return
    rec = {
        "step": _stat.next_step(layer, tag),
        "layer": layer,
        "cr": cr,
        "tag": tag,
        "shape": list(shape),
        "dtype": dtype,
        "int": True,
        "min": int(mn),
        "max": int(mx),
        "n_neg1": int(n_neg1),
        "mean": float(mean),
        "last8": [int(v) for v in last8],
        # Same host-side truncation as row_rms, so the two line up positionally.
        "row_valid": [int(v) for v in row_valid[:_ROWCAP]],
        "row_max": [int(v) for v in row_max[:_ROWCAP]],
        "row_sum": [int(v) for v in row_sum[:_ROWCAP]],
    }
    if extra:
        rec.update(extra)
    _stat.write_record(rec)


def emit_int(tag: str, t, layer=None, cr=None, **extra) -> None:
    """Record integer-tensor stats (top-k indices, block tables, seq lens)."""
    if not _ENABLED:
        return
    x = _to_jax(t)
    if x is None or x.size == 0:
        return
    try:
        # Deliberately NOT a plain sum: top-k index sums reach ~2.4e9, which
        # overflows int32 (jax has no int64 here without x64 enabled). The mean
        # carries the same fingerprint without wrapping.
        xi = x.reshape(1, -1) if x.ndim == 1 else x
        # Mask, not xi[-1] -- top-k indices are sharded across the leading axis
        # too, and slicing a sharded dim is illegal (see _last_row_mask).
        rmask = _last_row_mask(xi)
        lastv = jnp.where(rmask, xi, 0)
        last8 = lastv.sum(axis=0).reshape(-1)[:8]
        last8 = jnp.pad(last8, (0, max(0, 8 - last8.shape[0])))
        # PER-TOKEN-POSITION selection stats -- the integer analogue of
        # row_rms, and for the same reason. min/max/n_neg1 above reduce over
        # the whole [max_num_tokens, csa_topk] buffer, but only the first ~110
        # rows are real tokens; the rest is padding whose slots are filled with
        # stale garbage that differs per commit. A whole-tensor min/max is
        # therefore dominated by rows that affect no output token, which is how
        # a padding artefact gets mistaken for a selection change.
        #
        #   row_valid  how many keys this token actually selected (slots >= 0)
        #   row_max    the largest key index it selected
        #   row_sum    order-insensitive fingerprint of the selected SET, so
        #              two runs that pick the same keys in a different order
        #              still compare equal (bounded well inside int32: index
        #              <= seq_len, count <= csa_topk)
        flat = xi.reshape(xi.shape[0], -1)
        valid = flat >= 0
        row_valid = valid.sum(axis=-1)
        row_max = jnp.where(valid, flat, -1).max(axis=-1)
        row_sum = jnp.where(valid, flat, 0).sum(axis=-1)
        vals = (x.min(), x.max(), (x == -1).sum(),
                x.astype(jnp.float32).mean(), last8, row_valid, row_max,
                row_sum)
        # No ordered=True -- see the note in emit(); it is illegal on a
        # multi-device mesh and step ordering is handled by next_step().
        jax.debug.callback(
            functools.partial(_callback_int, tag, layer, cr, tuple(x.shape),
                              str(x.dtype), extra or None), *vals)
    except Exception as e:  # noqa: BLE001
        print(f"[DSV4STAT] emit_int({tag}) failed: {e!r}", flush=True)
