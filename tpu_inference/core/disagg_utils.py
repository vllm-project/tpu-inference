# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from tpu_inference import envs


def is_disagg_enabled() -> bool:
    # We triggrer our code path as long as prefill slices are set. This
    # allows us to test interleave mode effectively with the code path
    # for comparison purposes.
    return bool(envs.PREFILL_SLICES)


def _parse_slices(slices_str: str) -> Tuple[int, ...]:
    """Parse slices environment variable and return the a list of integers, each the size of a slice.

    For example, if slices_str is set to `2x2,2x1,2x4`, we should return `(4, 2, 8)`.

    Throws exception if the slice str is malformed.
    """
    if not slices_str:
        return ()

    try:
        slice_sizes = []
        for s in slices_str.split(','):
            dims = s.split('x')
            if len(dims) == 1:
                slice_sizes.append(int(dims[0]))
            elif len(dims) == 2:
                slice_sizes.append((int(dims[0]), int(dims[1])))
            else:
                raise ValueError("Each slice must be in 'N' or 'NxM' format.")
        return tuple(slice_sizes)
    except ValueError as e:
        raise ValueError(f"Malformed slice string: '{slices_str}'") from e


def get_prefill_slices() -> Tuple[int, ...]:
    if not envs.PREFILL_SLICES:
        return ()
    return _parse_slices(envs.PREFILL_SLICES)


def get_decode_slices() -> Tuple[int, ...]:
    if not envs.DECODE_SLICES:
        return ()
    return _parse_slices(envs.DECODE_SLICES)


# The hints resolved for this worker, remembered so that the runner can check
# every batch against them on the host without needing the config. Stays None
# on workers that never traced a GDN model.
_RESOLVED_GDN_SEGMENT_HINTS: Tuple[bool, bool] | None = None


def get_active_gdn_segment_hints(vllm_config=None) -> Tuple[bool, bool] | None:
    """Return resolved GDN segment hints, lazily resolving if config/env is present."""
    global _RESOLVED_GDN_SEGMENT_HINTS
    if _RESOLVED_GDN_SEGMENT_HINTS is None and (
            vllm_config is not None or envs.GDN_DISAGG_SEGMENTS is not None):
        _RESOLVED_GDN_SEGMENT_HINTS = _resolve_gdn_segment_hints(vllm_config)
    return _RESOLVED_GDN_SEGMENT_HINTS


def assert_batch_matches_gdn_segments(num_decode: int,
                                      num_reqs: int,
                                      vllm_config=None) -> None:
    """Fail loudly if a batch contains a segment this worker does not emit.

    `get_gdn_segment_hints` lets a disaggregated worker drop the kernel for the
    batch segment its role never produces. That assumption is cheap to verify
    on the host, and doing so turns a silently-skipped sequence (wrong tokens,
    no error) into a crash.
    """
    hints = get_active_gdn_segment_hints(vllm_config)
    if hints is None:
        return
    has_decode_seqs, has_prefill_seqs = hints
    if not has_decode_seqs and num_decode > 0:
        raise RuntimeError(
            f"GDN is configured for a prefill-only worker "
            f"(GDN_DISAGG_SEGMENTS={envs.GDN_DISAGG_SEGMENTS}) but the batch "
            f"has {num_decode} decode sequence(s). Set "
            "GDN_DISAGG_SEGMENTS=both to disable this optimization.")
    if not has_prefill_seqs and num_reqs > num_decode:
        raise RuntimeError(
            f"GDN is configured for a decode-only worker "
            f"(GDN_DISAGG_SEGMENTS={envs.GDN_DISAGG_SEGMENTS}) but the batch "
            f"has {num_reqs - num_decode} prefill sequence(s). Set "
            "GDN_DISAGG_SEGMENTS=both to disable this optimization.")


def get_gdn_segment_hints(vllm_config) -> Tuple[bool, bool]:
    """Return ``(has_decode_seqs, has_prefill_seqs)`` for this worker.

    The GDN kernel runs two `pallas_call`s per layer, one over the decode
    segment of the batch and one over the prefill/mixed segment. Under
    prefill/decode disaggregation a worker only ever sees one of the two, and
    the other launch is pure overhead (up to ~130us per layer on an 8K-token
    prefill). Because the role is fixed for the lifetime of the worker it can
    be resolved here, at trace time, and the empty launch simply not emitted.

    Dropping a segment is unconditional: a worker that declared itself
    decode-only would silently skip any prefill sequence it was handed. The
    runner therefore asserts the invariant on the host every step (see
    `PersistentBatchManager._reorder_batch`), so a violated assumption fails
    loudly instead of corrupting output. Set ``GDN_DISAGG_SEGMENTS=both`` to
    disable the optimization entirely, or to ``prefill`` / ``decode`` to force
    a role that cannot be derived from the config (for example the in-process
    `PREFILL_SLICES` / `DECODE_SLICES` layout).

    Returns:
        A pair of booleans for `fused_conv1d_gdn`'s `has_decode_seqs` and
        `has_prefill_seqs` arguments. ``(True, True)`` (emit both) whenever the
        worker's role cannot be determined, which is the safe default and the
        behaviour for non-disaggregated serving.
    """
    global _RESOLVED_GDN_SEGMENT_HINTS
    _RESOLVED_GDN_SEGMENT_HINTS = _resolve_gdn_segment_hints(vllm_config)
    return _RESOLVED_GDN_SEGMENT_HINTS


def _resolve_gdn_segment_hints(vllm_config) -> Tuple[bool, bool]:
    setting = (envs.GDN_DISAGG_SEGMENTS or "auto").lower()
    if setting == "both":
        return (True, True)
    if setting == "prefill":
        return (False, True)
    if setting == "decode":
        return (True, False)
    if setting != "auto":
        raise ValueError(
            f"Invalid GDN_DISAGG_SEGMENTS={setting!r}; expected one of "
            "'auto', 'both', 'prefill', 'decode'.")

    # "auto": derive from the KV-transfer role of a standard vLLM P/D
    # deployment, where the prefill instance is the KV producer and the decode
    # instance is the consumer. When kv_role='kv_both', both flags are True and
    # the worker must emit both segments.
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return (True, True)
    if kv_transfer_config.is_kv_producer and kv_transfer_config.is_kv_consumer:
        return (True, True)
    if kv_transfer_config.is_kv_producer:
        return (False, True)
    if kv_transfer_config.is_kv_consumer:
        return (True, False)
    return (True, True)
