# SPDX-License-Identifier: Apache-2.0
"""The GDN kernel writes ONE state checkpoint per forward pass; prefix caching
resumes from a checkpoint. This pins down why those two agree.

Write side (`kernels/gdn/v3/memory_ref.py`): ``dma_size`` is non-zero only on
``is_last_tile``, so a pass that covers many mamba blocks still writes a single
state -- the one after its *last* token -- into
``write_col = (seq_len - 1) // mamba_block_size``.

Read side (`layers/vllm/custom_ops/gdn_attention_op.py`, align mode): a request
resumes from ``read_col = (num_computed - 1) // mamba_block_size``.

Those only line up because vLLM clips prefill chunks to mamba block boundaries
(`Scheduler._mamba_block_aligned_split`), so every pass that is not a prompt's
final chunk ends exactly on a block boundary and its single write lands where
the next reader expects it. The prompt's final chunk is deliberately exempt --
it ends mid-block and decode walks that slot up to the boundary before the
block can be cached.

If that clipping regresses, or if the block sizes stop being coupled, a request
resumes from a slot the kernel never wrote: uninitialised state (NaN, i.e. '!'
forever) or a recycled block holding another request's state.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

# `tpu_platform.update_block_size_for_backend` forces
# `mamba_block_size == cache_config.block_size` in align mode, so one value
# stands for both here.
BLOCK_SIZES = [128, 256]
# `max_num_batched_tokens` is user-settable and is NOT required to be a
# multiple of the block size. Budgets that are a multiple happen to land every
# chunk on a boundary on their own; the ones that are not are what actually
# exercise `_mamba_block_aligned_split`, so keep both kinds here.
CHUNK_BUDGETS = [256, 300, 1000, 2048]
PROMPT_LENS = [1, 255, 256, 257, 700, 1024, 2048, 2300, 3000, 4500, 6144, 6500]


def _read_col(num_computed: int, block_size: int) -> int:
    """Mirror of gdn_attention_op.py align-mode `read_col`."""
    return max(num_computed - 1, 0) // block_size


def _write_col(seq_len: int, block_size: int) -> int:
    """Mirror of gdn_attention_op.py align-mode `write_col`."""
    return max(seq_len - 1, 0) // block_size


def _has_initial_state(seq_len: int, query_len: int) -> bool:
    """Mirror of kernels/gdn/v3/metadata.py."""
    return (seq_len - query_len) > 0


def _make_request(prompt_len: int) -> Request:
    return Request(request_id="r0",
                   prompt_token_ids=list(range(prompt_len)),
                   sampling_params=MagicMock(),
                   pooling_params=None)


def _split(request: Request, num_new_tokens: int, block_size: int,
           chunk_budget: int) -> int:
    """Invoke the real scheduler chunk-clipping rule on a stub `self`.

    Mirrors vLLM's own `tests/v1/core/test_mamba_align_chunk_split.py`, which
    avoids standing up a whole Scheduler just to exercise this method.
    """
    stub = SimpleNamespace(
        block_size=block_size,
        cache_config=SimpleNamespace(block_size=block_size),
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=chunk_budget,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        mamba_partial_cache_hit=False,
        hash_block_size=block_size,
        # Only Kimi-KDA/flashkda sets num_prefill_checkpoint_blocks > 0.
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


def _prefill_chunk_ends(prompt_len: int, block_size: int,
                        chunk_budget: int) -> list[int]:
    """Absolute token position each prefill pass ends at, chunking the prompt
    the way the scheduler would."""
    request = _make_request(prompt_len)
    ends: list[int] = []
    computed = 0
    while computed < prompt_len:
        request.num_computed_tokens = computed
        budgeted = min(prompt_len - computed, chunk_budget)
        num_new = _split(request, budgeted, block_size, chunk_budget)
        # A zero-token split would stall; the scheduler only emits it when the
        # chunk cannot reach a cacheable boundary within budget, which cannot
        # happen while block_size <= chunk_budget.
        assert num_new > 0, (f"split returned 0 tokens at computed={computed} "
                             f"(prompt_len={prompt_len}, block={block_size}, "
                             f"budget={chunk_budget})")
        computed += num_new
        ends.append(computed)
    assert ends[-1] == prompt_len
    return ends


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("chunk_budget", CHUNK_BUDGETS)
@pytest.mark.parametrize("prompt_len", PROMPT_LENS)
def test_intermediate_prefill_chunks_end_on_a_mamba_block_boundary(
        block_size: int, chunk_budget: int, prompt_len: int):
    """Every pass but the prompt's last must end on a block boundary.

    This is what makes "one state write per forward pass" sufficient: the one
    state the kernel writes is the state at a boundary, which is exactly what a
    later resume asks for.
    """
    if block_size > chunk_budget:
        pytest.skip("a block wider than the chunk budget advances sub-block")
    ends = _prefill_chunk_ends(prompt_len, block_size, chunk_budget)
    for end in ends[:-1]:
        assert end % block_size == 0, (
            f"prefill pass ends at {end}, which is not a multiple of the "
            f"mamba block size {block_size}. The GDN kernel writes its only "
            f"checkpoint for this pass into column "
            f"{_write_col(end, block_size)}, whose contract is the state "
            f"after {(_write_col(end, block_size) + 1) * block_size} tokens, "
            f"so that column is now wrong for anyone who resumes from it.")


def _columns_written_during_prefill(prompt_len: int, block_size: int,
                                    chunk_budget: int) -> dict[int, int]:
    """``{column: tokens encoded by the state the kernel left there}``.

    One entry per prefill pass, since a pass writes exactly one checkpoint.
    """
    written: dict[int, int] = {}
    for end in _prefill_chunk_ends(prompt_len, block_size, chunk_budget):
        written[_write_col(end, block_size)] = end
    return written


def _contract_violations(written: dict[int, int], prompt_len: int,
                         block_size: int) -> list[tuple[int, int, int]]:
    """Columns holding something other than what a resume would expect.

    Column ``c`` is only ever resumed from once its block is full and hashed,
    which fixes its meaning as "the state after ``(c + 1) * block_size``
    tokens". A column whose block is not full within this prompt is never
    hashed here, so it is exempt -- that is the prompt's final, mid-block
    chunk, which decode later advances to the boundary.
    """
    violations = []
    for col, tokens in written.items():
        contract = (col + 1) * block_size
        if contract > prompt_len:
            continue  # block not full within this prompt; not cacheable yet
        if tokens != contract:
            violations.append((col, tokens, contract))
    return violations


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("chunk_budget", CHUNK_BUDGETS)
@pytest.mark.parametrize("prompt_len", PROMPT_LENS)
def test_a_cacheable_column_holds_exactly_its_contract_state(
        block_size: int, chunk_budget: int, prompt_len: int):
    """The cross-request contract: any column a later request can resume from
    must hold the state after exactly ``(col + 1) * block_size`` tokens.

    Within a single request the read and write columns agree trivially --
    ``read_col`` of one pass is by construction ``write_col`` of the previous
    one -- so that says nothing. What matters is a *different* request
    resuming from a hashed checkpoint: it derives its read column from the
    cache-hit length, not from anything this request did. So a column that can
    be hashed has to mean what the hash says it means.
    """
    if block_size > chunk_budget:
        pytest.skip("a block wider than the chunk budget advances sub-block")

    written = _columns_written_during_prefill(prompt_len, block_size,
                                              chunk_budget)
    violations = _contract_violations(written, prompt_len, block_size)
    assert not violations, (
        f"prompt_len={prompt_len}, block={block_size}, "
        f"budget={chunk_budget}: column(s) {violations} (as "
        f"(column, holds_after_n_tokens, should_hold_after_n_tokens)) can be "
        f"hashed and resumed from, but hold a state from the middle of a "
        f"block. A later request resuming there restores a truncated state.")


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_a_fresh_request_never_reads_a_state_slot(block_size: int):
    """`has_initial_state` is what keeps an unwritten pool from ever being
    read on a cold request, so a poisoned/uninitialised slot cannot reach the
    logits. Both the DMA and its consumption are gated on it
    (`memory_ref.py` `should_read`, `vmem_ldst.py` `jnp.where`)."""
    assert not _has_initial_state(seq_len=block_size, query_len=block_size)
    assert not _has_initial_state(seq_len=1, query_len=1)
    # ...and a continuation always does read one.
    assert _has_initial_state(seq_len=block_size + 1, query_len=1)


@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_the_contract_check_is_not_vacuous(block_size: int):
    """Sensitivity control: drop the block-boundary clipping and
    `_contract_violations` must fire.

    Without this, a regression that stopped splitting prefill at block
    boundaries -- the very thing that makes one write per pass sufficient --
    could leave the test above green.
    """
    # A budget deliberately not a multiple of the block size, so unclipped
    # chunking lands mid-block, which is precisely what the real split
    # rule prevents.
    prompt_len, chunk_budget = 3000, 1000
    unclipped: dict[int, int] = {}
    computed = 0
    while computed < prompt_len:
        computed = min(computed + chunk_budget, prompt_len)
        unclipped[_write_col(computed, block_size)] = computed
    assert any(
        end % block_size != 0 for end in list(unclipped.values())[:-1]), (
            "this control needs at least one misaligned intermediate chunk")

    assert _contract_violations(unclipped, prompt_len, block_size), (
        "unclipped chunking should have left a hashable column holding a "
        "mid-block state; if it does not, the contract check proves nothing")

    # ...and the real, clipped chunking of the same prompt must be clean.
    clipped = _columns_written_during_prefill(prompt_len,
                                              block_size,
                                              chunk_budget=2048)
    assert not _contract_violations(clipped, prompt_len, block_size)
