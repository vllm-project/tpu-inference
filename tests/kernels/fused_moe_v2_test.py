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
"""Unit tests for the fused expert-parallel MoE kernel's host-side pieces.

Covers pallas_select, the sizing helpers, the routing tables and the three
shard cuts taken from them, the expert visit list, rowquant_fp8 and the
expert FFN body both weight formats share."""

import contextlib
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax.experimental import pallas as pl

from tpu_inference.kernels.fused_moe.v2 import host
from tpu_inference.kernels.fused_moe.v2 import kernel as v2_kernel
from tpu_inference.kernels.fused_moe.v2 import layer as v2_layer
from tpu_inference.kernels.fused_moe.v2.host import (
    BF16, FP4, FP8, FP8_MAX, HIDDEN_LANE_BLOCK, INT8, MAX_ROUTING_BLOCK,
    ROWBLK, WEIGHT_FORMAT_NAMES, WEIGHT_FORMS, WIDEN_KCHUNK, WeightFormat,
    align_up, build_routing_tables, expert_visit_list, ragged_stride_bound,
    routing_block, row_lane_blocks, scale_mirror_lanes, shard_expert_slabs,
    shard_push_tables_in_rows, shard_transport_tables_in_blocks,
    weight_format_of_dtype)
from tpu_inference.kernels.fused_moe.v2.kernel import rowquant_fp8
from tpu_inference.kernels.fused_moe.v2.layer import (_combine_arrivals,
                                                      fused_ep_moe_v2)
from tpu_inference.kernels.fused_moe.v2.router_ops import NEG, pallas_select
from tpu_inference.kernels.megablox.gmm_v2_fused_support import swigluoai

jax.config.parse_flags_with_absl()

# How far the biased FFN body may sit from a reference that biases the
# dequantized halves. Both sides are host arithmetic; the gap is the
# intermediate bf16 requantization. Measured 0.0231 on the four cases below,
# and a gate bias scaled by 1.05 reads 0.0955, so the bound separates them.
FFN_BIAS_RELATIVE_BOUND = 0.05

# How far a row-scaled intermediate value may sit from the value it came
# from, as a share of its row's largest. e4m3 carries three mantissa bits,
# so a value lands within 2^-4 of itself, and no value exceeds the row
# maximum the scale is taken from.
MID_QUANT_ROW_BOUND = 0.07

# Served geometry, for the device-marked cases.
SERVED_EXPERTS = 512
SERVED_TOPK = 10
SERVED_EP = 8


@contextlib.contextmanager
def interpret_pallas():
    """Run pallas_call under interpret for the duration of the block."""
    real = pl.pallas_call

    def interpreted(*args, **kwargs):
        kwargs.setdefault("interpret", True)
        return real(*args, **kwargs)

    with mock.patch.object(pl, "pallas_call", interpreted):
        yield


def ref_top_k(scores, topk):
    """Reference selection: XLA's sort-based top-k."""
    weights, indices = jax.lax.top_k(scores, topk)
    return np.asarray(weights), np.asarray(indices).astype(np.int32)


def ref_rowquant_scale(x_bf16):
    """Reference row scale from an f32-widened reduce."""
    x32 = x_bf16.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x32), axis=-1, keepdims=True)
    return amax / FP8_MAX


def softmax_scores(logits):
    return jax.nn.softmax(jnp.asarray(logits, jnp.float32), axis=-1)


def scatter_target_counts(slab_row, rows_alloc):
    """How many routed pairs land on each slab row."""
    counts = jnp.zeros((rows_alloc + 1, ), jnp.int32).at[slab_row].add(1)
    return np.asarray(counts)


def slab_stride(tokens, topk, e_total, tile_m):
    """The per-shard stride the tables are built against."""
    return align_up(tokens * topk + (ROWBLK - 1) * e_total + tile_m, tile_m)


def token_gather_all(routing, *, ep, stride):
    """The per-shard gather tables, laid back out as one ep-wide table.

    The gather table is no longer carried on the routing tables as one
    replicated field. It is built per shard, which is the whole point of the
    change: the plan stops being constructed against a globally indexed
    destination. A test that wants the old whole-batch view reassembles it
    from the shards rather than reading a field that no longer exists.
    """
    return np.concatenate([
        np.asarray(
            host.shard_token_gather(routing, jnp.int32(s),
                                    shard_stride=stride)) for s in range(ep)
    ])


def make_routing(idx, *, e_total, ep, block, tile_m):
    """The tables as the serving layer builds them: strided per-shard slabs."""
    tokens, topk = idx.shape
    stride = slab_stride(tokens, topk, e_total, tile_m)
    return build_routing_tables(jnp.asarray(idx, jnp.int32),
                                e_total=e_total,
                                ep=ep,
                                t_local=tokens // ep,
                                block=block,
                                tile_m=tile_m,
                                shard_stride=stride)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RouterSelectionTest(jtu.JaxTestCase):
    """pallas_select: the docstring contract and the NaN-row guarantee."""

    @parameterized.named_parameters(
        dict(testcase_name="_one_block",
             rows=32,
             experts=64,
             topk=4,
             block_rows=32),
        dict(testcase_name="_multi_block",
             rows=64,
             experts=32,
             topk=6,
             block_rows=16),
        dict(testcase_name="_topk_one",
             rows=16,
             experts=128,
             topk=1,
             block_rows=16),
    )
    def test_matches_lax_top_k_on_ordinary_scores(self, rows, experts, topk,
                                                  block_rows):
        """The stated contract: same weights, same indices, same bits."""
        rng = np.random.default_rng(0)
        scores = softmax_scores(rng.normal(size=(rows, experts)))
        with interpret_pallas():
            weights, indices = pallas_select(scores,
                                             topk=topk,
                                             block_rows=block_rows)
        want_w, want_i = ref_top_k(scores, topk)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)

    def test_ties_break_on_the_lowest_index(self):
        """Repeated scores select the lowest columns, as lax.top_k does."""
        rng = np.random.default_rng(1)
        rows, experts, topk = 32, 24, 8
        logits = rng.choice(np.array([-1.0, 0.0, 1.0], np.float32),
                            size=(rows, experts))
        scores = softmax_scores(logits)
        # Guard the premise: without ties this test proves nothing.
        self.assertLess(len(np.unique(np.asarray(scores)[0])), experts)
        with interpret_pallas():
            weights, indices = pallas_select(scores,
                                             topk=topk,
                                             block_rows=rows)
        want_w, want_i = ref_top_k(scores, topk)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)

    def test_nan_row_indices_stay_inside_the_expert_range(self):
        """A row of NaN scores must still name real experts."""
        rng = np.random.default_rng(2)
        rows, experts, topk = 32, 64, 6
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        nan_row = 7
        logits[nan_row, 3] = np.inf
        scores = softmax_scores(logits)
        self.assertTrue(np.isnan(np.asarray(scores)[nan_row]).all())
        with interpret_pallas():
            _, indices = pallas_select(scores, topk=topk, block_rows=rows)
        indices = np.asarray(indices)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < experts).all(),
                        f"out-of-range expert ids: {indices.max()}")

    def test_nan_row_selects_the_lowest_expert_with_defined_weights(self):
        """A NaN row selects expert 0 with sentinel weights, and
        renormalizing those weights gives exactly zero."""
        rng = np.random.default_rng(3)
        rows, experts, topk = 16, 32, 5
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        nan_row = 4
        logits[nan_row, 0] = np.inf
        scores = softmax_scores(logits)
        with interpret_pallas():
            weights, indices = pallas_select(scores,
                                             topk=topk,
                                             block_rows=rows)
        np.testing.assert_array_equal(
            np.asarray(indices)[nan_row], np.zeros((topk, ), np.int32))
        row_w = np.asarray(weights)[nan_row]
        np.testing.assert_array_equal(row_w, np.full((topk, ), NEG,
                                                     np.float32))
        self.assertTrue(np.isfinite(row_w).all())
        renormalized = weights / weights.sum(axis=-1, keepdims=True)
        renormalized = np.asarray(renormalized)
        self.assertTrue(np.isfinite(renormalized).all())
        np.testing.assert_array_equal(renormalized[nan_row],
                                      np.zeros((topk, ), np.float32))

    def test_the_drop_is_a_mask_rather_than_a_float32_accident(self):
        """The zeroing used to be an arithmetic accident: two sentinel
        weights summed to -inf in float32 and the renormalization returned
        exactly zero. Widen that one reduction and the same row renormalizes
        to full weight on expert zero instead -- so the guarantee depended on
        the accumulation dtype rather than on the design. The layer masks the
        row explicitly now, and this pins the difference."""
        rng = np.random.default_rng(23)
        rows, experts, topk = 8, 32, 4
        logits = rng.normal(size=(rows, experts)).astype(np.float32)
        nan_row = 3
        logits[nan_row, 0] = np.inf
        scores = softmax_scores(logits)
        with interpret_pallas():
            weights, _ = pallas_select(scores, topk=topk, block_rows=rows)

        # What the old guarantee rested on, and what a wider accumulator
        # does to it.
        wide = np.asarray(weights, np.float64)
        wide_renorm = wide / wide.sum(axis=-1, keepdims=True)
        self.assertAlmostEqual(float(wide_renorm[nan_row, 0]), 1.0 / topk)

        # The mask the layer applies, on the same weights.
        routes = np.any(np.isfinite(np.asarray(scores)),
                        axis=-1,
                        keepdims=True)
        self.assertFalse(bool(routes[nan_row, 0]))
        denom = np.where(routes, wide.sum(axis=-1, keepdims=True), 1.0)
        masked = np.where(routes, wide / denom, 0.0)
        np.testing.assert_array_equal(masked[nan_row],
                                      np.zeros((topk, ), np.float64))
        keep = [r for r in range(rows) if r != nan_row]
        np.testing.assert_allclose(masked[keep], wide_renorm[keep])
        self.assertEqual(int(np.count_nonzero(~routes)), 1)

    def test_nan_row_leaves_every_other_row_bitwise_unchanged(self):
        """Token locality: one bad row may not touch its neighbours."""
        rng = np.random.default_rng(4)
        rows, experts, topk = 32, 64, 6
        clean = rng.normal(size=(rows, experts)).astype(np.float32)
        poisoned = clean.copy()
        nan_row = 11
        poisoned[nan_row, 5] = np.inf
        with interpret_pallas():
            w_clean, i_clean = pallas_select(softmax_scores(clean),
                                             topk=topk,
                                             block_rows=rows)
            w_bad, i_bad = pallas_select(softmax_scores(poisoned),
                                         topk=topk,
                                         block_rows=rows)
        keep = [r for r in range(rows) if r != nan_row]
        np.testing.assert_array_equal(
            np.asarray(i_bad)[keep],
            np.asarray(i_clean)[keep])
        np.testing.assert_array_equal(
            np.asarray(w_bad)[keep],
            np.asarray(w_clean)[keep])

    def test_served_geometry_matches_lax_top_k(self):
        """The served router shape, on the device it is served on."""
        if not jtu.is_device_tpu_at_least(version=7):
            self.skipTest("Expect TPUv7+")
        rng = np.random.default_rng(5)
        rows = 8192 // SERVED_EP
        scores = softmax_scores(rng.normal(size=(rows, SERVED_EXPERTS)))
        weights, indices = pallas_select(scores,
                                         topk=SERVED_TOPK,
                                         block_rows=256)
        want_w, want_i = ref_top_k(scores, SERVED_TOPK)
        np.testing.assert_array_equal(np.asarray(indices), want_i)
        np.testing.assert_array_equal(np.asarray(weights), want_w)


# The routings the table below is asked to place, as (case, seed) builders.


def random_idx(case, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, case["e_total"], (case["tokens"], case["topk"]))


def one_expert_idx(expert):
    return lambda case, seed: np.full(
        (case["tokens"], case["topk"]), expert, np.int32)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RoutingTablesTest(jtu.JaxTestCase):
    """build_routing_tables: one slab row per routed pair, and no more."""

    # yapf: disable
    @parameterized.named_parameters(
        dict(testcase_name="_e8_ep4", e_total=8, ep=4, tokens=32, topk=2,
             block=8, tile_m=32),
        dict(testcase_name="_e16_ep8", e_total=16, ep=8, tokens=64, topk=4,
             block=32, tile_m=32),
        dict(testcase_name="_e4_ep2", e_total=4, ep=2, tokens=16, topk=3,
             block=8, tile_m=16),
        dict(testcase_name="_e64_ep8", e_total=64, ep=8, tokens=64, topk=6,
             block=16, tile_m=64),
        # The skewed extreme, on the first, the last and a mid-shard expert.
        dict(testcase_name="_all_on_the_first_expert", e_total=8, ep=4,
             tokens=32, topk=2, block=8, tile_m=32, idx=one_expert_idx(0)),
        dict(testcase_name="_all_on_the_last_expert", e_total=8, ep=4,
             tokens=32, topk=2, block=8, tile_m=32, idx=one_expert_idx(7)),
        dict(testcase_name="_all_on_a_mid_shard_expert", e_total=8, ep=4,
             tokens=32, topk=2, block=8, tile_m=32, idx=one_expert_idx(3)),
        dict(testcase_name="_served", e_total=SERVED_EXPERTS, ep=SERVED_EP,
             tokens=8192, topk=SERVED_TOPK, block=256, tile_m=128,
             seeds=(11, )),
    )
    # yapf: enable
    def test_every_routed_pair_lands_on_a_slab_row_of_its_own(
            self, idx=random_idx, seeds=range(4), **case):
        """Injectivity, and what it buys: each slab row holds exactly one
        token, so every token's table entries are recovered from it."""
        for seed in seeds:
            routing = make_routing(idx(case, seed),
                                   e_total=case["e_total"],
                                   ep=case["ep"],
                                   block=case["block"],
                                   tile_m=case["tile_m"])
            counts = scatter_target_counts(routing.slab_row,
                                           routing.rows_alloc)
            self.assertEqual(
                int(counts.max()), 1,
                f"two routed pairs share a slab row (seed {seed})")
            self.assertEqual(int(counts.sum()), case["tokens"] * case["topk"])
            slab_row = np.asarray(routing.slab_row)
            token_of_pair = np.arange(
                case["tokens"] * case["topk"]) // case["topk"]
            stride = slab_stride(case["tokens"], case["topk"], case["e_total"],
                                 case["tile_m"])
            np.testing.assert_array_equal(
                token_gather_all(routing, ep=case["ep"],
                                 stride=stride)[slab_row], token_of_pair)

    def test_an_empty_batch_is_refused_by_the_table_builder(self):
        """This is the defence that is actually load-bearing for T = 0. The
        serving adapter answers the same condition, but that is a different
        function and not the one a direct caller of the kernel package
        reaches."""
        with self.assertRaisesRegex(ValueError, "at least one token"):
            build_routing_tables(jnp.zeros((0, 2), jnp.int32),
                                 e_total=4,
                                 ep=2,
                                 t_local=0,
                                 block=8,
                                 tile_m=16,
                                 shard_stride=16)

    def test_the_gather_index_stays_inside_the_token_range(self):
        """The kernel fetches an input row by this value with bounds checks
        off, and two things hold it in range. The clamp bounds, whose upper
        end has to be floored at zero to say anything at all: at T = 0 it
        was T - 1 = -1, and a lower bound above the upper bound yields the
        upper bound, so the two-sided form returned -1 for every entry. And
        the table itself, which clamps an expert id past the last expert;
        the acceptance check refuses a gating width that could make one, and
        this is the backstop."""
        for tokens in (0, 1, 8, 8192):
            lo, hi = host.gather_clamp_bounds(tokens)
            self.assertEqual(lo, 0)
            self.assertLessEqual(lo, hi, f"empty clamp range at T={tokens}")
            self.assertLessEqual(hi, max(tokens - 1, 0))
            clipped = jnp.clip(jnp.zeros((4, ), jnp.int32), jnp.int32(lo),
                               jnp.int32(hi))
            self.assertEqual(int(clipped.min()), 0)
            self.assertEqual(int(clipped.max()), 0)

        e_total, ep, tokens, topk = 8, 4, 32, 2
        rng = np.random.default_rng(9)
        idx = rng.integers(0, e_total, size=(tokens, topk))
        idx[5, 0] = e_total  # one id past the last expert
        routing = make_routing(idx, e_total=e_total, ep=ep, block=8, tile_m=32)
        gather = token_gather_all(routing,
                                  ep=ep,
                                  stride=slab_stride(tokens, topk, e_total,
                                                     32))
        self.assertGreaterEqual(int(gather.min()), 0)
        self.assertLess(int(gather.max()), tokens)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class HostGeometryTest(jtu.JaxTestCase):
    """The sizing helpers the kernel layer and the serving gate both read."""

    @parameterized.named_parameters(
        dict(testcase_name="_served", t_local=1024, topk=10),
        dict(testcase_name="_odd_topk", t_local=64, topk=3),
        dict(testcase_name="_tiny", t_local=4, topk=1),
    )
    def test_the_routing_block_is_the_widest_one_that_divides(
            self, t_local, topk):
        """It has to divide the shard's routed pairs, sit under the ceiling
        and be the widest power of two that does both."""
        block = routing_block(t_local, topk)
        self.assertEqual((t_local * topk) % block, 0)
        self.assertLessEqual(block, MAX_ROUTING_BLOCK)
        self.assertEqual(block & (block - 1), 0)
        wider = 2 * block
        self.assertTrue(wider > MAX_ROUTING_BLOCK
                        or (t_local * topk) % wider != 0)

    def test_the_stride_bound_holds_the_worst_case_and_whole_tiles(self):
        """Every routed row landing on one shard, the block-alignment
        padding and one tail-read window, rounded up to whole tiles."""
        tokens, topk, e_total, capacity = 512, 4, 32, 128
        worst = tokens * topk + (ROWBLK - 1) * e_total + capacity
        bound = ragged_stride_bound(tokens, topk, e_total, capacity)
        self.assertEqual(bound % capacity, 0)
        self.assertGreaterEqual(bound, worst)
        self.assertLess(bound - capacity, worst)

    @parameterized.named_parameters(
        dict(testcase_name="_narrow", hidden=512, scale_lanes=1),
        dict(testcase_name="_one_ratio", hidden=1024, scale_lanes=1),
        dict(testcase_name="_served", hidden=4096, scale_lanes=4),
    )
    def test_the_row_geometry_follows_the_hidden_width(self, hidden,
                                                       scale_lanes):
        """A row is a whole number of lane blocks and its scale mirror is
        never narrower than one lane."""
        self.assertEqual(row_lane_blocks(hidden) * HIDDEN_LANE_BLOCK, hidden)
        self.assertEqual(scale_mirror_lanes(hidden), scale_lanes)

    def test_the_packed_word_tiling_is_checked_against_the_device_record(self):
        """The buffers holding the packed four-bit slabs are sized off the
        device record and the block-size refusal beside them is written
        against a constant. On a generation where the two disagree the
        accounting adapts and the acceptance check does not, and the
        disagreement lands as a Mosaic slice error with nothing naming it."""

        class _Info:

            def __init__(self, tiling):
                self._tiling = tiling

            def get_sublane_tiling(self, dtype):
                return self._tiling

        self.assertIsNone(
            host.check_u32_sublane_tile(_Info(host.U32_SUBLANE_TILE)))
        with self.assertRaisesRegex(ValueError, "sublanes"):
            host.check_u32_sublane_tile(_Info(host.U32_SUBLANE_TILE * 2))


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ExpertVisitListTest(jtu.JaxTestCase):
    """expert_visit_list: which local experts the kernel visits, in order."""

    # yapf: disable
    @parameterized.named_parameters(
        dict(testcase_name="_heaviest_first", rows=[0, 24, 8, 0, 40, 8],
             n_visit=4, order=[4, 1, 2, 5]),
        dict(testcase_name="_ties_break_on_the_ascending_index",
             rows=[8, 8, 16, 8], n_visit=4, order=[2, 0, 1, 3]),
    )
    # yapf: enable
    def test_the_visit_order(self, rows, n_visit, order):
        rows = jnp.asarray(rows, jnp.int32)
        visit, visited = expert_visit_list(rows, rows.shape[0])
        self.assertEqual(int(visited[0]), n_visit)
        np.testing.assert_array_equal(np.asarray(visit)[:n_visit], order)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class ShardCutTest(jtu.JaxTestCase):
    """The three cuts that turn the replicated routing tables into one
    shard's kernel operands."""

    E_TOTAL, EP, TOKENS, TOPK = 8, 4, 32, 2
    BLOCK, TILE_M = 8, 32

    def routing_for(self, seed):
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.E_TOTAL, size=(self.TOKENS, self.TOPK))
        return make_routing(idx,
                            e_total=self.E_TOTAL,
                            ep=self.EP,
                            block=self.BLOCK,
                            tile_m=self.TILE_M)

    def cuts(self, routing, me):
        kw = dict(e_total=self.E_TOTAL, ep=self.EP)
        return (shard_transport_tables_in_blocks(routing, me, **kw),
                shard_push_tables_in_rows(routing, me, **kw),
                shard_expert_slabs(routing, me, **kw))

    @parameterized.named_parameters(("_seed0", 0), ("_seed3", 3))
    def test_the_cuts_agree_with_the_replicated_tables(self, seed):
        """Four invariants, held on every shard of the same routing: the
        expert slabs are this shard's own experts, with base each slab's
        start relative to the first of them; the per-destination commit runs
        start at zero, follow each other and cover exactly the rows the slab
        holds; the row cut and the block cut agree on where a push lands,
        the data DMA taking rows and the scale DMA blocks; and only the
        pushed lengths shrink to the true row counts, while each cut's total
        counts what that cut's DMAs move."""
        routing = self.routing_for(seed)
        g_local = self.E_TOTAL // self.EP
        expert_rows = np.asarray(routing.expert_rows_aligned)
        for me in range(self.EP):
            ((start, aligned, _, _, push_len, push_dst,
              block_totals), (true_rows, recv_row_off, row_totals),
             (rows, base)) = self.cuts(routing, me)
            start = np.asarray(start)
            aligned = np.asarray(aligned)
            true_rows = np.asarray(true_rows)

            mine = expert_rows[me * g_local:(me + 1) * g_local]
            np.testing.assert_array_equal(np.asarray(rows), mine)
            np.testing.assert_array_equal(np.asarray(base),
                                          np.cumsum(mine) - mine)

            for g in range(g_local):
                np.testing.assert_array_equal(
                    start[g],
                    np.cumsum(aligned[g]) - aligned[g])
                self.assertEqual(int(aligned[g].sum()) * ROWBLK, int(rows[g]))

            np.testing.assert_array_equal(np.asarray(recv_row_off),
                                          np.asarray(push_dst) * ROWBLK)

            np.testing.assert_array_equal(np.asarray(push_len), aligned)
            self.assertTrue((true_rows <= aligned * ROWBLK).all())
            remote = [d for d in range(self.EP) if d != me]
            self.assertEqual(int(block_totals[0]), int(aligned[:,
                                                               remote].sum()))
            self.assertEqual(int(row_totals[0]), int(true_rows[:,
                                                               remote].sum()))


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class RowQuantTest(jtu.JaxTestCase):
    """rowquant_fp8: exact scales, no overflow, and the stated flushes."""

    @parameterized.named_parameters(
        dict(testcase_name="_narrow", rows=8, cols=128),
        dict(testcase_name="_served_row", rows=16, cols=4096),
        dict(testcase_name="_wide_range", rows=32, cols=512),
    )
    def test_scale_is_exact_against_an_f32_widened_reduce(self, rows, cols):
        """abs and max are exact in bf16, so the f32 scale is identical."""
        rng = np.random.default_rng(12)
        mags = np.exp(rng.uniform(-12, 12, size=(rows, 1)))
        x = (rng.normal(size=(rows, cols)) * mags).astype(np.float32)
        x_bf16 = jnp.asarray(x).astype(jnp.bfloat16)
        _, scale = rowquant_fp8(x_bf16)
        np.testing.assert_array_equal(
            np.asarray(scale.view(jnp.int32)),
            np.asarray(ref_rowquant_scale(x_bf16).view(jnp.int32)))

    def test_zero_row_scale_is_zero_and_the_row_stays_zero(self):
        x = jnp.zeros((4, 128), jnp.bfloat16)
        quantized, scale = rowquant_fp8(x)
        np.testing.assert_array_equal(np.asarray(scale),
                                      np.zeros((4, 1), np.float32))
        np.testing.assert_array_equal(
            np.asarray(quantized.astype(jnp.float32)),
            np.zeros((4, 128), np.float32))

    def test_no_fp8_overflow_through_the_bf16_rounded_inverse_scale(self):
        """Rounding the inverse scale to bf16 never pushes a row past 448."""
        rng = np.random.default_rng(13)
        for seed_row in range(64):
            mags = np.exp(rng.uniform(-20, 20, size=(1, 1)))
            x = (rng.normal(size=(1, 256)) * mags).astype(np.float32)
            x_bf16 = jnp.asarray(x).astype(jnp.bfloat16)
            quantized, _ = rowquant_fp8(x_bf16)
            widened = np.asarray(quantized.astype(jnp.float32))
            self.assertTrue(
                np.isfinite(widened).all(), f"row {seed_row} overflowed e4m3")
            self.assertLessEqual(float(np.abs(widened).max()), FP8_MAX)

    def test_no_fp8_overflow_on_rows_whose_maximum_sits_at_the_boundary(self):
        """The adversarial case: every bf16 row maximum in a window around
        448, where the inverse scale is as close to 1 as bf16 allows."""
        candidates = np.unique(
            np.asarray(
                jnp.asarray(np.linspace(440.0, 456.0, 2048,
                                        dtype=np.float32)).astype(
                                            jnp.bfloat16).astype(jnp.float32)))
        rng = np.random.default_rng(14)
        for peak in candidates:
            row = (rng.random((1, 128)) * peak).astype(np.float32)
            row[0, 0] = peak
            x_bf16 = jnp.asarray(row).astype(jnp.bfloat16)
            quantized, scale = rowquant_fp8(x_bf16)
            widened = np.asarray(quantized.astype(jnp.float32))
            self.assertTrue(
                np.isfinite(widened).all(), f"peak {peak} overflowed e4m3")
            self.assertLessEqual(float(np.abs(widened).max()), FP8_MAX)
            self.assertGreater(float(scale[0, 0]), 0.0)

    def test_a_non_finite_row_is_zeroed_whole_and_stays_local(self):
        """A row whose maximum is not finite has no usable scale. Zeroing it
        whole, with a zero row scale, is the only one of the three outcomes
        that is honest: the reciprocal of an infinite maximum is zero, so
        keeping the multiply would leave every real value at zero and only
        the infinity surviving as a NaN -- a plausible-looking mostly-zero
        row where the row's content was -- and would send an infinite scale
        over the wire for the destination shard's combine to multiply in."""
        for seed, bad in ((15, np.inf), (16, -np.inf), (17, np.nan)):
            rng = np.random.default_rng(seed)
            rows, cols = 4, 128
            x = rng.normal(size=(rows, cols)).astype(np.float32)
            clean_q, clean_s = rowquant_fp8(
                jnp.asarray(x).astype(jnp.bfloat16))
            poisoned = x.copy()
            bad_row = 2
            poisoned[bad_row, 5] = bad
            quantized, scale = rowquant_fp8(
                jnp.asarray(poisoned).astype(jnp.bfloat16))
            widened = np.asarray(quantized.astype(jnp.float32))
            self.assertEqual(float(scale[bad_row, 0]), 0.0, bad)
            np.testing.assert_array_equal(widened[bad_row],
                                          np.zeros(cols, np.float32))
            keep = [r for r in range(rows) if r != bad_row]
            np.testing.assert_array_equal(
                widened[keep],
                np.asarray(clean_q.astype(jnp.float32))[keep])
            np.testing.assert_array_equal(
                np.asarray(scale)[keep],
                np.asarray(clean_s)[keep])

    def test_an_all_non_finite_row_is_zeroed_the_same_way(self):
        for bad in (np.inf, np.nan):
            row = np.full((1, 128), bad, np.float32)
            quantized, scale = rowquant_fp8(
                jnp.asarray(row).astype(jnp.bfloat16))
            self.assertEqual(float(scale[0, 0]), 0.0)
            np.testing.assert_array_equal(
                np.asarray(quantized.astype(jnp.float32)),
                np.zeros((1, 128), np.float32))

    def test_a_zero_row_is_still_zeroed_with_a_zero_scale(self):
        """The case that was already guarded, kept beside the two that were
        not, because all three now take the same answer."""
        quantized, scale = rowquant_fp8(jnp.zeros((1, 128), jnp.bfloat16))
        self.assertEqual(float(scale[0, 0]), 0.0)
        np.testing.assert_array_equal(
            np.asarray(quantized.astype(jnp.float32)),
            np.zeros((1, 128), np.float32))


class RequantizeIntermediateTest(jtu.JaxTestCase):
    """The chunk loop every weight format hands its accumulator to, on the
    formats that ask it to requantize what it builds."""

    ROWS = 8

    def halves(self, seed, inter):
        """A gate|up accumulator whose first chunk holds the larger values."""
        rng = np.random.default_rng(seed)
        acc = rng.normal(size=(self.ROWS, 2 * inter)).astype(np.float32)
        chunk = v2_kernel.QCHUNK
        acc[:, chunk:inter] *= 0.01
        acc[:, inter + chunk:] *= 0.01
        return jnp.asarray(acc)

    def test_a_zero_accumulator_leaves_a_zero_row_and_a_zero_scale(self):
        inter = 256
        mid_q, mid_scale = v2_kernel._intermediate_rows(jnp.zeros(
            (self.ROWS, 2 * inter), jnp.float32),
                                                        jnp.ones(
                                                            (self.ROWS, 1),
                                                            jnp.float32),
                                                        inter,
                                                        w1s=None,
                                                        act_fn="silu",
                                                        w1b=None)
        np.testing.assert_array_equal(np.asarray(mid_q.astype(jnp.float32)),
                                      np.zeros((self.ROWS, inter), np.float32))
        np.testing.assert_array_equal(np.asarray(mid_scale),
                                      np.zeros((self.ROWS, 1), np.float32))

    def test_the_row_scale_spans_every_chunk_not_just_the_last(self):
        """The loop quantizes in column chunks but the scale is one per row,
        so a row whose largest value sits in the first chunk has to be
        scaled by that value."""
        inter = 2 * v2_kernel.QCHUNK
        acc1 = self.halves(31, inter)
        act_scale = jnp.ones((self.ROWS, 1), jnp.float32)
        _, mid_scale = v2_kernel._intermediate_rows(acc1,
                                                    act_scale,
                                                    inter,
                                                    w1s=None,
                                                    act_fn="silu",
                                                    w1b=None)
        gate, up = acc1[:, :inter], acc1[:, inter:]
        mid = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
        # Guard the premise: without the first chunk dominating, a loop that
        # kept only the last chunk's maximum would pass this.
        first = jnp.max(jnp.abs(mid[:, :v2_kernel.QCHUNK]), axis=-1)
        last = jnp.max(jnp.abs(mid[:, v2_kernel.QCHUNK:]), axis=-1)
        self.assertTrue(bool(jnp.all(first > last)))
        amax = jnp.max(jnp.abs(mid), axis=-1, keepdims=True)
        np.testing.assert_array_equal(
            np.asarray(mid_scale),
            np.asarray(amax.astype(jnp.float32) / FP8_MAX))

    def test_the_quantized_intermediate_dequantizes_back_to_its_rows(self):
        """One row scale for the whole row, so what a value may miss by is
        set by the row's largest value rather than by its own."""
        inter = 2 * v2_kernel.QCHUNK
        acc1 = self.halves(32, inter)
        act_scale = jnp.ones((self.ROWS, 1), jnp.float32)
        mid_q, mid_scale = v2_kernel._intermediate_rows(acc1,
                                                        act_scale,
                                                        inter,
                                                        w1s=None,
                                                        act_fn="silu",
                                                        w1b=None)
        gate, up = acc1[:, :inter], acc1[:, inter:]
        want = (jax.nn.silu(gate) * up).astype(jnp.float32)
        got = mid_q.astype(jnp.float32) * mid_scale
        worst = jnp.max(jnp.abs(got - want), axis=-1, keepdims=True)
        amax = jnp.max(jnp.abs(want), axis=-1, keepdims=True)
        self.assertTrue(bool(jnp.all(worst <= MID_QUANT_ROW_BOUND * amax)))


class FfnBiasTest(jtu.JaxTestCase):
    """The expert biases inside the kernel's own FFN body.

    Both weight paths are plain jnp, so the placement of each bias is
    checkable on a host: the gate and up halves must be biased after their
    scales and before the activation, and the down bias must land on the
    post-scale row that goes to the wire.
    """

    ROWS = 16
    HIDDEN = 256
    INTER = 128
    QB = 128

    def operands(self, seed=0, fp4=False):
        key = jax.random.key(seed)
        kx, k1, k2, kb1, kb2 = jax.random.split(key, 5)
        x = jax.random.normal(kx, (self.ROWS, self.HIDDEN), jnp.float32) / 4
        w1 = jax.random.normal(k1,
                               (self.HIDDEN, 2 * self.INTER), jnp.float32) / 16
        w2 = jax.random.normal(k2, (self.INTER, self.HIDDEN), jnp.float32) / 16
        b1 = jax.random.normal(kb1, (1, 2 * self.INTER), jnp.float32) / 2
        b2 = jax.random.normal(kb2, (1, self.HIDDEN), jnp.float32) * 2
        q, s = rowquant_fp8(x.astype(jnp.bfloat16))
        if fp4:
            q1, s1 = self.blockquant(w1)
            q2, s2 = self.blockquant(w2)
        else:
            q1, s1 = self.chanquant(w1)
            q2, s2 = self.chanquant(w2)
        return dict(q=q, s=s, q1=q1, s1=s1, q2=q2, s2=s2, b1=b1, b2=b2)

    def chanquant(self, w):
        amax = jnp.max(jnp.abs(w), axis=0, keepdims=True)
        sc = jnp.where(amax == 0, 1.0, amax / FP8_MAX).astype(jnp.float32)
        return (w / sc).astype(FP8), sc

    def blockquant(self, w):
        kdim, n = w.shape
        wb = w.reshape(kdim // self.QB, self.QB, n)
        amax = jnp.max(jnp.abs(wb), axis=1)
        sc = jnp.where(amax == 0, 1.0, amax / 6.0).astype(jnp.float32)
        return (wb / sc[:, None, :]).astype(FP4).reshape(kdim, n), sc

    def ffn(self, op, *, fp4, act_fn, w1b):
        if fp4:

            def w_of(q):
                return lambda b: q[b * self.QB:(b + 1) * self.QB].astype(FP8)

            acc2, s2 = v2_kernel.expert_ffn_blockscale(op["q"],
                                                       op["s"],
                                                       w_of(op["q1"]),
                                                       w_of(op["q2"]),
                                                       op["s1"],
                                                       op["s2"],
                                                       qb=self.QB,
                                                       act_fn=act_fn,
                                                       w1b=w1b)
            return acc2 * s2
        acc2, s2 = v2_kernel.expert_ffn_fp8(op["q"],
                                            op["s"],
                                            op["q1"],
                                            op["q2"],
                                            op["s1"],
                                            act_fn=act_fn,
                                            w1b=w1b)
        return (acc2 * s2) * op["s2"]

    @parameterized.product(fp4=(False, True), act_fn=("silu", "swigluoai"))
    def test_a_zero_bias_leaves_the_body_bitwise_unchanged(self, fp4, act_fn):
        op = self.operands(fp4=fp4)
        plain = self.ffn(op, fp4=fp4, act_fn=act_fn, w1b=None)
        zeroed = self.ffn(op,
                          fp4=fp4,
                          act_fn=act_fn,
                          w1b=jnp.zeros_like(op["b1"]))
        self.assertArraysEqual(zeroed, plain)

    @parameterized.product(fp4=(False, True), act_fn=("silu", "swigluoai"))
    def test_the_gate_and_up_bias_land_before_the_activation(
            self, fp4, act_fn):
        """Compare against a reference that biases the dequantized halves."""
        op = self.operands(fp4=fp4)
        got = self.ffn(op, fp4=fp4, act_fn=act_fn, w1b=op["b1"])
        x = op["q"].astype(jnp.float32) * op["s"]
        if fp4:
            w1 = (op["q1"].astype(jnp.float32).reshape(
                -1, self.QB, 2 * self.INTER) * op["s1"][:, None, :]).reshape(
                    self.HIDDEN, 2 * self.INTER)
            w2 = (op["q2"].astype(jnp.float32).reshape(
                -1, self.QB, self.HIDDEN) * op["s2"][:, None, :]).reshape(
                    self.INTER, self.HIDDEN)
        else:
            w1 = op["q1"].astype(jnp.float32) * op["s1"]
            w2 = op["q2"].astype(jnp.float32) * op["s2"]
        acc1 = x @ w1 + op["b1"]
        gate, up = acc1[:, :self.INTER], acc1[:, self.INTER:]
        mid = ((jax.nn.silu(gate) *
                up) if act_fn == "silu" else swigluoai(gate, up))
        want = mid.astype(jnp.bfloat16).astype(jnp.float32) @ w2
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        self.assertLess(rel, FFN_BIAS_RELATIVE_BOUND)

    @parameterized.product(fp4=(False, True))
    def test_a_zero_down_projection_leaves_the_row_zero(self, fp4):
        """The body up to the down bias, which the tile epilogue adds rather
        than this: with the down weights zeroed the FFN contributes nothing,
        so the row reaching the bias add is zero."""
        op = self.operands(fp4=fp4)
        op = dict(op, q2=jnp.zeros_like(op["q2"]))
        row = self.ffn(op, fp4=fp4, act_fn="swigluoai", w1b=op["b1"])
        self.assertArraysEqual(
            row, jnp.zeros((self.ROWS, self.HIDDEN), jnp.float32))

    @parameterized.product(wire_dtype=(FP8, BF16))
    def test_the_wire_row_is_the_epilogue_applied_exactly_once(
            self, wire_dtype):
        """The row handed to the wire is acc2 * mid_scale * w2s + w2b, and
        the down bias enters it once. This is the one correctness claim the
        bias work makes -- expert parallelism shards the second matmul on the
        expert axis, so a routed row's whole down projection is computed in
        one place -- and until the epilogue was a function of its own the
        only thing that could see it was a device-marked test."""
        op = self.operands()
        acc2 = jax.random.normal(jax.random.key(7), (self.ROWS, self.HIDDEN),
                                 jnp.float32)
        mid_scale = jnp.full((self.ROWS, 1), 0.25, jnp.float32)
        rows, scales = v2_kernel.wire_row_and_scale(acc2,
                                                    mid_scale,
                                                    w2s=op["s2"],
                                                    w2b=op["b2"],
                                                    wire_dtype=wire_dtype,
                                                    tile_m=self.ROWS)
        want = ((acc2 * mid_scale) * op["s2"] + op["b2"]).astype(jnp.bfloat16)
        if wire_dtype is BF16:
            self.assertArraysEqual(rows, want)
            self.assertArraysEqual(scales, jnp.ones((self.ROWS, 1),
                                                    jnp.float32))
            return
        # An eight-bit wire ships the row quantized against its own scale,
        # so what has to come back is the row, not its bits.
        self.assertEqual(jnp.dtype(rows.dtype), jnp.dtype(FP8))
        got = rows.astype(jnp.float32) * scales
        amax = jnp.max(jnp.abs(want.astype(jnp.float32)),
                       axis=-1,
                       keepdims=True)
        worst = jnp.max(jnp.abs(got - want.astype(jnp.float32)),
                        axis=-1,
                        keepdims=True)
        self.assertTrue(bool(jnp.all(worst <= MID_QUANT_ROW_BOUND * amax)))

    @parameterized.product(wire_dtype=(FP8, BF16))
    def test_doubling_the_down_bias_moves_the_wire_row(self, wire_dtype):
        """The control for the case above: a bias added twice has to be
        visible, or the case above would pass on a body that added it any
        number of times.

        The comparison is on the DEQUANTIZED row rather than on the wire
        bits, and that is not incidental. An eight-bit wire takes its scale
        from the row's own maximum, so doubling every value leaves the
        quantized bits identical and moves only the scale -- a body that
        doubled the bias would be invisible to anything reading the bits."""
        op = self.operands()
        acc2 = jnp.zeros((self.ROWS, self.HIDDEN), jnp.float32)

        def row_of(bias):
            rows, scales = v2_kernel.wire_row_and_scale(acc2,
                                                        None,
                                                        w2b=bias,
                                                        wire_dtype=wire_dtype,
                                                        tile_m=self.ROWS)
            return rows.astype(jnp.float32) * scales

        once, twice = row_of(op["b2"]), row_of(2.0 * op["b2"])
        self.assertFalse(bool(jnp.allclose(once, twice)))

    def test_the_epilogue_skips_what_the_format_does_not_carry(self):
        """A format whose second matmul took bf16 rows owes no row scale, a
        four-bit build has already folded its weight scale into the block
        sums, and a model with no down bias adds nothing. Each absent
        operand has to leave the row bitwise alone."""
        op = self.operands()
        acc2 = jax.random.normal(jax.random.key(8), (self.ROWS, self.HIDDEN),
                                 jnp.float32)
        bare, _ = v2_kernel.wire_row_and_scale(acc2,
                                               None,
                                               wire_dtype=BF16,
                                               tile_m=self.ROWS)
        self.assertArraysEqual(bare, acc2.astype(jnp.bfloat16))
        ones, _ = v2_kernel.wire_row_and_scale(acc2,
                                               jnp.ones((self.ROWS, 1),
                                                        jnp.float32),
                                               w2s=jnp.ones_like(op["s2"]),
                                               w2b=jnp.zeros_like(op["b2"]),
                                               wire_dtype=BF16,
                                               tile_m=self.ROWS)
        self.assertArraysEqual(ones, bare)

    def test_the_combine_weights_each_bias_once_per_selected_expert(self):
        """Expert parallelism computes a routed row's whole down projection
        on one shard, so the bias enters that row once. What the destination
        then owes is one router weight per selection slot -- not one per
        shard, and not one per token. Rows carrying nothing but their
        expert's bias make the difference visible as a multiple."""
        tokens, topk, experts, hidden = 8, 4, 6, 256
        key = jax.random.key(3)
        kb, kw, ki = jax.random.split(key, 3)
        bias = jax.random.normal(kb, (experts, hidden), jnp.float32)
        weights = jax.nn.softmax(jax.random.normal(kw, (tokens, topk),
                                                   jnp.float32),
                                 axis=-1)
        idx = jax.random.randint(ki, (tokens, topk), 0, experts)
        rows = bias[idx.reshape(-1)]  # one arrival row per selection slot
        recv3 = rows.astype(jnp.float32).reshape(-1, hidden // 128, 128)
        aux = jnp.ones((rows.shape[0], 1), jnp.float32)
        pos = jnp.arange(tokens * topk, dtype=jnp.int32).reshape(tokens, topk)
        got = _combine_arrivals(recv3, aux, pos, weights, jnp.float32)
        # The reference is what is being compared against, so it is pinned
        # to the highest precision: at the default the einsum itself sits
        # further from exact than the combine does.
        want = jnp.einsum("tk,tkh->th",
                          weights,
                          bias[idx],
                          precision=jax.lax.Precision.HIGHEST)
        self.assertAllClose(got, want, atol=1e-5, rtol=1e-5)


class UnquantizedFfnTest(jtu.JaxTestCase):
    """The two bodies that quantize nothing, held to what that means.

    The claim these make is negative -- that no quantization and no
    dequantization is inserted around a model that did not ask for one -- so
    each is checked against a reference written with the arithmetic spelled
    out, and the eight-bit bodies beside them are checked to still insert
    theirs.
    """

    ROWS = 16
    HIDDEN = 1024
    INTER = 512

    def operands(self, seed=0):
        key = jax.random.key(seed)
        kx, k1, k2, kb1 = jax.random.split(key, 4)
        x = jax.random.normal(kx, (self.ROWS, self.HIDDEN), jnp.float32) / 4
        w1 = jax.random.normal(k1,
                               (self.HIDDEN, 2 * self.INTER), jnp.float32) / 16
        w2 = jax.random.normal(k2, (self.INTER, self.HIDDEN), jnp.float32) / 16
        b1 = jax.random.normal(kb1, (1, 2 * self.INTER), jnp.float32) / 2
        return dict(x=x.astype(jnp.bfloat16),
                    w1=w1.astype(jnp.bfloat16),
                    w2=w2.astype(jnp.bfloat16),
                    b1=b1)

    def plain_body(self, x, w1, w2, w1b=None):
        """The FFN with every step written out and nothing quantized."""
        acc1 = jax.lax.dot_general(x,
                                   w1,
                                   v2_kernel.DN,
                                   preferred_element_type=jnp.float32)
        if w1b is not None:
            gate = acc1[:, :self.INTER] + w1b[:, :self.INTER]
            up = acc1[:, self.INTER:] + w1b[:, self.INTER:]
        else:
            gate, up = acc1[:, :self.INTER], acc1[:, self.INTER:]
        mid = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
        return jax.lax.dot_general(mid,
                                   w2,
                                   v2_kernel.DN,
                                   preferred_element_type=jnp.float32)

    def test_the_unquantized_body_is_the_plain_arithmetic_bitwise(self):
        """No rounding to fp8 anywhere, and no row scale to return."""
        op = self.operands()
        got, row_scale = v2_kernel.expert_ffn_bf16(op["x"], op["w1"], op["w2"])
        self.assertIsNone(row_scale)
        self.assertArraysEqual(got, self.plain_body(op["x"], op["w1"],
                                                    op["w2"]))

    def test_the_unquantized_body_biases_before_the_activation(self):
        op = self.operands()
        got, _ = v2_kernel.expert_ffn_bf16(op["x"],
                                           op["w1"],
                                           op["w2"],
                                           w1b=op["b1"])
        self.assertArraysEqual(
            got, self.plain_body(op["x"], op["w1"], op["w2"], w1b=op["b1"]))

    def test_a_zero_bias_leaves_the_unquantized_body_unchanged(self):
        op = self.operands()
        plain, _ = v2_kernel.expert_ffn_bf16(op["x"], op["w1"], op["w2"])
        zeroed, _ = v2_kernel.expert_ffn_bf16(op["x"],
                                              op["w1"],
                                              op["w2"],
                                              w1b=jnp.zeros_like(op["b1"]))
        self.assertArraysEqual(zeroed, plain)

    def integer_weights(self, w):
        """Per-output-channel integer quantization, as a loader does it."""
        peak = float(jnp.iinfo(INT8).max)
        amax = jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0, keepdims=True)
        scale = jnp.where(amax == 0, 1.0, amax / peak).astype(jnp.float32)
        rounded = jnp.clip(jnp.round(w.astype(jnp.float32) / scale), -peak,
                           peak)
        return rounded.astype(INT8), scale

    def test_the_integer_body_in_one_chunk_is_the_widened_arithmetic(self):
        """The integer weights reach the matrix unit widened to bf16, so a
        configuration that widens the whole contraction in ONE chunk has to
        reproduce the plain body over those widened weights exactly. That is
        what says the chunking is a memory decision and not a numerics one.
        """
        op = self.operands()
        q1, s1 = self.integer_weights(op["w1"])
        q2, _ = self.integer_weights(op["w2"])

        def whole(q):
            return lambda b: q.astype(BF16)

        got, row_scale = v2_kernel.expert_ffn_int8(op["x"],
                                                   whole(q1),
                                                   whole(q2),
                                                   s1,
                                                   n_chunks1=1,
                                                   n_chunks2=1,
                                                   kb=self.HIDDEN)
        # No intermediate quantization, so no row scale. The gate/up channel
        # scale enters before the activation; the down channel scale is the
        # kernel epilogue's and is deliberately not applied here.
        self.assertIsNone(row_scale)
        self.assertArraysEqual(got, self.scaled_reference(op["x"], q1, s1, q2))

    def scaled_reference(self, x, q1, s1, q2):
        """The plain body with the gate/up channel scale applied where the
        kernel applies it: on the post-matmul halves, before the
        activation."""
        acc1 = jax.lax.dot_general(x,
                                   q1.astype(BF16),
                                   v2_kernel.DN,
                                   preferred_element_type=jnp.float32) * s1
        mid = (jax.nn.silu(acc1[:, :self.INTER]) *
               acc1[:, self.INTER:]).astype(jnp.bfloat16)
        return jax.lax.dot_general(mid,
                                   q2.astype(BF16),
                                   v2_kernel.DN,
                                   preferred_element_type=jnp.float32)

    def test_the_integer_body_chunked_tracks_the_one_chunk_form(self):
        """Chunking changes the accumulation order and nothing else, so the
        two forms sit within accumulation noise of each other rather than
        being a different computation."""
        op = self.operands()
        q1, s1 = self.integer_weights(op["w1"])
        q2, _ = self.integer_weights(op["w2"])

        def chunks(q, kb):
            return lambda b: q[b * kb:(b + 1) * kb].astype(BF16)

        one, _ = v2_kernel.expert_ffn_int8(op["x"],
                                           chunks(q1, self.HIDDEN),
                                           chunks(q2, self.INTER),
                                           s1,
                                           n_chunks1=1,
                                           n_chunks2=1,
                                           kb=max(self.HIDDEN, self.INTER))
        many, _ = v2_kernel.expert_ffn_int8(
            op["x"],
            chunks(q1, WIDEN_KCHUNK),
            chunks(q2, WIDEN_KCHUNK),
            s1,
            n_chunks1=self.HIDDEN // WIDEN_KCHUNK,
            n_chunks2=self.INTER // WIDEN_KCHUNK,
            kb=WIDEN_KCHUNK)
        self.assertAllClose(many, one, atol=1e-3, rtol=1e-3)

    def test_the_eight_bit_body_still_inserts_its_requantization(self):
        """The negative half of the claim: the un-hardcoding must not have
        removed the quantization from the format that wants it."""
        op = self.operands()
        act_q, act_scale = rowquant_fp8(op["x"])
        w1_q = op["w1"].astype(FP8)
        w2_q = op["w2"].astype(FP8)
        ones = jnp.ones((1, 2 * self.INTER), jnp.float32)
        _, row_scale = v2_kernel.expert_ffn_fp8(act_q, act_scale, w1_q, w2_q,
                                                ones)
        self.assertIsNotNone(row_scale)
        self.assertEqual(row_scale.shape, (self.ROWS, 1))


class WeightFormatTableTest(jtu.JaxTestCase):
    """The table of accepted (weight dtype, scale layout) pairs."""

    def test_the_dtype_to_format_map_is_exact(self):
        """Every format resolves from its own weight dtype, and a dtype the
        table does not carry resolves to nothing."""
        for name, form in WEIGHT_FORMS.items():
            self.assertEqual(weight_format_of_dtype(form.weight_dtype), name)
        for dtype in (jnp.int4, jnp.float8_e5m2, jnp.float32):
            self.assertIsNone(weight_format_of_dtype(dtype))

    def test_a_format_name_outside_the_table_is_refused_by_name(self):
        """The lookup is what every caller reaches the record through, so a
        name it does not carry has to say which names it does rather than
        raise a bare KeyError from inside a kernel body."""
        for name in WEIGHT_FORMAT_NAMES:
            self.assertIs(host.weight_form(name), WEIGHT_FORMS[name])
        with self.assertRaises(NotImplementedError) as caught:
            host.weight_form("fp6")
        message = str(caught.exception)
        self.assertIn("fp6", message)
        for name in WEIGHT_FORMAT_NAMES:
            self.assertIn(name, message)

    def test_the_derived_flags_follow_the_format(self):
        """The un-hardcoding, as the table states it: the activation peak is
        what says whether the rows are quantized, the activation and wire
        element types follow it, and the scale layout is what says whether
        the weights carry scales."""
        for name, form in WEIGHT_FORMS.items():
            quantized = form.act_max is not None
            self.assertEqual(form.quantized_activations, quantized, name)
            self.assertEqual(
                jnp.dtype(form.act_dtype) == jnp.dtype(FP8), quantized, name)
            self.assertEqual(
                jnp.dtype(form.wire_dtype) == jnp.dtype(FP8), quantized, name)
            self.assertEqual(form.has_scales, form.scale_layout != "none",
                             name)

    @parameterized.parameters(
        (WeightFormat.INT8, BF16),
        (WeightFormat.BF16, INT8),
        (WeightFormat.FP8, BF16),
        (WeightFormat.BF16, FP8),
    )
    def test_the_layer_checks_the_weights_against_the_format_it_was_told(
            self, weight_format, wrong_dtype):
        """The layer validated that the SCALES matched the format and never
        looked at the weights. That was largely self-limiting while fp8 and
        fp4 were the whole table -- four-bit weights stream as packed u32
        words, so a wrong format name died on the ref-level bitcast or on a
        shape mismatch soon after. int8 and bf16 declare IDENTICAL slab
        shapes and differ only in element size, one byte against two, so
        naming one and passing the other is a half-slab read with
        correct-looking shapes all the way down."""
        struct = jax.ShapeDtypeStruct
        form = WEIGHT_FORMS[weight_format]
        scales = (struct(
            (4, 1, 1, 512), jnp.float32) if form.has_scales else None)
        with self.assertRaisesRegex(ValueError, "expert weights"):
            fused_ep_moe_v2(struct((64, 256), jnp.bfloat16),
                            struct((4, 256, 512), wrong_dtype),
                            struct((4, 256, 512), wrong_dtype),
                            scales,
                            scales,
                            struct((64, 4), jnp.float32),
                            topk=2,
                            renormalize=True,
                            mesh=None,
                            capacity=128,
                            weight_format=weight_format)


class RoutingBlobTest(jtu.JaxTestCase):
    """The blob the routing indices and the row scales ride the all-gather
    in. The scale is carried as raw bits beside integer indices, so the
    round trip has to be exact rather than close: a lossy one moves every
    destination shard's combine by the error in its scale."""

    def test_the_blob_round_trips_bit_exactly(self):
        tokens, topk = 16, 4
        key = jax.random.key(5)
        idx = jax.random.randint(key, (tokens, topk), 0, 64).astype(jnp.int32)
        scale = jax.random.normal(key, (tokens, 1), jnp.float32) / 8
        blob = v2_layer._pack_routing_blob(idx, scale)
        self.assertEqual(blob.shape, (tokens, topk + 1))
        back_idx, back_scale = v2_layer._unpack_routing_blob(blob, topk)
        self.assertArraysEqual(back_idx, idx)
        self.assertArraysEqual(back_scale.reshape(scale.shape), scale)

    def test_a_format_that_carries_no_row_scale_packs_the_indices_alone(self):
        """The unquantized formats have no scale to send, and the unpack has
        to answer None rather than reading a column that is not there."""
        idx = jnp.arange(32, dtype=jnp.int32).reshape(8, 4)
        blob = v2_layer._pack_routing_blob(idx, None)
        self.assertEqual(blob.shape, (8, 4))
        back_idx, back_scale = v2_layer._unpack_routing_blob(blob, 4)
        self.assertArraysEqual(back_idx, idx)
        self.assertIsNone(back_scale)


class PrivateJaxSurfacesTest(jtu.JaxTestCase):
    """The four-bit weight stream binds to one private jax method, and it is
    what makes that path move half the bytes an eight-bit slab would. It has
    no deprecation entry, so a bump that removes it would otherwise fail as a
    bare AttributeError from inside a kernel body."""

    def test_the_ref_level_bitcast_is_still_there(self):
        from jax._src.state import types as state_types
        self.assertTrue(
            hasattr(state_types.AbstractRef, "bitcast"),
            "the private ref-level bitcast the fp4 weight stream views its "
            "packed words through has moved; there is no public Pallas "
            "equivalent, so this needs a port rather than a rename")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
