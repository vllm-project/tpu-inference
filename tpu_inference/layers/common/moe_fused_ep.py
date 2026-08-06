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
"""Serving adapter for the fused expert-parallel MoE kernel.

moe_fused_ep_route is the whole entry point: with USE_MOE_FUSED_EP_KERNEL
on it asks unsupported_reason whether the kernel can take this MoE layer at
all -- a refusal there stops the build rather than running elsewhere -- and
then asks unsupported_batch_reason whether it can take this batch shape,
which returns None and routes to the general MoE path rather than raising.
Every setting the decision reads is read here, so the serving path that
calls it carries one guarded call and nothing else. With the switch off this
module is never imported.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference import envs
from tpu_inference.layers.common.sharding import (MESH_AXIS_NAMES,
                                                  ShardingAxisName)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_mesh_shape_product

logger = init_logger(__name__)

# The tile height the adapter asks the kernel for. It is the capacity
# argument and the unit the ragged stride rounds to.
_TILE_M = 128

_kernel = None
# Resolved from the kernel's own constants at import, so no literal here
# can go stale.
_row_block = None
_slot_field = None
_packed4_row_tile = None
_vmem_estimate = None
_vmem_budget = None
_weight_buffers = None
_hidden_lane_block = None
_hidden_max_blocks = None
_act_fns = None
_routing_block = None
_WeightFormat = None
_weight_forms = None
_weight_format_names = None
_weight_format_of_dtype = None
_widen_kchunk = None
_chip_generation = None
_min_generation = None


def _import_kernel():
    """Resolve the in-tree kernel entry point; raises ImportError."""
    global _kernel, _row_block, _slot_field, _packed4_row_tile
    global _vmem_estimate
    global _vmem_budget, _weight_buffers, _hidden_lane_block
    global _hidden_max_blocks, _act_fns, _routing_block
    global _WeightFormat, _weight_forms, _weight_format_names
    global _weight_format_of_dtype
    global _widen_kchunk, _chip_generation, _min_generation
    if _kernel is not None:
        return _kernel
    try:
        from tpu_inference.kernels.fused_moe import v2 as kernel_package
    except ImportError as e:
        raise ImportError(
            f"fused EP MoE: importing the fused_moe.v2 kernel package "
            f"failed ({e}). There is no fallback; either fix the tree or "
            "unset USE_MOE_FUSED_EP_KERNEL to run this model on the general "
            "MoE path.") from e
    # The single-axis mesh this file builds names its axis 'd', and the
    # kernel layer's shard_map is written against that name. A mismatch is a
    # layout contract broken between two files, so it is refused by name
    # rather than asserted: an assert is the one check `python -O` removes,
    # and removing this one leaves the mismatch to surface as a shard_map
    # error naming an axis neither file mentions.
    if kernel_package.AXIS != "d":
        raise ValueError(
            f"fused EP MoE: the kernel package names its device axis "
            f"{kernel_package.AXIS!r} and this adapter hands the kernel "
            "layer a mesh whose one axis is named 'd'; the two no longer "
            "describe one mesh")
    _row_block = kernel_package.ROWBLK
    _slot_field = kernel_package.ALIGNMENT_SLOT_FIELD
    # Eight four-bit values pack into a 32-bit word and those words tile to
    # U32_SUBLANE_TILE sublanes, so an addressable block of packed weights
    # is a whole number of this many ROWS. ROWBLK is the routing tables'
    # row block and has no part in the packed-weight layout; it is 8 today,
    # which is the same number the sublane tiling happens to be, so the two
    # were interchangeable by coincidence rather than by construction. The
    # kernel checks the same product against the same constant.
    _packed4_row_tile = (kernel_package.U32_SUBLANE_TILE *
                         kernel_package.PACK4)
    _vmem_estimate = kernel_package.vmem_estimate_bytes
    _vmem_budget = kernel_package.vmem_limit
    _weight_buffers = kernel_package.NBUF
    _hidden_lane_block = kernel_package.HIDDEN_LANE_BLOCK
    _hidden_max_blocks = kernel_package.HIDDEN_MAX_BLOCKS
    _act_fns = kernel_package.ACT_FNS
    # The accepted weight formats and what each one implies. The table is
    # the kernel's, so this file cannot accept a pair the kernel does not
    # build or refuse one it does.
    _WeightFormat = kernel_package.WeightFormat
    _weight_forms = kernel_package.WEIGHT_FORMS
    _weight_format_names = kernel_package.WEIGHT_FORMAT_NAMES
    _weight_format_of_dtype = kernel_package.weight_format_of_dtype
    _widen_kchunk = kernel_package.WIDEN_KCHUNK
    # The routing block has one spelling, in the kernel package; the gate
    # predicts it with the same function the kernel layer computes it with.
    _routing_block = kernel_package.routing_block
    # The chip generation floor and the reader for it, both the kernel's.
    _chip_generation = kernel_package.chip_generation
    _min_generation = kernel_package.MIN_GENERATION
    _kernel = kernel_package.fused_ep_moe_v2
    return _kernel


# The serving axis set minus 'pcp', which the re-wrap has not been shown to
# preserve. Derived rather than hand-copied, so a reorder upstream cannot
# silently change what this accepts.
_PROVEN_MESH_AXES = tuple(n for n in MESH_AXIS_NAMES if n != 'pcp')

# The element types a router's logits may arrive in. The kernel layer casts
# them to float32 and selects from that, so the question this answers is
# whether the value that arrived still carries the ordering the router
# computed -- which every type here does and float8 does not.
# Any ordinary float is routable; the refusal exists for the narrow types
# whose range makes softmax garbage (an fp8 gating array quantizes real
# logits into ties and infinities). This gate raises when the switch is on,
# so it lists what is refused rather than what is allowed.
_GATING_DTYPES = (jnp.dtype(jnp.float64), jnp.dtype(jnp.float32),
                  jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16))
_GATING_DTYPE_NAMES = tuple(d.name for d in _GATING_DTYPES)


def _accepted_weight_pairs() -> str:
    """The accepted (weight dtype, scale layout) pairs, for a refusal.

    Read off the kernel's own table, so a refusal cannot name a set the
    kernel has outgrown. Two four-bit forms are deliberately outside it and
    stay refused: integer-four, which has no scale layout here, and a
    four-bit block of 32, which the packed-weight row tile refuses further
    down -- eight values pack into a 32-bit word and those words tile to
    eight sublanes, so a block is a whole number of 64 rows or it is not
    addressable.
    """
    return ", ".join(f"({jnp.dtype(f.weight_dtype).name}, {f.scale_layout})"
                     for f in _weight_forms.values())


# The clause that marks a refusal whose condition is not this kernel's.
# Every expert-parallel MoE program shards the experts, so a width that
# does not divide the expert count is refused on the general path too --
# fused_moe_func raises on exactly this condition. A caller offering "unset
# the switch and run on the general path" as the remedy has to know which
# reasons it may offer it for, and this is how it knows.
_GENERAL_PATH_ALSO_REFUSES = ("which is a property of the expert sharding "
                              "rather than of this kernel")


def reason_survives_the_general_path(reason: str) -> bool:
    """Whether this reason would refuse on the general MoE path too.

    For nearly every refusal here -- a weight element type the kernel does
    not take, a mesh it cannot re-wrap, a scale layout, a routing modifier
    it has no operand for -- unsetting the switch really does run the model,
    so the caller's error says so. For the expert-count divisibility it does
    not: the general path shards the experts the same way and refuses the
    same condition, so that remedy sends an operator round a loop and back
    to the same message under a different program's name.
    """
    return _GENERAL_PATH_ALSO_REFUSES in reason


def _validated_top_k(layer) -> int | str:
    """The layer's selection width, or a reason string if it is not one.

    Read, never coerced, and read the same way everywhere. A router config
    that carried top_k as a float -- the shape of a value that arrived
    through JSON, or through a derived config -- would otherwise serve at
    the truncated width while the layer's own attribute still read the
    original everywhere else in the stack, and the general path this switch
    replaces does no such coercion, so the two paths would route the same
    layer differently.
    """
    topk = layer.top_k
    if isinstance(topk, bool) or not isinstance(topk, (int, np.integer)):
        return (f"top_k is {topk!r} ({type(topk).__name__}); the kernel's "
                "selection width is an integer and is read rather than "
                "converted")
    return int(topk)


def _check_failed(what: str, e: Exception) -> str:
    """A reason string for an acceptance check that could not answer.

    Distinguished in words from "this layer is outside the envelope",
    because the caller turns a reason into "this layer cannot: ..." and an
    operator reading that goes looking at their checkpoint. An escape from
    the body of a check is a defect in this adapter, not in the model.
    """
    return (f"the fused EP MoE {what} check could not be evaluated, which is "
            f"a defect in this kernel adapter rather than a property of the "
            f"model: {e!r}. Unsetting USE_MOE_FUSED_EP_KERNEL runs the model "
            "on the general MoE path while it is investigated")


def _attn_data_axes() -> tuple[str, ...]:
    """The axis names this runtime calls attention-data, always a tuple."""
    axes = ShardingAxisName.ATTN_DATA
    return (axes, ) if isinstance(axes, str) else tuple(axes)


def _mesh_reason(mesh: Mesh) -> str | None:
    """Why this serving mesh cannot be re-wrapped as one axis, or None."""
    ndev = mesh.devices.size
    names = tuple(mesh.axis_names)
    # Any axis outside this list is accepted only when degenerate: a
    # size-1 axis cannot permute the flattened device order.
    if tuple(n for n in names if n in _PROVEN_MESH_AXES) != _PROVEN_MESH_AXES:
        return (f"mesh axes {names} do not contain {_PROVEN_MESH_AXES} in "
                "that order")
    for name in names:
        if name not in _PROVEN_MESH_AXES and mesh.shape[name] != 1:
            return (f"mesh axis {name!r} has size {mesh.shape[name]} != 1; "
                    f"every axis outside {_PROVEN_MESH_AXES} must be "
                    "degenerate for the re-wrap to preserve device order")
    # The attention-data axis set is a property of the runtime sharding
    # selection rather than of the mesh, so it can name an axis this mesh
    # does not carry -- and the apply function's closing sharding constraint
    # names the same set. Answer that here rather than raising there.
    attn_axes = _attn_data_axes()
    absent = tuple(n for n in attn_axes if n not in names)
    if absent:
        return (f"the attention-data axis set {attn_axes} names {absent}, "
                f"which this mesh does not carry (its axes are {names}); the "
                "kernel's output is re-tagged onto that axis set")
    # The two-dimensional axis set calls the replica axis the whole of
    # attention-data, which asks for data == 1 (below) and data == device
    # count at once. Name the axis-set selection rather than leaving the
    # reader to read a contradiction as an attention misconfiguration.
    if attn_axes == ('data', ):
        return ("the attention-data axis set is ('data',), the replica axis, "
                "which the single-axis re-wrap needs at size 1: no mesh wider "
                "than one device satisfies both. Set NEW_MODEL_DESIGN=1 to "
                "select the axis set the kernel reconciles")
    if mesh.shape['data'] != 1:
        return (f"mesh data axis size {mesh.shape['data']} != 1; with "
                "data > 1 the expert-shard order diverges from the flat "
                "device order and the re-wrap would permute shards")
    attn_dp = get_mesh_shape_product(mesh, attn_axes)
    if attn_dp != ndev:
        return (f"attention is not pure DP over all devices "
                f"(enable_dp_attention): the attention-data axes "
                f"{attn_axes} multiply to {attn_dp}, not the device count "
                f"{ndev}")
    return None


def _single_axis_mesh(mesh: Mesh) -> Mesh:
    """The kernel-layer mesh: same devices, one axis named 'd'."""
    reason = _mesh_reason(mesh)
    if reason is not None:
        raise ValueError(f"fused EP MoE: {reason}")
    return Mesh(mesh.devices.reshape(-1), ("d", ))


def unsupported_reason(layer,
                       x: jax.Array,
                       gating_output,
                       weights,
                       mesh: Mesh,
                       activation: str,
                       scatter_results: bool,
                       extra_backend_kwargs: dict | None = None,
                       defer_all_reduce: bool = False) -> str | None:
    """Why the fused EP kernel cannot take this MoE LAYER, or None.

    Every condition here is a property of the model, the mesh and the
    caller's request rather than of the batch shape, so the answer is the
    same for every batch a deployment serves and the caller can raise on
    it. The batch-shape conditions live in unsupported_batch_reason. Every
    condition is a trace-time constant and the answer is never an
    exception, so the caller can pick the path at compile time.

    The caller has no try around this, so anything that escapes is an
    unhandled model-build failure with a traceback naming arithmetic rather
    than the operand that was wrong. Every operand below arrives from a
    checkpoint load or a serving config, and a half-finished load is exactly
    the case this function exists to catch, so the never-raises half of the
    contract is enforced here rather than left to inspection of the body.
    """
    try:
        return _unsupported_reason(layer, x, gating_output, weights, mesh,
                                   activation, scatter_results,
                                   extra_backend_kwargs, defer_all_reduce)
    except Exception as e:
        return _check_failed("layer acceptance", e)


def _unsupported_reason(layer, x, gating_output, weights, mesh, activation,
                        scatter_results, extra_backend_kwargs,
                        defer_all_reduce) -> str | None:
    """The conditions themselves; unsupported_reason keeps the contract."""
    try:
        _import_kernel()
    except ImportError as e:
        # Also a defect in the tree rather than in the model, and framed as
        # one: the caller turns this into "this layer cannot: ...".
        return (f"the fused EP MoE kernel package is not importable in this "
                f"tree, which is a defect in the install rather than a "
                f"property of the model: {e}")

    if getattr(x, "ndim", None) != 2:
        return (f"activations are {getattr(x, 'shape', None)}; the kernel "
                "takes a two-dimensional (tokens, hidden) batch")
    hidden = x.shape[-1]
    # Read before anything dereferences a shape: a bundle whose load did not
    # finish carries None here, and the failure should name the weight rather
    # than come back as an attribute error on a NoneType.
    for name, w in (("w13_weight", weights.w13_weight), ("w2_weight",
                                                         weights.w2_weight)):
        if w is None:
            return f"the weight bundle carries no {name}"
        if getattr(w, "ndim", None) != 3:
            return (f"{name} shape {getattr(w, 'shape', None)} is not the "
                    "three-dimensional (experts, contraction, output) form "
                    "the kernel reads")
    num_experts, w_hidden, two_inter = weights.w13_weight.shape
    # An expert count of zero was refused only by accident, because the top_k
    # check further down fires first and says "top_k is 10, over the 0
    # experts to choose from" -- which sends a loader debugging an empty
    # bundle to look at their router config.
    if num_experts < 1:
        return ("the weight bundle carries no experts; its first axis is the "
                "expert axis and it has length zero")
    # The first weight's output axis is the gate and up halves side by side,
    # so an odd width is not a pair: the integer division below would discard
    # the odd column silently and the failure would surface much later as a
    # reshape.
    if two_inter % 2:
        return (f"the first expert weight's output width {two_inter} is odd; "
                "it is the gate and up halves side by side")
    inter = two_inter // 2
    # The hidden axis has a lane-block floor further down and the
    # intermediate axis had none, so an expert whose FFN has no intermediate
    # channels was accepted here and built and cached a program: zero is a
    # whole number of every block size the checks below ask about, and the
    # VMEM estimate for such a build clears the budget comfortably. Same
    # floor and same reason as the hidden axis: an intermediate narrower
    # than the vector unit's lane count is not an FFN, and the diagnostic an
    # operator would eventually receive names a reduction or a reshape
    # rather than the weight that was empty.
    if inter < _hidden_lane_block:
        return (f"the first expert weight's output width {two_inter} leaves "
                f"an intermediate of {inter}, narrower than one "
                f"{_hidden_lane_block}-lane block; an expert whose FFN has "
                "no intermediate channels to speak of is not a layer this "
                "kernel can serve")

    if not isinstance(gating_output, jax.Array):
        return (f"gating output is {type(gating_output).__name__}; the "
                "kernel routes from one logits array")
    # The router's index range IS the width of this array, and the routing
    # tables bin expert ids against the expert count: an id at or above that
    # count matches no bin and lands on the row expert zero owns. The plan
    # has no sink for one, so the width is tied to the expert count here.
    # The height is tied to the token count for the same kind of reason: the
    # routing tables are built per routed pair, so a mismatch is a table of
    # the wrong length and the failure lands inside the table builder.
    if gating_output.shape != (x.shape[0], num_experts):
        return (
            f"gating output shape {gating_output.shape} is not "
            f"{(x.shape[0], num_experts)}, one row per token and one column "
            f"per expert; the routing tables have no sink for "
            "an expert id outside the expert range, and the router's "
            "index range is this array's width")
    # "Floating point" is not a narrow enough answer: float8_e4m3fn is a
    # floating point type and satisfies it, and an eight-bit logit array is
    # a routing decision made by the quantizer rather than by the router.
    # At ordinary logit magnitudes neighbouring experts tie exactly and the
    # selection falls to the lowest index by the tie rule; past about 448
    # the cast produces NaN and the whole row is masked out, deleting the
    # token. The accepted set is named rather than derived, so a type
    # narrower than the router's arithmetic cannot slip in as "floating".
    if gating_output.dtype not in _GATING_DTYPES:
        return (f"gating output is {gating_output.dtype}; the router "
                f"softmaxes these scores and selects from them, so the "
                f"logits have to arrive in one of {_GATING_DTYPE_NAMES} "
                "rather than in a type whose rounding decides which expert "
                "a token goes to")
    if not layer.use_ep:
        return "the kernel is expert-parallel and layer.use_ep is False"
    # Answered before the scatter_results question below, because on one
    # device that question has no content: with a single rank the per-rank
    # form and the all-reduced form are the same array, so a single-device
    # caller passing scatter_results=False was told the kernel "ends with
    # each rank holding its own token rows" and sent looking at a
    # distinction that does not exist on their mesh. The real answer is
    # that there is nothing here for an expert-parallel program to do.
    if mesh.devices.size < 2:
        return ("the mesh carries one device; the kernel shards the experts "
                "across ranks and combines each token's rows back to their "
                "owner inside itself, so it needs more than one device to "
                "have anything to serve")
    if not scatter_results:
        return ("the kernel layer ends with each rank holding its own token "
                "rows, which is scatter_results semantics; the caller asked "
                "for the all-reduced form")
    if layer.scoring_func != "softmax":
        return (f"kernel routing is softmax + top_k; got scoring_func="
                f"{layer.scoring_func!r}")
    if activation not in _act_fns:
        return (f"the kernel fuses one of {_act_fns} as its FFN activation; "
                f"got activation={activation!r}")
    if defer_all_reduce:
        return ("the caller asked for defer_all_reduce, which returns "
                "per-shard partial sums; the kernel combines rows to their "
                "token owners inside itself and has no partial-sum output")

    # Routing modifiers fused_moe_func honours and this kernel does not:
    # each one changes which experts a token goes to.
    kw = extra_backend_kwargs or {}
    for name, value in (("hash_based_topk_indices",
                         kw.get("hash_based_topk_indices")),
                        ("e_score_correction_bias",
                         kw.get("e_score_correction_bias")),
                        ("num_valid_tokens", kw.get("num_valid_tokens"))):
        if value is not None:
            return (f"the caller passed {name}, which selects experts (or "
                    "gates rows) differently from the kernel's own softmax "
                    "top-k; the kernel has no operand for it")
    if envs.MOE_APPROX_TOPK:
        return ("MOE_APPROX_TOPK asks for approximate top-k selection; the "
                "kernel selects exactly")
    # Same class: a setting the general path honours that changes WHICH
    # experts a token goes to. Ignoring it would leave a benchmark measuring
    # something other than what it asked for, and nothing would say so.
    if envs.FORCE_MOE_RANDOM_ROUTING:
        return ("FORCE_MOE_RANDOM_ROUTING asks for random expert assignment "
                "for benchmarking; the kernel routes from the router's own "
                "scores and has no operand for it, so a run with it set "
                "would measure real routing and say it measured random")

    # Which weight form this is, decided by the expert weight dtype against
    # the kernel's table of accepted (dtype, scale layout) pairs. Every
    # other dtype is refused by name, never reinterpreted.
    weight_format = _weight_format_of_dtype(weights.w13_weight.dtype)
    if weight_format is None:
        return (f"kernel weights are {weights.w13_weight.dtype}, which is "
                f"not one of the accepted weight element types "
                f"{_accepted_weight_pairs()}")
    rhs_packed4 = weight_format == _WeightFormat.FP4
    if weights.w2_weight.dtype != weights.w13_weight.dtype:
        return (f"both expert weights must carry one dtype; w13 is "
                f"{weights.w13_weight.dtype} and w2 is "
                f"{weights.w2_weight.dtype}")
    # Both weight forms: the intermediate width the second matmul contracts
    # has to be the one the first matmul produces. The kernel sizes the
    # second weight's resident buffer from the FIRST weight alone, and a
    # loader that pads one contraction and not the other produces exactly
    # this mismatch, so it is asked of both forms rather than only of fp4.
    if weights.w2_weight.shape != (num_experts, inter, w_hidden):
        return (f"w2 shape {weights.w2_weight.shape} != "
                f"{(num_experts, inter, w_hidden)}; the two expert weights "
                "disagree on the intermediate or hidden width")
    # The weights may be wider than the activations: a requantizing loader
    # pads each matmul's contraction up to a whole number of its blocks and
    # zero-fills, so the apply function pads the activations in and trims
    # the outputs, and everything below runs at the WEIGHT width.
    if w_hidden < hidden:
        return (f"weight hidden ({w_hidden}) is narrower than activation "
                f"hidden ({hidden}); the kernel contracts the whole "
                "activation row")
    # The transport stages a token row as a whole number of 128-lane
    # blocks, and its row staging holds a bounded number of them. Zero is a
    # whole number of them and is not a width: it builds a kernel whose row
    # quantizer reduces over an empty axis.
    if w_hidden < _hidden_lane_block:
        return (f"weight hidden {w_hidden} is narrower than one "
                f"{_hidden_lane_block}-lane block, which is the smallest row "
                "the kernel's transport stages")
    if w_hidden % _hidden_lane_block != 0:
        return (f"weight hidden {w_hidden} is not a whole number of "
                f"{_hidden_lane_block}-lane blocks, which the kernel's "
                "per-row transport geometry requires")
    if w_hidden > _hidden_lane_block * _hidden_max_blocks:
        return (f"weight hidden {w_hidden} is wider than the "
                f"{_hidden_lane_block * _hidden_max_blocks} the kernel's "
                "row staging holds")

    # Expert biases, one row per expert on each matmul's OUTPUT channels.
    # The two are optional and independent of each other. The width is taken
    # from the WEIGHT shapes, so a loader that padded a contraction and left
    # the bias beside it at the unpadded width is refused here rather than
    # silently mixing the two: the down bias enters the row before the result
    # is quantized for the wire, and a wrong-width one would corrupt the
    # quantization of every real column of that row.
    for name, bias, size_n in (("w13_bias", weights.w13_bias, two_inter),
                               ("w2_bias", weights.w2_bias, w_hidden)):
        if bias is None:
            continue
        if getattr(bias, "shape", None) != (num_experts, 1, size_n):
            return (f"{name} layout {getattr(bias, 'shape', None)} is not the "
                    f"per-expert form {(num_experts, 1, size_n)} the kernel "
                    "consumes")
        # The bias is added to a row before that row is quantized for the
        # wire, so it has to carry the magnitudes the weights were trained
        # with; an integer table is reinterpreted rather than converted.
        if not jnp.issubdtype(bias.dtype, jnp.floating):
            return (f"{name} is {bias.dtype}; the kernel adds it to a row "
                    "before quantizing that row, so it must be a floating "
                    "point table")

    w13_scale, w2_scale = (weights.w13_weight_scale, weights.w2_weight_scale)
    form = _weight_forms[weight_format]
    if not form.has_scales:
        # An unquantized weight has nothing to descale by. A bundle that
        # carries scales beside one came from a quantizing loader, and
        # ignoring them would run the model at the wrong magnitudes.
        if w13_scale is not None or w2_scale is not None:
            return (f"{weight_format} weights are unquantized and carry no "
                    f"scales, and this bundle supplies them; the accepted "
                    f"pairs are {_accepted_weight_pairs()}")
        rhs_qb = None
    elif w13_scale is None or w2_scale is None:
        return (f"{weight_format} weights carry {form.scale_layout} scales "
                f"and this bundle supplies none; the accepted pairs are "
                f"{_accepted_weight_pairs()}")
    elif any(s.dtype != jnp.float32 for s in (w13_scale, w2_scale)):
        # The kernel casts these to f32 on the way in, so a narrower table
        # is rounded to its own precision and an integer one is
        # reinterpreted -- neither is a descale factor any more.
        return (f"the weight scale tables are {w13_scale.dtype} and "
                f"{w2_scale.dtype}; the kernel descales in float32 and a "
                "narrower table has already lost the bits it would descale "
                "with")
    elif rhs_packed4:
        # Derive the block size from the w13 scale shape, then require w2
        # to carry the same one -- shape-verified, never assumed.
        if not (w13_scale.ndim == 4 and w13_scale.shape[0] == num_experts and
                w13_scale.shape[2] == 1 and w13_scale.shape[3] == two_inter):
            return (f"fp4 w13 scale layout {w13_scale.shape} is not the block "
                    f"form (E, hidden//qb, 1, 2*inter) = ({num_experts}, "
                    f"blocks, 1, {two_inter})")
        if w13_scale.shape[1] < 1:
            return ("the fp4 w13 scale table carries no contraction blocks, "
                    "so there is no block size to derive from it")
        # The format table presents four (weight dtype, scale layout) pairs
        # and every refusal reports them as if they were exclusive. They are
        # not: an (E, 1, 1, N) per-channel table satisfies the block-form
        # check above too, because that check leaves the block axis free. So
        # whenever hidden == inter the block count of one divides through
        # and an fp4 bundle carrying fp8's scale layout was ACCEPTED, with
        # the whole contraction read as one block; where the widths differ
        # it was refused for divisibility, which tells a loader debugging a
        # scale layout about arithmetic. Decide the layout by its shape
        # before deriving anything from it.
        if w13_scale.shape[1] < 2:
            return (f"the fp4 w13 scale table carries one contraction block, "
                    f"which is the per-channel layout rather than the "
                    f"{form.scale_layout} layout fp4 weights take; the "
                    f"accepted pairs are {_accepted_weight_pairs()}")
        if w_hidden % w13_scale.shape[1] != 0:
            return (f"fp4 hidden {w_hidden} is not divisible by the w13 "
                    f"scale block count {w13_scale.shape[1]}")
        rhs_qb = w_hidden // w13_scale.shape[1]
        if inter % rhs_qb != 0:
            return (f"fp4 intermediate size {inter} is not divisible by the "
                    f"derived block size {rhs_qb}; the kernel blocks BOTH "
                    "matmuls at one block size")
        if w2_scale.shape != (num_experts, inter // rhs_qb, 1, w_hidden):
            return (f"fp4 w2 scale layout {w2_scale.shape} != "
                    f"{(num_experts, inter // rhs_qb, 1, w_hidden)} -- both "
                    f"matmuls must carry the same block size {rhs_qb}, and "
                    "a mismatch is refused, never resampled")
        if rhs_qb % _packed4_row_tile != 0:
            return (f"four-bit block size {rhs_qb} is not a whole number of "
                    f"the kernel's packed-weight row tile "
                    f"({_packed4_row_tile} rows)")
    else:
        rhs_qb = None
        if w13_scale.shape != (num_experts, 1, 1, two_inter):
            return (
                f"w13 scale layout {w13_scale.shape} is not the per-channel "
                f"form {(num_experts, 1, 1, two_inter)} the kernel "
                "consumes")
        if w2_scale.shape != (num_experts, 1, 1, w_hidden):
            return (f"w2 scale layout {w2_scale.shape} != "
                    f"{(num_experts, 1, 1, w_hidden)}")
    # Integer weights reach the matrix unit through a widening, one fixed
    # contraction chunk at a time, and that chunk has to tile both matmuls'
    # contractions exactly.
    if weight_format == _WeightFormat.INT8:
        for name, size in (("hidden", w_hidden), ("intermediate", inter)):
            if size % _widen_kchunk != 0:
                return (f"int8 weights widen one {_widen_kchunk}-row "
                        f"contraction chunk at a time and {name} {size} is "
                        "not a whole number of them")

    # More selections than experts is a real break: the selector runs out of
    # columns and re-selects expert zero carrying the sentinel weight on a
    # row whose scores are finite, so the mask does not catch it and a
    # perfectly good token is zeroed.
    #
    # The two gates below are not that. They are envelope statements, and
    # the reason they used to give is gone: they said the kernel's NaN-score
    # guard zeroed a non-routing row by summing at least two sentinel
    # weights to -inf and dividing by that through the renormalization, so
    # both halves were required. That mechanism was replaced by an explicit
    # mask -- the kernel layer now takes jnp.any(jnp.isfinite(scores)) and
    # zeroes against it unconditionally -- precisely because the old
    # behaviour was an arithmetic accident of the accumulation dtype. The
    # mask holds at any selection width and with renormalization off, so
    # neither configuration is unsafe for the reason that was written down.
    # They stay refused because neither has ever been run: the selector's
    # tie and sentinel behaviour at one selection slot, and the combine
    # against un-renormalized weights, have no test anywhere in the tree.
    # Lifting either is a coverage question, not a redesign.
    #
    # Read rather than coerced; _validated_top_k says why.
    topk = _validated_top_k(layer)
    if isinstance(topk, str):
        return topk
    if topk < 2:
        return (f"top_k is {topk}; the kernel's router has only been run at "
                "two selections or more, and its behaviour at one slot -- "
                "where every tie and every sentinel weight is the row's "
                "whole answer -- has no test")
    if topk > num_experts:
        return (f"top_k is {topk}, over the {num_experts} experts to choose "
                "from; the kernel's selector re-selects expert zero at the "
                "sentinel weight once the columns run out, which zeroes the "
                "token the same way it zeroes a NaN row")
    if not bool(layer.renormalize):
        return ("the layer does not renormalize the selected weights; the "
                "kernel's combine has only been run against renormalized "
                "ones, and weights that do not sum to one have no test")

    reason = _mesh_reason(mesh)
    if reason is not None:
        return reason
    ep = mesh.devices.size
    if num_experts % ep != 0:
        return (f"expert count {num_experts} is not divisible by the "
                f"expert-parallel width {ep}; each shard owns "
                f"num_experts // ep experts and every expert has to land on "
                f"exactly one shard, {_GENERAL_PATH_ALSO_REFUSES}, so serve "
                f"at an expert-parallel width that divides the expert count")
    # The routing tables pack a per-shard position and an alignment slot
    # into one word. The slot runs up to (ROWBLK - 1) * (ep - 1), so past
    # ep = 10 it overflows its field and the two corrupt each other silently.
    if (_row_block - 1) * (ep - 1) >= _slot_field:
        return (f"expert-parallel width {ep} needs up to "
                f"{(_row_block - 1) * (ep - 1)} alignment slots and the "
                f"routing tables pack them into a {_slot_field}-wide field; "
                "the kernel holds widths up to "
                f"{1 + (_slot_field - 1) // (_row_block - 1)}")
    # VMEM fit. BOTH sides of this comparison are read off the device record
    # -- the budget is the chip's capacity, the accounting needs that chip's
    # lane count and sublane tiling to size the padding -- so both reads sit
    # under one guard. jax carries layout rules for the chip generations it
    # knows and raises for the rest, and the generation after the served one
    # is already an entry in its enum, so the first fleet to get newer chips
    # must be answered with a refusal naming the chip rather than handed a
    # traceback out of the middle of the model build.
    #
    # The buffered weight slabs dominate the estimate and do not shrink with
    # the token count, so the answer is a property of the model, not the
    # batch. A bias the model carries is a resident table the built kernel
    # holds, so the presence of each one is passed in rather than left out:
    # the figure this compares against the budget must be the figure for the
    # kernel this call would actually build.
    #
    # The chip's GENERATION is read under the same guard and answered first,
    # because a chip older than the one this kernel was written for fails
    # neither of the two questions below: its layout rules are all known to
    # jax, and its VMEM is large enough for a small enough model, so the
    # gate accepted it and the deployment ran a program that has never been
    # correctness-checked or timed on that generation.
    try:
        generation = _chip_generation()
        if generation < _min_generation:
            return (f"the chip this call is being built for is TPU "
                    f"generation {generation}; the fused expert-parallel MoE "
                    f"kernel is validated on TPU v{_min_generation} and has "
                    f"never been run on an earlier generation, so a build "
                    f"here would be the first. Unsetting "
                    f"USE_MOE_FUSED_EP_KERNEL runs the model on the general "
                    f"MoE path, which serves this chip")
        budget = _vmem_budget()
        est = _vmem_estimate(num_experts // ep,
                             _TILE_M,
                             w_hidden,
                             inter,
                             nbuf=_weight_buffers,
                             weight_format=weight_format,
                             rhs_qb=rhs_qb or w_hidden,
                             has_w1_bias=weights.w13_bias is not None,
                             has_w2_bias=weights.w2_bias is not None)
    except Exception as e:  # no device record, or a chip jax has no rules for
        return (f"the kernel's VMEM budget cannot be read for the chip this "
                f"call is being built for ({e!r}); the budget is that chip's "
                "capacity and the accounting needs its lane count and "
                "sublane tiling, so a host that can name no chip -- or a "
                "chip generation this kernel has not been built for -- "
                "cannot run the kernel")
    if est > budget:
        return (f"the kernel's buffers need {est / 2**20:.1f}MiB of VMEM "
                f"for {num_experts // ep} local experts of "
                f"{w_hidden}x{inter}, over the {budget / 2**20:.1f}MiB "
                "budget")
    return None


def unsupported_batch_reason(layer,
                             x: jax.Array,
                             mesh: Mesh,
                             num_experts: int | None = None) -> str | None:
    """Why the fused EP kernel cannot take this BATCH SHAPE, or None.

    These describe the batch rather than the model, so a caller routes the
    shape to the general MoE path instead of raising: the same layer serves
    every other shape on the kernel. Only valid on a layer
    unsupported_reason has already accepted.

    num_experts is the layer's expert count, which the arrival-row bound
    below needs. It is optional so a caller that only has a shape can still
    ask the other two questions; the bound is skipped when it is absent.

    The caller has no try around this either, and the asymmetry was
    invisible from the call site, which treats the two questions the same
    way: anything escaping here surfaced as an unhandled build failure for a
    condition the contract promises is a route-away. Same wrapper as
    unsupported_reason, for the same reason.
    """
    try:
        return _unsupported_batch_reason(layer, x, mesh, num_experts)
    except Exception as e:
        return _check_failed("batch shape", e)


def _unsupported_batch_reason(layer, x, mesh, num_experts) -> str | None:
    """The conditions themselves; the wrapper above keeps the contract."""
    _import_kernel()
    num_tokens = x.shape[0]
    # Read rather than coerced, the same way the layer gate reads it. On the
    # serving path the layer gate has already validated it, but this
    # function is exported beside the apply entry point and a direct caller
    # got int("10") and int(2.9) silently -- which is the coercion the layer
    # gate exists to refuse, two functions apart.
    topk = _validated_top_k(layer)
    if isinstance(topk, str):
        return topk
    ep = mesh.devices.size
    # An empty batch passes every divisibility check below, and the routing
    # tables it builds have no in-bounds row for the gather to clamp to.
    if num_tokens < 1:
        return "the batch has no tokens"
    if num_tokens % ep != 0:
        return (f"token count {num_tokens} is not divisible by the "
                f"expert-parallel width {ep}")
    if _routing_block(num_tokens // ep, topk) < 8:
        return (f"the routing tables need a block of at least 8 dividing "
                f"{(num_tokens // ep) * topk} rows per shard; tokens="
                f"{num_tokens}, top_k={topk}, ep={ep} admits none")
    # The routing tables pack an arrival position and an alignment slot into
    # one 32-bit word. The mesh-width check in unsupported_reason bounds the
    # slot; this bounds the OTHER factor of the same word. Past it the two
    # fields corrupt each other with no error anywhere near them, and it is
    # a statement about the batch, so an oversized bucket routes to the
    # general path rather than serving a corrupt plan.
    if num_experts is not None:
        max_pos = num_tokens * topk + (_row_block - 1) * num_experts
        if max_pos * _slot_field >= 2**31:
            return (f"this batch would carry up to {max_pos} arrival rows, "
                    f"and the routing tables pack one alongside a "
                    f"{_slot_field}-wide alignment slot in a 32-bit word, "
                    f"which holds {2**31 // _slot_field}")
    return None


def moe_fused_ep_apply(
    layer,
    x: jax.Array,
    gating_output: jax.Array,
    weights,
    mesh: Mesh,
    activation: str,
    scatter_results: bool,
    extra_backend_kwargs: dict | None = None,
    defer_all_reduce: bool = False,
    *,
    acceptance: tuple[str | None, str | None] | None = None,
) -> jax.Array:
    """Run one MoE call through the fused expert-parallel kernel layer.

    Not bit-identical to the general MoE path on the quantized weight
    formats: there each expert's output rows cross the wire as fp8 e4m3 with
    one f32 scale per row, where fused_moe_func carries them in bf16. On the
    unquantized formats the wire is bf16 too and that difference is gone,
    at twice the wire bytes.

    extra_backend_kwargs and defer_all_reduce are the operands that decide
    which experts a token goes to and what the layer returns. They take the
    same defaults moe_apply would pass for a caller that has neither, and
    they reach the acceptance check below, so a direct caller gets the same
    answer moe_apply gets rather than a weaker one.

    acceptance is the (layer reason, batch reason) pair a caller that has
    already asked both questions passes back in, and is how moe_apply avoids
    paying for the whole acceptance check twice on every accepted call --
    two thirds of that check is the VMEM estimate, which depends only on the
    model. A caller that passes nothing asks both questions here, so a direct
    caller is guarded exactly as before; a caller that passes a pair is
    saying it holds those answers for these operands, and they are raised on
    the same way.
    """
    kernel = _import_kernel()
    if acceptance is None:
        acceptance = (unsupported_reason(layer, x, gating_output, weights,
                                         mesh, activation, scatter_results,
                                         extra_backend_kwargs,
                                         defer_all_reduce),
                      unsupported_batch_reason(
                          layer,
                          x,
                          mesh,
                          num_experts=weights.w13_weight.shape[0]))
    for reason in acceptance:
        if reason is not None:
            # Same rule the layer caller applies: the remedy is offered only
            # for reasons the general MoE path does not refuse too.
            remedy = ("" if reason_survives_the_general_path(reason) else
                      " Unset USE_MOE_FUSED_EP_KERNEL to run this model on "
                      "the general MoE path instead.")
            raise ValueError(f"fused EP MoE: {reason}.{remedy}")

    hidden = x.shape[-1]
    num_experts, w_hidden, two_inter = weights.w13_weight.shape
    # Read the same way the gate reads it. Both checks above have passed by
    # here, so this cannot be a reason string -- but reading it through the
    # one function is what stops a third spelling of "the selection width"
    # appearing later.
    topk = _validated_top_k(layer)
    if isinstance(topk, str):
        raise ValueError(f"fused EP MoE: {topk}")
    w13_scale, w2_scale = (weights.w13_weight_scale, weights.w2_weight_scale)
    weight_format = _weight_format_of_dtype(weights.w13_weight.dtype)
    rhs_packed4 = weight_format == _WeightFormat.FP4
    rhs_qb = w_hidden // w13_scale.shape[1] if rhs_packed4 else None
    # The bundle's scale tables carry a singleton axis the kernel has no
    # operand for, and the kernel's own layout is [E, N] per output channel
    # and [E, blocks, N] per contraction block. The scales are constants of
    # the weights, so they take that shape here, where the bundle is read,
    # and the kernel layer takes them as they are: the same reshape below it
    # is a layout change standing between the parameter and the kernel call
    # on every call.
    if w13_scale is not None:
        if rhs_packed4:
            w13_scale = w13_scale.reshape(num_experts, -1, two_inter)
            w2_scale = w2_scale.reshape(num_experts, -1, w_hidden)
        else:
            w13_scale = w13_scale.reshape(num_experts, two_inter)
            w2_scale = w2_scale.reshape(num_experts, w_hidden)

    w13_bias, w2_bias = weights.w13_bias, weights.w2_bias

    # A requantizing loader pads each matmul's contraction up to a whole
    # number of its blocks and zero-fills, so the weights can be wider than
    # the activations. Pad the activations in and trim the outputs, the way
    # the FUSED_MOE backend does around its own kernel. The padded weight
    # rows are zero, so the padded columns add nothing to either matmul.
    if w_hidden > hidden:
        x = jnp.pad(x, ((0, 0), (0, w_hidden - hidden)))
        if w2_bias is not None:
            # The down bias's padded columns must be zero. They are trimmed
            # off the result, but they reach the row BEFORE it is quantized
            # for the wire, so a nonzero one would move the row's scale and
            # change every real column of that row. Imposed here instead of
            # assumed, so the padded-hidden path cannot depend on a loader
            # convention it does not control.
            w2_bias = jnp.pad(w2_bias[..., :hidden],
                              ((0, 0), (0, 0), (0, w_hidden - hidden)))

    mesh1 = _single_axis_mesh(mesh)

    # The router runs on f32 logits, so softmax and top-k tie behavior
    # follow the logit dtype rather than the activation dtype.
    gating_f32 = gating_output.astype(jnp.float32)

    # jax refuses a shard_map whose mesh differs from the ambient context,
    # so the trace enters the single-axis mesh for the kernel call alone.
    format_kwargs = {"weight_format": weight_format}
    if rhs_packed4:
        format_kwargs["rhs_qb"] = rhs_qb
    # The kernel layer derives its own routing block and no-drop stride.
    with jax.sharding.use_abstract_mesh(mesh1.abstract_mesh):
        out = kernel(x,
                     weights.w13_weight,
                     weights.w2_weight,
                     w13_scale,
                     w2_scale,
                     gating_f32,
                     w13_bias,
                     w2_bias,
                     topk=topk,
                     renormalize=bool(layer.renormalize),
                     mesh=mesh1,
                     capacity=_TILE_M,
                     act_fn=activation,
                     **format_kwargs)
    # The trim happens before the re-tag, so the sharding constraint is
    # asserted on the shape the caller asked for.
    if w_hidden > hidden:
        out = out[:, :hidden]
    return jax.lax.with_sharding_constraint(
        out, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))


def moe_fused_ep_route(
    layer,
    x: jax.Array,
    gating_output: jax.Array,
    weights,
    mesh: Mesh,
    activation: str,
    scatter_results: bool,
    extra_backend_kwargs: dict | None,
    defer_all_reduce: bool,
    moe_chunk_size: int,
    activation_dtype,
) -> jax.Array | None:
    """The whole USE_MOE_FUSED_EP_KERNEL decision for one MoE call.

    Returns this layer's output where the call runs on the fused
    expert-parallel kernel, and None where it does not -- which asks the
    caller to run its own general MoE path for this batch, unchanged.

    The two answers are not the same kind of answer. A LAYER the kernel
    cannot take raises: the switch is a statement about which program serves
    expert-parallel MoE, so a model it cannot serve is an error rather than
    a quiet run on a different program, and the question is asked on every
    call so the error does not depend on which batch shapes a deployment
    happens to compile. A BATCH it cannot take, and one below the token
    threshold, returns None: that is a statement about the batch, and the
    same layer is served by both programs across a serving run.

    Everything this decision needs is read here rather than at the call
    site, so the caller's own path carries one guarded call and nothing
    else, and with the switch off nothing in this module is reached at all.
    """
    fused_ep_min_tokens = envs.MOE_FUSED_EP_KERNEL_MIN_TOKENS
    reason = unsupported_reason(
        layer=layer,
        x=x,
        gating_output=gating_output,
        weights=weights,
        mesh=mesh,
        activation=activation,
        scatter_results=scatter_results,
        extra_backend_kwargs=extra_backend_kwargs,
        defer_all_reduce=defer_all_reduce,
    )
    if reason is not None:
        # The remedy is offered only where it works. Nearly every refusal is
        # about something this kernel does not take and the general path
        # does, so unsetting the switch really does run the model -- but the
        # expert-count divisibility is a property of the expert sharding,
        # which the general path does the same way and refuses the same
        # condition. Printing it there sends the operator round a loop and
        # back to the same sentence under another program's name.
        remedy = ("" if reason_survives_the_general_path(reason) else
                  " Unset USE_MOE_FUSED_EP_KERNEL to run this model on "
                  "fused_moe_func instead.")
        raise ValueError(
            f"USE_MOE_FUSED_EP_KERNEL is on, so every expert-parallel MoE "
            f"call has to run on the fused expert-parallel MoE kernel, and "
            f"this layer cannot: {reason}.{remedy}")
    if x.shape[0] < fused_ep_min_tokens:
        batch_reason = (f"below the {fused_ep_min_tokens}-token "
                        "MOE_FUSED_EP_KERNEL_MIN_TOKENS threshold")
    else:
        batch_reason = unsupported_batch_reason(
            layer, x, mesh, num_experts=weights.w13_weight.shape[0])
    # The token count is the GLOBAL one across every expert-parallel shard,
    # which is the number the threshold is compared against.
    ep_width = mesh.devices.size
    if batch_reason is not None:
        # One line per batch shape, in both directions, at INFO: this is the
        # only channel that reports what actually happened rather than what
        # the switch says.
        logger.info_once(
            "%s: MoE at %d tokens (global, over %d expert-parallel shards) "
            "runs on fused_moe_func: %s.", layer._get_name(), x.shape[0],
            ep_width, batch_reason)
        return None
    if moe_chunk_size > 0:
        # The kernel does its own transport and has no chunked form. Say so
        # rather than drop it: the value bounds peak memory.
        logger.warning_once(
            "%s: moe_chunk_size=%d is not honoured on the fused "
            "expert-parallel MoE kernel, which carries the whole call in one "
            "piece. Peak memory for these calls is therefore set by the "
            "batch shape rather than by the chunk size, and the setting "
            "still applies to calls routed to fused_moe_func.",
            layer._get_name(), moe_chunk_size)
    # The kernel carries its own transport and its own permute, so these
    # select nothing here. None of them changes the answer -- the ones that
    # would are refused by unsupported_reason -- but a deployment that set
    # one and expects it to still apply is owed a line rather than silence.
    carried_elsewhere = (
        ("MOE_ALL_GATHER_ACTIVATION_DTYPE", activation_dtype),
        ("ENABLE_RS_KERNEL", envs.ENABLE_RS_KERNEL),
        ("ONEHOT_MOE_PERMUTE_THRESHOLD", envs.ONEHOT_MOE_PERMUTE_THRESHOLD),
    )
    not_carried = [
        f"{name}={value}" for name, value in carried_elsewhere if value
    ]
    if not_carried:
        logger.warning_once(
            "%s: %s select how the general MoE path moves or permutes its "
            "data. The fused expert-parallel MoE kernel carries its own "
            "transport and permute, so they do nothing for calls it serves; "
            "they still apply to calls routed to fused_moe_func.",
            layer._get_name(), " and ".join(not_carried))
    logger.info_once(
        "%s: MoE at %d tokens (global, over %d expert-parallel shards) runs "
        "on the fused expert-parallel MoE kernel: at or above the %d-token "
        "MOE_FUSED_EP_KERNEL_MIN_TOKENS threshold.", layer._get_name(),
        x.shape[0], ep_width, fused_ep_min_tokens)
    return moe_fused_ep_apply(
        layer=layer,
        x=x,
        gating_output=gating_output,
        weights=weights,
        mesh=mesh,
        activation=activation,
        scatter_results=scatter_results,
        extra_backend_kwargs=extra_backend_kwargs,
        defer_all_reduce=defer_all_reduce,
        # Both questions were asked above, for these operands. Handing the
        # answers back stops the apply function asking them a second time on
        # every accepted call -- two thirds of that check is the VMEM
        # estimate, which depends only on the model and cannot have moved
        # between these two lines.
        acceptance=(reason, batch_reason),
    )
