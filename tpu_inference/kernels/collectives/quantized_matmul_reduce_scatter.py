# SPDX-License-Identifier: Apache-2.0
"""Row-parallel int8-weight matmul fused with a reduce-scatter over tokens.

Computes, per device inside a ``shard_map`` over a single mesh axis of D
devices::

    x [T, K_local]  (bf16, or int8 with a per-row f32 scale sx [T, 1])
    w [K_local, N]  int8, s [N] f32 per-output-channel scale
    out [T / D, N] = psum_scatter(dequant(x) @ (w * s), axis,
                                  scatter_dimension=0, tiled=True)

which is what a row-parallel linear produces under sequence parallelism
(token-sharded output). XLA runs the matmul and the reduce-scatter one after
the other; on a v6e-4 the reduce-scatter adds ~35-60% to the matmul time for
prefill-sized inputs. This kernel overlaps them.

Algorithm (ring reduce-scatter fused into the matmul). Tokens are split into D
chunks of tm = T / D rows and the output columns into grid_n tiles of bn. At
ring step j (0..D-1) the device at ring position p computes, tile by tile, the
partial product for the chunk owned by ring position p-1-j (clockwise) or
p+1+j (counter-clockwise), adds the partial for the same chunk and tile that
its neighbour sent at step j-1 and, for j < D-1, starts an async remote DMA of
the sum to the next device. At j = D-1 the chunk is the device's own and the
sum is final. Each tile's send starts as soon as the tile is done, so the
transfer overlaps the following tiles' matmuls, and the receiver only waits for
the one tile it is about to add.

With ``bidir=True`` even output tiles go clockwise and odd tiles
counter-clockwise, so both ICI links of every chip carry traffic at once.

Send/receive buffers are VMEM [D-1, grid_n, tm, bn] bf16: one slot per step and
tile, never reused within a call, so no flow control is needed. A double
barrier with both ring neighbours at kernel start keeps a fast device's next
call from writing into a neighbour's buffers before that neighbour has
finished the previous call.

Accuracy: partial sums travel in bf16 (as XLA's bf16 reduce-scatter does) and
are accumulated in f32 on arrival.
"""

import functools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.collectives import util

# Barrier-semaphore id for this kernel. Every instance syncs with the same two
# ring neighbours on the same axis, so all instances can share it. Distinct
# from the 0 used by all_gather_matmul.
COLLECTIVE_ID = 3


def _vmem_limits() -> tuple[int, int]:
    """(scoped VMEM limit to request, share of it tile selection may plan for).

    Sized from the chip: 7/8 of VMEM capacity as the limit and 3/4 for the
    plan, leaving headroom for compiler temporaries (112 / 96 MiB on v6e).
    """
    try:
        cap = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except Exception:  # noqa: BLE001  (no TPU backend, e.g. CPU unit tests)
        cap = 128 * 1024 * 1024
    return cap * 7 // 8, cap * 3 // 4


def _kernel(tbl_ref, *refs, D, grid_n, bk, int8_act, bidir, axis):
    # tbl_ref (SMEM int32): [0:D] clockwise chunk per step, [D:2D]
    # counter-clockwise chunk per step, [2D] right neighbour, [2D+1] left.
    refs = list(refs)
    x_cw = refs.pop(0)
    x_ccw = refs.pop(0) if bidir else None
    sx_cw = refs.pop(0) if int8_act else None
    sx_ccw = refs.pop(0) if (int8_act and bidir) else None
    w_ref, s_ref, o_ref, recv, send, send_sems, recv_sems = refs

    j = pl.program_id(0)
    n = pl.program_id(1)
    right = tbl_ref[2 * D]
    left = tbl_ref[2 * D + 1]
    K = w_ref.shape[0]

    def rcopy(step, tile, dev):
        return pltpu.make_async_remote_copy(
            src_ref=send.at[step, tile],
            dst_ref=recv.at[step, tile],
            send_sem=send_sems.at[step, tile],
            recv_sem=recv_sems.at[step, tile],
            device_id={axis: dev},
            device_id_type=pl.DeviceIdType.MESH)

    @pl.when(jnp.logical_and(j == 0, n == 0))
    def _barrier():
        util.local_barrier(left, right, axis_name=axis)

    def body(x_ref, sx_ref, dev):
        acc = None
        # Static K loop keeps the dequantized weight slice small.
        for k0 in range(0, K, bk):
            xs = x_ref[:, k0:k0 + bk]
            ws = w_ref[k0:k0 + bk, :]
            if int8_act:
                d = jnp.dot(xs, ws, preferred_element_type=jnp.int32)
            else:
                d = jnp.dot(xs,
                            ws.astype(jnp.bfloat16),
                            preferred_element_type=jnp.float32)
            acc = d if acc is None else acc + d
        acc = acc.astype(jnp.float32)
        if int8_act:
            acc = acc * sx_ref[...]
        acc = acc * s_ref[...]

        @pl.when(j > 0)
        def _wait():
            rcopy(j - 1, n, dev).wait_recv()

        prev = recv[jnp.maximum(j - 1, 0), n].astype(jnp.float32)
        acc = acc + jnp.where(j > 0, prev, 0.0)

        @pl.when(j < D - 1)
        def _send():
            send[j, n] = acc.astype(send.dtype)
            rcopy(j, n, dev).start()

        @pl.when(j == D - 1)
        def _out():
            o_ref[...] = acc.astype(o_ref.dtype)

    if bidir:

        @pl.when(n % 2 == 0)
        def _cw():
            body(x_cw, sx_cw, right)

        @pl.when(n % 2 == 1)
        def _ccw():
            body(x_ccw, sx_ccw, left)
    else:
        body(x_cw, sx_cw, right)

    @pl.when(jnp.logical_and(j == D - 1, n == grid_n - 1))
    def _drain():
        for st in range(D - 1):
            for t in range(grid_n):
                # Same destination as the .start() for this tile.
                dev = left if (bidir and t % 2 == 1) else right
                rcopy(st, t, dev).wait_send()


def axis_devices(mesh, axis: str) -> list:
    """The devices along `axis` in the group that holds mesh position 0 on
    every other axis, in axis order. All groups run the same program, so this
    group's physical layout picks the ring order for all of them."""
    i = mesh.axis_names.index(axis)
    index = tuple(
        slice(None) if j == i else 0 for j in range(mesh.devices.ndim))
    return list(mesh.devices[index])


def ring_order(devices: Sequence) -> list[int]:
    """Axis positions in an order where ring neighbours are ICI neighbours.

    `devices` are the devices of the axis in axis order. For a full 2D grid of
    chip coordinates with an even side this returns a Hamiltonian cycle
    (e.g. a 2x2 slice: (0,0) (1,0) (1,1) (0,1)); otherwise the axis order,
    which is still correct, just with some multi-hop transfers.
    """
    D = len(devices)
    try:
        coords = [tuple(int(c) for c in d.coords[:2]) for d in devices]
    except (AttributeError, TypeError):
        return list(range(D))
    xs = sorted({c[0] for c in coords})
    ys = sorted({c[1] for c in coords})
    if len(set(coords)) != D or len(xs) * len(ys) != D or D < 4:
        return list(range(D))
    transpose = len(ys) % 2 == 1
    if transpose:
        xs, ys = ys, xs
    if len(ys) % 2 == 1:  # both sides odd: no Hamiltonian cycle
        return list(range(D))
    X, Y = len(xs), len(ys)
    # Rows are ys (even count). Walk row 0 left to right, snake rows 1..Y-1
    # over columns X-1..1, and return along column 0.
    cycle = [(x, 0) for x in range(X)]
    for r in range(1, Y):
        cols = range(X - 1, 0, -1) if r % 2 == 1 else range(1, X)
        cycle += [(x, r) for x in cols]
    cycle += [(0, r) for r in range(Y - 1, 0, -1)]
    if X == 1:
        cycle = [(0, r) for r in range(Y)]
    index = {}
    for i, c in enumerate(coords):
        key = (xs.index(c[1]),
               ys.index(c[0])) if transpose else (xs.index(c[0]),
                                                  ys.index(c[1]))
        index[key] = i
    return [index[c] for c in cycle]


def _largest_div(total: int,
                 cap: int,
                 quantum: int = 128,
                 even_quotient=False):
    best = None
    for b in range(quantum, min(total, cap) + 1, quantum):
        if total % b == 0 and (not even_quotient or (total // b) % 2 == 0):
            best = b
    return best


def _vmem_bytes(tm, K, N, bn, bk, D, int8_act, bidir):
    xbytes = 1 if int8_act else 2
    n_x = 2 if bidir else 1
    scratch = 2 * (D - 1) * tm * N * 2  # send + recv, bf16
    x_blocks = n_x * 2 * tm * K * xbytes  # double-buffered
    w_blocks = 2 * K * bn
    temps = bk * bn * 2 + 3 * tm * bn * 4 + 2 * tm * bn * 2
    return scratch + x_blocks + w_blocks + temps


def choose_blocks(T: int, K: int, N: int, D: int, int8_act: bool):
    """(bn, bk, bidir) for per-shard x [T, K], w [K, N] on D devices.

    Returns None when the shape is unsupported or does not fit in VMEM; the
    caller should fall back to a matmul followed by psum_scatter.
    """
    if D < 2 or T % (D * 128) or K % 128 or N % 128:
        return None
    tm = T // D
    bk = _largest_div(K, 1024)
    bn = _largest_div(N, 896, even_quotient=True)
    bidir = bn is not None
    if not bidir:
        bn = _largest_div(N, 896)
    if _vmem_bytes(tm, K, N, bn, bk, D, int8_act, bidir) > _vmem_limits()[1]:
        return None
    return bn, bk, bidir


def quantized_matmul_reduce_scatter(x: jax.Array,
                                    w: jax.Array,
                                    s: jax.Array,
                                    *,
                                    axis: str,
                                    order: Sequence[int],
                                    sx: jax.Array | None = None,
                                    bn: int | None = None,
                                    bk: int | None = None,
                                    bidir: bool | None = None) -> jax.Array:
    """Per-shard entry point; call inside a shard_map over `axis`.

    Args:
        x: [T, K_local] bf16 activations, or int8 with per-row scale `sx`.
        w: [K_local, N] int8 weight shard.
        s: [N] f32 per-output-channel weight scale.
        axis: the (single) mesh axis the K dimension is sharded over.
        order: static permutation of the axis positions in ring order (see
            `ring_order`).
        sx: [T, 1] f32 per-row activation scale when `x` is int8.
        bn, bk, bidir: block overrides; default from `choose_blocks`.

    Returns:
        [T / D, N] bf16: this device's token chunk of the reduced output, the
        same layout as ``psum_scatter(.., scatter_dimension=0, tiled=True)``.
    """
    D = len(order)
    int8_act = sx is not None
    T, K = x.shape
    N = w.shape[1]
    if bn is None or bk is None or bidir is None:
        plan = choose_blocks(T, K, N, D, int8_act)
        if plan is None:
            raise ValueError(
                f"unsupported shape x {x.shape} w {w.shape} D={D}")
        bn = bn or plan[0]
        bk = bk or plan[1]
        bidir = plan[2] if bidir is None else bidir
    tm = T // D
    grid_n = N // bn
    assert not bidir or grid_n % 2 == 0, (N, bn)

    pos_of = np.zeros(D, np.int32)
    for p, m in enumerate(order):
        pos_of[m] = p
    order_a = jnp.asarray(order, jnp.int32)
    p = jnp.asarray(pos_of)[jax.lax.axis_index(axis)]
    steps = jnp.arange(D, dtype=jnp.int32)
    tbl = jnp.concatenate([
        order_a[(p - 1 - steps) % D], order_a[(p + 1 + steps) % D],
        order_a[(p + 1) % D][None], order_a[(p - 1) % D][None]
    ])

    in_specs = [pl.BlockSpec((tm, K), lambda j, n, t: (t[j], 0))]
    args = [x]
    if bidir:
        in_specs.append(pl.BlockSpec((tm, K), lambda j, n, t: (t[D + j], 0)))
        args.append(x)
    if int8_act:
        sx = sx.reshape(T, 1).astype(jnp.float32)
        in_specs.append(pl.BlockSpec((tm, 1), lambda j, n, t: (t[j], 0)))
        args.append(sx)
        if bidir:
            in_specs.append(
                pl.BlockSpec((tm, 1), lambda j, n, t: (t[D + j], 0)))
            args.append(sx)
    in_specs += [
        pl.BlockSpec((K, bn), lambda j, n, t: (0, n)),
        pl.BlockSpec((1, bn), lambda j, n, t: (0, n)),
    ]
    args += [w, s.reshape(1, N).astype(jnp.float32)]
    # The output block index only moves during the last step, so only final
    # tiles are written back to HBM.
    out_spec = pl.BlockSpec((tm, bn), lambda j, n, t:
                            (0, jnp.where(j == D - 1, n, 0)))

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(D, grid_n),
        in_specs=in_specs,
        out_specs=out_spec,
        scratch_shapes=[
            pltpu.VMEM((D - 1, grid_n, tm, bn), jnp.bfloat16),  # recv
            pltpu.VMEM((D - 1, grid_n, tm, bn), jnp.bfloat16),  # send
            pltpu.SemaphoreType.DMA((D - 1, grid_n)),
            pltpu.SemaphoreType.DMA((D - 1, grid_n)),
        ],
    )
    kernel = functools.partial(_kernel,
                               D=D,
                               grid_n=grid_n,
                               bk=bk,
                               int8_act=int8_act,
                               bidir=bidir,
                               axis=axis)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tm, N), jnp.bfloat16),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            collective_id=COLLECTIVE_ID,
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=_vmem_limits()[0],
        ),
        cost_estimate=pl.CostEstimate(flops=2 * T * K * N,
                                      bytes_accessed=K * N + 2 * T * K +
                                      2 * T * N,
                                      transcendentals=0),
        name=(f"quantized_matmul_reduce_scatter_bn{bn}_bk{bk}_"
              f"{'bidir' if bidir else 'uni'}_{'i8' if int8_act else 'bf16'}"),
    )(tbl, *args)
