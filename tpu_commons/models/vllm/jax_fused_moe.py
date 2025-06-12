import functools

import jax
import torch
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.tensor import t2j
from vllm.model_executor.layers.fused_moe import FusedMoE

P = PartitionSpec


def _round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without exceeding
    the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater than or
    equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest multiple of
    128 less than or equal to `limit` (down to 512) that divides `x` evenly, and
    returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128 and at least 128).

    Returns:
        int: The rounded value according to the rules above.

    Raises:
        AssertionError: If `limit` is less than 128 or not a multiple of 128.
    """
    assert limit >= 128 and limit % 128 == 0
    if x <= 128:
        return 128
    if x < limit:
        return (x + 127) // 128 * 128
    for candidate in range(limit, 511, -128):
        if x % candidate == 0:
            return candidate
    return limit


def _get_tiling_size_for_gmm_kernel(m: int, k: int, n: int,
                                    g: int) -> tuple[int, int, int]:
    """
    Calculate optimal tiling sizes for a GMM kernel in a Mixture of Experts
    (MoE) setting.

    Args:
        m (int): The total number of tokens.
        n (int): The output feature dimension.
        k (int): The input feature dimension.
        g (int): The number of experts.

    Returns:
        tuple[int, int, int]: A tuple (tm, tk, tn)
    """

    # TODO(Chengji): increase the upper limit tiling size of m when we can set
    # the vmem size to be used for gmm kernel.
    # NOTE: In average each expert has m // g tokens, but as it might be unbalanced,
    # here we doubled the token size when choosing tiling size of m. 2m//g can be
    # either greater or less than 512. If there are 32 tokens and topk=2,
    # m=topk * num_tokens=64, in this case, 2*m//g will be less than 512.
    tm = _round_up_to_multiple_of_128_within_limit(2 * m // g, 512)
    tm = min(tm, m)  # there's a requirement that m % tm == 0
    # k/n correspond to n_input_features/n_output_features in the matmul so they are
    # normally greater than 2048, unless the num shards is large.
    tk = _round_up_to_multiple_of_128_within_limit(k, 2048)
    tn = _round_up_to_multiple_of_128_within_limit(n, 2048)
    return tm, tk, tn


def sharded_gmm(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    transpose_rhs: bool,
    mesh: Mesh,
) -> jax.Array:
    # adapted from https://github.com/pytorch/xla/blob/1d409399474197c484894be90b75d9855393dda5/torch_xla/experimental/custom_kernel.py#L1401
    m, k, g = lhs.shape[0], lhs.shape[1], rhs.shape[0]
    n = rhs.shape[1] if transpose_rhs else rhs.shape[2]
    tm, tk, tn = _get_tiling_size_for_gmm_kernel(m, k, n, g)

    _gmm = functools.partial(
        gmm,
        preferred_element_type=lhs.dtype,
        tiling=(tm, tk, tn),
        transpose_rhs=transpose_rhs,
        group_offset=jnp.array(0),
    )

    return shard_map(
        _gmm,
        mesh=mesh,
        in_specs=(P(), P(None, 'model', None), P()),
        out_specs=(P(None, 'model')),
        check_rep=False,
    )(lhs, rhs, group_sizes)


def jax_fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating_output: jax.Array,
    topk: int,
    global_num_experts: int,
    renormalize: bool,
    reduce_results: bool,
    mesh: Mesh,
):
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
    """
    # adapted from https://github.com/vllm-project/vllm/blob/29fa5cac1cd731026f59084d93a822921507573c/vllm/model_executor/layers/fused_moe/moe_pallas.py#L26
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.size // hidden_size
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype
    assert (num_tokens * topk) % 16 == 0, (
        "The kernel requires num_tokens * topk to be a multiple of "
        f"16 but got {num_tokens}*{topk}={num_tokens*topk}")

    hidden_states = hidden_states.reshape(num_tokens, hidden_size)
    gating_output = gating_output.reshape(num_tokens, global_num_experts)

    topk_weights = jax.nn.softmax(gating_output.astype(jnp.float32), axis=-1)
    topk_weights, topk_indices = jax.lax.top_k(topk_weights, k=topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(dtype)

    topk_indices_flat = topk_indices.flatten()
    topk_argsort_indices = jnp.argsort(topk_indices_flat)
    topk_argsort_revert_indices = jnp.argsort(topk_argsort_indices)
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32).repeat(topk)
    token_indices_sorted = token_indices[topk_argsort_indices]
    group_sizes = jnp.bincount(topk_indices_flat, length=global_num_experts)

    x = hidden_states[token_indices_sorted]

    # x = torch.ops.xla.gmm(x, w1, group_sizes, transpose_rhs=True)
    x = sharded_gmm(x, w1, group_sizes, transpose_rhs=True, mesh=mesh)

    x = jax.nn.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]

    # x = torch.ops.xla.gmm(x, w2, group_sizes, transpose_rhs=True)
    x = sharded_gmm(x, w2, group_sizes, transpose_rhs=True, mesh=mesh)

    x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)
    x = x * jnp.expand_dims(topk_weights, axis=-1)
    x = x.sum(axis=-2)
    x = x.reshape(orig_shape)

    if reduce_results:
        x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))
    return x


def jax_fused_moe_func_padded(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    gating_output: jax.Array,
    topk: int,
    global_num_experts: int,
    renormalize: bool,
    reduce_results: bool,
    mesh: Mesh,
):
    # TODO(fanhongmin@google.com): Once the jax runner pads the input, we no longer need this.
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.size // hidden_size
    if num_tokens * topk < 16:
        assert 16 % (num_tokens *
                     topk) == 0, f"Cannot pad to 16: {num_tokens=}, {topk=}"
        n_repeats = 16 // (num_tokens * topk)

        reps = (n_repeats, ) + (1, ) * (hidden_states.ndim - 1)
        expanded_hidden_states = jnp.tile(hidden_states, reps)

        reps = (n_repeats, ) + (1, ) * (gating_output.ndim - 1)
        expanded_gating_output = jnp.tile(gating_output, reps)

        expanded_x = jax_fused_moe_func(expanded_hidden_states, w1, w2,
                                        expanded_gating_output, topk,
                                        global_num_experts, renormalize,
                                        reduce_results, mesh)
        x = expanded_x[:hidden_states.shape[0]]
        return x
    else:
        return jax_fused_moe_func(hidden_states, w1, w2, gating_output, topk,
                                  global_num_experts, renormalize,
                                  reduce_results, mesh)


class JaxFusedMoE(torch.nn.Module):

    def __init__(self, fused_moe: torch.nn.Module, mesh: Mesh):
        super().__init__()
        assert isinstance(fused_moe, FusedMoE)

        self.mesh = mesh
        self.top_k = fused_moe.top_k
        self.global_num_experts = fused_moe.global_num_experts
        self.renormalize = fused_moe.renormalize
        self.reduce_results = fused_moe.reduce_results

        self.w13_weight: Parameter
        self.w2_weight: Parameter

        self._load_weights_from_vllm_layer(fused_moe)
        self._shard_weight(mesh)

    def _shard_weight(self, mesh: Mesh):
        # Shard by the intermediate_size dim.
        self.w13_weight.apply_jax_(jax.device_put,
                                   NamedSharding(mesh, P(None, 'model', None)))
        self.w2_weight.apply_jax_(jax.device_put,
                                  NamedSharding(mesh, P(None, None, 'model')))

    def _load_weights_from_vllm_layer(self, fused_moe: torch.nn.Module):
        w13_weight = torch_view(t2j(fused_moe.w13_weight.data))
        w2_weight = torch_view(t2j(fused_moe.w2_weight.data))
        w13_weight = Parameter(w13_weight, requires_grad=False)
        w2_weight = Parameter(w2_weight, requires_grad=False)
        self.register_parameter("w13_weight", w13_weight)
        self.register_parameter("w2_weight", w2_weight)

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):

        _fused_moe_func = functools.partial(
            jax.jit(jax_fused_moe_func_padded,
                    static_argnames=[
                        "topk", "global_num_experts", "renormalize",
                        "reduce_results", "mesh"
                    ]),
            topk=self.top_k,
            global_num_experts=self.global_num_experts,
            renormalize=self.renormalize,
            reduce_results=self.reduce_results,
            mesh=self.mesh,
        )

        output = _fused_moe_func(
            jax_view(hidden_states),
            jax_view(self.w13_weight),
            jax_view(self.w2_weight),
            jax_view(router_logits),
        )

        return torch_view(output)
