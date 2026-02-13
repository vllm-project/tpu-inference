"""Explore the efficiency of gather via one-hot encoding (permutation matrix)."""

import jax
import jax.numpy as jnp
import time


def gather_onehot(indices, x):
    """Gather using one-hot permutation matrix + matmul."""
    num_input = x.shape[0]
    perm_matrix = jax.nn.one_hot(indices, num_input, dtype=x.dtype)
    return jnp.dot(perm_matrix, x)


def gather_direct(indices, x):
    """Gather using direct indexing."""
    return x[indices]


def benchmark(fn, indices, x, warmup=10, iters=100):
    """Benchmark a function on TPU."""
    jitted = jax.jit(fn)

    # Warmup
    for _ in range(warmup):
        result = jitted(indices, x)
        result.block_until_ready()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        result = jitted(indices, x)
        result.block_until_ready()
    elapsed = time.perf_counter() - start

    return elapsed / iters, result


def main():
    num_tokens = 8192
    hidden_size = 6144
    num_indices_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print(f"num_tokens={num_tokens}, hidden_size={hidden_size}\n")
    print(f"{'len(indices)':>14} {'onehot_ms':>12} {'direct_ms':>12} {'speedup':>10} {'match':>7}")
    print("-" * 65)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    hidden_states = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)

    for num_indices in num_indices_list:
        indices = jax.random.randint(k2, (num_indices,), 0, num_tokens)

        t_onehot, res_onehot = benchmark(gather_onehot, indices, hidden_states)
        t_direct, res_direct = benchmark(gather_direct, indices, hidden_states)

        match = jnp.allclose(res_onehot, res_direct, atol=1e-2)
        speedup = t_direct / t_onehot

        print(
            f"{num_indices:>14} "
            f"{t_onehot * 1000:>11.3f} {t_direct * 1000:>11.3f} "
            f"{speedup:>9.2f}x {str(match):>7}"
        )


if __name__ == "__main__":
    main()
