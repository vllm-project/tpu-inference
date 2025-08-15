"""Autotune for ragged paged attention kernel."""

import os
import random
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu
from jax._src.lib import _jax
from jax._src.pallas.mosaic import error_handling
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import \
    tuned_block_sizes
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.kernel import (
    cdiv, dynamic_validate_inputs, get_min_heads_per_blk,
    ragged_paged_attention)

jax.config.parse_flags_with_absl()
random.seed(1234)

MODEL_CONFIGS = {
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
    # https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
    # https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8/raw/main/config.json
    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/raw/main/config.json
    'Meta-Llama-3-8B': {
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'num_devices': [1],
    },
    # https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
    # https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json
    # https://huggingface.co/RedHatAI/Meta-Llama-3.1-70B-Instruct-quantized.w8a8/raw/main/config.json
    # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/raw/main/config.json
    # https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/raw/main/config.json
    'Meta-Llama-3-70B': {
        'num_q_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'num_devices': [1, 8],
    },
}


# This is a typical distrubution of kv_lens and cu_q_lens.
def get_qkv_lens_example(max_num_batched_tokens, max_model_len,
                         actual_num_seqs):
    assert max_num_batched_tokens >= actual_num_seqs
    cu_q_lens = [i for i in range(actual_num_seqs + 1)]
    cu_q_lens[-1] = min(max_num_batched_tokens, max_model_len)
    kv_lens = [max_model_len for _ in range(actual_num_seqs)]
    return cu_q_lens, kv_lens


def autotune(
    example,
    key,
    max_num_seqs,
    block_kv_pages,
    block_q_tokens,
    total_num_pages=1000,
    num_iterations=100,
):
    (
        q_dtype_name,
        kv_dtype_name,
        num_q_heads_per_blk,
        num_kv_heads_per_blk,
        head_dim,
        page_size,
        max_num_batched_tokens,
        max_model_len,
    ) = key
    pages_per_seq = cdiv(max_model_len, page_size)
    cu_q_lens, kv_lens = example
    actual_num_seqs = len(kv_lens)
    cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
    kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
    cu_q_lens = jnp.pad(cu_q_lens, (0, max_num_seqs + 1 - cu_q_lens.shape[0]))
    kv_lens = jnp.pad(kv_lens, (0, max_num_seqs - kv_lens.shape[0]))

    q = jnp.array(
        np.random.rand(max_num_batched_tokens, num_q_heads_per_blk, head_dim),
        dtype=jnp.dtype(q_dtype_name),
    )
    kv_pages = jnp.array(
        np.random.rand(total_num_pages, page_size, num_kv_heads_per_blk * 2,
                       head_dim),
        dtype=jnp.dtype(kv_dtype_name),
    )
    page_indices = np.random.randint(0,
                                     total_num_pages,
                                     size=(max_num_seqs, pages_per_seq),
                                     dtype=jnp.int32)
    num_seqs = jnp.array([actual_num_seqs], dtype=jnp.int32)
    dynamic_validate_inputs(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
    )

    best_block_size = None
    best_t = None
    for num_kv_pages_per_block in block_kv_pages:
        if num_kv_pages_per_block > pages_per_seq:
            continue
        for num_q_per_block in block_q_tokens:
            # Warm up.
            try:
                ragged_paged_attention(
                    q,
                    kv_pages,
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    num_seqs,
                    num_kv_pages_per_block=num_kv_pages_per_block,
                    num_queries_per_block=num_q_per_block,
                    # Temporarily set a large vmem limit for autotune.
                    vmem_limit_bytes=100 * 1024 * 1024,
                ).block_until_ready()
            except ValueError as e:
                # Check if the error message starts with "Not implemented:"
                if str(e).startswith(
                        "Not implemented: num_combined_kv_heads=20 can not be XLA fully"
                        " tiled"):
                    print(
                        f"{num_kv_pages_per_block=}, {num_q_per_block=} Not implemented error: {e}"
                    )
                else:
                    print(
                        f"{num_kv_pages_per_block=}, {num_q_per_block=} ValueError: {e}"
                    )
                continue
            except error_handling.MosaicError as e:
                print(
                    f"{num_kv_pages_per_block=}, {num_q_per_block=} Caught MosaicError: {e}"
                )
                continue
            except _jax.XlaRuntimeError as e:
                s = str(e)
                line = s.splitlines()[0]
                print(
                    f"{num_kv_pages_per_block=}, {num_q_per_block=} Caught XlaRuntimeError: {line}"
                )
                continue

            start_time = time.perf_counter_ns()
            for _ in range(num_iterations):
                ragged_paged_attention(
                    q,
                    kv_pages,
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    num_seqs,
                    num_kv_pages_per_block=num_kv_pages_per_block,
                    num_queries_per_block=num_q_per_block,
                    # Temporarily set a large vmem limit for autotune.
                    vmem_limit_bytes=100 * 1024 * 1024,
                ).block_until_ready()
            end_time = time.perf_counter_ns()
            t = (end_time - start_time) / num_iterations
            print(
                f"[Debug] {num_kv_pages_per_block=}, {num_q_per_block=}, {t=}")

            if best_t is None or t < best_t:
                best_block_size = (num_kv_pages_per_block, num_q_per_block)
                best_t = t
    return best_block_size


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class AutotuneTest(jtu.JaxTestCase):

    @parameterized.product(
        q_dtype=[jnp.bfloat16],
        kv_dtype=[jnp.bfloat16],
        model_name=list(MODEL_CONFIGS.keys()),
        max_model_len=[5120],
        max_num_batched_tokens=[16, 32, 64, 128, 512, 1024, 2048, 4096],
        max_num_seqs=[128, 256, 512],
        page_size=[512],
        block_kv_pages=[(4, 8, 16, 32, 64, 128, 256)],
        block_q_tokens=[(32, 64, 128, 256)],
    )
    def test_autotune(
        self,
        q_dtype,
        kv_dtype,
        model_name,
        max_model_len,
        max_num_batched_tokens,
        max_num_seqs,
        page_size,
        block_kv_pages,
        block_q_tokens,
    ):
        # Currently we only use one example to autotune. If necessary, we can
        # construct decode-heavy or prefill-heavy examples.
        example = get_qkv_lens_example(
            max_num_batched_tokens,
            max_model_len,
            actual_num_seqs=35,
        )
        print(f"[Debug] {example=}")
        model = MODEL_CONFIGS[model_name]
        head_dim = cdiv(model["head_dim"], 128) * 128
        num_q_heads = model["num_q_heads"]
        num_kv_heads = model["num_kv_heads"]
        if max_model_len < page_size:
            return
        pages_per_seq = cdiv(max_model_len, page_size)
        if pages_per_seq > tuned_block_sizes.MAX_PAGES_PER_SEQ:
            return
        rows = []
        for num_devices in model["num_devices"]:
            assert num_q_heads % num_devices == 0
            assert (num_kv_heads % num_devices == 0
                    ), f"{model_name=}, {num_kv_heads=}, {num_devices=}"
            num_q_heads_per_device = num_q_heads // num_devices
            num_kv_heads_per_device = num_kv_heads // num_devices
            num_q_heads_per_blk, num_combined_kv_heads_per_blk = (
                get_min_heads_per_blk(
                    num_q_heads_per_device,
                    num_kv_heads_per_device * 2,
                    q_dtype,
                    kv_dtype,
                ))
            # Based on how `jax.experimental.pallas.ops.tpu.ragged_paged_attention` is
            # implemented, we only need to consider these params for autotuning:
            # - q_dtype
            # - kv_dtype
            # - num_q_heads_per_blk
            # - num_combined_kv_heads_per_blk
            # - head_dim
            # - page_size
            # - max_num_batched_tokens
            # - max_model_len = page_size * pages_per_seq
            key = (
                jnp.dtype(q_dtype).name,
                jnp.dtype(kv_dtype).name,
                num_q_heads_per_blk,
                num_combined_kv_heads_per_blk // 2,
                head_dim,
                page_size,
                max_num_batched_tokens,
                page_size * pages_per_seq,
            )

            best_block_size = autotune(
                example,
                key,
                max_num_seqs,
                block_kv_pages,
                block_q_tokens,
                num_iterations=100,
            )
            if best_block_size is not None:
                rows.append(f"{key}: {best_block_size},")
                print(f"{key}: {best_block_size},")

        print("Finished autotuning.")
        print(rows)
        with tempfile.NamedTemporaryFile(mode="w+",
                                         encoding='utf-8',
                                         delete=False,
                                         dir=os.getcwd()) as f:
            print(rows, file=f)
            print(f"Saved to file {f.name}")


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
