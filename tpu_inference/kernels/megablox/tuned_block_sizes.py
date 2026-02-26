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

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# Key:
#   - m: int, total number of tokens/rows
#   - k: int, input feature dimension
#   - n: int, output feature dimension per group
#   - num_total_groups: int, total experts in the model
#   - num_current_groups: int, experts assigned to this TPU shard
#   - lhs_dtype: str, data type name of the LHS matrix
#   - rhs_dtype: str, data type name of the RHS (weights) matrix
#   - quant_block_size: int, granularity of quantization scales
# Value:
#   - tm: int, m-dimension tile size
#   - tk: int, k-dimension tile size
#   - tn: int, n-dimension tile size
TUNED_BLOCK_SIZES = {
    (128, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 5,
    ),
    (128, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (128, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (128, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (128, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (256, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (256, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (256, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 5,
        256 * 24,
    ),
    (256, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (256, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (512, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (512, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (512, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (512, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (512, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 6,
    ),
    (1024, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (1024, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (1024, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 8,
    ),
    (1024, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (1024, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 24,
    ),
    (2048, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (2048, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (2048, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (2048, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 8,
        256 * 20,
    ),
    (2048, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (4096, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (4096, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (4096, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 12,
    ),
    (4096, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (4096, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (8192, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (8192, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (8192, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (8192, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 5,
    ),
    (8192, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (16384, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (16384, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (16384, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (16384, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (16384, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 6,
    ),
    (32768, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (32768, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (32768, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (32768, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (32768, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        128,
        256 * 24,
        256 * 8,
    ),
    (65536, 320, 6144, 160, 160, 'bfloat16', 'float8_e4m3fn', 320): (
        128,
        640,
        256 * 24,
    ),
    (65536, 2560, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        128,
        256 * 10,
        256 * 20,
    ),
    (65536, 2560, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 2560): (
        256 * 1,
        256 * 10,
        256 * 12,
    ),
    (65536, 6144, 5120, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 5,
    ),
    (65536, 6144, 6144, 160, 20, 'bfloat16', 'float8_e4m3fn', 6144): (
        256 * 1,
        256 * 24,
        256 * 6,
    ),
}


# TODO (jacobplatin): make this more generic
def round_up_to_multiple_of_128_within_limit(x: int, limit: int) -> int:
    """
    Rounds the given integer `x` up to the nearest multiple of 128, without
    exceeding the specified `limit`.

    If `x` is less than or equal to 128, returns 128.
    If `x` is less than `limit`, returns the smallest multiple of 128 greater
    than or equal to `x`.
    If `x` is greater than or equal to `limit`, searches for the largest
    multiple of 128 less than or equal to `limit` (down to 512) that divides `x`
    evenly, and returns it.
    If no such candidate is found, returns `limit`.

    Args:
        x (int): The integer to round up.
        limit (int): The upper bound (must be a multiple of 128).

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


def get_default_gmm_block_sizes(m: int, k: int, n: int,
                                g: int) -> tuple[int, int, int]:
    """
    Heuristic-based defaults for GMM tiling.

    Args:
        m (int): The total number of tokens.
        n (int): The output feature dimension.
        k (int): The input feature dimension.

    Returns:
        tuple[int, int, int]: A tuple (tm, tk, tn)
    """

    # TODO(Chengji): increase the upper limit tiling size of m when we can set
    # the vmem size to be used for gmm kernel.
    # NOTE: In average each expert has m // g tokens, but as it might be
    # unbalanced, here we doubled the token size when choosing tiling size of m.
    # 2m//g can be either greater or less than 512. If there are 32 tokens and
    # topk=2, m=topk * num_tokens=64, in this case, 2*m//g will be less than
    # 512.
    tm = round_up_to_multiple_of_128_within_limit(2 * m // g, 512)
    # NOTE(catswe): this divisor-search approach may produce suboptimal tile
    # sizes (e.g., num_tokens=64 and topk=5, so m=num_tokens*topk=320, will
    # result in tm=80 and 4 tiles, instead of tm=128 and 3 tiles), though it's
    # unclear if such a case currently exists in practice. one solution is to
    # replace _calculate_num_tiles(m, tm) with
    # _calculate_irregular_num_tiles(m, tm) in make_group_metadata. this would
    # allow using tm=128 with a partial final tile, like k and n already do.
    # another solution is to pad the tensor so m is always a multiple of 128.
    for candidate in range(tm, 0, -1):
        if m % candidate == 0:  # there's a requirement that m % tm == 0
            tm = candidate
            break
    # k/n correspond to n_input_features/n_output_features in the matmul so they
    # are normally greater than 2048, unless the num shards is large.
    tk = round_up_to_multiple_of_128_within_limit(k, 2048)
    tn = round_up_to_multiple_of_128_within_limit(n, 2048)
    return tm, tk, tn


def get_tuned_block_sizes(
    m: int,
    k: int,
    n: int,
    num_total_groups: int,
    num_current_groups: int,
    lhs_dtype: str,
    rhs_dtype: str,
    quant_block_size: int,
):
    """
    Retrieves optimized (TM, TK, TN) tiling parameters for the GMM kernel.
    """
    # GMM inputs must align to tile sizes; however, tile sizes themselves
    # are often powers of 2 or mxu multiples.
    key = (
        m,
        k,
        n,
        num_total_groups,
        num_current_groups,
        str(lhs_dtype),
        str(rhs_dtype),
        quant_block_size,
    )

    if key not in TUNED_BLOCK_SIZES:
        default_val = get_default_gmm_block_sizes(m, k, n, num_current_groups)
        logger.warning_once(
            f'[GMM kernel] using default block sizes for key: {key}: {default_val}'
        )
        return default_val

    return TUNED_BLOCK_SIZES.get(key)