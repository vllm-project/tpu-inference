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

import dataclasses
import enum
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class GDNMode(enum.StrEnum):
    # Multiple sequences per tile, one tile per sequence. Each sequence
    # carries `window_size` tokens: a single decoded token, or a speculative
    # verify window. See `GDNConfig.window_size`.
    BATCHED = enum.auto()
    PER_SEQ = enum.auto()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Dtypes:
    act_in: jnp.dtype
    act_out: jnp.dtype
    compute: jnp.dtype
    recurrent_state: jnp.dtype
    conv_state: jnp.dtype


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class GDNConfig:
    mode: GDNMode
    dtypes: Dtypes
    batch_size: int
    dim_size: int
    kernel_size: int
    tile_size: int
    num_kq_heads: int
    num_v_heads: int
    kq_head_dim: int
    v_head_dim: int
    num_buffers: int = 2
    # Max tokens per speculative verify window (= num_speculative_tokens + 1),
    # which is also the number of state checkpoints kept per sequence. The
    # kernel reads a sequence's initial state from
    # `state_indices[s] + read_offset[s]` and writes one checkpoint per window
    # position to `state_indices[s] + t`, which is how rejected draft tokens
    # are rolled back (by checkpoint selection). It is 1 without speculative
    # decoding, where only the state after the last real token is kept, so
    # shapes and loops can be sized off it unconditionally and the extra axis
    # / iterations fold away in the non-speculative paths.
    window_size: int = 1

    @property
    def chunk_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return self.tile_size
        # One tile per sequence, holding its whole verify window. BATCHED is
        # the single-token case of that (window_size == 1).
        return self.window_size

    @property
    def seq_tile_size(self) -> int:
        if self.mode == GDNMode.PER_SEQ:
            return 1
        return self.tile_size

    @property
    def prev_kernel_size(self) -> int:
        return self.kernel_size - 1

    @property
    def use_recurrent(self) -> bool:
        """Whether GDN runs the token-recurrent scan instead of the chunked one.

        Keeping more than one state checkpoint mandates the recurrent scan,
        since the chunked path only ever produces the final state.
        """
        return self.chunk_size == 1 or self.window_size > 1

    @property
    def v_dim_size(self) -> int:
        return self.num_v_heads * self.v_head_dim

    @property
    def kq_dim_size(self) -> int:
        return self.num_kq_heads * self.kq_head_dim

    @property
    def v_per_kq_head(self) -> int:
        return self.num_v_heads // self.num_kq_heads

    @property
    def aligned_num_v_heads(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        num_lanes = tpu_info.num_lanes
        return pl.cdiv(self.num_v_heads, num_lanes) * num_lanes

    def get_kernel_name(self) -> str:
        # Windows of different sizes compile to different kernels; keep them
        # distinguishable in profiles.
        suffix = f"_w{self.window_size}" if self.window_size > 1 else ""
        return f"fused_conv1d_gdn_{self.mode.value}{suffix}"

    def get_metadata(self) -> dict[str, str | int | float]:
        cfgs_dict = dataclasses.asdict(self)
        ret = {}
        for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
            key = jax.tree_util.keystr(path, simple=True, separator=".")
            if not isinstance(val, str | int | float):
                val = str(val)
            ret[key] = val
        return ret

    def get_out_shape(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (self.batch_size, self.num_v_heads, self.v_head_dim),
            self.dtypes.act_out,
        )

    # Fraction of VMEM the kernel may use. A multi-token window holds one
    # state checkpoint per position in VMEM, so for large-head models
    # (e.g. Qwen3.5-397B: 64 local v-heads -> 21M per buffered window even
    # at tile_size=1) the default 0.7 budget is not enough; those get a
    # higher limit and the wrapper sizes the tile against the same factor.
    DEFAULT_VMEM_FRACTION = 0.7
    WINDOWED_VMEM_FRACTION = 0.9

    def get_vmem_limit_bytes(self) -> int:
        tpu_info = pltpu.get_tpu_info()
        fraction = (self.WINDOWED_VMEM_FRACTION
                    if self.window_size > 1 else self.DEFAULT_VMEM_FRACTION)
        return int(fraction * tpu_info.vmem_capacity_bytes)

    def get_scratch_shape_dict(self) -> dict[str, Any]:
        conv_shape = (self.seq_tile_size, self.prev_kernel_size, 1,
                      self.dim_size)
        recurrent_shape = (
            self.seq_tile_size,
            self.num_v_heads,
            self.kq_head_dim,
            self.v_head_dim,
        )

        carry_conv_scratch = carry_recurrent_scratch = None
        # NOTE: In batched mode 1 seq = 1 tile, so inter-tile carry is not
        # needed.
        if self.mode == GDNMode.PER_SEQ:
            carry_conv_scratch = pltpu.VMEM(conv_shape, jnp.float32)
            carry_recurrent_scratch = pltpu.VMEM(recurrent_shape, jnp.float32)

        return dict(
            carry_conv_scratch_ref=carry_conv_scratch,
            carry_recurrent_scratch_ref=carry_recurrent_scratch,
        )
