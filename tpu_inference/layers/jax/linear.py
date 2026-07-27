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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import QuantizationConfig
from tpu_inference.logger import init_logger

logger = init_logger(__name__)


class JaxEinsum(nnx.Einsum, JaxModule):
    """Einsum layer for JAX.

    Args:
        einsum_str: a string to denote the einsum equation.
        kernel_shape: the shape of the kernel.
        bias_shape: the shape of the bias. If this is None, a bias won't be used.
        param_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
    """

    def __init__(self,
                 einsum_str: str,
                 kernel_shape: tuple[int, ...],
                 rngs,
                 bias_shape: Optional[tuple[int, ...]] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 kernel_metadata=None,
                 bias_metadata=None,
                 **kwargs):
        # nnx.Einsum uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility, but forward it
        # as `param_dtype` so that weights are created with the correct dtype.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        # Use None sentinels: a shared mutable {} default would be populated
        # (eager_sharding) by the first instance and leak that state into every
        # other default-constructed layer.
        kernel_metadata = dict(kernel_metadata) if kernel_metadata else {}
        bias_metadata = dict(bias_metadata) if bias_metadata else {}
        if "eager_sharding" not in kernel_metadata:
            kernel_metadata["eager_sharding"] = False
        if "eager_sharding" not in bias_metadata:
            bias_metadata["eager_sharding"] = False
        nnx.Einsum.__init__(self,
                            rngs=rngs,
                            einsum_str=einsum_str,
                            kernel_shape=kernel_shape,
                            bias_shape=bias_shape,
                            kernel_metadata=kernel_metadata,
                            bias_metadata=bias_metadata,
                            **kwargs)
        self.kernel_init = kwargs.get("kernel_init",
                                      jax.nn.initializers.lecun_normal())
        # For compatibility. HF model use 'weight' as name suffix, we alias `self.kernel` to
        # `self.weight` such that `named_parameters()` can match the names in HF models.
        self.weight = self.kernel
        delattr(self, 'kernel')
        self.prefix = prefix

        if quant_config is None:
            self.quant_method = None
        elif (quant_method := quant_config.get_quant_method(self,
                                                            prefix=prefix)):
            assert isinstance(quant_method, QuantizeMethodBase)
            self.quant_method = quant_method
            self.quant_method.create_weights_jax(self, rngs=rngs)
        else:
            self.quant_method = None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        if self.quant_method is not None:
            return self.quant_method.apply_jax(self, inputs)

        output = jax.numpy.einsum(self.einsum_str, inputs, self.weight.value)
        if self.bias is not None:
            output += self.bias
        return output


class JaxLinear(JaxEinsum):
    """Linear layer for JAX.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        use_bias: If false, skip adding bias.
        param_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
        prefix: Prefix for parameter names.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 rngs,
                 *,
                 use_bias: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        JaxEinsum.__init__(self,
                           rngs=rngs,
                           einsum_str="mn,np->mp",
                           kernel_shape=(input_size, output_size),
                           bias_shape=(output_size, ) if use_bias else None,
                           quant_config=quant_config,
                           prefix=prefix,
                           **kwargs)


class JaxLmHead(nnx.Einsum, JaxModule):
    """Output projection (vocab head).

    Mirrors vLLM's `ParallelLMHead`: it is NOT a `LinearBase`-equivalent,
    so quant configs do not dispatch onto it. Use this for `lm_head` even
    on native FP8 / INT8 checkpoints whose `lm_head.weight` is bf16 — they
    would otherwise be assigned to a quantized slot pinned to `cpu_mesh()`
    and stranded on CPU at `create_jit_model` time.

    Intentionally does NOT inherit from `JaxEinsum` so that any
    `isinstance(layer, JaxEinsum)` check in a quant config naturally
    skips this class.
    """

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 rngs,
                 *,
                 prefix: str = "lm_head",
                 kernel_metadata=None,
                 **kwargs):
        # nnx.Einsum uses `param_dtype` for parameter initialization dtype.
        # Accept `dtype` as an alias for backward compatibility, but forward it
        # as `param_dtype` so that weights are created with the correct dtype.
        if "dtype" in kwargs and "param_dtype" not in kwargs:
            kwargs["param_dtype"] = kwargs.pop("dtype")
        # None sentinel: a shared mutable {} default would leak eager_sharding
        # state across default-constructed lm_head instances.
        kernel_metadata = dict(kernel_metadata) if kernel_metadata else {}
        if "eager_sharding" not in kernel_metadata:
            kernel_metadata["eager_sharding"] = False
        nnx.Einsum.__init__(self,
                            rngs=rngs,
                            einsum_str="TD,DV->TV",
                            kernel_shape=(hidden_size, vocab_size),
                            kernel_metadata=kernel_metadata,
                            **kwargs)
        # HF stores this weight under `lm_head.weight`; alias for named_parameters().
        self.weight = self.kernel
        delattr(self, 'kernel')
        self.prefix = prefix
        self.quant_method = None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return jax.numpy.einsum(self.einsum_str, inputs, self.weight.value)


class JaxMergedColumnParallelLinear(JaxLinear):
    """Merged version of JaxLinear. This is used to fuse multiple
    JaxLinear layers into one for better efficiency.

    The einsum string is the same as JaxLinear, but the weight is expected to
    have multiple output dimensions concatenated together, and the output will
    be split accordingly.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: a list of output dimensions for each fused linear layer.
        use_bias: If false, skip adding bias.
        param_dtype: Data type for the parameters.
        quant_config: Quantization configuration.
        prefix: Prefix for parameter names.
    """

    def __init__(self,
                 input_size: int,
                 output_sizes: list[int],
                 rngs,
                 *,
                 use_bias: bool = True,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 **kwargs):
        # Must be set before super().__init__(): JaxEinsum.__init__ calls
        # quant_config.get_quant_method(self), which reads self.output_sizes to
        # build the merged linear's QuantLinearConfig.
        self.output_sizes = output_sizes
        if quant_config is None:
            # When no quant_config is provided, UnquantizedConfig is inserted
            # to ensure quant_method is hooked up, which is necessary to properly
            # fuse sharded weights.
            # Imported locally to avoid an import cycle (the quantization
            # package imports this module).
            from tpu_inference.layers.jax.quantization.unquantized import \
                UnquantizedConfig
            quant_config = UnquantizedConfig({})
        super().__init__(input_size=input_size,
                         output_size=sum(output_sizes),
                         rngs=rngs,
                         use_bias=use_bias,
                         quant_config=quant_config,
                         prefix=prefix,
                         **kwargs)


class JaxQKVParallelLinear(JaxMergedColumnParallelLinear):
    """Fused QKV Parallel Linear layer for JAX-native models.

    Performs fused Q, K, and V projections in a single HBM read pass and
    partitions them locally per TPU device without incurring all-to-all collectives.
    """

    def __init__(self,
                 *,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 use_bias: bool,
                 dtype: jnp.dtype,
                 rngs: nnx.Rngs,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_size = num_heads * head_dim
        self.k_size = num_kv_heads * head_dim
        self.v_size = num_kv_heads * head_dim
        super().__init__(
            input_size=hidden_size,
            output_sizes=[self.q_size, self.k_size, self.v_size],
            rngs=rngs,
            use_bias=use_bias,
            quant_config=quant_config,
            prefix=prefix,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                              (None, "model")),
            bias_init=nnx.with_partitioning(nnx.initializers.uniform(),
                                            ("model", )) if use_bias else None,
        )

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        # Single projection operation (loads inputs from HBM once)
        outs = super().__call__(x)

        # Slice the concatenated [Q, K, V] output directly into individual Q, K, V components.
        # This is safe and 100% correct because super().__call__ (via apply_jax/_apply_fused)
        # has already un-interleaved the TP-sharded outputs and concatenated them into
        # a continuous [Q, K, V] layout across features.
        q_sz = self.q_size
        k_sz = self.k_size

        q_flat = outs[..., :q_sz]
        k_flat = outs[..., q_sz:q_sz + k_sz]
        v_flat = outs[..., q_sz + k_sz:]

        # Reshape directly to their global multi-head shapes (metadata change under JAX!)
        q = q_flat.reshape(outs.shape[:-1] + (self.num_heads, self.head_dim))
        k = k_flat.reshape(outs.shape[:-1] +
                           (self.num_kv_heads, self.head_dim))
        v = v_flat.reshape(outs.shape[:-1] +
                           (self.num_kv_heads, self.head_dim))

        return q, k, v
