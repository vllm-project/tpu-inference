import os
import warnings
from typing import Callable, Optional

import jax
import jaxlib
import torch
import torch_xla
from jax import numpy as jnp
from torch.library import impl
# Re-expose this API used that is referenced by docs
from torch_xla._internal.jax_workarounds import \
    jax_import_guard  # noqa: F401, pylint: disable=unused-import
from torch_xla._internal.jax_workarounds import requires_jax
from torch_xla.core.xla_model import XLA_LIB

_XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0") == "1"
DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)


@requires_jax
def convert_torch_dtype_to_jax(dtype: torch.dtype) -> "jnp.dtype":
    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    import jax.numpy as jnp
    if _XLA_USE_BF16:
        raise RuntimeError(
            "Pallas kernel does not support XLA_USE_BF16, please unset the env var"
        )
    if dtype == torch.float32:
        return jnp.float32
    elif dtype == torch.float64:
        return jnp.float64
    elif dtype == torch.float16:
        return jnp.float16
    elif dtype == torch.bfloat16:
        return jnp.bfloat16
    elif dtype == torch.int32:
        return jnp.int32
    elif dtype == torch.int64:
        return jnp.int64
    elif dtype == torch.int16:
        return jnp.int16
    elif dtype == torch.int8:
        return jnp.int8
    elif dtype == torch.uint8:
        return jnp.uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@requires_jax
def to_jax_shape_dtype_struct(tensor: torch.Tensor) -> "jax.ShapeDtypeStruct":
    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    import jax

    return jax.ShapeDtypeStruct(tensor.shape,
                                convert_torch_dtype_to_jax(tensor.dtype))


def _extract_backend_config(
        module: "jaxlib.mlir._mlir_libs._mlir.ir.Module") -> Optional[str]:
    """
  This algorithm intends to extract the backend config from the compiler IR like the following,
  and it is not designed to traverse any generic MLIR module.

  module @jit_add_vectors attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
    func.func public @main(%arg0: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}, %arg1: tensor<8xi32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<8xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
      %0 = call @add_vectors(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @add_vectors(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @wrapped(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @wrapped(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = call @apply_kernel(%arg0, %arg1) : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
    func.func private @apply_kernel(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
      %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSMTkuMC4wZ2l0AAErCwEDBQcJAQMLAwUDDQcFDxEJBRMVA3lZDQFVBwsPEw8PCw8PMwsLCwtlCwsLCwsPCw8PFw8LFw8PCxcPCxcTCw8LDxcLBQNhBwNZAQ0bBxMPGw8CagMfBRcdKy0DAycpHVMREQsBBRkVMzkVTw8DCxUXGRsfCyELIyUFGwEBBR0NCWFmZmluZV9tYXA8KGQwKSAtPiAoZDApPgAFHwUhBSMFJQUnEQMBBSkVLw8dDTEXA8IfAR01NwUrFwPWHwEVO0EdPT8FLRcD9h8BHUNFBS8XA3InAQMDSVcFMR1NEQUzHQ1RFwPGHwEFNSN0cHUubWVtb3J5X3NwYWNlPHZtZW0+ACNhcml0aC5vdmVyZmxvdzxub25lPgAXVQMhBx0DJwMhBwECAgUHAQEBAQECBASpBQEQAQcDAQUDEQETBwMVJwcBAQEBAQEHAwUHAwMLBgUDBQUBBwcDBQcDAwsGBQMFBQMLCQdLRwMFBQkNBwMJBwMDCwYJAwUFBRENBAkHDwURBQABBgMBBQEAxgg32wsdE2EZ2Q0LEyMhHSknaw0LCxMPDw8NCQsRYnVpbHRpbgBmdW5jAHRwdQBhcml0aAB2ZWN0b3IAbW9kdWxlAHJldHVybgBjb25zdGFudABhZGRpAGxvYWQAc3RvcmUAL3dvcmtzcGFjZXMvd29yay9weXRvcmNoL3hsYS90ZXN0L3Rlc3Rfb3BlcmF0aW9ucy5weQBhZGRfdmVjdG9yc19rZXJuZWwAZGltZW5zaW9uX3NlbWFudGljcwBmdW5jdGlvbl90eXBlAHNjYWxhcl9wcmVmZXRjaABzY3JhdGNoX29wZXJhbmRzAHN5bV9uYW1lAG1haW4AdmFsdWUAL2dldFt0cmVlPVB5VHJlZURlZigoQ3VzdG9tTm9kZShOREluZGV4ZXJbKFB5VHJlZURlZigoQ3VzdG9tTm9kZShTbGljZVsoMCwgOCldLCBbXSksKSksICg4LCksICgpKV0sIFtdKSwpKV0AYWRkX3ZlY3RvcnMAdGVzdF90cHVfY3VzdG9tX2NhbGxfcGFsbGFzX2V4dHJhY3RfYWRkX3BheWxvYWQAPG1vZHVsZT4Ab3ZlcmZsb3dGbGFncwAvYWRkAC9zd2FwW3RyZWU9UHlUcmVlRGVmKChDdXN0b21Ob2RlKE5ESW5kZXhlclsoUHlUcmVlRGVmKChDdXN0b21Ob2RlKFNsaWNlWygwLCA4KV0sIFtdKSwpKSwgKDgsKSwgKCkpXSwgW10pLCkpXQA=\22, \22needs_layout_passes\22: true}}", kernel_name = "add_vectors_kernel", operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<0> : tensor<1xindex>]} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
      return %0 : tensor<8xi32>
    }
  }

  Basically, what we are looking for is a two level of operations, and the tpu_custom_call operation in the inner level. It will return None if the payload is not found.
  """
    for operation in module.body.operations:
        assert len(operation.body.blocks
                   ) == 1, "The passing module is not compatible."
        for op in operation.body.blocks[0].operations:
            if op.name == "stablehlo.custom_call":
                return op.backend_config.value
    return None


@requires_jax
def trace_pallas(kernel: Callable,
                 *args,
                 static_argnums=None,
                 static_argnames=None,
                 use_cache=False,
                 **kwargs):
    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    import jax
    import jax._src.pallas.mosaic.pallas_call_registration

    jax_args = []  # for tracing
    tensor_args = []  # for execution
    for i, arg in enumerate(args):
        # TODO: Could the args be a tuple of tensors or a list of tensors? Flattern them?
        if torch.is_tensor(arg):
            # ShapeDtypeStruct doesn't have any storage and thus is very suitable for generating the payload.
            jax_meta_tensor = to_jax_shape_dtype_struct(arg)
            jax_args.append(jax_meta_tensor)
            tensor_args.append(arg)
        else:
            jax_args.append(arg)

    hash_key = ()
    if use_cache:
        global trace_pallas_arg_to_payload
        # implcit assumption here that everything in kwargs is hashable and not a tensor,
        # which is true for the gmm and tgmm.
        hash_key = (jax.config.jax_default_matmul_precision, kernel,
                    static_argnums, tuple(static_argnames)
                    if static_argnames is not None else static_argnames,
                    tuple(jax_args), repr(sorted(kwargs.items())).encode())
        if hash_key in trace_pallas_arg_to_payload:
            torch_xla._XLAC._xla_increment_counter('trace_pallas_cache_hit', 1)
            return trace_pallas_arg_to_payload[hash_key], tensor_args

    # Here we ignore the kwargs for execution as most of the time, the kwargs is only used in traced code.
    ir = jax.jit(kernel,
                 static_argnums=static_argnums,
                 static_argnames=static_argnames).lower(
                     *jax_args, **kwargs).compiler_ir()
    payload = _extract_backend_config(ir)

    if use_cache:
        # if we reach here it means we have a cache miss.
        trace_pallas_arg_to_payload[hash_key] = payload

    return payload, tensor_args


@requires_jax
def ragged_paged_attention(
    q,  # [max_num_batched_tokens, num_q_heads, head_dim]
    kv_pages,  # [total_num_pages, page_size, num_combined_kv_heads, head_dim]
    kv_lens,  # i32[max_num_seqs]
    page_indices,  # i32[max_num_seqs, pages_per_seq]
    cu_q_lens,  # i32[max_num_seqs + 1]
    num_seqs,  # i32[1]
    *,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=None,
    # kernel tuning parameters
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    # Import JAX within the function such that we don't need to call the jax_import_guard()
    # in the global scope which could cause problems for xmp.spawn.
    from tpu_commons.kernels.ragged_paged_attention.kernel import \
        ragged_paged_attention as ragged_attention

    if vmem_limit_bytes is None:
        vmem_limit_bytes = 64 * 1024 * 1024

    payload, _ = trace_pallas(
        ragged_attention,
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        static_argnames=[
            "sm_scale",
            "sliding_window",
            "soft_cap",
            "mask_value",
            "num_kv_pages_per_block",
            "num_queries_per_block",
            "vmem_limit_bytes",
        ],
    )

    seq_buf_idx = torch.tensor([0, 0], dtype=torch.int32).to("xla")
    output = torch_xla._XLAC._xla_tpu_custom_call(
        [
            kv_lens,
            page_indices,
            cu_q_lens,
            seq_buf_idx,
            num_seqs,
            q,
            kv_pages,
        ],
        payload,
        [  # output shape
            q.shape
        ],
        [  # output dtype
            q.dtype,
        ])
    return output[0]


def non_xla_ragged_paged_attention(q, kv, attention_type):
    # This will be called when dynamo use fake tensor to construct the fake output.
    # We need to make sure output tensor's shape is correct.
    if kv.device != torch.device("meta"):
        warnings.warn(
            f'XLA {attention_type} attention should only be applied to tensors on XLA device'
        )

    # Return orignal shape of q.
    return torch.empty_like(q)


XLA_LIB.define(
    "ragged_paged_attention(Tensor q, Tensor kv_pages, Tensor kv_lens, Tensor page_indices, "
    "Tensor cu_q_lens, Tensor num_seqs, float sm_scale=1, int? sliding_window=None, "
    "float? soft_cap=None, float? mask_value=None,"
    "int? num_kv_pages_per_block=None, int? num_queries_per_block=None, int? vmem_limit_bytes=None) -> Tensor",
)


@impl(XLA_LIB, "ragged_paged_attention", "XLA")
def ragged_paged_attention_xla(
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    kv_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    num_seqs: torch.Tensor,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=None,
    # kernel tuning parameters
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
    return ragged_paged_attention(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes)


@impl(XLA_LIB, "ragged_paged_attention", "CompositeExplicitAutograd")
def ragged_paged_attention_non_xla(
    q: torch.Tensor,
    kv_pages: torch.Tensor,
    kv_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    num_seqs: torch.Tensor,
    sm_scale=1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value=None,
    use_kernel=True,
    # kernel tuning parameters
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
    return non_xla_ragged_paged_attention(q, kv_pages, "paged")
