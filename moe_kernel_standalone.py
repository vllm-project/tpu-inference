from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from tpu_inference.layers.common.fused_moe_gmm import fused_moe_func
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding

jax.config.parse_flags_with_absl()


def gen_moe_inputs(
    dtype,
    top_k,
    num_experts,
    hidden_size,
    intermediate_size,
    num_tokens,
    *,
    seed=1234,
    has_bias=False,
):
  key = jax.random.key(seed)

  a = (
      jax.random.normal(
          key, (num_tokens, hidden_size), dtype=jnp.float32
      ).astype(dtype)
      / 10
  )

  w1 = (
      jax.random.normal(
          key,
          (num_experts, hidden_size, 2 * intermediate_size),
          dtype=jnp.float32,
      )
      / 10
  ).astype(dtype)
  w2 = (
      jax.random.normal(
          key, (num_experts, intermediate_size, hidden_size), dtype=jnp.float32
      )
      / 10
  ).astype(dtype)

  if has_bias:
    b1 = (
        jax.random.normal(
            key, (num_experts, 1, 2 * intermediate_size), dtype=jnp.float32
        )
        / 10
    ).astype(dtype)
    b2 = (
        jax.random.normal(key, (num_experts, 1, hidden_size), dtype=jnp.float32)
        / 10
    ).astype(dtype)
  else:
    b1 = b2 = None

  gating_output = (
      jax.random.normal(key, (num_tokens, num_experts), dtype=jnp.float32)
      + jnp.arange(num_tokens * num_experts, dtype=jnp.float32).reshape(
          num_tokens, num_experts
      )
      / 100
  )

  # To generate unique top-k!
  top_k_indices = jax.random.randint(
      key,
      (num_tokens, top_k),
      minval=0,
      maxval=num_experts - 1,
      dtype=jnp.int32,
  )

  one_hot = (
      jnp.sum(
          jax.nn.one_hot(top_k_indices, num_experts, dtype=jnp.float32),
          axis=1,
      )
      * 30
  )

  gating_output = (gating_output + one_hot).astype(dtype)

  return a, w1, w2, b1, b2, gating_output


def sub_channel_quantize(x: jax.Array, quant_dtype: jnp.dtype, wsz: int = 256):
  """Quantizes x with sub-channel quantization on the 2nd minor."""
  if jnp.issubdtype(quant_dtype, jnp.floating):
    dtype_info = jnp.finfo(quant_dtype)
  else:
    dtype_info = jnp.iinfo(quant_dtype)
  dtype_max = float(dtype_info.max)
  w_lst, scale_lst = [], []
  assert len(x.shape) >= 2
  assert x.shape[-2] % wsz == 0
  for i in range(0, x.shape[1], wsz):
    y = x[:, i : i + wsz, :]
    abs_max = jnp.abs(y).max(axis=1, keepdims=True)
    scale = (abs_max / dtype_max).astype(jnp.float32)
    w = (y / scale).astype(quant_dtype)
    w_lst.append(w)
    scale_lst.append(scale)
  return jnp.concat(w_lst, axis=1), jnp.concat(scale_lst, axis=1)


@jtu.with_config(jax_numpy_dtype_promotion="standard")
class MoEKernelTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.mesh_devices = sorted(
        jax.devices(),
        key=lambda x: (
            x.coords[0],
            (-1 if x.coords[0] % 2 else 1) * x.coords[1],
        ),
    )

  def _test_moe(
      self,
      dtype,
      top_k,
      num_experts,
      hidden_size,
      intermediate_size,
      num_tokens,
      seed,
      renormalize_topk_logits: bool,
      act_fn: str = "silu",
      w_dtype: jnp.dtype | None = None,
      subc_quant_wsz: int | None = None,
      has_bias: bool = False,
      use_ep: bool = False,
      atol: float = 2e-1,
      rtol: float = 2e-1,
      capture_xprof=False,
      num_cores: int | None = None,
  ):
    if num_cores is None:
      num_cores = len(self.mesh_devices)
    mesh = Mesh(
        np.array(self.mesh_devices[:num_cores]).reshape(1, num_cores),
        axis_names=("data", "model"),
    )

    a, w1, w2, b1, b2, gating_output = gen_moe_inputs(
        dtype,
        top_k,
        num_experts,
        hidden_size,
        intermediate_size,
        num_tokens,
        seed=seed,
        has_bias=has_bias,
    )

    # Run baseline without any sharding or quantization.
    single_chip_mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("data", "model")
    )
    print("Running expected...")
    expected = jax.block_until_ready(
        fused_moe_func(
            a,
            w1,
            w2,
            w1_scale=None,
            w2_scale=None,
            w1_bias=b1,
            w2_bias=b2,
            activation=act_fn,
            gating_output=gating_output,
            topk=top_k,
            renormalize=renormalize_topk_logits,
            mesh=single_chip_mesh,
            use_ep=False,
        )
    )
    # Pre-process weights.
    n_shards = mesh.shape["model"]
    if not use_ep:
      w1 = reorder_concatenated_tensor_for_sharding(
          w1, [intermediate_size, intermediate_size], n_shards, 2
      )
      if has_bias:
        b1 = reorder_concatenated_tensor_for_sharding(
            b1, [intermediate_size, intermediate_size], n_shards, 2
        )

    # Quantize weights.
    w1_scale = w2_scale = None
    if w_dtype is not None:
      if subc_quant_wsz is None:
        subc_quant_wsz = 256
      w1, w1_scale = sub_channel_quantize(w1, w_dtype, subc_quant_wsz)
      w2, w2_scale = sub_channel_quantize(w2, w_dtype, subc_quant_wsz)

      w1_scale = jnp.expand_dims(w1_scale, axis=2)
      w2_scale = jnp.expand_dims(w2_scale, axis=2)

    # Shard weights.
    if use_ep:
      ep_sharding = NamedSharding(mesh, P("model"))

      w1 = jax.device_put(w1, ep_sharding)
      w2 = jax.device_put(w2, ep_sharding)
      if w_dtype is not None:
        w1_scale = jax.device_put(w1_scale, ep_sharding)
        w2_scale = jax.device_put(w2_scale, ep_sharding)
      if has_bias:
        b1 = jax.device_put(b1, ep_sharding)
        b2 = jax.device_put(b2, ep_sharding)
    else:
      w1 = jax.device_put(w1, NamedSharding(mesh, P(None, None, "model")))
      w2 = jax.device_put(w2, NamedSharding(mesh, P(None, "model", None)))
      if w_dtype is not None:
        w1_scale = jax.device_put(
            w1_scale, NamedSharding(mesh, P(None, None, None, "model"))
        )
        w2_scale = jax.device_put(
            w2_scale, NamedSharding(mesh, P(None, "model"))
        )
      if has_bias:
        b1 = jax.device_put(b1, NamedSharding(mesh, P(None, None, "model")))
        b2 = jax.device_put(b2, NamedSharding(mesh, P(None, None)))

    if capture_xprof:
      xprof_sess = xprof_session.XprofSession()
      # xprof_sess.start_session(trace_mode="TRACE_COMPUTE_AND_DMA")
      xprof_sess.start_session(trace_mode="TRACE_COMPUTE_AND_SYNC")
      # xprof_sess.start_session(trace_mode="TRACE_ALL")
    print("Running actual...")
    actual = jax.block_until_ready(
        fused_moe_func(
            a,
            w1,
            w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_bias=b1,
            w2_bias=b2,
            activation=act_fn,
            gating_output=gating_output,
            topk=top_k,
            renormalize=renormalize_topk_logits,
            mesh=mesh,
            use_ep=use_ep,
        )
    )
    if capture_xprof:
      url = xprof_sess.end_session_and_get_url()
      print(f"XProf URL: {url}")
    jax.debug.print("actual: {} {}", actual, actual.shape)
    jax.debug.print("expected: {} {}", expected, expected.shape)
    
    self.assertEqual(actual.dtype, a.dtype)
    self.assertEqual(expected.dtype, a.dtype)
    self.assertAllClose(actual, expected, atol=atol, rtol=rtol)

  @parameterized.product(
      num_tokens=[64],
      w_dtype=[jnp.float4_e2m1fn],
      has_bias=[True],
      block_size=[512],
      use_ep=[False],
  )
  def test_benchmark_gpt_oss_120(
      self, num_tokens, w_dtype, has_bias, block_size, use_ep
  ):
    num_experts = 128
    top_k = 4

    subc_quant_wsz = None
    if w_dtype is not None:
      subc_quant_wsz = block_size
    hidden_size = 3072
    intermediate_size = 3072

    dtype = jnp.bfloat16

    seed = 54321
    renormalize_topk_logits = True
    self._test_moe(
        dtype=dtype,
        top_k=top_k,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_tokens=num_tokens,
        seed=seed,
        renormalize_topk_logits=renormalize_topk_logits,
        subc_quant_wsz=subc_quant_wsz,
        has_bias=has_bias,
        w_dtype=w_dtype,
        use_ep=use_ep,
        act_fn="swigluoai",
        atol=1e1,
        rtol=1e1,
        num_cores=2,
        capture_xprof=False,
    )


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
