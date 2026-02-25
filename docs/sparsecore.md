# SparseCore Kernel Writing

[SparseCores](https://openxla.org/xla/sparsecore) specialize in sparse memory access and operations, and have been an essential part of modern TPU for multiple versions. While most of the matmul and heavy-compute work will happen on TensorCores, offloading certain computation to SparseCores could improve overall performance.

This guide will give an overview on SparseCore architecture and show how to write Pallas kernels that runs on TPU SparseCores.


```python
from functools import partial
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

assert pltpu.get_tpu_info().sparse_core is not None, "No SparseCore found"
```

## Hardware overview

Depending on the version, a recent TPU chip may have 2 or 4 SparseCores. A SparseCore consists of multiple subcores, each having its own VMEM space. Below is a diagram for a SparseCore inside the TPU.



A walkthrough on each of the components:

* **Vector subcore (tiles)**: The vector processing subcores of a SparseCore. Each subcore has its own memory, so the data flow is independent.

* **Lanes (SIMD width)**: An SC vector subcore does computation on size-N vectors in a "Single instruction multiple data (SIMD) fashion. Computations will happen on all the numbers in a lane in a single instruction.

* **Scalar subcore**: The scalar processing subcore of a SparseCore. It is capable of scalar operations, dynamic indexing as well as initiating DMAs and streams.

* **Memory spaces**: Each vector subcore has its own VMEM and SMEM (omitted in the graph) space. They also have access to a shared VMEM space. The scalar subcore has its own SMEM. All these spaces connect with the TPU's HBM.

  * In Pallas, the VMEM spaces are denoted as `pltpu.VMEM` and `pltpu.VMEM_SHARED`, and SMEM is denoted as `pltpu.SMEM`.

  * In some other documentations, the shared VMEM could be called "Spmem", and the per-subcore VMEM called "TileSpmem" or "local Spmem".


Actual specs vary by TPU version. Here are some published TPU specs:

| Attribute | TPU v4 | TPU v5p | TPU v6e (Trillium) | TPU 7x (Ironwood) |
| :--- | :--- | :--- | :--- | :--- |
| SparseCores / Chip | 4 | 4 | 2 | 2 (4 physical cores) |
| Vector subcores / SparseCore | 16 | 16 | 16 | 16 |
| SIMD Width | 8 | 8 | 8 (F32)<br>16 (BF16) | 16 (F32)<br>32(BF16) |
| HBM Capacity | 32 GiB | 96 GiB | 32 GiB | 192 GB |

You can also use `pltpu.get_tpu_info()` to quickly obtain specs for your current hardware.


```python
# Quick way to query basic SC info

assert (sc_info := pltpu.get_tpu_info().sparse_core)
print(f"SparseCore info for TPU {pltpu.get_tpu_info().chip_version}:")
print(sc_info)
```

    SparseCore info for TPU 7x:
    SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16, dma_granule_size_bytes=64)


## Operations and workloads

A SparseCore consists of 16 smaller processing units, each with its own data flow. That makes it good for workloads that have the following characteristics:

* Highly parallel and irregular

* Random data access

* Medium-to-low amount of computation

* Frequent data communications

Some of the useful operations on SparseCore are:

* Small vector arithmetics

* Gather and scatter (indexed fetch & send)

* Sorting, unique, counts, histograms

* Ragged operations

## Express SparseCore hardware

Similar to in TensorCore, Pallas uses mesh to express the compute units in SparseCore. Depending on the processing unit you want to use, create a `ScalarSubcoreMesh` or a `VectorSubcoreMesh`.

Note that a `VectorSubcoreMesh` has two dimensions - `core` for the different SparseCores, and `subcore` for the multiple subcores on each SparseCore.

This allows you to apply the same programming model of TensorCores to write collectives on SparseCores. Check out our [collectives guide](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html) if you want to learn more.


```python
scalar_mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=sc_info.num_cores)
print(scalar_mesh)

vector_mesh = plsc.VectorSubcoreMesh(
    core_axis_name="core", subcore_axis_name="subcore"
)
print(vector_mesh)
```

    ScalarSubcoreMesh(axis_name='core', num_cores=2)
    VectorSubcoreMesh(core_axis_name='core', subcore_axis_name='subcore', num_cores=2, num_subcores=16)


## A basic SparseCore kernel

See below for a simple scalar subcore kernel that includes DMAs, per-core customizing and compute operations. Note that the scalar subcore can only do scalar operations.


```python
@jax.jit
def cumsum(x):
  @pl.kernel(out_shape=x, mesh=scalar_mesh,
             scratch_shapes=[pltpu.SMEM((x.shape[1],), x.dtype),
                             pltpu.SemaphoreType.DMA])
  def kernel(x_ref, o_ref, tmp_ref, sem):
    idx = jax.lax.axis_index('core')
    pltpu.async_copy(x_ref.at[idx], tmp_ref, sem).wait()

    @pl.loop(1, x.shape[1])
    def _(i):
      tmp_ref[i] += tmp_ref[i - 1]

    pltpu.async_copy(tmp_ref, o_ref.at[idx], sem).wait()

  return kernel(x)

x_shape = (sc_info.num_cores, sc_info.num_lanes)
x = jax.random.randint(jax.random.key(0), x_shape, 0, 64, jnp.int32)
np.testing.assert_array_equal(cumsum(x), jnp.cumsum(x, axis=1))
```

## Pipelining in SparseCore kernels

You can `pltpu.emit_pipeline` to write pipelined SparseCore kernels. The `core_axis_name` and `dimension_semantics` arguments to `emit_pipeline` enable partitioning the pipeline across SparseCores/subcores.


```python
SC_REG_OP_SHAPE = (1, sc_info.num_lanes)
dma_block = (8, 128)

@jax.jit
def sc_add_one(x):
  @pl.kernel(out_shape=x, mesh=vector_mesh, scratch_shapes=[])
  def sc_add_one_kernel(x_hbm_ref, o_hbm_ref):
    in_shape = x_hbm_ref.shape

    def sc_add_one_body(in_vmem, out_vmem):
      @pl.loop(0, in_vmem.shape[0], step=SC_REG_OP_SHAPE[0])
      def _(c0):
        @pl.loop(0, in_vmem.shape[1], step=SC_REG_OP_SHAPE[1])
        def _(c1):
          slc = (pl.ds(c0, SC_REG_OP_SHAPE[0]), pl.ds(c1, SC_REG_OP_SHAPE[1]))
          out_vmem.at[*slc][...] = in_vmem.at[*slc][...] + 1

    pltpu.emit_pipeline(
        sc_add_one_body,
        grid=(in_shape[0] // dma_block[0], in_shape[1] // dma_block[1]),
        in_specs=[pl.BlockSpec(block_shape=dma_block, index_map=lambda i, j: (i, j))],
        out_specs=[pl.BlockSpec(block_shape=dma_block, index_map=lambda i, j: (i, j))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
    )(x_hbm_ref, o_hbm_ref)
  return sc_add_one_kernel(x)

x = jax.random.randint(jax.random.key(0), (4096, 128), 0, 64, jnp.int32)
y = sc_add_one(x)
np.testing.assert_array_equal(y, x + 1)
```

Alternatively, you can use axis_index to compute the core index and use it to split up work across cores (example [here](https://docs.jax.dev/en/latest/pallas/tpu/core_map.html#mapping-over-sparsecores)).

## Overlapping TensorCore and SparseCore

It is very simple to overlap kernels written in TensorCore vs SparseCore: just put them together inside a `jax.jit`. The XLA compiler will handle their scheduling.


```python
@jax.jit
def tc_add_one(x):
  return x + 1
np.testing.assert_array_equal(tc_add_one(x), jnp.add(x, 1))

@jax.jit
def two_add_ones(x):
  return sc_add_one(x), tc_add_one(x)
jax.tree.map(np.testing.assert_array_equal, two_add_ones(x), (x + 1, x + 1));
```

A benchmark here shows the total time is less than the two functions combined.


```python
%timeit sc_add_one(x).block_until_ready()
%timeit tc_add_one(x).block_until_ready()

%timeit jax.block_until_ready(two_add_ones(x))
```

    136 µs ± 1.69 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    108 µs ± 4.38 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    200 µs ± 925 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)


## Gather and scatter

SparseCore has specific optimized ops for indexed retrievals and updates. Given an input or output ref in HBM (named `data`) and an array of indices in VMEM (named `indices`), it can quickly read from ("gather") or write to ("scatter") `data[indices]`.

We can use these gather/scatter by indexing a Ref with an indices Ref as part of an `async_copy` or `sync_copy`. For example, `sync_copy(data_ref.at[indices_ref], target_ref)` will trigger a gather.

Below is a kernel that pipelines loading indices into a vector subcore's VMEM. In the body, we execute a gather using those indices.


```python
batch_size = 4096
value_dim = 128
gather_window_size = 128
num_steps = 1024
sc_num_cores, sc_num_subcores = sc_info.num_cores, sc_info.num_subcores
num_indices = gather_window_size * sc_num_cores * sc_num_subcores * num_steps
data = jnp.arange(batch_size * value_dim).reshape(batch_size, value_dim)
indices = jax.random.randint(jax.random.key(0), (num_indices,), 0, batch_size, jnp.int32)


@jax.jit
def gather(data, indices):
  indices = indices.reshape((1, num_indices))
  @pl.kernel(out_shape=jax.ShapeDtypeStruct((num_indices, value_dim), data.dtype),
             mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)  # The gather op

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // gather_window_size,),
        in_specs=[pl.BlockSpec((1, gather_window_size),
                               index_map=lambda i: (0, i))],
        out_specs=[pl.BlockSpec((gather_window_size, value_dim),
                                index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  return kernel(data, indices)

out = gather(data, indices)
np.testing.assert_array_equal(out, jnp.take(data, indices, axis=0))

```

If you are doing indexed retrieval at the beginning of a kernel, you could use the `indexed_by` and `indexed_dim` argument of `plsc.BlockSpec` on the top-level `pl.pallas_call` to refer to another input as the indices of this input on this axis.

This call will parallelize the DMA from HBM to VMEM and the gather operation that does the indexed lookup, resulting in 4 pipeline stages: indices copy-in, gather, kernel computation and output copy-out. This allows you to overlap gather and any further computation on gathered outputs.

Note that the `plsc.BlockSpec` is experimental and subject to change.


```python
@jax.jit
def gather_add_one(data, indices):
  @partial(
      pl.pallas_call,
      out_shape=jax.ShapeDtypeStruct(shape=(num_indices, value_dim), dtype=data.dtype),
      grid=(num_indices // gather_window_size,),
      in_specs=(
          plsc.BlockSpec((gather_window_size, value_dim), indexed_by=1, indexed_dim=0),
          pl.BlockSpec((gather_window_size,), lambda i: i),
      ),
      out_specs=pl.BlockSpec((gather_window_size, value_dim), lambda i: (i, 0)),
      compiler_params=pltpu.CompilerParams(
          kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE,
          dimension_semantics=(pltpu.PARALLEL,),
      ),
  )
  def kernel(gathered_ref, _, o_ref):
    # gathered_ref is the gathered content of x[indices]
    @pl.loop(0, gather_window_size)
    def _(c0):
      @pl.loop(0, o_ref.shape[1], step=16)
      def _(c1):
        slc = (pl.ds(c0, 1), pl.ds(c1, 16))
        o_ref.at[*slc][...] = gathered_ref.at[*slc][...] + 1

  return kernel(data, indices)

out = gather_add_one(data, indices)
np.testing.assert_array_equal(out, jnp.take(data, indices, axis=0) + 1)

```

A scatter (indexed overwrite) is the opposite of a gather. See an example kernel below.


```python
@jax.jit
def scatter(data, indices):
  indices = indices.reshape((1, num_indices))
  @pl.kernel(out_shape=jax.ShapeDtypeStruct((batch_size, value_dim), data.dtype),
             mesh=vector_mesh, scratch_shapes=[])
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(x_vmem, i_vmem):
      pltpu.sync_copy(x_vmem, o_hbm.at[i_vmem.at[0]])  # The scatter op

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // gather_window_size,),
        in_specs=[pl.BlockSpec((gather_window_size, value_dim), index_map=lambda i: (i, 0)),
                  pl.BlockSpec((1, gather_window_size,), index_map=lambda i: (0, i))],
        out_specs=[],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(x_hbm, i_hbm)

  return kernel(data, indices)

gathered = jnp.take(data, indices, axis=0)
out = scatter(gathered, indices)
np.testing.assert_array_equal(out, data)

```

## Benchmark against TensorCore

Sparsecores are particularly good at gather and scatter operations. We can implement the same using vanilla JAX APIs, which by default run on TensorCore, and compare the results.


```python
%timeit jax.block_until_ready(gather(data, indices))

gather_tc = jax.jit(lambda x, i: jnp.take(x, i, axis=0))
gather_tc(data, indices).block_until_ready()

%timeit jax.block_until_ready(gather_tc(data, indices))
```

    4.07 ms ± 6.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    18.1 ms ± 4.17 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
