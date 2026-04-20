class PjRtCopyFuture:
  def Await(self) -> None: ...

def transfer_d2h_async(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> PjRtCopyFuture:
  """Asynchronously copies data from device to host.

  If the offset and size lists are omitted or empty, it performs a full copy.
  Otherwise, it performs partial copies based on the provided lists.

  Requirements:
    - Array rank must be >= 3.
    - The product of all non-major dimensions multiplied by the element size
      in bytes must be a multiple of the device tile size (typically 4KB).
    - Partial copy is NOT supported if the array is sharded on the major
      dimension.

  WARNING: The data layout on the host destination array may NOT be standard
  row-major. It preserves the physical layout of the TPU, which may include
  padding and tiling. Do NOT use the host array for any computations or
  assume standard NumPy layout. It is intended primarily for round-trip
  transfers or for saving/restoring raw buffers.

  Args:
    src_arr: The source JAX array on device.
    dst_arr: The destination JAX array on host.
    src_offsets_major_dim: Offsets in the source array along the major
      dimension.
    dst_offsets_major_dim: Offsets in the destination array along the major
      dimension.
    copy_sizes_major_dim: Sizes to copy along the major dimension.

  Returns:
    A list of PjRtCopyFuture objects that can be awaited.
  """
  ...

def transfer_h2d_async(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> PjRtCopyFuture:
  """Asynchronously copies data from host to device.

  If the offset and size lists are omitted or empty, it performs a full copy.
  Otherwise, it performs partial copies based on the provided lists.

  Requirements:
    - Array rank must be >= 3.
    - The product of all non-major dimensions multiplied by the element size
      in bytes must be a multiple of the device tile size (typically 4KB).
    - Partial copy is NOT supported if the array is sharded on the major
      dimension.

  WARNING: The source host array must have the specific physical layout
  expected by the TPU (including padding and tiling) if it was produced by
  a previous `transfer_d2h_async` call. Using a standard NumPy array or host
  array with different layout will result in corrupted data on the device.

  Args:
    src_arr: The source JAX array on host.
    dst_arr: The destination JAX array on device.
    src_offsets_major_dim: Offsets in the source array along the major
      dimension.
    dst_offsets_major_dim: Offsets in the destination array along the major
      dimension.
    copy_sizes_major_dim: Sizes to copy along the major dimension.

  Returns:
    A list of PjRtCopyFuture objects that can be awaited.
  """
  ...

def transfer_d2h(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None:
  """Synchronously copies data from device to host.

  Blocks until the transfer is complete.
  See `transfer_d2h_async` for requirements and warnings regarding layout.
  """
  ...

def transfer_h2d(
    src_arr,
    dst_arr,
    *,
    src_offsets_major_dim: list[int] = ...,
    dst_offsets_major_dim: list[int] = ...,
    copy_sizes_major_dim: list[int] = ...,
) -> None:
  """Synchronously copies data from host to device.

  Blocks until the transfer is complete.
  See `transfer_h2d_async` for requirements and warnings regarding layout.
  """
  ...
