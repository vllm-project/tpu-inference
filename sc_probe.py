from jax.experimental.pallas import tpu as pltpu

info = pltpu.get_tpu_info()
sc = info.sparse_core
print("=== SPARSECORE PROBE ===")
print("generation:", info.generation)
print("sparse_core:", sc)
print("num_lanes:", sc.num_lanes)
print("num_cores:", sc.num_cores)
print("num_subcores:", sc.num_subcores)

# Replicate the kernel's output-block-row math for Qwen3-30B-A3B (topk=8, bf16).
reduce_group_size = 8
packing = 2  # 32 // bits(bf16)
out_rows_per_step = sc.num_lanes // reduce_group_size
print("topk(reduce_group_size):", reduce_group_size, "packing(bf16):", packing)
print("out_rows_per_step = num_lanes // topk =", out_rows_per_step)
print("output_block_rows = out_rows_per_step // packing =",
      out_rows_per_step // packing,
      "(0 => zero-height block => swap crash)")
