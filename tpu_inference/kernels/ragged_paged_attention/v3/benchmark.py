"Benchmark for ragged paged attention."

# Run
# - v5e:
#   blaze test --test_output=errors //experimental/users/jevinjiang/ullm:google/ragged_paged_attention/v3/benchmark_vl
# - v6e:
#   blaze test --test_output=errors //experimental/users/jevinjiang/ullm:google/ragged_paged_attention/v3/benchmark_gl
# - v7:
#   blaze test --test_output=errors //experimental/users/jevinjiang/ullm:google/ragged_paged_attention/v3/benchmark_gf --test_env=LIBTPU_INIT_ARGS=--xla_tpu_dvfs_p_state=7 --test_arg='--xla_tpu_dvfs_p_state=7'
import uuid
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import ragged_paged_attention, get_kv_cache_shape
from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    cdiv,
    next_power_of_2,
)

from tensorflow.tsl.profiler.protobuf import xplane_pb2

MODEL_CONFIGS = {}


head_dim_options = [128, 256]
num_kv_heads_per_card_options = [2, 4, 8] # TODO: 1
num_q_heads_per_card_options = [4, 8, 16]
batch_sizes = [256]  # [256, 512, 1024]


head_dim_options = [128]
num_kv_heads_per_card_options = [2] # TODO: 1
num_q_heads_per_card_options = [4]
batch_sizes = [256]  # [256, 512, 1024]

for head_dim in head_dim_options:
    for num_kv_heads in num_kv_heads_per_card_options:
        for num_q_heads in num_q_heads_per_card_options:
            for batch_size in batch_sizes:
                if num_q_heads >= num_kv_heads:
                    key = f'batch_{batch_size}_q_{num_q_heads}_kv_{num_kv_heads}_head_{head_dim}_2_devices'
                    MODEL_CONFIGS[key] = {
                        'num_q_heads': num_q_heads,
                        'num_kv_heads': num_kv_heads,
                        'head_dim': head_dim,
                        'batch_size': batch_size,
                        'num_devices': [2],
                    }

def get_device_name(num_devices: int | None = None):
  kind = jax.devices()[0].device_kind
  if 'TPU' not in kind:
    raise RuntimeError('Expected TPU devices')
  suffix = ''
  if kind.endswith(' lite'):
    kind = kind[: -len(' lite')]
    suffix = 'e'
  if kind == 'TPU7x':
    kind = 'TPU v7'
  assert kind[:-1] == 'TPU v', kind
  kind += suffix
  if num_devices is not None:
    kind += f'-{num_devices}'
  return kind


import os
import csv


import gzip
import json
def extract_op_times_from_json(profiler_path, name):
    """Parses a trace.json.gz file and extracts operation durations."""
    # walk the directory and find the trace.json.gz file
    json_gz_path = None
    for root, dirs, files in os.walk(profiler_path):
        for file in files:
            if file.endswith('.trace.json.gz'):
                json_gz_path = os.path.join(root, file)
                break
    with gzip.open(json_gz_path, 'rt', encoding='utf-8') as f:
        trace_data = json.load(f)
    t = 0
    cnt = 0
    trace_events = trace_data.get('traceEvents', [])
    for event in trace_events:
        if event.get('ph') == 'X':  # Complete event
            op_name = event["name"]
            if name in op_name:
                t += event['dur']
                cnt += 1

    return t, cnt, json_gz_path



def extract_op_times_from_xplane(profiler_path, op_name):
  xplane_pb_path = None
  for root, dirs, files in os.walk(profiler_path):
      for file in files:
          if file.endswith('.xplane.pb'):
              xplane_pb_path = os.path.join(root, file)
              break

  # Read the serialized XSpace proto from the file
  with open(xplane_pb_path, 'rb') as f:
      serialized_space = f.read()

  # Deserialize the XSpace message
  space = xplane_pb2.XSpace()
  space.ParseFromString(serialized_space)

  total_duration_ps = 0
  event_count = 0

  # Iterate through planes, lines, and events
  for plane in space.planes:
      # Metadata map for event names for the current plane
      event_metadata_map = plane.event_metadata

      for line in plane.lines:
          for event in line.events:
              # Get the event name from the metadata map
              metadata = event_metadata_map.get(event.metadata_id)
              if not metadata:
                  continue

              event_name = metadata.name
              if op_name in event_name:
                  total_duration_ps += event.duration_ps
                  event_count += 1

  # Convert total duration from picoseconds to microseconds
  total_duration_us = total_duration_ps / 1000000.0

  return total_duration_us, event_count, xplane_pb_path


os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_dvfs_p_state=7"

def schedule(
    num_reqs,
    inp_len,
    out_len,
    max_num_tokens,
    max_num_seqs,
):
  # Schedule the requets.
  tasks = []
  pq_inp = [inp_len for _ in range(num_reqs)]
  pq_out = [out_len for _ in range(num_reqs)]
  completed = 0

  while completed < num_reqs:
    kv_lens = []
    cu_q_lens = [0]
    decode_only = True
    decode_end = 0
    for i in range(completed, num_reqs):
      q_len = min(pq_inp[i], max_num_tokens - cu_q_lens[-1])
      assert q_len > 0
      pq_inp[i] -= q_len
      if pq_inp[i] == 0:
        if pq_out[i] == 0:
          completed += 1
        else:
          pq_out[i] -= 1
          pq_inp[i] += 1
      if decode_only:
        if q_len == 1:
          decode_end += 1
        else:
          decode_only = False
      cu_q_lens.append(cu_q_lens[-1] + q_len)
      kv_lens.append(inp_len + out_len - pq_inp[i] - pq_out[i])
      if len(kv_lens) >= max_num_seqs or cu_q_lens[-1] >= max_num_tokens:
        break
    num_seq = len(kv_lens)
    kv_lens_ = jnp.array(kv_lens, dtype=jnp.int32)
    cu_q_lens_ = jnp.array(cu_q_lens, dtype=jnp.int32)
    padding = max_num_seqs - num_seq
    distribution = jnp.array([decode_end, decode_end, num_seq], dtype=jnp.int32)
    kv_lens_ = jnp.pad(kv_lens_, (0, padding))
    cu_q_lens_ = jnp.pad(cu_q_lens_, (0, padding))
    tasks.append((distribution, cu_q_lens_, kv_lens_))
  return tasks


def run_benchmark(
    model_config,
    *,
    num_devices=1,
    num_reqs=1000,
    inp_len=1800,
    out_len=128,
    max_num_tokens=512,
    max_num_seqs=512,
    num_pages=1000,
    page_size=16,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
    # TODO(jevinjiang): remove after the autotuned table is ready.
    vmem_limit_bytes=64 * 1024 * 1024,
):
  pages_per_seq = cdiv(inp_len + out_len, page_size)
  actual_num_q_heads = model_config["num_q_heads"]
  actual_num_kv_heads = model_config["num_kv_heads"]
  actual_head_dim = model_config["head_dim"]
  assert actual_num_q_heads % num_devices == 0
  assert actual_num_kv_heads % num_devices == 0
  actual_num_q_heads //= num_devices
  actual_num_kv_heads //= num_devices

  q_shape = (max_num_tokens, actual_num_q_heads, actual_head_dim)
  kv_shape = (max_num_tokens, actual_num_kv_heads, actual_head_dim)
  kv_cache_shape = get_kv_cache_shape(
      num_pages,
      page_size,
      actual_num_kv_heads,
      actual_head_dim,
      kv_dtype,
  )
  q = jnp.array(
      np.random.rand(*q_shape),
      dtype=q_dtype,
  )
  k = jnp.array(
      np.random.rand(*kv_shape),
      dtype=kv_dtype,
  )
  v = jnp.array(
      np.random.rand(*kv_shape),
      dtype=kv_dtype,
  )
  kv_cache = jnp.array(
      np.random.rand(*kv_cache_shape),
      dtype=kv_dtype,
  )
  page_indices = np.random.randint(
      0, num_pages, size=(max_num_seqs * pages_per_seq,), dtype=jnp.int32
  )

  k_scale = 0.5 if q_dtype != kv_dtype else None
  v_scale = 0.5 if q_dtype != kv_dtype else None

  tasks = schedule(num_reqs, inp_len, out_len, max_num_tokens, max_num_seqs)

  # Xprof the kernel performance for all tasks.
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = os.getenv("PYTHON_TRACER_LEVEL", 0)

  uid = f"uid_{uuid.uuid4()}_num_reqs={num_reqs},inp_len={inp_len},out_len={out_len},max_num_tokens={max_num_tokens},max_num_seqs={max_num_seqs},num_pages={num_pages},page_size={page_size},q_dtype={q_dtype},kv_dtype={kv_dtype}_num_devices={num_devices}_num_q_heads={actual_num_q_heads}_num_kv_heads={actual_num_kv_heads}_head_dim={actual_head_dim}"
  uid_profile_dir = f"/tmp/jax-trace/{uid}"
  count = 0
  with jax.profiler.trace(uid_profile_dir, profiler_options=options):
    # try:
    for distribution, cu_q_lens, kv_lens in tasks:
      count += 1
      _, kv_cache = jax.block_until_ready(
          ragged_paged_attention(
              q,
              k,
              v,
              kv_cache,
              kv_lens,
              page_indices,
              cu_q_lens,
              distribution,
              vmem_limit_bytes=vmem_limit_bytes,
              k_scale=k_scale,
              v_scale=v_scale,
          )
      )
    # except KeyError as e:
    #   print(f"Failed to run {uid} due to missing block size.")
    #   raise e
  t_us, num_calls, xplane_pb_path = extract_op_times_from_xplane(uid_profile_dir, "jit_ragged_paged_attention")
  assert num_calls == len(tasks)
  return t_us / num_reqs, xplane_pb_path


def print_table(header, rows, *, col_width_extra=None):
  if col_width_extra is None:
    col_width_extra = {}
  sz = len(header)
  col_width = [(len(str(h)) + 3 + col_width_extra.get(h, 0)) for h in header]
  start_separator = "╒" + "╤".join("═" * w for w in col_width) + "╕"
  middle_separator = "╞" + "╪".join("═" * w for w in col_width) + "╡"
  end_separator = "╘" + "╧".join("═" * w for w in col_width) + "╛"
  fmt = "│" + "│".join("{{:<{}}}".format(w) for w in col_width) + "│"
  print(start_separator)
  print(fmt.format(*header))
  for row in rows:
    assert len(row) == sz
    print(middle_separator)
    print(fmt.format(*row))
  print(end_separator)


class PagedAttentionKernelBenchmark(jtu.JaxTestCase):

  @parameterized.product(
      model_name=list(MODEL_CONFIGS.keys()),
  )
  def test_benchmark(self, model_name):
    header = [
        "device",
        "model",
        "inp_len",
        "out_len",
        "num_reqs",
        "max_tokens",
        "max_seqs",
        "q_dtype",
        "kv_dtype",
        "baseline (ms)",
        "runtime/req (ms)",
        "speedup",
        "xprof",
    ]
    col_width_extra = {
        "model": 20,
        "q_dtype": 2,
        "kv_dtype": 2,
        "xprof": 55,
    }
    rows = []
    headers = ["device", "device_number", "model", "num_pages", "page_size", "max_tokens", "num_reqs", "max_seqs", "inp_len", "out_len", "total_len", "num_q_heads", "num_kv_heads", "head_dim", "q_dtype", "kv_dtype", "total kernel execution time (ms)", "runtime/req (ms)","xprof"]
    result = {}
    # test_set: (num_pages, inp_len, out_len, num_reqs,  q_dtype, kv_dtype)
    batch_size = MODEL_CONFIGS[model_name]["batch_size"]
    test_set = [
        (1000, 4096,12288, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 1024, 3072, batch_size,  'bfloat16', 'float8_e4m3fn'),
        (1000, 1024, 15360, batch_size,  'bfloat16', 'bfloat16'),
        (1000, 1024, 15360, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 1024, 64512, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 1024, 64512, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 1024, 130048, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 1024, 130048, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 4096, 0, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 4096, 0, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 4096, 12288, batch_size,  'bfloat16', 'bfloat16'),
        (1000, 4096, 12288, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 4096, 61440, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 4096, 61440, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 4096, 126976, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 4096, 126976, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 65536, 0, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 65536, 0, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 65536, 65536, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 65536, 65536, batch_size,  'bfloat16', 'float8_e4m3fn'),
        # (1000, 131072, 0, batch_size,  'bfloat16', 'bfloat16'),
        # (1000, 131072, 0, batch_size,  'bfloat16', 'float8_e4m3fn'),
    ]
    # # test_set: (num_pages, inp_len, out_len, num_reqs, max_tokens, max_seqs, q_dtype, kv_dtype)
    # test_set = [
    #     (1000, 1024, 3072, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     (1000, 1024, 3072, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 1024, 15360, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 1024, 15360, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 1024, 64512, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 1024, 64512, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 1024, 130048, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 1024, 130048, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     (1000, 4096, 0, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     (1000, 4096, 0, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 4096, 12288, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 4096, 12288, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 4096, 61440, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 4096, 61440, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 4096, 126976, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 4096, 126976, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 65536, 0, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 65536, 0, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 65536, 65536, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 65536, 65536, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    #     # (1000, 131072, 0, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'bfloat16'),
    #     # (1000, 131072, 0, batch_size, max_num_batched_tokens, max_num_seqs, 'bfloat16', 'float8_e4m3fn'),
    # ]

    max_num_seqs_options = [128, 256, 512]
    max_num_tokens_options = [1024, 2048, 4096, 8192]
    # model_perf_baseline = perf.PERF_BASELINE[model_name]
    for num_devices in MODEL_CONFIGS[model_name]["num_devices"]:
      device_name = get_device_name(num_devices)
      if device_name not in result:
        result[device_name] = {}
      for spec in test_set:
        (
            num_pages,
            inp_len,
            out_len,
            num_reqs,
            q_dtype_name,
            kv_dtype_name,
        ) = spec
        page_size = 256

        lowest_latency_max_num_seqs, lowest_latency_max_num_tokens, lowest_latency_session_id = None, None, None
        lowest_latency_t_us_per_req = float("inf")
        for max_num_seqs in max_num_seqs_options:
          for max_num_tokens in max_num_tokens_options:
            t_us_per_req, session_id = run_benchmark(
                MODEL_CONFIGS[model_name],
                num_devices=num_devices,
                num_reqs=num_reqs,
                inp_len=inp_len,
                out_len=out_len,
                max_num_tokens=max_num_tokens,
                max_num_seqs=max_num_seqs,
                q_dtype=jnp.dtype(q_dtype_name),
                kv_dtype=jnp.dtype(kv_dtype_name),
                page_size=page_size,
                num_pages=num_pages,
            )
            print(
                f"max_num_seqs={max_num_seqs}, max_num_tokens={max_num_tokens}, t_us_per_req={t_us_per_req}, session_id={session_id}")
            if t_us_per_req is not None and t_us_per_req < lowest_latency_t_us_per_req:
              lowest_latency_t_us_per_req = t_us_per_req
              lowest_latency_max_num_seqs = max_num_seqs
              lowest_latency_max_num_tokens = max_num_tokens
              lowest_latency_session_id = session_id

        result[device_name][spec] = lowest_latency_t_us_per_req

        num_q_heads = MODEL_CONFIGS[model_name]["num_q_heads"]
        num_kv_heads = MODEL_CONFIGS[model_name]["num_kv_heads"]
        head_dim = MODEL_CONFIGS[model_name]["head_dim"]
        device_number = MODEL_CONFIGS[model_name]["num_devices"][0]
        rows.append((
            device_name,
            device_number,
            model_name,
            num_pages,
            page_size,
            lowest_latency_max_num_tokens,
            num_reqs,
            lowest_latency_max_num_seqs,
            inp_len,
            out_len,
            inp_len + out_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_dtype_name,
            kv_dtype_name,
            f"{(lowest_latency_t_us_per_req * num_reqs)/1000:.3f}",
            f"{lowest_latency_t_us_per_req / 1000:.3f}",
            f"{lowest_latency_session_id}",
        ))
    print_table(headers, rows, col_width_extra=col_width_extra)
    print()
    print(
        "* Please update the baseline with the latest results once the speedup"
        f" is confirmed for {model_name=}:"
    )
    print(result)
    print()

    with open(os.path.join("result.csv"), "w") as f:
      writer = csv.writer(f)
      writer.writerow(headers)
      writer.writerows(rows)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
