"""Benchmark for Ragged Paged Attention-V3

This will run either a single step of prefill, where we assume:
* Batch size and maximum number of sequences is 1
* Output length is 0
* Input length is just the context length
* Maximum number of (batched) tokens is the context length (i.e. no chunking)


Or a single step of decode, where we assume:
* Batch size and the maximum number of sequences are given (e.g. 64, 1024)
* Output length is 1
* Input length is just the context length
* Maximum number of (batched) tokens is the batch size (i.e. number of total tokens to generate)

We currently make the following assumptions:
* The number of pages is 1000
* The page size is 256
* The query dtype is bfloat16
* Only using 1 core

To run, simply run:
python benchmarking/benchmark_janus.py

Noting the following arguments:
* --is_prefill: Set to true to run prefill configs.
* --outdir: Directory to save the result CSV and profiles.

"""
import csv
import os
import shutil
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from absl.testing import absltest
from benchmark_block_sizes import TUNED_BLOCK_SIZES_PER_RESULT
from google.cloud import storage
from jax._src import test_util as jtu
from tensorflow.tsl.profiler.protobuf import xplane_pb2

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape, ragged_paged_attention)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 import \
    get_kv_cache_shape as get_kv_cache_shape_hd64
from tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64 import \
    ragged_paged_attention_hd64
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv
from tpu_inference.utils import get_device_name

os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_dvfs_p_state=7"

FLAGS = flags.FLAGS

DEFAULT_LOCAL_PATH = "outdir"

flags.DEFINE_bool("is_prefill", False, "Set to true to run prefill configs.")
flags.DEFINE_string("outdir", DEFAULT_LOCAL_PATH,
                    "Directory to save results and profiles.")

# Assume using 1 core
DEVICE_NUMBER = 1
# NOTE: this is arbitrary for now
NUMBER_OF_PAGES = 1000
PAGE_SIZE = 256
Q_DTYPE = "bfloat16"
VMEM_LIMIT_BYTES = 120 * 1024 * 1024

# The subir for profiles within the outdir
PROFILE_DIR_NAME = "rpa-v3-benchmark-profiles"


def upload_to_gcs(local_results_dir: str, remote_dir: str) -> None:
    """
    Uploads a local file to a specified storage location.
    """

    if remote_dir.startswith("gs://"):  # Google Cloud Storage (G\CS)
        # recursively count the number of files in the directory to upload
        num_expected_files = 0
        for _, _, files in os.walk(local_results_dir):
            for _ in files:
                num_expected_files += 1

        path_parts = remote_dir[5:].split("/", 1)
        bucket_name = path_parts[0]
        # If there's a path after the bucket, use it; otherwise empty string
        gcs_prefix = path_parts[1] if len(path_parts) > 1 else ""

        # Remove trailing slash from prefix if present to avoid double slashes later
        gcs_prefix = gcs_prefix.rstrip("/")

        # 2. Initialize the client and bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # 3. Determine the base folder name to mimic gsutil behavior
        # gsutil cp -r /local/dir gs://bucket/dest -> gs://bucket/dest/dir/file.txt
        base_folder_name = os.path.basename(
            os.path.normpath(local_results_dir))

        num_uploaded_files = 0
        # 4. Walk the local directory and upload files
        for root, _, files in os.walk(local_results_dir):
            for filename in files:
                local_file_path = os.path.join(root, filename)

                # Calculate the relative path from the directory being uploaded
                relative_path = os.path.relpath(local_file_path,
                                                local_results_dir)

                # Construct the remote blob path
                # Structure: {remote_prefix}/{local_dir_name}/{relative_path_to_file}
                if gcs_prefix:
                    blob_path = f"{gcs_prefix}/{base_folder_name}/{relative_path}"
                else:
                    blob_path = f"{base_folder_name}/{relative_path}"

                # Upload the file
                # Note: Standardize paths to use forward slashes for GCS
                blob_path = blob_path.replace(os.sep, "/")
                blob = bucket.blob(blob_path)

                blob.upload_from_filename(local_file_path)

                num_uploaded_files += 1

        if num_uploaded_files != num_expected_files:
            raise ValueError(
                f"Expected to upload {num_expected_files} files, but uploaded {num_uploaded_files}."
            )
        print(
            f"Successfully uploaded {num_expected_files} files from {local_results_dir} to {remote_dir}."
        )

    else:
        raise KeyError(f"{local_results_dir} is not a valid GCS path.")


def generate_run_configs(is_prefill: bool) -> dict:
    """
  Generate run configs for both prefill and decode benchmarks over the given search space.

  Args:
    is_prefill: Whether to generate configs for prefill or decode.

  Returns:
    A dictionary of run configs where the keys are in the format:
      num_q_heads_{num_q_heads}_num_kv_heads_{num_kv_heads}_head_dim_{head_dim}_batch_size_{batch_size}_context_length_{context_length}_kv_dtype_{kv_dtype}_is_prefill_{is_prefill}

    And each value has the following structure:
      {
        'num_q_heads': num_q_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'batch_size': batch_size,
        'context_length': context_length,
        'q_dtype': Q_DTYPE,
        'kv_dtype': kv_dtype,
        'page_size': PAGE_SIZE,
        'is_prefill': is_prefill,
      }
  """
    run_configs = {}
    head_dim_options = [256]
    num_kv_heads_per_card_options = [8]
    num_q_heads_per_card_options = [16]
    batch_size_options = [1] if is_prefill else [4]
    kv_dtype_options = ['bfloat16', 'float8_e4m3fn']
    context_length_options = [512, 1024, 2048]
    for head_dim in head_dim_options:
        for num_kv_heads in num_kv_heads_per_card_options:
            for num_q_heads in num_q_heads_per_card_options:
                for batch_size in batch_size_options:
                    for kv_dtype in kv_dtype_options:
                        for context_length in context_length_options:
                            if num_q_heads >= num_kv_heads:
                                key = f'num_q_heads_{num_q_heads}_num_kv_heads_{num_kv_heads}_head_dim_{head_dim}_batch_size_{batch_size}_context_length_{context_length}_kv_dtype_{kv_dtype}_is_prefill_{is_prefill}'
                                run_configs[key] = {
                                    'num_q_heads': num_q_heads,
                                    'num_kv_heads': num_kv_heads,
                                    'head_dim': head_dim,
                                    'batch_size': batch_size,
                                    'context_length': context_length,
                                    'q_dtype': Q_DTYPE,
                                    'kv_dtype': kv_dtype,
                                    'page_size': PAGE_SIZE,
                                    'is_prefill': is_prefill
                                }
    print(f"Generated {len(run_configs)} configs")
    return run_configs


def schedule_single_step_of_pure_decode_or_prefill(num_reqs: int,
                                                   context_len: int,
                                                   is_prefill: bool) -> list:
    """
    Returns a list of tuples of the form (distribution, cu_q_lens, kv_lens) for
    a single step of pure decode or prefill.

    Args:
      num_reqs: Number of requests.
      context_len: Length of the context.
      is_prefill: Whether to generate configs for prefill or decode.

    Returns:
      A list of tuples of the form (distribution, cu_q_lens, kv_lens)
    """
    if is_prefill:
        # For a single prefill request, we want distribution to be [0, 1, 1]
        distribution = jnp.array([0, 1, 1], dtype=jnp.int32)
        # For a single prefill request, we want cu_q_lens to be [0, input_len]
        cu_q_lens = jnp.zeros(num_reqs + 1, dtype=jnp.int32)
        cu_q_lens = cu_q_lens.at[-1].set(context_len)
        # For a single prefill request, we want kv_lens to be [input_len]
        kv_lens = jnp.full(shape=(1, ),
                           fill_value=context_len,
                           dtype=jnp.int32)
    else:
        distribution = jnp.array([num_reqs, num_reqs, num_reqs],
                                 dtype=jnp.int32)
        cu_q_lens = jnp.arange(num_reqs + 1, dtype=jnp.int32)
        # Fill the request lengths with the context length here
        kv_lens = jnp.full(shape=(num_reqs, ),
                           fill_value=context_len,
                           dtype=jnp.int32)
    return [(distribution, cu_q_lens, kv_lens)]


def extract_op_times_from_xplane(profiler_path: str,
                                 op_name: str) -> Tuple[float, int, str]:
    """
    Parses an xplane.pb file and extracts operation durations for a given operation name.

    Args:
      profiler_path: The path to the profiler directory.
      op_name: The name of the operation to extract.

    Returns:
      A tuple of the form (t_us, bq_sz, bkv_p, uid_profile_dir), where
        t_us: The total duration of the RPA operation in microseconds.
        bq_sz: The query block size.
        bkv_p: The KV block size.
        uid_profile_dir: The path to the trace.

    NOTE: in the case that the benchmark fails, this will return None for each
    """
    xplane_pb_path = None
    for root, _, files in os.walk(profiler_path):
        for file in files:
            if file.endswith('.xplane.pb'):
                xplane_pb_path = os.path.join(root, file)
                break

    with open(xplane_pb_path, 'rb') as f:
        serialized_space = f.read()

    space = xplane_pb2.XSpace()
    space.ParseFromString(serialized_space)

    total_duration_ps = 0
    event_count = 0

    for plane in space.planes:
        event_metadata_map = plane.event_metadata

        for line in plane.lines:
            for event in line.events:
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


def run_benchmark(run_config: Dict[str, Any]) -> Tuple[float, int, int, str]:
    """
    Runs the benchmark for a given run configuration.

    Args:
      run_config: The run configuration dictionary.

    Returns:
      A tuple of the form (total_duration_us, event_count, xplane_pb_path)
    """
    num_q_heads = run_config["num_q_heads"]
    num_kv_heads = run_config["num_kv_heads"]
    head_dim = run_config["head_dim"]
    num_reqs = run_config["batch_size"]
    context_length = run_config["context_length"]
    is_prefill = run_config["is_prefill"]
    q_dtype = jnp.dtype(run_config["q_dtype"])
    kv_dtype = jnp.dtype(run_config["kv_dtype"])
    page_size = run_config["page_size"]

    # This would normally handle chunked prefill, but we're not doing that here
    max_num_tokens = context_length if is_prefill else num_reqs
    max_num_seqs = num_reqs
    pages_per_seq = cdiv(context_length, page_size)

    q_shape = (max_num_tokens, num_q_heads, head_dim)
    kv_shape = (max_num_tokens, num_kv_heads, head_dim)
    get_kv_cache_shape_fn = get_kv_cache_shape_hd64 if head_dim == 64 else get_kv_cache_shape
    kv_cache_shape = get_kv_cache_shape_fn(
        NUMBER_OF_PAGES,
        page_size,
        num_kv_heads,
        head_dim,
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
    page_indices = np.random.randint(0,
                                     NUMBER_OF_PAGES,
                                     size=(max_num_seqs * pages_per_seq, ),
                                     dtype=jnp.int32)

    k_scale = 0.5 if q_dtype != kv_dtype else None
    v_scale = 0.5 if q_dtype != kv_dtype else None

    tasks = schedule_single_step_of_pure_decode_or_prefill(
        num_reqs,
        context_length,
        is_prefill,
    )
    # We should only run a single step of either prefill or decode
    assert len(tasks) == 1
    ragged_paged_attention_fn = (ragged_paged_attention_hd64
                                 if head_dim == 64 else ragged_paged_attention)

    distribution, cu_q_lens, kv_lens = tasks[0]
    key = f'num_q_heads_{num_q_heads}_num_kv_heads_{num_kv_heads}_head_dim_{head_dim}_batch_size_{num_reqs}_context_length_{context_length}_kv_dtype_{kv_dtype}_is_prefill_{is_prefill}'
    try:
        bkv_p, bq_sz = TUNED_BLOCK_SIZES_PER_RESULT[key]
    except KeyError:
        print(f"Could not find tuned blocks for key {key}, skipping...")
        return None, None, None, None
    try:
        # Warmup the kernel
        _, kv_cache = jax.block_until_ready(
            ragged_paged_attention_fn(
                q,
                k,
                v,
                kv_cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
                k_scale=k_scale,
                v_scale=v_scale,
                num_kv_pages_per_block=bkv_p,
                num_queries_per_block=bq_sz,
            ))

        uid_profile_dir = f"{DEFAULT_LOCAL_PATH}/{PROFILE_DIR_NAME}/{key}"
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = os.getenv("PYTHON_TRACER_LEVEL", 0)

        with jax.profiler.trace(uid_profile_dir, profiler_options=options):
            _, kv_cache = jax.block_until_ready(
                ragged_paged_attention_fn(
                    q,
                    k,
                    v,
                    kv_cache,
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    distribution,
                    vmem_limit_bytes=VMEM_LIMIT_BYTES,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    num_kv_pages_per_block=bkv_p,
                    num_queries_per_block=bq_sz,
                ))
        t_us, num_calls, _ = extract_op_times_from_xplane(
            uid_profile_dir, "jit_ragged_paged_attention")
        assert num_calls == len(tasks)
        return t_us, bq_sz, bkv_p, uid_profile_dir

    except Exception as e:
        print(
            f"WARNING: test case {key} failed with exception {e}: skipping...")
        return None, None, None, None


class PagedAttentionKernelBenchmark(jtu.JaxTestCase):

    def test_benchmark(self):
        rows = []
        # Device	# Devices	# of Pages	Page Size	Max Num Batched Tokens	Batch Size	Max Num Seqs	Input Length	Output Length	Context Length	Q Heads (Per-Core)	KV Heads (Per-Core)	Q heads per KV	Head Size	Q/Output-DType	KV-DType
        headers = [
            "device", "device_number", "num_pages", "page_size",
            "max_batche_tokens", "num_reqs", "max_seqs", "inp_len", "out_len",
            "context_length", "num_q_heads (per core)",
            "num_kv_heads (per core)", "head_dim", "q_dtype", "kv_dtype",
            "Causal Mask Kernel Execution Time (ms)", "xprof"
        ]

        is_prefill = FLAGS.is_prefill
        run_configs = generate_run_configs(is_prefill=is_prefill)

        outdir = flags.FLAGS.outdir
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(DEFAULT_LOCAL_PATH, exist_ok=True)
        output_result_csv_file_path = os.path.join(DEFAULT_LOCAL_PATH,
                                                   "benchmark_results.csv")
        # NOTE: we are overwriting the file
        with open(output_result_csv_file_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        print(outdir, DEFAULT_LOCAL_PATH, is_prefill)

        for run_name in run_configs:
            print(f"Starting run: {run_name}")
            run_latency, run_q_block_size, run_kv_page_block_size, run_uid_profile_dir = run_benchmark(
                run_configs[run_name])

            if run_latency is None:
                continue

            device_name = get_device_name(DEVICE_NUMBER)
            page_size = run_configs[run_name]["page_size"]
            num_reqs = run_configs[run_name]["batch_size"]
            num_q_heads = run_configs[run_name]["num_q_heads"]
            num_kv_heads = run_configs[run_name]["num_kv_heads"]
            head_dim = run_configs[run_name]["head_dim"]
            q_dtype_name = run_configs[run_name]["q_dtype"]
            kv_dtype_name = run_configs[run_name]["kv_dtype"]
            is_prefill = run_configs[run_name]["is_prefill"]
            context_length = run_configs[run_name]["context_length"]

            max_num_tokens = context_length if is_prefill else num_reqs
            max_num_seqs = num_reqs
            inp_len = context_length if is_prefill else 0
            out_len = 0 if is_prefill else 1

            rows.append((
                device_name,
                DEVICE_NUMBER,
                NUMBER_OF_PAGES,
                page_size,
                max_num_tokens,
                num_reqs,
                max_num_seqs,
                inp_len,
                out_len,
                context_length,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q_dtype_name,
                kv_dtype_name,
                f"{run_latency / 1000:.9f}",
                run_uid_profile_dir,
                # run_kv_page_block_size,
                # run_q_block_size,
            ))

            print(
                f"Obtained latency of {run_latency / 1000:.9f} ms for run: {run_name}\n"
            )

        with open(output_result_csv_file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        # After writing, copy the results to the output directory
        if outdir.startswith("gs://"):
            upload_to_gcs(DEFAULT_LOCAL_PATH, outdir)
        else:
            shutil.copytree(DEFAULT_LOCAL_PATH, outdir)
        # delete the temp dir
        shutil.rmtree(DEFAULT_LOCAL_PATH)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
