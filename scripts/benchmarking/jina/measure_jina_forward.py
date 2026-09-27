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
"""Times the jina-embeddings-v2 forward (and pooling) on one TPU chip.

Run on the TPU VM with vLLM stopped (the chip must be free), from the repo
root, in the serving environment:

    python scripts/benchmarking/jina/measure_jina_forward.py \
        --output /tmp/jina_forward.json [--trace-dir /tmp/jina-trace]

What is timed (all on the real checkpoint, float32 params, like the server):

* ``forward``: the jitted model forward (embeddings + encoder) of one engine
  step, dispatch to `block_until_ready`, for 1x1024, 2x1024 packed and 1x2048
  tokens. Paths: the encoder megakernel (each kernel version available at
  this commit, `JINA_BERT_MEGAKERNEL_VERSION`) and the per-layer XLA path
  (`USE_JINA_BERT_MEGAKERNEL=0`), each at matmul precision ``default`` and
  ``highest``. For the XLA path, ``highest`` means
  `jax.default_matmul_precision("highest")` for its XLA matmuls; its Pallas
  flash-attention kernel is unchanged.
* ``step`` (unless --no-pooling): forward + copy to host + vLLM's embed pooler
  (mean pooling + normalization), i.e. the device and pooling part of the
  engine step, for the pooling paths available at this commit: ``old`` (the
  pooler under torchax's dispatch modes, `TPU_POOLING_FAST_PATH=0`),
  ``fast`` (plain torch, the default) and ``device`` (mean pooling on the
  TPU, `JINA_BERT_DEVICE_POOLING=1`).

Each measurement: `--warmup` untimed runs, then `--runs` (>= 100) timed runs;
the JSON reports median / p90 / mean / min in milliseconds. `--trace-dir`
additionally records a `jax.profiler` trace of a few runs of every
measurement (open it with xprof / TensorBoard; steps are annotated).
"""

import argparse
import contextlib
import functools
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import types
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np

MODEL_ID = "jinaai/jina-embeddings-v2-small-en"
CLS_ID, SEP_ID = 101, 102
CASES = {  # name -> (seq_lens, padded step size as the runner buckets it)
    "1x1024": ([1024], 1024),
    "2x1024": ([1024, 1024], 2048),
    "1x2048": ([2048], 2048),
}


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=200, help="timed runs (>=100)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--cases", default=",".join(CASES))
    p.add_argument("--precisions", default="default,highest")
    p.add_argument("--paths",
                   default="megakernel,xla",
                   help="comma list of: megakernel, xla")
    p.add_argument("--versions",
                   default="all",
                   help="megakernel versions to time (default: all available)")
    p.add_argument("--no-pooling",
                   action="store_true",
                   help="skip the forward + pooling (step) measurements")
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="request slots in seq_lens (the runner's max_num_reqs)")
    p.add_argument("--trace-dir", default=None)
    p.add_argument("--trace-runs", type=int, default=5)
    p.add_argument("--output", default=None, help="also write JSON here")
    args = p.parse_args()
    if args.runs < 100:
        p.error("--runs must be >= 100")
    return args


def _shim_transformers_onnx():
    # The jina remote-code config imports transformers.onnx (removed in
    # transformers v5); it is never used at inference.
    try:
        import transformers.onnx  # noqa: F401
    except Exception:
        import transformers
        mod = types.ModuleType("transformers.onnx")
        mod.OnnxConfig = type("OnnxConfig", (), {})
        sys.modules["transformers.onnx"] = mod
        transformers.onnx = mod


class _VllmConfig:
    """Just enough of VllmConfig to build and load JinaBertForMaskedLM."""

    def __init__(self):
        from vllm.config import ModelConfig
        self.model_config = ModelConfig(MODEL_ID,
                                        runner="pooling",
                                        trust_remote_code=True)
        self.model_config.dtype = jnp.float32
        self.load_config = MagicMock()
        self.load_config.download_dir = None
        self.cache_config = MagicMock(cache_dtype="auto")
        self.quant_config = None
        self.parallel_config = None


def _available_versions():
    from tpu_inference import envs
    if "JINA_BERT_MEGAKERNEL_VERSION" in envs.environment_variables:
        from tpu_inference.kernels.jina_bert_megakernel import kernel
        return list(kernel.KERNEL_VERSIONS)
    return ["v1"]  # Before kernel versions existed.


def _build_model(vllm_config, mesh, precision):
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import LoadConfig, get_model_loader

    from tpu_inference.models.jax.jina_bert import JinaBertForMaskedLM

    os.environ["USE_JINA_BERT_MEGAKERNEL"] = "1"
    os.environ["JINA_BERT_MEGAKERNEL_PRECISION"] = precision
    with jax.set_mesh(mesh):
        model = JinaBertForMaskedLM(vllm_config, jax.random.PRNGKey(0), mesh)
    if not model.model.encoder.use_megakernel:
        raise RuntimeError("The megakernel is not supported here; see the "
                           "warning above.")
    with jax.set_mesh(mesh), set_current_vllm_config(vllm_config):
        get_model_loader(LoadConfig(load_format="hf")).load_weights(
            model, vllm_config.model_config)
    return model


def _forward_fn(model):
    """Jitted forward over flat state leaves, as the model runner does."""
    from flax import nnx
    graphdef, state = nnx.split(model)
    leaves, treedef = jax.tree_util.tree_flatten(state)

    @jax.jit
    def forward(leaves, input_ids, metadata):
        m = nnx.merge(graphdef, jax.tree_util.tree_unflatten(treedef, leaves))
        _, hidden, _, _ = m(kv_caches=[],
                            input_ids=input_ids,
                            attention_metadata=metadata)
        return hidden

    return lambda ids, md: forward(leaves, ids, md)


def _make_step(seq_lens, num_tokens, max_num_seqs, vocab_size, seed=0):
    from tpu_inference.layers.common.attention_metadata import \
        AttentionMetadata
    rng = np.random.default_rng(seed)
    ids = np.zeros(num_tokens, np.int32)
    positions = np.zeros(num_tokens, np.int32)
    off = 0
    for n in seq_lens:
        tokens = rng.integers(1000, vocab_size, size=n, dtype=np.int32)
        tokens[0], tokens[-1] = CLS_ID, SEP_ID
        ids[off:off + n] = tokens
        positions[off:off + n] = np.arange(n)
        off += n
    lens = np.zeros(max_num_seqs, np.int32)
    lens[:len(seq_lens)] = seq_lens
    metadata = AttentionMetadata(
        input_positions=jnp.asarray(positions),
        block_tables=jnp.zeros(lens.shape, jnp.int32),
        seq_lens=jnp.asarray(lens),
        query_start_loc=jnp.asarray(np.concatenate([[0], np.cumsum(lens)]),
                                    dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, len(seq_lens)], jnp.int32),
    )
    return jnp.asarray(ids), metadata


def _timed(fn, runs, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {
        "median_ms": round(statistics.median(times), 4),
        "p90_ms": round(times[min(len(times) - 1, int(0.9 * len(times)))], 4),
        "mean_ms": round(statistics.fmean(times), 4),
        "min_ms": round(times[0], 4),
        "runs": runs,
    }


def _poolers(mesh, max_num_seqs):
    """{name: fn(hidden, seq_lens) -> pooler output}, as available here."""
    import torch
    from vllm.model_executor.layers.pooler.activations import PoolerNormalize
    from vllm.model_executor.layers.pooler.seqwise.heads import \
        EmbeddingPoolerHead
    from vllm.model_executor.layers.pooler.seqwise.methods import MeanPool
    from vllm.model_executor.layers.pooler.seqwise.poolers import \
        SequencePooler
    from vllm.model_executor.layers.pooler.special import DispatchPooler
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

    # vLLM's embed pooler for this model: mean pooling, no ST projector,
    # float32 head, L2 normalization.
    pooler = DispatchPooler({
        "embed":
        SequencePooler(
            MeanPool(),
            EmbeddingPoolerHead(projector=None,
                                head_dtype=torch.float32,
                                activation=PoolerNormalize()))
    })

    def metadata(lens):
        return PoolingMetadata(
            prompt_lens=torch.from_numpy(np.array(lens, np.int32)),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[PoolingParams(task="embed") for _ in lens],
            pooling_states=[PoolingStates() for _ in lens])

    def old(hidden, lens):
        # The original compute_pooler_output body (TPU_POOLING_FAST_PATH=0).
        import torchax
        from torchax.interop import torch_view
        nst = np.array(lens, np.int32)
        md = metadata(lens)
        torch_states = torch_view(hidden)
        with torchax.default_env():
            torch_states = torch_states.to('cpu', non_blocking=True)
            md.build_pooling_cursor(nst,
                                    torch.tensor(nst),
                                    device=torch_states.device)
            return pooler(torch_states, md)

    poolers = {"old": old}
    try:
        from tpu_inference.models.common.pooling import (DeviceMeanPooler,
                                                         pool_on_host)
    except ImportError:
        return poolers  # Before the fast paths existed.

    def fast(hidden, lens):
        nst = np.array(lens, np.int32)
        return pool_on_host(pooler, hidden, metadata(lens), nst, nst)

    device_pooler = DeviceMeanPooler.maybe_create(pooler, mesh, max_num_seqs)

    def device(hidden, lens):
        out = device_pooler(hidden, metadata(lens), np.array(lens, np.int32))
        assert out is not None
        return out

    poolers["fast"] = fast
    poolers["device"] = device
    return poolers


def main():
    args = _parse_args()
    if jax.devices()[0].platform != "tpu":
        sys.exit("Needs a TPU (the megakernel is a Mosaic kernel).")
    _shim_transformers_onnx()
    from tpu_inference.models.vllm.experimental import register_models
    register_models()
    from jax.sharding import Mesh

    devices = np.array(jax.local_devices()[:1]).reshape((1, 1, 1, 1))
    mesh = Mesh(devices, axis_names=("data", "attn_dp", "expert", "model"))
    vllm_config = _VllmConfig()
    vocab = vllm_config.model_config.hf_config.vocab_size

    cases = [c for c in args.cases.split(",") if c]
    precisions = [p for p in args.precisions.split(",") if p]
    paths = [p for p in args.paths.split(",") if p]
    versions = _available_versions()
    if args.versions != "all":
        versions = [v for v in args.versions.split(",") if v in versions]

    steps = {c: _make_step(*CASES[c], args.max_num_seqs, vocab) for c in cases}
    results, step_results, fns = [], [], []

    with jax.set_mesh(mesh):
        poolers = {} if args.no_pooling else _poolers(mesh, args.max_num_seqs)
        for precision in precisions:
            model = _build_model(vllm_config, mesh, precision)
            encoder = model.model.encoder
            configs = []
            if "megakernel" in paths:
                configs += [("megakernel", v) for v in versions]
            if "xla" in paths:
                configs.append(("xla", None))
            for path, version in configs:
                encoder.use_megakernel = path == "megakernel"
                if version is not None and hasattr(encoder,
                                                   "megakernel_version"):
                    encoder.megakernel_version = version
                forward = _forward_fn(model)
                if path == "xla" and precision == "highest":
                    make_ctx = functools.partial(jax.default_matmul_precision,
                                                 "highest")
                else:
                    make_ctx = contextlib.nullcontext
                label = dict(path=path, version=version, precision=precision)
                for case in cases:
                    ids, md = steps[case]
                    seq_lens, num_tokens = CASES[case]

                    def run_forward(forward=forward,
                                    ids=ids,
                                    md=md,
                                    make_ctx=make_ctx):
                        with make_ctx():
                            return forward(ids, md).block_until_ready()

                    stats = _timed(run_forward, args.runs, args.warmup)
                    row = dict(label,
                               case=case,
                               num_tokens=num_tokens,
                               seq_lens=seq_lens,
                               **stats)
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    fns.append((f"forward {path} {version} {precision} "
                                f"{case}", run_forward))
                    # The engine step's device + pooling part, for the
                    # default-precision megakernel paths.
                    if path != "megakernel" or precision != "default":
                        continue
                    for name, pool in poolers.items():

                        def run_step(forward=forward,
                                     ids=ids,
                                     md=md,
                                     pool=pool,
                                     seq_lens=seq_lens):
                            return pool(forward(ids, md), seq_lens)

                        stats = _timed(run_step, args.runs, args.warmup)
                        row = dict(label,
                                   case=case,
                                   num_tokens=num_tokens,
                                   seq_lens=seq_lens,
                                   pooling=name,
                                   **stats)
                        step_results.append(row)
                        print(json.dumps(row), flush=True)
                        fns.append(
                            (f"step {version} pool={name} {case}", run_step))
            encoder.use_megakernel = True

        if args.trace_dir:
            jax.profiler.start_trace(args.trace_dir)
            try:
                for name, fn in fns:
                    for i in range(args.trace_runs):
                        with jax.profiler.StepTraceAnnotation(name,
                                                              step_num=i):
                            fn()
            finally:
                jax.profiler.stop_trace()

    report = dict(
        model=MODEL_ID,
        device=str(jax.devices()[0].device_kind),
        versions={
            "jax": jax.__version__,
            "python": platform.python_version(),
        },
        git_commit=_git_commit(),
        runs=args.runs,
        warmup=args.warmup,
        forward=results,
        step=step_results,
        trace_dir=args.trace_dir,
        notes=[
            "forward = jitted embeddings + encoder of one engine step, "
            "dispatch to block_until_ready (wall clock).",
            "step = forward + host copy + vLLM embed pooler (mean + "
            "normalize); pooling: old = under torchax modes "
            "(TPU_POOLING_FAST_PATH=0), fast = plain torch (default), "
            "device = mean on TPU (JINA_BERT_DEVICE_POOLING=1).",
            "xla/highest = jax.default_matmul_precision('highest') for the "
            "XLA matmuls; the Pallas flash-attention kernel is unchanged.",
        ],
    )
    for pkg in ("jaxlib", "libtpu", "vllm-tpu", "torch", "torchax"):
        try:
            report["versions"][pkg] = importlib.metadata.version(pkg)
        except Exception:  # noqa: BLE001
            pass
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(
                                           os.path.abspath(__file__)),
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    main()
