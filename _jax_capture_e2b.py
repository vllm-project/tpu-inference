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
"""E2B oracle capture driver.

Sets E2B_CAPTURE_DIR before importing tpu_inference. The gated capture hooks
in tpu_inference/models/jax/gemma4.py write per-layer hidden-state .npy files
to that dir. After offline_inference, this script bundles them into one .npz
written to /tmp/hf_home/jax_e2b_hidden.npz, which is bind-mounted to host's
/mnt/disks/persist/models/ for buildkite-agent artifact upload.

THROWAWAY. Reverted before opening any production PR.
"""

import os

# Set BEFORE importing vllm / tpu_inference. Subprocesses (EngineCore) inherit
# this env, re-import gemma4.py, and see _E2B_CAPTURE_DIR.
CAPTURE_DIR = "/tmp/hf_home/e2b_capture"
OUTPUT_NPZ = "/tmp/hf_home/jax_e2b_hidden.npz"

os.environ["E2B_CAPTURE_DIR"] = CAPTURE_DIR
os.environ["VLLM_XLA_CHECK_RECOMPILATION"] = "0"
os.environ["JITTED_MM_MODULE_KEYS"] = "model.vision_tower.encoder"
os.environ["REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES"] = (
    "transformers.modeling_outputs.BaseModelOutputWithPast")
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

import glob  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    # Wipe any stale captures from a previous run.
    for f in glob.glob(os.path.join(CAPTURE_DIR, "*.npy")):
        os.remove(f)

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model="google/gemma-4-E2B-it",
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
    )
    print(f"[load] {time.time() - t0:.1f}s", flush=True)

    prompt = "The capital of France is"
    out = llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))
    print(f"[output] {out[0].outputs[0].text!r}", flush=True)

    # Bundle .npy files into one .npz.
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "*.npy")))
    print(f"[capture] {len(files)} npy files", flush=True)
    if not files:
        print(f"[capture] WARNING: no captures in {CAPTURE_DIR}", flush=True)
        # Still write an empty npz so the artifact upload step doesn't fail.
        np.savez_compressed(OUTPUT_NPZ)
        return

    bundle = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        bundle[name] = np.load(f)
        print(f"  {name:<25} {bundle[name].shape} dtype={bundle[name].dtype}",
              flush=True)
    np.savez_compressed(OUTPUT_NPZ, **bundle)
    sz = os.path.getsize(OUTPUT_NPZ)
    print(f"[save] {OUTPUT_NPZ}  ({sz / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
