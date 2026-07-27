#!/bin/bash
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

# ---------------------------------------------------------------------------
# Gemma3-4B MaxText inference smoke test.
#
# This is the tpu-inference-side counterpart of MaxText's RL validation script
#   https://github.com/AI-Hypercomputer/maxtext/blob/main/tests/end_to_end/tpu/gemma3/4b/test_gemma3_rl.sh
# The MaxText script validates a full RL pipeline (inference -> GRPO training ->
# inference). tpu-inference only participates in that pipeline as the vLLM TPU
# backend that MaxText drives via the `MaxTextForCausalLM` plugin, so here we
# port only the inference legs (steps 1 & 3 of that script): run
# `maxtext.inference.vllm_decode` against a pre-converted MaxText Gemma3-4B
# checkpoint and assert that decoding succeeds and produces non-empty output.
#
# This guards against tpu-inference regressions that would break MaxText's RL
# rollouts, without baking maxtext into the shared CI image.
#
# maxtext is installed from HEAD (see MAXTEXT_REF), so this tracks the upstream
# script rather than a months-old PyPI release. Doing so requires upgrading the
# image's torch 2.10.0+cpu to the torch==2.11.0+cpu that maxtext HEAD pins; the
# image's vLLM is compiled against 2.10, so the script verifies vLLM still
# imports after the upgrade and fails loudly if it does not.
#
# Usage (inside the CI docker image):
#   BUILDKITE_COMMIT=<sha> .buildkite/scripts/run_in_docker.sh \
#     bash /workspace/tpu_inference/tests/e2e/test_gemma3_maxtext_inference.sh
#
# Requires:
#   - HF_TOKEN in the environment (Gemma3 tokenizer/config are gated on HF).
#   - Read access to the checkpoint GCS path (see GEMMA3_MT_CKPT below).
#   - Network access to github.com and download.pytorch.org from the agent.
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-gemma3-4b}"

# maxtext is not baked into the shared CI image; install it in-step so this test
# does not affect the image size / build time of every other CI job.
#
# Install maxtext from HEAD rather than a PyPI release, so this tracks the same
# code as the upstream script this test is ported from
# (tests/end_to_end/tpu/gemma3/4b/test_gemma3_rl.sh on maxtext main). Pin
# MAXTEXT_REF to a SHA if a green run needs to be reproducible.
MAXTEXT_REF="${MAXTEXT_REF:-main}"
MAXTEXT_SPEC="${MAXTEXT_SPEC:-maxtext[tpu-post-train] @ git+https://github.com/AI-Hypercomputer/maxtext@${MAXTEXT_REF}}"

# maxtext HEAD's [tpu-post-train] extra pins torch==2.11.0+cpu /
# torchvision==0.26.0+cpu exactly, while the image ships torch 2.10.0+cpu (the
# version its source-built vLLM was compiled against). We deliberately let torch
# upgrade to 2.11 here rather than holding maxtext back to an old release -- see
# ALLOW_TORCH_UPGRADE below. The +cpu local-version wheels only exist on
# PyTorch's own index, so that index has to be available to the resolver.
TORCH_CPU_INDEX="${TORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}"

# Set to 0 to keep the image's torch and fail instead of upgrading it.
ALLOW_TORCH_UPGRADE="${ALLOW_TORCH_UPGRADE:-1}"

# Pre-converted (HF -> MaxText) unscanned Gemma3-4B checkpoint.
#
# This is a tpu-inference-owned copy in gs://tpu-commons-ci, the bucket the CI
# agent already reads its other model checkpoints from. It was copied from
# MaxText's CI bucket (gs://runner-maxtext-logs/gemma3-4b/to_maxtext/unscanned/
# sft-test-2026-06-24-09-06/0/items), which the agent has no read access to and
# which makes no permanence guarantees -- hence the local copy.
GEMMA3_MT_CKPT="${GEMMA3_MT_CKPT:-gs://tpu-commons-ci/gemma3-4b/to_maxtext/unscanned/items}"

PROMPT="${PROMPT:-Suggest some famous landmarks in London.}"
HBM_UTILIZATION="${HBM_UTILIZATION:-0.5}"

LOG_FILE="${LOG_FILE:-gemma3_maxtext_decode.log}"

echo "--- Gemma3-4B MaxText inference smoke test"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MAXTEXT_REF=${MAXTEXT_REF}"
echo "MAXTEXT_SPEC=${MAXTEXT_SPEC}"
echo "GEMMA3_MT_CKPT=${GEMMA3_MT_CKPT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN must be set (Gemma3 config/tokenizer are gated on HuggingFace)." >&2
    exit 1
fi

# Preflight the checkpoint before spending ~10 minutes on the install and model
# load. Without this, an unreadable path surfaces as a permission traceback from
# inside orbax on rank0, long after the expensive work is done.
#
# gsutil is not reliably present (google-cloud-cli is only installed in the image
# when BM_INFRA=true), but gcsfs is in requirements.txt, so use that. If the check
# itself cannot run, warn and continue rather than failing a test that might
# otherwise pass.
echo "--- Preflight: checking checkpoint is readable"
python3 - "${GEMMA3_MT_CKPT}" <<'PY' || preflight_rc=$?
import sys
path = sys.argv[1].removeprefix("gs://")
try:
    import gcsfs
    fs = gcsfs.GCSFileSystem()
except Exception as e:  # noqa: BLE001 - environment problem, not a test failure
    print(f"WARNING: could not run checkpoint preflight ({e!r}); continuing.")
    sys.exit(0)
try:
    entries = fs.ls(path)
except Exception as e:  # noqa: BLE001
    print(f"ERROR: cannot read checkpoint at gs://{path}", file=sys.stderr)
    print(f"  {e!r}", file=sys.stderr)
    sys.exit(2)
names = {p.rsplit("/", 1)[-1] for p in entries}
print(f"checkpoint readable: {len(entries)} entries")
# orbax writes commit_success.txt only after a checkpoint is fully flushed.
if "commit_success.txt" not in names:
    print("WARNING: no commit_success.txt -- checkpoint may be incomplete.")
PY
if [[ "${preflight_rc:-0}" -eq 2 ]]; then
    echo "The CI agent service account must have storage.objects.get on that path." >&2
    echo "Override the location with GEMMA3_MT_CKPT if the checkpoint has moved." >&2
    exit 1
fi

echo "--- Recording pre-install versions"
TORCH_BEFORE="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
echo "torch before install: ${TORCH_BEFORE}"

echo "--- Installing ${MAXTEXT_SPEC}"
CONSTRAINTS_FILE="$(mktemp)"
CONSTRAIN_RE='^(jax|jaxlib|libtpu|flax|numpy)=='
if [[ "${ALLOW_TORCH_UPGRADE}" != "1" ]]; then
    # Also hold torch, which makes maxtext HEAD unresolvable -- opt-in only.
    CONSTRAIN_RE='^(jax|jaxlib|libtpu|flax|numpy|torch|torchvision|torchax)=='
fi
pip freeze 2>/dev/null | grep -iE "${CONSTRAIN_RE}" > "${CONSTRAINTS_FILE}" || true
echo "Pinning the following image packages during the maxtext install:"
cat "${CONSTRAINTS_FILE}"

# torch/torchvision are intentionally absent from the constraints above so the
# resolver can take maxtext's torch==2.11.0+cpu. Those +cpu local-version wheels
# are only published on PyTorch's index, hence --extra-index-url.
if ! pip install "${MAXTEXT_SPEC}" \
        --constraint "${CONSTRAINTS_FILE}" \
        --extra-index-url "${TORCH_CPU_INDEX}"; then
    echo "ERROR: could not install ${MAXTEXT_SPEC} against this image." >&2
    echo "If the failure mentions a torch conflict, maxtext HEAD requires" >&2
    echo "torch==2.11.0+cpu; check that ${TORCH_CPU_INDEX} is reachable." >&2
    exit 1
fi

echo "--- Post-install versions of the shared stack"
pip freeze 2>/dev/null | grep -iE '^(jax|jaxlib|libtpu|flax|torch|torchvision|torchax|transformers|numpy|tensorflow)==' || true

TORCH_AFTER="$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
echo "torch: ${TORCH_BEFORE} -> ${TORCH_AFTER}"

# The image's vLLM is built from source against the torch it shipped with, so a
# torch upgrade can break its compiled extensions. Check that explicitly -- a
# clear failure here is far easier to read than an opaque crash inside decode.
if [[ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]]; then
    echo "--- torch changed; verifying vLLM still imports against it"
    if ! python3 -c "import vllm; print('vllm', vllm.__version__, 'imports OK on torch', __import__('torch').__version__)"; then
        echo "ERROR: vLLM fails to import after torch ${TORCH_BEFORE} -> ${TORCH_AFTER}." >&2
        echo "The image's vLLM is compiled against ${TORCH_BEFORE}; its native" >&2
        echo "extensions do not load on ${TORCH_AFTER}. Either rebuild vLLM against" >&2
        echo "the new torch, or re-run with ALLOW_TORCH_UPGRADE=0 (which will instead" >&2
        echo "fail to resolve maxtext HEAD)." >&2
        exit 1
    fi
fi

# Fail fast with a clear message if the install left imports missing, rather than
# surfacing a bare ModuleNotFoundError from deep inside maxtext.
echo "--- Verifying maxtext imports"
python3 -c "import maxtext; import omegaconf; print('maxtext import OK')"

# Force the value: run_in_docker.sh already exports this as 1 in the container,
# so a ${VLLM_XLA_CHECK_RECOMPILATION:-0} default would preserve the 1.
export VLLM_XLA_CHECK_RECOMPILATION="${CHECK_RECOMPILATION_OVERRIDE:-0}"
echo "VLLM_XLA_CHECK_RECOMPILATION=${VLLM_XLA_CHECK_RECOMPILATION}"

echo "--- Running maxtext.inference.vllm_decode"
set -x
python3 -m maxtext.inference.vllm_decode \
    model_name="${MODEL_NAME}" \
    load_parameters_path="${GEMMA3_MT_CKPT}" \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    hbm_utilization_vllm="${HBM_UTILIZATION}" \
    prompt="${PROMPT}" \
    use_chat_template=True \
    scan_layers=false \
    enable_single_controller=false \
    2>&1 | tee "${LOG_FILE}"
set +x

# `set -o pipefail` above means a non-zero exit from vllm_decode already fails
# the test. As an extra correctness guard, require that the run actually emitted
# generated text rather than exiting 0 after an early no-op.
echo "--- Verifying decode produced output"
if ! grep -qiE "output|generated|completion" "${LOG_FILE}"; then
    echo "ERROR: vllm_decode did not appear to produce any generated output." >&2
    echo "See ${LOG_FILE} for details." >&2
    exit 1
fi

echo "+++ Gemma3-4B MaxText inference smoke test PASSED"
