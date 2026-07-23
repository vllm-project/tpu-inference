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
# rollouts, without pulling the full training stack (tunix/orbax/optax) into the
# shared CI image.
#
# Usage (inside the CI docker image):
#   BUILDKITE_COMMIT=<sha> .buildkite/scripts/run_in_docker.sh \
#     bash /workspace/tpu_inference/tests/e2e/test_gemma3_maxtext_inference.sh
#
# Requires:
#   - HF_TOKEN in the environment (Gemma3 tokenizer/config are gated on HF).
#   - Read access to the checkpoint GCS path (see GEMMA3_MT_CKPT below).
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-gemma3-4b}"

# maxtext is not baked into the shared CI image; install it in-step so this test
# does not affect the image size / build time of every other CI job. Pin a
# version for reproducibility; override with MAXTEXT_VERSION if a bump is needed.
MAXTEXT_VERSION="${MAXTEXT_VERSION:-0.2.3}"

# Pre-converted (HF -> MaxText) unscanned Gemma3-4B checkpoint.
#
# TODO(tpu-inference): stage a permanent copy under a tpu-inference-owned bucket
# (e.g. gs://tpu-commons-ci/gemma3-4b/to_maxtext/unscanned/items) and point the
# default here at it. The default below is a complete converted checkpoint that
# currently lives in MaxText's CI bucket; it is NOT guaranteed to be permanent,
# so treat GEMMA3_MT_CKPT as the source of truth and override in CI as needed.
GEMMA3_MT_CKPT="${GEMMA3_MT_CKPT:-gs://runner-maxtext-logs/gemma3-4b/to_maxtext/unscanned/sft-test-2026-06-24-09-06/0/items}"

PROMPT="${PROMPT:-Suggest some famous landmarks in London.}"
HBM_UTILIZATION="${HBM_UTILIZATION:-0.5}"

LOG_FILE="${LOG_FILE:-gemma3_maxtext_decode.log}"

echo "--- Gemma3-4B MaxText inference smoke test"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MAXTEXT_VERSION=${MAXTEXT_VERSION}"
echo "GEMMA3_MT_CKPT=${GEMMA3_MT_CKPT}"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN must be set (Gemma3 config/tokenizer are gated on HuggingFace)." >&2
    exit 1
fi

echo "--- Installing maxtext[tpu]==${MAXTEXT_VERSION}"
# NOTE: the maxtext wheel declares *no* unconditional dependencies -- every dep
# sits behind an extra (cuda12 / docs / runner / tpu / tpu-post-train). A plain
# `pip install maxtext` therefore installs the package with none of its imports
# available (omegaconf, etc.), so the [tpu] extra is required.
#
# The [tpu] extra specifies its deps as lower bounds (jax>=0.10.0,
# jaxlib>=0.10.0, libtpu>=0.0.40, ...). pip resolves `>=` to the *newest*
# release, so an unconstrained install would upgrade the image's pinned
# jax/jaxlib/libtpu out from under vLLM and break the TPU runtime, even though
# the pinned versions already satisfy the bounds. Constrain the critical stack
# to whatever the image already has and let the rest resolve freely.
CONSTRAINTS_FILE="$(mktemp)"
pip freeze 2>/dev/null \
    | grep -iE '^(jax|jaxlib|libtpu|numpy|torch|torchvision|torchax)==' \
    > "${CONSTRAINTS_FILE}" || true
echo "Pinning the following image packages during the maxtext install:"
cat "${CONSTRAINTS_FILE}"

pip install "maxtext[tpu]==${MAXTEXT_VERSION}" --constraint "${CONSTRAINTS_FILE}"

# Fail fast with a clear message if the install still left imports missing,
# rather than surfacing a bare ModuleNotFoundError from deep inside maxtext.
echo "--- Verifying maxtext imports"
python3 -c "import maxtext; import omegaconf; print('maxtext import OK')"

# The constrained install must not have moved the TPU stack. flax and
# transformers are NOT constrained because maxtext requires flax>=0.12.7 while
# tpu-inference pins flax==0.12.4 -- those bounds are mutually exclusive, so flax
# is necessarily upgraded here. Print the resulting versions so any breakage from
# that upgrade is attributable from the log.
echo "--- Post-install versions of the shared stack"
pip freeze 2>/dev/null | grep -iE '^(jax|jaxlib|libtpu|flax|transformers|numpy)==' || true

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
