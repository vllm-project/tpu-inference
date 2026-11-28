#!/bin/bash
# One-command DAPO-on-TPU launch with the 299s STRICT-BF16 recipe env levers.
# Prereqs: a PR-image whose libtpu has the v15 RPA/continue_decode Mosaic kernels; a 4x4x4 tpu7x jobset.
set -euo pipefail

# --- the three env levers (default-off in the engine; enabled here) ---
export GRR_RPA_DECODE_BLOCKS="1,16384,1,4096"   # RPA-v3 decode fetch/compute block split
export GRR_CD_EOS_CHECK_INTERVAL="8"            # continue_decode EOS-check interval
# continue_decode + async_scheduling are set in config_template.yaml.

CONFIG="${1:-config_template.yaml}"
echo "Launching DAPO/GRPO RL rollout on TPU v7x with recipe: $CONFIG"
echo "  continue_decode=on  RPA_DECODE_BLOCKS=$GRR_RPA_DECODE_BLOCKS  EOS_INTERVAL=$GRR_CD_EOS_CHECK_INTERVAL"
# Hand off to your MaxText/tunix RL entrypoint, e.g.:
#   python -m your_rl_trainer --mode dapo --config "$CONFIG"
echo "(wire the final line to your RL trainer entrypoint)"
