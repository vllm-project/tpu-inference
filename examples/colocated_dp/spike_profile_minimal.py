#!/usr/bin/env python
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Minimal Pathways profiling smoke test — no vLLM, no DP code.

Purpose: isolate whether the
    ValueError: Mismatch between out_handlers and num_results: 0 vs 1
crash seen in the colocated-DP run is a pathwaysutils ↔ Pathways runtime
mismatch on this image (in which case it'll reproduce here) or something
specific to our N-RankExecutor init (in which case this script will succeed
and we'll need to look harder at our setup).

Run on the same node as the failing job, with the same env vars Pathways
needs (JAX_PLATFORMS=proxy + whatever your Pathways coordinator setup wants):

  JAX_PLATFORMS=proxy \
  python examples/colocated_dp/spike_profile_minimal.py \
      [gs://path/for/trace]    # optional; defaults to /tmp

Outcome interpretation:
  - "PROFILE OK" → pathwaysutils/Pathways profiling works on this node in
    isolation. The crash in the DP run is then triggered by something we do
    (multiple RankExecutor inits, etc.) and we should fix that.
  - Same crash as the DP run → pathwaysutils profile path is broken on this
    install. Fix is `pip install -U pathwaysutils` (or pin a known-good
    version), not in tpu-inference code.
"""


import jax
import pathwaysutils
pathwaysutils.initialize()


profile_dir = "gs://wenxindong-vm/profiling/spike_minimal"  
jax.profiler.start_trace(profile_dir)

x = jax.numpy.ones((1024, 1024))
for _ in range(5):
    y = jax.jit(lambda v: (v * v).sum() + v.sum())(x)
    y.block_until_ready()
jax.profiler.stop_trace()

