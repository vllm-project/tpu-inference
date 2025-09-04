#!/bin/bash

echo "--- Starting Special Buildkite Bootstrap ---"

buildkite-agent pipeline upload .buildkite/pipeline_jax.yml

echo "--- Buildkite Special Bootstrap Finished ---"
