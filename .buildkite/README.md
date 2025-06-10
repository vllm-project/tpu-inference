# Buildkite

The GitHub webhook is configured to trigger the Buildkite pipeline. The current step configuration of the pipeline:

```
steps:
  - label: ":pipeline: Upload Pipeline"
    agents:
      queue: tpu_v6e_queue
    command: "bash .buildkite/scripts/bootstrap.sh"
```
