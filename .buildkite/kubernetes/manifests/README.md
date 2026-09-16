# Workload manifests

Handed to the launcher with `--manifest`, for workloads the built-in one-pod
Job cannot express: several roles that talk to each other, or the hosts of one
multi-host slice.

`workloads/` names what it runs. `qwen3-coder-480b-1p1d.yaml` serves that model
in that topology and nothing else, so it is named for the model and the shape.
One of these belongs to one step.

A manifest that is told what to run instead - the step passing a command it
executes - is a template, named for the bring-up rather than any benchmark,
and several steps can share one. A workload that acquires a second caller has
to be parameterised first, which makes it a template and renames it.

`v7x/` predates both and is not referenced by any pipeline.
