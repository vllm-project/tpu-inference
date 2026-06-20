# How to add a new kernel target

This guide walks through wiring a new Pallas kernel into the evolve
loop end-to-end. Five steps; the synthetic-matmul example is the
shortest working reference.

## Step 1: Build a `KernelHost`

The orchestrator and evaluator depend on the `KernelHost` Protocol
(see `evaluator.py:KernelHost`). Minimum surface:

```python
class MyKernelHost:
    kernel_name = "my_kernel"
    kernel_symbol = "the_pallas_fn_name"  # name in your kernel.py

    def __init__(self, **shape_kwargs) -> None:
        # generate fixed inputs once; the evaluator pins them in a closure
        ...
        self.inputs = {"x": x, "y": y, ...}

    @property
    def baseline_path(self) -> str:
        return "tpu_inference/kernels/my_kernel/kernel.py"

    def read_baseline_source(self) -> str:
        return (REPO_ROOT / self.baseline_path).read_text()

    def build_kernel_fn(self, module) -> Callable[[], Any]:
        kernel = getattr(module, self.kernel_symbol)
        x = self.inputs["x"]
        y = self.inputs["y"]

        def fn():
            return kernel(jnp.copy(x), jnp.copy(y))

        return fn

    def get_oracle(self):
        return MyReferenceOracle()  # see Step 2

    def anti_cheat_skip_keys(self) -> tuple[str, ...]:
        return ()  # or e.g. ("kv_cache",) for in-place updates
```

Important: `jnp.copy(x)` inside `fn()` is required when Pallas donates
input buffers — otherwise the second iter of `bench.measure` fails with
"Donation requested for invalid buffer." See the RPA v3 host for the
canonical pattern.

## Step 2: Define a reference oracle

The verifier compares your kernel's output to whatever `oracle.compute`
returns. Easiest path: pure JAX reference.

```python
class MyReferenceOracle:
    def compute(self, inputs: dict) -> jax.Array:
        return jnp.matmul(inputs["x"], inputs["y"])

    def dtype_tolerance(self, dtype) -> tuple[float, float]:
        bits = jnp.dtype(dtype).itemsize * 8
        return {32: (0.05, 0.05), 16: (0.1, 0.1)}.get(bits, (0.2, 0.2))
```

The verifier tolerances should *exactly match* what your kernel's unit
tests use. Production wins under tighter tolerance are stronger
evidence.

## Step 3: Build the evolve example

```python
# tools/kernel/evolve/examples/my_kernel_evolve.py
from tools.kernel.evolve.archive import Archive
from tools.kernel.evolve.genome import Genome
from tools.kernel.evolve.orchestrator import EvolutionConfig, Orchestrator
from tools.kernel.evolve.mutator.programmatic import (
    ProgrammaticMutator, LiteralRewriteRule)


def main():
    host = MyKernelHost(M=2048, N=2048, K=2048)
    mutator = ProgrammaticMutator(
        baseline_path=host.baseline_path,
        literal_rules=[
            LiteralRewriteRule("BLOCK_M",
                               values=["64", "128", "256", "512"]),
            LiteralRewriteRule("BLOCK_N",
                               values=["64", "128", "256", "512"]),
        ],
        seed=0,
    )
    archive = Archive(baseline=Genome.baseline(host.baseline_path),
                      num_islands=2, persist_path="/tmp/my_kernel.jsonl")
    orch = Orchestrator(host=host, mutator=mutator, archive=archive,
                        config=EvolutionConfig(generations=3))
    orch.run()
```

## Step 4: Write a CI recipe

```python
# tools/kernel/evolve/ci/recipes/my_kernel.py
from pathlib import Path


def run(*, out_dir: Path) -> dict:
    # The recipe is just a `run(out_dir)` callable that returns a CI
    # summary dict. The simplest pattern is to subprocess to your own
    # CLI entrypoint and parse its output.
    ...
    return {
        "target_kernel": "my_kernel",
        "wins_count": n_wins,
        "diff_path": str(diff_file) if n_wins else None,
        "summary_path": str(summary_file),
        "telemetry_path": str(telemetry_file),
    }
```

`nightly_evolve` discovers recipes by filename and runs them
automatically when files under the corresponding kernel directory have
been touched in the last 24h.

## Step 5: Test it locally

```bash
# Stub run (no TPU work) to confirm the recipe loads:
python -m tools.kernel.evolve.ci.nightly \
    --recipe tools/kernel/evolve/ci/recipes/my_kernel.py --dry-run

# Real run:
python -m tools.kernel.evolve.examples.my_kernel_evolve --generations 3
```

## Common pitfalls

* **Donation errors on iter 2**: wrap `inputs` in `jnp.copy(...)` inside
  the returned `fn()`. See RPA v3 host.
* **Baseline fails verifier**: usually means your oracle disagrees with
  your kernel on some input region. Either tighten the oracle, narrow
  the input range, or relax the tolerance — but don't relax it past
  what the kernel's own unit tests use.
* **Anti-cheat false positives**: if your kernel legitimately returns
  an input verbatim (e.g. unchanged kv_cache when `update_kv_cache=False`),
  add that input's key to `anti_cheat_skip_keys()`.
* **Diff applier silently rejects diffs**: the unified diff format the
  LLM emits must reference lines that exist in the *current* baseline.
  If you're applying multiple diffs sequentially, regenerate the diff
  after each apply — line numbers shift.

## Production checklist

Before merging your kernel's recipe into `nightly_evolve`:

* [ ] Baseline verifies against oracle on `seed=0..9`.
* [ ] At least one programmatic mutation candidate VERIFIED.
* [ ] If using LLM mutator: ≥1 critic rejection seen and reasoning
      is on-topic.
* [ ] Telemetry events emit (`/tmp/<kernel>_telemetry.jsonl` non-empty).
* [ ] `nightly_evolve --recipe <yours> --dry-run` lists your recipe.
* [ ] Auto-PR diff applies cleanly when piped through `git apply --check`.
