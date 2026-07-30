# Installation

This guide provides instructions for installing and running `tpu-inference`.

There are three ways to install `tpu-inference`: using `uv pip`, running a Docker container, or building from source.

## Interactive Installation

Select your preferred installation method and hardware target to generate the exact setup command.

<div class="command-generator-container">
  <div class="cg-options-group">
    <span class="cg-label">Method</span>
    <button class="cg-btn active" role="button" aria-pressed="true" data-group="method" data-val="uv_pip">uv pip</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="method" data-val="docker">Docker</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="method" data-val="source">Source</button>
  </div>
  
  <div id="cg-output-instructions" class="cg-instructions"></div>
  <div class="cg-output-container">
    <pre><code id="cg-output-command" class="language-shell"></code></pre>
  </div>
</div>

## Verify Installation

To quickly verify that the installation was successful under any of the above methods and `vllm-tpu` is correctly configured:

```shell
python -c '
import jax
import vllm
import importlib.metadata
from vllm.platforms import current_platform

tpu_version = importlib.metadata.version("tpu_inference")
print(f"vllm version: {vllm.__version__}")
print(f"tpu_inference version: {tpu_version}")
print(f"vllm platform: {current_platform.get_device_name()}")
print(f"jax backends: {jax.devices()}")
'
# Expected output:
# vllm version: 0.x.x
# tpu_inference version: 0.x.x
# vllm platform: TPU V6E (or your specific TPU architecture)
# jax backends: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), ...]
```
