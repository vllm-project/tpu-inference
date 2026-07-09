# Cloud TPU Setup

This guide provides information on setting up and provisioning Google Cloud TPUs for use with `tpu-inference`.

## TPU Versions and Topologies

Tensor Processing Units (TPUs) are Google's custom-developed application-specific
integrated circuits (ASICs) used to accelerate machine learning workloads. TPUs
are available in different versions each with different hardware specifications.
For more information about TPUs, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm).

The following TPU versions are compatible with `tpu-inference`. Selecting a topology allows you to configure the physical arrangements of the TPU chips, improving throughput and networking performance.

### Recommended

<div class="grid cards" markdown>

- __TPU v7x (Ironwood)__
  <span class="cg-badge cg-badge-recommended">Recommended</span>
  
  Next-generation architecture for bleeding-edge research and ultra-large model training.
  
  [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/tpu7x)

- __TPU v6e (Trillium)__
  <span class="cg-badge cg-badge-recommended">Recommended</span>
  
  Optimal performance for mainstream AI workloads with balanced memory and compute.
  
  [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v6e#configurations)

- __TPU v5e__
  <span class="cg-badge cg-badge-recommended">Recommended</span>
  
  Cost-effective performance for medium-to-large scale inference and training.
  
  [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)

</div>

### Experimental

<div class="grid cards" markdown>

- __TPU v5p__
  <span class="cg-badge cg-badge-experimental">Experimental</span>
  
  High-performance architecture optimized for peak compute and memory bandwidth.
  
  [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)

- __TPU v4__
  <span class="cg-badge cg-badge-experimental">Experimental</span>
  
  Previous generation flagship architecture for large-scale training.
  
  [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)

- __TPU v3__
  <span class="cg-badge cg-badge-experimental">Experimental</span>
  
  Legacy architecture suitable for smaller scale experimentation.
  
  [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v3)

</div>

## Quota and Pricing

In order for you to use Cloud TPUs you need to have TPU quota granted to your
Google Cloud project. For more information, see [TPU quota](https://cloud.google.com/tpu/docs/quota#tpu_quota).

For TPU pricing information, see [Cloud TPU pricing](https://cloud.google.com/tpu/pricing).

You may need additional persistent storage for your TPU VMs. For more
information, see [Storage options for Cloud TPU data](https://cloud.devsite.corp.google.com/tpu/docs/storage-options).

## Provisioning Cloud TPUs

You can provision Cloud TPUs using the [Cloud TPU API](https://cloud.google.com/tpu/docs/reference/rest)
or the [queued resources](https://cloud.google.com/tpu/docs/queued-resources)
API (preferred). This section shows how to create TPUs using the queued resource API.

### Provision a Cloud TPU with the queued resource API

Select your desired TPU hardware and number of chips to generate the exact provisioning command. Be sure to replace the placeholder variables (like `PROJECT_ID` and `SERVICE_ACCOUNT`) with your own values before running.

<div class="command-generator-container" id="prov-generator">
  <div class="cg-options-group">
    <span class="cg-label">Hardware</span>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_hw" data-val="v7x">TPU v7x</button>
    <button class="cg-btn active" role="button" aria-pressed="true" data-group="prov_hw" data-val="v6e">TPU v6e</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_hw" data-val="v5e">TPU v5e</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_hw" data-val="v5p">TPU v5p</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_hw" data-val="v4">TPU v4</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_hw" data-val="v3">TPU v3</button>
  </div>
  <div class="cg-options-group">
    <span class="cg-label">Chips</span>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_chips" data-val="1">1</button>
    <button class="cg-btn active" role="button" aria-pressed="true" data-group="prov_chips" data-val="4">4</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_chips" data-val="8">8</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_chips" data-val="16">16</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_chips" data-val="32">32</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_chips" data-val="64">64</button>
  </div>
  
  <div id="prov-output-instructions" class="cg-instructions"></div>
  <div class="cg-output-container">
    <pre><code id="prov-output-command" class="language-shell"></code></pre>
  </div>
</div>

Connect to your TPU VM using SSH:

```bash
gcloud compute tpus tpu-vm ssh my-tpu-name --project PROJECT_ID --zone ZONE
```

!!! note
    When configuring `RUNTIME_VERSION` ("TPU software version") for your TPU, ensure it matches the TPU generation you've selected by referencing the [TPU VM images] compatibility matrix. Using an incompatible version may prevent vLLM from running correctly.

[TPU versions]: https://cloud.google.com/tpu/docs/runtimes
[TPU VM images]: https://cloud.google.com/tpu/docs/runtimes
[TPU regions and zones]: https://cloud.google.com/tpu/docs/regions-zones
