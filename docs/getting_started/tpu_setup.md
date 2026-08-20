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

## Provisioning Cloud TPUs

Google Cloud supports two primary APIs for provisioning TPUs:

- **[Compute Engine API](https://cloud.google.com/tpu/docs/tpus-in-compute-engine) (`gcloud compute instances create`)**: Recommended for modern TPU generations starting with **TPU v6e (Trillium)** and **TPU v5p**.
- **[Cloud TPU API](https://cloud.google.com/tpu/docs/queued-resources) (`gcloud alpha compute tpus queued-resources create`)**: Legacy API used for earlier generations like **TPU v5e**. Note that the Cloud TPU API is no longer under active development.

!!! note "TPU v7x (Ironwood)"
    TPU v7x (Ironwood) is in preview status and is provisioned via **Google Kubernetes Engine (GKE)** rather than standalone on-demand VM commands. Note that TPU v7x does not support Flex-start (DWS). For v7x provisioning and cluster orchestration, see [About TPUs in GKE](https://cloud.google.com/tpu/docs/tpus-in-gke) and the official [TPU v7x documentation](https://cloud.google.com/tpu/docs/tpu7x).

You can also choose between two capacity models:

- **Standard (On-Demand)**: Immediate allocation at standard pay-as-you-go rates.
- **Flex-start (DWS)**: Discounted capacity via Dynamic Workload Scheduler (DWS) that runs uninterrupted for up to 7 days. Note that Flex-start is supported on **TPU v5e**, **TPU v5p**, and **TPU v6e** (**TPU v7x**, as well as older generations like **v3** and **v4**, do not support Flex-start). For more details, see [About Flex-start VMs](https://cloud.google.com/compute/docs/instances/about-flex-start-vms) and [DWS Pricing](https://cloud.google.com/products/dws/pricing#flex-start-tpu-vm-pricing).

Select your desired TPU hardware, number of chips, and capacity model to generate the exact provisioning command. Be sure to replace placeholder variables (like `PROJECT_ID` and `SERVICE_ACCOUNT`) with your own values before running.

<div class="command-generator-container" id="prov-generator">
  <div class="cg-options-group">
    <span class="cg-label">Hardware</span>
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
  <div class="cg-options-group">
    <span class="cg-label">Model</span>
    <button class="cg-btn active" role="button" aria-pressed="true" data-group="prov_model" data-val="standard">Standard</button>
    <button class="cg-btn" role="button" aria-pressed="false" data-group="prov_model" data-val="flex_start">Flex-start (DWS)</button>
  </div>
  
  <div id="prov-output-instructions" class="cg-instructions"></div>
  <div class="cg-output-container">
    <pre><code id="prov-output-command" class="language-shell"></code></pre>
  </div>
</div>

| Parameter | Description |
|-----------|-------------|
| `PROJECT_ID` | Your Google Cloud project ID. |
| `ZONE` | The Google Cloud zone where you have TPU quota (e.g., `us-east5-a`, `europe-west4-a`, `us-central2-b`). See [TPU regions and zones](https://cloud.google.com/tpu/docs/regions-zones) for availability. |
| `SERVICE_ACCOUNT` | The email address for your service account, found in the Cloud Console under IAM Service Accounts (e.g., `tpu-service-account@<your_project_ID>.iam.gserviceaccount.com`). Required for legacy Cloud TPU API calls. |
| `RUNTIME_VERSION` | Automatically populated by the generator above based on your selected TPU hardware generation. |

### Connecting and Checking Status

**Connect to your TPU VM using SSH**:

- **For GCE VM Instances (TPU v6e, v5p)**:

    ```bash
    gcloud compute ssh my-tpu-vm --zone ZONE
    ```

- **For Legacy TPU VMs (TPU v5e, v4, v3)**:

    ```bash
    gcloud compute tpus tpu-vm ssh my-tpu-name --project PROJECT_ID --zone ZONE
    ```

**Check Provisioning Status**:

To check whether your TPU VM or Flex-start request has been allocated and is running:

- **For GCE VM Instances (TPU v6e, v5p)**:

    ```bash
    gcloud compute instances describe my-tpu-vm --zone ZONE
    ```

- **For Queued Resources (TPU v5e, v4, v3)**:

    ```bash
    gcloud alpha compute tpus queued-resources describe my-queued-resource --zone ZONE
    ```

[TPU versions]: https://cloud.google.com/tpu/docs/runtimes
[TPU VM images]: https://cloud.google.com/tpu/docs/runtimes
[TPU regions and zones]: https://cloud.google.com/tpu/docs/regions-zones
