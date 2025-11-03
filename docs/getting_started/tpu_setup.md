# Cloud TPU Setup

This guide provides information on setting up and provisioning Google Cloud TPUs for use with `tpu-inference`.

## TPU Versions and Topologies

Tensor Processing Units (TPUs) are Google's custom-developed application-specific
integrated circuits (ASICs) used to accelerate machine learning workloads. TPUs
are available in different versions each with different hardware specifications.
For more information about TPUs, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm).

The following TPU versions are compatible with `tpu-inference`:

### Recommended
- [TPU v6e](https://cloud.google.com/tpu/docs/v6e)
- [TPU v5e](https://cloud.google.com/tpu/docs/v5e)

### Experimental
- [TPU v5p](https://cloud.google.com/tpu/docs/v5p)
- [TPU v4](https://cloud.google.com/tpu/docs/v4)
- [TPU v3](https://cloud.google.com/tpu/docs/v3)

These TPU versions allow you to configure the physical arrangements of the TPU
chips. This can improve throughput and networking performance. For more
information see:

- [TPU v6e topologies](https://cloud.google.com/tpu/docs/v6e#configurations)
- [TPU v5e topologies](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)
- [TPU v5p topologies](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)
- [TPU v4 topologies](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)

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

Use the following command to provision a Cloud TPU. Replace the parameters in all caps with your own values.

```bash
gcloud alpha compute tpus queued-resources create QUEUED_RESOURCE_ID \
  --node-id TPU_NAME \
  --project PROJECT_ID \
  --zone ZONE \
  --accelerator-type ACCELERATOR_TYPE \
  --runtime-version RUNTIME_VERSION \
  --service-account SERVICE_ACCOUNT
```

| Parameter name     | Description                                                                                                                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| QUEUED_RESOURCE_ID | The user-assigned ID of the queued resource request.                                                                                                                                                     |
| TPU_NAME           | The user-assigned name of the TPU which is created when the queued resource request is allocated.                                                                                                        |
| PROJECT_ID         | Your Google Cloud project                                                                                                                                                                                |
| ZONE               | The Google Cloud zone where you want to create your Cloud TPU. The value you use depends on the version of TPUs you are using. For more information, see [TPU regions and zones]                                  |
| ACCELERATOR_TYPE   |  Specify the TPU version, for example `v5litepod-4` specifies a v5e TPU with 4 cores, `v6e-1` specifies a v6e TPU with 1 core. For more information, see [TPU versions]. |
| RUNTIME_VERSION    | The TPU VM runtime version to use. For example, use `v2-alpha-tpuv6e` for a VM loaded with one or more v6e TPU(s).  For more information, see [TPU software versions](https://docs.cloud.google.com/tpu/docs/runtimes)                                          |
| SERVICE_ACCOUNT    | The email address for your service account. You can find it in the IAM Cloud Console under *Service Accounts*. For example: `tpu-service-account@<your_project_ID>.iam.gserviceaccount.com`              |

Connect to your TPU VM using SSH:

```bash
gcloud compute tpus tpu-vm ssh TPU_NAME --project PROJECT_ID --zone ZONE
```

!!! note
    When configuring `RUNTIME_VERSION` ("TPU software version") for your TPU, ensure it matches the TPU generation you've selected by referencing the [TPU VM images] compatibility matrix. Using an incompatible version may prevent vLLM from running correctly.

[TPU versions]: https://cloud.google.com/tpu/docs/runtimes
[TPU VM images]: https://cloud.google.com/tpu/docs/runtimes
[TPU regions and zones]: https://cloud.google.com/tpu/docs/regions-zones
