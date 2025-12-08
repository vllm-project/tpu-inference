# Recommended Model and Feature Matrices

Although vLLM TPU’s new unified backend makes out-of-the-box high performance serving possible with any model supported in vLLM, the reality is that we're still in the process of implementing a few core components.
For this reason, until we land more capabilities, we recommend starting from this list of stress tested models and features below.

We are still landing components in tpu-inference that will improve performance for larger scale, higher complexity models (XL MoE, +vision encoders, MLA, etc.).

If you’d like us to prioritize something specific, please submit a GitHub feature request [here](https://github.com/vllm-project/tpu-inference/issues/new/choose).

## Recommended Models

These tables show the models currently tested for accuracy and performance.

### Text-Only Models

{{ read_csv('../support_matrices/text_only_model_support_matrix.csv', keep_default_na=False) }}

### Multimodal Models

{{ read_csv('../support_matrices/multimodal_model_support_matrix.csv', keep_default_na=False) }}

## Recommended Features

This table shows the features currently tested for accuracy and performance.

{{ read_csv('../support_matrices/feature_support_matrix.csv', keep_default_na=False) }}

## Kernel Support

This table shows the current kernel support status.

{{ read_csv('../support_matrices/kernel_support_matrix.csv', keep_default_na=False) }}

## Parallelism Support

This table shows the current parallelism support status.

{{ read_csv('../support_matrices/parallelism_support_matrix.csv', keep_default_na=False) }}

## Quantization Support

This table shows the current quantization support status.

{{ read_csv('../support_matrices/quantization_support_matrix.csv', keep_default_na=False) }}
