"""Sampling kernels for TPU inference."""

from tpu_inference.kernels.sampling.sampling import topk_topp_and_sample
from tpu_inference.kernels.sampling.top_p_and_sample import top_p_and_sample
from tpu_inference.kernels.sampling.divide_and_filter_topk import topk as top_k

__all__ = [
  "top_k",
  "top_p_and_sample",
  "topk_topp_and_sample",
]
