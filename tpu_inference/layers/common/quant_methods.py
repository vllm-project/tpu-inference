UNQUANTIZED = "unquantized"
MXFP4 = "mxfp4"
AWQ = "awq"
COMPRESSED_TENSORS = "compressed-tensors"
FP8 = "fp8"


def get_tpu_quant_method(quant_method: str) -> str:
    return "tpu-" + quant_method
