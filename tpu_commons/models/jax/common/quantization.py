class QuantizationConfig:
    quantize_kvcache: bool = False
    quantize_bit: int | None = 8

    def __init__(self, flag_cfg):
        ...
    def get_aqt_config(self):
        ...
    def make(self):
        ...

class Quantization:
    cfg: QuantizationConfig

    def __init__(self):
        ...
    def get_quantization_config(self) -> QuantizationConfig:
        ...
    def create_weight(self):
        ...
    