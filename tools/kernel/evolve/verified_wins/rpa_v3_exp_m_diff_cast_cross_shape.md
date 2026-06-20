| shape | description | baseline μs | patched μs | speedup | p | direction |
|---|---|---|---|---|---|---|
| `qwen3_0_6b_short` | Qwen3-0.6B-style, max_len=1024 | 502.31 | 497.75 | **1.0219×** | 0.2049 | tie  |
| `qwen3_0_6b_long` | Qwen3-0.6B-style, max_len=4096 | 513.84 | 504.50 | **1.0317×** | 0.0134 | WIN  |
| `llama3_8b_mid` | Llama-3-8B attention shape, max_len=2048 | 531.43 | 503.47 | **1.0508×** | 0.0048 | WIN  |
| `qwen3_0_6b_fp8_kv` | Qwen3-0.6B with fp8 KV cache | 497.92 | 510.16 | **0.9862×** | 0.3508 | tie  |
