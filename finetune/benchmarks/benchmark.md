# Call-agent benchmark (prompted, no fine-tune)

Same system prompt and held-out questions for every model. Pass = no false promise, no made-up fact, and the required facts present. Latency budget: 1500 ms p95.

| Model | Role | Pass (both) | test_real | test_synth | Honesty | False promises | Made-up facts | p50 ms | p95 ms | $/1k answers | Errors | Blank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Keyword router (today) | baseline | - | 55.3% | n/a | 38.1% | 0.0% | 0.0% | 3.3 | 2050.2 | $0 | - | - |
| Kimi K3 | reference | **90.3%** | 96.7% | 84.0% | 93.8% | 0.0% | 3.3% | 1751.2 | 3982.9 | 1.9085 | 0.0% | 0.0% |
| DeepSeek V4.1 Flash 1017 | reference | **87.7%** | 91.3% | 84.0% | 89.2% | 0.0% | 0.7% | 1104.7 | 1631.1 | 0.1791 | 0.0% | 0.0% |
| GLM 5.3 Flash | reference | **85.7%** | 90.0% | 81.3% | 80.7% | 0.7% | 4.0% | 773.4 | 1468.3 | 0.0859 | 0.0% | 0.0% |
| Qwen3.6 35B A3B | candidate | **77.7%** | 85.3% | 70.0% | 71.9% | 0.0% | 1.4% | 587.5 | 814.1 | 0.2332 | 0.0% | 0.0% |
| Qwen3.8 27B | candidate | **75.0%** | 75.3% | 74.7% | 60.0% | 2.0% | 2.0% | 612.3 | 832.0 | 0.9833 | 0.0% | 0.0% |

## Pick
**Fine-tune Qwen3.6 35B A3B** (`accounts/fireworks/models/qwen3p6-35b-a3b`): highest pass rate (77.7%) among candidates inside the 1500 ms p95 budget.
Gap to the best reference (Kimi K3, 90.3%): 12.6 points. This gap is what the fine-tune has to close.

Latency is measured with parallel requests, so it runs high; recheck the pick with `--workers 1`. Training-pool cost uses the pool's prefill and sample rates.
