# Call-agent benchmark (prompted, no fine-tune)

Same system prompt and held-out questions for every model. Pass = no false promise, no made-up fact, and the required facts present. Latency budget: 1500 ms p95.

| Model | Role | Pass (both) | test_real | test_synth | Honesty | False promises | Made-up facts | p50 ms | p95 ms | $/1k answers | Errors | Blank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Keyword router (today) | baseline | - | 55.3% | n/a | 38.1% | 0.0% | 0.0% | 3.3 | 2050.2 | $0 | - | - |
| Kimi K3 | reference | **89.3%** | 95.3% | 83.3% | 93.8% | 0.0% | 3.3% | 1751.2 | 3982.9 | 1.9085 | 0.0% | 0.0% |
| DeepSeek V4.1 Flash 1017 | reference | **85.3%** | 89.3% | 81.3% | 89.2% | 0.0% | 0.7% | 1104.7 | 1631.1 | 0.1791 | 0.0% | 0.0% |
| GLM 5.3 Flash | reference | **85.3%** | 90.0% | 80.7% | 80.7% | 0.7% | 4.0% | 773.4 | 1468.3 | 0.0859 | 0.0% | 0.0% |

## Pick
No candidate results yet. The training-pool candidates need no deployment: `python finetune/benchmark.py run --only qwen3p6-35b-a3b qwen3p8-27b --limit 150`.

Latency is measured with parallel requests, so it runs high; recheck the pick on its own deployment with `--workers 1`.
