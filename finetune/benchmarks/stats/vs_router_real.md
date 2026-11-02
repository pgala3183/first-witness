# Is the gain real? (test_real, vs `baseline`)

McNemar = exact two-sided binomial on the disagreements. Bootstrap = 5,000 resamples of whole clusters (one Seville second / one synthetic snapshot), 95% interval.

| System | n | Clusters | Before | After | Gap (pts) | 95% CI | Wins / losses | McNemar p | Bootstrap ahead | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-qwen3p6-35b-a3b | 150 | 90 | 61.3% | 85.3% | +24.0 | +14.5 to +33.6 | 49 / 13 | 4.8e-06 | 5,000 / 5,000 | real gain |
| bench-qwen3p8-27b | 150 | 90 | 61.3% | 75.3% | +14.0 | +3.3 to +24.4 | 43 / 22 | 0.013 | 4,972 / 5,000 | real gain |
| bench-glm-5p3-flash | 150 | 90 | 61.3% | 90.0% | +28.7 | +19.1 to +38.5 | 53 / 10 | 3.4e-08 | 5,000 / 5,000 | real gain |
| bench-deepseek-v4p1-flash | 150 | 90 | 61.3% | 91.3% | +30.0 | +20.8 to +38.9 | 53 / 8 | 3e-09 | 5,000 / 5,000 | real gain |
| bench-kimi-k3 | 150 | 90 | 61.3% | 96.7% | +35.3 | +26.7 to +44.1 | 56 / 3 | 1.2e-13 | 5,000 / 5,000 | real gain |
