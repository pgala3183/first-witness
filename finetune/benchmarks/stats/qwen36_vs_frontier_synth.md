# Is the gain real? (test_synth, vs `bench-qwen3p6-35b-a3b`)

McNemar = exact two-sided binomial on the disagreements. Bootstrap = 5,000 resamples of whole clusters (one Seville second / one synthetic snapshot), 95% interval.

| System | n | Clusters | Before | After | Gap (pts) | 95% CI | Wins / losses | McNemar p | Bootstrap ahead | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-kimi-k3 | 150 | 150 | 70.0% | 84.0% | +14.0 | +6.7 to +21.3 | 28 / 7 | 0.00051 | 5,000 / 5,000 | real gain |
| bench-glm-5p3-flash | 150 | 150 | 70.0% | 81.3% | +11.3 | +4.7 to +18.7 | 24 / 7 | 0.0033 | 4,994 / 5,000 | real gain |
| bench-deepseek-v4p1-flash | 150 | 150 | 70.0% | 84.0% | +14.0 | +7.3 to +20.7 | 25 / 4 | 0.0001 | 5,000 / 5,000 | real gain |
