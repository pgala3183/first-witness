# Is the gain real? (test_real, vs `bench-qwen3p6-35b-a3b`)

McNemar = exact two-sided binomial on the disagreements. Bootstrap = 5,000 resamples of whole clusters (one Seville second / one synthetic snapshot), 95% interval.

| System | n | Clusters | Before | After | Gap (pts) | 95% CI | Wins / losses | McNemar p | Bootstrap ahead | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-kimi-k3 | 150 | 90 | 85.3% | 96.7% | +11.3 | +5.6 to +17.5 | 20 / 3 | 0.00049 | 4,999 / 5,000 | real gain |
| bench-glm-5p3-flash | 150 | 90 | 85.3% | 90.0% | +4.7 | -1.4 to +10.9 | 15 / 8 | 0.21 | 4,599 / 5,000 | could be luck |
| bench-deepseek-v4p1-flash | 150 | 90 | 85.3% | 91.3% | +6.0 | +1.3 to +10.9 | 12 / 3 | 0.035 | 4,959 / 5,000 | real gain |
